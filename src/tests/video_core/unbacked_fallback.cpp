// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>

#include "video_core/renderer_vulkan/vk_state_tracker.h"

#include "video_core/nx2.h"
#include "video_core/renderer_vulkan/blit_image.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_render_pass_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/texture_cache/image_view_info.h"
#include "video_core/texture_cache/util.h"
#include "video_core/vulkan_common/unbacked_fallback.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace {

constexpr u64 Texels(u64 width, u64 height, u64 depth = 1, u64 layers = 1) {
    return width * height * depth * layers;
}

constexpr u64 Rgba8Bytes(u64 texels) {
    return texels * 4;
}

} // namespace

TEST_CASE("Dense image limit requires both a large image and a large allocation",
          "[video_core][vulkan][unbacked_fallback]") {
    REQUIRE_FALSE(
        Vulkan::IsTooLargeToBack(Texels(1280, 720), Rgba8Bytes(Texels(1280, 720))));
    REQUIRE_FALSE(
        Vulkan::IsTooLargeToBack(Texels(1024, 1024), Rgba8Bytes(Texels(1024, 1024))));
    REQUIRE_FALSE(Vulkan::IsTooLargeToBack(Vulkan::MaxBackedImageTexels,
                                         Rgba8Bytes(Vulkan::MaxBackedImageTexels)));
    REQUIRE(Vulkan::IsTooLargeToBack(Vulkan::MaxBackedImageTexels + 1,
                                   Rgba8Bytes(Vulkan::MaxBackedImageTexels + 1)));
}

TEST_CASE("The BC6H atlas fits at mip 3 with every array layer retained",
          "[video_core][vulkan][unbacked_fallback]") {
    const u64 full = Vulkan::Bc6hMipTailBytes(8192, 8192, 12, 68, 0);
    const u64 compact = Vulkan::Bc6hMipTailBytes(8192, 8192, 12, 68, 3);
    REQUIRE(Vulkan::IsTooLargeToBack(Texels(8192, 8192, 1, 68), full));
    CHECK(Vulkan::CompactBc6hFirstMip(8192, 8192, 12, 68) == 3);
    CHECK((compact >> 20) == 90);
    CHECK_FALSE(Vulkan::IsTooLargeToBack(Texels(1024, 1024, 1, 68), compact));
    CHECK(Vulkan::Bc6hMipTailBytes(8192, 8192, 12, 68, 11) == 68 * 16);
}

TEST_CASE("A compressed volume over the texel limit is backed",
          "[video_core][vulkan][unbacked_fallback]") {
    // Metroid Prime 4's light volumes: 1408x256x1008 BC1 (8 bytes a 4x4 block) and BC6H (16).
    constexpr u64 texels = Texels(1408, 256, 1008);
    constexpr u64 bc1_bytes = (1408 / 4) * (256 / 4) * 1008 * 8;
    REQUIRE(texels > Vulkan::MaxBackedImageTexels);
    REQUIRE_FALSE(Vulkan::IsTooLargeToBack(texels, bc1_bytes));
    REQUIRE_FALSE(Vulkan::IsTooLargeToBack(texels, bc1_bytes * 2));
}

TEST_CASE("The oversized BC6H array is rejected or backed without sparse residency",
          "[video_core][vulkan][unbacked_fallback]") {
    const auto library = Vulkan::OpenLibrary();
    if (!library) {
        SKIP("No Vulkan library");
    }
    Vulkan::vk::InstanceDispatch dld;
    std::optional<Vulkan::vk::Instance> instance;
    try {
        instance.emplace(Vulkan::CreateInstance(*library, dld, VK_API_VERSION_1_1));
    } catch (const Vulkan::vk::Exception&) {
        SKIP("No Vulkan instance");
    }
    std::optional<Vulkan::Device> device;
    for (const VkPhysicalDevice physical : instance->EnumeratePhysicalDevices()) {
        try {
            device.emplace(**instance, Vulkan::vk::PhysicalDevice(physical, dld), VK_NULL_HANDLE,
                           dld);
            break;
        } catch (const Vulkan::vk::Exception&) {
            continue;
        }
    }
    if (!device) {
        SKIP("No Vulkan device");
    }
    if (!device->IsOptimalBcnSupported()) {
        SKIP("No native Vulkan BCn support");
    }
    if (device->IsSparseResidencyImage2DSupported() && device->CanUseUnbackedSparseImages()) {
        SKIP("Need a Vulkan device without usable unbound sparse images");
    }

    Vulkan::MemoryAllocator allocator(*device);
    Vulkan::StateTracker tracker;
    Vulkan::Scheduler scheduler(*device, tracker);
    Vulkan::StagingBufferPool staging(*device, allocator, scheduler, nullptr, 8);
    Vulkan::DescriptorPool descriptors(*device, scheduler);
    Vulkan::BlitImageHelper blit(*device, scheduler, tracker, descriptors);
    Vulkan::RenderPassCache render_passes(*device);
    Vulkan::UpdateDescriptorQueue queue(*device, Vulkan::UpdateDescriptorQueue::COMPUTE_FRAME_PAYLOAD_SIZE);
    Vulkan::TextureCacheRuntime runtime(*device, scheduler, allocator, staging, blit,
                                        render_passes, descriptors, queue);

    VideoCommon::ImageInfo info;
    info.format = VideoCore::Surface::PixelFormat::BC6H_UFLOAT;
    info.type = VideoCommon::ImageType::e2D;
    info.size = {8192, 8192, 1};
    info.resources = {12, 68};
    info.layer_stride = 0x5560000;
    info.native_tiling = true;
    constexpr GPUVAddr address = 0x82027b0000;
    {
        Vulkan::Image oversized(runtime, info, address, 0);
        CHECK(oversized.OutOfMemory());
        CHECK(allocator.FailedImages() == 0);
    }
    struct SparseRegion {
        explicit SparseRegion(GPUVAddr addr, u64 size) : address{addr} {
            Tegra::Nx2AddSparseSpan(address, size);
            Tegra::Nx2SparseRegionCreated(address, size, 4U << 20);
        }
        ~SparseRegion() {
            Tegra::Nx2SparseRegionDestroyed(address);
        }
        GPUVAddr address;
    } region{address, VideoCommon::CalculateGuestSizeInBytes(info)};

    Vulkan::Image compact(runtime, info, address, 0);
    REQUIRE_FALSE(compact.OutOfMemory());
    REQUIRE(True(compact.flags & VideoCommon::ImageFlagBits::CompactLayers));
    REQUIRE_FALSE(True(compact.flags & VideoCommon::ImageFlagBits::Unbacked));
    REQUIRE(compact.MemoryBlock().size > 0);
    CHECK(compact.host_level_offset >= 3);
    if (const char* expected = std::getenv("LICHII_TEST_COMPACT_BC6H_MIP")) {
        CHECK(compact.host_level_offset == std::strtoul(expected, nullptr, 10));
    }
    CHECK(compact.guest_size_bytes == 0x16ad80000);
    const auto last_layer = compact.TryFindBase(address + 67ULL * info.layer_stride);
    REQUIRE(last_layer.has_value());
    CHECK(last_layer->layer == 67);

    const VideoCommon::ImageViewInfo view_info(
        VideoCommon::ImageViewType::e2DArray, info.format, {{0, 0}, {12, 68}});
    const Vulkan::ImageView view(runtime, view_info, VideoCommon::ImageId{}, compact);
    REQUIRE(view.Handle(Shader::TextureType::ColorArray2D) != VK_NULL_HANDLE);

    const auto upload = staging.Request(16, Vulkan::MemoryUsage::Upload);
    const auto download = staging.Request(16, Vulkan::MemoryUsage::Download);
    REQUIRE(upload.mapped_span.size() >= 16);
    REQUIRE(download.mapped_span.size() >= 16);
    std::ranges::fill(upload.mapped_span.first(16), u8{0x42});
    const VideoCommon::BufferImageCopy copy{
        .buffer_size = 16,
        .image_subresource = {static_cast<s32>(compact.HostMipLevels()) - 1, 67, 1},
        .image_extent = {4, 4, 1},
    };
    compact.UploadMemory(upload, std::span{&copy, 1});
    compact.DownloadMemory(download.buffer, download.offset, std::span{&copy, 1});
    scheduler.Finish();
    CHECK(std::memcmp(upload.mapped_span.data(), download.mapped_span.data(), 16) == 0);
}
