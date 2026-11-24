// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "video_core/frame_gen/flyhigh_library.h"
#include "video_core/renderer_vulkan/present/flyhigh_frame_gen.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

namespace {

[[noreturn]] void Fail(const std::string& what) {
    throw std::runtime_error(what);
}

const VideoCore::FrameGen::FlyHighLibrary& RequireLibrary() {
    const auto* library = VideoCore::FrameGen::LoadFlyHighLibrary();
    if (library == nullptr) {
        Fail("the FlyHigh library is missing or too old");
    }
    return *library;
}

VkImageMemoryBarrier Transition(VkImage image, VkAccessFlags src_access, VkAccessFlags dst_access,
                                VkImageLayout old_layout, VkImageLayout new_layout) {
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
}

VkImageBlit WholeBlit(VkExtent2D extent) {
    const VkImageSubresourceLayers layers{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    const VkOffset3D corner{static_cast<s32>(extent.width), static_cast<s32>(extent.height), 1};
    return VkImageBlit{
        .srcSubresource = layers,
        .srcOffsets = {{0, 0, 0}, corner},
        .dstSubresource = layers,
        .dstOffsets = {{0, 0, 0}, corner},
    };
}

VkBufferImageCopy RowsCopy(VkExtent2D extent) {
    return VkBufferImageCopy{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = {},
        .imageExtent = {extent.width, extent.height, 1},
    };
}

bool IsSrgb(VkFormat format) {
    return format == VK_FORMAT_B8G8R8A8_SRGB || format == VK_FORMAT_R8G8B8A8_SRGB ||
           format == VK_FORMAT_A8B8G8R8_SRGB_PACK32;
}

} // Anonymous namespace

struct FlyHighFrameGen::Worker {
    struct Engine {
        const VideoCore::FrameGen::FlyHighLibrary& fh = RequireLibrary();
        FhEngine* engine{};
        FhModel* model{};
        FhStream* stream{};
        std::chrono::steady_clock::time_point start{std::chrono::steady_clock::now()};

        ~Engine() {
            if (stream) {
                fh.fh_stream_destroy(stream);
            }
            if (model) {
                fh.fh_model_destroy(model);
            }
            if (engine) {
                fh.fh_engine_destroy(engine);
            }
        }

        void Check(int result, const char* call) const {
            if (result < 0) {
                Fail(fmt::format("{} failed: {}", call, fh.LastError()));
            }
        }
    };

    std::mutex mutex;
    std::condition_variable ready;
    std::packaged_task<void(Engine&)> pending;
    bool stopped{};
    bool owns_slot{};
    inline static std::atomic_uint running{};

    ~Worker() {
        if (owns_slot) {
            running.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    static std::shared_ptr<Worker> Start() {
        (void)RequireLibrary();
        auto worker = std::make_shared<Worker>();
        if (running.fetch_add(1, std::memory_order_relaxed) >= 2) {
            running.fetch_sub(1, std::memory_order_relaxed);
            Fail("two FlyHigh workers are still running");
        }
        worker->owns_slot = true;
        std::thread([worker] {
            Engine engine;
            for (;;) {
                std::packaged_task<void(Engine&)> job;
                {
                    std::unique_lock lock{worker->mutex};
                    worker->ready.wait(lock, [&] { return worker->stopped || worker->pending.valid(); });
                    if (worker->stopped) {
                        return;
                    }
                    job = std::move(worker->pending);
                }
                job(engine);
            }
        }).detach();
        return worker;
    }

    template<typename Task>
    auto Run(std::chrono::steady_clock::time_point deadline, Task task) {
        using Result = std::invoke_result_t<Task, Engine&>;
        std::packaged_task<Result(Engine&)> job{std::move(task)};
        auto result = job.get_future();
        {
            std::scoped_lock lock{mutex};
            pending = std::packaged_task<void(Engine&)>{std::move(job)};
        }
        ready.notify_one();
        if (result.wait_until(deadline) != std::future_status::ready) {
            Fail("the FlyHigh engine did not finish before the deadline");
        }
        return result.get();
    }

    void Stop() {
        {
            std::scoped_lock lock{mutex};
            stopped = true;
        }
        ready.notify_one();
    }
};

FlyHighFrameGen::FlyHighFrameGen(const Device& device, MemoryAllocator& memory_allocator,
                                 VkExtent2D extent_, VkFormat format, bool protect_hud)
    : extent{extent_} {
    using namespace std::chrono_literals;

    if (!VideoCore::FrameGen::FlyHighModelInstalled()) {
        Fail("the FlyHigh model is not installed");
    }

    const VkFormat rgba_format =
        IsSrgb(format) ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    constexpr auto blit_features = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
    const auto physical = device.GetPhysical();
    const auto present_features = physical.GetFormatProperties(format).optimalTilingFeatures;
    const auto rgba_features = physical.GetFormatProperties(rgba_format).optimalTilingFeatures;
    if ((present_features & rgba_features & blit_features) != blit_features) {
        Fail("this device can't blit between the present format and RGBA8");
    }
    const std::string adapter =
        fmt::format("{:04x}:{:04x}", device.GetVendorID(), device.GetDeviceID());
    const auto tune_dir = Common::FS::PathToUTF8String(VideoCore::FrameGen::FlyHighTuneDir());
    const auto model_dir = Common::FS::PathToUTF8String(VideoCore::FrameGen::FlyHighModelDir());
    const auto init =
        [adapter, tune_dir, model_dir, extent = extent, protect_hud](Worker::Engine& state) {
            const auto& fh = state.fh;
            FhEngineOptions options = fh.fh_engine_options_default();
            options.backend = 1;
            options.adapter = adapter.c_str();
            options.tune_cache = tune_dir.c_str();
            state.engine = fh.fh_engine_create(&options);
            if (!state.engine) {
                Fail(fmt::format("no FlyHigh engine on {}: {}", adapter, fh.LastError()));
            }
            state.model = fh.fh_model_load(model_dir.c_str());
            if (!state.model) {
                Fail(fmt::format("the model did not load: {}", fh.LastError()));
            }
            FhSessionOptions session = fh.fh_session_options_default(extent.width, extent.height);
            session.keep_static = protect_hud ? 1 : 0;
            FhStreamOptions stream = fh.fh_stream_options_default();
            stream.outputs = static_cast<u32>(MAX_GENERATIONS);
            state.stream = fh.fh_stream_create(state.engine, state.model, &session, &stream);
            if (!state.stream) {
                Fail(fmt::format("no FlyHigh stream at {}x{}: {}", extent.width, extent.height,
                                 fh.LastError()));
            }
            state.Check(fh.fh_stream_warm_up(state.stream, state.engine), "fh_stream_warm_up");
            char info[512]{};
            fh.fh_engine_info(state.engine, info, sizeof(info));
            return std::string(info);
        };

    buffer_size = static_cast<VkDeviceSize>(extent.width) * extent.height * 4;
    staging = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        // The network sees the bytes as they are encoded, the way it was trained.
        .format = rgba_format,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    readback = CreateWrappedBuffer(memory_allocator, buffer_size, MemoryUsage::Download);
    for (auto& upload : uploads) {
        upload = CreateWrappedBuffer(memory_allocator, buffer_size, MemoryUsage::Upload);
    }

    worker = Worker::Start();
    std::string info;
    try {
        info = worker->Run(std::chrono::steady_clock::now() + 5s, init);
    } catch (...) {
        worker->Stop();
        throw;
    }
    LOG_INFO(Render_Vulkan, "FlyHigh CPU staging at {}x{}: {}", extent.width, extent.height, info);
}

FlyHighFrameGen::~FlyHighFrameGen() {
    worker->Stop();
}

size_t FlyHighFrameGen::Process(Scheduler& scheduler, VkImage source, size_t generations) {
    using namespace std::chrono_literals;

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([source, staging_image = *staging, buffer = *readback, extent = extent](
                         vk::CommandBuffer cmdbuf) {
        const std::array before{
            Transition(source, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
            Transition(staging_image, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {}, {}, before);
        cmdbuf.BlitImage(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, WholeBlit(extent),
                         VK_FILTER_NEAREST);
        const std::array after{
            Transition(source, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Transition(staging_image, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT |
                                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                               0, {}, {}, after);
        cmdbuf.CopyImageToBuffer(staging_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer,
                                 RowsCopy(extent));
        const VkBufferMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                               {}, barrier, {});
    });

    scheduler.Finish();
    readback.Invalidate();
    const auto mapped = readback.Mapped().first(static_cast<size_t>(buffer_size));
    std::vector<u8> source_bytes(mapped.begin(), mapped.end());

    const auto budget = !generated_once && generations != 0 ? 2s : 50ms;
    const auto deadline = std::chrono::steady_clock::now() + budget;
    auto [made, frames] = worker->Run(
        deadline, [source = std::move(source_bytes), wanted = std::min(generations, MAX_GENERATIONS),
                   bytes = static_cast<size_t>(buffer_size)](Worker::Engine& state) {
            const auto& fh = state.fh;
            const int slot = fh.fh_stream_next_input(state.stream, nullptr, 0);
            state.Check(slot, "fh_stream_next_input");
            state.Check(fh.fh_stream_write_rgba(state.stream, state.engine, static_cast<u32>(slot),
                                                source.data(), source.size()),
                        "fh_stream_write_rgba");
            const double pts = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - state.start).count();
            FhPush push{};
            state.Check(fh.fh_stream_push(state.stream, state.engine, static_cast<u32>(slot),
                                          pts, &push), "fh_stream_push");

            const size_t count = push.ready ? wanted : 0;
            std::array<std::vector<u8>, MAX_GENERATIONS> frames;
            for (size_t i = 0; i < count; ++i) {
                const float t = static_cast<float>(i + 1) / static_cast<float>(count + 1);
                const u32 out = static_cast<u32>(i);
                const int generated = fh.fh_stream_generate(state.stream, state.engine, t, out);
                state.Check(generated, "fh_stream_generate");
                if (generated == 0) {
                    Fail("FlyHigh did not write an intermediate frame");
                }
                frames[i].resize(bytes);
                const int written = fh.fh_stream_read_rgba(state.stream, state.engine, out,
                                                           frames[i].data(), frames[i].size());
                state.Check(written, "fh_stream_read_rgba");
                if (static_cast<size_t>(written) != bytes) {
                    Fail("FlyHigh returned an incomplete frame");
                }
            }
            return std::pair{count, std::move(frames)};
        });
    for (size_t i = 0; i < made; ++i) {
        std::ranges::copy(frames[i], uploads[i].Mapped().begin());
        uploads[i].Flush();
    }
    generated_once |= made != 0;
    return made;
}

void FlyHighFrameGen::Generate(Scheduler& scheduler, VkImage destination, size_t generation) {
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([destination, staging_image = *staging, buffer = *uploads[generation],
                      extent = extent](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
            Transition(staging_image, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
        cmdbuf.CopyBufferToImage(buffer, staging_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 RowsCopy(extent));
        const std::array before{
            Transition(staging_image, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
            Transition(destination, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT |
                                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {}, {}, before);
        cmdbuf.BlitImage(staging_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, WholeBlit(extent),
                         VK_FILTER_NEAREST);
        cmdbuf.PipelineBarrier(
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
            Transition(destination, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL));
    });
}

} // namespace Vulkan
