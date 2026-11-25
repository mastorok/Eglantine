// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <tuple>
#include <set>
#include <span>
#include <fstream>
#include <mutex>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/renderer_vulkan.h"

#include "common/assert.h"
#include "common/dkb_frame_trace.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/exception.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/control/channel_state.h"
#include "video_core/buffer_cache/indirect_buffer.h"
#include "video_core/native_query_trace.h"
#include "video_core/native_indirect_reset.h"
#include "video_core/render_trace.h"
#include "video_core/indirect_reset.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/ocarina_camera.h"
#include "video_core/renderer_vulkan/blit_image.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/framebuffer_clear_layers.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_ocarina_camera_motion.h"
#include "video_core/renderer_vulkan/vk_fsr3_upscaler.h"
#ifdef EGLANTINE_STREAMLINE
#include "video_core/renderer_vulkan/vk_streamline_dlss.h"
#include "video_core/vulkan_common/streamline.h"
#endif
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_backx.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_render_view.h"
#include "video_core/renderer_vulkan/submission_policy.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_render_view.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/shader_handle_program.h"
#include "video_core/texture_cache/texture_cache_base.h"
#include "video_core/textures/texture.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

// EDEN_NX2_RENDER_OWNS=1: on the back, an entry into GPURender's objects runs on GPURender, the
// back parked (Fritchii ead877857's ONEF_LEND); other threads run it where they are.
#define RENDER_LEND(call)                                                                          \
    if (VideoCommon::GpuSplit::LendsToRender()) [[unlikely]] {                                     \
        VideoCommon::GpuSplit::Get().LendRender([&] { call; });                                    \
        return;                                                                                    \
    }
#define RENDER_LEND_RESULT(call)                                                                   \
    if (VideoCommon::GpuSplit::LendsToRender()) [[unlikely]] {                                     \
        std::optional<std::remove_cvref_t<decltype(call)>> lent_result;                            \
        VideoCommon::GpuSplit::Get().LendRender([&] { lent_result.emplace(call); });               \
        return std::move(*lent_result);                                                            \
    }

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using VideoCommon::ImageViewId;
using VideoCommon::ImageViewType;


namespace {

/// read_gpu_memory: the first address space (the bound channel's first) that maps the whole range,
/// read as the GPU reads it, without flushing GPU caches.
VideoCore::Nx2Debug::GpuReadResult ReadGpuMemory(std::span<Tegra::MemoryManager* const> spaces,
                                                 GPUVAddr address, u64 size) {
    VideoCore::Nx2Debug::GpuReadResult result{};
    if (spaces.empty()) {
        result.error = "no GPU address space is bound yet";
        return result;
    }
    for (Tegra::MemoryManager* const space : spaces) {
        if (space->IsFullyMappedRange(address, size)) {
            result.bytes.resize(size);
            space->ReadBlockUnsafe(address, result.bytes.data(), size);
            result.address_space = space->GetID();
            return result;
        }
    }
    constexpr u64 page_mask = 0xfff;
    const GPUVAddr end = address + size;
    GPUVAddr first = address;
    while (first < end) {
        const GPUVAddr next = (std::min)((first | page_mask) + 1, end);
        if (!spaces.front()->IsFullyMappedRange(first, next - first)) {
            break;
        }
        first = next;
    }
    result.error = fmt::format("GPU address {:#x} is not mapped (the first unmapped byte in the "
                               "bound channel's address space; {} address spaces checked)",
                               first, spaces.size());
    return result;
}

struct DrawParams {
    u32 base_instance;
    u32 num_instances;
    u32 base_vertex;
    u32 num_vertices;
    u32 first_index;
    bool is_indexed;
};

void RecordDirectDraw(Scheduler& scheduler, DrawParams params, std::pair<u64, u64> hashes,
                      std::source_location location = std::source_location::current()) {
    scheduler.Record([&scheduler, params, hashes](vk::CommandBuffer cmdbuf) {
        if (scheduler.GraphicsPipelineMissing()) {
            return;
        }
        scheduler.TraceGpuWork(cmdbuf, hashes.first, hashes.second,
                               {params.num_vertices, params.num_instances, 0}, [&] {
            if (params.is_indexed) {
                cmdbuf.DrawIndexed(params.num_vertices, params.num_instances, params.first_index,
                                   params.base_vertex, params.base_instance);
            } else {
                cmdbuf.Draw(params.num_vertices, params.num_instances, params.base_vertex,
                            params.base_instance);
            }
        });
    }, location);
}

/// A warm direct draw's recording, which GPURender plays (GpuSplit::PostRender).
struct RenderDraw {
    Scheduler* scheduler;
    GraphicsPipeline* pipeline;
    GraphicsPipeline::PreparedBinding binding;
    std::vector<Common::UniqueFunction<void, const vk::CommandBuffer&>> dynamic;
    DrawParams params;

    void operator()() {
        namespace Profile = VideoCommon::Nx2Profile;
        {
            const Profile::Scope scope{Profile::Zone::ConfigureDraw};
            pipeline->RecordPreparedBinding(binding);
        }
        if (!dynamic.empty()) {
            const Profile::Scope scope{Profile::Zone::DynamicState};
            scheduler->Record([commands = std::move(dynamic)](vk::CommandBuffer cmdbuf) {
                for (const auto& command : commands) {
                    command(cmdbuf);
                }
            });
        }
        const Profile::Scope scope{Profile::Zone::DrawRecord};
        const auto& hashes = pipeline->ShaderHashes();
        scheduler->TraceGraphicsPipeline(hashes);
        RecordDirectDraw(*scheduler, params, {hashes[1], hashes[5]});
    }
};

VkViewport GetViewportState(const Device& device, const Maxwell& regs, size_t index, float scale) {
    const auto& src = regs.viewport_transform[index];
    const auto conv = [scale](float value) {
        float new_value = value * scale;
        if (scale < 1.0f) {
            const bool sign = std::signbit(value);
            new_value = std::round(std::abs(new_value));
            new_value = sign ? -new_value : new_value;
        }
        return new_value;
    };
    const float x = conv(src.translate_x - src.scale_x);
    const float width = conv(src.scale_x * 2.0f);
    float y = conv(src.translate_y - src.scale_y);
    float height = conv(src.scale_y * 2.0f);

    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const bool y_negate = !device.IsNvViewportSwizzleSupported() &&
                          src.swizzle.y == Maxwell::ViewportSwizzle::NegativeY;

    if (lower_left) {
        // Flip by surface clip height
        y += conv(static_cast<f32>(regs.surface_clip.height));
        height = -height;
    }

    if (y_negate) {
        // Flip by viewport height
        y += height;
        height = -height;
    }

    const float reduce_z = regs.depth_mode == Maxwell::DepthMode::MinusOneToOne ? 1.0f : 0.0f;
    VkViewport viewport{
        .x = x,
        .y = y,
        .width = width != 0.0f ? width : 1.0f,
        .height = height != 0.0f ? height : 1.0f,
        .minDepth = src.translate_z - src.scale_z * reduce_z,
        .maxDepth = src.translate_z + src.scale_z,
    };
    if (!device.IsExtDepthRangeUnrestrictedSupported()) {
        viewport.minDepth = std::clamp(viewport.minDepth, 0.0f, 1.0f);
        viewport.maxDepth = std::clamp(viewport.maxDepth, 0.0f, 1.0f);
    }
    return viewport;
}

VkRect2D GetScissorState(const Maxwell& regs, size_t index, u32 up_scale = 1, u32 down_shift = 0) {
    const auto& src = regs.scissor_test[index];
    VkRect2D scissor{};
    const auto scale_up = [&](s32 value) -> s32 {
        if (value == 0) {
            return 0U;
        }
        const s32 upset = value * up_scale;
        s32 acumm = 0;
        if ((up_scale >> down_shift) == 0) {
            acumm = upset % 2;
        }
        const s32 converted_value = (value * up_scale) >> down_shift;
        return value < 0 ? std::min<s32>(converted_value - acumm, -1)
                         : std::max<s32>(converted_value + acumm, 1);
    };

    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const s32 clip_height = regs.surface_clip.height;

    // Flip coordinates if lower left
    s32 min_y = lower_left ? (clip_height - src.max_y) : src.min_y.Value();
    s32 max_y = lower_left ? (clip_height - src.min_y) : src.max_y.Value();

    // Bound to render area
    min_y = (std::max)(min_y, 0);
    max_y = (std::max)(max_y, 0);

    if (src.enable) {
        scissor.offset.x = scale_up(src.min_x);
        scissor.offset.y = scale_up(min_y);
        scissor.extent.width = scale_up(src.max_x - src.min_x);
        scissor.extent.height = scale_up(max_y - min_y);
    } else {
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        scissor.extent.width = (std::numeric_limits<s32>::max)();
        scissor.extent.height = (std::numeric_limits<s32>::max)();
    }
    return scissor;
}

// The part of a volume slice a clear writes: the scissor (when the clear uses it) limited to the
// render area and the slice.
VkRect2D VolumeClearRect(const Maxwell& regs, VkExtent2D render_area, VideoCommon::Extent3D slice) {
    VkRect2D rect = regs.clear_control.use_scissor ? GetScissorState(regs, 0)
                                                   : VkRect2D{{0, 0}, {slice.width, slice.height}};
    const auto clamp_axis = [](s32& offset, u32& extent, u32 limit) {
        const s64 begin = std::clamp<s64>(offset, 0, limit);
        const s64 end = std::clamp<s64>(s64{offset} + extent, begin, limit);
        offset = static_cast<s32>(begin);
        extent = static_cast<u32>(end - begin);
    };
    clamp_axis(rect.offset.x, rect.extent.width, (std::min)(render_area.width, slice.width));
    clamp_axis(rect.offset.y, rect.extent.height, (std::min)(render_area.height, slice.height));
    return rect;
}

DrawParams MakeDrawParams(const Tegra::Engines::Maxwell3D::DrawManager::State& draw_state, u32 num_instances, bool is_indexed) {
    DrawParams params{
        .base_instance = draw_state.base_instance,
        .num_instances = num_instances,
        .base_vertex = is_indexed ? draw_state.base_index : draw_state.vertex_buffer.first,
        .num_vertices = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count,
        .first_index = is_indexed ? draw_state.index_buffer.first : 0,
        .is_indexed = is_indexed,
    };
    // 6 triangle vertices per quad, base vertex is part of the index
    // See BindQuadIndexBuffer for more details
    if (draw_state.topology == Maxwell::PrimitiveTopology::Quads) {
        params.num_vertices = (params.num_vertices / 4) * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    } else if (draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
        params.num_vertices = (params.num_vertices - 2) / 2 * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    }
    return params;
}

bool SupportsPrimitiveRestart(VkPrimitiveTopology topology) {
    switch (topology) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST:
        return false;
    default:
        return true;
    }
}

bool IsPrimitiveRestartSupported(const Device& device, VkPrimitiveTopology topology) {
    return ((topology != VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
             device.IsTopologyListPrimitiveRestartSupported()) ||
            SupportsPrimitiveRestart(topology) ||
            (topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
             device.IsPatchListPrimitiveRestartSupported()));
}

/// Polygon offset the game enabled for the primitive type of the draw
bool PolygonOffsetEnabled(const Maxwell& regs, Maxwell::PrimitiveTopology topology) {
    constexpr size_t POINT = 0;
    constexpr size_t LINE = 1;
    constexpr size_t POLYGON = 2;
    static constexpr std::array POLYGON_OFFSET_ENABLE_LUT = {
        POINT,   // Points
        LINE,    // Lines
        LINE,    // LineLoop
        LINE,    // LineStrip
        POLYGON, // Triangles
        POLYGON, // TriangleStrip
        POLYGON, // TriangleFan
        POLYGON, // Quads
        POLYGON, // QuadStrip
        POLYGON, // Polygon
        LINE,    // LinesAdjacency
        LINE,    // LineStripAdjacency
        POLYGON, // TrianglesAdjacency
        POLYGON, // TriangleStripAdjacency
        POLYGON, // Patches
    };
    const std::array enabled_lut{
        regs.polygon_offset_point_enable,
        regs.polygon_offset_line_enable,
        regs.polygon_offset_fill_enable,
    };
    return enabled_lut[POLYGON_OFFSET_ENABLE_LUT[static_cast<u32>(topology)]] != 0;
}

/// Viewport 0 passes position z and w through unswizzled
bool ViewportKeepsDepth(const Maxwell& regs) {
    const auto& swizzle = regs.viewport_transform[0].swizzle;
    return swizzle.z == Maxwell::ViewportSwizzle::PositiveZ &&
           swizzle.w == Maxwell::ViewportSwizzle::PositiveW;
}

/// Depth bias, in units of the depth format, that pushes far plane draws past the far value so
/// that depth clamping puts them exactly on it. A vertex shader that stores one value to position
/// z and w puts every vertex on the far plane, the usual way to draw a sky. NVIDIA fills such a
/// triangle with exactly the far value; AMD's rasterizer lands some pixels a step short, so two
/// sky meshes fight and a dome tested EQUAL or LEQUAL against an earlier one loses in blocks
/// (Mario Kart 8 Deluxe's Sky-High Sundae, #1905).
constexpr float FAR_PLANE_DEPTH_BIAS = 16.0f;

/// Earlier writes must be visible to a dispatch, including its indirect arguments.
void RecordDispatchBarrier(vk::CommandBuffer cmdbuf) {
    static constexpr VkMemoryBarrier READ_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
    };
    cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                           VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, READ_BARRIER);
}
} // Anonymous namespace

/// Reads back the mesh emulation's output for the probe frame and logs how many primitives of
/// PrimitiveVertices() indices it holds: unused slots, primitives with every vertex off screen
/// and bad indices.
void RasterizerVulkan::ReadMeshEmulationProbe() {
    const MeshEmulationProbe probe{mesh_probe};
    mesh_probe.pending = false;
    const size_t num_words{probe.data.size() / sizeof(u32)};
    const auto word = [&](u64 index) -> u32 {
        if (index >= num_words) {
            return 0;
        }
        u32 value;
        std::memcpy(&value, probe.data.data() + index * sizeof(u32), sizeof(value));
        return value;
    };
    const auto position = [&](u64 base, u32 vertex) {
        std::array<f32, 4> result{};
        for (u32 element = 0; element < 4; ++element) {
            const u32 line{probe.position_lines[element]};
            if (line != ~0U) {
                result[element] = std::bit_cast<f32>(word(base + u64{vertex} * probe.vertex_lines +
                                                          line));
            }
        }
        return result;
    };
    u64 triangles{}, degenerate{}, outside{}, bad_index{};
    std::string sample;
    const u32 corners{probe.primitive_vertices};
    for (u32 group = 0; group < probe.groups; ++group) {
        const u64 base{u64{group} * probe.block_words};
        const u64 index_base{base + u64{probe.max_vertices} * probe.vertex_lines};
        for (u32 primitive = 0; primitive < probe.max_primitives; ++primitive) {
            std::array<u32, 3> storage{};
            for (u32 corner = 0; corner < corners; ++corner) {
                storage[corner] = word(index_base + u64{primitive} * corners + corner);
            }
            const std::span<const u32> indices{storage.data(), corners};
            // Slots past the primitive count: 0 0 0 for triangles, the unused marker otherwise.
            const bool unused{
                corners == 3
                    ? indices[0] == indices[1] && indices[1] == indices[2]
                    : indices[0] == Shader::Backend::SPIRV::MESH_EMULATION_UNUSED_VERTEX};
            if (unused) {
                ++degenerate;
                continue;
            }
            ++triangles;
            bool any_bad{};
            bool any_inside{};
            for (const u32 vertex : indices) {
                if (vertex >= probe.max_vertices) {
                    any_bad = true;
                    continue;
                }
                const auto p{position(base, vertex)};
                const f32 w{std::abs(p[3])};
                any_inside |= w > 0.0f && std::abs(p[0]) <= w && std::abs(p[1]) <= w;
            }
            bad_index += any_bad;
            outside += !any_inside;
            if (sample.size() < 400 && triangles <= 2) {
                for (const u32 vertex : indices) {
                    const auto p{position(base, std::min(vertex, probe.max_vertices - 1))};
                    sample += fmt::format(" g{} v{} ({:.3g} {:.3g} {:.3g} {:.3g})", group, vertex,
                                          p[0], p[1], p[2], p[3]);
                }
            }
        }
    }
    LOG_INFO(Render_Vulkan,
             "Mesh emulation probe frame {}: {} groups, {} primitives of {} vertices ({} unused "
             "slots), {} with every vertex off screen, {} with a bad index; first primitives:{}",
             probe.frame, probe.groups, triangles, corners, degenerate, outside, bad_index, sample);
}

const RasterizerVulkan::MeshArena& RasterizerVulkan::AcquireMeshArena() {
    const u64 tick{scheduler.CurrentTick()};
    if (mesh_arena_current < mesh_arenas.size() && mesh_arenas[mesh_arena_current].tick == tick) {
        return mesh_arenas[mesh_arena_current];
    }
    // A finished submission's counter holds the words its draws asked for, including those
    // that did not fit: grow later arenas to hold them.
    const u64 max_size{(std::min)(u64{1} << 30, u64{device.GetMaxStorageBufferRange()})};
    for (MeshArena& arena : mesh_arenas) {
        if (arena.counted || !scheduler.IsFree(arena.tick)) {
            continue;
        }
        arena.counted = true;
        arena.counter.Invalidate();
        u32 words;
        std::memcpy(&words, arena.counter.Mapped().data(), sizeof(words));
        const u64 needed{u64{words} * 4};
        static const bool task_log{std::getenv("EDEN_NX2_MESH_TASK_LOG") != nullptr};
        if (task_log && needed != 0) {
            LOG_INFO(Render_Vulkan, "Mesh arena: a submission asked for {} KiB of its {} MiB",
                     needed >> 10, arena.size >> 20);
        }
        if (needed > arena.size && mesh_arena_size < max_size) {
            mesh_arena_size = (std::min)(std::bit_ceil(needed + needed / 4), max_size);
            LOG_INFO(Render_Vulkan, "Mesh emulation: a submission needed {} MiB of output, arenas "
                                    "grow to {} MiB",
                     needed >> 20, mesh_arena_size >> 20);
        }
    }
    std::erase_if(mesh_arenas, [&](const MeshArena& arena) {
        return arena.counted && arena.size < mesh_arena_size;
    });
    auto it{std::ranges::find_if(mesh_arenas, [](const MeshArena& arena) { return arena.counted; })};
    if (it == mesh_arenas.end()) {
        const auto make_buffer = [&](u64 size, VkBufferUsageFlags usage, MemoryUsage memory) {
            return memory_allocator.CreateBuffer(
                VkBufferCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .size = size,
                    .usage = usage,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                    .queueFamilyIndexCount = 0,
                    .pQueueFamilyIndices = nullptr,
                },
                memory);
        };
        it = mesh_arenas.insert(
            mesh_arenas.end(),
            MeshArena{
                .buffer = make_buffer(mesh_arena_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      MemoryUsage::DeviceLocal),
                .counter = make_buffer(sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       MemoryUsage::Download),
                .size = mesh_arena_size,
            });
    }
    // The GPU is done with this arena, and this submission's commands have not been sent yet.
    const u32 zero{};
    std::memcpy(it->counter.Mapped().data(), &zero, sizeof(zero));
    it->counter.Flush();
    it->tick = tick;
    it->counted = false;
    mesh_arena_current = static_cast<size_t>(std::distance(mesh_arenas.begin(), it));
    return *it;
}

RasterizerVulkan::MeshTaskLogSlot* RasterizerVulkan::AcquireMeshTaskLogSlot() {
    constexpr u64 args_bytes{Shader::Backend::SPIRV::MESH_EMULATION_ARGS_WORDS * 4};
    ReadMeshTaskLog();
    MeshTaskLogSlot* slot{};
    for (MeshTaskLogSlot& candidate : mesh_task_log) {
        if (!candidate.pending) {
            slot = &candidate;
            break;
        }
    }
    if (!slot) {
        if (mesh_task_log.size() >= 256) {
            return nullptr;
        }
        slot = &mesh_task_log.emplace_back();
        slot->buffer = memory_allocator.CreateBuffer(
            VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = args_bytes,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
            },
            MemoryUsage::Download);
    }
    // A draw that never records its commands leaves zeros, which the log reports as unwritten.
    std::memset(slot->buffer.Mapped().data(), 0, args_bytes);
    slot->buffer.Flush();
    return slot;
}

void RasterizerVulkan::ReadMeshTaskLog() {
    for (MeshTaskLogSlot& slot : mesh_task_log) {
        // The draw's commands may sit one submission after the tick it was requested in.
        if (!slot.pending || !scheduler.IsFree(slot.tick + 1)) {
            continue;
        }
        slot.pending = false;
        slot.buffer.Invalidate();
        std::array<u32, Shader::Backend::SPIRV::MESH_EMULATION_ARGS_WORDS> words{};
        std::memcpy(words.data(), slot.buffer.Mapped().data(), sizeof(words));
        using Shader::Backend::SPIRV::MESH_EMULATION_ARGS_BASE;
        using Shader::Backend::SPIRV::MESH_EMULATION_ARGS_REQUESTED;
        using Shader::Backend::SPIRV::MESH_EMULATION_ARGS_TASKS;
        using Shader::Backend::SPIRV::MESH_EMULATION_DRAW_ARGS_OFFSET;
        constexpr u32 draw_args{MESH_EMULATION_DRAW_ARGS_OFFSET / 4};
        LOG_INFO(Render_Vulkan,
                 "Mesh task log: frame {} (odd {}) draw {} task {:016x} mesh {:016x}: {} tasks, "
                 "{} mesh workgroups asked, {} dispatched, draw {} vertices x {} instances, "
                 "first output word {} of {}{}",
                 slot.frame, slot.frame & 1, slot.draw, slot.task_hash, slot.mesh_hash,
                 slot.tasks, words[MESH_EMULATION_ARGS_REQUESTED], words[0],
                 words[draw_args], words[draw_args + 1], words[MESH_EMULATION_ARGS_BASE],
                 slot.capacity,
                 words[MESH_EMULATION_ARGS_TASKS] != slot.tasks ? " (arguments not written)" : "");
    }
}

void RasterizerVulkan::ReleaseMeshEmulationBuffers() {
    for (StagingBufferRef& ref : mesh_draw_buffers) {
        staging_pool.FreeDeferred(ref);
    }
    mesh_draw_buffers.clear();
    if (mesh_arena_used) {
        mesh_arena_used = false;
        mesh_arenas[mesh_arena_current].tick = scheduler.CurrentTick();
    }
}

RasterizerVulkan::RasterizerVulkan(Core::Frontend::EmuWindow& emu_window_, Tegra::GPU& gpu_,
                                   Tegra::MaxwellDeviceMemoryManager& device_memory_,
                                   const Device& device_, MemoryAllocator& memory_allocator_,
                                   StateTracker& state_tracker_, Scheduler& scheduler_)
    : gpu{gpu_}, device_memory{device_memory_}, device{device_},
      memory_allocator{memory_allocator_}, state_tracker{state_tracker_}, scheduler{scheduler_},
      staging_pool(device, memory_allocator, scheduler, nullptr,
                   scheduler.BufferArenaEnabled() ? 2 : 1),
      descriptor_pool(device, scheduler),
      guest_descriptor_queue(device, UpdateDescriptorQueue::GUEST_FRAME_PAYLOAD_SIZE,
                             device.IsExtDescriptorBufferSupported()),
      compute_pass_descriptor_queue(device, UpdateDescriptorQueue::COMPUTE_FRAME_PAYLOAD_SIZE),
      descriptor_buffer_ring(device, memory_allocator),
      blit_image(device, scheduler, state_tracker, descriptor_pool), render_pass_cache(device),
      texture_cache_runtime{
          device,     scheduler,         memory_allocator, staging_pool,
          blit_image, render_pass_cache, descriptor_pool,  compute_pass_descriptor_queue},
      texture_cache(texture_cache_runtime, device_memory),
      buffer_cache_runtime(device, memory_allocator, scheduler, staging_pool,
                           guest_descriptor_queue, compute_pass_descriptor_queue, descriptor_pool),
      buffer_cache(device_memory, buffer_cache_runtime),
      query_cache_runtime(this, device_memory, buffer_cache, device, memory_allocator, scheduler,
                          staging_pool, compute_pass_descriptor_queue, descriptor_pool,
                          texture_cache),
      query_cache(gpu, *this, device_memory, query_cache_runtime),
      pipeline_cache(device_memory, device, scheduler, descriptor_pool, guest_descriptor_queue,
                     descriptor_buffer_ring, render_pass_cache, buffer_cache, texture_cache,
                     gpu.ShaderNotify()),
      accelerate_dma(buffer_cache, texture_cache, scheduler),
      fence_manager(*this, gpu, texture_cache, buffer_cache, query_cache, device, scheduler),
      frame_dump(device, memory_allocator, scheduler) {
    scheduler.SetQueryCache(query_cache);
    buffer_cache.SetPendingCpuWrites(
        [this](const std::function<void(DAddr, size_t)>& callback) {
            gpu.ForEachPendingCpuWrite(callback);
        });
    VideoCommon::GpuSplit::Get().SetIdleHook(
        [](void* argument) {
            auto* const rasterizer = static_cast<RasterizerVulkan*>(argument);
            Scheduler* const owner = &rasterizer->scheduler;
            if (VideoCommon::GpuSplit::LendsToRender()) [[unlikely]] {
                if (rasterizer->PostRenderOp(VideoCommon::GpuSplit::PostedOp::IdleHook,
                                             [owner] { owner->FlushDeferred(); })) {
                    return;
                }
                VideoCommon::GpuSplit::Get().LendRender([owner] { owner->FlushDeferred(); });
                return;
            }
            owner->FlushDeferred();
        },
        this);
    VideoCommon::GpuSplit::Get().SetLendHooks(&LendEnter, &LendLeave, this);
    VideoCommon::GpuSplit::Get().ConfigureRenderFollower(
        scheduler.RenderRecordersEnabled(), scheduler.BufferArenaEnabled(),
        device.GetDriverID() != VK_DRIVER_ID_AMD_PROPRIETARY && !device.IsMoltenVK() &&
        device.IsHostQueryResetSupported());
    VideoCommon::GpuSplit::Get().WatchRenderLocks({&buffer_cache.mutex, &texture_cache.mutex});
    oot_performance_scale =
        Settings::AppliedOcarinaRenderScale() == Settings::NvidiaDlssQuality::Performance;
}

RasterizerVulkan::~RasterizerVulkan() {
    VideoCommon::GpuSplit::Get().ConfigureRenderFollower(false, false, false);
    VideoCommon::GpuSplit::Get().WatchRenderLocks({});
    VideoCommon::GpuSplit::Get().SetIdleHook(nullptr, nullptr);
    // The channels are gone by now, and submitting invalidates the state tracker's flags.
    state_tracker.ClearChannel();
    scheduler.WaitWorker();
    scheduler.Finish();
    buffer_cache_runtime.ClearMultiRangeBuffers();
}

GraphicsPipeline* RasterizerVulkan::TakeReusableDraw(bool is_indexed, bool is_indirect) {
    namespace FastDraw = VideoCommon::Nx2FastDraw;
    const ReusableDraw draw = std::exchange(reusable_draw, ReusableDraw{});
    FastDraw::TakeDelta(taken_delta);
    const FastDraw::Delta& delta = taken_delta;
    // A draw that stopped before binding its uniforms never handed its writes to the keeps.
    uniform_writes_lost |= std::exchange(taken_delta_unconsumed, true);
    const auto full = [&](FullDraw reason) -> GraphicsPipeline* {
        ++fast_draw_stats.full[static_cast<size_t>(reason)];
        return nullptr;
    };
    if (draw.pipeline == nullptr) {
        return full(FullDraw::NoCandidate);
    }
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    if (is_indirect || draw.is_indexed != is_indexed || draw.maxwell3d != maxwell3d ||
        draw.topology != draw_state.topology || !draw_state.inline_index_draw_indexes.empty() ||
        frame_dump.Capturing()) {
        return full(FullDraw::Kind);
    }
    if (draw.serial != FastDraw::Serial()) {
        const u32 last_break = FastDraw::g_last_break.load(std::memory_order_relaxed);
        ++fast_draw_stats.breaks[last_break & 0xff];
        if (const u32 method = last_break >> 8; method < Maxwell::NUM_REGS) {
            ++fast_draw_stats.break_methods[method];
        }
        return full(FullDraw::StateChanged);
    }
    if (draw.map_serial != gpu_memory->MapSerial()) {
        ++fast_draw_stats.breaks[static_cast<size_t>(FastDraw::Reason::Map)];
        return full(FullDraw::StateChanged);
    }
    if (draw.tick != scheduler.CurrentTick()) {
        return full(FullDraw::Submitted);
    }
    if (draw.render_pass != scheduler.RenderPassSerial() || !scheduler.IsRenderPassActive() ||
        scheduler.BoundGraphicsPipeline() != draw.pipeline) {
        return full(FullDraw::RenderPass);
    }
    if (!draw.pipeline->TakesDeltaDraws()) {
        ++fast_draw_stats.delta_blockers[static_cast<size_t>(draw.pipeline->DeltaDrawBlocker())];
        return full(FullDraw::DeltaPipeline);
    }
    // Vertex attribute formats are draw parameters with dynamic vertex input (UpdateVertexInput
    // records them for every draw); only the type class of each attribute is in the pipeline key.
    if (maxwell3d->dirty.flags[Dirty::VertexInput] &&
        (!draw.pipeline->HasDynamicVertexInput() ||
         FixedPipelineState::DynamicAttributeTypes(maxwell3d->regs) !=
             draw.pipeline->DynamicAttributeTypes())) {
        return full(FullDraw::VertexInput);
    }
    const auto& handles = draw.pipeline->HandleConstantBuffers();
    for (size_t stage = 0; stage < handles.size() && !delta.Empty(); ++stage) {
        if ((delta.binds[stage] & handles[stage]) != 0) {
            return full(FullDraw::HandleBind);
        }
        if (delta.num_writes == 0) {
            continue;
        }
        const auto& cbufs = maxwell3d->state.shader_stages[stage].const_buffers;
        for (u32 mask = handles[stage]; mask != 0; mask &= mask - 1) {
            const u32 slot = std::countr_zero(mask);
            const auto& cbuf = cbufs[slot];
            // Only the words the handles are read from: draws write their own data next to them.
            const auto [begin, end] = draw.pipeline->HandleRange(stage, slot);
            FastDraw::DeviceSpans& spans = handle_spans[stage][slot];
            spans.Update(*gpu_memory, cbuf.address + begin, end - begin);
            if (delta.Reaches(cbuf.address + begin, end - begin, spans)) {
                if (delta.ReachesWords(cbuf.address, draw.pipeline->HandleWords(stage, slot),
                                       spans)) {
                    return full(FullDraw::HandleData);
                }
                taken_delta.no_uniform_keeps = true;
            }
        }
    }
    taken_draw = draw;
    return draw.pipeline;
}

bool RasterizerVulkan::DeltaStillHolds(u64 serial, u64 map_serial) {
    const auto full = [&](FullDraw reason) {
        ++fast_draw_stats.full[static_cast<size_t>(reason)];
        return false;
    };
    // TakeReusableDraw ran before the caches were locked, while other threads could still write,
    // unmap or delete what the delta draw keeps.
    if (taken_draw.serial != serial || taken_draw.map_serial != map_serial) {
        return full(FullDraw::StateChanged);
    }
    if (taken_draw.tick != scheduler.CurrentTick()) {
        return full(FullDraw::Submitted);
    }
    if (taken_draw.render_pass != scheduler.RenderPassSerial() || !scheduler.IsRenderPassActive()) {
        return full(FullDraw::RenderPass);
    }
    // A delta draw prepares none of the images it samples: one with an upload pending or a newer
    // alias to copy in (a render target aliasing it, written by the draw before) needs Configure.
    if (taken_draw.pipeline->DeltaImagesChange()) {
        return full(FullDraw::SampledImages);
    }
    // Handles computed at draw time come from memory the delta doesn't track (storage buffers).
    if (taken_draw.pipeline->ComputedHandlesChanged()) {
        return full(FullDraw::ComputedHandles);
    }
    return true;
}

void RasterizerVulkan::NoteResolvedDraw(GraphicsPipeline* pipeline, bool is_indexed,
                                        bool is_indirect, u64 serial, u64 map_serial) {
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const auto topology = draw_state.topology;
    const auto ineligible = [&]() -> std::optional<Ineligible> {
        if (is_indirect) return Ineligible::Indirect;
        if (mesh_draw) return Ineligible::Mesh;
        if (!pipeline->IsBuilt()) return Ineligible::NotBuilt;
        if (pipeline->WritesImages()) return Ineligible::WritesImages;
        if (maxwell3d->regs.transform_feedback_enabled != 0) return Ineligible::TransformFeedback;
        if (!draw_state.inline_index_draw_indexes.empty()) return Ineligible::InlineIndex;
        if (topology == Maxwell::PrimitiveTopology::Quads ||
            topology == Maxwell::PrimitiveTopology::QuadStrip) {
            return Ineligible::Quads;
        }
        if (buffer_cache_runtime.AnyVertexRepacked()) return Ineligible::Repacked;
        if (!scheduler.IsRenderPassActive()) return Ineligible::NoRenderPass;
        if (scheduler.BoundGraphicsPipeline() != pipeline) return Ineligible::PipelineNotBound;
        if (!VideoCommon::GpuSplit::CanReuseDrawState()) return Ineligible::OffBackThread;
        return std::nullopt;
    }();
    if (ineligible) {
        ++fast_draw_stats.ineligible[static_cast<size_t>(*ineligible)];
        return;
    }
    reusable_draw = ReusableDraw{
        .pipeline = pipeline,
        .maxwell3d = maxwell3d,
        .serial = serial,
        .map_serial = map_serial,
        .tick = scheduler.CurrentTick(),
        .render_pass = scheduler.RenderPassSerial(),
        .topology = topology,
        .is_indexed = is_indexed,
    };
}

void RasterizerVulkan::ReportFastDraws() {
    const u64 now = VideoCommon::Nx2Profile::Now();
    FastDrawStats& stats = fast_draw_stats;
    if (stats.window_start == 0) {
        stats.window_start = now;
        return;
    }
    if (now - stats.window_start < 10'000'000'000ULL) {
        return;
    }
    // Logged only with EDEN_NX2_GPU_PROFILE or EDEN_NX2_FAST_DRAW_VERIFY.
    if (!VideoCommon::Nx2Profile::Enabled() && !fast_draw_verify) {
        stats = FastDrawStats{.window_start = now};
        return;
    }
    const double seconds = static_cast<double>(now - stats.window_start) / 1e9;
    const auto per_second = [&](FullDraw reason) {
        return static_cast<double>(stats.full[static_cast<size_t>(reason)]) / seconds;
    };
    LOG_INFO(Render_Vulkan,
             "Fast draw per second: {:.0f} delta | full: {:.0f} after an ineligible draw, {:.0f} "
             "different kind, {:.0f} state changed, {:.0f} after a submit, {:.0f} render pass or "
             "pipeline changed, {:.0f} pipeline takes no delta draws, {:.0f} handle buffer bound, "
             "{:.0f} handle data written, {:.0f} sampled image to prepare, {:.0f} computed handle "
             "changed, {:.0f} vertex attribute types changed",
             static_cast<double>(stats.delta) / seconds, per_second(FullDraw::NoCandidate),
             per_second(FullDraw::Kind), per_second(FullDraw::StateChanged),
             per_second(FullDraw::Submitted), per_second(FullDraw::RenderPass),
             per_second(FullDraw::DeltaPipeline), per_second(FullDraw::HandleBind),
             per_second(FullDraw::HandleData), per_second(FullDraw::SampledImages),
             per_second(FullDraw::ComputedHandles), per_second(FullDraw::VertexInput));
    static constexpr std::array<const char*, static_cast<size_t>(Ineligible::Count)>
        ineligible_names{"indirect",       "mesh",          "not built",     "writes images",
                         "transform feedback", "inline index", "quads",       "repacked vertices",
                         "no render pass", "pipeline not bound", "thread ownership gate"};
    std::string ineligible;
    for (size_t reason = 0; reason < stats.ineligible.size(); ++reason) {
        if (stats.ineligible[reason] != 0) {
            ineligible += fmt::format(" {} {:.0f}", ineligible_names[reason],
                                      static_cast<double>(stats.ineligible[reason]) / seconds);
        }
    }
    const auto micros = [&](DrawPath path) {
        const size_t index = static_cast<size_t>(path);
        return stats.prepare_draws[index] == 0
                   ? 0.0
                   : static_cast<double>(stats.prepare_ns[index]) / 1e3 /
                         static_cast<double>(stats.prepare_draws[index]);
    };
    const u64 all_draws = stats.prepare_draws[0] + stats.prepare_draws[1];
    const u64 all_ns = stats.prepare_ns[0] + stats.prepare_ns[1];
    static constexpr std::array<const char*,
                                static_cast<size_t>(GraphicsPipeline::DeltaBlocker::Count)>
        blocker_names{"none", "writes images", "mesh", "not configured"};
    std::string blockers;
    for (size_t blocker = 0; blocker < stats.delta_blockers.size(); ++blocker) {
        if (stats.delta_blockers[blocker] != 0) {
            blockers += fmt::format(" {} {:.0f}", blocker_names[blocker],
                                    static_cast<double>(stats.delta_blockers[blocker]) / seconds);
        }
    }
    ineligible += " | pipelines without delta draws:" + (blockers.empty() ? std::string{" none"} : blockers);
    LOG_INFO(Render_Vulkan,
             "Fast draw ineligible per second:{} | prepare us per draw: full {:.2f}, delta {:.2f}, "
             "all {:.2f} ({:.0f} draws/s)",
             ineligible, micros(DrawPath::Full), micros(DrawPath::Delta),
             all_draws == 0 ? 0.0 : static_cast<double>(all_ns) / 1e3 / static_cast<double>(all_draws),
             static_cast<double>(all_draws) / seconds);
    if (fast_draw_verify) {
        LOG_INFO(Render_Vulkan,
                 "Fast draw verify per second: {:.0f} checked, {:.0f} pipeline mismatches, {:.0f} "
                 "view or sampler mismatches, {:.0f} kept buffer binding mismatches, {:.0f} "
                 "vertex input mismatches (totals {} {} {} {} {})",
                 static_cast<double>(stats.verified) / seconds,
                 static_cast<double>(stats.pipeline_mismatches) / seconds,
                 static_cast<double>(stats.view_mismatches) / seconds,
                 static_cast<double>(stats.binding_mismatches) / seconds,
                 static_cast<double>(stats.vertex_input_mismatches) / seconds, stats.verified,
                 stats.pipeline_mismatches, stats.view_mismatches, stats.binding_mismatches,
                 stats.vertex_input_mismatches);
    }
    std::string breaks;
    for (size_t reason = 0; reason < stats.breaks.size(); ++reason) {
        if (stats.breaks[reason] != 0) {
            breaks += fmt::format(" {} {:.0f}", VideoCommon::Nx2FastDraw::reason_names[reason],
                                  static_cast<double>(stats.breaks[reason]) / seconds);
        }
    }
    std::vector<std::pair<u32, u32>> methods;
    for (u32 method = 1; method < stats.break_methods.size(); ++method) {
        if (stats.break_methods[method] != 0) {
            methods.emplace_back(stats.break_methods[method], method);
        }
    }
    const size_t shown = std::min<size_t>(methods.size(), 12);
    std::partial_sort(methods.begin(), methods.begin() + shown, methods.end(), std::greater{});
    std::string top;
    for (size_t i = 0; i < shown; ++i) {
        top += fmt::format(" {:#x}:{:.0f}", methods[i].second,
                           static_cast<double>(methods[i].first) / seconds);
    }
    LOG_INFO(Render_Vulkan, "Fast draw state changes per second by last break:{} | by method:{}",
             breaks, top);
    stats = FastDrawStats{.window_start = now};
}

RasterizerVulkan::DrawReuse RasterizerVulkan::NoteDrawReuse(const GraphicsPipeline& pipeline,
                                                            bool is_indexed) {
    u64 key = buffer_cache.KeptGraphicsBindingsHash(true, true, false);
    const auto mix = [&key](u64 value) { key = (key ^ value) * 0x100000001b3ULL; };
    mix(reinterpret_cast<u64>(&pipeline));
    if (const Framebuffer* const framebuffer = texture_cache.GetFramebuffer()) {
        mix(reinterpret_cast<u64>(framebuffer->Handle()));
        mix(reinterpret_cast<u64>(framebuffer->RenderPass()));
    }
    for (const auto& image : pipeline.SampledImages()) {
        mix((u64{image.id.index} << 32) | image.sampler.index);
    }
    const auto& streams = maxwell3d->regs.vertex_streams;
    for (u32 index = 0; index < streams.size(); ++index) {
        if (streams[index].IsEnabled()) {
            mix((u64{index} << 32) | streams[index].stride);
        }
    }
    mix(is_indexed ? 1 : 0);
    if (is_indexed) {
        const auto& index_buffer = maxwell3d->draw_manager.draw_state.index_buffer;
        mix(index_buffer.StartAddress());
        mix(static_cast<u64>(index_buffer.format));
    }
    if (draw_reuse.previous_frame.contains(key)) {
        draw_reuse.frame.insert(key);
        return DrawReuse::Hit;
    }
    return draw_reuse.frame.insert(key).second ? DrawReuse::Miss : DrawReuse::Repeat;
}

void RasterizerVulkan::TickDrawReuse() {
    auto& probe = draw_reuse;
    std::swap(probe.previous_frame, probe.frame);
    probe.frame.clear();
    ++probe.frames;
    const u64 now = VideoCommon::Nx2Profile::Now();
    if (probe.window_start == 0) {
        probe.window_start = now;
    }
    if (now - probe.window_start < 10'000'000'000ULL) {
        return;
    }
    const auto churn = [&] {
        std::scoped_lock lock{texture_cache.mutex};
        return texture_cache.TakeChurn();
    }();
    const auto [memo_hits, memo_misses] = GraphicsPipeline::TakeTextureMemoCounts();
    const double frames = static_cast<double>(probe.frames);
    const auto per_frame = [frames](u64 count) { return static_cast<double>(count) / frames; };
    static constexpr std::array<const char*, static_cast<size_t>(DrawReuse::Count)> names{
        "hit", "same-frame repeat", "miss"};
    std::string classes;
    u64 classed = 0;
    for (size_t index = 0; index < names.size(); ++index) {
        const auto& draws = probe.draws[index];
        classed += draws[0] + draws[1];
        const auto& ns = probe.prepare_ns[index];
        const auto micros = [&](DrawPath path) {
            const size_t i = static_cast<size_t>(path);
            return draws[i] == 0 ? 0.0
                                 : static_cast<double>(ns[i]) / 1e3 / static_cast<double>(draws[i]);
        };
        classes += fmt::format("{} {} full {:.0f} ({:.2f} us) delta {:.0f} ({:.2f} us)",
                               index == 0 ? "" : ",", names[index],
                               per_frame(draws[static_cast<size_t>(DrawPath::Full)]),
                               micros(DrawPath::Full),
                               per_frame(draws[static_cast<size_t>(DrawPath::Delta)]),
                               micros(DrawPath::Delta));
    }
    LOG_INFO(Render_Vulkan,
             "Draw reuse per frame over {} frames:{}; gate {:.3f} ms/frame (full-path hits over "
             "1.1 us); probe {:.2f} us/draw (left out of the times); {:.1f} images created, "
             "{:.1f} deleted, {:.1f} joins, {:.1f} TIC table invalidations, {:.1f} TIC/TSC table "
             "moves; texture memo {:.1f} hits, {:.1f} misses",
             probe.frames, classes, per_frame(probe.gate_ns) / 1e6,
             classed == 0 ? 0.0
                          : static_cast<double>(probe.probe_ns) / 1e3 / static_cast<double>(classed),
             per_frame(churn.created), per_frame(churn.deleted), per_frame(churn.joins),
             per_frame(churn.invalidations), per_frame(churn.refreshes), per_frame(memo_hits),
             per_frame(memo_misses));
    probe.draws = {};
    probe.prepare_ns = {};
    probe.gate_ns = 0;
    probe.probe_ns = 0;
    probe.frames = 0;
    probe.window_start = now;
}

// Experimental adaptation of PokeDinner commit 76df4591a747: execute the reset shader's
// indirect record writes on the host until general GPU-address translation is available.
bool RasterizerVulkan::TryNx2IndirectReset(bool is_indexed, u32 instance_count) {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_RESETFIX");
        return !value || std::string_view{value} != "0";
    }();
    const auto& regs = maxwell3d->regs;
    const auto& draw = maxwell3d->draw_manager.draw_state;
    if (!enabled || is_indexed || instance_count != 1 || regs.IsMeshEnabled() ||
        draw.topology != Maxwell::PrimitiveTopology::Points ||
        !maxwell3d->GetNativeMacroEngine().native_turing) {
        return false;
    }
    const GPUVAddr program =
        (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
    std::array<u32, VideoCommon::NativeIndirectReset::Program.size()> code{};
    if (!gpu_memory->IsFullyMappedRange(program + 128, sizeof(code))) {
        return false;
    }
    gpu_memory->ReadBlock(program + 128, code.data(), sizeof(code));
    if (!VideoCommon::NativeIndirectReset::Matches(code)) {
        return false;
    }
    VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::IndirectReset);
    FlushWork();
    gpu_memory->FlushCaching();
    const auto reject = [](std::string_view reason) {
        static u32 reported_rejections{};
        if (reported_rejections++ < 8) {
            LOG_WARNING(HW_GPU, "Experimental indirect reset skipped: {}", reason);
        }
        return false;
    };
    const auto in_gpu_range = [this](u64 address, size_t size) {
        return size != 0 && address <= (std::numeric_limits<u64>::max)() - size &&
               gpu_memory->IsWithinGPUAddressRange(address) &&
               gpu_memory->IsWithinGPUAddressRange(address + size - 1);
    };
    const auto& cbuf = maxwell3d->state.shader_stages[0].const_buffers[0];
    if (!cbuf.enabled || cbuf.size < 0x120 || !in_gpu_range(cbuf.address, 0x120) ||
        !gpu_memory->IsFullyMappedRange(cbuf.address + 0x110, 16)) {
        return reject("descriptor unavailable");
    }
    std::array<u64, 2> descriptor{};
    gpu_memory->ReadBlock(cbuf.address + 0x110, descriptor.data(), sizeof(descriptor));
    const u64 list = descriptor[0];
    const u64 first = draw.vertex_buffer.first;
    const u64 count = draw.vertex_buffer.count;
    const auto range = VideoCommon::IndirectResetRange(list, static_cast<u32>(descriptor[1]),
                                                        first, count);
    if (!range || !in_gpu_range(range->begin, range->size) ||
        !gpu_memory->IsFullyMappedRange(range->begin, range->size)) {
        return reject("list outside mapped descriptor");
    }
    std::vector<u64> pointers(count);
    gpu_memory->ReadBlock(range->begin, pointers.data(), range->size);
    static constexpr std::array<u32, 5> record{0, 1, 0, 0, 0};
    for (const u64 pointer : pointers) {
        if (!pointer) {
            return reject("record is null");
        }
        if (!in_gpu_range(pointer, sizeof(record)) ||
            !gpu_memory->IsFullyMappedRange(pointer, sizeof(record))) {
            return reject("record not mapped");
        }
        if (pointer < range->begin + range->size && range->begin < pointer + sizeof(record)) {
            return reject("record overlaps pointer list");
        }
        const auto segments = gpu_memory->GetSubmappedRange(pointer, sizeof(record));
        if (segments.size() != 1 || segments[0].first != pointer ||
            segments[0].second != sizeof(record)) {
            return reject("record crosses a mapping boundary");
        }
    }
    const std::span<const u8> bytes{reinterpret_cast<const u8*>(record.data()), sizeof(record)};
    for (const u64 pointer : pointers) {
        AccelerateInlineToMemory(pointer, bytes.size(), bytes);
    }
    static u32 reported{};
    if (reported++ < 5) {
        LOG_INFO(HW_GPU, "Experimental indirect reset: {} records", count);
    }
    gpu.TickWork();
    return true;
}

/// Common start of every draw: resolves the pipeline and the bound resources, then calls
/// `draw_func` to record the draw itself. A draw that cannot be prepared is skipped and logged.
template <typename Func>
void RasterizerVulkan::PrepareDraw(bool is_indexed, Func&& draw_func, bool is_indirect,
                                   std::optional<u32> direct_instances) {
    auto& split = VideoCommon::GpuSplit::Get();
    std::optional<VideoCommon::Nx2RenderView::DrawCheck> view_check;
    if (VideoCommon::Nx2RenderView::Enabled()) {
        view_check.emplace(*maxwell3d, std::exchange(unlend_verify_draw, false));
        if (auto* const view = view_check->View()) {
            VerifyRenderViewBuild(*view, *maxwell3d, pipeline_cache.Features());
        }
    }
    DrawReuse reuse = DrawReuse::Count;
    DrawPath reuse_path = DrawPath::Full;
    u64 reuse_probe_ns = 0;
    const u64 reuse_start = DrawReuseProbeEnabled() ? VideoCommon::Nx2Profile::Now() : 0;
    SCOPE_EXIT {
        if (reuse == DrawReuse::Count) {
            return;
        }
        const u64 ns = VideoCommon::Nx2Profile::Now() - reuse_start - reuse_probe_ns;
        draw_reuse.probe_ns += reuse_probe_ns;
        const size_t index = static_cast<size_t>(reuse);
        ++draw_reuse.draws[index][static_cast<size_t>(reuse_path)];
        draw_reuse.prepare_ns[index][static_cast<size_t>(reuse_path)] += ns;
        if (reuse == DrawReuse::Hit && reuse_path == DrawPath::Full && ns > 1100) {
            draw_reuse.gate_ns += ns - 1100;
        }
    };

    if (Common::DkbFrameTrace::Enabled()) {
        ++trace_draws;
    }

    SCOPE_EXIT {
        gpu.TickWork();
    };
    bool submitted = false;
    SCOPE_EXIT {
        auto& native = maxwell3d->GetNativeMacroEngine();
        if (native.native_turing && !submitted) {
            native.native_execution_failed = true;
            const auto& regs = maxwell3d->regs;
            const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) |
                           regs.pipelines[1].reserved[1];
            const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) |
                           regs.pipelines[5].reserved[1];
            LOG_DEBUG(HW_GPU, "Native graphics draw not recorded: frame={} vs={:#x} ps={:#x}",
                      render_trace_frame, vs, ps);
        }
    };
    std::optional<RenderDraw> pending_render;
    SCOPE_EXIT {
        if (pending_render) {
            split.PostRender(std::move(*pending_render));
        }
    };
    SCOPE_EXIT {
        ReleaseMeshEmulationBuffers();
    };
    namespace Profile = VideoCommon::Nx2Profile;
    namespace FastDraw = VideoCommon::Nx2FastDraw;
    // Latched here, on the thread that draws: turning fast draws on starts with a full resolve,
    // and with them off nothing below runs.
    if (const bool on = FastDraw::Enabled(); on != fast_draw_on) {
        fast_draw_on = on;
        reusable_draw = {};
        FastDraw::g_delta = {};
        uniform_writes_lost = true;
        fast_draw_stats = {};
    }
    fast_draw_verify = fast_draw_on && FastDraw::VerifyEnabled();
    // The prepare time per draw (a fenced time stamp at each end): ReportFastDraws logs it only
    // with the profile or the verify on.
    const bool time_fast_draw = fast_draw_on && (Profile::Enabled() || fast_draw_verify);
    const u64 fast_draw_start = time_fast_draw ? Profile::Now() : 0;
    {
        Profile::Scope scope{Profile::Zone::FlushWork};
        FlushWork();
        gpu_memory->FlushCaching();
    }
    if (frame_dump.Capturing()) {
        frame_dump.DrainIfFull();
    }

    const auto report_failure = [&](std::string_view reason) {
        TraceDrawFailure(reason);
        static std::set<std::tuple<u64, u64, std::string_view>> seen;
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        if (seen.size() < 64 && seen.emplace(vs, ps, reason).second) {
            LOG_WARNING(HW_GPU, "Draw skipped ({}): vs={:#x} ps={:#x} rt0={:#x} {}x{}", reason, vs, ps,
                        regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height);
        }
    };
    GraphicsPipeline* const delta_pipeline =
        fast_draw_on ? TakeReusableDraw(is_indexed, is_indirect) : nullptr;
    // EDEN_NX2_FAST_DRAW_VERIFY: resolve in full and compare with what the delta path keeps.
    GraphicsPipeline* const verify_pipeline = fast_draw_verify ? delta_pipeline : nullptr;
    GraphicsPipeline* const pipeline{delta_pipeline && !verify_pipeline ? delta_pipeline : [&] {
        Profile::Scope scope{Profile::Zone::PipelineLookup};
        return pipeline_cache.CurrentGraphicsPipeline();
    }()};
    if (!pipeline) {
        report_failure("no pipeline");
        TraceRenderState(is_indexed, "no-pipeline", nullptr);
        // Mesh emulation survey: how many task/mesh draws a device without
        // VK_EXT_mesh_shader drops, how many are indirect and how many groups they launch.
        if (maxwell3d->regs.IsMeshEnabled() && !device.IsExtMeshShaderSupported()) {
            static u64 draws{}, indirect{}, groups{}, max_groups{};
            ++draws;
            if (is_indirect) {
                ++indirect;
            } else {
                const u64 count{maxwell3d->draw_manager.draw_state.vertex_buffer.count};
                groups += count;
                max_groups = std::max(max_groups, count);
            }
            if (draws == 1 || draws % 5000 == 0) {
                LOG_WARNING(Render_Vulkan,
                            "Mesh emulation survey: {} mesh draws dropped, {} indirect, {} groups "
                            "in direct ones (max {} per draw), frame {}",
                            draws, indirect, groups, max_groups, render_trace_frame);
            }
        }
        return;
    }
    if (VideoCommon::OotNetwork::BypassEnabled(Settings::GetCurrentProgramID())) {
        TrackOotTonemappedImage();
    }
    // Task/mesh pipelines: the draw's vertex count is the number of task workgroups.
    mesh_draw = pipeline->IsMeshPipeline();
    // EDEN_NX2_NO_MESH=1: skip task/mesh draws as before they were supported (A/B comparisons).
    // EDEN_NX2_NO_MESH_FILE=<path>: skip them while that file exists (checked every 1024 draws),
    // to compare the same scene with and without them in one run.
    static const bool no_mesh_always{std::getenv("EDEN_NX2_NO_MESH") != nullptr};
    static const char* const no_mesh_file{std::getenv("EDEN_NX2_NO_MESH_FILE")};
    static bool no_mesh{no_mesh_always};
    static u32 no_mesh_check{};
    if (no_mesh_file && (no_mesh_check++ % 1024) == 0) {
        no_mesh = std::filesystem::exists(no_mesh_file);
    }
    const bool emulated_mesh{mesh_draw && pipeline->IsEmulatedMesh()};
    if (mesh_draw &&
        (no_mesh || (!device.IsExtMeshShaderSupported() && !emulated_mesh) ||
         (emulated_mesh && is_indirect))) {
        report_failure(no_mesh         ? "EDEN_NX2_NO_MESH"
                       : emulated_mesh ? "indirect emulated mesh draw"
                                       : "no VK_EXT_mesh_shader");
        TraceRenderState(is_indexed, "mesh-skipped", nullptr);
        return;
    }
    // Mesh emulation: one output block per mesh workgroup, written by the pipeline's compute
    // dispatch and drawn as groups * vertices-per-group triangle list vertices.
    mesh_emulation_groups = 0;
    mesh_emulation_vertices_per_group = 0;
    if (emulated_mesh) {
        // The same group count the native path hands vkCmdDrawMeshTasksEXT (MakeDrawParams:
        // the index count for indexed draws), within the device's dispatch limit.
        const auto& draw_state{maxwell3d->draw_manager.draw_state};
        const u32 groups{(std::min)(MakeDrawParams(draw_state, 1, is_indexed).num_vertices,
                                    device.GetMaxComputeWorkGroupCount()[0])};
        // Mesh emulation statistics: draws with and without groups, and the frames with any
        // groups (the game culls on the GPU and sets the counts itself).
        {
            static u64 draws{}, empty{}, total_groups{}, frames{}, frames_drawn{};
            static u64 last_frame{~0ULL};
            static bool frame_drawn{};
            if (render_trace_frame != last_frame) {
                if (last_frame != ~0ULL) {
                    ++frames;
                    frames_drawn += frame_drawn;
                }
                last_frame = render_trace_frame;
                frame_drawn = false;
            }
            ++draws;
            empty += groups == 0;
            total_groups += groups;
            frame_drawn |= groups != 0;
            if (draws % 2000 == 0) {
                LOG_INFO(Render_Vulkan,
                         "Mesh emulation draws: {} ({} with no groups), {} groups; frames with "
                         "emulated draws: {} of {}",
                         draws, empty, total_groups, frames_drawn, frames);
            }
        }
        static const bool task_log{std::getenv("EDEN_NX2_MESH_TASK_LOG") != nullptr};
        if (task_log && !mesh_task_log.empty()) {
            ReadMeshTaskLog();
        }
        // Read the pending probe before Request can hand its buffer out again.
        if (mesh_probe.pending && scheduler.IsFree(mesh_probe.tick)) {
            ReadMeshEmulationProbe();
        }
        // EDEN_NX2_MESH_EMULATION_PROBE=1: every 60 frames, read one draw's output back (from a
        // host-visible download buffer, as device-local ones are not mapped on every GPU).
        static const bool probe{std::getenv("EDEN_NX2_MESH_EMULATION_PROBE") != nullptr};
        const bool has_task{pipeline->MeshEmulationHasTask()};
        const bool probe_draw{probe && !has_task && !mesh_probe.pending && groups != 0 &&
                              render_trace_frame % 60 == 0};
        // With a task program the draw's groups are task workgroups. The mesh workgroups they ask
        // for are only known on the GPU, so their output goes to the submission's arena.
        const u64 block_bytes{pipeline->MeshEmulationBlockBytes()};
        StagingBufferRef output{};
        u64 bytes{};
        if (has_task) {
            const MeshArena& arena{AcquireMeshArena()};
            mesh_arena_used = true;
            bytes = arena.size;
            pipeline->SetMeshEmulationOutput(*arena.buffer, 0, bytes, groups);
        } else {
            bytes = (std::max)(u64{groups} * block_bytes, u64{4});
            output = staging_pool.Request(
                bytes, probe_draw ? MemoryUsage::Download : MemoryUsage::DeviceLocal, true);
            mesh_draw_buffers.push_back(output);
            pipeline->SetMeshEmulationOutput(output.buffer, output.offset, bytes, groups);
        }
        mesh_emulation_indirect = {};
        if (mesh_task_probe.pending && scheduler.IsFree(mesh_task_probe.tick)) {
            mesh_task_probe.pending = false;
            std::array<u32, Shader::Backend::SPIRV::MESH_EMULATION_ARGS_WORDS> words{};
            std::memcpy(words.data(), mesh_task_probe.args.data(), sizeof(words));
            LOG_INFO(Render_Vulkan,
                     "Mesh emulation task probe: {} tasks asked for {} mesh workgroups, {} "
                     "dispatched at arena word {} (arena {} words)",
                     mesh_task_probe.tasks,
                     words[Shader::Backend::SPIRV::MESH_EMULATION_ARGS_REQUESTED], words[0],
                     words[Shader::Backend::SPIRV::MESH_EMULATION_ARGS_BASE],
                     mesh_task_probe.capacity);
        }
        if (has_task && groups != 0) {
            const bool task_probe{probe && !mesh_task_probe.pending &&
                                  render_trace_frame % 60 == 0};
            const auto request = [&](u64 size, MemoryUsage usage = MemoryUsage::DeviceLocal) {
                size = (std::max)(size, u64{16});
                const StagingBufferRef ref{staging_pool.Request(size, usage, true)};
                mesh_draw_buffers.push_back(ref);
                return std::make_pair(GraphicsPipeline::MeshEmulationBuffer{ref.buffer, ref.offset,
                                                                            size},
                                      ref.mapped_span.subspan(
                                          (std::min)(ref.offset, u64{ref.mapped_span.size()})));
            };
            const auto payload{
                request(u64{groups} * pipeline->MeshEmulationPayloadStride() * 4).first};
            const auto offsets{request(u64{groups} * 4).first};
            constexpr u64 args_bytes{Shader::Backend::SPIRV::MESH_EMULATION_ARGS_WORDS * 4};
            MeshTaskLogSlot* const log_slot{task_log && render_trace_frame % 120 < 2
                                                ? AcquireMeshTaskLogSlot()
                                                : nullptr};
            const auto [args, args_span] =
                log_slot ? std::make_pair(GraphicsPipeline::MeshEmulationBuffer{
                                              *log_slot->buffer, 0, args_bytes},
                                          std::span<u8>{})
                         : request(args_bytes,
                                   task_probe ? MemoryUsage::Download : MemoryUsage::DeviceLocal);
            const u32 capacity_words{static_cast<u32>(
                (std::min)(bytes / 4, u64{(std::numeric_limits<u32>::max)()}))};
            if (task_probe && args_span.size() >= args_bytes) {
                mesh_task_probe = {
                    .args = args_span,
                    .tick = scheduler.CurrentTick(),
                    .tasks = groups,
                    .capacity = capacity_words,
                    .pending = true,
                };
            }
            if (log_slot) {
                log_slot->pending = true;
                log_slot->tick = scheduler.CurrentTick();
                log_slot->frame = render_trace_frame;
                log_slot->draw = render_trace_draws;
                log_slot->tasks = groups;
                log_slot->capacity = capacity_words;
                log_slot->task_hash = pipeline->ShaderHashes()[1];
                log_slot->mesh_hash = pipeline->ShaderHashes()[3];
            }
            const MeshArena& arena{mesh_arenas[mesh_arena_current]};
            pipeline->SetMeshEmulationTaskBuffers(
                payload, offsets, args, {*arena.counter, 0, sizeof(u32)}, capacity_words);
            mesh_emulation_indirect = {args.buffer, args.offset};
        }
        if (probe_draw && output.mapped_span.size() >= output.offset + bytes) {
            const MeshEmulationSetup& setup{*pipeline->MeshEmulation()};
            mesh_probe = {
                .data = output.mapped_span.subspan(output.offset, bytes),
                .tick = scheduler.CurrentTick(),
                .frame = render_trace_frame,
                .groups = groups,
                .block_words = setup.block_words,
                .vertex_lines = setup.vertex_lines,
                .max_vertices = setup.max_vertices,
                .max_primitives = setup.max_primitives,
                .primitive_vertices = setup.primitive_vertices,
                .position_lines = setup.position_lines,
                .pending = true,
            };
        } else if (probe_draw && output.mapped_span.empty()) {
            static bool warned{};
            if (!warned) {
                warned = true;
                LOG_WARNING(Render_Vulkan, "Mesh emulation probe: output buffer is not mapped");
            }
        }
        mesh_emulation_groups = groups;
        mesh_emulation_vertices_per_group = pipeline->MeshEmulationVerticesPerGroup();
    }
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    // What the next draw's TakeReusableDraw compares with, read once the caches are locked: a
    // break from here on, this draw's own resolve included, ends the delta chain.
    const u64 draw_serial = FastDraw::Serial();
    const u64 draw_map_serial = gpu_memory->MapSerial();
    // update engine as channel may be different.
    pipeline->SetEngine(maxwell3d, gpu_memory);
    const u32 repack_dirty = buffer_cache_runtime.SetVertexRepackLayout(
        maxwell3d->regs, pipeline->HasDynamicVertexInput());
    if (repack_dirty) {
        auto& flags = maxwell3d->dirty.flags;
        flags[VideoCommon::Dirty::VertexBuffers] = true;
        flags[Dirty::VertexInput] = true;
        for (u32 binding = 0; binding < 32; ++binding) {
            if (repack_dirty & (1U << binding)) {
                flags[VideoCommon::Dirty::VertexBuffer0 + binding] = true;
            }
        }
    }
    const bool delta_holds =
        delta_pipeline != nullptr && DeltaStillHolds(draw_serial, draw_map_serial);
    const bool verify_views =
        delta_holds && verify_pipeline != nullptr && verify_pipeline == pipeline;
    if (delta_holds && verify_pipeline != nullptr) {
        ++fast_draw_stats.verified;
        fast_draw_stats.pipeline_mismatches += verify_pipeline != pipeline ? 1 : 0;
    }
    const DrawPath path = delta_holds && !verify_pipeline ? DrawPath::Delta : DrawPath::Full;
    // EDEN_NX2_FAST_DRAW_VERIFY compares only draws that would have taken the delta path.
    const bool verify_kept = verify_views && delta_holds;
    verify_vertex_input = verify_kept;
    // Every draw keeps the uniform buffers no write reached since they were streamed.
    buffer_cache.BeginGraphicsUniforms(fast_draw_on && !uniform_writes_lost ? &taken_delta
                                                                            : nullptr);
    uniform_writes_lost = taken_delta_unconsumed = false;
    std::optional<GraphicsPipeline::PreparedBinding> prepared;
    std::optional<GraphicsPipeline::PreparedBinding>* output = nullptr;
    static const bool native_draw_capture = std::getenv("EDEN_NATIVE_DRAW_CAPTURE") != nullptr;
    if (direct_instances && scheduler.RenderRecordersEnabled() && split.CanPostRender() &&
        !mesh_draw && !fast_draw_verify &&
        !frame_dump.Capturing() && !Common::DkbFrameTrace::Enabled() &&
        !VideoCommon::RenderTrace::Enabled() && !GPU::Logging::IsActive() &&
        !native_draw_capture &&
        maxwell3d->regs.transform_feedback_enabled == 0 &&
        maxwell3d->draw_manager.draw_state.inline_index_draw_indexes.empty()) {
        const auto topology = maxwell3d->draw_manager.draw_state.topology;
        if (topology != Maxwell::PrimitiveTopology::Quads &&
            topology != Maxwell::PrimitiveTopology::QuadStrip) {
            output = &prepared;
        }
    }
    if (path == DrawPath::Delta) {
        if (Profile::Scope scope{Profile::Zone::Configure};
            !pipeline->ConfigureDelta(is_indexed, output)) {
            report_failure("configure failed");
            TraceRenderState(is_indexed, "configure-failed", nullptr);
            return;
        }
        ++fast_draw_stats.delta;
    } else {
        // What the delta draw would have kept: the texture buffer bindings.
        const u64 kept_bindings =
            verify_kept ? buffer_cache.KeptGraphicsBindingsHash(false, false, true) : 0;
        if (fast_draw_on) {
            pipeline->CaptureNextDelta(verify_kept);
            FastDraw::NoteHandlesRead();
        }
        if (Profile::Scope scope{Profile::Zone::Configure}; !pipeline->Configure(is_indexed, output)) {
            report_failure("configure failed");
            TraceRenderState(is_indexed, "configure-failed", nullptr);
            return;
        }
        if (verify_kept && pipeline->DeltaMismatch()) {
            ++fast_draw_stats.view_mismatches;
        }
        if (verify_kept &&
            kept_bindings != buffer_cache.KeptGraphicsBindingsHash(false, false, true)) {
            ++fast_draw_stats.binding_mismatches;
        }
    }
    if (reuse_start != 0) {
        const u64 probe_start = VideoCommon::Nx2Profile::Now();
        reuse = NoteDrawReuse(*pipeline, is_indexed);
        reuse_path = path;
        reuse_probe_ns = VideoCommon::Nx2Profile::Now() - probe_start;
    }

    if (OcarinaTemporalEnabled()) {
        TrackOcarinaDraw(pipeline->ShaderHashes()[1], is_indexed, is_indirect);
    }
    TracePendingCpuWrites(false, pipeline->ShaderHashes());
    TraceRenderState(is_indexed, "configured", texture_cache.GetFramebuffer());
    if (frame_dump.Capturing()) {
        FrameDumpPass(texture_cache.GetFramebuffer());
        frame_dump.NoteDraw(pipeline->ShaderHashes());
        if (frame_dump.IncludesGraphicsPass() && frame_dump.WantsShaders(pipeline)) {
            frame_dump.NoteGraphicsShaders(
                *pipeline, pipeline_cache.CaptureGraphicsPrograms(pipeline->ShaderHashes()));
        }
        bool recorded = false;
        if (frame_dump.IncludesGraphicsPass() && frame_dump.CapturingSampled()) {
            boost::container::small_vector<FrameDump::Sampled, 16> sampled;
            for (const auto& image : pipeline->SampledImages()) {
                auto& sample = sampled.emplace_back(FrameDump::Sampled{
                    image.stage, image.tic, &texture_cache.GetImageView(image.id),
                    &texture_cache.GetSampler(image.sampler)});
                if (frame_dump.Focused()) {
                    const auto& header = maxwell3d->regs.tex_header;
                    const u32 index = Tegra::Texture::TextureHeaderIndex(
                        image.tic, header.limit,
                        Tegra::Texture::UsesHeaderByteOffsetHandles(
                            gpu_memory->UsesNativeTiling(), Settings::GetCurrentProgramID()));
                    if (index <= header.limit) {
                        Tegra::Texture::TICEntry tic{};
                        gpu_memory->ReadBlockUnsafe(header.Address() + u64{index} * sizeof(tic),
                                                    &tic, sizeof(tic));
                        sample.tic_raw = tic.raw;
                    }
                }
            }
            recorded = frame_dump.NoteSampled(sampled);
        }
        if (frame_dump.CapturingBuffers()) {
            recorded |= FrameDumpBuffers(false, pipeline->ShaderHashes());
        }
        if (recorded) {
            scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
        }
    }
    if (VideoCommon::RenderTrace::SampleFrame(render_trace_frame) && render_trace_draws <= 2048) {
        LOG_DEBUG(Render_Vulkan,
                  "Render trace pipeline: frame={} draw={} hashes={:016x} dynamic_vertex_input={}",
                  render_trace_frame, render_trace_draws, fmt::join(pipeline->ShaderHashes(), ","),
                  pipeline->HasDynamicVertexInput());
    }

    if (is_indirect && !mesh_draw && Common::DkbFrameTrace::Enabled()) {
        static const bool readback = [] {
            const char* value = std::getenv("EDEN_DKB_INDIRECT_READBACK");
            return value && std::string_view{value} == "1";
        }();
        if (readback) {
            CaptureIndirectTrace();
        }
    }

    if (prepared) {
        query_cache.NotifySegment(true);
        HandleTransformFeedback();
        DynamicCommands dynamic;
        {
            dynamic_commands = &dynamic;
            SCOPE_EXIT { dynamic_commands = nullptr; };
            UpdateOcarinaJitter();
            UpdateDynamicStates(pipeline);
        }
        verify_vertex_input = false;
        const DrawParams params =
            MakeDrawParams(maxwell3d->draw_manager.draw_state, *direct_instances, is_indexed);
        // The query cache stays on the back: its begin lands before the packet's bindings.
        query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64,
                                  maxwell3d->regs.zpass_pixel_count_enable);
        pending_render.emplace(
            RenderDraw{&scheduler, pipeline, std::move(*prepared), std::move(dynamic), params});
    } else {
        {
            Profile::Scope scope{Profile::Zone::DynamicState};
            UpdateOcarinaJitter();
            UpdateDynamicStates(pipeline);
            verify_vertex_input = false;
        }

        query_cache.NotifySegment(true);
        HandleTransformFeedback();
        query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64,
                                  maxwell3d->regs.zpass_pixel_count_enable);
        Profile::Scope scope{Profile::Zone::DrawRecord};
        scheduler.TraceGraphicsPipeline(pipeline->ShaderHashes());
        trace_draw_hashes = {pipeline->ShaderHashes()[1], pipeline->ShaderHashes()[5]};
        draw_func();
    }
    submitted = true;
    buffer_cache.NoteWorkRecorded(false);
    NoteOcarinaDraw(*pipeline);
    if (fast_draw_on) {
        NoteResolvedDraw(pipeline, is_indexed, is_indirect, draw_serial, draw_map_serial);
    }
    if (time_fast_draw) {
        const size_t index = static_cast<size_t>(path);
        fast_draw_stats.prepare_ns[index] += Profile::Now() - fast_draw_start;
        ++fast_draw_stats.prepare_draws[index];
    }
    static u32 native_capture_count{};
    if (maxwell3d->GetNativeMacroEngine().native_turing &&
        native_capture_count < 16 && [] {
            static const bool capture = std::getenv("EDEN_NATIVE_DRAW_CAPTURE") != nullptr;
            return capture;
        }()) {
        const u32 capture = native_capture_count++;
        const auto prefix = fmt::format("native-draw-{}", capture);
        const auto dump = [&](const std::string& name, GPUVAddr address, size_t size) {
            if (!size || !gpu_memory->IsFullyMappedRange(address, size)) return;
            std::vector<u8> bytes(size);
            gpu_memory->ReadBlockUnsafe(address, bytes.data(), bytes.size());
            std::ofstream file(prefix + name + ".bin", std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            LOG_INFO(HW_GPU, "Native draw capture {}{}: address={:#x}, bytes={}, nonzero={}",
                     capture, name, address, size,
                     std::count_if(bytes.begin(), bytes.end(), [](u8 byte) { return byte != 0; }));
        };
        const auto& regs = maxwell3d->regs;
        const auto& rt = regs.rt[0];
        const size_t size = static_cast<size_t>(rt.array_pitch) * 4;
        if (size && size <= 16 * 1024 * 1024) {
            if (const auto address = gpu_memory->GpuToCpuAddress(rt.Address(), size)) {
                LOG_INFO(HW_GPU, "Native draw {} render target physical address={:#x}", capture, *address);
                texture_cache.DownloadMemory(*address, size);
                dump("-rt0", rt.Address(), size);
            }
        }
        std::ofstream registers(prefix + "-registers.bin", std::ios::binary);
        registers.write(reinterpret_cast<const char*>(&regs), sizeof(regs));
        for (size_t stream = 0; stream < 4; ++stream) {
            if (regs.vertex_streams[stream].IsEnabled())
                dump(fmt::format("-vertex-{}", stream), regs.vertex_streams[stream].Address(), 128);
        }
        for (size_t stage : {0U, 4U}) {
            for (size_t bank : {2U, 3U}) {
                const auto& cb = maxwell3d->state.shader_stages[stage].const_buffers[bank];
                if (cb.enabled) dump(fmt::format("-cbuf-{}-{}", stage, bank), cb.address, std::min<size_t>(cb.size, 128));
            }
        }
    }
}

void RasterizerVulkan::NoteOcarinaDraw(const GraphicsPipeline& pipeline) {
    constexpr u64 final_vertex = 0x16aff33d999f993fULL;
    constexpr u64 final_fragment = 0x52402d8e8ab55263ULL;
    const auto& hashes = pipeline.ShaderHashes();
    if (oot_4k_scene_frame == render_trace_frame && hashes[5] == final_fragment &&
        std::ranges::find(hashes, final_vertex) != hashes.end()) {
        const auto target_id = texture_cache.GetRenderTargets().color_buffer_ids[0];
        if (target_id) {
            const auto& target = texture_cache.GetImageView(target_id);
            if (target.size.width == 2560 && target.size.height == 1440 &&
                target.format == VideoCore::Surface::PixelFormat::A8B8G8R8_SRGB) {
                oot_4k_guest_present = target.ImageHandle();
            }
        }
    }
    // The game draws Navi's glow, fireflies and dust right after the fog. They write no depth and
    // move on their own, so FSR must not take them from its history. DLSS doesn't use the mask.
    constexpr u64 fog_fragment = 0x3ddfb1d7a854c849ULL;
    if (hashes[5] == fog_fragment && oot_live_dlss && OcarinaTemporalEnabled() &&
        oot_backend == VideoCommon::OcarinaUpscaleBackend::Fsr &&
        oot_camera_sample_frame == render_trace_frame && !frame_dump.Capturing()) {
        const auto scene_id = texture_cache.GetRenderTargets().color_buffer_ids[0];
        if (scene_id) {
            const auto& scene = texture_cache.GetImageView(scene_id);
            if (scene.format == VideoCore::Surface::PixelFormat::B10G11R11_FLOAT &&
                scene.size.width == oot_motion.extent.width &&
                scene.size.height == oot_motion.extent.height && !scene.IsRescaled() &&
                scene.range == VideoCommon::SubresourceRange{}) {
                oot_effects_scene = scene.Handle(Shader::TextureType::Color2D);
                oot_effects_address = scene.gpu_addr;
                oot_effects_frame = render_trace_frame;
                oot_live_dlss->SaveEffectsBase(oot_effects_scene, oot_motion.extent,
                                               render_trace_frame);
            }
        }
    }
}

namespace {
// EDEN_NX2_WINDOW_TRACE: describe clears and draws into screen-sized colour targets (debug aid).
void TraceWindowTarget(const Tegra::Engines::Maxwell3D& maxwell3d, Tegra::MemoryManager& memory,
                       const char* kind, u32 count) {
    static const bool trace = std::getenv("EDEN_NX2_WINDOW_TRACE") != nullptr;
    static u32 traced{};
    const auto& regs = maxwell3d.regs;
    static const u64 skip = [] {
        const char* value = std::getenv("EDEN_NX2_WINDOW_TRACE_SKIP");
        return value != nullptr ? std::strtoull(value, nullptr, 10) : 0ULL;
    }();
    static u64 seen_events{};
    static const u64 only_ps = [] {
        const char* value = std::getenv("EDEN_NX2_TRACE_PS");
        return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ULL;
    }();
    const u64 pixel_shader =
        (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
    if (!trace || regs.rt[0].width != 3840 || traced >= 400 || seen_events++ < skip ||
        (only_ps != 0 && pixel_shader != only_ps)) {
        return;
    }
    ++traced;
    const auto& vp = regs.viewport_transform[0];
    const auto& sc = regs.scissor_test[0];
    LOG_INFO(HW_GPU,
             "nx2 window {} rt0={:#x} count={} clear={:#x} color=({},{},{},{}) mask_common={} "
             "mask0={:#x} rt_count={} vp scale=({},{}) translate=({},{}) scissor en={} x={}..{} "
             "y={}..{} blend0={} vs={:#x} ps={:#x} rasterize={}",
             kind, regs.rt[0].Address(), count, regs.clear_surface.raw, regs.clear_color[0],
             regs.clear_color[1], regs.clear_color[2], regs.clear_color[3], regs.color_mask_common,
             regs.color_mask[0].raw, regs.rt_control.count.Value(), vp.scale_x, vp.scale_y,
             vp.translate_x, vp.translate_y, sc.enable, sc.min_x.Value(), sc.max_x.Value(),
             sc.min_y.Value(), sc.max_y.Value(), regs.blend.enable[0],
             (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1],
             (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1],
             regs.rasterize_enable);
    if (kind[0] != 'd') {
        return;
    }
    // Pixel-stage constant buffer 3: colour constants of the loading-screen passes.
    if (const auto& cb3 = maxwell3d.state.shader_stages[4].const_buffers[3]; cb3.enabled) {
        std::array<f32, 8> values{};
        memory.ReadBlockUnsafe(cb3.address, values.data(), sizeof(values));
        LOG_INFO(HW_GPU, "nx2 window ps c[3]@{:#x} size={}: {}", cb3.address, cb3.size,
                 fmt::join(values, " "));
    }
    static u32 texture_traces{};
    if (texture_traces++ > 200) {
        return;
    }
    // Texture used by the pixel shader's first bound TEX (handle in c[2][0x20]) and its TIC.
    const auto& cb = maxwell3d.state.shader_stages[4].const_buffers[2];
    const u32 handle = cb.enabled ? memory.Read<u32>(cb.address + 0x20) : 0;
    const u32 tic_index = handle & 0xfffff;
    const GPUVAddr tic_address = regs.tex_header.Address() + u64{tic_index} * 32;
    std::array<u32, 8> tic{};
    memory.ReadBlockUnsafe(tic_address, tic.data(), sizeof(tic));
    LOG_INFO(HW_GPU,
             "nx2 window texture: cb2={:#x} en={} handle={:#x} tic pool={:#x} limit={} "
             "sampler pool={:#x} tic[{}]={:08x}",
             cb.address, cb.enabled, handle, regs.tex_header.Address(), regs.tex_header.limit,
             regs.tex_sampler.Address(), tic_index, fmt::join(tic, " "));
    // First bytes of the texture's memory (block-linear, so just a sample).
    const GPUVAddr texture = (GPUVAddr{tic[2] & 0xffff} << 32) | tic[1];
    std::array<u32, 16> texels{};
    memory.ReadBlockUnsafe(texture, texels.data(), sizeof(texels));
    LOG_INFO(HW_GPU, "nx2 window texture {:#x} {}x{}: {:08x}", texture, (tic[4] & 0xffff) + 1,
             (tic[5] & 0xffff) + 1, fmt::join(texels, " "));
    // EDEN_NX2_TEXTURE_DUMP=<file>: save the first 4 MiB of the last traced texture's memory.
    if (const char* path = std::getenv("EDEN_NX2_TEXTURE_DUMP"); path != nullptr) {
        std::vector<u8> bytes(2048u * 10240u);
        memory.ReadBlockUnsafe(texture, bytes.data(), bytes.size());
        if (std::FILE* file = std::fopen(path, "wb")) {
            std::fwrite(bytes.data(), 1, bytes.size(), file);
            std::fclose(file);
        }
    }
}
} // Anonymous namespace

/// EDEN_NX2_RENDER_VERIFY_EVERY: one in this many unlent draws, and back flush answers, is
/// checked against GPURender's own answer.
static u64 RenderVerifyEvery() {
    static const u64 every = [] {
        const char* const value = std::getenv("EDEN_NX2_RENDER_VERIFY_EVERY");
        return value != nullptr ? std::strtoull(value, nullptr, 10) : 0ULL;
    }();
    return every;
}

template <typename Op>
bool RasterizerVulkan::PostRenderOp(VideoCommon::GpuSplit::PostedOp kind, Op&& op) {
    auto& split = VideoCommon::GpuSplit::Get();
    if (!split.Posts(kind) || maxwell3d == nullptr) {
        return false;
    }
    // GPURender records the op on the view's flags: an engine that has unlent no draw yet gets
    // its view here, seeded by its first unlent draw.
    if (!maxwell3d->render_view_writes) {
        maxwell3d->render_view_writes =
            std::make_unique<VideoCommon::Nx2RenderView::Writes>(*maxwell3d);
    }
    split.PostOp(kind, [this, engine = maxwell3d, op = std::forward<Op>(op)]() mutable {
        auto* const flags =
            state_tracker.SwapFlags(&engine->render_view_writes->View().dirty.flags);
        SCOPE_EXIT {
            state_tracker.SwapFlags(flags);
        };
        op();
    });
    return true;
}

bool RasterizerVulkan::FencePostable(VideoCommon::GpuSplit::PostedOp kind) {
    using VideoCommon::GpuSplit;
    auto& split = GpuSplit::Get();
    if (!split.Posts(kind)) {
        return false;
    }
    // Fritchii 440c984e7: GPURender's fence must not enter the back's buffer cache: nothing to
    // commit there, no query sync owed, and the closing gather runs here.
    const bool delayed =
        Settings::IsGPUFenceBehaviorDefault()
            ? Settings::IsGPULevelHigh()
            : Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
    if (delayed) {
        split.NoteFenceLent(GpuSplit::FenceLent::Delayed);
        return false;
    }
    if (QuerySyncOwed()) {
        split.NoteFenceLent(GpuSplit::FenceLent::QueryOwed);
        return false;
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.HasUncommittedFlushes()) {
            split.NoteFenceLent(GpuSplit::FenceLent::BufferHalf);
            return false;
        }
    }
    if (maxwell3d == nullptr) {
        split.NoteFenceLent(GpuSplit::FenceLent::NoView);
        return false;
    }
    return true;
}

void RasterizerVulkan::CollectQuerySync(QuerySyncPacket& packet) {
    if (back_payload_sync.empty() && samples_slots.empty()) {
        return;
    }
    std::vector<DAddr> pages;
    pages.reserve(samples_slots.size() * 2);
    for (const DAddr slot : samples_slots) {
        pages.push_back(Common::AlignDown(slot, Core::DEVICE_PAGESIZE));
        pages.push_back(Common::AlignDown(slot + 15, Core::DEVICE_PAGESIZE));
    }
    std::ranges::sort(pages);
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    {
        std::scoped_lock lock{buffer_cache.mutex};
        const BufferArena::Collect collect{packet.commands};
        // Maximal runs of contiguous pages over a superset of the reports GPURender syncs: each
        // run SyncValues groups there lies within one of them.
        buffer_cache.BufferOperations([&] {
            packet.targets.clear();
            for (size_t i = 0; i < pages.size();) {
                const DAddr begin = pages[i];
                DAddr end = begin + Core::DEVICE_PAGESIZE;
                for (++i; i < pages.size() && pages[i] == end; ++i) {
                    end += Core::DEVICE_PAGESIZE;
                }
                const auto [buffer, offset] = buffer_cache.ObtainCPUBuffer(
                    begin, static_cast<u32>(end - begin),
                    VideoCommon::ObtainBufferSynchronize::FullSynchronize,
                    VideoCommon::ObtainBufferOperation::DoNothing);
                if (buffer != nullptr) {
                    packet.targets.push_back({begin, end - begin, buffer->Handle(), offset});
                }
            }
        });
        // As SyncValues does before its copies.
        for (const DAddr slot : samples_slots) {
            buffer_cache.OutdateReadDownloads(slot, 16);
        }
        if (!back_payload_sync.empty()) {
            RecordPayloadSyncs(back_payload_sync);
        }
    }
    packet.stamp = packet.commands.Empty() ? 0 : buffer_cache_runtime.Arena().CloseCollected();
    VideoCommon::GpuSplit::Get().NoteQuerySync(packet.targets.size(), back_payload_sync.size());
    back_payload_sync.clear();
    samples_slots.clear();
}

bool RasterizerVulkan::QuerySyncOwed() const {
    using VideoCommon::GpuSplit;
    if (!GpuSplit::QuerySyncOnBack()) {
        return fence_query_owed;
    }
    return query_sync_lent_only || GpuSplit::Get().QuerySyncStopped();
}

void RasterizerVulkan::NoteSyncLent() {
    fence_query_owed = false;
    query_sync_lent_only = false;
    samples_slots.clear();
}

template <typename Op>
void RasterizerVulkan::PostCheckedFence(VideoCommon::GpuSplit::PostedOp kind, Op&& op) {
    using VideoCommon::GpuSplit;
    InvalidateGPUCache();
    // The value syncs its commit runs, after the closing gather as there (collected with
    // EDEN_NX2_RENDER_QUERY_SYNC_E; without it only a fence report's own value is listed).
    QuerySyncPacket sync;
    if (GpuSplit::QuerySyncOnBack() || kind == GpuSplit::PostedOp::FenceReport) {
        CollectQuerySync(sync);
    } else if (!back_payload_sync.empty()) [[unlikely]] {
        GpuSplit::Get().NotePayloadSyncsLeft();
    }
    PostRenderOp(kind, [this, sync = std::move(sync), op = std::forward<Op>(op)]() mutable {
        if (!sync.commands.Empty()) {
            buffer_cache_runtime.Arena().PlayCollected(sync.commands, sync.stamp);
        }
        t_query_sync_targets = &sync.targets;
        posted_fence_playing = true;
        fence_manager.skip_buffer_half = true;
        SCOPE_EXIT {
            t_query_sync_targets = nullptr;
            posted_fence_playing = false;
            fence_manager.skip_buffer_half = false;
        };
        op();
    });
    // Its texture downloads and the guest increment land on GPURender without a lend.
    VideoCommon::GpuSplit::Get().MarkGuestWrite();
}

template <typename Op>
bool RasterizerVulkan::PostFence(Op&& op) {
    if (!FencePostable(VideoCommon::GpuSplit::PostedOp::Fence)) {
        return false;
    }
    PostCheckedFence(VideoCommon::GpuSplit::PostedOp::Fence, std::forward<Op>(op));
    return true;
}

struct RasterizerVulkan::UnlentDraw {
    RasterizerVulkan* rasterizer;
    Tegra::Engines::Maxwell3D* engine;
    GraphicsPipeline* pipeline;
    VideoCommon::Nx2RenderView::Packet view;
    GraphicsPipeline::BackHalf half;
    DrawParams params;
    bool fast_delta{};
    u64 position{};
    std::unique_ptr<OcarinaDrawFacts> ocarina;

    void operator()() {
        rasterizer->PlayUnlentDraw(*this);
    }
};

GuestDescriptorQueue& RasterizerVulkan::UnlendCapture() {
    if (!unlend_capture) {
        unlend_capture.emplace(device, 0x1000, device.IsExtDescriptorBufferSupported());
    }
    return *unlend_capture;
}

bool RasterizerVulkan::TryUnlendDraw(bool is_indexed, u32 instance_count) {
    using Reason = VideoCommon::GpuSplit::UnlendReason;
    namespace Profile = VideoCommon::Nx2Profile;
    namespace View = VideoCommon::Nx2RenderView;
    auto& split = VideoCommon::GpuSplit::Get();
    const auto lend = [&split](Reason reason) {
        split.NoteLent(reason);
        return false;
    };
    const auto& regs = maxwell3d->regs;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    static const bool diagnostics = [] {
        for (const char* name :
             {"EDEN_NX2_FRAME_DUMP", "EDEN_NATIVE_DRAW_CAPTURE", "EDEN_NX2_CB24_TRACE",
              "EDEN_NX2_TRACE_VS", "EDEN_NX2_SKIP_VS", "EDEN_NX2_RT_TRACE", "EDEN_NX2_WINDOW_TRACE",
              "EDEN_NX2_PENDING_WRITE_TRACE", "EDEN_NX2_NO_MESH", "EDEN_NX2_NO_MESH_FILE"}) {
            if (std::getenv(name) != nullptr) {
                return true;
            }
        }
        return false;
    }();
    if (diagnostics || !split.RenderPosting() || frame_dump.Capturing() ||
        Tegra::Nx2ListIndirect::Stats::g_check || Common::DkbFrameTrace::Enabled() ||
        VideoCommon::RenderTrace::Enabled() || GPU::Logging::IsActive() ||
        DrawReuseProbeEnabled() || VideoCommon::HandleTableNotes::Wanted() ||
        Tegra::Engines::g_nx2_trace_draws_after_upscale > 0) {
        return lend(Reason::Diagnostics);
    }
    const u64 first = is_indexed ? draw_state.index_buffer.first : draw_state.vertex_buffer.first;
    const u64 count = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count;
    if (first + count > (u64{1} << 24)) {
        return lend(Reason::Diagnostics);
    }
    if (VideoCommon::Nx2FastDraw::Enabled() || fast_draw_on) {
        return lend(Reason::FastDraw);
    }
    if (regs.transform_feedback_enabled != 0) {
        return lend(Reason::Feedback);
    }
    if (!draw_state.inline_index_draw_indexes.empty()) {
        return lend(Reason::InlineIndex);
    }
    if (draw_state.topology == Maxwell::PrimitiveTopology::Quads ||
        draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
        return lend(Reason::Quads);
    }
    if (is_indexed && draw_state.index_buffer.format == Maxwell::IndexFormat::UnsignedByte &&
        !device.IsExtIndexTypeUint8Supported()) {
        return lend(Reason::Uint8Index);
    }
    // EDEN_NX2_RENDER_UNLEND_MESH: native task/mesh pipelines take the unlent draw (Fritchii
    // c1bf9ddf94); the compute emulation stays lent.
    const bool unlend_mesh =
        VideoCommon::GpuSplit::UnlendsMesh() && device.IsExtMeshShaderSupported();
    if (regs.IsMeshEnabled() && !unlend_mesh) {
        return lend(Reason::Mesh);
    }
    if (!is_indexed && instance_count == 1 &&
        draw_state.topology == Maxwell::PrimitiveTopology::Points &&
        maxwell3d->GetNativeMacroEngine().native_turing) {
        return lend(Reason::IndirectReset);
    }
    // A direct draw with no vertices or no instances records nothing (as Draw skips it).
    const DrawParams params = MakeDrawParams(draw_state, instance_count, is_indexed);
    if (params.num_vertices == 0 || instance_count == 0) {
        const GraphicsPipeline* const pipeline =
            params.num_vertices == 0 ? nullptr : pipeline_cache.CurrentGraphicsPipeline();
        if (pipeline == nullptr || !pipeline->IsMeshPipeline()) {
            if (gpu.HasSyncRequests()) {
                split.LendRender([this] { gpu.TickWork(); });
            }
            return true;
        }
        if (params.num_vertices == 0 || !unlend_mesh) {
            return lend(Reason::Mesh);
        }
    }
    if (const u64 verify_every = RenderVerifyEvery();
        verify_every != 0 && ++unlend_candidates % verify_every == 0) {
        split.NoteUnlendVerify();
        unlend_verify_draw = true;
        return lend(Reason::Verify);
    }
    // The draw's start: cached writes reach the caches, lent where they reach GPURender's.
    unlend_gather_hit = false;
    gpu_memory->FlushCaching();
    if (unlend_gather_hit) {
        return lend(Reason::Gather);
    }
    const View::Flags& forwarded = View::Forwarded();
    const View::Flags start_flags = maxwell3d->dirty.flags & forwarded;
    // EDEN_NX2_RENDER_UNLEND_FAST_DELTA: only draw parameters written since the last unlent draw,
    // whose lookup was the cache's last, keep its pipeline.
    const bool fast_delta = VideoCommon::GpuSplit::UnlendsFastDelta();
    const bool key_unchanged = fast_delta && maxwell3d->render_view_writes &&
                               unlent_lookup.engine == maxwell3d &&
                               unlent_lookup.lookups == pipeline_cache.GraphicsLookups() &&
                               unlent_lookup.topology == draw_state.topology &&
                               !maxwell3d->render_view_writes->KeyWordsWritten(*maxwell3d);
    GraphicsPipeline* pipeline = [&] {
        Profile::Scope scope{Profile::Zone::PipelineLookup};
        return pipeline_cache.CurrentGraphicsPipeline(key_unchanged);
    }();
    if (fast_delta) {
        const bool kept = pipeline_cache.LastLookupKept();
        split.NoteFastDelta(kept);
        if (kept && RenderVerifyEvery() != 0) {
            GraphicsPipeline* const looked_up = pipeline_cache.CurrentGraphicsPipeline();
            if (looked_up != pipeline) {
                split.NoteFastDeltaDiffers();
                pipeline = looked_up;
            }
        }
    }
    if (pipeline == nullptr || !pipeline->CanUnlend() ||
        (pipeline->IsMeshPipeline() && (!unlend_mesh || pipeline->IsEmulatedMesh()))) {
        return lend(pipeline != nullptr && pipeline->IsMeshPipeline() ? Reason::Mesh
                                                                      : Reason::NotBuilt);
    }
    // Owed list words (Nx2ListIndirect) in the draw's buffers are patched in its packet, but
    // the back reads texture handles and storage buffer descriptors from the constant buffers
    // itself: one over an owed word would read its placeholder. Only the buffers it reads them
    // from are looked at.
    const auto constant_buffers_owe = [&] {
        const auto& slots = buffer_cache_runtime.Nx2Slots();
        const auto& handle_cbufs = pipeline->HandleConstBuffers();
        for (size_t stage = 0; stage < handle_cbufs.size(); ++stage) {
            const auto& cbufs = maxwell3d->state.shader_stages[stage].const_buffers;
            for (u32 bits = handle_cbufs[stage]; bits != 0; bits &= bits - 1) {
                const u32 index = static_cast<u32>(std::countr_zero(bits));
                if (index >= cbufs.size()) {
                    break;
                }
                const auto& cbuf = cbufs[index];
                if (!cbuf.enabled || cbuf.address == 0 || cbuf.size == 0) {
                    continue;
                }
                for (const auto& [device_addr, size] :
                     gpu_memory->GetSubmappedDeviceRange(cbuf.address, cbuf.size)) {
                    if (slots.Owes(device_addr, size)) {
                        return true;
                    }
                }
            }
        }
        return false;
    };
    if (buffer_cache_runtime.Nx2Slots().HasOwed() && constant_buffers_owe()) {
        return lend(Reason::ListOwed);
    }
    // Ocarina of Time's No Sharpening bypass: the back's next dispatches read the tracked target.
    if (VideoCommon::OotNetwork::BypassEnabled(Settings::GetCurrentProgramID())) {
        TrackOotTonemappedImage();
    }
    std::unique_lock lock{buffer_cache.mutex};
    pipeline->SetEngine(maxwell3d, gpu_memory);
    const u32 repack_dirty = buffer_cache_runtime.SetVertexRepackLayout(
        maxwell3d->regs, pipeline->HasDynamicVertexInput());
    if (repack_dirty) {
        auto& flags = maxwell3d->dirty.flags;
        flags[VideoCommon::Dirty::VertexBuffers] = true;
        flags[Dirty::VertexInput] = true;
        for (u32 binding = 0; binding < 32; ++binding) {
            if (repack_dirty & (1U << binding)) {
                flags[VideoCommon::Dirty::VertexBuffer0 + binding] = true;
            }
        }
    }
    if (buffer_cache_runtime.AnyVertexRepacked()) {
        lock.unlock();
        return lend(Reason::Repack);
    }
    // GPURender can submit before it plays the draw, and a new command buffer has no vertex
    // buffers: bind every enabled one (the bindings stay resolved while VertexBuffers is clear).
    for (u32 binding = 0; binding < Maxwell::NumVertexArrays; ++binding) {
        maxwell3d->dirty.flags[VideoCommon::Dirty::VertexBuffer0 + binding] = true;
    }
    if (!maxwell3d->render_view_writes) {
        maxwell3d->render_view_writes = std::make_unique<View::Writes>(*maxwell3d);
    }
    UnlentDraw draw{
        .rasterizer = this,
        .engine = maxwell3d,
        .pipeline = pipeline,
        .view = maxwell3d->render_view_writes->Pack(*maxwell3d, true),
        .half = {},
        .params = params,
        .fast_delta = fast_delta,
    };
    if (fast_delta) {
        maxwell3d->render_view_writes->KeyWordsSeen();
        unlent_lookup = {maxwell3d, pipeline_cache.GraphicsLookups(), draw_state.topology};
    }
    buffer_cache.BeginGraphicsUniforms(nullptr);
    uniform_writes_lost = taken_delta_unconsumed = false;
    {
        Profile::Scope scope{Profile::Zone::Configure};
        const BufferArena::Collect collect{draw.half.commands};
        pipeline->ConfigureBack(is_indexed, draw.half, UnlendCapture());
    }
    draw.ocarina = CaptureOcarinaFacts(pipeline->ShaderHashes()[1], is_indexed, false);
    buffer_cache.NoteWorkRecorded(false);
    lock.unlock();
    draw.half.stamp = buffer_cache_runtime.Arena().CloseCollected();
    // What GPURender consumes goes in the packet, raised by events: the back's copy starts over.
    draw.view.raised |= start_flags | (maxwell3d->dirty.flags & forwarded);
    maxwell3d->dirty.flags &= ~forwarded;
    split.NotePools(regs.tex_header.Address(), (u64{regs.tex_header.limit} + 1) * 32,
                    regs.tex_sampler.Address(), (u64{regs.tex_sampler.limit} + 1) * 32);
    split.PostUnlent(std::move(draw));
    if (gpu.HasSyncRequests()) {
        split.LendRender([this] { gpu.TickWork(); });
    }
    return true;
}

void RasterizerVulkan::PlayUnlentDraw(UnlentDraw& draw) {
    namespace Profile = VideoCommon::Nx2Profile;
    Tegra::Engines::Maxwell3D& view = draw.engine->render_view_writes->View();
    VideoCommon::Nx2RenderView::Apply(view, draw.view);
    VideoCommon::GpuSplit::SetPlayingPosition(draw.position);
    render_engine = &view;
    Tegra::Engines::Maxwell3D* const texture_engine = texture_cache.SwapMaxwell3D(&view);
    Tegra::Engines::Maxwell3D* const query_engine = query_cache.SwapMaxwell3D(&view);
    auto* const flags = state_tracker.SwapFlags(&view.dirty.flags);
    SCOPE_EXIT {
        state_tracker.SwapFlags(flags);
        query_cache.SwapMaxwell3D(query_engine);
        texture_cache.SwapMaxwell3D(texture_engine);
        render_engine = nullptr;
        VideoCommon::GpuSplit::SetPlayingPosition(~u64{0});
    };
    {
        // A submit here invalidates the view's flags: the dynamic state is GPURender's.
        Profile::Scope scope{Profile::Zone::FlushWork};
        FlushWork();
    }
    std::scoped_lock lock{texture_cache.mutex};
    GraphicsPipeline& pipeline = *draw.pipeline;
    bool configured;
    {
        Profile::Scope scope{Profile::Zone::Configure};
        configured = pipeline.ConfigureRender(draw.half, view);
    }
    if (!configured) {
        VideoCommon::GpuSplit::Get().NoteUnlentFailed();
        TraceDrawFailure("configure failed (unlent)");
        return;
    }
    {
        Profile::Scope scope{Profile::Zone::DynamicState};
        if (draw.ocarina) {
            ApplyOcarinaFacts(*draw.ocarina);
        }
        UpdateOcarinaJitter();
        auto& view_flags = view.dirty.flags;
        const bool input_kept = draw.fast_delta && pipeline.HasDynamicVertexInput() &&
                                !view_flags[Dirty::VertexInput] &&
                                !view_flags[VideoCommon::Dirty::VertexBuffers];
        UpdateDynamicStates(&pipeline);
        if (draw.fast_delta) {
            // On the view only the vertex input reads VertexBuffers (the back's buffer cache binds
            // the vertex buffers): recorded, it is consumed.
            view_flags[VideoCommon::Dirty::VertexBuffers] = false;
        }
        if (input_kept) {
            auto& split = VideoCommon::GpuSplit::Get();
            split.NoteVertexInputKept();
            if (RenderVerifyEvery() != 0 &&
                (recorded_input_tick != scheduler.CurrentTick() ||
                 recorded_input_key != VertexInputKeyOf(view.regs,
                                                        recorded_input_key.num_attributes,
                                                        recorded_input_key.num_bindings))) {
                split.NoteVertexInputDiffers();
            }
        }
    }
    query_cache.NotifySegment(true);
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64,
                              view.regs.zpass_pixel_count_enable);
    {
        Profile::Scope scope{Profile::Zone::DrawRecord};
        const auto& hashes = pipeline.ShaderHashes();
        scheduler.TraceGraphicsPipeline(hashes);
        trace_draw_hashes = {hashes[1], hashes[5]};
        if (pipeline.IsMeshPipeline()) {
            if (pipeline.IsEmulatedMesh()) [[unlikely]] {
                VideoCommon::GpuSplit::Get().NoteUnlentEmulatedMesh();
            } else {
                scheduler.Record(
                    [this, groups = draw.params.num_vertices](vk::CommandBuffer cmdbuf) {
                        if (scheduler.GraphicsPipelineMissing()) {
                            return;
                        }
                        cmdbuf.DrawMeshTasksEXT(groups, 1, 1);
                    });
            }
        } else {
            RecordDirectDraw(scheduler, draw.params, trace_draw_hashes);
        }
    }
    NoteOcarinaDraw(pipeline);
}

void* RasterizerVulkan::LendEnter(void* argument) {
    auto& self = *static_cast<RasterizerVulkan*>(argument);
    // EDEN_NX2_RENDER_PAYLOAD_E: before anything the call syncs. A report sets fence_query_owed,
    // so no sync is posted until a lent fence clears it after this. With
    // EDEN_NX2_RENDER_QUERY_SYNC_E the uploads are recorded here instead, so nothing is owed.
    if (!self.back_payload_sync.empty()) {
        if (VideoCommon::GpuSplit::QuerySyncEnabled()) {
            self.RecordPayloadSyncs(self.back_payload_sync);
        } else {
            for (const PayloadSync& sync : self.back_payload_sync) {
                self.query_cache.AddPayloadSync(sync.address, sync.value, sync.size);
            }
        }
        self.back_payload_sync.clear();
    }
    Tegra::Engines::Maxwell3D* const engine = self.maxwell3d;
    if (engine == nullptr || !engine->render_view_writes) {
        return nullptr;
    }
    // GPURender's submits outside a lent call invalidate the view's flags, the per-binding vertex
    // buffer flags among them: the back's own, which the lent call consumes.
    auto& view_flags = engine->render_view_writes->View().dirty.flags;
    engine->dirty.flags |= view_flags;
    view_flags &= VideoCommon::Nx2RenderView::Forwarded();
    return engine;
}

void RasterizerVulkan::LendLeave(void*, void* token) {
    // Buffer commands the call queued with no scheduler use after them (an inline write into
    // GPU-written buffer memory, InlineMemoryImplementation) take their arena stamp here, the
    // back parked: played at GPURender's next scheduler use, they took it while the back stamps.
    BufferArena::PlayPending();
    if (token == nullptr) {
        return;
    }
    auto* const engine = static_cast<Tegra::Engines::Maxwell3D*>(token);
    auto& view_flags = engine->render_view_writes->View().dirty.flags;
    const auto& forwarded = VideoCommon::Nx2RenderView::Forwarded();
    view_flags = (view_flags & ~forwarded) | (engine->dirty.flags & forwarded);
}

void RasterizerVulkan::Draw(bool is_indexed, u32 instance_count) {
    if (VideoCommon::GpuSplit::LendsToRender()) [[unlikely]] {
        if (VideoCommon::GpuSplit::Unlends() && TryUnlendDraw(is_indexed, instance_count)) {
            return;
        }
        VideoCommon::GpuSplit::Get().LendRender([&] { Draw(is_indexed, instance_count); });
        return;
    }
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Draw};
    {
        // A draw's first and count can come from memory a skipped (rejected) compute program
        // should have written. Past 16M vertices they are garbage: building the quad index
        // buffer for one threw out of the GPU thread and ended the process (Hyrule Warriors:
        // Age of Imprisonment, entering a new area), and indexed ones read far out of bounds.
        const auto& state = maxwell3d->draw_manager.draw_state;
        const u64 first = is_indexed ? state.index_buffer.first : state.vertex_buffer.first;
        const u64 count = is_indexed ? state.index_buffer.count : state.vertex_buffer.count;
        constexpr u64 MaxVertices = u64{1} << 24;
        if (first + count > MaxVertices) {
            static std::atomic<u32> skipped{};
            if (skipped.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_WARNING(HW_GPU, "Draw skipped: {} {} from {} (vertex/index count past {})",
                            count, is_indexed ? "indices" : "vertices", first, MaxVertices);
            }
            return;
        }
    }
    if (TryNx2IndirectReset(is_indexed, instance_count)) {
        return;
    }
    TraceWindowTarget(*maxwell3d, *gpu_memory, is_indexed ? "draw-indexed" : "draw",
                      is_indexed ? maxwell3d->draw_manager.draw_state.index_buffer.count
                                 : maxwell3d->draw_manager.draw_state.vertex_buffer.count);
    // EDEN_NX2_CB24_TRACE: for draws whose vertex shader reads bank 24, show the inline
    // constant buffer upload target (CB_ADDRESS) and its first words.
    if (static const bool trace = std::getenv("EDEN_NX2_CB24_TRACE") != nullptr; trace) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        static u32 logged{};
        if ((vs == 0x11471c9a00 || vs == 0x11471ca600 || vs == 0x11471cd300) && logged++ < 3) {
            std::array<u32, 24> words{};
            gpu_memory->ReadBlockUnsafe(regs.const_buffer.Address(), words.data(), sizeof(words));
            LOG_INFO(HW_GPU, "bank24 draw vs={:#x}: CB_ADDRESS {:#x} size {:#x} offset {:#x}: {:08x}", vs,
                     regs.const_buffer.Address(), regs.const_buffer.size, regs.const_buffer.offset,
                     fmt::join(words, " "));
            // Search every bound vertex-stage buffer for render-target-size values.
            const auto& stage = maxwell3d->state.shader_stages[0];
            for (u32 slot = 0; slot < Tegra::Engines::Maxwell3D::Regs::MaxConstBuffers; ++slot) {
                const auto& buffer = stage.const_buffers[slot];
                if (!buffer.enabled || buffer.size == 0) continue;
                std::vector<u32> data(std::min<u32>(buffer.size, 0x10000) / 4);
                gpu_memory->ReadBlockUnsafe(buffer.address, data.data(), data.size() * 4);
                for (size_t i = 0; i < data.size(); ++i) {
                    const u32 w = data[i];
                    if (w == 0x45200000 || w == 0x44b40000 || w == 0x39cccccd || w == 0x3a360b61 ||
                        w == 0x44a00000 || w == 0x44340000 || w == 2559 || w == 1439) {
                        LOG_INFO(HW_GPU, "  slot {} +{:#x} = {:#x}", slot, i * 4, w);
                    }
                }
            }
        }
    }
    // EDEN_NX2_TRACE_VS=<address>: describe the next few draws using that vertex shader like the
    // after-upscale trace does (render target, blend, fragment textures).
    static const u64 trace_vs = [] {
        const char* value = std::getenv("EDEN_NX2_TRACE_VS");
        return value ? std::strtoull(value, nullptr, 16) : 0ULL;
    }();
    if (trace_vs != 0) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        static u32 traced{};
        if (vs == trace_vs && traced++ % 200 == 0 && traced < 1000) {
            Tegra::Engines::g_nx2_trace_draws_after_upscale = 1;
        }
    }
    if (Tegra::Engines::g_nx2_trace_draws_after_upscale > 0) {
        --Tegra::Engines::g_nx2_trace_draws_after_upscale;
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        LOG_INFO(HW_GPU, "after upscale: draw rt0 {:#x} {}x{} fmt {:#x} block {}/{}/{} pitch_linear {} depth {} array_pitch {:#x} ctrl {:#x} vs {:#x} ps {:#x}",
                 regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height,
                 static_cast<u32>(regs.rt[0].format), regs.rt[0].tile_mode.block_width.Value(),
                 regs.rt[0].tile_mode.block_height.Value(), regs.rt[0].tile_mode.block_depth.Value(),
                 regs.rt[0].tile_mode.is_pitch_linear.Value(), regs.rt[0].depth.Value(),
                 regs.rt[0].array_pitch, regs.rt_control.count.Value(), vs, ps);
        // Fragment-stage texture handles (bound buffer 2) and the images they name.
        const auto& handles = maxwell3d->state.shader_stages[4].const_buffers[2];
        if (handles.enabled) {
            for (u32 i = 0; i < 24; ++i) {
                const u32 handle = gpu_memory->Read<u32>(handles.address + i * 4);
                if (handle == 0 || handle == 0xffffffff) continue;
                Tegra::Texture::TICEntry tic{};
                gpu_memory->ReadBlockUnsafe(regs.tex_header.Address() + u64{handle & 0xfffff} * sizeof(tic),
                                            &tic, sizeof(tic));
                LOG_INFO(HW_GPU, "  fs handle[{}] = {:#x}: {:#x} {}x{} format {:#x}", i, handle,
                         tic.Address(), tic.Width(), tic.Height(), static_cast<u32>(tic.format.Value()));
            }
        }
        const auto& bt = regs.blend_per_target[0];
        LOG_INFO(HW_GPU, "  blend enable {} per-target {} global op {} src {:#x} dst {:#x} | rt0 op {} src {:#x} dst {:#x} alpha {:#x}/{:#x} mask {:#x}",
                 regs.blend.enable[0], regs.blend_per_target_enabled,
                 static_cast<u32>(regs.blend.color_op), static_cast<u32>(regs.blend.color_source),
                 static_cast<u32>(regs.blend.color_dest), static_cast<u32>(bt.color_op),
                 static_cast<u32>(bt.color_source), static_cast<u32>(bt.color_dest),
                 static_cast<u32>(bt.alpha_source), static_cast<u32>(bt.alpha_dest),
                 regs.color_mask[0].raw);
    }
    // EDEN_NX2_SKIP_VS=<address>[,<address>...]: drop draws using those vertex shaders (debug aid).
    static const std::vector<u64> skip_list = [] {
        std::vector<u64> list;
        if (const char* value = std::getenv("EDEN_NX2_SKIP_VS")) {
            for (char* end = nullptr; *value; value = *end ? end + 1 : end) {
                list.push_back(std::strtoull(value, &end, 16));
                if (end == value) break;
            }
        }
        return list;
    }();
    if (!skip_list.empty()) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        if (std::find(skip_list.begin(), skip_list.end(), vs) != skip_list.end()) return;
    }
    // EDEN_NX2_RT_TRACE: log the first draw to every distinct colour target (debug aid).
    if (static const bool trace = std::getenv("EDEN_NX2_RT_TRACE") != nullptr; trace) {
        static std::set<std::tuple<GPUVAddr, u32, u32, u64, u64>> seen;
        const auto& regs = maxwell3d->regs;
        const auto& rt = regs.rt[0];
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        const auto key = std::make_tuple(rt.Address(), rt.width, static_cast<u32>(rt.format), vs, ps);
        if (rt.width >= 640 && seen.size() < 600 && seen.insert(key).second) {
            LOG_INFO(HW_GPU, "nx2 draw target {:#x} {}x{} format={:#x} vs={:#x} ps={:#x} blend={} (pairs so far {})",
                     rt.Address(), rt.width, rt.height, static_cast<u32>(rt.format), vs, ps,
                     regs.blend.enable[0], seen.size());
        }
    }
    // A direct draw with no vertices, or no instances outside task/mesh pipelines, records
    // nothing. Skip it before the pipeline, buffer and descriptor work: dirty state, deferred
    // clears and query counter toggles stay pending for the next recorded draw, clear or render
    // pass end, which apply them before anything can observe them. With transform feedback on
    // the draw goes through, since its counters are begun and closed around draws.
    if (maxwell3d->regs.transform_feedback_enabled == 0 &&
        (MakeDrawParams(maxwell3d->draw_manager.draw_state, instance_count, is_indexed)
                 .num_vertices == 0 ||
         (instance_count == 0 && [this] {
             const GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
             return pipeline != nullptr && !pipeline->IsMeshPipeline();
         }()))) {
        if (Common::DkbFrameTrace::Enabled()) {
            ++trace_draws;
            ++trace_zero_direct;
        }
        gpu.TickWork();
        return;
    }
    PrepareDraw(is_indexed, [this, is_indexed, instance_count] {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        const u32 num_instances{instance_count};
        const DrawParams draw_params{MakeDrawParams(draw_state, num_instances, is_indexed)};
        if (VideoCommon::RenderTrace::SampleFrame(render_trace_frame) && render_trace_draws <= 2048) {
            LOG_DEBUG(Render_Vulkan,
                      "Render trace direct: frame={} draw={} count={} instances={} first_index={} "
                      "base_vertex={} base_instance={} mesh={}",
                      render_trace_frame, render_trace_draws, draw_params.num_vertices,
                      draw_params.num_instances, draw_params.first_index, draw_params.base_vertex,
                      draw_params.base_instance, mesh_draw);
        }

        if (Common::DkbFrameTrace::Enabled() &&
            (draw_params.num_vertices == 0 || (!mesh_draw && draw_params.num_instances == 0))) {
            ++trace_zero_direct;
        }

        if (mesh_draw && mesh_emulation_vertices_per_group != 0) {
            const u32 vertices{mesh_emulation_groups * mesh_emulation_vertices_per_group};
            if (vertices == 0) {
                return;
            }
            if (mesh_emulation_indirect.buffer) {
                // With a task program the distribution pass wrote the vertex count.
                scheduler.Record([this, indirect = mesh_emulation_indirect](vk::CommandBuffer cmdbuf) {
                    if (scheduler.GraphicsPipelineMissing()) {
                        return;
                    }
                    cmdbuf.DrawIndirect(indirect.buffer,
                                        indirect.offset +
                                            Shader::Backend::SPIRV::MESH_EMULATION_DRAW_ARGS_OFFSET,
                                        1, 0);
                });
                return;
            }
            scheduler.Record([this, vertices](vk::CommandBuffer cmdbuf) {
                if (scheduler.GraphicsPipelineMissing()) {
                    return;
                }
                cmdbuf.Draw(vertices, 1, 0, 0);
            });
            return;
        }
        if (mesh_draw) {
            scheduler.Record([this, groups = draw_params.num_vertices](vk::CommandBuffer cmdbuf) {
                if (scheduler.GraphicsPipelineMissing()) {
                    return;
                }
                cmdbuf.DrawMeshTasksEXT(groups, 1, 1);
            });
            return;
        }
        RecordDirectDraw(scheduler, draw_params, trace_draw_hashes);

        // Log draw call
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            const std::string params = is_indexed ?
                fmt::format("vertices={}, instances={}, firstIndex={}, baseVertex={}, baseInstance={}",
                    draw_params.num_vertices, draw_params.num_instances,
                    draw_params.first_index, draw_params.base_vertex, draw_params.base_instance) :
                fmt::format("vertices={}, instances={}, firstVertex={}, firstInstance={}",
                    draw_params.num_vertices, draw_params.num_instances,
                    draw_params.base_vertex, draw_params.base_instance);
            GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                is_indexed ? "vkCmdDrawIndexed" : "vkCmdDraw", params, VK_SUCCESS);
        }
    }, false, instance_count);
}

void RasterizerVulkan::DrawIndirect() {
    if (VideoCommon::GpuSplit::Unlends()) {
        VideoCommon::GpuSplit::Get().NoteLent(VideoCommon::GpuSplit::UnlendReason::Indirect);
    }
    RENDER_LEND(DrawIndirect())
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Draw};
    const auto& params = maxwell3d->draw_manager.indirect_state;
    if (Common::DkbFrameTrace::Enabled()) {
        ++trace_indirect;
    }
    buffer_cache.SetDrawIndirect(&params);
    PrepareDraw(params.is_indexed, [this, &params] {
        if (VideoCommon::RenderTrace::SampleFrame(render_trace_frame) && render_trace_draws <= 2048) {
            LOG_DEBUG(Render_Vulkan,
                      "Render trace indirect: frame={} draw={} address={:#x} count_address={:#x} "
                      "bytes={} max_draws={} stride={} include_count={} byte_count={} mesh={}",
                      render_trace_frame, render_trace_draws, params.indirect_start_address,
                      params.count_start_address, params.buffer_size, params.max_draw_counts,
                      params.stride, params.include_count, params.is_byte_count, mesh_draw);
        }
        if (mesh_draw) {
            TraceDrawFailure("indirect task/mesh unsupported");
            // Indirect task/mesh draws would need vkCmdDrawMeshTasksIndirectEXT's layout.
            static bool logged{};
            if (!std::exchange(logged, true)) {
                LOG_WARNING(Render_Vulkan, "Indirect task/mesh draw skipped");
            }
            return;
        }
        const auto indirect_buffer = buffer_cache.GetDrawIndirectBuffer();
        const auto& buffer = indirect_buffer.first;
        const auto& offset = indirect_buffer.second;
        if (params.is_byte_count) {
            scheduler.Record([this, buffer_obj = buffer->Handle(), offset,
                              stride = params.stride](vk::CommandBuffer cmdbuf) {
                if (scheduler.GraphicsPipelineMissing()) {
                    return;
                }
                cmdbuf.DrawIndirectByteCountEXT(1, 0, buffer_obj, offset, 0,
                                                static_cast<u32>(stride));
            });
            return;
        }
        if (params.include_count) {
            const auto count = buffer_cache.GetDrawIndirectCount();
            const auto& draw_buffer = count.first;
            const auto& offset_base = count.second;
            scheduler.Record([this, draw_buffer_obj = draw_buffer->Handle(),
                              buffer_obj = buffer->Handle(), offset_base, offset,
                              params](vk::CommandBuffer cmdbuf) {
                if (scheduler.GraphicsPipelineMissing()) {
                    return;
                }
                if (params.is_indexed) {
                    cmdbuf.DrawIndexedIndirectCount(
                        buffer_obj, offset, draw_buffer_obj, offset_base,
                        static_cast<u32>(params.max_draw_counts), static_cast<u32>(params.stride));
                } else {
                    cmdbuf.DrawIndirectCount(buffer_obj, offset, draw_buffer_obj, offset_base,
                                             static_cast<u32>(params.max_draw_counts),
                                             static_cast<u32>(params.stride));
                }
            });
            return;
        }
        scheduler.Record([this, buffer_obj = buffer->Handle(), offset, params,
                          hashes = trace_draw_hashes](vk::CommandBuffer cmdbuf) {
            if (scheduler.GraphicsPipelineMissing()) {
                return;
            }
            scheduler.TraceGpuWork(cmdbuf, hashes.first, hashes.second, {}, [&] {
                if (params.is_indexed) {
                    cmdbuf.DrawIndexedIndirect(buffer_obj, offset,
                                               static_cast<u32>(params.max_draw_counts),
                                               static_cast<u32>(params.stride));
                } else {
                    cmdbuf.DrawIndirect(buffer_obj, offset,
                                        static_cast<u32>(params.max_draw_counts),
                                        static_cast<u32>(params.stride));
                }
            });
        });

        // Log indirect draw call
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            const std::string log_params = fmt::format("drawCount={}, stride={}",
                params.max_draw_counts, params.stride);
            GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                params.is_indexed ? "vkCmdDrawIndexedIndirect" : "vkCmdDrawIndirect",
                log_params, VK_SUCCESS);
        }
    }, true);
    buffer_cache.SetDrawIndirect(nullptr);
}

VideoCommon::OotNetwork::BypassState* RasterizerVulkan::OotNetworkBypass() {
    // Temporal upscaling replaces the same output pass and takes precedence until it fails.
    if (OcarinaTemporalEnabled() && !(oot_live_dlss && oot_live_dlss->EvaluationFailed())) {
        static bool logged = false;
        if (!std::exchange(logged, true)) {
            LOG_INFO(Render_Vulkan, "Ocarina of Time: No sharpening is off while temporal upscaling is on");
        }
        return nullptr;
    }
    return &oot_network_bypass;
}

void RasterizerVulkan::TrackOotTonemappedImage() {
    const auto& target = maxwell3d->regs.rt[0];
    auto& bypass = oot_network_bypass;
    constexpr auto relaxed = std::memory_order_relaxed;
    // The scene draws repeat one target; only a change of target takes the lock.
    if (target.format != Tegra::RenderTargetFormat::B10G11R11_FLOAT ||
        target.tile_mode.is_pitch_linear || bypass.latest_address.load(relaxed) == target.Address() ||
        !VideoCommon::OotNetwork::IsTonemapCandidate(
            target.Address(), target.width, target.height, bypass.input_width.load(relaxed),
            bypass.input_height.load(relaxed), bypass.output_address.load(relaxed))) {
        return;
    }
    std::scoped_lock lock{bypass.mutex};
    bypass.latest_draw = target;
    bypass.latest_address.store(target.Address(), relaxed);
}

void RasterizerVulkan::TraceDrawFailure(std::string_view reason) {
    if (!Common::DkbFrameTrace::Enabled()) {
        return;
    }
    ++trace_skipped;
    if (trace_failure_details++ >= 4) {
        return;
    }
    const auto& regs = maxwell3d->regs;
    std::array<GPUVAddr, 6> stages{};
    for (size_t i = 0; i < stages.size(); ++i) {
        if (regs.IsShaderConfigEnabled(i)) {
            stages[i] = maxwell3d->GetNativeMacroEngine().native_turing
                            ? (u64{regs.pipelines[i].reserved[0] & 0xff} << 32) |
                                  regs.pipelines[i].reserved[1]
                            : regs.program_region.Address() + regs.pipelines[i].offset;
        }
    }
    LOG_INFO(Render_Vulkan,
             "DKB draw skipped frame={} reason={} stages[0..5]={:#x} rt0={:#x} {}x{}",
             trace_frame, reason, fmt::join(stages, ","), regs.rt[0].Address(),
             regs.rt[0].width, regs.rt[0].height);
}

void RasterizerVulkan::CaptureIndirectTrace() {
    if (!trace_sample_frame || trace_captures >= 4) {
        return;
    }
    ++trace_captures;
    const auto slot = std::find_if(indirect_traces.begin(), indirect_traces.end(),
                                   [](const auto& trace) { return !trace.pending; });
    if (slot == indirect_traces.end()) {
        ++trace_capture_dropped;
        return;
    }
    const auto& params = maxwell3d->draw_manager.indirect_state;
    const auto [commands, command_offset] = buffer_cache.GetDrawIndirectBuffer();
    const u32 max_draws = static_cast<u32>(params.max_draw_counts);
    std::array<VkBufferCopy, 8> copies{};
    u32 sampled{};
    const u64 command_bytes = params.is_byte_count ? 4 : 8;
    const u64 available = command_offset <= commands->SizeBytes()
                              ? std::min<u64>(params.buffer_size,
                                              commands->SizeBytes() - command_offset)
                              : 0;
    const u32 limit = params.is_byte_count ? 1 : std::min(max_draws, 8U);
    for (u32 i = 0; i < limit; ++i) {
        if (available < command_bytes || (i && params.stride > (available - command_bytes) / i)) {
            break;
        }
        const u64 offset = command_offset + i * params.stride;
        copies[sampled++] = VkBufferCopy{offset, i * 8ULL, command_bytes};
    }
    VkBuffer count_buffer{};
    VkBufferCopy count_copy{};
    if (params.include_count && !params.is_byte_count) {
        const auto [count, offset] = buffer_cache.GetDrawIndirectCount();
        if (offset > count->SizeBytes() || sizeof(u32) > count->SizeBytes() - offset) {
            ++trace_capture_dropped;
            return;
        }
        count_buffer = count->Handle();
        count_copy = VkBufferCopy{offset, 64, sizeof(u32)};
    }
    if (!sampled && !count_buffer) {
        LOG_INFO(Render_Vulkan, "DKB indirect frame={} address={:#x} max_draws={} sampled=0",
                 trace_frame, params.indirect_start_address, max_draws);
        return;
    }
    if (!slot->readback) {
        slot->readback = memory_allocator.CreateBuffer(
            VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = 68,
                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            },
            MemoryUsage::Download);
    }
    slot->frame = trace_frame;
    slot->address = params.indirect_start_address;
    slot->count_address = params.count_start_address;
    slot->max_draws = max_draws;
    slot->sampled_commands = sampled;
    slot->include_count = count_buffer != VK_NULL_HANDLE;
    slot->byte_count = params.is_byte_count;
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([source = commands->Handle(), destination = *slot->readback, copies, sampled,
                      count_buffer, count_copy](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier before{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, before);
        if (sampled) {
            cmdbuf.CopyBuffer(source, destination, vk::Span(copies.data(), sampled));
        }
        if (count_buffer) {
            cmdbuf.CopyBuffer(count_buffer, destination, count_copy);
        }
        const VkMemoryBarrier after{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                               0, after);
    });
    slot->tick = scheduler.CurrentTick();
    slot->pending = true;
    scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
    scheduler.NoteRenderPassStores();
}

void RasterizerVulkan::CollectIndirectTrace() {
    for (auto& trace : indirect_traces) {
        if (!trace.pending || !scheduler.IsFree(trace.tick)) {
            continue;
        }
        trace.readback.Invalidate();
        std::array<u32, 17> words{};
        std::memcpy(words.data(), trace.readback.Mapped().data(), sizeof(words));
        if (trace.byte_count) {
            LOG_INFO(Render_Vulkan, "DKB indirect frame={} address={:#x} gpu_byte_count={}",
                     trace.frame, trace.address, words[0]);
        } else {
            const u32 count = trace.include_count ? words[16] : trace.max_draws;
            const u32 requested_draws = std::min(count, trace.max_draws);
            const u32 sampled = std::min(requested_draws, trace.sampled_commands);
            u32 zero_commands{};
            std::string commands;
            for (u32 i = 0; i < sampled; ++i) {
                zero_commands += words[2 * i] == 0 || words[2 * i + 1] == 0;
                commands += fmt::format("{}{}x{}", i ? "," : "", words[2 * i], words[2 * i + 1]);
            }
            LOG_INFO(Render_Vulkan,
                     "DKB indirect frame={} address={:#x} count_address={:#x} draw_count={} "
                     "count_source={} max_draws={} requested_draws={} sampled={} zero_commands={} "
                     "vertices_or_indices_x_instances=[{}] conditional_execution=unobserved",
                     trace.frame, trace.address, trace.count_address, count,
                     trace.include_count ? "gpu" : "command-parameter",
                     trace.max_draws, requested_draws, sampled, zero_commands, commands);
        }
        trace.pending = false;
    }
}

void RasterizerVulkan::DrawTexture() {
    if (VideoCommon::GpuSplit::Unlends()) {
        VideoCommon::GpuSplit::Get().NoteLent(VideoCommon::GpuSplit::UnlendReason::Texture);
    }
    RENDER_LEND(DrawTexture())

    SCOPE_EXIT {
        gpu.TickWork();
    };
    FlushWork();

    std::scoped_lock l{texture_cache.mutex};
    texture_cache.SynchronizeDescriptors(false);
    texture_cache.UpdateRenderTargets(false);

    UpdateDynamicStates(nullptr);

    query_cache.NotifySegment(true);
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    const auto& draw_texture_state = maxwell3d->draw_manager.draw_texture_state;
    const auto& sampler = texture_cache.GetSampler(draw_texture_state.src_sampler, false);
    const auto& texture = texture_cache.GetImageView(draw_texture_state.src_texture);
    const auto* framebuffer = texture_cache.GetFramebuffer();
    if (frame_dump.Capturing()) {
        FrameDumpPass(framebuffer);
    }

    const bool src_rescaling = texture_cache.IsRescaling() && texture.IsRescaled();
    const bool dst_rescaling = texture_cache.IsRescaling() && framebuffer->IsRescaled();

    const auto ScaleSrc = [&](auto dim_f) -> s32 {
        auto dim = static_cast<s32>(dim_f);
        return src_rescaling ? Settings::values.resolution_info.ScaleUp(dim) : dim;
    };

    const auto ScaleDst = [&](auto dim_f) -> s32 {
        auto dim = static_cast<s32>(dim_f);
        return dst_rescaling ? Settings::values.resolution_info.ScaleUp(dim) : dim;
    };

    Region2D dst_region = {Offset2D{.x = ScaleDst(draw_texture_state.dst_x0),
                                    .y = ScaleDst(draw_texture_state.dst_y0)},
                           Offset2D{.x = ScaleDst(draw_texture_state.dst_x1),
                                    .y = ScaleDst(draw_texture_state.dst_y1)}};
    Region2D src_region = {Offset2D{.x = ScaleSrc(draw_texture_state.src_x0),
                                    .y = ScaleSrc(draw_texture_state.src_y0)},
                           Offset2D{.x = ScaleSrc(draw_texture_state.src_x1),
                                    .y = ScaleSrc(draw_texture_state.src_y1)}};
    Extent3D src_size = {static_cast<u32>(ScaleSrc(texture.size.width)),
                         static_cast<u32>(ScaleSrc(texture.size.height)), texture.size.depth};
    blit_image.BlitColor(framebuffer, texture.RenderTarget(), texture.ImageHandle(),
                         sampler->Handle(), dst_region, src_region, src_size);
}

// A 1-layer depth target bound with a 3D color target limits the framebuffer to one layer, so
// vkCmdClearAttachments can't reach the target's other slices. Those slices are zeroed by copying
// zeros into the clear's rectangle of exactly the slices the clear names; the rest of the volume
// keeps its contents. The rectangle can be smaller than the target: in handheld mode the game
// clears a 48-wide fog volume through a 64-wide render target.
bool RasterizerVulkan::ClearVolumeSlices(const Maxwell& regs, const Framebuffer* framebuffer,
                                         u32 first_slice, u32 slice_count) {
    const u32 target_index = regs.clear_surface.RT;
    const auto& color_buffer_ids = texture_cache.GetRenderTargets().color_buffer_ids;
    // Depth and stencil clears take the normal path, so only a colour-only clear qualifies.
    if (target_index >= color_buffer_ids.size() || regs.clear_surface.Z || regs.clear_surface.S) {
        return false;
    }
    const auto view_id = color_buffer_ids[target_index];
    const u32 depth = regs.rt[target_index].depth;
    if (!view_id || depth <= framebuffer->Layers() || first_slice >= depth) {
        return false;
    }
    const auto& view = texture_cache.GetImageView(view_id);
    const bool is_volume = True(view.flags & VideoCommon::ImageViewFlagBits::Slice);
    const bool clears_all_channels = regs.clear_surface.R && regs.clear_surface.G &&
                                     regs.clear_surface.B && regs.clear_surface.A;
    const bool clears_to_zero =
        std::ranges::all_of(regs.clear_color, [](f32 value) { return value == 0.0f; });
    if (!is_volume || !clears_all_channels || !clears_to_zero) {
        return false;
    }
    const VkRect2D rect = VolumeClearRect(regs, framebuffer->RenderArea(), view.size);
    if (rect.extent.width == 0 || rect.extent.height == 0) {
        return true;
    }
    const u32 width = rect.extent.width;
    const u32 height = rect.extent.height;
    // The host image can use a wider format than the guest's (B10G11R11 falls back to
    // R16G16B16A16): size the zeros for the widest color texel, 16 bytes.
    constexpr size_t WidestTexelBytes = 16;
    const size_t slice_bytes = size_t{width} * height * WidestTexelBytes;
    const auto zeros = staging_pool.Request(slice_bytes, MemoryUsage::Upload);
    std::memset(zeros.mapped_span.data(), 0, slice_bytes);
    const u32 count = std::min(slice_count, depth - first_slice);
    const VkImage image = view.ImageHandle();
    const u32 level = view.range.base.level;
    // The clear's slices count from the view's first slice.
    const u32 base_slice = static_cast<u32>(view.range.base.layer);
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image, level, base_slice, rect, width, height, first_slice, count,
                      buffer = zeros.buffer, offset = zeros.offset](vk::CommandBuffer cmdbuf) {
        // Earlier passes may still be reading or writing the volume.
        constexpr VkMemoryBarrier clear_waits{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, clear_waits);
        for (u32 slice = first_slice; slice < first_slice + count; ++slice) {
            const VkBufferImageCopy zero_slice{
                .bufferOffset = offset,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
                .imageOffset = {rect.offset.x, rect.offset.y, static_cast<s32>(base_slice + slice)},
                .imageExtent = {width, height, 1},
            };
            cmdbuf.CopyBufferToImage(buffer, image, VK_IMAGE_LAYOUT_GENERAL, zero_slice);
        }
        constexpr VkMemoryBarrier clear_finished{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               0, clear_finished);
    });
    return true;
}

struct RasterizerVulkan::UnlentClear {
    RasterizerVulkan* rasterizer;
    Tegra::Engines::Maxwell3D* engine;
    VideoCommon::Nx2RenderView::Packet view;
    u32 layer_count;
    u64 position{};

    void operator()() {
        rasterizer->PlayUnlentClear(*this);
    }
};

bool RasterizerVulkan::TryUnlendClear(u32 layer_count) {
    using Lent = VideoCommon::GpuSplit::ClearLent;
    namespace View = VideoCommon::Nx2RenderView;
    auto& split = VideoCommon::GpuSplit::Get();
    const auto lend = [&split](Lent reason) {
        split.NoteClearLent(reason);
        return false;
    };
    static const bool window_trace = std::getenv("EDEN_NX2_WINDOW_TRACE") != nullptr;
    if (window_trace || !split.RenderPosting() || frame_dump.Capturing() ||
        VideoCommon::RenderTrace::Enabled() || GPU::Logging::IsActive() ||
        Tegra::Engines::g_nx2_trace_draws_after_upscale > 0) {
        return lend(Lent::Diagnostics);
    }
    const auto& regs = maxwell3d->regs;
    const auto& surface = regs.clear_surface;
    // Colour-only clears of a volume target can take ClearVolumeSlices (a staging upload).
    if (!surface.Z && !surface.S && surface.RT < regs.rt.size() && regs.rt[surface.RT].depth > 1) {
        return lend(Lent::Volume);
    }
    if (const u64 verify_every = RenderVerifyEvery();
        verify_every != 0 && ++unlend_clear_candidates % verify_every == 0) {
        unlend_verify_clear = true;
        return lend(Lent::Verify);
    }
    unlend_gather_hit = false;
    gpu_memory->FlushCaching();
    if (unlend_gather_hit) {
        return lend(Lent::Gather);
    }
    if (surface.R || surface.G || surface.B || surface.A || surface.Z || surface.S) {
        if (!maxwell3d->render_view_writes) {
            maxwell3d->render_view_writes = std::make_unique<View::Writes>(*maxwell3d);
        }
        UnlentClear clear{
            .rasterizer = this,
            .engine = maxwell3d,
            .view = maxwell3d->render_view_writes->Pack(*maxwell3d, true),
            .layer_count = layer_count,
        };
        // The packet raised the forwarded flags: GPURender consumes them on the view. Shaders
        // stays: the back's next pipeline lookup consumes it (ShaderCache::RefreshStages), and a
        // clear runs none, so dropping it kept the previous draw's shaders.
        static const View::Flags taken = [] {
            View::Flags flags = View::Forwarded();
            flags.reset(VideoCommon::Dirty::Shaders);
            return flags;
        }();
        maxwell3d->dirty.flags &= ~taken;
        split.PostUnlent(std::move(clear), VideoCommon::GpuSplit::UnlentKind::Clear);
    }
    if (gpu.HasSyncRequests()) {
        split.LendRender([this] { gpu.TickWork(); });
    }
    return true;
}

void RasterizerVulkan::PlayUnlentClear(UnlentClear& clear) {
    Tegra::Engines::Maxwell3D& view = clear.engine->render_view_writes->View();
    VideoCommon::Nx2RenderView::Apply(view, clear.view);
    VideoCommon::GpuSplit::SetPlayingPosition(clear.position);
    render_engine = &view;
    Tegra::Engines::Maxwell3D* const texture_engine = texture_cache.SwapMaxwell3D(&view);
    Tegra::Engines::Maxwell3D* const query_engine = query_cache.SwapMaxwell3D(&view);
    auto* const flags = state_tracker.SwapFlags(&view.dirty.flags);
    SCOPE_EXIT {
        state_tracker.SwapFlags(flags);
        query_cache.SwapMaxwell3D(query_engine);
        texture_cache.SwapMaxwell3D(texture_engine);
        render_engine = nullptr;
        VideoCommon::GpuSplit::SetPlayingPosition(~u64{0});
    };
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Clear};
    FlushWork();
    ClearImpl(view.regs, clear.layer_count);
}

void RasterizerVulkan::Clear(u32 layer_count) {
    if (VideoCommon::GpuSplit::LendsToRender()) [[unlikely]] {
        if (VideoCommon::GpuSplit::UnlendsClear() && TryUnlendClear(layer_count)) {
            return;
        }
        VideoCommon::GpuSplit::Get().LendRender([&] { Clear(layer_count); });
        return;
    }
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Clear};
    if (std::exchange(unlend_verify_clear, false)) {
        const VideoCommon::Nx2RenderView::DrawCheck view_check{*maxwell3d, true};
        if (view_check.Differed()) {
            VideoCommon::GpuSplit::Get().NoteUnlentClearView();
        }
    }
    if (Tegra::Engines::g_nx2_trace_draws_after_upscale > 0) {
        const auto& regs = maxwell3d->regs;
        LOG_INFO(HW_GPU, "after upscale: clear rt0 {:#x} {}x{} fmt {:#x} surface {:#x} colour {} {} {} {}",
                 regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height,
                 static_cast<u32>(regs.rt[0].format), regs.clear_surface.raw, regs.clear_color[0],
                 regs.clear_color[1], regs.clear_color[2], regs.clear_color[3]);
    }
    TraceWindowTarget(*maxwell3d, *gpu_memory, "clear", layer_count);
    FlushWork();
    gpu_memory->FlushCaching();
    ClearImpl(maxwell3d->regs, layer_count);
}

void RasterizerVulkan::ClearImpl(Maxwell& regs, u32 layer_count) {
    const bool use_color = regs.clear_surface.R || regs.clear_surface.G || regs.clear_surface.B ||
                           regs.clear_surface.A;
    const bool use_depth = regs.clear_surface.Z;
    const bool use_stencil = regs.clear_surface.S;
    if (!use_color && !use_depth && !use_stencil) {
        return;
    }

    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.UpdateRenderTargets(true);
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    if (frame_dump.Capturing()) {
        FrameDumpPass(framebuffer);
    }
    const VkExtent2D render_area = framebuffer->RenderArea();
    const u32 base_array_layer = regs.clear_surface.layer;
    const auto clear_layers =
        GetFramebufferClearLayers(base_array_layer, layer_count, framebuffer->Layers());
    if (clear_layers.count != layer_count) {
        LOG_DEBUG(Render_Vulkan,
                  "Framebuffer clear layers: base={}, requested={}, clipped={}, framebuffer={}",
                  base_array_layer, layer_count, clear_layers.count, framebuffer->Layers());
    }
    if (ClearVolumeSlices(regs, framebuffer, base_array_layer, layer_count)) {
        return;
    }
    if (clear_layers.count == 0) {
        return;
    }

    constexpr bool ENABLE_DEFERRED_CLEAR = true;
    const bool color_full_channels = regs.clear_surface.R && regs.clear_surface.G &&
                                     regs.clear_surface.B && regs.clear_surface.A;
    const bool stencil_partial = use_stencil && framebuffer->HasAspectStencilBit() &&
                                 regs.stencil_front_mask != 0xFF && regs.stencil_front_mask != 0;
    const bool ds_used = use_depth || use_stencil;
    const bool ds_deferrable =
        !ds_used || ((!framebuffer->HasAspectDepthBit() || use_depth) &&
                     (!framebuffer->HasAspectStencilBit() || use_stencil) && !stencil_partial);
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (texture_cache.IsRescaling()) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    VkRect2D default_scissor{};
    default_scissor.offset.x = 0;
    default_scissor.offset.y = 0;
    default_scissor.extent.width = (std::numeric_limits<s32>::max)();
    default_scissor.extent.height = (std::numeric_limits<s32>::max)();

    VkClearRect clear_rect{
        .rect = regs.clear_control.use_scissor ? GetScissorState(regs, 0, up_scale, down_shift)
                                               : default_scissor,
        .baseArrayLayer = base_array_layer,
        .layerCount = clear_layers.count,
    };
    const auto clamp_rect_to_render_area = [render_area](VkRect2D& rect) -> bool {
        const auto clamp_axis = [](s32& offset, u32& extent, u32 limit) {
            auto clamp_offset = [&offset, limit]() {
                if (limit == 0) {
                    offset = 0;
                    return;
                }
                offset = std::clamp(offset, 0, static_cast<s32>(limit));
            };

            if (extent == 0) {
                clamp_offset();
                return;
            }
            if (offset < 0) {
                const u32 shrink = (std::min)(extent, static_cast<u32>(-offset));
                extent -= shrink;
                offset = 0;
            }
            if (limit == 0) {
                extent = 0;
                offset = 0;
                return;
            }
            if (offset >= s32(limit)) {
                offset = s32(limit);
                extent = 0;
                return;
            }
            const u64 end_coord = u64(offset) + extent;
            if (end_coord > limit) {
                extent = limit - u32(offset);
            }
        };

        clamp_axis(rect.offset.x, rect.extent.width, render_area.width);
        clamp_axis(rect.offset.y, rect.extent.height, render_area.height);
        return rect.extent.width != 0 && rect.extent.height != 0;
    };
    if (!clamp_rect_to_render_area(clear_rect.rect)) {
        return;
    }

    if (VideoCommon::RenderTrace::SampleFrame(render_trace_frame) && render_trace_clears++ < 2048) {
        const u32 target = regs.clear_surface.RT;
        LOG_DEBUG(Render_Vulkan,
                  "Render trace clear: frame={} docked={} color={} depth={} stencil={} "
                  "color_target={} color_address={:#x} depth_address={:#x} depth_value={} base_layer={} layers={} "
                  "framebuffer_layers={} rect=({},{} {}x{})",
                  render_trace_frame, Settings::IsDockedMode(), use_color, use_depth, use_stencil,
                  target, target < regs.rt.size() ? regs.rt[target].Address() : 0,
                  regs.zeta.Address(), regs.clear_depth,
                  clear_rect.baseArrayLayer, clear_rect.layerCount, framebuffer->Layers(),
                  clear_rect.rect.offset.x, clear_rect.rect.offset.y,
                  clear_rect.rect.extent.width, clear_rect.rect.extent.height);
    }

    const bool clear_covers_render_area =
        clear_rect.rect.offset.x == 0 && clear_rect.rect.offset.y == 0 &&
        clear_rect.rect.extent.width >= render_area.width &&
        clear_rect.rect.extent.height >= render_area.height;
    const bool can_defer_clear = ENABLE_DEFERRED_CLEAR && (!regs.clear_control.use_scissor || clear_covers_render_area) &&
                                 clear_layers.covers_all_layers &&
                                 !scheduler.IsRenderPassActive() &&
                                 (!use_color || color_full_channels) && ds_deferrable;
    if (!can_defer_clear) {
        scheduler.RequestRenderpass(framebuffer);
    }

    query_cache.NotifySegment(true);
    // Deferred load-op clears have no render pass yet. Starting a query here
    // would begin it outside the pass and end it inside the next draw's pass.
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64,
                              !can_defer_clear && regs.zpass_pixel_count_enable);
    UpdateViewportsState(regs);

    const u32 color_attachment = regs.clear_surface.RT;
    if (use_color && framebuffer->HasAspectColorBit(color_attachment)) {
        const auto format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[color_attachment].format);
        bool is_integer = IsPixelFormatInteger(format);
        bool is_signed = IsPixelFormatSignedInteger(format);
        size_t int_size = PixelComponentSizeBitsInteger(format);
        VkClearValue clear_value{};
        if (!is_integer) {
            std::memcpy(clear_value.color.float32, regs.clear_color.data(), regs.clear_color.size() * sizeof(f32));
        } else if (!is_signed) {
            for (size_t i = 0; i < 4; i++)
                clear_value.color.uint32[i] = u32(f32(u64(int_size) << 1U) * regs.clear_color[i]);
        } else {
            for (size_t i = 0; i < 4; i++)
                clear_value.color.int32[i] = s32(f32(s64(int_size - 1) << 1) * (regs.clear_color[i] - 0.5f));
        }

        if (color_full_channels) {
            if (can_defer_clear) {
                scheduler.DeferColorClear(framebuffer, color_attachment, clear_value);
            } else {
                scheduler.Record([color_attachment, clear_value, clear_rect](vk::CommandBuffer cmdbuf) {
                    const VkClearAttachment attachment{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .colorAttachment = color_attachment,
                        .clearValue = clear_value,
                    };
                    cmdbuf.ClearAttachments(attachment, clear_rect);
                });
            }
        } else {
            u8 color_mask = u8(regs.clear_surface.R | regs.clear_surface.G << 1 | regs.clear_surface.B << 2 | regs.clear_surface.A << 3);
            Region2D dst_region = {
                Offset2D{.x = clear_rect.rect.offset.x, .y = clear_rect.rect.offset.y},
                Offset2D{.x = clear_rect.rect.offset.x + s32(clear_rect.rect.extent.width),
                         .y = clear_rect.rect.offset.y + s32(clear_rect.rect.extent.height)}};
            blit_image.ClearColor(framebuffer, color_mask, regs.clear_color, dst_region);
        }
    }

    if (!use_depth && !use_stencil) {
        return;
    }
    VkImageAspectFlags aspect_flags = 0;
    if (use_depth && framebuffer->HasAspectDepthBit()) {
        aspect_flags |= VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    if (use_stencil && framebuffer->HasAspectStencilBit()) {
        aspect_flags |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    if (aspect_flags == 0) {
        return;
    }

    if (use_stencil && framebuffer->HasAspectStencilBit() && regs.stencil_front_mask != 0xFF &&
        regs.stencil_front_mask != 0) {
        Region2D dst_region = {
            Offset2D{.x = clear_rect.rect.offset.x, .y = clear_rect.rect.offset.y},
            Offset2D{.x = clear_rect.rect.offset.x + s32(clear_rect.rect.extent.width),
                     .y = clear_rect.rect.offset.y + s32(clear_rect.rect.extent.height)}};
        blit_image.ClearDepthStencil(framebuffer, use_depth, regs.clear_depth,
                                     u8(regs.stencil_front_mask), regs.clear_stencil,
                                     regs.stencil_front_func_mask, dst_region);
    } else if (can_defer_clear) {
        VkClearValue ds_value{};
        ds_value.depthStencil.depth = regs.clear_depth;
        ds_value.depthStencil.stencil = regs.clear_stencil;
        scheduler.DeferDepthStencilClear(framebuffer, ds_value);
    } else {
        scheduler.Record([clear_depth = regs.clear_depth, clear_stencil = regs.clear_stencil,
                          clear_rect, aspect_flags](vk::CommandBuffer cmdbuf) {
            VkClearAttachment attachment;
            attachment.aspectMask = aspect_flags;
            attachment.colorAttachment = 0;
            attachment.clearValue.depthStencil.depth = clear_depth;
            attachment.clearValue.depthStencil.stencil = clear_stencil;
            cmdbuf.ClearAttachments(attachment, clear_rect);
        });
    }
}

struct RasterizerVulkan::UnlentDispatch {
    RasterizerVulkan* rasterizer;
    Tegra::Engines::Maxwell3D* engine;
    ComputePipeline* pipeline;
    ComputePipeline::BackHalf half;
    VideoCommon::ComputePools pools;
    std::array<u32, 3> grid;
    bool indirect{};
    VkBuffer indirect_buffer{};
    VkDeviceSize indirect_offset{};
    std::optional<u32> list_slot;
    u64 position{};

    void operator()() {
        rasterizer->PlayUnlentDispatch(*this);
    }
};

bool RasterizerVulkan::TryUnlendDispatch() {
    using Lent = VideoCommon::GpuSplit::DispatchLent;
    namespace Profile = VideoCommon::Nx2Profile;
    auto& split = VideoCommon::GpuSplit::Get();
    const auto lend = [&split](Lent reason) {
        split.NoteDispatchLent(reason);
        return false;
    };
    static const bool pending_write_trace = std::getenv("EDEN_NX2_PENDING_WRITE_TRACE") != nullptr;
    if (pending_write_trace || !split.RenderPosting() || frame_dump.Capturing() ||
        Tegra::Nx2ListIndirect::Stats::g_check || Common::DkbFrameTrace::Enabled() ||
        VideoCommon::RenderTrace::Enabled() || GPU::Logging::IsActive() ||
        VideoCommon::HandleTableNotes::Wanted()) {
        return lend(Lent::Diagnostics);
    }
    if (maxwell3d == nullptr) {
        return lend(Lent::NoView);
    }
    const auto& qmd = kepler_compute->launch_description;
    const auto indirect_address = kepler_compute->GetIndirectComputeAddress();
    const auto list_slot = kepler_compute->GetIndirectListSlot();
    const std::array<u32, 3> grid{qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z};
    if (!indirect_address && !list_slot) {
        const std::array<u32, 3> max_dim{device.GetMaxComputeWorkGroupCount()};
        if (grid[0] > max_dim[0] || grid[1] > max_dim[1] || grid[2] > max_dim[2]) {
            return lend(Lent::Grid);
        }
    }
    auto& slots = buffer_cache_runtime.Nx2Slots();
    if (list_slot && !slots.Held(*list_slot)) {
        return lend(Lent::SlotUnheld);
    }
    unlend_gather_hit = false;
    gpu_memory->FlushCaching();
    if (unlend_gather_hit) {
        return lend(Lent::Gather);
    }
    u64 native_key{};
    ComputePipeline* const pipeline = [&] {
        Profile::Scope scope{Profile::Zone::PipelineLookup};
        return pipeline_cache.UnlentComputePipeline(native_key);
    }();
    if (pipeline == nullptr) {
        return lend(Lent::Lookup);
    }
    if (!pipeline->CanUnlend()) {
        return lend(Lent::NotBuilt);
    }
    // Ocarina of Time's hooks read the dispatch's registers and GPURender's state together.
    if (native_key == VideoCommon::OcarinaDepthReaderKey || native_key == 0x6e0ae200ee9aa220ULL ||
        VideoCommon::FindOcarinaUpscalePass(native_key, pipeline->ShaderHash()) !=
            VideoCommon::OcarinaUpscalePass::None) {
        return lend(Lent::Hook);
    }
    // As for draws: the back reads handles and storage buffer descriptors from these constant
    // buffers itself, and an owed list word there would read its placeholder.
    if (slots.HasOwed()) {
        const auto& cbufs = qmd.const_buffer_config;
        for (u32 bits = pipeline->HandleConstBuffers() & qmd.const_buffer_enable_mask; bits != 0;
             bits &= bits - 1) {
            const u32 index = static_cast<u32>(std::countr_zero(bits));
            if (index >= cbufs.size()) {
                break;
            }
            const GPUVAddr address = cbufs[index].Address();
            const u32 size = cbufs[index].size;
            if (address == 0 || size == 0) {
                continue;
            }
            for (const auto& [device_addr, bytes] :
                 gpu_memory->GetSubmappedDeviceRange(address, size)) {
                if (slots.Owes(device_addr, bytes)) {
                    return lend(Lent::ListOwed);
                }
            }
        }
    }
    if (!maxwell3d->render_view_writes) {
        maxwell3d->render_view_writes =
            std::make_unique<VideoCommon::Nx2RenderView::Writes>(*maxwell3d);
    }
    const bool linked_tsc = qmd.linked_tsc != 0;
    const auto& compute_regs = kepler_compute->regs;
    UnlentDispatch dispatch{
        .rasterizer = this,
        .engine = maxwell3d,
        .pipeline = pipeline,
        .half = {},
        .pools =
            {
                .tic = compute_regs.tic.Address(),
                .tic_limit = compute_regs.tic.limit,
                .tsc = compute_regs.tsc.Address(),
                .tsc_limit = linked_tsc ? compute_regs.tic.limit : compute_regs.tsc.limit,
                .linked_tsc = linked_tsc,
            },
        .grid = grid,
        .indirect = indirect_address.has_value(),
        .list_slot = list_slot,
    };
    std::unique_lock lock{buffer_cache.mutex};
    buffer_cache.Nx2CollectListCopies(true);
    {
        Profile::Scope scope{Profile::Zone::Configure};
        const BufferArena::Collect collect{dispatch.half.commands};
        pipeline->ConfigureBack(*kepler_compute, *gpu_memory, buffer_cache, dispatch.half,
                                UnlendCapture());
        if (indirect_address) {
            const auto [buffer, offset] =
                VideoCommon::ObtainIndirectBuffer(buffer_cache, *indirect_address);
            dispatch.indirect_buffer = buffer->Handle();
            dispatch.indirect_offset = offset;
        }
    }
    // Early downloads and list copies are recorded behind the dispatch: lent after it.
    const bool early_downloads = buffer_cache.HasEarlyDownloads();
    const bool list_copies = buffer_cache.Nx2HasListCopyRequests();
    if (!early_downloads) {
        Profile::Scope scope{Profile::Zone::ComputeDownloads};
        buffer_cache.NoteWorkRecorded(true);
    }
    if (!list_copies) {
        buffer_cache.Nx2CollectListCopies(false);
    }
    lock.unlock();
    dispatch.half.stamp = buffer_cache_runtime.Arena().CloseCollected();
    const auto& pools = dispatch.pools;
    split.NotePools(pools.tic, (u64{pools.tic_limit} + 1) * 32, pools.tsc,
                    (u64{pools.tsc_limit} + 1) * 32);
    split.PostUnlent(std::move(dispatch), VideoCommon::GpuSplit::UnlentKind::Dispatch);
    split.NoteUnlentDispatch(indirect_address.has_value(), list_slot.has_value());
    if (early_downloads || list_copies) {
        split.NoteDispatchTail();
        split.LendRender([this, early_downloads] {
            std::scoped_lock tail_lock{buffer_cache.mutex};
            Profile::Scope scope{Profile::Zone::ComputeDownloads};
            if (early_downloads) {
                buffer_cache.NoteWorkRecorded(true);
            }
            SubmitNx2ListCopies();
            buffer_cache.Nx2CollectListCopies(false);
        });
    }
    if (gpu.HasSyncRequests()) {
        split.LendRender([this] { gpu.TickWork(); });
    }
    return true;
}

void RasterizerVulkan::PlayUnlentDispatch(UnlentDispatch& dispatch) {
    namespace Profile = VideoCommon::Nx2Profile;
    auto& split = VideoCommon::GpuSplit::Get();
    VideoCommon::GpuSplit::SetPlayingPosition(dispatch.position);
    // Its submits invalidate the view's flags, not the back's engine's.
    auto* const flags =
        state_tracker.SwapFlags(&dispatch.engine->render_view_writes->View().dirty.flags);
    SCOPE_EXIT {
        state_tracker.SwapFlags(flags);
        VideoCommon::GpuSplit::SetPlayingPosition(~u64{0});
    };
    Profile::Scope profile_scope{Profile::Zone::Compute};
    FlushWork(true);
    auto& slots = buffer_cache_runtime.Nx2Slots();
    if (dispatch.list_slot && !slots.Held(*dispatch.list_slot)) {
        split.NoteUnlentSlotUnheld();
        return;
    }
    std::scoped_lock lock{texture_cache.mutex};
    ComputePipeline* const pipeline = dispatch.pipeline;
    const bool direct = !dispatch.indirect && !dispatch.list_slot;
    // Until a dispatch overwrites the scene, each direct one binds only after the effects are
    // marked.
    const bool marking_effects =
        direct && oot_effects_frame == render_trace_frame && OcarinaTemporalEnabled();
    ComputePipeline::BindCommand deferred_bind;
    bool configured;
    {
        Profile::Scope scope{Profile::Zone::Configure};
        configured = pipeline->ConfigureRender(dispatch.half, dispatch.pools, scheduler,
                                               texture_cache, buffer_cache_runtime,
                                               marking_effects ? &deferred_bind : nullptr);
    }
    if (!configured) {
        split.NoteUnlentDispatchFailed();
        return;
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.NoteComputeWrites();
    if (!direct) {
        VkBuffer buffer = dispatch.indirect_buffer;
        VkDeviceSize offset = dispatch.indirect_offset;
        if (dispatch.list_slot) {
            std::tie(buffer, offset) = slots.DispatchArguments(*dispatch.list_slot);
            Tegra::Nx2ListIndirect::Stats::Count(
                Tegra::Nx2ListIndirect::Stats::Get().slot_dispatches);
        } else if (buffer == VK_NULL_HANDLE) [[unlikely]] {
            // An unmapped argument address on a nullDescriptor device (Fritchii 763871b779).
            return;
        }
        scheduler.TraceComputePipeline(pipeline->ShaderHash());
        scheduler.Record([this, pipeline, buffer, offset](vk::CommandBuffer cmdbuf) {
            RecordDispatchBarrier(cmdbuf);
            if (!pipeline->IsBound()) {
                return;
            }
            scheduler.TraceGpuWork(cmdbuf, pipeline->ShaderHash(), 0, {},
                                   [&] { cmdbuf.DispatchIndirect(buffer, offset); });
        });
        return;
    }
    if (marking_effects) {
        scheduler.Record([](vk::CommandBuffer cmdbuf) { RecordDispatchBarrier(cmdbuf); });
        if (std::ranges::any_of(pipeline->WrittenImages(), [this](VideoCommon::ImageViewId id) {
                return id && texture_cache.GetImageView(id).gpu_addr == oot_effects_address;
            })) {
            oot_live_dlss->MarkEffects(oot_effects_scene, render_trace_frame);
            oot_effects_frame.reset();
        }
    }
    scheduler.TraceComputePipeline(pipeline->ShaderHash());
    scheduler.Record([this, pipeline, grid = dispatch.grid, bind = std::move(deferred_bind),
                      marking_effects](vk::CommandBuffer cmdbuf) {
        if (!marking_effects) {
            RecordDispatchBarrier(cmdbuf);
        }
        if (bind) {
            bind(cmdbuf);
        }
        if (!pipeline->IsBound()) {
            return;
        }
        scheduler.TraceGpuWork(cmdbuf, pipeline->ShaderHash(), 0, grid,
                               [&] { cmdbuf.Dispatch(grid[0], grid[1], grid[2]); });
    });
    if (marking_effects) {
        // The deferred native binding may follow DLSS's command buffer bindings.
        scheduler.InvalidateState();
    }
}

void RasterizerVulkan::DispatchCompute() {
    if (VideoCommon::GpuSplit::LendsToRender()) [[unlikely]] {
        if (VideoCommon::GpuSplit::UnlendsDispatch() && TryUnlendDispatch()) {
            return;
        }
        VideoCommon::GpuSplit::Get().LendRender([&] { DispatchCompute(); });
        return;
    }
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Compute};
    FlushWork(true);
    gpu_memory->FlushCaching();
    if (frame_dump.Capturing()) {
        frame_dump.EndPass();
        frame_dump.DrainIfFull();
    }

    u64 native_key{};
    ComputePipeline* const pipeline{[&] {
        VideoCommon::Nx2Profile::Scope scope{VideoCommon::Nx2Profile::Zone::PipelineLookup};
        return pipeline_cache.CurrentComputePipeline(native_key);
    }()};
    if (!pipeline) {
        if (kepler_compute->IsNativeMode()) throw Shader::RuntimeError("Native compute pipeline unavailable");
        return;
    }
    std::scoped_lock lock{texture_cache.mutex, buffer_cache.mutex};
    std::optional<ByteAddressedShadow> byte_addressed_shadow;
    buffer_cache.Nx2CollectListCopies(true);
    SCOPE_EXIT {
        buffer_cache.Nx2CollectListCopies(false);
        if (byte_addressed_shadow) {
            texture_cache_runtime.RetireByteAddressedShadow(*byte_addressed_shadow);
        }
    };
    const auto indirect_address = kepler_compute->GetIndirectComputeAddress();
    const auto list_slot = kepler_compute->GetIndirectListSlot();
    const bool can_replace =
        OcarinaTemporalEnabled() &&
        VideoCommon::FindOcarinaUpscalePass(native_key, pipeline->ShaderHash()) !=
            VideoCommon::OcarinaUpscalePass::None &&
        pipeline->CanReplaceOcarinaUpscale() && !indirect_address && !list_slot &&
        !frame_dump.Capturing();
    // Until a dispatch overwrites the scene, each direct one binds only after the effects are
    // marked. The indirect path records its bind at once.
    const bool marking_effects = oot_effects_frame == render_trace_frame &&
                                 OcarinaTemporalEnabled() && !indirect_address && !list_slot &&
                                 !frame_dump.Capturing();
    ComputePipeline::BindCommand deferred_bind;
    switch (pipeline->Configure(*kepler_compute, *gpu_memory, scheduler, buffer_cache,
                                texture_cache, texture_cache_runtime, pipeline_cache,
                                byte_addressed_shadow,
                                can_replace || marking_effects ? &deferred_bind : nullptr)) {
    case ComputePipeline::ConfigureResult::Ready:
        break;
    case ComputePipeline::ConfigureResult::Failed:
        if (kepler_compute->IsNativeMode()) throw Shader::RuntimeError("Native compute resources unavailable");
        return;
    case ComputePipeline::ConfigureResult::Skipped:
        return;
    }
    SCOPE_EXIT {
        if (!frame_dump.Capturing()) {
            return;
        }
        boost::container::small_vector<ImageView*, 8> written;
        for (const VideoCommon::ImageViewId id : pipeline->WrittenImages()) {
            written.push_back(&texture_cache.GetImageView(id));
        }
        frame_dump.CaptureDispatch(pipeline->ShaderHash(), written);
    };
    TracePendingCpuWrites(true, std::array{pipeline->ShaderHash()});
    if (!frame_dump.Focused() && frame_dump.WantsShaders(pipeline)) {
        const auto& launch = kepler_compute->launch_description;
        frame_dump.NoteComputeShaders(
            *pipeline, pipeline_cache.CaptureComputeProgram(),
            std::span(reinterpret_cast<const u8*>(&launch), sizeof(launch)));
    }
    if (!frame_dump.Focused() && frame_dump.CapturingSampled()) {
        boost::container::small_vector<FrameDump::Sampled, 16> sampled;
        for (const auto& image : pipeline->SampledImages()) {
            sampled.push_back({5, image.tic, &texture_cache.GetImageView(image.id),
                               &texture_cache.GetSampler(image.sampler)});
        }
        frame_dump.NoteDispatchSampled(pipeline->ShaderHash(), sampled);
    }
    if (frame_dump.CapturingBuffers()) {
        FrameDumpBuffers(true, std::array{pipeline->ShaderHash()});
    }

    const auto& qmd{kepler_compute->launch_description};
    const auto publish_collision_map = [&] {
        if (native_key != 0x6e0ae200ee9aa220ULL ||
            (qmd.const_buffer_enable_mask & 1) == 0) {
            return;
        }
        const auto& constants = qmd.const_buffer_config[0];
        if (constants.size < 0x3c) {
            return;
        }
        // WaterSurfaceCollisionHeightMap exposes its output handle in CB0.
        const u32 handle = gpu_memory->Read<u32>(constants.Address() + 0x38);
        const u32 index = Tegra::Texture::TexturePair(handle, qmd.linked_tsc != 0).first;
        texture_cache.PublishComputeCollisionImage(index);
    };
    if (frame_dump.CapturingBuffers()) {
        std::optional<std::array<u32, 3>> indirect_grid;
        if (indirect_address) {
            indirect_grid = std::array{gpu_memory->Read<u32>(*indirect_address),
                                       gpu_memory->Read<u32>(*indirect_address + 4),
                                       gpu_memory->Read<u32>(*indirect_address + 8)};
        }
        frame_dump.NoteLaunch(pipeline->ShaderHash(),
                              {qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z},
                              {qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z},
                              indirect_address, indirect_grid);
    }
    if (list_slot) {
        // The grid is GPU-written list words (Nx2ListIndirect): the host GPU copied them into the
        // slot at the list's place, and dispatches from there.
        auto& slots = buffer_cache_runtime.Nx2Slots();
        if (!slots.Held(*list_slot)) {
            LOG_ERROR(Render_Vulkan, "Nx2 list slot {} dispatched without a hold", *list_slot);
            return;
        }
        const auto [args_buffer, args_offset] = slots.DispatchArguments(*list_slot);
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.NoteComputeWrites();
        scheduler.TraceComputePipeline(pipeline->ShaderHash());
        scheduler.Record([this, pipeline, byte_addressed_shadow, indirect_buffer = args_buffer,
                          indirect_offset = args_offset](vk::CommandBuffer cmdbuf) {
            RecordDispatchBarrier(cmdbuf);
            if (!pipeline->IsBound()) {
                return;
            }
            if (byte_addressed_shadow) {
                TextureCacheRuntime::CopyByteAddressedShadow(cmdbuf, *byte_addressed_shadow, false);
            }
            scheduler.TraceGpuWork(cmdbuf, pipeline->ShaderHash(), 0, {}, [&] {
                cmdbuf.DispatchIndirect(indirect_buffer, indirect_offset);
            });
            if (byte_addressed_shadow) {
                TextureCacheRuntime::CopyByteAddressedShadow(cmdbuf, *byte_addressed_shadow, true);
            }
        });
        Tegra::Nx2ListIndirect::Stats::Count(Tegra::Nx2ListIndirect::Stats::Get().slot_dispatches);
        if (Tegra::Nx2ListIndirect::Stats::g_check) {
            slots.CheckDispatch(*list_slot);
        }
        {
            VideoCommon::Nx2Profile::Scope scope{VideoCommon::Nx2Profile::Zone::ComputeDownloads};
            buffer_cache.NoteWorkRecorded(true);
            SubmitNx2ListCopies();
        }
        publish_collision_map();
        TryOcarinaUpscale(*pipeline, native_key, false);
        return;
    }
    if (indirect_address) {
        // DispatchIndirect
        const auto [buffer, offset] =
            VideoCommon::ObtainIndirectBuffer(buffer_cache, *indirect_address);
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.NoteComputeWrites();
        scheduler.TraceComputePipeline(pipeline->ShaderHash());
        scheduler.Record([this, pipeline, indirect_buffer = buffer->Handle(),
                          indirect_offset = offset](vk::CommandBuffer cmdbuf) {
            RecordDispatchBarrier(cmdbuf);
            if (!pipeline->IsBound()) {
                return;
            }
            scheduler.TraceGpuWork(cmdbuf, pipeline->ShaderHash(), 0, {}, [&] {
                cmdbuf.DispatchIndirect(indirect_buffer, indirect_offset);
            });
        });
        {
            VideoCommon::Nx2Profile::Scope scope{VideoCommon::Nx2Profile::Zone::ComputeDownloads};
            buffer_cache.NoteWorkRecorded(true);
            SubmitNx2ListCopies();
        }
        publish_collision_map();
        return;
    }
    const std::array<u32, 3> dim{qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z};
    const std::array<u32, 3> max_dim{device.GetMaxComputeWorkGroupCount()};
    if (dim[0] > max_dim[0] || dim[1] > max_dim[1] || dim[2] > max_dim[2]) {
        return;
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.NoteComputeWrites();
    const bool barrier_recorded = can_replace || marking_effects;
    if (barrier_recorded) {
        // Effects snapshots and temporal upscaling can record work before the native dispatch.
        scheduler.Record([](vk::CommandBuffer cmdbuf) { RecordDispatchBarrier(cmdbuf); });
    }
    if (marking_effects) {
        const auto written = pipeline->WrittenImages();
        if (std::ranges::any_of(written, [this](VideoCommon::ImageViewId id) {
                return id && texture_cache.GetImageView(id).gpu_addr == oot_effects_address;
            })) {
            oot_live_dlss->MarkEffects(oot_effects_scene, render_trace_frame);
            oot_effects_frame.reset();
        }
    }
    const auto* const replacement =
        can_replace && !byte_addressed_shadow && TryOcarinaUpscale(*pipeline, native_key, true)
            ? oot_live_dlss.get()
            : nullptr;
    if (!replacement) {
        scheduler.TraceComputePipeline(pipeline->ShaderHash());
    }
    scheduler.Record([this, pipeline, dim, byte_addressed_shadow, bind = std::move(deferred_bind),
                      replacement, barrier_recorded](vk::CommandBuffer cmdbuf) {
        if (!barrier_recorded) {
            RecordDispatchBarrier(cmdbuf);
        }
        if (replacement && !replacement->EvaluationFailed()) {
            return;
        }
        if (replacement) {
            LOG_WARNING(Render_Vulkan, "Ocarina temporal evaluation failed; dispatching native upscaler");
        }
        if (bind) {
            bind(cmdbuf);
        }
        if (!pipeline->IsBound()) {
            return;
        }
        if (byte_addressed_shadow) {
            TextureCacheRuntime::CopyByteAddressedShadow(cmdbuf, *byte_addressed_shadow, false);
        }
        scheduler.TraceGpuWork(cmdbuf, pipeline->ShaderHash(), 0, dim,
                               [&] { cmdbuf.Dispatch(dim[0], dim[1], dim[2]); });
        if (byte_addressed_shadow) {
            TextureCacheRuntime::CopyByteAddressedShadow(cmdbuf, *byte_addressed_shadow, true);
        }
    });
    if (can_replace || marking_effects) {
        // The deferred native binding may follow DLSS's command buffer bindings.
        scheduler.InvalidateState();
    }
    {
        VideoCommon::Nx2Profile::Scope scope{VideoCommon::Nx2Profile::Zone::ComputeDownloads};
        buffer_cache.NoteWorkRecorded(true);
        SubmitNx2ListCopies();
    }
    publish_collision_map();

    if (!can_replace) {
        TryOcarinaUpscale(*pipeline, native_key, false);
    }

    // Log compute dispatch
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string params = fmt::format("groupCountX={}, groupCountY={}, groupCountZ={}",
            dim[0], dim[1], dim[2]);
        GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
            "vkCmdDispatch", params, VK_SUCCESS);
    }
}

void RasterizerVulkan::ResetCounter(VideoCommon::QueryType type) {
    // EDEN_NX2_RENDER_POST_QUERY: a Z-pass reset only changes GPURender's samples streamer and
    // queues its accumulator reset behind it. The streaming counters read live registers.
    if (type == VideoCommon::QueryType::ZPassPixelCount64 &&
        PostRenderOp(VideoCommon::GpuSplit::PostedOp::QueryReset, [this] {
            query_cache.CounterReset(VideoCommon::QueryType::ZPassPixelCount64);
        })) {
        return;
    }
    RENDER_LEND(ResetCounter(type))
    switch (type) {
    case VideoCommon::QueryType::ZPassPixelCount64:
    case VideoCommon::QueryType::StreamingByteCount:
    case VideoCommon::QueryType::StreamingPrimitivesSucceeded:
    case VideoCommon::QueryType::VtgPrimitivesOut:
        query_cache.CounterReset(type);
        return;
    default:
        LOG_DEBUG(Render_Vulkan, "Unimplemented counter reset={}", type);
        return;
    }
}

void RasterizerVulkan::RecordPayloadSyncs(std::span<const PayloadSync> syncs) {
    boost::container::small_vector<std::pair<Buffer*, u32>, 4> targets;
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.BufferOperations([&] {
        targets.clear();
        for (const PayloadSync& sync : syncs) {
            targets.push_back(
                buffer_cache.ObtainCPUBuffer(sync.address, static_cast<u32>(sync.size),
                                             VideoCommon::ObtainBufferSynchronize::FullSynchronize,
                                             VideoCommon::ObtainBufferOperation::DoNothing));
        }
    });
    u64 total_size = 0;
    for (const PayloadSync& sync : syncs) {
        total_size += sync.size;
    }
    const auto staging = buffer_cache_runtime.UploadStagingBuffer(total_size);
    buffer_cache_runtime.PreCopyBarrier();
    u64 staged = 0;
    for (size_t i = 0; i < syncs.size(); ++i) {
        const auto [buffer, offset] = targets[i];
        if (buffer == nullptr) {
            continue;
        }
        std::memcpy(staging.mapped_span.data() + staged, &syncs[i].value, syncs[i].size);
        const std::array copies{VideoCommon::BufferCopy{
            .src_offset = staging.offset + staged,
            .dst_offset = offset,
            .size = syncs[i].size,
        }};
        buffer_cache_runtime.CopyBuffer(buffer->Handle(), staging.buffer, copies, false, false);
        staged += syncs[i].size;
    }
    buffer_cache_runtime.PostCopyBarrier();
    // The copies are GPU writes the buffer cache doesn't record (as SyncValues).
    for (const PayloadSync& sync : syncs) {
        buffer_cache.OutdateReadDownloads(sync.address, sync.size);
    }
}

bool RasterizerVulkan::PostFenceReport(GPUVAddr gpu_addr, VideoCommon::QueryPropertiesFlags flags,
                                       u32 payload, u32 subreport) {
    using VideoCommon::GpuSplit;
    if (!GpuSplit::Get().Posts(GpuSplit::PostedOp::FenceReport)) {
        return false;
    }
    const auto cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!cpu_addr || !FencePostable(GpuSplit::PostedOp::FenceReport)) {
        return false;
    }
    // As the posted Z-pass reports: the buffer note, page mark and pending count at the report's
    // place. Later writes to the page see the mark and invalidate the query behind it on
    // GPURender, and query flushes wait for it to land.
    const bool wide = True(flags & VideoCommon::QueryPropertiesFlags::HasTimeout);
    const QueryCache::PostedReport posted{
        .device = *cpu_addr,
        .noted = NoteReportPending(*cpu_addr, wide ? 16 : 4),
        .by_value = true,
    };
    query_cache.page_filter.Mark(*cpu_addr, 16);
    if (GpuSplit::TracksPendingReports()) {
        query_cache.NotePendingReport(*cpu_addr);
    }
    VideoCommon::Nx2FastDraw::NoteSync();
    // Its own fence's NotifyWFI synced the value to the host: PostCheckedFence collects that
    // sync and GPURender plays it before the fence, so nothing is owed after it.
    back_payload_sync.push_back({*cpu_addr, payload, wide ? u64{8} : u64{4}});
    PostCheckedFence(GpuSplit::PostedOp::FenceReport,
                     [this, gpu_addr, flags, payload, subreport, posted] {
                         query_cache.CounterReport(gpu_addr, VideoCommon::QueryType::Payload, flags,
                                                   payload, subreport, &posted);
                     });
    return true;
}

void RasterizerVulkan::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                             VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    if (type == VideoCommon::QueryType::Payload &&
        True(flags & VideoCommon::QueryPropertiesFlags::IsAFence) &&
        PostFenceReport(gpu_addr, flags, payload, subreport)) {
        return;
    }
    if (VideoCommon::GpuSplit::LendsToRender()) {
        fence_query_owed = true;
    }
    using VideoCommon::GpuSplit;
    if (GpuSplit::PayloadsOnBack() && type == VideoCommon::QueryType::Payload &&
        False(flags & VideoCommon::QueryPropertiesFlags::IsAFence) && !Settings::IsGPULevelHigh()) {
        // EDEN_NX2_RENDER_PAYLOAD_E: such a release registers no query and writes at once, so the
        // back writes it at its place in the stream. Its later writes follow on this thread; an
        // unplayed draw reading the pools there is waited for. The host sync waits in
        // back_payload_sync for the next lent call, and fence_query_owed lends the next sync.
        const auto cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
        if (!cpu_addr) {
            return;
        }
        const bool wide = True(flags & VideoCommon::QueryPropertiesFlags::HasTimeout);
        const bool mirrored = True(flags & VideoCommon::QueryPropertiesFlags::Nx2Mirror);
        const u64 size = wide ? 16 : 4;
        auto& split = GpuSplit::Get();
        split.GuardStreamWrite(gpu_addr, size);
        split.StampWrite(*cpu_addr, size, GpuSplit::Writer::Payload);
        VideoCommon::Nx2FastDraw::NoteSync();
        // A read download still to land there is finished on GPURender.
        const bool lent = VideoCore::ReadDownloadsMayCover(*cpu_addr, size);
        split.NotePayloadOnBack(lent);
        if (lent) {
            split.LendRender(
                [&] { query_cache.WriteImmediate(gpu_addr, *cpu_addr, payload, wide, mirrored); });
        } else {
            query_cache.WriteImmediate(gpu_addr, *cpu_addr, payload, wide, mirrored);
        }
        back_payload_sync.push_back({*cpu_addr, payload, wide ? u64{8} : u64{4}});
        return;
    }
    if (type == VideoCommon::QueryType::ZPassPixelCount64 &&
        False(flags & VideoCommon::QueryPropertiesFlags::IsAFence) &&
        GpuSplit::Get().Posts(GpuSplit::PostedOp::QueryReport)) {
        // EDEN_NX2_RENDER_POST_QUERY: the back takes the report's buffer cache note and marks
        // its page at the report's place in the stream, so its later invalidations and flushes
        // there reach GPURender behind it. GPURender records and registers the report.
        if (const auto cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr)) {
            const bool wide = True(flags & VideoCommon::QueryPropertiesFlags::HasTimeout);
            const QueryCache::PostedReport posted{
                .device = *cpu_addr,
                .noted = NoteReportPending(*cpu_addr, wide ? 16 : 4),
            };
            query_cache.page_filter.Mark(*cpu_addr, 16);
            if (GpuSplit::TracksPendingReports()) {
                query_cache.NotePendingReport(*cpu_addr);
            }
            VideoCommon::Nx2FastDraw::NoteSync();
            if (PostRenderOp(GpuSplit::PostedOp::QueryReport, [this, gpu_addr, type, flags, payload,
                                                               subreport, posted] {
                    query_cache.CounterReport(gpu_addr, type, flags, payload, subreport, &posted);
                })) {
                GpuSplit::Get().MarkGuestWrite();
                if (GpuSplit::QuerySyncOnBack()) {
                    samples_slots.push_back(*cpu_addr);
                }
                return;
            }
            if (posted.noted) {
                FinishReport(posted.device, wide ? 16 : 4);
            }
        }
    }
    if (GpuSplit::QuerySyncOnBack()) {
        query_sync_lent_only = true;
    }
    RENDER_LEND(Query(gpu_addr, type, flags, payload, subreport))
    // Reports and semaphore releases reach guest memory the CPU may wait on.
    VideoCommon::Nx2FastDraw::NoteSync();
    query_cache.CounterReport(gpu_addr, type, flags, payload, subreport);
}

void RasterizerVulkan::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                 u32 size) {
    buffer_cache.BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
}

void Vulkan::RasterizerVulkan::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    buffer_cache.DisableGraphicsUniformBuffer(stage, index);
}

void RasterizerVulkan::FlushAll() {}

void RasterizerVulkan::SubmitNx2ListCopies() {
    // Batched like fences (DeferFlush): one submission per dispatch was ~6,000 a second in Hyrule
    // Warriors: Age of Imprisonment fights. A front that waits on a copy whose tick isn't
    // submitted yet has the back submit it between entries (IsGpuTickDone, WaitSubmittedTick).
    if (buffer_cache.Nx2CopyListPages(scheduler.CurrentTick()) && !scheduler.DeferFlush()) {
        scheduler.Flush();
    }
}

VideoCore::RasterizerInterface::GpuWrittenList RasterizerVulkan::ReadGpuWrittenCommandList(
    DAddr addr, std::span<u8> out, bool past_sync, u64* pending_tick) {
    // No cache lock: the front only reads the copies, under their own lock.
    switch (buffer_cache.Nx2ReadListCopy(addr, out, past_sync, pending_tick)) {
    case BufferCache::Nx2ListRead::Copied:
        return GpuWrittenList::Copied;
    case BufferCache::Nx2ListRead::Synced:
        return GpuWrittenList::Synced;
    case BufferCache::Nx2ListRead::NeedsSync:
        return GpuWrittenList::NeedsSync;
    case BufferCache::Nx2ListRead::Pending:
        return GpuWrittenList::Pending;
    default:
        return GpuWrittenList::Unknown;
    }
}

bool RasterizerVulkan::IsGpuTickDone(u64 tick) {
    if (scheduler.IsFree(tick)) {
        return true;
    }
    // A deferred submission (Scheduler::DeferFlush) is only made by the split's back: ask it to
    // submit, or the tick never completes while the front polls.
    if (!scheduler.IsSubmitted(tick) && VideoCommon::GpuSplit::Get().Active()) {
        VideoCommon::GpuSplit::Get().RequestIdleHook();
    }
    scheduler.GetMasterSemaphore().Refresh();
    return scheduler.IsFree(tick);
}

void RasterizerVulkan::NoteGpuWrittenCommandListSynced(DAddr page) {
    buffer_cache.Nx2NoteListSynced(page);
}

bool RasterizerVulkan::Nx2ListSlotsSupported() const {
    // A direct dispatch skips a grid over the device's limits; an indirect one must not get one.
    // The slot's height and depth are 16 bits, and a width over 31 bits is no culling result.
    const std::array<u32, 3> max_dim{device.GetMaxComputeWorkGroupCount()};
    const bool fits = max_dim[0] >= 0x7fffffffU && max_dim[1] >= 0xffffU && max_dim[2] >= 0xffffU;
    if (static std::atomic<bool> logged{}; !fits && !logged.exchange(true)) {
        LOG_WARNING(Render_Vulkan,
                    "Nx2 list indirect off: the device dispatches at most {}x{}x{} groups",
                    max_dim[0], max_dim[1], max_dim[2]);
    }
    return fits;
}

std::optional<u32> RasterizerVulkan::Nx2AcquireListSlot() {
    return buffer_cache_runtime.Nx2Slots().Acquire();
}

void RasterizerVulkan::Nx2ReleaseListSlot(u32 slot) {
    buffer_cache_runtime.Nx2Slots().Release(slot);
}

bool RasterizerVulkan::ListWordsOnBack() {
    // Lent until the slots' buffers exist (their fill records on the scheduler), and with the
    // list indirect check, which reads them back.
    auto& split = VideoCommon::GpuSplit::Get();
    if (!split.Posts(VideoCommon::GpuSplit::PostedOp::ListWords) || maxwell3d == nullptr ||
        BufferArena::Collecting()) {
        return false;
    }
    if (!buffer_cache_runtime.Nx2Slots().Ready() || Tegra::Nx2ListIndirect::Stats::g_check) {
        split.NoteListWordsLent();
        return false;
    }
    return true;
}

void RasterizerVulkan::PostListWords(bool snapshot, u64 noted) {
    if (list_word_commands.Empty()) {
        return;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    VideoCommon::CommandArena commands;
    commands.Swap(list_word_commands);
    const u64 stamp = buffer_cache_runtime.Arena().CloseCollected();
    if (noted != 0 && noted != stamp) {
        split.NoteOwedStampMoved();
    }
    split.NoteListWords(snapshot);
    PostRenderOp(VideoCommon::GpuSplit::PostedOp::ListWords,
                 [this, commands = std::move(commands), stamp] {
                     buffer_cache_runtime.Arena().PlayCollected(commands, stamp);
                 });
}

void RasterizerVulkan::Nx2SnapshotListWords(DAddr device_addr, u32 count, u32 slot) {
    // EDEN_NX2_RENDER_LIST_WORDS_E: the copy goes in a posted op, played at this place.
    const bool on_back = ListWordsOnBack();
    if (!on_back) {
        RENDER_LEND(Nx2SnapshotListWords(device_addr, count, slot))
    }
    // The words as GPU work sent before the list left them: the buffer cache's, its CPU writes
    // applied (as a dispatch reading them finds them).
    gpu_memory->FlushCaching();
    {
        std::optional<BufferArena::Collect> collect;
        if (on_back) {
            collect.emplace(list_word_commands);
        }
        std::scoped_lock lock{buffer_cache.mutex};
        const u32 bytes = count * static_cast<u32>(sizeof(u32));
        const auto [buffer, offset] = buffer_cache.ObtainCPUBuffer(
            device_addr, bytes, VideoCommon::ObtainBufferSynchronize::FullSynchronize,
            VideoCommon::ObtainBufferOperation::DoNothing);
        // As ObtainIndirectBuffer: a later upload of the words must not be reordered ahead of the
        // copy.
        buffer->MarkUsage(offset, bytes);
        buffer_cache_runtime.Nx2Slots().Snapshot(buffer->Handle(), offset, count, slot);
    }
    if (on_back) {
        PostListWords(true, 0);
    }
}

void RasterizerVulkan::Nx2NoteOwedListWord(DAddr device_addr, u32 value) {
    // EDEN_NX2_RENDER_LIST_WORDS_E: the slot word's copy to its host copy goes in a posted op,
    // and the word stays owed until the tick GPURender plays the op in completes.
    const bool on_back = ListWordsOnBack();
    if (!on_back) {
        RENDER_LEND(Nx2NoteOwedListWord(device_addr, value))
    }
    auto& slots = buffer_cache_runtime.Nx2Slots();
    slots.LandDone(std::bind_front(&RasterizerVulkan::Nx2LandOwedWord, this));
    bool owed;
    u64 noted = 0;
    {
        std::optional<BufferArena::Collect> collect;
        if (on_back) {
            collect.emplace(list_word_commands);
            noted = buffer_cache_runtime.Arena().Now();
        }
        owed = slots.NoteOwed(device_addr, value, device_memory.Read<u32>(device_addr));
    }
    if (owed) {
        // A guest CPU read of the word before it lands takes the trap, and GetFlushArea lands it.
        device_memory.ArmReadTraps(device_addr, sizeof(u32));
    }
    if (on_back) {
        PostListWords(false, noted);
    }
}

bool RasterizerVulkan::Nx2LandOwedWord(DAddr device_addr, u32 value, u32 real) {
    VideoCommon::GpuSplit::CheckParked(VideoCommon::GpuSplit::Unparked::OwedLanding);
    if (!Tegra::Nx2ListIndirect::LandOwedWord(device_memory, device_addr, value, real)) {
        return false;
    }
    // The caches learn of it as of the inline upload that wrote the placeholder. On the back
    // (EDEN_NX2_RENDER_LIST_WORDS_E) as a draw's cached writes: the buffer half there, the rest
    // lent or posted.
    const std::array<std::pair<DAddr, std::size_t>, 1> written{{{device_addr, sizeof(u32)}}};
    InnerInvalidation(written);
    return true;
}

void RasterizerVulkan::Nx2NoteInlineQmdGrid(DAddr device_addr, u32 x, u32 y, u32 z) {
    buffer_cache_runtime.Nx2Slots().NoteInlineQmdGrid(device_addr, x, y, z);
}

void RasterizerVulkan::Nx2ReadListSlots(std::span<const u32> slots, std::span<u32> words) {
    buffer_cache_runtime.Nx2Slots().Read(slots, words);
}

void RasterizerVulkan::Nx2CheckListSlot(u32 slot, std::span<const u32> words) {
    buffer_cache_runtime.Nx2Slots().CheckSnapshot(slot, words);
}

void RasterizerVulkan::FlushGpuWrittenCommandListPages(std::span<const DAddr> pages) {
    RENDER_LEND(FlushGpuWrittenCommandListPages(pages))
    gpu.InvalidateGPUCache();
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.Nx2FlushListPages(pages);
}

void RasterizerVulkan::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    using VideoCommon::CacheType;
    using VideoCommon::GpuSplit;
    if (GpuSplit::FlushesOnBack()) {
        // EDEN_NX2_RENDER_FLUSH_E: the back gathers CPU writes itself, as before a posted fence,
        // and lends only a flush with work to do. Images: GPURender played every packet (read
        // before the relaxed filter, which it only ever sets) and none was registered there.
        // Buffers: the back's own cache. Reports: as FlushQueryResults.
        auto& split = GpuSplit::Get();
        bool clean = true;
        if (addr != 0 && size != 0) {
            if (True(which & (CacheType::TextureCache | CacheType::BufferCache))) {
                gpu.InvalidateGPUCache();
            }
            if (True(which & CacheType::TextureCache)) {
                clean = !split.Guarding() && !texture_cache.page_filter.Any(addr, size);
            }
            if (clean && True(which & CacheType::BufferCache)) {
                clean = !buffer_cache_runtime.Nx2Slots().Owes(addr, size);
                if (clean) {
                    std::scoped_lock lock{buffer_cache.mutex};
                    clean = buffer_cache.DownloadsNothing(addr, size);
                }
            }
            if (clean && True(which & CacheType::QueryCache)) {
                clean = !query_cache.page_filter.Any(addr, size);
            }
        }
        split.NoteFlushOnBack(GpuSplit::FlushSite::Flush, clean);
        if (clean) {
            VerifyBackFlushAnswer(addr, size, which);
            return;
        }
    }
    RENDER_LEND(FlushRegion(addr, size, which))
    if (addr == 0 || size == 0) {
        return;
    }
    if (VideoCommon::GpuSplit::PlayingUnlent() && VideoCommon::GpuSplit::UnlendEnabled()) {
        // GPURender reading guest memory for an unlent draw (image uploads) stays out of the
        // back's buffer cache and CPU write gather (Fritchii ba647008e).
        auto& split = VideoCommon::GpuSplit::Get();
        split.CheckRenderRead(addr, size);
        if (True(which & VideoCommon::CacheType::BufferCache)) {
            split.NoteSkippedBackFlush();
        }
        if (True(which & VideoCommon::CacheType::TextureCache)) {
            std::scoped_lock lock{texture_cache.mutex};
            texture_cache.DownloadMemory(addr, size);
        }
        if (True(which & VideoCommon::CacheType::QueryCache)) {
            query_cache.FlushRegion(addr, size);
        }
        return;
    }
    if (True(which & VideoCommon::CacheType::BufferCache) &&
        VideoCommon::GpuSplit::OwnsScheduler()) {
        // A reader that flushes first gets the GPU-written list words owed there (Nx2ListIndirect).
        buffer_cache_runtime.Nx2Slots().LandRange(
            addr, size, Tegra::Nx2ListIndirect::OwedReader::Flush,
            std::bind_front(&RasterizerVulkan::Nx2LandOwedWord, this));
    }
    if (True(which & (VideoCommon::CacheType::TextureCache | VideoCommon::CacheType::BufferCache))) {
        // CPU writes into GPU-written memory are only collected when they happen. Apply them
        // first, or the download writes the stale GPU copy over them: the Yoshi demo writes a
        // native program over memory a compute pass had bound as a written texel buffer, and its
        // title menu or background read as zeros once the shader cache's read flushed it.
        gpu.InvalidateGPUCache();
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.DownloadMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.DownloadMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::QueryCache))) {
        query_cache.FlushRegion(addr, size);
    }
}

void RasterizerVulkan::FlushRegionForRead(DAddr addr, u64 size) {
    RENDER_LEND(FlushRegionForRead(addr, size))
    if (addr == 0 || size == 0) {
        return;
    }
    // As FlushRegion, but the buffer cache doesn't wait for the host GPU here (Scheduler::Finish,
    // with its lock held): the reading thread does, in FinishRead. GPU-written list words owed
    // there land at once (Nx2ListIndirect, GetFlushArea), which waits here.
    buffer_cache_runtime.Nx2Slots().LandRange(
        addr, size, Tegra::Nx2ListIndirect::OwedReader::CpuRead,
        std::bind_front(&RasterizerVulkan::Nx2LandOwedWord, this));
    gpu.InvalidateGPUCache();
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.DownloadMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.DownloadMemoryForRead(addr, size);
    }
    query_cache.FlushRegion(addr, size);
}

void RasterizerVulkan::FinishRead(DAddr addr, u64 size) {
    buffer_cache.FinishReadDownloads(addr, size);
}

bool RasterizerVulkan::NoteReportPending(DAddr addr, u64 size) {
    // A report per draw (Ocarina of Time's occlusion counts), rarely on a page the guest reads
    // back: most skip the buffer cache's lock.
    if (!buffer_cache.MayBeReadBackPage(addr)) {
        return false;
    }
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.NoteReportPending(addr, size);
}

void RasterizerVulkan::FinishReport(DAddr addr, u64 size) {
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.FinishReport(addr, size);
}

bool RasterizerVulkan::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    using VideoCommon::CacheType;
    using VideoCommon::GpuSplit;
    if (GpuSplit::FlushesOnBack()) {
        // EDEN_NX2_RENDER_FLUSH_E: the buffer verdict is the back's own cache; an image verdict
        // is lent unless GPURender played every packet and no image was registered there.
        auto& split = GpuSplit::Get();
        if (True(which & CacheType::BufferCache)) {
            bool modified = buffer_cache_runtime.Nx2Slots().Owes(addr, size);
            if (!modified && (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) ||
                              Tegra::Nx2MayBeGpuWritten(addr, size))) {
                std::scoped_lock lock{buffer_cache.mutex};
                modified = buffer_cache.IsRegionGpuModified(addr, size);
            }
            if (modified) {
                split.NoteFlushOnBack(GpuSplit::FlushSite::MustFlush, true);
                return true;
            }
        }
        const bool answered = !Settings::IsGPULevelHigh() ||
                              False(which & CacheType::TextureCache) ||
                              (!split.Guarding() && !texture_cache.page_filter.Any(addr, size));
        split.NoteFlushOnBack(GpuSplit::FlushSite::MustFlush, answered);
        if (answered) {
            VerifyBackFlushAnswer(
                addr, size, Settings::IsGPULevelHigh() ? which : which & ~CacheType::TextureCache);
            return false;
        }
    }
    RENDER_LEND_RESULT(MustFlushRegion(addr, size, which))
    // GPU-written list words owed there land in guest memory at the flush (Nx2ListIndirect).
    if (True(which & VideoCommon::CacheType::BufferCache) &&
        buffer_cache_runtime.Nx2Slots().Owes(addr, size)) {
        return true;
    }
    // The split's front asks this before each macro DMA read, and the back holds the buffer
    // cache's lock through each draw: pages the GPU never wrote need no look under it.
    if (True(which & VideoCommon::CacheType::BufferCache) &&
        (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) ||
         Tegra::Nx2MayBeGpuWritten(addr, size))) {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.IsRegionGpuModified(addr, size)) {
            return true;
        }
    }
    if (!Settings::IsGPULevelHigh()) {
        return false;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        return texture_cache.IsRegionGpuModified(addr, size);
    }
    return false;
}

void RasterizerVulkan::VerifyBackFlushAnswer(DAddr addr, u64 size, VideoCommon::CacheType which) {
    using VideoCommon::CacheType;
    const u64 every = RenderVerifyEvery();
    if (every == 0 || ++back_flush_answers % every != 0) {
        return;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    bool modified = false;
    split.LendRender([&] {
        if (True(which & CacheType::TextureCache)) {
            std::scoped_lock lock{texture_cache.mutex};
            modified |= texture_cache.IsRegionGpuModified(addr, size);
        }
        if (True(which & CacheType::BufferCache)) {
            std::scoped_lock lock{buffer_cache.mutex};
            modified |= buffer_cache.IsRegionGpuModified(addr, size);
        }
    });
    split.NoteFlushVerified(modified);
}

bool RasterizerVulkan::HasPendingQueries(DAddr addr, u64 size) {
    RENDER_LEND_RESULT(HasPendingQueries(addr, size))
    return query_cache.IsRegionGpuModified(addr, size);
}

bool RasterizerVulkan::HoldsQueries(DAddr addr, u64 size) {
    RENDER_LEND_RESULT(HoldsQueries(addr, size))
    return query_cache.HoldsQueries(addr, size);
}

bool RasterizerVulkan::MayHoldImagesOrReports(DAddr addr, u64 size) {
    RENDER_LEND_RESULT(MayHoldImagesOrReports(addr, size))
    return texture_cache.page_filter.Any(addr, size) || query_cache.HoldsQueries(addr, size);
}

VideoCore::RasterizerDownloadArea RasterizerVulkan::GetFlushArea(DAddr addr, u64 size) {
    RENDER_LEND_RESULT(GetFlushArea(addr, size))
    // Guest cores ask this on reads of cached pages, and the back holds both caches' locks through
    // each draw. Pages no image was ever registered on, and Switch 2 pages the GPU never wrote,
    // get the clean area the caches would return without a look under their locks (after
    // Stephen Jayakar's EDEN_NX2_READ_FASTPATH, eb3553bcb8).
    const VideoCore::RasterizerDownloadArea owed_area{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = false,
    };
    // GPU-written list words owed on these pages land in guest memory first (FlushRegionForRead):
    // the core keeps the area it gets, so later reads in it would not ask again.
    if (buffer_cache_runtime.Nx2Slots().Owes(owed_area.start_address,
                                             owed_area.end_address - owed_area.start_address)) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.ForgetReadTraps(owed_area.start_address,
                                     owed_area.end_address - owed_area.start_address);
        return owed_area;
    }
    std::optional<VideoCore::RasterizerDownloadArea> texture_area;
    if (texture_cache.page_filter.Any(addr, size)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_area = texture_cache.GetFlushArea(addr, size);
    }
    if (texture_area) {
        // The core disarmed the read's pages before asking, and skips later reads in this area,
        // even through another device alias of those pages. The buffer cache arms only pages it
        // doesn't count as armed, so it must forget the whole area.
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.ForgetReadTraps(texture_area->start_address,
                                     texture_area->end_address - texture_area->start_address);
        return *texture_area;
    }
    if (buffer_cache.FlushesOnCpuRead() &&
        (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) ||
         Tegra::Nx2MayBeGpuWritten(addr, size))) {
        std::scoped_lock lock{buffer_cache.mutex};
        auto area = buffer_cache.GetFlushArea(addr, size);
        if (area) {
            return *area;
        }
    }
    VideoCore::RasterizerDownloadArea new_area{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
    return new_area;
}

void RasterizerVulkan::PrepareRegionWrite(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0 || False(which & VideoCommon::CacheType::BufferCache)) {
        return;
    }
    std::scoped_lock lock{buffer_cache.mutex};
    // MemoryManager landed the read downloads here just before (FinishRead) and writes guest memory
    // next. Landing a newer one would wait for the host GPU with the lock held, from any thread.
    buffer_cache.ClearReplacedDownload(addr, size);
}

void RasterizerVulkan::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    const std::array<std::pair<DAddr, u64>, 1> range{{{addr, size}}};
    const bool buffer_half_only =
        VideoCommon::GpuSplit::InvalidatesOnBack() &&
        BackTakesInvalidation(
            std::span<const std::pair<DAddr, u64>>{range}, VideoCommon::GpuSplit::InvalSite::Region,
            which & (VideoCommon::CacheType::TextureCache | VideoCommon::CacheType::QueryCache));
    if (!buffer_half_only) {
        RENDER_LEND(InvalidateRegion(addr, size, which))
    }
    if (addr == 0 || size == 0) {
        return;
    }
    if (!buffer_half_only && True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    if (buffer_half_only) {
        return;
    }
    if ((True(which & VideoCommon::CacheType::QueryCache))) {
        query_cache.InvalidateRegion(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::ShaderCache))) {
        pipeline_cache.InvalidateRegion(addr, size);
    }
}

template <typename Size>
bool RasterizerVulkan::BackTakesInvalidation(std::span<const std::pair<DAddr, Size>> ranges,
                                             VideoCommon::GpuSplit::InvalSite site,
                                             VideoCommon::CacheType rest, bool gpu_written) {
    using Site = VideoCommon::GpuSplit::InvalSite;
    using VideoCommon::CacheType;
    auto& split = VideoCommon::GpuSplit::Get();
    const auto valid = [](const auto& range) { return range.first != 0 && range.second != 0; };
    const auto hit = [&]() -> std::optional<Site> {
        for (const auto& range : ranges) {
            if (!valid(range)) {
                continue;
            }
            if (texture_cache.page_filter.Any(range.first, range.second)) {
                return Site::Texture;
            }
            if (True(rest & CacheType::QueryCache) &&
                query_cache.page_filter.Any(range.first, range.second)) {
                return Site::Query;
            }
        }
        return std::nullopt;
    };
    const auto lend = [&split](Site site) {
        split.NoteInvalidation(site);
        return false;
    };
    // Shaders are looked up on the back as well: their invalidation stays lent.
    if (std::ranges::any_of(ranges, [&](const auto& range) {
            return valid(range) && pipeline_cache.page_filter.Any(range.first, range.second);
        })) {
        return lend(Site::Shader);
    }
    // EDEN_NX2_RENDER_POST_INVAL: GPURender's half (images, queries) is posted at this place.
    const bool posts =
        split.Posts(VideoCommon::GpuSplit::PostedOp::Invalidation) && maxwell3d != nullptr;
    auto first = hit();
    if (first && !posts) {
        return lend(*first);
    }
    {
        // Off the scheduler's owner, BufferCache::WriteMemory leaves out the download of
        // GPU-written bytes on the pages; only the back changes these ranges while it is not
        // lending. It uploads the write into the buffer instead where it collects the commands.
        std::scoped_lock lock{buffer_cache.mutex};
        for (const auto& [addr, size] : ranges) {
            const DAddr begin = Common::AlignDown(addr, Core::DEVICE_PAGESIZE);
            const DAddr end = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE);
            if (size == 0 || !buffer_cache.IsRegionGpuModified(begin, end - begin)) {
                continue;
            }
            if (!gpu_written) {
                return lend(Site::GpuWritten);
            }
            if (!buffer_cache.InlinesPageWrite(addr, size)) {
                split.NoteGpuWrittenGather(false);
                return lend(Site::GpuWritten);
            }
        }
    }
    if (!first) {
        // GPURender registers images while the back goes on: stamp, fence, look again. Either
        // this sees its registration, or its upload sees the stamp (CheckRenderRead, must be 0).
        for (const auto& [addr, size] : ranges) {
            split.StampWrite(addr, size, VideoCommon::GpuSplit::Writer::Gather);
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        first = hit();
        if (first && !posts) {
            return lend(Site::Recheck);
        }
    }
    if (first) {
        boost::container::small_vector<std::pair<DAddr, u64>, 8> posted;
        for (const auto& range : ranges) {
            if (valid(range)) {
                posted.emplace_back(range.first, range.second);
            }
        }
        PostRenderOp(VideoCommon::GpuSplit::PostedOp::Invalidation,
                     [this, rest, posted = std::move(posted)] {
                         if (True(rest & CacheType::TextureCache)) {
                             std::scoped_lock lock{texture_cache.mutex};
                             for (const auto& [addr, size] : posted) {
                                 texture_cache.WriteMemory(addr, size);
                             }
                         }
                         if (True(rest & CacheType::QueryCache)) {
                             for (const auto& [addr, size] : posted) {
                                 query_cache.InvalidateRegion(addr, size);
                             }
                         }
                     });
    }
    split.NoteInvalidation(site);
    return true;
}

void RasterizerVulkan::InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) {
    const bool buffer_half_only =
        VideoCommon::GpuSplit::InvalidatesOnBack() &&
        BackTakesInvalidation(sequences, VideoCommon::GpuSplit::InvalSite::Draw,
                              VideoCommon::CacheType::TextureCache |
                                  VideoCommon::CacheType::QueryCache);
    if (!buffer_half_only) {
        RENDER_LEND(InnerInvalidation(sequences))
    }
    InvalidateCachedWrites(sequences, false, buffer_half_only);
}

void RasterizerVulkan::InvalidateConstantBufferData(
    std::span<const std::pair<DAddr, std::size_t>> sequences) {
    const bool buffer_half_only =
        VideoCommon::GpuSplit::InvalidatesOnBack() &&
        BackTakesInvalidation(sequences, VideoCommon::GpuSplit::InvalSite::Draw,
                              VideoCommon::CacheType::TextureCache |
                                  VideoCommon::CacheType::QueryCache);
    if (!buffer_half_only) {
        RENDER_LEND(InvalidateConstantBufferData(sequences))
    }
    InvalidateCachedWrites(sequences, true, buffer_half_only);
}

void RasterizerVulkan::InvalidateCachedWrites(
    std::span<const std::pair<DAddr, std::size_t>> sequences, bool constant_buffer_data,
    bool buffer_half_only) {
    // The GPU's own cached writes (constant buffer data, every draw): most land where no image,
    // query or shader was ever registered, which page_filter tells without the caches' locks.
    const auto any = [&](const VideoCommon::PageFilter& filter) {
        for (const auto& [addr, size] : sequences) {
            if (filter.Any(addr, size)) {
                return true;
            }
        }
        return false;
    };
    const bool texture_hit = !buffer_half_only && any(texture_cache.page_filter);
    // A draw the back wanted to unlend waits for these to reach GPURender's caches.
    if (VideoCommon::GpuSplit::OnBackThread() && VideoCommon::GpuSplit::UnlendEnabled()) {
        unlend_gather_hit = !buffer_half_only &&
                            (texture_hit || any(query_cache.page_filter) ||
                             any(pipeline_cache.page_filter));
    }
    // Even with skip_cpu_inner_invalidation: engines writing images in software (DMA pitch to
    // block linear copies of decoded video frames) reach the texture cache only through here.
    if (texture_hit) {
        std::scoped_lock lock{texture_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            texture_cache.WriteMemory(addr, size);
        }
    }
    if (Settings::SkipCpuInnerInvalidation()) {
        return;
    }
    // Constant buffer data only streamed uniform buffers read needs no buffer cache work: those
    // pages stay CPU-modified, and uniform keeps learn of the write from the draw's fast draw
    // delta (BufferCache::ConstantDataReachesBuffers).
    const auto reaches_buffers = [&](const std::pair<DAddr, std::size_t>& range) {
        return !constant_buffer_data ||
               buffer_cache.ConstantDataReachesBuffers(range.first, range.second);
    };
    if (std::ranges::any_of(sequences, reaches_buffers)) {
        std::scoped_lock lock{buffer_cache.mutex};
        boost::container::small_vector<std::pair<DAddr, std::size_t>, 16> applied;
        for (const auto& range : sequences) {
            if (reaches_buffers(range)) {
                applied.push_back(range);
            }
        }
        // As OnCacheInvalidations: out of the GPU-written ranges first.
        buffer_cache.ClaimCpuWrites(applied);
        for (const auto& [addr, size] : applied) {
            buffer_cache.WriteMemory(addr, size, true, true);
        }
    }
    if (buffer_half_only) {
        return;
    }
    if (any(query_cache.page_filter)) {
        for (const auto& [addr, size] : sequences) {
            query_cache.InvalidateRegion(addr, size);
        }
    }
    if (any(pipeline_cache.page_filter)) {
        pipeline_cache.InvalidateRegions(sequences);
    }
}

bool RasterizerVulkan::OnCPUWrite(DAddr addr, u64 size) {
    RENDER_LEND_RESULT(OnCPUWrite(addr, size))
    DEBUG_ASSERT(addr != 0 || size != 0);
    {
        // The pages the write untracks change their host protection after the lock is dropped:
        // drawing waited for those system calls (~1 µs each, thousands a second at Ocarina of
        // Time's water). A write trapping on a page still protected until then finds the page
        // already untracked and changes nothing. Off the scheduler's owner the write only
        // untracks pages (BufferCache::WriteMemory uploads only on the owner), which is all that
        // may wait. Device unmaps wait for these changes, as one applied to an unmapped page is
        // lost (BeginDeferredCachedUpdates).
        static thread_local VideoCommon::DeferredPageCaching deferred;
        const bool defer = VideoCommon::t_deferred_page_caching == nullptr &&
                           !VideoCommon::GpuSplit::OwnsScheduler();
        bool gpu_modified;
        {
            std::scoped_lock lock{buffer_cache.mutex};
            if (defer) {
                VideoCommon::t_deferred_page_caching = &deferred;
            }
            gpu_modified = buffer_cache.OnCPUWrite(addr, size);
            if (defer) {
                VideoCommon::t_deferred_page_caching = nullptr;
                if (!deferred.Empty()) {
                    device_memory.BeginDeferredCachedUpdates();
                }
            }
        }
        if (defer && !deferred.Empty()) {
            deferred.ApplyAll();
            device_memory.EndDeferredCachedUpdates();
        }
        if (gpu_modified) {
            return true;
        }
    }
    // Pages no image was ever registered on hold no image or stale page to update, as in
    // GetFlushArea, so the texture cache's lock (held by the back through each draw) is not
    // waited for there.
    if (texture_cache.page_filter.Any(addr, size)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return false;
}

void RasterizerVulkan::OnCacheInvalidations(std::span<const std::pair<DAddr, u64>> ranges) {
    const bool gpu_written = VideoCommon::GpuSplit::GathersGpuWrittenOnBack() &&
                             maxwell3d != nullptr && !BufferArena::Collecting();
    const bool buffer_half_only =
        VideoCommon::GpuSplit::InvalidatesOnBack() &&
        BackTakesInvalidation(ranges, VideoCommon::GpuSplit::InvalSite::Gather,
                              VideoCommon::CacheType::TextureCache, gpu_written);
    if (!buffer_half_only) {
        RENDER_LEND(OnCacheInvalidations(ranges))
    }
    // Each cache's lock is taken once for all the writes, not once a write (jpkto/Lichii
    // f5edeabdbb): guest CPU write traps hold the buffer cache's for long. The caches are
    // independent (TextureCache::WriteMemory reads and changes only texture state, and the page
    // counts both change commute), so each ends as with the writes applied one by one.
    const auto valid = [](const std::pair<DAddr, u64>& range) {
        return range.first != 0 && range.second != 0;
    };
    if (!buffer_half_only) {
        std::scoped_lock lock{texture_cache.mutex};
        for (const auto& range : ranges) {
            if (valid(range)) {
                texture_cache.WriteMemory(range.first, range.second);
            }
        }
    }
    {
        // EDEN_NX2_RENDER_GATHER_E: uploads into GPU-written buffer data go in a posted op.
        std::optional<BufferArena::Collect> collect;
        if (buffer_half_only && gpu_written) {
            collect.emplace(gather_commands);
        }
        std::scoped_lock lock{buffer_cache.mutex};
        // Out of the GPU-written ranges first: applying one write downloads the GPU's bytes
        // elsewhere on its pages (WriteMemory), which would write over the next collected write.
        buffer_cache.ClaimCpuWrites(ranges);
        for (const auto& range : ranges) {
            if (valid(range)) {
                buffer_cache.WriteMemory(range.first, range.second);
            }
        }
    }
    if (!gather_commands.Empty()) {
        VideoCommon::CommandArena commands;
        commands.Swap(gather_commands);
        const u64 stamp = buffer_cache_runtime.Arena().CloseCollected();
        VideoCommon::GpuSplit::Get().NoteGpuWrittenGather(true);
        PostRenderOp(VideoCommon::GpuSplit::PostedOp::Invalidation,
                     [this, commands = std::move(commands), stamp] {
                         buffer_cache_runtime.Arena().PlayCollected(commands, stamp);
                     });
    }
    if (buffer_half_only) {
        return;
    }
    for (const auto& range : ranges) {
        if (valid(range)) {
            pipeline_cache.InvalidateRegion(range.first, range.second);
        }
    }
}

void RasterizerVulkan::OnCacheInvalidation(DAddr addr, u64 size) {
    RENDER_LEND(OnCacheInvalidation(addr, size))
    if (addr == 0 || size == 0) {
        return;
    }

    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
}

bool RasterizerVulkan::TryCacheInvalidation(DAddr addr, u64 size) {
    RENDER_LEND_RESULT(TryCacheInvalidation(addr, size))
    if (addr == 0 || size == 0) return true;
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        // WriteMemory downloads GPU-written bytes elsewhere on the written pages, which only the
        // GPU thread may do: leave those pages to the fallback.
        const DAddr page_begin = Common::AlignDown(addr, Core::DEVICE_PAGESIZE);
        const DAddr page_end = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE);
        if (buffer_cache.IsRegionGpuModified(page_begin, page_end - page_begin)) return false;
        texture_cache.WriteMemory(addr, size);
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return true;
}

void RasterizerVulkan::InvalidateGPUCache() {
    if (posted_fence_playing) {
        return;
    }
    gpu.InvalidateGPUCache();
}

void RasterizerVulkan::UnmapMemory(DAddr addr, u64 size) {
    RENDER_LEND(UnmapMemory(addr, size))
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size, false);
    }
    pipeline_cache.OnCacheInvalidation(addr, size);
}

void RasterizerVulkan::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {
    RENDER_LEND(ModifyGPUMemory(as_id, addr, size))
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapGPUMemory(as_id, addr, size);
    }
    buffer_cache.UnmapGPUMemory(as_id, addr, size);
}

void RasterizerVulkan::SignalFence(std::function<void()>&& func) {
    // EDEN_NX2_RENDER_POST_SIGNAL_FENCE: as a posted sync point. The callback runs where it did,
    // on the fence's completion; only the fence is made on GPURender in stream order.
    if (FencePostable(VideoCommon::GpuSplit::PostedOp::SignalFence)) {
        PostCheckedFence(VideoCommon::GpuSplit::PostedOp::SignalFence,
                         [this, func = std::move(func)]() mutable {
                             VideoCommon::Nx2FastDraw::NoteSync();
                             fence_manager.SignalFence(std::move(func));
                         });
        return;
    }
    if (VideoCommon::GpuSplit::LendsToRender()) {
        NoteSyncLent();
    }
    RENDER_LEND(SignalFence(std::move(func)))
    VideoCommon::Nx2FastDraw::NoteSync();
    fence_manager.SignalFence(std::move(func));
}

void RasterizerVulkan::SyncOperation(std::function<void()>&& func) {
    RENDER_LEND(SyncOperation(std::move(func)))
    VideoCommon::Nx2FastDraw::NoteSync();
    fence_manager.SyncOperation(std::move(func));
}

void RasterizerVulkan::SignalSyncPoint(u32 value) {
    if (PostFence([this, value] {
            VideoCommon::Nx2FastDraw::NoteSync();
            fence_manager.SignalSyncPoint(value);
        })) {
        return;
    }
    if (VideoCommon::GpuSplit::LendsToRender()) {
        NoteSyncLent();
    }
    RENDER_LEND(SignalSyncPoint(value))
    VideoCommon::Nx2FastDraw::NoteSync();
    fence_manager.SignalSyncPoint(value);
}

void RasterizerVulkan::SignalReference() {
    if (PostFence([this] {
            VideoCommon::Nx2FastDraw::NoteSync();
            fence_manager.SignalReference();
        })) {
        return;
    }
    if (VideoCommon::GpuSplit::LendsToRender()) {
        NoteSyncLent();
    }
    RENDER_LEND(SignalReference())
    VideoCommon::Nx2FastDraw::NoteSync();
    fence_manager.SignalReference();
}

void RasterizerVulkan::ReleaseFences(bool force) {
    // EDEN_NX2_RENDER_SOFT_RELEASE_E: with the fencing thread an unforced release does nothing
    // (FenceManager::WaitPendingFences), so the back has nothing to lend (Fritchii 554125b40).
    static_assert(FenceManagerParams::HAS_ASYNC_CHECK);
    if (!force && VideoCommon::GpuSplit::SoftReleasesOnBack()) {
        VideoCommon::GpuSplit::Get().NoteSoftRelease();
        return;
    }
    // With immediate fences the forced release's wait ends inside SignalFence (Fritchii 440c984e7).
    if (force && PostFence([this] { fence_manager.WaitPendingFences(true); })) {
        return;
    }
    if (VideoCommon::GpuSplit::LendsToRender() && force) {
        NoteSyncLent();
    }
    RENDER_LEND(ReleaseFences(force))
    fence_manager.WaitPendingFences(force);
}

void RasterizerVulkan::FlushQueryResults(DAddr addr, u64 size) {
    using VideoCommon::GpuSplit;
    // EDEN_NX2_RENDER_POST_QUERY: SemiFlushRegion does nothing on pages no report was registered
    // on, and the back sees every mark: lent reports mark them with the back parked, posted ones
    // are marked by the back before they are posted. The filter is never cleared.
    if (GpuSplit::Get().Posts(GpuSplit::PostedOp::QueryReport) &&
        (addr == 0 || size == 0 || !query_cache.page_filter.Any(addr, size))) {
        GpuSplit::Get().NoteQueryFlushSkipped();
        return;
    }
    // EDEN_NX2_RENDER_QUERY_FLUSH_E: SemiFlushRegion only writes out values a report's own
    // operation writes when it lands, and waits for the reports still to land: none means the
    // range already holds every value. The back counts a report pending before it posts or lends
    // it; its operation uncounts it once the value is written.
    if (GpuSplit::QueryFlushesOnBack() && !query_cache.HasPendingReports(addr, size)) {
        GpuSplit::Get().NoteQueryFlushNothingPending();
        return;
    }
    RENDER_LEND(FlushQueryResults(addr, size))
    if (addr == 0 || size == 0 || !query_cache.SemiFlushRegion(addr, size)) {
        return;
    }
    // Below High GPU accuracy a fence runs its own callback at once, so wait for an operation
    // queued behind the pending report writes instead. On the split's back the fence's
    // submission may be deferred, and the fencing thread would wait for this thread to submit it.
    std::mutex wait_mutex;
    std::condition_variable wait_cv;
    bool written{};
    fence_manager.SyncOperation([&] {
        std::scoped_lock lock{wait_mutex};
        written = true;
        wait_cv.notify_all();
    });
    fence_manager.SignalFence([] {});
    scheduler.FlushDeferred();
    std::unique_lock lock{wait_mutex};
    wait_cv.wait(lock, [&written] { return written; });
}

void RasterizerVulkan::FlushAndInvalidateRegion(DAddr addr, u64 size,
                                                VideoCommon::CacheType which) {
    RENDER_LEND(FlushAndInvalidateRegion(addr, size, which))
    if (Settings::IsGPULevelHigh()) {
        FlushRegion(addr, size, which);
    }
    InvalidateRegion(addr, size, which);
}

void RasterizerVulkan::WaitForIdle() {
    // EDEN_NX2_RENDER_POST_WFI: with no report since the last lent fence (whose commit synced
    // the queries), no streamer has a sync pending and NotifyWFI leaves the back's buffer cache
    // alone: the rest is GPURender's recording.
    // EDEN_NX2_RENDER_QUERY_SYNC_E: also with reports whose value syncs the back collects here.
    using VideoCommon::GpuSplit;
    if (GpuSplit::Get().Posts(GpuSplit::PostedOp::WaitForIdle) && maxwell3d != nullptr &&
        !QuerySyncOwed()) {
        QuerySyncPacket sync;
        if (GpuSplit::QuerySyncOnBack()) {
            CollectQuerySync(sync);
        } else if (!back_payload_sync.empty()) [[unlikely]] {
            GpuSplit::Get().NotePayloadSyncsLeft();
        }
        PostRenderOp(GpuSplit::PostedOp::WaitForIdle, [this, sync = std::move(sync)]() mutable {
            if (!sync.commands.Empty()) {
                buffer_cache_runtime.Arena().PlayCollected(sync.commands, sync.stamp);
            }
            if (!GpuSplit::QuerySyncEnabled() && query_cache.HasPendingSync()) [[unlikely]] {
                GpuSplit::Get().NotePostedWaitSynced();
            }
            t_query_sync_targets = &sync.targets;
            SCOPE_EXIT {
                t_query_sync_targets = nullptr;
            };
            WaitForIdle();
        });
        return;
    }
    if (GpuSplit::QuerySyncOnBack()) {
        // Lent: NotifyWFI syncs everything pending, the back parked.
        query_sync_lent_only = false;
        samples_slots.clear();
    }
    RENDER_LEND(WaitForIdle())
    // Everything but wait pixel operations. This intentionally includes FRAGMENT_SHADER_BIT because
    // fragment shaders can still write storage buffers.
    VkPipelineStageFlags flags =
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
        VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT |
        VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (device.IsExtTransformFeedbackSupported()) {
        flags |= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
    }

    query_cache.NotifyWFI();

    if (device.IsMoltenVK() && scheduler.IsRenderPassActive()) {
        scheduler.DeferFragmentBarrier();
        fence_manager.SignalOrdering();
        return;
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([flags](vk::CommandBuffer cmdbuf) {
        // Later work waits for the earlier work and sees its writes.
        static constexpr VkMemoryBarrier WFI_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(flags, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, WFI_BARRIER);
    });
    fence_manager.SignalOrdering();
}

void RasterizerVulkan::FragmentBarrier() {
    // GPURender's scheduler alone, nothing back to the back (Fritchii f5afca4c9, fdd2b9472).
    if (PostRenderOp(VideoCommon::GpuSplit::PostedOp::FragmentBarrier,
                     [this] { FragmentBarrier(); })) {
        return;
    }
    RENDER_LEND(FragmentBarrier())
    // The next draw breaks the render pass only if it reads what this one wrote (#161): Ocarina
    // ended ~5,900 render passes a second here.
    scheduler.DeferFragmentBarrier();
}

void RasterizerVulkan::TiledCacheBarrier() {
    // TODO: Implementing tiled barriers requires rewriting a good chunk of the Vulkan backend
}

void RasterizerVulkan::FlushCommands() {
    // GPURender's own counters (draw_counter, pending_native_work) and scheduler, nothing back to
    // the back (Fritchii f5afca4c9).
    if (PostRenderOp(VideoCommon::GpuSplit::PostedOp::FlushCommands, [this] { FlushCommands(); })) {
        return;
    }
    RENDER_LEND(FlushCommands())
    if (draw_counter == 0 && !pending_native_work) {
        return;
    }
    draw_counter = 0;
    pending_native_work = false;
    // A fence with a download is submitted by InnerFence::Queue, and SignalFence then comes here
    // with nothing recorded since: ~4,600 empty submits a second in Ocarina of Time's Castle
    // Town, each followed by a full rebind (from Fritchii c338269b92).
    if (!scheduler.RecordedSinceSubmit()) {
        Scheduler::CountSkippedFlush();
        return;
    }
    // Fences with downloads or queries come here right after queueing: ~9,400 a second in
    // Hyrule Warriors: Age of Imprisonment fights. Batched like the fences themselves.
    if (!scheduler.DeferFlush()) {
        scheduler.Flush();
    }
}

void RasterizerVulkan::TraceRenderState(bool is_indexed, const char* outcome,
                                        const Framebuffer* framebuffer) {
    if (!VideoCommon::RenderTrace::SampleFrame(render_trace_frame)) {
        return;
    }
    if (render_trace_draws++ >= 2048) {
        if (render_trace_draws == 2049) {
            LOG_DEBUG(Render_Vulkan, "Render trace frame={} truncated after 2048 draws",
                      render_trace_frame);
        }
        return;
    }
    const auto& regs = maxwell3d->regs;
    const auto& draw = maxwell3d->draw_manager.draw_state;
    std::array<u64, 6> shaders{};
    u32 stages{};
    for (size_t stage = 0; stage < shaders.size(); ++stage) {
        shaders[stage] = (u64{regs.pipelines[stage].reserved[0] & 0xff} << 32) |
                         regs.pipelines[stage].reserved[1];
        stages |= static_cast<u32>(regs.IsShaderConfigEnabled(stage)) << stage;
    }
    const auto extent = framebuffer ? framebuffer->RenderArea() : VkExtent2D{};
    LOG_DEBUG(Render_Vulkan,
              "Render trace frame={} draw={} docked={} outcome={} indexed={} stages={:#x} "
              "shaders={:x} guest_count={} guest_instances={} framebuffer={}x{} layers={}",
              render_trace_frame, render_trace_draws, Settings::IsDockedMode(), outcome, is_indexed,
              stages, fmt::join(shaders, ","),
              is_indexed ? draw.index_buffer.count : draw.vertex_buffer.count, draw.instance_count,
              extent.width, extent.height, framebuffer ? framebuffer->Layers() : 0);
    LOG_DEBUG(Render_Vulkan,
              "Render trace targets: color={:#x} {}x{} layers={} format={:#x} depth={:#x} "
              "{}x{} layers={} format={:#x} enabled={} clip={}x{}",
              regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height, regs.rt[0].depth.Value(),
              static_cast<u32>(regs.rt[0].format), regs.zeta.Address(), regs.zeta_size.width,
              regs.zeta_size.height, regs.zeta_size.depth.Value(), static_cast<u32>(regs.zeta.format),
              regs.zeta_enable, regs.surface_clip.width.Value(), regs.surface_clip.height.Value());
    const auto& viewport = regs.viewport_transform[0];
    const auto& scissor = regs.scissor_test[0];
    LOG_DEBUG(Render_Vulkan,
              "Render trace tests: raster={} depth_test={} depth_write={} depth_func={} "
              "cull={} face={} alpha={} func={} ref={} color_mask={:#x} "
              "viewport_scale=({},{},{}) translate=({},{},{}) scissor={} {}..{} {}..{}",
              regs.rasterize_enable, regs.depth_test_enable, regs.depth_write_enabled,
              static_cast<u32>(regs.depth_test_func), regs.gl_cull_test_enabled,
              static_cast<u32>(regs.gl_cull_face), regs.alpha_test_enabled,
              static_cast<u32>(regs.alpha_test_func), regs.alpha_test_ref, regs.color_mask[0].raw,
              viewport.scale_x, viewport.scale_y, viewport.scale_z, viewport.translate_x,
              viewport.translate_y, viewport.translate_z, scissor.enable, scissor.min_x.Value(),
              scissor.max_x.Value(), scissor.min_y.Value(), scissor.max_y.Value());
    for (size_t binding = 0; binding < regs.vertex_streams.size(); ++binding) {
        const auto& stream = regs.vertex_streams[binding];
        if (stream.IsEnabled()) {
            LOG_DEBUG(Render_Vulkan,
                      "Render trace vertex: binding={} address={:#x} limit={:#x} stride={} "
                      "frequency={} repacked={}",
                      binding, stream.Address(), regs.vertex_stream_limits[binding].Address(),
                      stream.stride.Value(), stream.frequency,
                      buffer_cache_runtime.IsVertexRepacked(static_cast<u32>(binding)));
        }
    }
    if (is_indexed) {
        LOG_DEBUG(Render_Vulkan, "Render trace index: address={:#x} limit={:#x} format={} first={}",
                  regs.index_buffer.StartAddress(), regs.index_buffer.EndAddress(),
                  static_cast<u32>(regs.index_buffer.format), draw.index_buffer.first);
    }
    LOG_DEBUG(Render_Vulkan, "Render trace condition: address={:#x} mode={} override={}",
              regs.render_enable.Address(), static_cast<u32>(regs.render_enable.mode),
              static_cast<u32>(regs.render_enable_override));
}

void RasterizerVulkan::TracePendingCpuWrites(bool compute, std::span<const u64> shader_hashes) {
    // Guest CPU writes into GPU-written buffer memory are collected, and applied to the host
    // buffers only at the next fence or cache invalidation. A hit here reads the older bytes.
    static const bool enabled = std::getenv("EDEN_NX2_PENDING_WRITE_TRACE") != nullptr;
    if (!enabled) {
        return;
    }
    static constexpr u64 max_lines = 100000;
    static u64 lines{};
    buffer_cache.ForEachBoundBuffer(compute, [&](u32 stage, bool storage, u32 index, bool written,
                                                 const auto& binding, u32 bound_size,
                                                 const auto*, u32) {
        if (lines > max_lines || !gpu.HasPendingCPUWrites(binding.device_addr, bound_size)) {
            return;
        }
        if (++lines > max_lines) {
            LOG_INFO(HW_GPU, "Pending write trace stopped after {} lines", max_lines);
            return;
        }
        const size_t program = compute ? 0 : stage + 1;
        LOG_INFO(HW_GPU,
                 "Pending CPU write frame={} {} shader={:016x} stage={} {}[{}]{} addr={:#x} "
                 "device={:#x} size={:#x}",
                 render_trace_frame, compute ? "dispatch" : "draw",
                 program < shader_hashes.size() ? shader_hashes[program] : 0, stage,
                 storage ? "ssbo" : "cbuf", index, written ? " written" : "", binding.gpu_addr,
                 binding.device_addr, bound_size);
    });
}

bool RasterizerVulkan::OcarinaTemporalEnabled() const {
    constexpr u64 ocarina_program = 0x0400486024DEC000ULL;
    if (program_id != ocarina_program || oot_live_failed) {
        return false;
    }
    if (oot_backend) {
        return *oot_backend != VideoCommon::OcarinaUpscaleBackend::Native;
    }
    oot_backend = ResolveOcarinaUpscaler(device);
    if (*oot_backend == VideoCommon::OcarinaUpscaleBackend::Native &&
        Settings::values.ocarina_fsr.GetValue() && !Fsr3Upscaler::IsSupported(device)) {
        LOG_WARNING(Render_Vulkan, "Ocarina FSR 3.1 unsupported on this device; using native upscale");
    }
    return *oot_backend != VideoCommon::OcarinaUpscaleBackend::Native;
}

void RasterizerVulkan::TrackOcarinaDraw(u64 shader_hash, bool is_indexed, bool is_indirect) {
    if (const auto* color = OcarinaCameraTarget()) {
        if (const auto camera = CaptureOcarinaCamera()) {
            ApplyOcarinaCamera(*color, *camera, gpu_memory);
        }
    }
    if (shader_hash == 0x28c6e527634f814aULL && is_indexed && !is_indirect &&
        oot_actor.frame != render_trace_frame) {
        if (const auto actor = CaptureOcarinaActor()) {
            ApplyOcarinaActor(*actor);
        }
    }
}

/// Capture the two camera matrices from a draw targeting the scene color and depth. Terrain and
/// materials draw with several vertex shaders, so any draw into a scene-sized target qualifies.
const ImageView* RasterizerVulkan::OcarinaCameraTarget() const {
    const auto& targets = texture_cache.GetRenderTargets();
    if (!targets.color_buffer_ids[0] || !targets.depth_buffer_id) {
        return nullptr;
    }
    const auto& color = texture_cache.GetImageView(targets.color_buffer_ids[0]);
    if (color.size.width < 640 || color.size.height < 360 ||
        color.format != VideoCore::Surface::PixelFormat::B10G11R11_FLOAT) {
        return nullptr;
    }
    // Reflections and half-size passes share the camera layout; the upscale reads the largest.
    if (oot_camera_sample_frame == render_trace_frame &&
        u64{color.size.width} * color.size.height <=
            u64{oot_motion.image_extent.width} * oot_motion.image_extent.height) {
        return nullptr;
    }
    return &color;
}

void RasterizerVulkan::ApplyOcarinaCamera(const ImageView& color, const OcarinaCameraFacts& facts,
                                          const Tegra::MemoryManager* channel) {
    const auto refuse = [&](const char* reason, f32 width, f32 height) {
        if (oot_camera_refusal && oot_camera_refusal->frame == render_trace_frame &&
            u64{color.size.width} * color.size.height <
                u64{oot_camera_refusal->target_width} * oot_camera_refusal->target_height) {
            return;
        }
        oot_camera_refusal = OcarinaCameraRefusal{render_trace_frame, reason, width, height,
                                                  color.size.width, color.size.height};
    };
    if (facts.gpu_written) {
        refuse("no scene camera: its buffer is GPU-written", 0.0f, 0.0f);
        return;
    }
    const auto& values = facts.values;
    const auto camera = VideoCommon::DecodeOcarinaCamera(
        values, color.size.width, color.size.height,
        oot_backend == VideoCommon::OcarinaUpscaleBackend::Dlss);
    if (camera.issue != VideoCommon::OcarinaCameraIssue::None) {
        refuse(camera.issue == VideoCommon::OcarinaCameraIssue::Extent
                   ? "no scene camera: its size does not match the scene target"
                   : "no scene camera: unexpected projection",
               values[0x120 / sizeof(f32)], values[0x124 / sizeof(f32)]);
        return;
    }
    oot_camera_sample_frame = render_trace_frame;
    oot_motion.camera_channel = channel;
    oot_motion.extent = {camera.width, camera.height};
    oot_motion.image_extent = {color.size.width, color.size.height};
    oot_motion.current = camera.current;
    oot_motion.previous = camera.previous;
}

void RasterizerVulkan::ApplyOcarinaActor(const OcarinaActorPose& pose) {
    oot_previous_actor = oot_actor;
    oot_actor = pose;
    oot_actor.frame = render_trace_frame;
}

void RasterizerVulkan::ApplyOcarinaFacts(const OcarinaDrawFacts& facts) {
    if (!OcarinaTemporalEnabled()) {
        return;
    }
    if (facts.actor && oot_actor.frame != render_trace_frame) {
        ApplyOcarinaActor(*facts.actor);
    }
    if (facts.camera) {
        if (const auto* color = OcarinaCameraTarget()) {
            ApplyOcarinaCamera(*color, *facts.camera, facts.channel);
        }
    }
}

void RasterizerVulkan::UpdateOcarinaJitter() {
    std::array<f32, 2> want{};
    // Only while FSR replaced the previous frame's upscale: a frame the game upscales itself
    // would show the offset as a shake. Only draws that depth-test against the scene depth at
    // the scene size, so a depth pre-pass and the colour passes move together; shadow maps,
    // half-size passes and full-screen passes (which keep the depth bound without using it, as
    // #678 found, some with depth writes on: with the test off nothing writes depth) stay put.
    const auto& regs = DrawEngine().regs;
    if (oot_backend == VideoCommon::OcarinaUpscaleBackend::Fsr && oot_motion.extent.width != 0 &&
        oot_dlss_frame && *oot_dlss_frame + 1 >= render_trace_frame && regs.depth_test_enable) {
        const auto& targets = texture_cache.GetRenderTargets();
        if (targets.depth_buffer_id) {
            const auto& depth = texture_cache.GetImageView(targets.depth_buffer_id);
            if (depth.size.width == oot_motion.extent.width &&
                depth.size.height == oot_motion.extent.height && !depth.IsRescaled()) {
                // AMD's phase count for the last FSR frame's scale.
                const u32 index = static_cast<u32>(render_trace_frame % oot_jitter_phases) + 1;
                const auto driver = device.GetDriverID();
                want = VideoCommon::OcarinaJitter(index, driver == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                                             driver == VK_DRIVER_ID_AMD_OPEN_SOURCE);
                oot_jitter_frame = render_trace_frame;
                oot_jitter_offset = want;
            }
        }
    }
    if (want != oot_jitter_applied) {
        oot_jitter_applied = want;
        DrawEngine().dirty.flags[Dirty::Viewports] = true;
    }
}

std::unique_ptr<RasterizerVulkan::OcarinaDrawFacts> RasterizerVulkan::CaptureOcarinaFacts(
    u64 shader_hash, bool is_indexed, bool is_indirect) {
    constexpr u64 ocarina_program = 0x0400486024DEC000ULL;
    if (program_id != ocarina_program) {
        return nullptr;
    }
    std::unique_ptr<OcarinaDrawFacts> facts;
    const auto take = [&]() -> OcarinaDrawFacts& {
        if (!facts) {
            // Default-initialized: the actor pose's bones (13 KiB) are written only when present.
            facts.reset(new OcarinaDrawFacts);
            facts->channel = gpu_memory;
        }
        return *facts;
    };
    const auto& regs = maxwell3d->regs;
    if (regs.rt[0].format == Tegra::RenderTargetFormat::B10G11R11_FLOAT && regs.zeta_enable != 0) {
        if (auto camera = CaptureOcarinaCamera()) {
            take().camera = std::move(*camera);
        }
    }
    if (shader_hash == 0x28c6e527634f814aULL && is_indexed && !is_indirect) {
        if (auto actor = CaptureOcarinaActor()) {
            take().actor = std::move(*actor);
        }
    }
    return facts;
}

std::optional<RasterizerVulkan::OcarinaCameraFacts> RasterizerVulkan::CaptureOcarinaCamera() {
    // Tracking needs the guest binding: the shader-visible view omits unused slots and
    // truncates ranges to shader usage.
    const auto& binding = maxwell3d->state.shader_stages[0].const_buffers[3];
    constexpr size_t bytes = VideoCommon::OcarinaCameraBytes;
    if (!binding.enabled || binding.address == 0 || binding.size < bytes ||
        !gpu_memory->IsFullyMappedRange(binding.address, bytes) ||
        !gpu_memory->IsContinuousRange(binding.address, bytes)) {
        return std::nullopt;
    }
    const auto device_address = gpu_memory->GpuToCpuAddress(binding.address);
    if (!device_address || buffer_cache.IsRegionGpuModified(*device_address, bytes)) {
        return OcarinaCameraFacts{.values = {}, .gpu_written = true};
    }
    OcarinaCameraFacts camera;
    camera.gpu_written = false;
    device_memory.ReadBlockUnsafe(*device_address, camera.values.data(), bytes);
    return camera;
}

/// Capture the skinning pose of the recognized Link draw, for frame-to-frame motion.
std::optional<RasterizerVulkan::OcarinaActorPose> RasterizerVulkan::CaptureOcarinaActor() {
    const auto& regs = maxwell3d->regs;
    const auto& draw = maxwell3d->draw_manager.draw_state;
    const auto& stream = regs.vertex_streams[0];
    if (draw.index_buffer.count != 9702 || draw.index_buffer.FormatSizeInBytes() != 2 ||
        !stream.IsEnabled() || stream.stride != 48 || regs.vertex_stream_instances.IsInstancingEnabled(0)) {
        return std::nullopt;
    }
    const auto attribute = [&](u32 location, VkFormat format, u32 offset) {
        const auto& attrib = regs.vertex_attrib_format[location];
        return attrib.buffer == 0 && !attrib.constant && attrib.offset == offset &&
            MaxwellToVK::VertexFormat(device, attrib.type, attrib.size) == format;
    };
    if (!attribute(0, VK_FORMAT_R32G32B32_SFLOAT, 0) ||
        !attribute(4, VK_FORMAT_R8G8B8A8_UNORM, 40) || !attribute(5, VK_FORMAT_R8G8B8A8_UNORM, 44) ||
        !attribute(6, VK_FORMAT_R16G16B16A16_UINT, 12) || !attribute(7, VK_FORMAT_R16G16B16A16_UINT, 20)) {
        return std::nullopt;
    }

    OcarinaActorPose pose{
        .channel = gpu_memory,
        .vertex = stream.Address(),
        .indices = draw.index_buffer.IndexStart(),
        .base_vertex = static_cast<s32>(draw.base_index),
    };
    bool origin_found{}, bones_found{};
    buffer_cache.ForEachBoundBuffer(false, [&](u32 stage, bool storage, u32 index, bool,
                                                const auto& binding, u32 size, const auto*, u32) {
        if (stage != 0 || storage || (index != 4 && index != 6) ||
            buffer_cache.IsRegionGpuModified(binding.device_addr, size)) {
            return;
        }
        if (index == 4 && size >= 0x40 + 3 * sizeof(f32)) {
            device_memory.ReadBlockUnsafe(binding.device_addr + 0x40, pose.origin.data(), 3 * sizeof(f32));
            origin_found = true;
        }
        if (index == 6 && size == 13568) {
            device_memory.ReadBlockUnsafe(binding.device_addr, pose.bones.data(), sizeof(pose.bones));
            bones_found = true;
        }
    });
    if (!origin_found || !bones_found) {
        return std::nullopt;
    }
    return pose;
}

/// Read the shared mesh only when both poses refer to the same vertex and index buffers.
void RasterizerVulkan::ReadOcarinaActor(OcarinaActorMesh& mesh) {
    const auto& now = oot_actor;
    const auto& before = oot_previous_actor;
    if (now.frame != render_trace_frame || before.frame + 1 != now.frame ||
        now.channel != gpu_memory || before.channel != now.channel || now.vertex != before.vertex ||
        now.indices != before.indices || now.base_vertex != before.base_vertex) {
        return;
    }
    const auto read = [&](GPUVAddr address, void* dst, size_t size) {
        if (!gpu_memory->IsContinuousRange(address, size) ||
            !gpu_memory->IsFullyMappedRange(address, size)) {
            return false;
        }
        const auto device_address = gpu_memory->GpuToCpuAddress(address);
        if (!device_address || buffer_cache.IsRegionGpuModified(*device_address, size) ||
            gpu.HasPendingCPUWrites(*device_address, size)) {
            return false;
        }
        gpu_memory->ReadBlockUnsafe(address, dst, size);
        return true;
    };
    constexpr size_t index_count = 9702;
    constexpr size_t stride = 48;
    std::vector<u16> indices(index_count);
    if (!read(now.indices, indices.data(), indices.size() * sizeof(u16))) {
        return;
    }
    const auto [minimum, maximum] = std::ranges::minmax(indices);
    const s64 first = static_cast<s64>(now.base_vertex) + minimum;
    const u32 vertex_count = static_cast<u32>(maximum) - minimum + 1;
    if (first < 0 || first > (1 << 20) || vertex_count > 16384) {
        return;
    }
    std::vector<u8> vertices(vertex_count * stride);
    if (!read(now.vertex + first * stride, vertices.data(), vertices.size())) {
        return;
    }
    for (u16& index : indices) {
        index -= minimum;
    }

    mesh = {std::move(vertices), std::move(indices), now.origin, before.origin, now.bones, before.bones};
}

/// Track the scene depth on its reader dispatch, or try to replace a recognized upscale dispatch.
bool RasterizerVulkan::TryOcarinaUpscale(const ComputePipeline& pipeline, u64 native_key,
                                         bool replace) {
    if ((!replace && native_key != VideoCommon::OcarinaDepthReaderKey) ||
        !OcarinaTemporalEnabled()) {
        return false;
    }
    const bool native_resolution =
        oot_backend == VideoCommon::OcarinaUpscaleBackend::Dlss &&
        Settings::GetNvidiaDlssRuntimeSettings().quality == Settings::NvidiaDlssQuality::Native;
    const auto read_tic = [this](u32 index) -> std::optional<Tegra::Texture::TICEntry> {
        const auto& table = kepler_compute->regs.tic;
        using Entry = Tegra::Texture::TICEntry;
        const GPUVAddr address = table.Address() + u64{index} * sizeof(Entry);
        if (index > table.limit || !gpu_memory->IsFullyMappedRange(address, sizeof(Entry))) {
            return std::nullopt;
        }
        Entry entry{};
        gpu_memory->ReadBlockUnsafe(address, &entry, sizeof(entry));
        return entry;
    };
    if (!replace) {
        constexpr u32 depth_handle_offset = 0x128;
        const auto& qmd = kepler_compute->launch_description;
        const auto& cbuf = qmd.const_buffer_config[0];
        if (!(qmd.const_buffer_enable_mask & 1) || cbuf.size < depth_handle_offset + sizeof(u32) ||
            !gpu_memory->IsFullyMappedRange(cbuf.Address(), depth_handle_offset + sizeof(u32))) {
            return false;
        }
        const u32 index = gpu_memory->Read<u32>(cbuf.Address() + depth_handle_offset) & 0xfffff;
        const auto tic = read_tic(index);
        const bool scene_size = oot_camera_sample_frame != render_trace_frame ||
                                (tic && oot_motion.SceneSized(tic->Width(), tic->Height()));
        if (tic && tic->Address() && scene_size) {
            oot_motion.depth_channel = gpu_memory;
            oot_motion.depth_frame = render_trace_frame;
            oot_motion.depth_tic = index;
            oot_motion.depth_address = tic->Address();
        }
        return false;
    }
    // Logs why the game's own upscale draws, when the reason changes and every 120th repeat.
    const auto fail = [this](const char* reason, std::string_view detail = {}) {
        if (reason != oot_fallback_reason || ++oot_fallback_repeats % 120 == 0) {
            LOG_INFO(Render_Vulkan, "Ocarina temporal: the game's upscale draws frame {} ({}{})",
                     render_trace_frame, reason, detail);
        }
        if (reason != oot_fallback_reason) {
            oot_fallback_reason = reason;
            oot_fallback_repeats = 0;
        }
        return false;
    };
    if (oot_camera_sample_frame != render_trace_frame) {
        if (!oot_camera_refusal || oot_camera_refusal->frame != render_trace_frame) {
            return fail("no scene camera this frame");
        }
        const auto& refusal = *oot_camera_refusal;
        return fail(refusal.reason, fmt::format(": camera {}x{}, scene target {}x{}",
                                                refusal.camera_width, refusal.camera_height,
                                                refusal.target_width, refusal.target_height));
    }
    if (oot_motion.depth_frame != render_trace_frame) {
        return fail("no scene depth this frame");
    }
    if (oot_motion.camera_channel != gpu_memory || oot_motion.depth_channel != gpu_memory) {
        return fail("camera and depth from another channel");
    }
    const auto tic = read_tic(oot_motion.depth_tic);
    if (!tic || tic->Address() != oot_motion.depth_address ||
        !oot_motion.SceneSized(tic->Width(), tic->Height())) {
        return fail("scene depth descriptor changed");
    }

    texture_cache.SynchronizeDescriptors(true);
    std::array images{VideoCommon::ImageViewInOut{.index = oot_motion.depth_tic}};
    texture_cache.FillImageViews(images, true);
    if (!images[0].id) {
        return fail("no scene depth image");
    }
    auto& depth = texture_cache.GetImageView(images[0].id);
    const auto* backing = depth.BackingImage();
    if (!backing || depth.IsRescaled() || depth.format != VideoCore::Surface::PixelFormat::D32_FLOAT ||
        depth.range != VideoCommon::SubresourceRange{} || backing->info.num_samples != 1) {
        return fail("unsupported scene depth");
    }

    const auto sampled = pipeline.SampledImages();
    const auto written = pipeline.WrittenImages();
    if (sampled.size() != 1 || written.size() != 1 || !sampled[0].id || !written[0]) {
        return fail("unexpected upscale bindings");
    }
    const auto& color = texture_cache.GetImageView(sampled[0].id);
    auto& target = texture_cache.GetImageView(written[0]);
    using VideoCore::Surface::PixelFormat;
    const auto extent = oot_motion.extent;
    using enum VideoCommon::OcarinaUpscalePass;
    const auto pass = VideoCommon::FindOcarinaUpscalePass(native_key, pipeline.ShaderHash());
    const bool target_1440p = target.size.width == 2560 && target.size.height == 1440;
    // At the DLSS Performance render scale (1920x1080) the game upscales to 2560x1440 with its
    // 3/4 pass instead (QuincyForbes' f7d0938b5b). The scale is the one the game started with,
    // whatever quality is chosen since.
    const bool pass_1440p = target_1440p && (pass == OneToOne || pass == Scaled ||
                                             (pass == ThreeQuarter && oot_performance_scale));
    // Handheld with the Max Resolution add-on: FSR runs as native AA. DLAA through it is untested.
    const bool native_size = pass == OneToOne && !target_1440p;
    if (native_size && oot_backend != VideoCommon::OcarinaUpscaleBackend::Fsr) {
        return fail("the full-size pass is replaced only by FSR");
    }
    const bool upscale_4k = pass_1440p && !native_resolution;
    // The 1440p and scaled passes take any 16:9 scene up to their target: the DLSS render scales
    // shrink it docked, and the game's dynamic resolution handheld (854x480 at its floor).
    const bool scaled_scene = extent.width <= target.size.width &&
                              extent.height <= target.size.height &&
                              std::abs(static_cast<int>(extent.width * 9) -
                                       static_cast<int>(extent.height * 16)) <= 32;
    const bool valid_extents =
        pass_1440p || pass == Scaled ? scaled_scene
        : native_size ? target.size.width == extent.width && target.size.height == extent.height
                      : target.size.width == extent.width * 4 / 3 &&
                            target.size.height == extent.height * 4 / 3;
    if (!valid_extents || !color.BackingImage() || !target.BackingImage() ||
        color.IsRescaled() || target.IsRescaled() ||
        color.Samples() != VK_SAMPLE_COUNT_1_BIT || target.Samples() != VK_SAMPLE_COUNT_1_BIT ||
        color.range != VideoCommon::SubresourceRange{} || target.range != VideoCommon::SubresourceRange{} ||
        color.format != PixelFormat::E5B9G9R9_FLOAT || target.format != PixelFormat::B10G11R11_FLOAT ||
        !oot_motion.SceneSized(color.size.width, color.size.height) ||
        color.ImageHandle() == target.ImageHandle() || depth.ImageHandle() == target.ImageHandle()) {
        return fail("unexpected upscale images");
    }

    constexpr VkFormat output_format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    const auto color_view = color.Handle(Shader::TextureType::Color2D);
    if (!color_view || !device.IsStorageImageExtendedFormatsSupported() ||
        !device.IsFormatSupported(output_format, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT, FormatType::Optimal) ||
        (target.BackingImage()->UsageFlags() & VK_IMAGE_USAGE_STORAGE_BIT) == 0 ||
        MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, true, target.format).format != output_format) {
        return fail("unsupported storage format");
    }
    // Without Link's pose the mesh stays empty and the motion comes from the camera alone.
    OcarinaActorMesh mesh{};
    ReadOcarinaActor(mesh);
    try {
        const auto target_view =
            target.StorageView(Shader::TextureType::Color2D, Shader::ImageFormat::Typeless, false);
        const auto scene_reference = upscale_4k ? target.Handle(Shader::TextureType::Color2D)
                                                : VK_NULL_HANDLE;
        if (!target_view || (upscale_4k && !scene_reference)) {
            return fail("no upscale target view");
        }
        if (!oot_live_dlss) {
            oot_live_dlss = std::make_unique<OcarinaUpscaler>(
                device, scheduler, descriptor_pool, compute_pass_descriptor_queue, memory_allocator,
                staging_pool, *oot_backend);
        }
        const VkExtent2D target_extent{target.size.width, target.size.height};
        // The offset this frame's scene was drawn with (none when no scene draw was offset).
        std::array<f32, 2> jitter{};
        if (oot_jitter_frame == render_trace_frame) {
            jitter = oot_jitter_offset;
        }
        const VkExtent2D output_extent = native_resolution ? extent
                                         : upscale_4k      ? VkExtent2D{3840, 2160}
                                                           : target_extent;
        const bool recorded = oot_live_dlss->Record(color_view, depth.DepthView(), target_view,
                                                   extent, output_extent, target_extent, oot_motion.current,
                                                   oot_motion.previous, mesh, depth.ImageHandle(),
                                                   render_trace_frame, jitter,
                                                   {depth.size.width, depth.size.height});
        if (!recorded) {
            return fail("temporal upscaler declined the frame");
        }
        if (upscale_4k) {
            oot_live_dlss->SaveSceneReference(scene_reference);
            oot_4k_scene_frame = render_trace_frame;
            oot_4k_guest_present = VK_NULL_HANDLE;
        }
        oot_dlss_frame = render_trace_frame;
        // AMD's jitter phase count for FSR: 8 * (output / render width)^2, so 18 for 1440p -> 4K
        // and 8 for native AA.
        const f32 ratio = static_cast<f32>(output_extent.width) / static_cast<f32>(extent.width);
        oot_jitter_phases = std::max(1U, static_cast<u32>(8.0f * ratio * ratio));
        oot_fallback_reason = nullptr;
        return true;
    } catch (const vk::Exception& error) {
        LOG_WARNING(Render_Vulkan, "Ocarina temporal upscaling unavailable: {}", error.what());
        oot_live_failed = true;
        texture_cache_runtime.ocarina_fsr_failed = true; // samplers drop the FSR LOD bias
        return false;
    }
}

bool RasterizerVulkan::FrameDumpBuffers(bool compute, std::span<const u64> shader_hashes) {
    static constexpr u32 BUFFER_CAP = 4U << 20;
    std::vector<FrameDump::BoundBuffer> buffers;
    buffer_cache.ForEachBoundBuffer(compute, [&](u32 stage, bool storage, u32 index, bool written,
                                                 const auto& binding, u32 bound_size,
                                                 const auto* host, u32 host_offset) {
        const u32 size = std::min(bound_size, BUFFER_CAP);
        const bool gpu_modified = buffer_cache.IsRegionGpuModified(binding.device_addr, size);
        FrameDump::BoundBuffer& buffer = buffers.emplace_back(FrameDump::BoundBuffer{
            .stage = stage,
            .storage = storage,
            .index = index,
            .written = written,
            .gpu_addr = binding.gpu_addr,
            .device_addr = binding.device_addr,
            .bound_size = bound_size,
            .size = size,
            .gpu_modified = gpu_modified,
            .host = host ? host->Handle() : VK_NULL_HANDLE,
            .host_offset = host_offset,
            .guest = {},
        });
        if (!gpu_modified || !host) {
            buffer.guest.resize(size);
            device_memory.ReadBlockUnsafe(binding.device_addr, buffer.guest.data(), size);
        }
    });
    return frame_dump.NoteBuffers(compute, shader_hashes, buffers);
}

void RasterizerVulkan::FrameDumpPass(const Framebuffer* framebuffer) {
    std::array<ImageView*, VideoCommon::NUM_RT> colors{};
    const auto& ids = texture_cache.GetRenderTargets().color_buffer_ids;
    for (size_t slot = 0; slot < ids.size(); ++slot) {
        if (ids[slot]) {
            colors[slot] = &texture_cache.GetImageView(ids[slot]);
        }
    }
    if (frame_dump.BeginPass(framebuffer, colors)) {
        scheduler.RequestRenderpass(framebuffer);
    }
}

void RasterizerVulkan::TickFrame() {
    RENDER_LEND(TickFrame())
    VideoCommon::GpuSplit::CheckParked(VideoCommon::GpuSplit::Unparked::FrameTick);
    buffer_cache_runtime.Nx2Slots().LandDone(
        std::bind_front(&RasterizerVulkan::Nx2LandOwedWord, this));
    if (VideoCore::Nx2Debug::HasRequests()) {
        if (auto capture = VideoCore::Nx2Debug::TakeCapture()) {
            frame_dump.Arm(std::move(*capture));
        }
        if (const auto change = VideoCore::Nx2Debug::TakeRejectChange()) {
            pipeline_cache.RefreshDebugRejections(change->unrejected);
            VideoCore::Nx2Debug::NoteRejectApplied(change->generation);
        }
        while (const auto read = VideoCore::Nx2Debug::TakeGpuRead()) {
            std::vector<Tegra::MemoryManager*> spaces;
            if (gpu_memory != nullptr) {
                spaces.push_back(gpu_memory);
            }
            {
                std::scoped_lock lock{config_mutex};
                for (const auto& [id, space] : address_spaces) {
                    if (space.gpu_memory != gpu_memory) {
                        spaces.push_back(space.gpu_memory);
                    }
                }
            }
            VideoCore::Nx2Debug::FinishGpuRead(read->id,
                                               ReadGpuMemory(spaces, read->gpu_address, read->size));
        }
    }
    if (fast_draw_on) {
        ReportFastDraws();
        VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Frame);
    }
    if (DrawReuseProbeEnabled()) {
        TickDrawReuse();
    }
    frame_dump.TickFrame();
    ++render_trace_frame;
    render_trace_draws = 0;
    render_trace_clears = 0;
    if (Common::DkbFrameTrace::Enabled()) {
        CollectIndirectTrace();
        const auto now = std::chrono::steady_clock::now();
        if (trace_cpu_rate.Ready(now)) {
            auto snapshot = Common::DkbCpuTrace::ReadSnapshot();
            if (trace_cpu_snapshot && trace_frame > trace_cpu_frame) {
                const u64 frames = trace_frame - trace_cpu_frame;
                std::string roles;
                for (size_t i = 0; i < Common::DkbCpuTrace::RoleCount; ++i) {
                    const auto name = Common::DkbCpuTrace::RoleNames[i];
                    const auto delta = Common::DkbCpuTrace::CpuTimeDelta(
                        trace_cpu_snapshot->threads[i], snapshot.threads[i]);
                    if (delta) {
                        const double cpu_ms = static_cast<double>(*delta) / 1e6;
                        roles += fmt::format(" {}_cpu_ms={:.3f} {}_cpu_ms_per_frame={:.3f}",
                                             name, cpu_ms, name, cpu_ms / frames);
                    } else {
                        roles += fmt::format(" {}={}", name,
                                             snapshot.threads[i].active ? "unavailable" : "inactive");
                    }
                }
                LOG_INFO(Render_Vulkan,
                         "DKB cpu through_frame={} frames={} window_wall_ms={:.3f}{}",
                         trace_frame, frames,
                         std::chrono::duration<double, std::milli>(
                             snapshot.wall_time - trace_cpu_snapshot->wall_time).count(),
                         roles);
            }
            trace_cpu_snapshot = std::move(snapshot);
            trace_cpu_frame = trace_frame;
        }
        ++trace_window_frames;
        if (trace_sample_frame) {
            LOG_INFO(Render_Vulkan,
                     "DKB draws through_frame={} frames={} attempts={} skipped={} indirect={} "
                     "zero_direct={} capture_dropped={} attempts_per_frame={:.1f} "
                     "zero_direct_per_frame={:.1f} (totals since previous line)",
                     trace_frame, trace_window_frames, trace_draws, trace_skipped, trace_indirect,
                     trace_zero_direct, trace_capture_dropped,
                     static_cast<double>(trace_draws) / trace_window_frames,
                     static_cast<double>(trace_zero_direct) / trace_window_frames);
            trace_draws = trace_skipped = trace_indirect = trace_zero_direct = 0;
            trace_window_frames = 0;
            trace_capture_dropped = trace_failure_details = 0;
            trace_rate.next = now + std::chrono::seconds{1};
        }
        ++trace_frame;
        trace_captures = 0;
        trace_sample_frame = trace_frame < 4 || trace_rate.Ready(now);
    }
    ComputePipeline::TickUpscalerTrace();
    draw_counter = 0;
    pending_native_work = false;
    pipeline_cache.TickFrame();
    guest_descriptor_queue.TickFrame();
    compute_pass_descriptor_queue.TickFrame();
    descriptor_buffer_ring.TickFrame();
    fence_manager.TickFrame();
    staging_pool.TickFrame();
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.TickFrame();
    }
    scheduler.JoinRecorders();
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.TickFrame();
    }
}

bool RasterizerVulkan::AccelerateConditionalRendering() {
    using VideoCommon::GpuSplit;
    using RenderEnable = Tegra::Engines::Maxwell3D::Regs::RenderEnable;
    if (GpuSplit::Get().Posts(GpuSplit::PostedOp::ConditionalEnd) && maxwell3d != nullptr) {
        // EDEN_NX2_RENDER_POST_CONDITIONAL: a static render enable (an override, or True or
        // False) is decided from the back's registers, and only ends host conditional
        // rendering on GPURender, posted in order. The dynamic modes read the query cache.
        const auto& regs = maxwell3d->regs;
        const bool dynamic =
            regs.render_enable_override == RenderEnable::Override::UseRenderEnable &&
            regs.render_enable.mode != RenderEnable::Mode::True &&
            regs.render_enable.mode != RenderEnable::Mode::False;
        if (!dynamic) {
            gpu_memory->FlushCaching();
            if (host_conditional_may_be_set) {
                host_conditional_may_be_set = false;
                PostRenderOp(GpuSplit::PostedOp::ConditionalEnd,
                             [this] { query_cache.EndHostConditionalRendering(); });
            }
            return false;
        }
        host_conditional_may_be_set = true;
    }
    RENDER_LEND_RESULT(AccelerateConditionalRendering())
    gpu_memory->FlushCaching();
    return query_cache.AccelerateHostConditionalRendering();
}

bool RasterizerVulkan::HasDrawTransformFeedback() {
    return device.IsTransformFeedbackDrawSupported();
}

bool RasterizerVulkan::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                             const Tegra::Engines::Fermi2D::Surface& dst,
                                             const Tegra::Engines::Fermi2D::Config& copy_config) {
    RENDER_LEND_RESULT(AccelerateSurfaceCopy(src, dst, copy_config))
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.BlitImage(dst, src, copy_config);
}

bool RasterizerVulkan::UsesNvidiaDlss() const {
    return texture_cache_runtime.UsesNvidiaDlss();
}

bool RasterizerVulkan::SupportsGameDlssNetwork() const {
    // Compute pipelines are created 32 threads wide where the device can be told to.
    return device.IsGuestWarpSizeSupported(VK_SHADER_STAGE_COMPUTE_BIT);
}

std::optional<VideoCore::GameDlssProbe> RasterizerVulkan::ProbeGameDlssOutput(
    const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
    std::optional<std::array<u32, 2>> src_region) {
    RENDER_LEND_RESULT(ProbeGameDlssOutput(src, dst, src_region))
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.ProbeGameDlssOutput(dst, src, src_region);
}

bool RasterizerVulkan::AccelerateTemporalUpscale(
    const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
    const Tegra::Engines::Fermi2D::Surface& motion_vectors, const Tegra::Texture::TICEntry* depth,
    std::array<f32, 2> jitter, std::array<f32, 2> motion_scale, bool use_history,
    std::optional<std::array<u32, 2>> render_region) {
    RENDER_LEND_RESULT(AccelerateTemporalUpscale(src, dst, motion_vectors, depth, jitter,
                                                 motion_scale, use_history, render_region))
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.TemporalUpscale(dst, src, motion_vectors, depth, jitter, motion_scale,
                                          use_history, render_region);
}

Tegra::Engines::AccelerateDMAInterface& RasterizerVulkan::AccessAccelerateDMA() {
    return accelerate_dma;
}

namespace {
// EDEN_NX2_INLINE_DEFER=0: invalidate every cache at once (for comparisons).
bool InlineDefers() {
    static const bool defer = [] {
        const char* value = std::getenv("EDEN_NX2_INLINE_DEFER");
        return value == nullptr || value[0] != '0';
    }();
    return defer;
}
} // namespace

void RasterizerVulkan::TraceInlineToMemory(GPUVAddr address, size_t copy_size,
                                           std::span<const u8> memory,
                                           std::optional<DAddr> cpu_addr) {
    if (gpu_memory->UsesNativeTiling()) {
        u32 first_word{};
        if (memory.size() >= sizeof(first_word)) std::memcpy(&first_word, memory.data(), sizeof(first_word));
        VideoCommon::TraceNativeQuery(11, gpu_memory->GetID(), address, cpu_addr.value_or(0),
            0, first_word, static_cast<u32>(copy_size));
        static const GPUVAddr watch = [] {
            const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
            return value ? GPUVAddr{std::strtoull(value, nullptr, 16)} : GPUVAddr{};
        }();
        if (watch != 0 && address < watch + 16 && address + copy_size > watch) {
            LOG_INFO(Render_Vulkan, "Semaphore watch: inline write {:#x} size {} first word {:#x}",
                     address, copy_size, first_word);
        }
    }
}

bool RasterizerVulkan::InlineBufferHalf(DAddr cpu_addr, size_t copy_size,
                                        std::span<const u8> memory, bool defer) {
    bool uploaded = false;
    // One cache at a time: holding both while waiting for either (std::lock) waited longer, the
    // fencing thread and CPU write traps holding one of them most of the time.
    // InlineMemory uploads only into GPU-written buffer data, and Nx2NoteListHostWrite acts only
    // on command list pages and kept words (GPU-written too). Every range the buffer cache records
    // as GPU-written marks its pages in Tegra::g_nx2_gpu_written_pages first, under its lock, on
    // this thread, and the bits never clear. So when no written page has either mark the locked
    // section changes nothing, and the write skips the lock (jpkto/Lichii 7a5e5357cb).
    const DAddr page_begin = Common::AlignDown(cpu_addr, Core::DEVICE_PAGESIZE);
    const DAddr page_end = Common::AlignUp(cpu_addr + copy_size, Core::DEVICE_PAGESIZE);
    const auto reaches_list_page = [&] {
        if (copy_size == 0 ||
            !Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed)) {
            return false;
        }
        for (DAddr page = page_begin; page < page_end;
             page += DAddr{1} << Tegra::Nx2WrittenPageBits) {
            if (Tegra::Nx2IsCommandListPage(page)) {
                return true;
            }
        }
        return false;
    };
    const bool lock_free = defer && Tegra::g_nx2_ampere.load(std::memory_order_relaxed) &&
                           !Tegra::Nx2MayBeGpuWritten(page_begin, page_end - page_begin) &&
                           !reaches_list_page();
    if (lock_free) {
        if (VideoCommon::BackX::g_verify) {
            std::scoped_lock lock{buffer_cache.mutex};
            if (buffer_cache.IsRegionGpuModified(page_begin, page_end - page_begin)) {
                VideoCommon::BackX::Mismatch("inline write into GPU-written buffer data", cpu_addr,
                                             copy_size);
                buffer_cache.Nx2NoteListHostWrite(cpu_addr, copy_size);
                uploaded = buffer_cache.InlineMemory(cpu_addr, copy_size, memory);
            }
        }
    } else {
        std::scoped_lock lock{buffer_cache.mutex};
        // Guest memory has the words now: a command list copy of the page no longer does.
        buffer_cache.Nx2NoteListHostWrite(cpu_addr, copy_size);
        uploaded = buffer_cache.InlineMemory(cpu_addr, copy_size, memory);
        if (!uploaded && !defer) {
            buffer_cache.WriteMemory(cpu_addr, copy_size);
        }
    }
    return uploaded;
}

bool RasterizerVulkan::PostInlineToMemory(GPUVAddr address, size_t copy_size,
                                          std::span<const u8> memory) {
    using VideoCommon::GpuSplit;
    if (!GpuSplit::Get().Posts(GpuSplit::PostedOp::Inline) || !InlineDefers()) {
        return false;
    }
    const auto cpu_addr = gpu_memory->GpuToCpuAddress(address);
    // The back looks up shaders itself: an upload over one stays lent.
    if (!cpu_addr || pipeline_cache.page_filter.Any(*cpu_addr, copy_size)) {
        return false;
    }
    TraceInlineToMemory(address, copy_size, memory, cpu_addr);
    gpu_memory->WriteBlockUnsafe(address, memory.data(), copy_size);
    bool uploaded;
    {
        // Uploads into GPU-written buffer data go in the op, played where the lent call ran.
        const BufferArena::Collect collect{inline_commands};
        uploaded = InlineBufferHalf(*cpu_addr, copy_size, memory, true);
    }
    const bool query = query_cache.page_filter.Any(*cpu_addr, copy_size);
    if (!uploaded) {
        gpu_memory->AccumulateInvalidation(address, copy_size);
    }
    if (inline_commands.Empty() && !query && !uploaded) {
        return true;
    }
    VideoCommon::CommandArena commands;
    commands.Swap(inline_commands);
    const u64 stamp = commands.Empty() ? 0 : buffer_cache_runtime.Arena().CloseCollected();
    PostRenderOp(GpuSplit::PostedOp::Inline, [this, commands = std::move(commands), stamp,
                                              addr = *cpu_addr, copy_size, query, uploaded] {
        if (!commands.Empty()) {
            buffer_cache_runtime.Arena().PlayCollected(commands, stamp);
        }
        // Pending query results must not land over the new words later.
        if (query) {
            query_cache.InvalidateRegion(addr, copy_size);
        }
        if (uploaded) {
            std::scoped_lock lock{texture_cache.mutex};
            texture_cache.WriteMemory(addr, copy_size);
        }
    });
    return true;
}

void RasterizerVulkan::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                                std::span<const u8> memory) {
    if (scheduler.RenderRecordersEnabled() && PostInlineToMemory(address, copy_size, memory)) {
        return;
    }
    RENDER_LEND(AccelerateInlineToMemory(address, copy_size, memory))
    auto cpu_addr = gpu_memory->GpuToCpuAddress(address);
    TraceInlineToMemory(address, copy_size, memory, cpu_addr);
    if (!cpu_addr) [[unlikely]] {
        gpu_memory->WriteBlock(address, memory.data(), copy_size);
        return;
    }
    gpu_memory->WriteBlockUnsafe(address, memory.data(), copy_size);
    const bool defer = InlineDefers();
    const bool uploaded = InlineBufferHalf(*cpu_addr, copy_size, memory, defer);
    // Pending query results must not land over the new words later: drop them now.
    query_cache.InvalidateRegion(*cpu_addr, copy_size);
    if (!uploaded && defer) {
        // Title 1 makes thousands of small inline writes between draws (~10% of the GPU
        // thread invalidating every cache for each). Guest memory is written; the buffer,
        // texture and shader caches see the merged ranges at the next FlushCaching, which runs
        // before anything of theirs is used. Writes into GPU-written buffer data were uploaded
        // in place above, as before.
        gpu_memory->AccumulateInvalidation(address, copy_size);
        return;
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(*cpu_addr, copy_size);
    }
    pipeline_cache.InvalidateRegion(*cpu_addr, copy_size);
}

std::optional<FramebufferTextureInfo> RasterizerVulkan::AccelerateDisplay(
    const Tegra::FramebufferConfig& config, DAddr framebuffer_addr, u32 pixel_stride) {
    RENDER_LEND_RESULT(AccelerateDisplay(config, framebuffer_addr, pixel_stride))
    if (!framebuffer_addr) {
        return {};
    }
    std::scoped_lock lock{texture_cache.mutex};
    const auto [image_view, scaled] =
        texture_cache.TryFindFramebufferImageView(config, framebuffer_addr);
    static u32 display_trace_count{};
    static const bool frame_capture = std::getenv("EDEN_NATIVE_FRAME_CAPTURE") != nullptr;
    if (frame_capture && display_trace_count++ < 64) {
        LOG_INFO(HW_GPU, "Native display source: address={:#x}, offset={:#x}, dimensions={}x{}, stride={}, accelerated={}, mask={:#x}",
                 framebuffer_addr, config.offset, config.width, config.height, pixel_stride,
                 image_view != nullptr, config.layer_stack_mask);
    }
    if (!image_view) {
        return {};
    }
    if (frame_dump.Capturing()) {
        frame_dump.EndPass();
        frame_dump.CapturePresent(*image_view);
    }

    query_cache.NotifySegment(false);

    const auto& resolution = Settings::values.resolution_info;

    FramebufferTextureInfo info{};
    info.image = image_view->ImageHandle();
    info.image_view = image_view->Handle(Shader::TextureType::Color2D);
    info.width = image_view->size.width;
    info.height = image_view->size.height;
    info.scaled_width = scaled ? resolution.ScaleUp(info.width) : info.width;
    info.scaled_height = scaled ? resolution.ScaleUp(info.height) : info.height;
    info.anti_aliased = oot_dlss_frame == render_trace_frame && !oot_live_dlss->EvaluationFailed();
    if (oot_4k_scene_frame == render_trace_frame && info.image &&
        info.image == oot_4k_guest_present && info.image_view &&
        info.width == 2560 && info.height == 1440 && !scaled &&
        image_view->format == VideoCore::Surface::PixelFormat::A8B8G8R8_UNORM &&
        !oot_live_dlss->EvaluationFailed() && !frame_dump.Capturing()) {
        try {
            oot_live_dlss->Composite4k(info.image_view, render_trace_frame);
            info.image = oot_live_dlss->CompositedImage();
            info.image_view = oot_live_dlss->CompositedView();
            info.scaled_width = 3840;
            info.scaled_height = 2160;
            info.already_upscaled = true;
        } catch (const vk::Exception& error) {
            LOG_WARNING(Render_Vulkan, "Ocarina temporal 4K presentation unavailable: {}", error.what());
            oot_live_failed = true;
            texture_cache_runtime.ocarina_fsr_failed = true; // samplers drop the FSR LOD bias
        }
    }
    return info;
}

void RasterizerVulkan::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                         const VideoCore::DiskResourceLoadCallback& callback) {
    pipeline_cache.LoadDiskResources(title_id, stop_loading, callback);
}

void RasterizerVulkan::FlushWork(bool ends_pass) {
#ifdef __ANDROID__
    static constexpr bool android = true;
#else
    static constexpr bool android = false;
#endif // __ANDROID__
    const bool native = gpu_memory->UsesNativeTiling();
    if (descriptor_buffer_ring.NeedsSubmit(scheduler)) {
        draw_counter = 0;
        scheduler.Flush();
        pending_native_work = native && !android;
        return;
    }
    u32 budget = SubmissionPolicy::DefaultNativeBudget;
    if (native && !android) {
        static const std::optional<u32> configured_budget = []() -> std::optional<u32> {
            if (const char* value = std::getenv("EDEN_NX2_FLUSH_DRAWS")) {
                if (const auto parsed = SubmissionPolicy::ParseBudget(value)) {
                    return parsed;
                }
                LOG_WARNING(Render_Vulkan,
                            "Invalid EDEN_NX2_FLUSH_DRAWS; expected an integer from 1 to {}",
                            (std::numeric_limits<u32>::max)());
            }
            return std::nullopt;
        }();
        budget = configured_budget.value_or(SubmissionPolicy::DefaultNativeBudget);
    }
    u32 legacy_budget = 4096;
    if (!native && !android) {
        static const u32 forced_draws = [] {
            const char* value = std::getenv("EDEN_NX2_FLUSH_DRAWS");
            return value != nullptr
                       ? static_cast<u32>(std::max(32L, std::strtol(value, nullptr, 10)) & ~31L)
                       : 0U;
        }();
        legacy_budget = forced_draws != 0 ? forced_draws : (device.IsMoltenVK() ? 128U : 4096U);
    }
    bool pass_boundary = false;
    if (native && !android && draw_counter + 1 >= budget / 2 && draw_counter + 1 < budget) {
        pass_boundary = ends_pass || scheduler.IsBetweenRenderPasses() ||
                        (texture_cache.RenderTargetsDirty() && scheduler.IsRenderPassActive());
    }
    switch (SubmissionPolicy::Advance(draw_counter, native, android, budget, legacy_budget,
                                      pass_boundary)) {
    case SubmissionPolicy::Action::None:
        break;
    case SubmissionPolicy::Action::Dispatch:
        scheduler.DispatchWork();
        break;
    case SubmissionPolicy::Action::Flush:
        scheduler.Flush();
        pending_native_work = false;
        break;
    }
    if (native && !android) {
        pending_native_work = true;
    }
}

AccelerateDMA::AccelerateDMA(BufferCache& buffer_cache_, TextureCache& texture_cache_,
                             Scheduler& scheduler_)
    : buffer_cache{buffer_cache_}, texture_cache{texture_cache_}, scheduler{scheduler_} {}

bool AccelerateDMA::BufferClear(GPUVAddr src_address, u64 amount, u32 value) {
    RENDER_LEND_RESULT(BufferClear(src_address, amount, value))
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.Nx2CollectListCopies(true);
    const bool cleared = buffer_cache.DMAClear(src_address, amount, value);
    if (buffer_cache.Nx2CopyListPages(scheduler.CurrentTick())) {
        scheduler.Flush();
    }
    buffer_cache.Nx2CollectListCopies(false);
    return cleared;
}

bool AccelerateDMA::BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    RENDER_LEND_RESULT(BufferCopy(src_address, dest_address, amount))
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    buffer_cache.Nx2CollectListCopies(true);
    boost::container::small_vector<std::pair<DAddr, size_t>, 4> gpu_written;
    const bool copied = buffer_cache.DMACopy(src_address, dest_address, amount, &gpu_written);
    if (buffer_cache.Nx2CopyListPages(scheduler.CurrentTick())) {
        scheduler.Flush();
    }
    buffer_cache.Nx2CollectListCopies(false);
    // DMACopy copies guest memory without telling the texture cache. Images over the destination
    // (pitch linear data textures such as Kirby Air Riders' vertex lighting) would keep their old
    // contents, and stale pages would skip the upload of images made there later.
    if (copied) {
        texture_cache.WriteDmaCopy(src_address, dest_address, amount, gpu_written);
    }
    return copied;
}

template <bool IS_IMAGE_UPLOAD>
bool AccelerateDMA::DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                                       const Tegra::DMA::BufferOperand& buffer_operand,
                                       const Tegra::DMA::ImageOperand& image_operand) {
    RENDER_LEND_RESULT(DmaBufferImageCopy<IS_IMAGE_UPLOAD>(copy_info, buffer_operand, image_operand))
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    const auto image_id = texture_cache.DmaImageId(image_operand, IS_IMAGE_UPLOAD);
    if (image_id == VideoCommon::NULL_IMAGE_ID) {
        return false;
    }
    const u32 buffer_size = static_cast<u32>(buffer_operand.pitch * buffer_operand.height);
    static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
    const auto post_op = IS_IMAGE_UPLOAD ? VideoCommon::ObtainBufferOperation::DoNothing
                                         : VideoCommon::ObtainBufferOperation::MarkAsWritten;
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(buffer_operand.address, buffer_size, sync_info, post_op);
    // Downloads to unmapped memory must not write the shared null buffer.
    if (!IS_IMAGE_UPLOAD && buffer->IsNull()) {
        return false;
    }

    const auto [image, copy] = texture_cache.DmaBufferImageCopy(
        copy_info, buffer_operand, image_operand, image_id, IS_IMAGE_UPLOAD);
    const std::span copy_span{&copy, 1};

    if constexpr (IS_IMAGE_UPLOAD) {
        texture_cache.PrepareImage(image_id, true, false);
        image->UploadMemory(buffer->Handle(), offset, copy_span);
    } else {
        if (offset % BytesPerBlock(image->info.format)) {
            return false;
        }
        texture_cache.DownloadImageIntoBuffer(image, buffer->Handle(), offset, copy_span,
                                              buffer_operand.address, buffer_size);
    }
    return true;
}

bool AccelerateDMA::ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::ImageOperand& image_operand,
                                  const Tegra::DMA::BufferOperand& buffer_operand) {
    return DmaBufferImageCopy<false>(copy_info, buffer_operand, image_operand);
}

bool AccelerateDMA::BufferToImage(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::BufferOperand& buffer_operand,
                                  const Tegra::DMA::ImageOperand& image_operand) {
    return DmaBufferImageCopy<true>(copy_info, buffer_operand, image_operand);
}

template <typename F>
void RasterizerVulkan::RecordDynamic(F&& command, std::source_location location) {
    if (dynamic_commands != nullptr) {
        dynamic_commands->emplace_back(std::move(command));
    } else {
        scheduler.Record(std::forward<F>(command), location);
    }
}

void RasterizerVulkan::UpdateDynamicStates(const GraphicsPipeline* pipeline) {
    auto& regs = DrawEngine().regs;
    auto& flags = DrawEngine().dirty.flags;
    // A mesh emulation pipeline has static input assembly and vertex input. Binding it replaces
    // that dynamic state, so leave it to the next regular draw to set again.
    const bool emulated_mesh{pipeline != nullptr && pipeline->IsEmulatedMesh()};
    if (emulated_mesh) {
        state_tracker.InvalidateInputState();
    }
    const auto topology = DrawEngine().draw_manager.draw_state.topology;
    const bool topology_changed = !emulated_mesh && state_tracker.ChangePrimitiveTopology(topology);
    if (topology_changed) {
        flags[Dirty::DepthBiasEnable] = true;
        flags[Dirty::PrimitiveRestartEnable] = true;
    }
    const bool far_plane = pipeline != nullptr && pipeline->OnFarPlane() &&
                           !PolygonOffsetEnabled(regs, topology) && ViewportKeepsDepth(regs);
    if (far_plane || far_plane != far_plane_draw) {
        flags[Dirty::DepthBias] = true;
    }
    if (far_plane != far_plane_draw) {
        far_plane_draw = far_plane;
        flags[Dirty::DepthBiasEnable] = true;
        flags[Dirty::DepthClampEnable] = true;
    }

    UpdateViewportsState(regs);
    UpdateScissorsState(regs);
    UpdateDepthBias(regs);
    UpdateBlendConstants(regs);
    UpdateDepthBounds(regs);
    UpdateStencilFaces(regs);
    UpdateLineWidth(regs);

    if (device.IsExtExtendedDynamicStateSupported()) {
        UpdateCullMode(regs);
        UpdateDepthCompareOp(regs);
        UpdateFrontFace(regs);
        UpdateStencilOp(regs);
        if (state_tracker.TouchStateEnable()) {
            UpdateDepthBoundsTestEnable(regs);
            UpdateDepthTestEnable(regs);
            UpdateDepthWriteEnable(regs);
            UpdateStencilTestEnable(regs);
        }
        if (topology_changed) {
            RecordDynamic([topology_vk = MaxwellToVK::PrimitiveTopology(device, topology)](
                                 vk::CommandBuffer cmdbuf) {
                cmdbuf.SetPrimitiveTopologyEXT(topology_vk);
            });
        }
    }

    if (device.IsExtExtendedDynamicState2Supported()) {
        if (!emulated_mesh) {
            UpdatePrimitiveRestartEnable(regs);
        }
        UpdateRasterizerDiscardEnable(regs);
        UpdateDepthBiasEnable(regs);
    }

    if (device.IsExtExtendedDynamicState2ExtrasSupported()) {
        UpdateLogicOp(regs);
    }

    if (device.IsExtExtendedDynamicState3EnablesSupported()) {
        using namespace Tegra::Engines;
        // AMD Workaround: LogicOp incompatible with float render targets
        if (device.GetDriverID() == VkDriverIdKHR::VK_DRIVER_ID_AMD_OPEN_SOURCE ||
            device.GetDriverID() == VkDriverIdKHR::VK_DRIVER_ID_AMD_PROPRIETARY) {
            const auto has_float = std::any_of(
                regs.vertex_attrib_format.begin(), regs.vertex_attrib_format.end(),
                [](const auto& attrib) {
                    return attrib.type == Maxwell3D::Regs::VertexAttribute::Type::Float;
                }
            );
            if (regs.logic_op.enable) {
                regs.logic_op.enable = static_cast<u32>(!has_float);
            }
        }
        UpdateLogicOpEnable(regs);
        UpdateDepthClampEnable(regs);
        UpdateLineRasterizationMode(regs);
        UpdateLineStippleEnable(regs);
        UpdateConservativeRasterizationMode(regs);
        UpdateAlphaToCoverageEnable(regs);
        UpdateAlphaToOneEnable(regs);
    }

    if (device.IsExtExtendedDynamicState3BlendingSupported()) {
        UpdateBlending(regs, pipeline);
    } else if (device.IsExtColorWriteEnableSupported()) {
        UpdateColorWriteEnable(regs);
    }

    if (device.IsExtVertexInputDynamicStateSupported() && !emulated_mesh) {
        // A draw passes the pipeline it looked up: a second lookup was ~2.5% of the back thread.
        const GraphicsPipeline* const gp =
            pipeline != nullptr ? pipeline : pipeline_cache.CurrentGraphicsPipeline();
        if (gp && gp->HasDynamicVertexInput()) {
            UpdateVertexInput(regs);
        }
    }
}

void RasterizerVulkan::HandleTransformFeedback() {
    static std::once_flag warn_unsupported;

    const auto& regs = DrawEngine().regs;
    if (!device.IsExtTransformFeedbackSupported()) {
        if (regs.transform_feedback_enabled != 0) {
            std::call_once(warn_unsupported, [&] {
                LOG_WARNING(Render_Vulkan, "Transform feedback requested by guest but VK_EXT_transform_feedback is unavailable; queries disabled");
            });
        } else {
            std::call_once(warn_unsupported, [&] {
                LOG_INFO(Render_Vulkan, "VK_EXT_transform_feedback not available on device");
            });
        }
        return;
    }
    query_cache.CounterEnable(VideoCommon::QueryType::StreamingByteCount,
                              regs.transform_feedback_enabled);
    if (regs.transform_feedback_enabled != 0) {
        // Log extension usage for transform feedback
        if (GPU::Logging::IsActive()) {
            GPU::Logging::GPULogger::GetInstance().LogExtensionUsage(
                "VK_EXT_transform_feedback", "HandleTransformFeedback");
        }
        UNIMPLEMENTED_IF(regs.IsShaderConfigEnabled(Maxwell::ShaderType::TessellationInit) ||
                         regs.IsShaderConfigEnabled(Maxwell::ShaderType::Tessellation));
    }
}

void RasterizerVulkan::UpdateViewportsState(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchViewports()) {
        return;
    }

    DrawEngine().dirty.flags[Dirty::Scissors] = true;

    if (!regs.viewport_scale_offset_enabled) {
        float x = static_cast<float>(regs.surface_clip.x);
        float y = static_cast<float>(regs.surface_clip.y);
        float width = (std::max)(1.0f, static_cast<float>(regs.surface_clip.width));
        float height = (std::max)(1.0f, static_cast<float>(regs.surface_clip.height));
        if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
            y += height;
            height = -height;
        }
        VkViewport viewport{
            .x = x + oot_jitter_applied[0],
            .y = y + oot_jitter_applied[1],
            .width = width,
            .height = height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        RecordDynamic([this, viewport](vk::CommandBuffer cmdbuf) {
            const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
            std::array<VkViewport, Maxwell::NumViewports> viewport_list{};
            viewport_list.fill(viewport);
            const vk::Span<VkViewport> viewports(viewport_list.data(), num_viewports);
            cmdbuf.SetViewport(0, viewports);
        });
        return;
    }
    const bool is_rescaling{texture_cache.IsRescaling()};
    const float scale = is_rescaling ? Settings::values.resolution_info.up_factor : 1.0f;
    std::array viewport_list{
        GetViewportState(device, regs, 0, scale),  GetViewportState(device, regs, 1, scale),
        GetViewportState(device, regs, 2, scale),  GetViewportState(device, regs, 3, scale),
        GetViewportState(device, regs, 4, scale),  GetViewportState(device, regs, 5, scale),
        GetViewportState(device, regs, 6, scale),  GetViewportState(device, regs, 7, scale),
        GetViewportState(device, regs, 8, scale),  GetViewportState(device, regs, 9, scale),
        GetViewportState(device, regs, 10, scale), GetViewportState(device, regs, 11, scale),
        GetViewportState(device, regs, 12, scale), GetViewportState(device, regs, 13, scale),
        GetViewportState(device, regs, 14, scale), GetViewportState(device, regs, 15, scale),
    };
    if (oot_jitter_applied != std::array<f32, 2>{}) {
        for (VkViewport& viewport : viewport_list) {
            viewport.x += oot_jitter_applied[0];
            viewport.y += oot_jitter_applied[1];
        }
    }
    RecordDynamic([this, viewport_list](vk::CommandBuffer cmdbuf) {
        const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
        const vk::Span<VkViewport> viewports(viewport_list.data(), num_viewports);
        cmdbuf.SetViewport(0, viewports);
    });
}

void RasterizerVulkan::UpdateScissorsState(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchScissors()) {
        return;
    }
    if (!regs.viewport_scale_offset_enabled) {
        u32 x = regs.surface_clip.x;
        u32 y = regs.surface_clip.y;
        u32 width = (std::max)(1u, static_cast<u32>(regs.surface_clip.width));
        u32 height = (std::max)(1u, static_cast<u32>(regs.surface_clip.height));
        if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
            y = regs.surface_clip.height - (y + height);
        }
        VkRect2D scissor{};
        scissor.offset.x = static_cast<int32_t>(x);
        scissor.offset.y = static_cast<int32_t>(y);
        scissor.extent.width  = width;
        scissor.extent.height = height;
        RecordDynamic([this, scissor](vk::CommandBuffer cmdbuf) {
            const u32 num_scissors = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
            std::array<VkRect2D, Maxwell::NumViewports> scissor_list{};
            scissor_list.fill(scissor);
            const vk::Span<VkRect2D> scissors(scissor_list.data(), num_scissors);
            cmdbuf.SetScissor(0, scissors);
        });
        return;
    }
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (texture_cache.IsRescaling()) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    const std::array scissor_list{
        GetScissorState(regs, 0, up_scale, down_shift),
        GetScissorState(regs, 1, up_scale, down_shift),
        GetScissorState(regs, 2, up_scale, down_shift),
        GetScissorState(regs, 3, up_scale, down_shift),
        GetScissorState(regs, 4, up_scale, down_shift),
        GetScissorState(regs, 5, up_scale, down_shift),
        GetScissorState(regs, 6, up_scale, down_shift),
        GetScissorState(regs, 7, up_scale, down_shift),
        GetScissorState(regs, 8, up_scale, down_shift),
        GetScissorState(regs, 9, up_scale, down_shift),
        GetScissorState(regs, 10, up_scale, down_shift),
        GetScissorState(regs, 11, up_scale, down_shift),
        GetScissorState(regs, 12, up_scale, down_shift),
        GetScissorState(regs, 13, up_scale, down_shift),
        GetScissorState(regs, 14, up_scale, down_shift),
        GetScissorState(regs, 15, up_scale, down_shift),
    };
    RecordDynamic([this, scissor_list](vk::CommandBuffer cmdbuf) {
        const u32 num_scissors = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
        const vk::Span<VkRect2D> scissors(scissor_list.data(), num_scissors);
        cmdbuf.SetScissor(0, scissors);
    });
}

void RasterizerVulkan::UpdateDepthBias(Tegra::Engines::Maxwell3D::Regs& regs) {
    // The slope term is per pixel, so a rescaled viewport needs it scaled back to guest pixels.
    const float slope_scale = texture_cache.IsRescaling() && regs.viewport_scale_offset_enabled
                                  ? Settings::values.resolution_info.up_factor
                                  : 1.0f;
    if (!state_tracker.TouchDepthBias() && slope_scale == depth_bias_slope_scale) {
        return;
    }
    depth_bias_slope_scale = slope_scale;
    float units = regs.depth_bias / 2.0f;
    const bool is_d24 = regs.zeta.format == Tegra::DepthFormat::Z24_UNORM_S8_UINT ||
                        regs.zeta.format == Tegra::DepthFormat::X8Z24_UNORM ||
                        regs.zeta.format == Tegra::DepthFormat::S8Z24_UNORM ||
                        regs.zeta.format == Tegra::DepthFormat::V8Z24_UNORM;

    if (is_d24 && !device.SupportsD24DepthBuffer()) {
        static constexpr const size_t length = sizeof(NEEDS_D24) / sizeof(NEEDS_D24[0]);

        static constexpr const u64* start = NEEDS_D24;
        static constexpr const u64* end = NEEDS_D24 + length;

        const u64* it = std::find(start, end, program_id);

        if (it != end) {
            // the base formulas can be obtained from here:
            //   https://docs.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-output-merger-stage-depth-bias
            const double rescale_factor =
                static_cast<double>(1ULL << (32 - 24)) / (static_cast<double>(0x1.ep+127));
            units = static_cast<float>(static_cast<double>(units) * rescale_factor);
        }
    }
    float clamp = regs.depth_bias_clamp;
    float factor = regs.slope_scale_depth_bias * slope_scale;
    if (far_plane_draw) {
        const bool reversed =
            regs.viewport_scale_offset_enabled && regs.viewport_transform[0].scale_z < 0.0f;
        units = reversed ? -FAR_PLANE_DEPTH_BIAS : FAR_PLANE_DEPTH_BIAS;
        clamp = 0.0f;
        factor = 0.0f;
    }

    RecordDynamic([constant = units, clamp, factor, this](vk::CommandBuffer cmdbuf) {
        if (device.IsExtDepthBiasControlSupported()) {
            static VkDepthBiasRepresentationInfoEXT bias_info{
                .sType = VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT,
                .pNext = nullptr,
                .depthBiasRepresentation =
                    VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT,
                .depthBiasExact = VK_FALSE,
            };

            cmdbuf.SetDepthBias(constant, clamp, factor, &bias_info);
        } else {
            cmdbuf.SetDepthBias(constant, clamp, factor);
        }
    });
}

void RasterizerVulkan::UpdateBlendConstants(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchBlendConstants()) {
        return;
    }
    const std::array blend_color = {regs.blend_color.r, regs.blend_color.g, regs.blend_color.b,
                                    regs.blend_color.a};
    RecordDynamic(
        [blend_color](vk::CommandBuffer cmdbuf) { cmdbuf.SetBlendConstants(blend_color.data()); });
}

void RasterizerVulkan::UpdateDepthBounds(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBounds()) {
        return;
    }
    RecordDynamic([min = regs.depth_bounds[0], max = regs.depth_bounds[1]](
                         vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthBounds(min, max); });
}

void RasterizerVulkan::UpdateStencilFaces(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilProperties()) {
        return;
    }
    bool update_references = state_tracker.TouchStencilReference();
    bool update_write_mask = state_tracker.TouchStencilWriteMask();
    bool update_compare_masks = state_tracker.TouchStencilCompare();
    if (state_tracker.TouchStencilSide(regs.stencil_two_side_enable != 0)) {
        update_references = true;
        update_write_mask = true;
        update_compare_masks = true;
    }
    if (update_references) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilReferenceFront(regs.stencil_front_ref);
                const bool back_changed =
                    state_tracker.CheckStencilReferenceBack(regs.stencil_back_ref);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilReferenceFront(regs.stencil_front_ref);
                const bool back_changed =
                    state_tracker.CheckStencilReferenceBack(regs.stencil_front_ref);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            RecordDynamic([front_ref = regs.stencil_front_ref, back_ref = regs.stencil_back_ref,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_ref != back_ref;
                // Front face
                cmdbuf.SetStencilReference(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                    : VK_STENCIL_FACE_FRONT_AND_BACK,
                                           front_ref);
                if (set_back) {
                    cmdbuf.SetStencilReference(VK_STENCIL_FACE_BACK_BIT, back_ref);
                }
            });
        }();
    }
    if (update_write_mask) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilWriteMaskFront(regs.stencil_front_mask);
                const bool back_changed =
                    state_tracker.CheckStencilWriteMaskBack(regs.stencil_back_mask);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilWriteMaskFront(regs.stencil_front_mask);
                const bool back_changed =
                    state_tracker.CheckStencilWriteMaskBack(regs.stencil_front_mask);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            RecordDynamic([front_write_mask = regs.stencil_front_mask,
                              back_write_mask = regs.stencil_back_mask,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_write_mask != back_write_mask;
                // Front face
                cmdbuf.SetStencilWriteMask(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                    : VK_STENCIL_FACE_FRONT_AND_BACK,
                                           front_write_mask);
                if (set_back) {
                    cmdbuf.SetStencilWriteMask(VK_STENCIL_FACE_BACK_BIT, back_write_mask);
                }
            });
        }();
    }
    if (update_compare_masks) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilCompareMaskFront(regs.stencil_front_func_mask);
                const bool back_changed =
                    state_tracker.CheckStencilCompareMaskBack(regs.stencil_back_func_mask);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilCompareMaskFront(regs.stencil_front_func_mask);
                const bool back_changed =
                    state_tracker.CheckStencilCompareMaskBack(regs.stencil_front_func_mask);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            RecordDynamic([front_test_mask = regs.stencil_front_func_mask,
                              back_test_mask = regs.stencil_back_func_mask,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_test_mask != back_test_mask;
                // Front face
                cmdbuf.SetStencilCompareMask(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                      : VK_STENCIL_FACE_FRONT_AND_BACK,
                                             front_test_mask);
                if (set_back) {
                    cmdbuf.SetStencilCompareMask(VK_STENCIL_FACE_BACK_BIT, back_test_mask);
                }
            });
        }();
    }
    state_tracker.ClearStencilReset();
}

void RasterizerVulkan::UpdateLineWidth(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLineWidth()) {
        return;
    }
    const float width =
        regs.line_anti_alias_enable ? regs.line_width_smooth : regs.line_width_aliased;
    RecordDynamic([width](vk::CommandBuffer cmdbuf) { cmdbuf.SetLineWidth(width); });
}

void RasterizerVulkan::UpdateCullMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchCullMode()) {
        return;
    }
    RecordDynamic([enabled = regs.gl_cull_test_enabled,
                      cull_face = regs.gl_cull_face](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetCullModeEXT(enabled ? MaxwellToVK::CullFace(cull_face) : VK_CULL_MODE_NONE);
    });
}

void RasterizerVulkan::UpdateDepthBoundsTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBoundsTestEnable()) {
        return;
    }
    bool enabled = regs.depth_bounds_enable;
    if (enabled && !device.IsDepthBoundsSupported()) {
        LOG_WARNING(Render_Vulkan, "Depth bounds is enabled but not supported");
        enabled = false;
    }
    RecordDynamic([enable = enabled](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthBoundsTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateDepthTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthTestEnable()) {
        return;
    }
    RecordDynamic([enable = regs.depth_test_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateDepthWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthWriteEnable()) {
        return;
    }
    RecordDynamic([enable = regs.depth_write_enabled](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthWriteEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdatePrimitiveRestartEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchPrimitiveRestartEnable()) {
        return;
    }

    bool enable = regs.primitive_restart.enabled != 0;
    if (device.IsMoltenVK()) {
        enable = true;
    } else if (enable) {
        const auto topology =
            MaxwellToVK::PrimitiveTopology(device, DrawEngine().draw_manager.draw_state.topology);
        enable = IsPrimitiveRestartSupported(device, topology);
    }

    RecordDynamic([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetPrimitiveRestartEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateRasterizerDiscardEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchRasterizerDiscardEnable()) {
        return;
    }
    RecordDynamic([disable = regs.rasterize_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetRasterizerDiscardEnableEXT(disable == 0);
    });
}

void RasterizerVulkan::UpdateConservativeRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchConservativeRasterizationMode()) {
        return;
    }

    if (!device.SupportsDynamicState3ConservativeRasterizationMode()) {
        return;
    }

    RecordDynamic([enable = regs.conservative_raster_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetConservativeRasterizationModeEXT(
            enable ? VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT
                   : VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);
    });
}

void RasterizerVulkan::UpdateLineStippleEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLineStippleEnable()) {
        return;
    }

    if (!device.SupportsDynamicState3LineStippleEnable()) {
        return;
    }

    RecordDynamic([enable = regs.line_stipple_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLineStippleEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateLineRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!device.IsExtLineRasterizationSupported()) {
        return;
    }
    if (!state_tracker.TouchLineRasterizationMode()) {
        return;
    }

    if (!device.SupportsDynamicState3LineRasterizationMode()) {
        static std::once_flag warn_missing_rect;
        std::call_once(warn_missing_rect, [] {
            LOG_WARNING(Render_Vulkan,
                        "Driver lacks rectangular line rasterization support; skipping dynamic "
                        "line state updates");
        });
        return;
    }

    const bool wants_smooth = regs.line_anti_alias_enable != 0;
    VkLineRasterizationModeEXT mode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT;
    if (wants_smooth) {
        if (device.SupportsSmoothLines()) {
            mode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT;
        } else {
            static std::once_flag warn_missing_smooth;
            std::call_once(warn_missing_smooth, [] {
                LOG_WARNING(Render_Vulkan,
                            "Line anti-aliasing requested but smoothLines feature unavailable; "
                            "using rectangular rasterization");
            });
        }
    }
    RecordDynamic([mode](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLineRasterizationModeEXT(mode);
    });
}

void RasterizerVulkan::UpdateDepthBiasEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBiasEnable()) {
        return;
    }
    const bool enable =
        far_plane_draw || PolygonOffsetEnabled(regs, DrawEngine().draw_manager.draw_state.topology);
    RecordDynamic([enable](vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthBiasEnableEXT(enable); });
}

void RasterizerVulkan::UpdateLogicOpEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLogicOpEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3LogicOpEnable()) {
        return;
    }
    RecordDynamic([enable = regs.logic_op.enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLogicOpEnableEXT(enable != 0);
    });
}

void RasterizerVulkan::UpdateDepthClampEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthClampEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3DepthClampEnable()) {
        return;
    }
    bool is_enabled = far_plane_draw ||
                      !(regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::Passthrough ||
                        regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::FrustumXYZ ||
                        regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::FrustumZ);
    RecordDynamic(
        [is_enabled](vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthClampEnableEXT(is_enabled); });
}

void RasterizerVulkan::UpdateAlphaToCoverageEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchAlphaToCoverageEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3AlphaToCoverageEnable()) {
        return;
    }
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    const bool enable = pipeline != nullptr && pipeline->SupportsAlphaToCoverage() &&
                        regs.anti_alias_alpha_control.alpha_to_coverage != 0;
    RecordDynamic([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetAlphaToCoverageEnableEXT(enable ? VK_TRUE : VK_FALSE);
    });
}

void RasterizerVulkan::UpdateAlphaToOneEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchAlphaToOneEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3AlphaToOneEnable()) {
        static std::once_flag warn_alpha_to_one;
        std::call_once(warn_alpha_to_one, [] {
            LOG_WARNING(Render_Vulkan,
                        "Alpha-to-one is not supported on this device; forcing it disabled");
        });
        return;
    }
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    const bool enable = pipeline != nullptr && pipeline->SupportsAlphaToOne() &&
                        regs.anti_alias_alpha_control.alpha_to_one != 0;
    RecordDynamic([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetAlphaToOneEnableEXT(enable ? VK_TRUE : VK_FALSE);
    });
}

void RasterizerVulkan::UpdateDepthCompareOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthCompareOp()) {
        return;
    }
    RecordDynamic([func = regs.depth_test_func](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthCompareOpEXT(MaxwellToVK::ComparisonOp(func));
    });
}

void RasterizerVulkan::UpdateFrontFace(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchFrontFace()) {
        return;
    }

    VkFrontFace front_face = MaxwellToVK::FrontFace(regs.gl_front_face);
    if (regs.window_origin.flip_y != 0) {
        front_face = front_face == VK_FRONT_FACE_CLOCKWISE ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                           : VK_FRONT_FACE_CLOCKWISE;
    }
    RecordDynamic(
        [front_face](vk::CommandBuffer cmdbuf) { cmdbuf.SetFrontFaceEXT(front_face); });
}

void RasterizerVulkan::UpdateStencilOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilOp()) {
        return;
    }
    const Maxwell::StencilOp::Op fail = regs.stencil_front_op.fail;
    const Maxwell::StencilOp::Op zfail = regs.stencil_front_op.zfail;
    const Maxwell::StencilOp::Op zpass = regs.stencil_front_op.zpass;
    const Maxwell::ComparisonOp compare = regs.stencil_front_op.func;
    if (regs.stencil_two_side_enable) {
        // Separate stencil op per face
        const Maxwell::StencilOp::Op back_fail = regs.stencil_back_op.fail;
        const Maxwell::StencilOp::Op back_zfail = regs.stencil_back_op.zfail;
        const Maxwell::StencilOp::Op back_zpass = regs.stencil_back_op.zpass;
        const Maxwell::ComparisonOp back_compare = regs.stencil_back_op.func;
        RecordDynamic([fail, zfail, zpass, compare, back_fail, back_zfail, back_zpass,
                          back_compare](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_FRONT_BIT, MaxwellToVK::StencilOp(fail),
                                   MaxwellToVK::StencilOp(zpass), MaxwellToVK::StencilOp(zfail),
                                   MaxwellToVK::ComparisonOp(compare));
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_BACK_BIT, MaxwellToVK::StencilOp(back_fail),
                                   MaxwellToVK::StencilOp(back_zpass),
                                   MaxwellToVK::StencilOp(back_zfail),
                                   MaxwellToVK::ComparisonOp(back_compare));
        });
    } else {
        // Front face defines the stencil op of both faces
        RecordDynamic([fail, zfail, zpass, compare](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_FRONT_AND_BACK, MaxwellToVK::StencilOp(fail),
                                   MaxwellToVK::StencilOp(zpass), MaxwellToVK::StencilOp(zfail),
                                   MaxwellToVK::ComparisonOp(compare));
        });
    }
}

void RasterizerVulkan::UpdateLogicOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLogicOp()) {
        return;
    }
    const auto op_value = static_cast<u32>(regs.logic_op.op);
    auto op = op_value >= 0x1500 && op_value < 0x1510 ? static_cast<VkLogicOp>(op_value - 0x1500)
                                                      : VK_LOGIC_OP_NO_OP;
    RecordDynamic([op](vk::CommandBuffer cmdbuf) { cmdbuf.SetLogicOpEXT(op); });
}

void RasterizerVulkan::UpdateBlending(Tegra::Engines::Maxwell3D::Regs& regs,
                                      const GraphicsPipeline* pipeline) {
    // The write mask also excludes components the fragment shader doesn't output, so it has to be
    // recorded again whenever the bound pipeline outputs a different set of components.
    const u32 color_output_mask = pipeline ? pipeline->ColorOutputMask() : ~0U;
    const bool color_output_changed = color_output_mask != last_color_output_mask;
    if (!state_tracker.TouchBlending() && !color_output_changed) {
        return;
    }

    if (state_tracker.TouchColorMask() || color_output_changed) {
        last_color_output_mask = color_output_mask;
        std::array<VkColorComponentFlags, Maxwell::NumRenderTargets> setup_masks{};
        for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
            const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
            auto& current = setup_masks[index];
            if (mask.R) {
                current |= VK_COLOR_COMPONENT_R_BIT;
            }
            if (mask.G) {
                current |= VK_COLOR_COMPONENT_G_BIT;
            }
            if (mask.B) {
                current |= VK_COLOR_COMPONENT_B_BIT;
            }
            if (mask.A) {
                current |= VK_COLOR_COMPONENT_A_BIT;
            }
            current &= (color_output_mask >> (index * 4)) & 0xF;
        }
        RecordDynamic([setup_masks](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorWriteMaskEXT(0, setup_masks);
        });
    }

    if (state_tracker.TouchBlendEnable()) {
        std::array<VkBool32, Maxwell::NumRenderTargets> setup_enables{};
        for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
            bool is_integer = false;
            if (regs.rt[index].format != Tegra::RenderTargetFormat::NONE) {
                const auto format =
                    VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[index].format);
                is_integer = IsPixelFormatInteger(format);
            }
            setup_enables[index] =
                (!is_integer && regs.blend.enable[index] != 0) ? VK_TRUE : VK_FALSE;
        }
        RecordDynamic([setup_enables](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorBlendEnableEXT(0, setup_enables);
        });
    }

    if (state_tracker.TouchBlendEquations()) {
        std::array<VkColorBlendEquationEXT, Maxwell::NumRenderTargets> setup_blends{};

        const auto blend_setup = [&](auto& host_blend, const auto& guest_blend) {
            host_blend.srcColorBlendFactor = MaxwellToVK::BlendFactor(guest_blend.color_source);
            host_blend.dstColorBlendFactor = MaxwellToVK::BlendFactor(guest_blend.color_dest);
            host_blend.colorBlendOp = MaxwellToVK::BlendEquation(guest_blend.color_op);
            host_blend.srcAlphaBlendFactor = MaxwellToVK::BlendFactor(guest_blend.alpha_source);
            host_blend.dstAlphaBlendFactor = MaxwellToVK::BlendFactor(guest_blend.alpha_dest);
            host_blend.alphaBlendOp = MaxwellToVK::BlendEquation(guest_blend.alpha_op);
        };

        // Single blend equation for all targets
        if (!regs.blend_per_target_enabled) {
            // Temporary workaround for games that use iterated blending
            if (regs.iterated_blend.enable && Settings::values.use_squashed_iterated_blend) {
                setup_blends[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
                setup_blends[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
                setup_blends[0].colorBlendOp = VK_BLEND_OP_ADD;
                setup_blends[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
                setup_blends[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
                setup_blends[0].alphaBlendOp = VK_BLEND_OP_ADD;
            } else {
                blend_setup(setup_blends[0], regs.blend);
            }

            // Copy first blend state to all other targets
            for (size_t index = 1; index < Maxwell::NumRenderTargets; index++) {
                setup_blends[index] = setup_blends[0];
            }
        } else {
            // Per-target blending
            for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
                blend_setup(setup_blends[index], regs.blend_per_target[index]);
            }
        }

        RecordDynamic([setup_blends](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorBlendEquationEXT(0, setup_blends);
        });
    }
}

void RasterizerVulkan::UpdateColorWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchColorMask()) {
        return;
    }
    std::array<VkBool32, Maxwell::NumRenderTargets> setup_enables{};
    for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
        const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
        setup_enables[index] = (mask.R || mask.G || mask.B || mask.A) ? VK_TRUE : VK_FALSE;
    }
    RecordDynamic([setup_enables](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetColorWriteEnableEXT(setup_enables);
    });
}

void RasterizerVulkan::UpdateStencilTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilTestEnable()) {
        return;
    }
    RecordDynamic([enable = regs.stencil_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetStencilTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateVertexInput(Tegra::Engines::Maxwell3D::Regs& regs) {
    auto& dirty{DrawEngine().dirty.flags};
    const bool record = dirty[Dirty::VertexInput] || dirty[VideoCommon::Dirty::VertexBuffers];
    if (!record && !verify_vertex_input) {
        return;
    }

    const u32 max_attributes =
        static_cast<u32>(std::min<size_t>(Maxwell::NumVertexAttributes,
                                          device.GetMaxVertexInputAttributes()));
    const u32 max_bindings =
        static_cast<u32>(std::min<size_t>(Maxwell::NumVertexArrays,
                                          device.GetMaxVertexInputBindings()));
    const auto same_state = [](const VertexInputState& lhs, const VertexInputState& rhs) {
        return std::ranges::equal(lhs.bindings, rhs.bindings,
                                  [](const auto& a, const auto& b) {
                                      return a.binding == b.binding && a.stride == b.stride &&
                                             a.inputRate == b.inputRate && a.divisor == b.divisor;
                                  }) &&
               std::ranges::equal(lhs.attributes, rhs.attributes, [](const auto& a, const auto& b) {
                   return a.location == b.location && a.binding == b.binding &&
                          a.format == b.format && a.offset == b.offset;
               });
    };

    if (!record) {
        // EDEN_NX2_FAST_DRAW_VERIFY: no dirty flag asked for it, so the vertex input recorded
        // last must be what the registers give.
        VertexInputState state;
        BuildVertexInput(regs, max_attributes, max_bindings, state);
        if (vertex_input_recorded && !same_state(state, recorded_vertex_input)) {
            ++fast_draw_stats.vertex_input_mismatches;
        }
        return;
    }
    dirty[Dirty::VertexInput] = false;
    for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
        dirty[Dirty::VertexAttribute0 + index] = false;
    }
    for (size_t index = 0; index < Maxwell::NumVertexArrays; ++index) {
        dirty[Dirty::VertexBinding0 + index] = false;
    }

    // The state these registers built before, recorded by reference instead of built and copied
    // into the command again (jpkto/Lichii e7581fa9f6): converting every attribute's format and
    // copying both arrays into the command was ~0.5 ms a frame of Mario Kart World's back.
    const VideoCommon::VertexInputKey key = VertexInputKeyOf(regs, max_attributes, max_bindings);
    if (RenderVerifyEvery() != 0) {
        recorded_input_key = key;
        recorded_input_tick = scheduler.CurrentTick();
    }
    const u64 hash = key.Hash();
    std::shared_ptr<const VertexInputState> state = vertex_input_cache.Find(key, hash);
    if (!state) {
        auto built = std::make_shared<VertexInputState>();
        BuildVertexInput(regs, max_attributes, max_bindings, *built);
        state = std::move(built);
        vertex_input_cache.Insert(key, hash, state);
    } else if (fast_draw_verify) {
        // The key must hold everything the build reads.
        VertexInputState built;
        BuildVertexInput(regs, max_attributes, max_bindings, built);
        if (!same_state(built, *state)) {
            ++fast_draw_stats.vertex_input_mismatches;
        }
    }
    if (fast_draw_verify) {
        recorded_vertex_input = *state;
        vertex_input_recorded = true;
    }

    RecordDynamic([input = std::move(state)](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetVertexInputEXT(input->bindings, input->attributes);
    });
}

VideoCommon::VertexInputKey RasterizerVulkan::VertexInputKeyOf(
    const Tegra::Engines::Maxwell3D::Regs& regs, u32 max_attributes, u32 max_bindings) const {
    VideoCommon::VertexInputKey key{
        .num_attributes = max_attributes,
        .num_bindings = max_bindings,
    };
    for (u32 index = 0; index < max_attributes; ++index) {
        key.attributes[index] = regs.vertex_attrib_format[index].hex;
    }
    for (u32 binding = 0; binding < max_bindings; ++binding) {
        const auto& stream{regs.vertex_streams[binding]};
        key.strides[binding] = stream.stride.Value();
        key.divisors[binding] = stream.frequency;
        key.instanced |= (regs.vertex_stream_instances.IsInstancingEnabled(binding) ? 1U : 0U)
                         << binding;
        key.repacked |= (buffer_cache_runtime.IsVertexRepacked(binding) ? 1U : 0U) << binding;
    }
    return key;
}

void RasterizerVulkan::BuildVertexInput(const Tegra::Engines::Maxwell3D::Regs& regs,
                                        u32 max_attributes, u32 max_bindings,
                                        VertexInputState& state) {
    for (u32 index = 0; index < max_attributes; ++index) {
        const Maxwell::VertexAttribute attribute{regs.vertex_attrib_format[index]};
        const u32 binding{attribute.buffer};
        if (attribute.constant || binding >= max_bindings) {
            continue;
        }
        state.attributes.push_back({
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
            .pNext = nullptr,
            .location = index,
            .binding = binding,
            .format = MaxwellToVK::VertexFormat(device, attribute.type, attribute.size),
            .offset = attribute.offset +
                      (buffer_cache_runtime.IsVertexRepacked(binding) && attribute.offset >= 65 ? 3 : 0),
        });
    }

    for (u32 binding = 0; binding < max_bindings; ++binding) {
        const auto& input_binding{regs.vertex_streams[binding]};
        const bool is_instanced{regs.vertex_stream_instances.IsInstancingEnabled(binding)};
        state.bindings.push_back({
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
            .pNext = nullptr,
            .binding = binding,
            .stride = buffer_cache_runtime.IsVertexRepacked(binding) ? 72U : input_binding.stride.Value(),
            .inputRate = is_instanced ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX,
            .divisor = is_instanced ? input_binding.frequency : 1,
        });
    }
}

void RasterizerVulkan::InitializeChannel(Tegra::Control::ChannelState& channel) {
    RENDER_LEND(InitializeChannel(channel))
    CreateChannel(channel);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.CreateChannel(channel);
        buffer_cache.CreateChannel(channel);
    }
    pipeline_cache.CreateChannel(channel);
    query_cache.CreateChannel(channel);
    state_tracker.SetupTables(channel);
}

void RasterizerVulkan::BindChannel(Tegra::Control::ChannelState& channel) {
    RENDER_LEND(BindChannel(channel))
    const s32 channel_id = channel.bind_id;
    BindToChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.BindToChannel(channel_id);
        buffer_cache.BindToChannel(channel_id);
    }
    pipeline_cache.BindToChannel(channel_id);
    query_cache.BindToChannel(channel_id);
    state_tracker.ChangeChannel(channel);
    state_tracker.InvalidateState();
    bound_channel_id = channel_id;
}

void RasterizerVulkan::ReleaseChannel(s32 channel_id) {
    RENDER_LEND(ReleaseChannel(channel_id))
    if (channel_id == bound_channel_id) {
        state_tracker.ClearChannel();
    }
    EraseChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.EraseChannel(channel_id);
        buffer_cache.EraseChannel(channel_id);
    }
    pipeline_cache.EraseChannel(channel_id);
    query_cache.EraseChannel(channel_id);
}

} // namespace Vulkan
