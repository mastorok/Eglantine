// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>

#include "common/logging.h"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_perf.h"
#include "video_core/renderer_vulkan/vk_buffer_arena.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {
namespace {
constexpr VkMemoryBarrier COPY_READ_BARRIER{
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .pNext = nullptr,
    .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
};
constexpr VkMemoryBarrier COPY_WRITE_BARRIER{
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .pNext = nullptr,
    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
};
constexpr VkMemoryBarrier POST_COPY_BARRIER{
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .pNext = nullptr,
    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
    .dstAccessMask =
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT,
};
using VertexArray = boost::container::static_vector<VkDeviceSize, VideoCommon::NUM_VERTEX_BUFFERS>;
using HandleArray = boost::container::static_vector<VkBuffer, VideoCommon::NUM_VERTEX_BUFFERS>;

void RecordVertices(Scheduler& scheduler, u32 first, bool extended, HandleArray&& buffers,
                    VertexArray&& offsets, VertexArray&& sizes, VertexArray&& strides) {
    if (extended) {
        scheduler.Record([first, buffers = std::move(buffers), offsets = std::move(offsets),
                          sizes = std::move(sizes), strides = std::move(strides)](
                             vk::CommandBuffer cmdbuf) {
            cmdbuf.BindVertexBuffers2EXT(first, static_cast<u32>(buffers.size()), buffers.data(),
                                         offsets.data(), sizes.data(), strides.data());
        });
        return;
    }
    scheduler.Record([first, buffers = std::move(buffers), offsets = std::move(offsets)](
                         vk::CommandBuffer cmdbuf) {
        cmdbuf.BindVertexBuffers(first, static_cast<u32>(buffers.size()), buffers.data(),
                                 offsets.data());
    });
}

void RecordFeedback(Scheduler& scheduler, u32 first, HandleArray&& buffers, VertexArray&& offsets,
                    VertexArray&& sizes) {
    scheduler.Record([first, buffers = std::move(buffers), offsets = std::move(offsets),
                      sizes = std::move(sizes)](vk::CommandBuffer cmdbuf) {
        cmdbuf.BindTransformFeedbackBuffersEXT(first, static_cast<u32>(buffers.size()),
                                               buffers.data(), offsets.data(), sizes.data());
    });
}
} // Anonymous namespace

BufferArena::BufferArena(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_}, enabled{scheduler_.BufferArenaEnabled()} {
    checks_records.store(enabled, std::memory_order_relaxed);
    if (enabled) {
        LOG_INFO(Render_Vulkan, "EDEN_NX2_BUFFER_ARENA: buffer commands queued by value");
    }
}

BufferArena::~BufferArena() {
    if (pending == this) {
        pending = nullptr;
    }
    if (!enabled) {
        return;
    }
    LOG_INFO(Render_Vulkan,
             "Buffer arena: {} records in {} plays, {} run at once off the scheduler's owner, {} "
             "immediate buffer side paths; {} buffer side records outside the arena (must be 0)",
             played_records, plays, direct_records, immediates.load(std::memory_order_relaxed),
             outside_records.load(std::memory_order_relaxed));
    if (unlend_seen) {
        LOG_INFO(Render_Vulkan,
                 "Buffer arena clock: must be 0: {} stamps taken on GPURender outside a lent call, "
                 "{} collected batches whose stamp moved before the post",
                 unowned_stamps, moved_collects);
    }
}

template <typename Body, typename E>
void BufferArena::Submit(Op op, const Body& body, std::span<const E> array) {
    if (enabled) {
        if (collecting != nullptr) {
            if (collecting->Empty()) {
                collect_stamp = clock.Now();
            }
            collecting->Push(static_cast<u16>(op), body, array);
            return;
        }
        if (VideoCommon::GpuSplit::OwnsScheduler()) {
            queued.Push(static_cast<u16>(op), body, array);
            pending = this;
            return;
        }
        ++direct_records;
    }
    Exec(body, array);
}

void BufferArena::OutsideRenderPass(std::source_location location) {
    Submit(Op::OutsideRenderPass, OutsideBody{location});
}

void BufferArena::Copy(VkBuffer src, VkBuffer dst, std::span<const VkBufferCopy> copies,
                       CopyMode mode, bool barrier) {
    Submit(Op::Copy, CopyBody{src, dst, mode, barrier}, copies);
}

void BufferArena::CopyBarrier(Barrier barrier) {
    Submit(Op::CopyBarrier, BarrierBody{barrier});
}

void BufferArena::Clear(VkBuffer buffer, u32 offset, size_t size, u32 value) {
    Submit(Op::Clear, ClearBody{buffer, u64{size}, offset, value});
}

void BufferArena::Fill(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                       VkAccessFlags dst_access) {
    Submit(Op::Fill, FillBody{buffer, offset, size, dst_access});
}

void BufferArena::IndexTable(VkBuffer src, VkDeviceSize src_offset, VkBuffer dst,
                             VkDeviceSize size) {
    Submit(Op::IndexTable, IndexTableBody{src, dst, src_offset, size});
}

void BufferArena::BindIndex(VkBuffer buffer, VkDeviceSize offset, VkIndexType type,
                            bool tracked) {
    Submit(Op::BindIndex, BindIndexBody{buffer, offset, type, tracked});
}

void BufferArena::BindVertex(u32 index, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                             VkDeviceSize stride, bool extended) {
    Submit(Op::BindVertex, BindVertexBody{buffer, offset, size, stride, index, extended});
}

void BufferArena::BindVertices(u32 first, std::span<const VkBuffer> buffers,
                               std::span<const u64> offsets, std::span<const u64> sizes,
                               std::span<const u64> strides, bool extended) {
    if (!enabled) {
        const size_t count = buffers.size();
        offsets = offsets.first(count);
        sizes = sizes.first(count);
        strides = strides.first(count);
        RecordVertices(scheduler, first, extended, HandleArray{buffers.begin(), buffers.end()},
                       VertexArray{offsets.begin(), offsets.end()},
                       extended ? VertexArray{sizes.begin(), sizes.end()} : VertexArray{},
                       extended ? VertexArray{strides.begin(), strides.end()} : VertexArray{});
        return;
    }
    boost::container::static_vector<VertexBinding, VideoCommon::NUM_VERTEX_BUFFERS> bindings;
    for (size_t i = 0; i < buffers.size(); ++i) {
        bindings.push_back({buffers[i], offsets[i], sizes[i], strides[i]});
    }
    Submit(Op::BindVertices, BindVerticesBody{first, extended},
           std::span<const VertexBinding>{bindings});
}

void BufferArena::BindTransformFeedback(u32 first, std::span<const VkBuffer> buffers,
                                        std::span<const u64> offsets, std::span<const u64> sizes) {
    if (!enabled) {
        offsets = offsets.first(buffers.size());
        sizes = sizes.first(buffers.size());
        RecordFeedback(scheduler, first, HandleArray{buffers.begin(), buffers.end()},
                       VertexArray{offsets.begin(), offsets.end()},
                       VertexArray{sizes.begin(), sizes.end()});
        return;
    }
    boost::container::static_vector<VertexBinding, VideoCommon::NUM_VERTEX_BUFFERS> bindings;
    for (size_t i = 0; i < buffers.size(); ++i) {
        bindings.push_back({buffers[i], offsets[i], sizes[i], 0});
    }
    Submit(Op::BindTransformFeedback, BindFeedbackBody{first}, std::span<const VertexBinding>{bindings});
}

u64 BufferArena::Now() const noexcept {
    return enabled ? clock.Now() : scheduler.CurrentTick();
}

bool BufferArena::IsFree(u64 stamp) const noexcept {
    if (!enabled) {
        return scheduler.IsFree(stamp);
    }
    return clock.IsFree(stamp, [this](u64 tick) { return scheduler.IsFree(tick); });
}

std::optional<u64> BufferArena::TickOf(u64 stamp) const noexcept {
    if (!enabled) {
        return stamp;
    }
    return clock.TickOf(stamp);
}

void BufferArena::Play() {
    if (pending == this) {
        pending = nullptr;
    }
    if (queued.Empty()) {
        return;
    }
    play.Swap(queued);
    NoteStamp();
    clock.Map(scheduler.CurrentTick());
    PlayRecords(play);
    play.Clear();
}

void BufferArena::PlayCollected(const VideoCommon::CommandArena& commands, u64 stamp) {
    clock.Map(stamp, scheduler.CurrentTick());
    if (!commands.Empty()) {
        PlayRecords(commands);
    }
}

void BufferArena::PlayRecords(const VideoCommon::CommandArena& commands) {
    ++plays;
    played_records += commands.Records();
    playing = true;
    using Arena = VideoCommon::CommandArena;
    commands.ForEach([this](u16 kind, const u8* record) {
        switch (static_cast<Op>(kind)) {
        case Op::OutsideRenderPass:
            Exec(Arena::Body<OutsideBody>(record), {});
            break;
        case Op::Copy:
            Exec(Arena::Body<CopyBody>(record), Arena::Array<CopyBody, VkBufferCopy>(record));
            break;
        case Op::CopyBarrier:
            Exec(Arena::Body<BarrierBody>(record), {});
            break;
        case Op::Clear:
            Exec(Arena::Body<ClearBody>(record), {});
            break;
        case Op::Fill:
            Exec(Arena::Body<FillBody>(record), {});
            break;
        case Op::IndexTable:
            Exec(Arena::Body<IndexTableBody>(record), {});
            break;
        case Op::BindIndex:
            Exec(Arena::Body<BindIndexBody>(record), {});
            break;
        case Op::BindVertex:
            Exec(Arena::Body<BindVertexBody>(record), {});
            break;
        case Op::BindVertices:
            Exec(Arena::Body<BindVerticesBody>(record),
                 Arena::Array<BindVerticesBody, VertexBinding>(record));
            break;
        case Op::BindTransformFeedback:
            Exec(Arena::Body<BindFeedbackBody>(record),
                 Arena::Array<BindFeedbackBody, VertexBinding>(record));
            break;
        }
    });
    playing = false;
}

void BufferArena::TickFrame() {
    if (enabled && queued.Empty()) {
        NoteStamp();
        clock.Map(scheduler.CurrentTick());
    }
}

void BufferArena::CheckRecord(const char* where, std::source_location location) {
    if (side_depth == 0 || playing || immediate_depth != 0 ||
        !checks_records.load(std::memory_order_relaxed)) {
        return;
    }
    if (outside_records.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_ERROR(Render_Vulkan, "{} from the buffer side outside the arena at {}:{}", where,
                  location.file_name(), location.line());
    }
}

BufferArena::ImmediateScope::ImmediateScope(BufferArena& arena) noexcept : enabled{arena.enabled} {
    if (enabled) {
        ++immediate_depth;
        arena.immediates.fetch_add(1, std::memory_order_relaxed);
    }
}

void BufferArena::Exec(const OutsideBody& body, std::span<const u8>) {
    scheduler.RequestOutsideRenderPassOperationContext(body.location);
}

void BufferArena::Exec(const CopyBody& body, std::span<const VkBufferCopy> span) {
    if (body.mode == CopyMode::Batched) {
        scheduler.RecordUpload(body.src, body.dst, span);
        return;
    }
    boost::container::small_vector<VkBufferCopy, 8> copies(span.begin(), span.end());
    if (body.mode == CopyMode::UploadBuffer) {
        scheduler.RecordWithUploadBuffer(
            [src = body.src, dst = body.dst, copies = std::move(copies)](
                vk::CommandBuffer, vk::CommandBuffer upload_cmdbuf) {
                upload_cmdbuf.CopyBuffer(src, dst, copies);
            });
        return;
    }
    scheduler.Record([src = body.src, dst = body.dst, copies = std::move(copies),
                      barrier = body.barrier](vk::CommandBuffer cmdbuf) {
        if (barrier) {
            cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, COPY_READ_BARRIER);
        }
        cmdbuf.CopyBuffer(src, dst, copies);
        if (barrier) {
            // Later transfers too: a reordered upload of the same bytes in a later submission
            // has no barrier of its own before it.
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, 0,
                                   COPY_WRITE_BARRIER);
        }
    });
}

void BufferArena::Exec(const BarrierBody& body, std::span<const u8>) {
    switch (body.barrier) {
    case Barrier::PreCopy:
        scheduler.Record([](vk::CommandBuffer cmdbuf) {
            cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, COPY_READ_BARRIER);
        });
        return;
    case Barrier::PostCopy:
        scheduler.Record([](vk::CommandBuffer cmdbuf) {
            // Copies bracketed explicitly need the same ordering as CopyBuffer(barrier = true),
            // and host reads as well, as the query cache's value syncs had (Barriers).
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER_HOST, 0,
                                   POST_COPY_BARRIER);
        });
        return;
    case Barrier::HostRead:
        scheduler.Record([](vk::CommandBuffer cmdbuf) {
            const VkMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                   0, barrier);
        });
        return;
    }
}

void BufferArena::Exec(const ClearBody& body, std::span<const u8>) {
    scheduler.Record([buffer = body.buffer, offset = body.offset, size = body.size,
                      value = body.value](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, COPY_READ_BARRIER);
        cmdbuf.FillBuffer(buffer, offset, size, value);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, vk::PIPELINE_STAGE_GRAPHICS_COMPUTE,
                               0, COPY_WRITE_BARRIER);
    });
}

void BufferArena::Exec(const FillBody& body, std::span<const u8>) {
    scheduler.Record([buffer = body.buffer, offset = body.offset, size = body.size,
                      dst_access = body.dst_access](vk::CommandBuffer cmdbuf) {
        cmdbuf.FillBuffer(buffer, offset, size, 0);
        const VkMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = dst_access,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               0, barrier);
    });
}

void BufferArena::Exec(const IndexTableBody& body, std::span<const u8>) {
    scheduler.Record([src = body.src, src_offset = body.src_offset, dst = body.dst,
                      size = body.size](vk::CommandBuffer cmdbuf) {
        const VkBufferCopy copy{
            .srcOffset = src_offset,
            .dstOffset = 0,
            .size = size,
        };
        const VkBufferMemoryBarrier write_barrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_INDEX_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = dst,
            .offset = 0,
            .size = size,
        };
        cmdbuf.CopyBuffer(src, dst, copy);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                               0, write_barrier);
    });
}

void BufferArena::Exec(const BindIndexBody& body, std::span<const u8>) {
    if (!body.tracked) {
        bound_index_valid = false;
    } else if (!VideoCommon::Nx2Perf::Verify()) {
        // The same index buffer is still bound in this command buffer (PR #161): skip the rebind.
        const u64 epoch = scheduler.RecordingEpoch();
        if (bound_index_valid && body.buffer == bound_index_buffer &&
            body.offset == bound_index_offset && body.type == bound_index_type &&
            epoch == bound_index_epoch) {
            return;
        }
        bound_index_valid = true;
        bound_index_buffer = body.buffer;
        bound_index_offset = body.offset;
        bound_index_type = body.type;
        bound_index_epoch = epoch;
    }
    scheduler.Record([buffer = body.buffer, offset = body.offset,
                      type = body.type](vk::CommandBuffer cmdbuf) {
        cmdbuf.BindIndexBuffer(buffer, offset, type);
    });
}

void BufferArena::Exec(const BindVertexBody& body, std::span<const u8>) {
    if (body.extended) {
        scheduler.Record([index = body.index, buffer = body.buffer, offset = body.offset,
                          size = body.size, stride = body.stride](vk::CommandBuffer cmdbuf) {
            const VkDeviceSize vk_offset = buffer != VK_NULL_HANDLE ? offset : 0;
            const VkDeviceSize vk_size = buffer != VK_NULL_HANDLE ? size : VK_WHOLE_SIZE;
            cmdbuf.BindVertexBuffers2EXT(index, 1, &buffer, &vk_offset, &vk_size, &stride);
        });
        return;
    }
    scheduler.Record([index = body.index, buffer = body.buffer,
                      offset = body.offset](vk::CommandBuffer cmdbuf) {
        cmdbuf.BindVertexBuffer(index, buffer, offset);
    });
}

void BufferArena::Exec(const BindVerticesBody& body, std::span<const VertexBinding> bindings) {
    HandleArray buffers;
    VertexArray offsets, sizes, strides;
    for (const VertexBinding& binding : bindings) {
        buffers.push_back(binding.buffer);
        offsets.push_back(binding.offset);
        sizes.push_back(binding.size);
        strides.push_back(binding.stride);
    }
    RecordVertices(scheduler, body.first, body.extended, std::move(buffers), std::move(offsets),
                   std::move(sizes), std::move(strides));
}

void BufferArena::Exec(const BindFeedbackBody& body, std::span<const VertexBinding> bindings) {
    HandleArray buffers;
    VertexArray offsets, sizes;
    for (const VertexBinding& binding : bindings) {
        buffers.push_back(binding.buffer);
        offsets.push_back(binding.offset);
        sizes.push_back(binding.size);
    }
    RecordFeedback(scheduler, body.first, std::move(buffers), std::move(offsets), std::move(sizes));
}

} // namespace Vulkan
