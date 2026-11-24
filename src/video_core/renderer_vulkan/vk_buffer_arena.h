// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <optional>
#include <source_location>
#include <span>

#include "common/common_types.h"
#include "video_core/buffer_cache/command_arena.h"
#include "video_core/gpu_split.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// Every Vulkan command the buffer cache runtime records goes through here. With
/// EDEN_NX2_BUFFER_ARENA=1 the scheduler's owner queues them by value in a command arena and
/// plays them at its next use of the scheduler (Scheduler::JoinRecorders), so they keep their
/// place in the command stream while the buffer side runs without the scheduler. Otherwise each
/// command runs at once. Play is the only copy of their Vulkan code.
/// Design after Fritchii's memory stage recorder and player (c351072099, 0225a76fbb).
class BufferArena {
public:
    enum class Barrier : u8 { PreCopy, PostCopy, HostRead };
    enum class CopyMode : u8 { Plain, Batched, UploadBuffer };

    explicit BufferArena(const Device& device, Scheduler& scheduler);
    ~BufferArena();

    BufferArena(const BufferArena&) = delete;
    BufferArena& operator=(const BufferArena&) = delete;

    void OutsideRenderPass(std::source_location location = std::source_location::current());
    void Copy(VkBuffer src, VkBuffer dst, std::span<const VkBufferCopy> copies, CopyMode mode,
              bool barrier);
    void CopyBarrier(Barrier barrier);
    void Clear(VkBuffer buffer, u32 offset, size_t size, u32 value);
    /// Zeroes [offset, +size) and makes it visible to dst_access.
    void Fill(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, VkAccessFlags dst_access);
    /// Copies a quad index table from staging and makes it visible to index reads.
    void IndexTable(VkBuffer src, VkDeviceSize src_offset, VkBuffer dst, VkDeviceSize size);
    /// tracked: skipped while the same buffer is still bound in the command buffer.
    void BindIndex(VkBuffer buffer, VkDeviceSize offset, VkIndexType type, bool tracked);
    void BindVertex(u32 index, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                    VkDeviceSize stride, bool extended);
    void BindVertices(u32 first, std::span<const VkBuffer> buffers, std::span<const u64> offsets,
                      std::span<const u64> sizes, std::span<const u64> strides, bool extended);
    void BindTransformFeedback(u32 first, std::span<const VkBuffer> buffers,
                               std::span<const u64> offsets, std::span<const u64> sizes);

    /// The stamp of what is recorded now (buffer usage, the buffer staging pool).
    [[nodiscard]] u64 Now() const noexcept;
    [[nodiscard]] bool IsFree(u64 stamp) const noexcept;
    /// The tick a stamp went in, or a later one; nullopt while it is not mapped.
    [[nodiscard]] std::optional<u64> TickOf(u64 stamp) const noexcept;
    /// With the arena on, the buffer staging pool's clock; null otherwise.
    [[nodiscard]] const VideoCommon::ArenaClock* Clock() const noexcept {
        return enabled ? &clock : nullptr;
    }

    /// Plays the queued commands. Scheduler entry points call PlayPending.
    void Play();

    /// EDEN_NX2_RENDER_UNLEND: while one lives, the back's commands go to `into` by value, the
    /// buffer half of a draw GPURender records.
    struct Collect {
        explicit Collect(VideoCommon::CommandArena& into) noexcept {
            collecting = &into;
            collect_stamp = 0;
        }
        ~Collect() {
            collecting = nullptr;
        }
    };
    [[nodiscard]] static bool Collecting() noexcept {
        return collecting != nullptr;
    }
    /// The back, posting what it collected: the stamp GPURender maps when it plays it.
    [[nodiscard]] u64 CloseCollected() noexcept {
        const u64 stamp = clock.Close();
        moved_collects += collect_stamp != 0 && collect_stamp != stamp ? 1 : 0;
        collect_stamp = 0;
        return stamp;
    }
    /// GPURender: plays what the back collected for a draw, stamped `stamp`.
    void PlayCollected(const VideoCommon::CommandArena& commands, u64 stamp);
    /// Once a frame: frees the stamp of commands that ran without the arena.
    void TickFrame();

    /// The scheduler's entry points: plays what this thread queued.
    static void PlayPending() {
        if (BufferArena* const arena = pending) [[unlikely]] {
            arena->Play();
        }
    }

    /// EDEN_NX2_SPLIT_CHECK: Scheduler::Record and RecordUpload report themselves here; a record
    /// in a buffer side scope that is neither an arena play nor a declared immediate is counted
    /// (must be 0).
    static void CheckRecord(const char* where, std::source_location location);

    /// Marks code whose scheduler use belongs to the buffer side.
    struct SideScope {
        SideScope() noexcept
            : tracking{checks_records.load(std::memory_order_relaxed) &&
                       VideoCommon::GpuSplit::check_owner} {
            if (tracking) {
                ++side_depth;
            }
        }
        ~SideScope() {
            if (tracking) {
                --side_depth;
            }
        }

    private:
        const bool tracking;
    };
    /// A buffer side path that records or syncs on the scheduler directly (compute passes, host
    /// read syncs, list word patches): counted, and lent or posted when GPURender owns the
    /// scheduler.
    struct ImmediateScope {
        explicit ImmediateScope(BufferArena& arena_) noexcept;
        ~ImmediateScope() {
            if (enabled) {
                --immediate_depth;
            }
        }

    private:
        const bool enabled;
    };

private:
    enum class Op : u16 {
        OutsideRenderPass,
        Copy,
        CopyBarrier,
        Clear,
        Fill,
        IndexTable,
        BindIndex,
        BindVertex,
        BindVertices,
        BindTransformFeedback,
    };
    struct OutsideBody {
        std::source_location location;
    };
    struct CopyBody {
        VkBuffer src;
        VkBuffer dst;
        CopyMode mode;
        bool barrier;
    };
    struct BarrierBody {
        Barrier barrier;
    };
    struct ClearBody {
        VkBuffer buffer;
        u64 size;
        u32 offset;
        u32 value;
    };
    struct FillBody {
        VkBuffer buffer;
        VkDeviceSize offset;
        VkDeviceSize size;
        VkAccessFlags dst_access;
    };
    struct IndexTableBody {
        VkBuffer src;
        VkBuffer dst;
        VkDeviceSize src_offset;
        VkDeviceSize size;
    };
    struct BindIndexBody {
        VkBuffer buffer;
        VkDeviceSize offset;
        VkIndexType type;
        bool tracked;
    };
    struct BindVertexBody {
        VkBuffer buffer;
        VkDeviceSize offset;
        VkDeviceSize size;
        VkDeviceSize stride;
        u32 index;
        bool extended;
    };
    struct VertexBinding {
        VkBuffer buffer;
        VkDeviceSize offset;
        VkDeviceSize size;
        VkDeviceSize stride;
    };
    struct BindVerticesBody {
        u32 first;
        bool extended;
    };
    struct BindFeedbackBody {
        u32 first;
    };

    template <typename Body, typename E = u8>
    void Submit(Op op, const Body& body, std::span<const E> array = {});

    void Exec(const OutsideBody& body, std::span<const u8>);
    void Exec(const CopyBody& body, std::span<const VkBufferCopy> copies);
    void Exec(const BarrierBody& body, std::span<const u8>);
    void Exec(const ClearBody& body, std::span<const u8>);
    void Exec(const FillBody& body, std::span<const u8>);
    void Exec(const IndexTableBody& body, std::span<const u8>);
    void Exec(const BindIndexBody& body, std::span<const u8>);
    void Exec(const BindVertexBody& body, std::span<const u8>);
    void Exec(const BindVerticesBody& body, std::span<const VertexBinding> bindings);
    void Exec(const BindFeedbackBody& body, std::span<const VertexBinding> bindings);

    void PlayRecords(const VideoCommon::CommandArena& commands);

    static inline thread_local BufferArena* pending{};
    static inline thread_local VideoCommon::CommandArena* collecting{};
    /// The stamp of the open collect's first record (Now()), 0 before it.
    static inline thread_local u64 collect_stamp{};
    static inline thread_local u32 side_depth{};
    static inline thread_local u32 immediate_depth{};
    static inline std::atomic<bool> checks_records{};
    static inline thread_local bool playing{};

    const Device& device;
    Scheduler& scheduler;
    const bool enabled;
    VideoCommon::CommandArena queued;
    VideoCommon::CommandArena play;
    VideoCommon::ArenaClock clock;

    /// The index buffer bound last, at scheduler recording epoch bound_index_epoch (PR #161).
    VkBuffer bound_index_buffer = VK_NULL_HANDLE;
    VkDeviceSize bound_index_offset = 0;
    VkIndexType bound_index_type = VK_INDEX_TYPE_UINT32;
    u64 bound_index_epoch = 0;
    bool bound_index_valid = false;

    u64 plays{};
    u64 played_records{};
    u64 direct_records{};
    std::atomic<u64> immediates{};
    /// EDEN_NX2_RENDER_UNLEND: the back takes stamps while GPURender plays, so GPURender takes
    /// them only inside a lent call, the back parked (must be 0 outside).
    u64 unowned_stamps{};
    /// Collected batches whose records were stamped other than CloseCollected returns: another
    /// thread took a stamp while the back collected (must be 0).
    u64 moved_collects{};
    bool unlend_seen{};
    void NoteStamp() noexcept {
        if (VideoCommon::GpuSplit::UnlendEnabled()) {
            unlend_seen = true;
            unowned_stamps += VideoCommon::GpuSplit::PlayingUnlent() ? 1 : 0;
        }
    }
    static inline std::atomic<u64> outside_records{};
};

} // namespace Vulkan
