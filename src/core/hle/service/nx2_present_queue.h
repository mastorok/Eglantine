// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include "common/common_types.h"
#include "common/logging.h"
#include "core/core_timing.h"
#include "video_core/host1x/syncpoint_manager.h"

namespace Service {

/**
 * Release fences for presented native buffers. A buffer is released, in present order, once its
 * display deadline has passed and the renderer has composited it: a reached GPU fence only says
 * the game's rendering finished, while the composite request is still queued and reads the image
 * later on the GPU thread.
 */
class Nx2PresentReleaseQueue : public std::enable_shared_from_this<Nx2PresentReleaseQueue> {
public:
    // The service advertises fifteen buffer slots.
    static constexpr size_t Capacity = 15;

    /// Starts an empty queue on an already allocated, session-owned release syncpoint.
    Nx2PresentReleaseQueue(Core::Timing::CoreTiming& timing_,
                           Tegra::Host1x::SyncpointManager& syncpoints_, u32 release_id_)
        : timing{timing_}, syncpoints{syncpoints_}, release_id{release_id_},
          next_value{syncpoints.GetGuestSyncpointValue(release_id)}, target{next_value},
          drained{next_value} {
        timer = Core::Timing::CreateEvent(
            "nx2:vi2 buffer release",
            [this](s64, std::chrono::nanoseconds) -> std::optional<std::chrono::nanoseconds> {
                std::unique_lock lock{mutex};
                timer_armed = false;
                ReleaseReady();
                Drain(lock);
                return std::nullopt;
            });
    }

    ~Nx2PresentReleaseQueue() {
        Stop();
    }

    /// Stops the timer before waiting for its callback, then wakes outstanding release waits.
    /// Nothing signals the syncpoint after this returns, so its owner may free it.
    void Stop() {
        {
            std::scoped_lock lock{mutex};
            if (std::exchange(stopped, true)) {
                return;
            }
        }
        timing.UnscheduleEvent(timer);
        std::unique_lock lock{mutex};
        if (!pending.empty()) {
            target = pending.back().value;
            pending.clear();
        }
        if (draining) {
            drain_done.wait(lock, [this] { return !draining; });
        } else {
            Drain(lock);
        }
    }

    /// How long the newest pending release still waits for its deadline (zero when none does).
    /// A present holds this long before queueing, so the game runs at most one frame ahead of the
    /// display however many buffer slots it cycles through.
    std::chrono::nanoseconds RunAhead() {
        std::scoped_lock lock{mutex};
        if (pending.empty()) {
            return {};
        }
        return (std::max)(std::chrono::nanoseconds{0},
                          pending.back().deadline - timing.GetGlobalTimeNs());
    }

    /// Queues a release at the CoreTiming time `deadline`, without blocking. Returns the fence
    /// value, or nothing when every slot is pending.
    std::optional<u32> Queue(std::chrono::nanoseconds deadline) {
        std::scoped_lock lock{mutex};
        if (stopped || pending.size() == Capacity) {
            return std::nullopt;
        }
        pending.push_back({deadline, ++next_value, false});
        return next_value;
    }

    /// The renderer has composited the buffer released at `value`.
    void Composited(u32 value) {
        std::unique_lock lock{mutex};
        if (stopped) {
            return;
        }
        const auto entry = std::ranges::find(pending, value, &PendingRelease::value);
        if (entry == pending.end()) {
            return;
        }
        entry->composited = true;
        ReleaseReady();
        Drain(lock);
    }

    /// For GPU::RequestComposite, which may run it on the GPU thread after this queue is gone.
    std::function<void()> OnComposited(u32 value) {
        return [queue = weak_from_this(), value] {
            if (const auto shared = queue.lock()) {
                shared->Composited(value);
            }
        };
    }

private:
    struct PendingRelease {
        std::chrono::nanoseconds deadline;
        u32 value;
        bool composited;
    };

    /**
     * Signals dequeued releases up to `target`, in order, from a single thread at a time. The
     * increments run outside the lock: each runs the syncpoint's due guest actions on this thread,
     * and one may composite and call Composited. Any other caller, that re-entrant one included,
     * only advances `target` and leaves the signal to the thread already draining.
     */
    void Drain(std::unique_lock<std::mutex>& lock) {
        if (draining) {
            return;
        }
        draining = true;
        while (drained != target) {
            const u32 last = target;
            lock.unlock();
            for (u32 value = drained + 1; static_cast<s32>(value - last) <= 0; ++value) {
                // Without chasing a counter that has already passed the release.
                if (static_cast<s32>(value - syncpoints.GetGuestSyncpointValue(release_id)) > 0) {
                    syncpoints.IncrementGuest(release_id);
                }
            }
            lock.lock();
            drained = last;
        }
        draining = false;
        drain_done.notify_all();
    }

    /// Under the lock: dequeues ready buffers from the front into `target`, arming the timer for a
    /// deadline.
    void ReleaseReady() {
        if (stopped) {
            return;
        }
        while (!pending.empty()) {
            const auto& entry = pending.front();
            if (!entry.composited) {
                break;
            }
            if (timing.GetGlobalTimeNs() < entry.deadline) {
                if (!timer_armed) {
                    timer_armed = true;
                    timing.ScheduleEvent(entry.deadline, timer, true);
                }
                break;
            }
            target = entry.value;
            pending.pop_front();
        }
    }

    Core::Timing::CoreTiming& timing;
    Tegra::Host1x::SyncpointManager& syncpoints;
    const u32 release_id;
    u32 next_value;
    u32 target;  ///< Newest dequeued release; releases are consecutive and dequeued in order
    u32 drained; ///< Newest release Drain has signalled
    std::mutex mutex;
    std::condition_variable drain_done;
    bool draining{};
    bool stopped{};
    bool timer_armed{};
    std::deque<PendingRelease> pending;
    std::shared_ptr<Core::Timing::EventType> timer;
};

/**
 * A session's release fences: the syncpoint is allocated on the first queued present and freed
 * after the queue has woken its waits. Every failure leaves the caller on its blocking pacing.
 */
class Nx2PresentReleases {
public:
    using Allocate = std::function<std::optional<u32>()>;
    using Free = std::function<void(u32)>;

    struct Release {
        u32 id;
        u32 value;
        std::function<void()> on_composited;
    };

    Nx2PresentReleases(Core::Timing::CoreTiming& timing_,
                       Tegra::Host1x::SyncpointManager& syncpoints_, Allocate allocate_,
                       Free free_)
        : timing{timing_}, syncpoints{syncpoints_}, allocate{std::move(allocate_)},
          free{std::move(free_)} {}

    ~Nx2PresentReleases() {
        if (queue) {
            // A composite callback may still hold the queue; it does nothing once stopped.
            queue->Stop();
            free(id);
        }
    }

    /// See Nx2PresentReleaseQueue::RunAhead.
    std::chrono::nanoseconds RunAhead() {
        return queue ? queue->RunAhead() : std::chrono::nanoseconds{};
    }

    /// Queues a release `wait` from now (zero or negative: at once). Returns nothing when no
    /// syncpoint can be allocated or every slot is pending.
    std::optional<Release> Queue(std::chrono::nanoseconds wait) {
        if (!queue) {
            const auto allocated = allocate();
            if (!allocated) {
                if (!std::exchange(allocation_logged, true)) {
                    LOG_WARNING(Service, "nx2:vi2 release fences unavailable: no free syncpoint");
                }
                return std::nullopt;
            }
            id = *allocated;
            queue = std::make_shared<Nx2PresentReleaseQueue>(timing, syncpoints, id);
        }
        const auto deadline =
            timing.GetGlobalTimeNs() + (std::max)(std::chrono::nanoseconds{0}, wait);
        const auto value = queue->Queue(deadline);
        if (!value) {
            return std::nullopt;
        }
        return Release{id, *value, queue->OnComposited(*value)};
    }

private:
    Core::Timing::CoreTiming& timing;
    Tegra::Host1x::SyncpointManager& syncpoints;
    Allocate allocate;
    Free free;
    std::shared_ptr<Nx2PresentReleaseQueue> queue;
    u32 id{};
    bool allocation_logged{};
};

} // namespace Service
