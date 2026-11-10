// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <string>

namespace Nx2Sampler {

// EDEN_NX2_SAMPLE=<thread names>: a sampling profiler for named host threads (Windows only).
// Names are comma separated; a name ending in '*' matches by prefix (CPUCore_*). From
// EDEN_NX2_SAMPLE_START seconds (default 60) for EDEN_NX2_SAMPLE_SECONDS (default 30), each
// thread is paused about 1000 times a second and its call stack recorded; the counts per
// function (where the thread is, and everything on its stack) go to EDEN_NX2_SAMPLE_OUT
// (default nx2_samples.txt).
//
// EDEN_NX2_STALL_MS=<n>: a watchdog that notices when the game's presents stop for n ms or more
// and, until the next present, takes the call stacks of the GPU, GPUBack, Vulkan, guest-core and
// worker threads every few ms. When the present comes it logs, per thread group, the most
// frequent stacks and the deltas of the counters below ("nx2 stall #k ..."). Nothing runs while
// the game keeps its pace.
//
// EDEN_NX2_FRAME_LOG=1: one line per present in <log dir>/nx2_frames.csv.
void StartFromEnvironment();

/// The guest state of one emulated core, read while its host thread is paused by the sampler.
struct GuestSample {
    std::uint64_t pc{};
    std::uint64_t lr{};
    std::uint64_t sp{};
    std::uint64_t fp{};
    std::uint64_t thread_id{}; ///< Guest thread running on the core
    /// Return addresses along the frame pointer chain (filled by the sampler), innermost first.
    std::array<std::uint64_t, 10> chain{};
    std::uint32_t chain_depth{};
};

/// Registered by the kernel: reads the guest state of core `core`. Called with that core's host
/// thread paused, so it must not wait for any lock; false when the core has nothing to report.
/// With it, the profile of a CPUCore_N thread also lists the guest addresses its time went to
/// (<out>.guest.txt, see EDEN_NX2_SAMPLE).
void SetGuestProbe(std::function<bool(unsigned core, GuestSample&)> probe);

/// Registered by the kernel: reads guest memory. The profile of the guest cores then also keeps the
/// code around their hottest addresses (<out>.guest_code.txt), to read without the game's files.
void SetGuestMemoryReader(std::function<bool(std::uint64_t address, void* data, std::size_t size)> reader);

/// Called by the kernel when a guest thread is created (svcCreateThread). With EDEN_NX2_SAMPLE,
/// the thread table (<out>.guest_threads.txt) lists them: entry point, argument, priority, core.
void NoteGuestThread(std::uint64_t id, std::uint64_t creator, std::uint64_t entry,
                     std::uint64_t arg, std::int32_t priority, std::int32_t core, std::uint64_t tls);

/// The guest thread whose service request the calling host thread is handling (nvdrv ioctls).
void SetClientThread(std::uint64_t id);

/// A GPFIFO submission with `entries` pushbuffer entries, credited to the client thread.
void NoteGpuSubmit(std::uint32_t entries);

/// EDEN_NX2_AUDIO_LOG (default on): an audio sink stream found no buffer to play ("underrun",
/// the output repeats its last sample = a click) with `queued` buffers waiting; logged with the
/// time on the same clock as the frame log (t_s), so a click can be matched to a stall.
void NoteAudioUnderrun(const char* stream, std::uint32_t queued);
/// The DSP renderer started a tick `gap_ms` after the previous one (nominally 5 ms), and the
/// previous one took `process_ms` to render.
void NoteAudioRenderGap(double gap_ms, double process_ms);

/// Samples the named threads for `seconds` now (blocking), writes the report to `out` (and
/// the folded stacks to `out`.folded) and returns the report.
std::string Sample(const char* names, double seconds, const std::string& out);

// Counters the renderer bumps, in the frame log and in each stall report.
inline std::atomic<std::uint64_t> g_astc_decodes{0};
inline std::atomic<std::uint64_t> g_pipelines_built{0};
inline std::atomic<std::uint64_t> g_pipeline_build_us{0};
// Sparse image binds (queue bind + wait for its fence), texture cache images made and deleted,
// and time spent in the texture cache's garbage collector.
inline std::atomic<std::uint64_t> g_sparse_binds{0};
inline std::atomic<std::uint64_t> g_sparse_bind_us{0};
// Part of g_sparse_bind_us up to the end of vkQueueBindSparse (the rest is the wait for its fence).
inline std::atomic<std::uint64_t> g_sparse_call_us{0};
inline std::atomic<std::uint64_t> g_images_created{0};
inline std::atomic<std::uint64_t> g_images_deleted{0};
inline std::atomic<std::uint64_t> g_gc_us{0};
// Why images are deleted, and what the collector sees (state, not counters: the last values).
inline std::atomic<std::uint64_t> g_del_gc{0};          // collector
inline std::atomic<std::uint64_t> g_del_gc_download{0}; // collector, after downloading the image
inline std::atomic<std::uint64_t> g_del_unmap{0};       // guest memory unmapped
inline std::atomic<std::uint64_t> g_del_join{0};        // replaced by an overlapping new image
inline std::atomic<std::uint64_t> g_created_replace{0}; // new images that replaced overlaps
inline std::atomic<std::uint64_t> g_gc_finish_us{0};    // GPU waits inside the collector
// Images created again at the same address and shape as one deleted earlier, by the time since
// that deletion in frames: thrash (soon again) against streaming (never seen before, or long ago).
inline std::atomic<std::uint64_t> g_recreated_30{0};    // under 30 frames
inline std::atomic<std::uint64_t> g_recreated_150{0};   // 30 to 149 frames
inline std::atomic<std::uint64_t> g_recreated_600{0};   // 150 to 599 frames
inline std::atomic<std::uint64_t> g_recreated_old{0};   // 600 frames or more
// Waits of the split's front thread for its back thread (writers: before reading a command list
// entry; drains: everything queued), and the back's waits for work.
inline std::atomic<std::uint64_t> g_writer_waits{0};
inline std::atomic<std::uint64_t> g_writer_wait_us{0};
inline std::atomic<std::uint64_t> g_drains{0};
inline std::atomic<std::uint64_t> g_drain_us{0};
inline std::atomic<std::uint64_t> g_back_idles{0};
inline std::atomic<std::uint64_t> g_back_idle_us{0};
// Of the drains: those of semaphore acquires (the puller) and of the macros' DMA reads.
inline std::atomic<std::uint64_t> g_sem_drains{0};
inline std::atomic<std::uint64_t> g_sem_drain_us{0};
inline std::atomic<std::uint64_t> g_dma_drains{0};
inline std::atomic<std::uint64_t> g_dma_drain_us{0};
// Command list entries read without waiting for the writers because their pages had been found
// untouched by the GPU before (EDEN_NX2_TRUST_CLEAN), and pages that were written after that.
inline std::atomic<std::uint64_t> g_trust_skips{0};
inline std::atomic<std::uint64_t> g_clean_violations{0};
// Semaphore acquires of the puller: evaluated; met by a look at the memory before the back was
// drained, so the drain was skipped (EDEN_NX2_ACQ_SKIP); and, of those read both before and after
// a drain, how the two compare: met both times, met only after (the drain was needed), met
// neither time (a real wait), met only before (a skip would have gone wrong). Pages the GPU wrote
// are not looked at before. Acquires from SEM_EXECUTE, and polls of suspended channels.
inline std::atomic<std::uint64_t> g_acq_checks{0};
inline std::atomic<std::uint64_t> g_acq_skips{0};
inline std::atomic<std::uint64_t> g_acq_same{0};
inline std::atomic<std::uint64_t> g_acq_needed{0};
inline std::atomic<std::uint64_t> g_acq_waits{0};
inline std::atomic<std::uint64_t> g_acq_wrong{0};
inline std::atomic<std::uint64_t> g_acq_written{0};
inline std::atomic<std::uint64_t> g_acq_exec{0};
inline std::atomic<std::uint64_t> g_resume_polls{0};
inline std::atomic<std::uint64_t> g_resume_skips{0};
// Draws, indirect draws and compute dispatches the rasterizer ran (calls, not primitives).
inline std::atomic<std::uint64_t> g_draws{0};
inline std::atomic<std::uint64_t> g_indirect_draws{0};
inline std::atomic<std::uint64_t> g_dispatches{0};
inline std::atomic<std::uint64_t> g_list_waits{0};      // GPU-written command list reads that waited
inline std::atomic<std::uint64_t> g_list_wait_us{0};
// Audio: output buffers the sink found empty (the game heard a repeated sample) and DSP render
// ticks that came late.
inline std::atomic<std::uint64_t> g_audio_underruns{0};
inline std::atomic<std::uint64_t> g_audio_late_ticks{0};
inline std::atomic<std::uint64_t> g_tc_used_mib{0};
inline std::atomic<std::uint64_t> g_tc_expected_mib{0};
inline std::atomic<std::uint64_t> g_tc_critical_mib{0};
inline std::atomic<std::uint64_t> g_tc_mode{0}; // collector: 0 idle, 1 high priority, 2 aggressive

/// EDEN_NX2_AB_SECONDS=<n>: from EDEN_NX2_AB_START seconds (default 60) after the first call, the
/// variant alternates between 0 and 1 every n seconds, so two settings meet the same scene in one
/// run. The frame log's `ab` column holds 2 before the start, then 0 or 1.
inline int AbLabel() {
    static const double period = [] {
        const char* value = std::getenv("EDEN_NX2_AB_SECONDS");
        return value != nullptr ? std::atof(value) : 0.0;
    }();
    if (period <= 0.0) {
        return 0;
    }
    static const double start = [] {
        const char* value = std::getenv("EDEN_NX2_AB_START");
        return value != nullptr ? std::atof(value) : 60.0;
    }();
    static const auto begin = std::chrono::steady_clock::now();
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    if (now < start) {
        return 2;
    }
    return (static_cast<long long>((now - start) / period) & 1) != 0 ? 1 : 0;
}

/// A setting read from EDEN_NX2_<name> (variant 0) and EDEN_NX2_<name>_B (variant 1, falls back to
/// the first); only the variant switches while EDEN_NX2_AB_SECONDS is set.
struct Knob {
    long long a{};
    long long b{};

    [[nodiscard]] long long Get() const {
        return AbLabel() == 1 ? b : a;
    }
};

inline Knob MakeKnob(const char* name, long long fallback) {
    const std::string base = std::string{"EDEN_NX2_"} + name;
    Knob knob{fallback, fallback};
    if (const char* value = std::getenv(base.c_str()); value != nullptr && *value != '\0') {
        knob.a = knob.b = std::strtoll(value, nullptr, 10);
    }
    if (const char* value = std::getenv((base + "_B").c_str()); value != nullptr && *value != '\0') {
        knob.b = std::strtoll(value, nullptr, 10);
    }
    return knob;
}

/// Adds the lifetime of the object, in microseconds, to `total` (and one to `count`, if given).
class ScopedMicros {
public:
    explicit ScopedMicros(std::atomic<std::uint64_t>& total_,
                          std::atomic<std::uint64_t>* count_ = nullptr)
        : total{total_}, count{count_}, begin{std::chrono::steady_clock::now()} {}

    ~ScopedMicros() {
        total.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                       std::chrono::steady_clock::now() - begin)
                                                       .count()),
                        std::memory_order_relaxed);
        if (count != nullptr) {
            count->fetch_add(1, std::memory_order_relaxed);
        }
    }

    /// Adds the time since construction to `into` now (a part of the total).
    void Split(std::atomic<std::uint64_t>& into) const {
        into.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                      std::chrono::steady_clock::now() - begin)
                                                      .count()),
                       std::memory_order_relaxed);
    }

    ScopedMicros(const ScopedMicros&) = delete;
    ScopedMicros& operator=(const ScopedMicros&) = delete;

private:
    std::atomic<std::uint64_t>& total;
    std::atomic<std::uint64_t>* count;
    std::chrono::steady_clock::time_point begin;
};

/// Called by the present path with the times (microseconds) of the present that just ended.
void NotePresent(long long frame_us, long long fence_us, long long composite_us, long long pace_us);

} // namespace Nx2Sampler
