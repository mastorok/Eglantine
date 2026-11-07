// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdint>
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

/// Samples the named threads for `seconds` now (blocking), writes the report to `out` (and
/// the folded stacks to `out`.folded) and returns the report.
std::string Sample(const char* names, double seconds, const std::string& out);

// Counters the renderer bumps, in the frame log and in each stall report.
inline std::atomic<std::uint64_t> g_astc_decodes{0};
inline std::atomic<std::uint64_t> g_pipelines_built{0};
inline std::atomic<std::uint64_t> g_pipeline_build_us{0};

/// Called by the present path with the times (microseconds) of the present that just ended.
void NotePresent(long long frame_us, long long fence_us, long long composite_us, long long pace_us);

} // namespace Nx2Sampler
