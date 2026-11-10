// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/nx2_sampler.h"

#ifdef _WIN32

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/thread.h"

#include <windows.h>
// windows.h first: the other two use its types.
#include <dbghelp.h>
#include <tlhelp32.h>

namespace Nx2Sampler {
namespace {

constexpr size_t MaxDepth = 40;

struct Sample {
    unsigned thread;
    unsigned depth;
    std::array<DWORD64, MaxDepth> frames;
    bool has_guest;   ///< `guest` was read (an emulated core's thread, probe registered)
    GuestSample guest;
};

std::mutex g_guest_probe_mutex;
std::function<bool(unsigned, GuestSample&)> g_guest_probe;
std::function<bool(std::uint64_t, void*, std::size_t)> g_guest_reader;

/// Reads guest memory through the reader the kernel registered (false without one).
bool ReadGuest(std::uint64_t address, void* data, std::size_t size) {
    const std::scoped_lock reader_lock{g_guest_probe_mutex};
    return g_guest_reader && g_guest_reader(address, data, size);
}

// Guest threads created by the game (NoteGuestThread) and the GPU submissions of each guest thread
// (NoteGpuSubmit); only kept when EDEN_NX2_SAMPLE is set.
struct GuestThreadInfo {
    std::uint64_t creator;
    std::uint64_t entry;
    std::uint64_t arg;
    std::int32_t priority;
    std::int32_t core;
    std::uint64_t tls;
    std::chrono::steady_clock::time_point created;
};
struct SubmitCount {
    std::uint64_t submits{};
    std::uint64_t entries{};
};
std::atomic<bool> g_thread_stats{false};
std::mutex g_threads_mutex;
std::map<std::uint64_t, GuestThreadInfo> g_guest_threads;
std::map<std::uint64_t, SubmitCount> g_gpu_submits;
thread_local std::uint64_t t_client_thread = 0;

/// Return addresses along the frame pointer chain of a guest sample (AArch64 frame records:
/// [fp] = caller's fp, [fp + 8] = return address). Games built without frame pointers give a short
/// or nonsensical chain, which the stack checks cut off. The caller holds g_guest_probe_mutex.
void UnwindGuest(GuestSample& guest) {
    constexpr std::uint64_t MaxStackSpan = std::uint64_t{16} << 20;
    guest.chain_depth = 0;
    if (!g_guest_reader) {
        return;
    }
    std::uint64_t fp = guest.fp;
    while (guest.chain_depth < guest.chain.size()) {
        if ((fp & 0xf) != 0 || fp - guest.sp > MaxStackSpan) {
            break;
        }
        std::array<std::uint64_t, 2> record{};
        if (!g_guest_reader(fp, record.data(), sizeof(record)) || record[1] == 0) {
            break;
        }
        guest.chain[guest.chain_depth++] = record[1];
        if (record[0] <= fp) {
            break;
        }
        fp = record[0];
    }
}

/// The emulated core of a "CPUCore_<n>" host thread, or -1.
int GuestCoreOf(const std::string& name) {
    static constexpr char prefix[] = "CPUCore_";
    if (name.compare(0, sizeof(prefix) - 1, prefix) != 0 || name.size() == sizeof(prefix) - 1) {
        return -1;
    }
    return std::atoi(name.c_str() + sizeof(prefix) - 1);
}

struct Target {
    HANDLE handle;
    std::string name;
};

double EnvSeconds(const char* name, double fallback) {
    const char* text = std::getenv(name);
    return text != nullptr ? std::strtod(text, nullptr) : fallback;
}

std::vector<std::string> SplitNames(const char* text) {
    std::vector<std::string> names;
    std::string current;
    for (const char* c = text; *c != '\0'; ++c) {
        if (*c == ',') {
            names.push_back(current);
            current.clear();
        } else {
            current += *c;
        }
    }
    if (!current.empty()) {
        names.push_back(current);
    }
    return names;
}

bool Matches(const std::string& thread_name, const std::vector<std::string>& patterns) {
    for (const auto& pattern : patterns) {
        if (!pattern.empty() && pattern.back() == '*') {
            if (thread_name.compare(0, pattern.size() - 1, pattern, 0, pattern.size() - 1) == 0) {
                return true;
            }
        } else if (thread_name == pattern) {
            return true;
        }
    }
    return false;
}

// A beep that wireless headsets do not swallow: a short silence first wakes the device. It plays
// through the normal audio engine (PlaySound, loaded on demand), with Beep() as the fallback.
void Tone(unsigned frequency, unsigned milliseconds, unsigned lead_milliseconds) {
    constexpr unsigned SampleRate = 22050;
    const unsigned lead = SampleRate * lead_milliseconds / 1000;
    const unsigned tone = SampleRate * milliseconds / 1000;
    const unsigned count = lead + tone;
    std::vector<unsigned char> wav(44 + static_cast<size_t>(count) * 2);
    const auto put32 = [&](size_t at, std::uint32_t value) { std::memcpy(wav.data() + at, &value, 4); };
    const auto put16 = [&](size_t at, std::uint16_t value) { std::memcpy(wav.data() + at, &value, 2); };
    std::memcpy(wav.data(), "RIFF", 4);
    put32(4, 36 + count * 2);
    std::memcpy(wav.data() + 8, "WAVEfmt ", 8);
    put32(16, 16);
    put16(20, 1); // PCM
    put16(22, 1); // mono
    put32(24, SampleRate);
    put32(28, SampleRate * 2);
    put16(32, 2);
    put16(34, 16);
    std::memcpy(wav.data() + 36, "data", 4);
    put32(40, count * 2);
    for (unsigned i = 0; i < tone; ++i) {
        const double fade = (std::min)({1.0, i / 400.0, (tone - i) / 400.0});
        const auto value = static_cast<std::int16_t>(
            20000.0 * fade * std::sin(6.283185307 * frequency * i / SampleRate));
        std::memcpy(wav.data() + 44 + static_cast<size_t>(lead + i) * 2, &value, 2);
    }
    using PlaySoundFn = BOOL(WINAPI*)(LPCSTR, HMODULE, DWORD);
    const HMODULE winmm = LoadLibraryA("winmm.dll");
    const auto play =
        winmm != nullptr ? reinterpret_cast<PlaySoundFn>(GetProcAddress(winmm, "PlaySoundA")) : nullptr;
    // SND_MEMORY | SND_NODEFAULT, synchronous.
    if (play == nullptr || !play(reinterpret_cast<LPCSTR>(wav.data()), nullptr, 0x0004 | 0x0002)) {
        Beep(frequency, milliseconds);
    }
}

// Names are matched like EDEN_NX2_SAMPLE says; "@busiest" or "@busiest:N" adds the N threads
// (default 8) that used the most CPU in the 2 seconds after the call, whatever their name.
std::vector<Target> FindThreads(const std::vector<std::string>& patterns, std::string& listing) {
    using GetThreadDescriptionFn = HRESULT(WINAPI*)(HANDLE, PWSTR*);
    const auto get_description = reinterpret_cast<GetThreadDescriptionFn>(
        GetProcAddress(GetModuleHandleA("KernelBase.dll"), "GetThreadDescription"));
    std::vector<std::string> name_patterns;
    size_t busiest = 0;
    for (const auto& pattern : patterns) {
        if (pattern.compare(0, 8, "@busiest") == 0) {
            const size_t colon = pattern.find(':');
            const long requested =
                colon != std::string::npos ? std::strtol(pattern.c_str() + colon + 1, nullptr, 10) : 8;
            busiest = static_cast<size_t>((std::max)(1L, requested));
        } else {
            name_patterns.push_back(pattern);
        }
    }
    std::vector<Target> targets;
    std::vector<Target> others;
    listing = get_description == nullptr ? "GetThreadDescription is not available. " : "";
    unsigned unnamed = 0;
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return targets;
    }
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    const DWORD self = GetCurrentThreadId();
    for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
        if (entry.th32OwnerProcessID != GetCurrentProcessId() || entry.th32ThreadID == self) {
            continue;
        }
        const HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT |
                                             THREAD_QUERY_LIMITED_INFORMATION,
                                         FALSE, entry.th32ThreadID);
        if (handle == nullptr) {
            continue;
        }
        PWSTR wide{};
        std::string name;
        if (get_description != nullptr && SUCCEEDED(get_description(handle, &wide)) &&
            wide != nullptr) {
            char buffer[256]{};
            WideCharToMultiByte(CP_UTF8, 0, wide, -1, buffer, sizeof(buffer) - 1, nullptr, nullptr);
            name = buffer;
            LocalFree(wide);
        }
        if (name.empty()) {
            ++unnamed;
        } else {
            listing += fmt::format("{}={} ", entry.th32ThreadID, name);
        }
        const std::string label = name.empty() ? fmt::format("tid {}", entry.th32ThreadID) : name;
        if (Matches(name, name_patterns)) {
            targets.push_back({handle, label});
        } else if (busiest > 0) {
            others.push_back({handle, label});
        } else {
            CloseHandle(handle);
        }
    }
    CloseHandle(snapshot);
    listing += fmt::format("({} unnamed)", unnamed);
    if (busiest > 0) {
        const auto cpu_time = [](HANDLE handle) {
            FILETIME created{}, exited{}, kernel{}, user{};
            if (!GetThreadTimes(handle, &created, &exited, &kernel, &user)) {
                return ULONGLONG{0};
            }
            const auto to_integer = [](const FILETIME& time) {
                return (static_cast<ULONGLONG>(time.dwHighDateTime) << 32) | time.dwLowDateTime;
            };
            return to_integer(kernel) + to_integer(user);
        };
        std::vector<ULONGLONG> before;
        for (const Target& other : others) {
            before.push_back(cpu_time(other.handle));
        }
        Sleep(2000);
        std::vector<std::pair<ULONGLONG, size_t>> used;
        for (size_t i = 0; i < others.size(); ++i) {
            const ULONGLONG after = cpu_time(others[i].handle);
            used.emplace_back(after > before[i] ? after - before[i] : ULONGLONG{0}, i);
        }
        std::sort(used.rbegin(), used.rend());
        std::vector<bool> keep(others.size(), false);
        for (size_t i = 0; i < (std::min)(busiest, used.size()); ++i) {
            keep[used[i].second] = used[i].first > 0;
        }
        for (size_t i = 0; i < others.size(); ++i) {
            if (keep[i]) {
                targets.push_back(others[i]);
            } else {
                CloseHandle(others[i].handle);
            }
        }
    }
    return targets;
}

// Walks a suspended thread's stack from its context. Nothing here may allocate: the thread
// could be paused holding the heap lock. `depth` is updated as frames are found, so that what was
// read before a fault is kept.
void WalkFrames(CONTEXT context, std::array<DWORD64, MaxDepth>& frames, unsigned& depth) {
    while (depth < MaxDepth && context.Rip != 0) {
        frames[depth++] = context.Rip;
        DWORD64 image_base{};
        const PRUNTIME_FUNCTION function =
            RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
        if (function == nullptr) {
            // A leaf function (no unwind data): its return address is on top of the stack. JIT
            // code has none either; past the first frame, stop there.
            if (depth != 1) {
                break;
            }
            context.Rip = *reinterpret_cast<const DWORD64*>(context.Rsp);
            context.Rsp += 8;
            continue;
        }
        PVOID handler_data{};
        DWORD64 establisher_frame{};
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, function, &context,
                         &handler_data, &establisher_frame, nullptr);
    }
}

// The unwind data of a paused thread can be half-written (JIT code, a driver, a module being
// unloaded): RtlVirtualUnwind then faults inside ntdll. That took the whole process down when the
// window lost focus and presents stopped for seconds (build 6, alt-tab). A fault ends the walk
// here and the thread is resumed as usual.
unsigned Walk(CONTEXT context, std::array<DWORD64, MaxDepth>& frames) {
    unsigned depth = 0;
#if defined(_MSC_VER)
    __try {
        WalkFrames(context, frames, depth);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
#else
    WalkFrames(context, frames, depth);
#endif
    return depth;
}

// dbghelp is initialised once per process: the profiler's report and the stall watchdog share
// this one, and hold `mutex` while they look names up.
struct Symbolizer {
    using SymInitializeFn = BOOL(WINAPI*)(HANDLE, PCSTR, BOOL);
    using SymSetOptionsFn = DWORD(WINAPI*)(DWORD);
    using SymFromAddrFn = BOOL(WINAPI*)(HANDLE, DWORD64, PDWORD64, PSYMBOL_INFO);
    using SymGetLineFn = BOOL(WINAPI*)(HANDLE, DWORD64, PDWORD, PIMAGEHLP_LINE64);

    static Symbolizer& Shared() {
        static Symbolizer instance;
        return instance;
    }

    Symbolizer() {
        const HMODULE dbghelp = LoadLibraryA("dbghelp.dll");
        if (dbghelp == nullptr) {
            return;
        }
        const auto set_options =
            reinterpret_cast<SymSetOptionsFn>(GetProcAddress(dbghelp, "SymSetOptions"));
        const auto initialize =
            reinterpret_cast<SymInitializeFn>(GetProcAddress(dbghelp, "SymInitialize"));
        from_address = reinterpret_cast<SymFromAddrFn>(GetProcAddress(dbghelp, "SymFromAddr"));
        from_line = reinterpret_cast<SymGetLineFn>(GetProcAddress(dbghelp, "SymGetLineFromAddr64"));
        if (set_options == nullptr || initialize == nullptr || from_address == nullptr) {
            from_address = nullptr;
            from_line = nullptr;
            return;
        }
        set_options(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
        if (!initialize(GetCurrentProcess(), nullptr, TRUE)) {
            from_address = nullptr;
            from_line = nullptr;
        }
    }

    /// "file.cpp:123" of an address, or nothing without line information.
    std::string Line(DWORD64 address) {
        if (from_line == nullptr) {
            return {};
        }
        IMAGEHLP_LINE64 line{};
        line.SizeOfStruct = sizeof(line);
        DWORD displacement{};
        if (!from_line(GetCurrentProcess(), address, &displacement, &line) ||
            line.FileName == nullptr) {
            return {};
        }
        const char* base = std::strrchr(line.FileName, '\\');
        return fmt::format("{}:{}", base != nullptr ? base + 1 : line.FileName, line.LineNumber);
    }

    const std::string& Name(DWORD64 address) {
        const auto [it, is_new] = cache.try_emplace(address);
        if (!is_new) {
            return it->second;
        }
        if (from_address != nullptr) {
            alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 1024]{};
            auto* const symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
            symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
            symbol->MaxNameLen = 1024;
            DWORD64 displacement{};
            if (from_address(GetCurrentProcess(), address, &displacement, symbol)) {
                it->second.assign(symbol->Name, std::min<size_t>(symbol->NameLen, 160));
                return it->second;
            }
        }
        HMODULE module{};
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCSTR>(address), &module)) {
            char path[MAX_PATH] = "?";
            GetModuleFileNameA(module, path, sizeof(path));
            const char* base = std::strrchr(path, '\\');
            it->second = std::string{base != nullptr ? base + 1 : path} + "!?";
        } else {
            it->second = "(jit)";
        }
        return it->second;
    }

    std::mutex mutex;
    SymFromAddrFn from_address{};
    SymGetLineFn from_line{};
    std::unordered_map<DWORD64, std::string> cache;
};

// <path>.guest.txt: for each emulated core, where its guest threads were (guest addresses at the
// moment the host thread was paused). The PC is the one the JIT last stored, i.e. the start of the
// block being run or of the last block entered, so a hot loop shows as a few addresses. Turn an
// address into a module offset with the module bases of the log ("Module main loaded at ...").
void GuestReport(const std::vector<Target>& targets, const std::vector<Sample>& samples,
                 const std::string& path) {
    struct Agg {
        unsigned count{};
        unsigned in_jit{};
        std::map<std::uint64_t, unsigned> callers;
        std::map<std::uint64_t, unsigned> threads;
    };
    Symbolizer& symbolizer = Symbolizer::Shared();
    const std::scoped_lock symbol_lock{symbolizer.mutex};
    std::string text;
    std::string code;
    for (unsigned thread = 0; thread < targets.size(); ++thread) {
        std::map<std::uint64_t, Agg> by_pc;
        std::map<std::uint64_t, unsigned> by_block;
        std::map<std::uint64_t, unsigned> by_thread;
        unsigned total = 0;
        unsigned with_guest = 0;
        unsigned in_jit = 0;
        for (const Sample& sample : samples) {
            if (sample.thread != thread || sample.depth == 0) {
                continue;
            }
            ++total;
            if (!sample.has_guest) {
                continue;
            }
            ++with_guest;
            const bool jit = symbolizer.Name(sample.frames[0]) == "(jit)";
            in_jit += jit ? 1 : 0;
            Agg& agg = by_pc[sample.guest.pc];
            ++agg.count;
            agg.in_jit += jit ? 1 : 0;
            ++agg.callers[sample.guest.lr];
            ++agg.threads[sample.guest.thread_id];
            ++by_block[sample.guest.pc & ~std::uint64_t{0xff}];
            ++by_thread[sample.guest.thread_id];
        }
        if (with_guest == 0) {
            continue;
        }
        text += fmt::format("=== {}: {} samples, {} with a guest state, {} of them in JIT code ({:.1f}%)\n",
                            targets[thread].name, total, with_guest, in_jit,
                            100.0 * in_jit / (std::max)(with_guest, 1U));
        const auto top = [](const auto& counts, size_t limit) {
            std::vector<std::pair<unsigned, std::uint64_t>> sorted;
            for (const auto& [key, value] : counts) {
                sorted.emplace_back(value, key);
            }
            std::sort(sorted.rbegin(), sorted.rend());
            sorted.resize((std::min)(limit, sorted.size()));
            return sorted;
        };
        text += "-- guest threads (id: samples)\n";
        for (const auto& [count, id] : top(by_thread, 8)) {
            text += fmt::format("  {:>8} {:5.1f}%  thread {}\n", count, 100.0 * count / with_guest, id);
        }
        text += "-- 256-byte blocks of guest code (address: samples)\n";
        for (const auto& [count, block] : top(by_block, 40)) {
            text += fmt::format("  {:#014x} {:>8} {:5.1f}%\n", block, count, 100.0 * count / with_guest);
        }
        // The code of the hottest blocks, 512 bytes from 128 bytes before each (one line of hex
        // per 32 bytes, "<address>: <bytes>"), for a disassembler.
        code += fmt::format("=== {}\n", targets[thread].name);
        std::set<std::uint64_t> dumped;
        std::map<std::uint64_t, unsigned> callees; // bl targets of the hot blocks, by their samples
        for (const auto& [count, block] : top(by_block, 24)) {
            const std::uint64_t start = block >= 0x80 ? block - 0x80 : 0;
            if (!dumped.insert(start).second) {
                continue;
            }
            std::array<unsigned char, 512> bytes{};
            bool ok = false;
            {
                const std::scoped_lock reader_lock{g_guest_probe_mutex};
                ok = g_guest_reader && g_guest_reader(start, bytes.data(), bytes.size());
            }
            if (!ok) {
                code += fmt::format("# {:#x}: unreadable\n", start);
                continue;
            }
            code += fmt::format("# block {:#x}: {} samples\n", block, count);
            for (size_t offset = 0; offset < bytes.size(); offset += 32) {
                code += fmt::format("{:#x}: ", start + offset);
                for (size_t i = 0; i < 32; ++i) {
                    code += fmt::format("{:02x}", bytes[offset + i]);
                }
                code += '\n';
            }
            for (size_t offset = 0; offset + 4 <= bytes.size(); offset += 4) {
                std::uint32_t word{};
                std::memcpy(&word, bytes.data() + offset, sizeof(word));
                if ((word & 0xfc000000U) == 0x94000000U) { // bl
                    const std::int64_t delta = static_cast<std::int32_t>(word << 6) >> 6;
                    callees[static_cast<std::uint64_t>(static_cast<std::int64_t>(start + offset) +
                                                       delta * 4)] += count;
                }
            }
        }
        // The functions the hot blocks call: their first 192 bytes, to tell locks, copies and
        // decompressors from the game's own code.
        for (const auto& [weight, target] : top(callees, 16)) {
            bool covered = false;
            for (const std::uint64_t start : dumped) {
                covered = covered || (target >= start && target < start + 512);
            }
            std::array<unsigned char, 192> bytes{};
            if (covered || !ReadGuest(target, bytes.data(), bytes.size())) {
                continue;
            }
            code += fmt::format("# callee {:#x}: called from the hot blocks ({} samples)\n", target,
                                weight);
            for (size_t offset = 0; offset < bytes.size(); offset += 32) {
                code += fmt::format("{:#x}: ", target + offset);
                for (size_t i = 0; i < 32; ++i) {
                    code += fmt::format("{:02x}", bytes[offset + i]);
                }
                code += '\n';
            }
        }
        text += "-- guest addresses (address: samples, in JIT; most frequent LR; most frequent thread)\n";
        std::vector<std::pair<unsigned, std::uint64_t>> pcs;
        for (const auto& [pc, agg] : by_pc) {
            pcs.emplace_back(agg.count, pc);
        }
        std::sort(pcs.rbegin(), pcs.rend());
        pcs.resize((std::min)(size_t{100}, pcs.size()));
        for (const auto& [count, pc] : pcs) {
            const Agg& agg = by_pc[pc];
            const auto best = [](const std::map<std::uint64_t, unsigned>& counts) {
                std::uint64_t key = 0;
                unsigned most = 0;
                for (const auto& [k, v] : counts) {
                    if (v > most) {
                        most = v;
                        key = k;
                    }
                }
                return key;
            };
            text += fmt::format("  {:#014x} {:>8} {:5.1f}%  jit {:>6}  lr {:#014x}  thread {}\n", pc, count,
                                100.0 * count / with_guest, agg.in_jit, best(agg.callers),
                                best(agg.threads));
        }
        text += "\n";
    }
    if (text.empty()) {
        return;
    }
    if (std::FILE* const file = std::fopen((path + ".guest.txt").c_str(), "w")) {
        std::fputs(text.c_str(), file);
        std::fclose(file);
    }
    if (std::FILE* const file = std::fopen((path + ".guest_code.txt").c_str(), "w")) {
        std::fputs(code.c_str(), file);
        std::fclose(file);
    }
}

std::string Report(const std::vector<Target>& targets, const std::vector<Sample>& samples,
                   const char* path) {
    Symbolizer& symbolizer = Symbolizer::Shared();
    const std::scoped_lock symbol_lock{symbolizer.mutex};
    std::string text;
    for (unsigned thread = 0; thread < targets.size(); ++thread) {
        std::map<std::string, unsigned> self;
        std::map<std::string, unsigned> inclusive;
        std::map<std::string, unsigned> self_caller;
        unsigned total = 0;
        for (const Sample& sample : samples) {
            if (sample.thread != thread || sample.depth == 0) {
                continue;
            }
            ++total;
            const std::string& top = symbolizer.Name(sample.frames[0]);
            ++self[top];
            if (sample.depth > 1) {
                ++self_caller[top + "  <-  " + symbolizer.Name(sample.frames[1])];
            }
            std::set<std::string> seen;
            for (unsigned i = 0; i < sample.depth; ++i) {
                const std::string& name = symbolizer.Name(sample.frames[i]);
                if (seen.insert(name).second) {
                    ++inclusive[name];
                }
            }
        }
        const auto print = [&](const char* title, const std::map<std::string, unsigned>& counts,
                               size_t limit) {
            std::vector<std::pair<unsigned, std::string>> sorted;
            for (const auto& [name, count] : counts) {
                sorted.emplace_back(count, name);
            }
            std::sort(sorted.rbegin(), sorted.rend());
            text += fmt::format("-- {}\n", title);
            for (size_t i = 0; i < (std::min)(limit, sorted.size()); ++i) {
                text += fmt::format("{:6} {:5.1f}% {}\n", sorted[i].first,
                                    100.0 * sorted[i].first / (std::max)(total, 1U),
                                    sorted[i].second.c_str());
            }
        };
        text += fmt::format("=== thread \"{}\": {} samples\n", targets[thread].name, total);
        print("self (where the thread is)", self, 60);
        // The hottest instructions as module offsets, for llvm-symbolizer --relative-address
        // (which line of a function, inlined code included).
        std::map<DWORD64, unsigned> addresses;
        for (const Sample& sample : samples) {
            if (sample.thread == thread && sample.depth != 0) {
                ++addresses[sample.frames[0]];
            }
        }
        std::vector<std::pair<unsigned, DWORD64>> hot;
        for (const auto& [address, count] : addresses) {
            hot.emplace_back(count, address);
        }
        std::sort(hot.rbegin(), hot.rend());
        text += "-- self by instruction (module+offset)\n";
        for (size_t i = 0; i < (std::min<size_t>)(hot.size(), 60); ++i) {
            const DWORD64 address = hot[i].second;
            HMODULE module{};
            char module_path[MAX_PATH] = "?";
            DWORD64 offset = address;
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   reinterpret_cast<LPCSTR>(address), &module)) {
                GetModuleFileNameA(module, module_path, sizeof(module_path));
                offset = address - reinterpret_cast<DWORD64>(module);
            }
            const char* base = std::strrchr(module_path, '\\');
            text += fmt::format("{:6} {}+{:#x} {}\n", hot[i].first,
                                base != nullptr ? base + 1 : module_path, offset,
                                symbolizer.Name(address));
        }
        print("self <- caller", self_caller, 50);
        print("inclusive (on the stack)", inclusive, 90);
        text += "\n";
    }
    if (std::FILE* const file = std::fopen(path, "w")) {
        std::fwrite(text.data(), 1, text.size(), file);
        std::fclose(file);
    }

    // <out>.folded: every distinct stack with its count, root first ("thread;a;b;c count").
    std::map<std::string, unsigned> folded;
    for (const Sample& sample : samples) {
        std::string line = targets[sample.thread].name;
        for (unsigned i = sample.depth; i-- > 0;) {
            line += ';';
            line += symbolizer.Name(sample.frames[i]);
        }
        ++folded[line];
    }
    if (std::FILE* const out = std::fopen((std::string{path} + ".folded").c_str(), "w")) {
        for (const auto& [line, count] : folded) {
            std::fprintf(out, "%s %u\n", line.c_str(), count);
        }
        std::fclose(out);
    }
    return text;
}

// ---- Stall watchdog (EDEN_NX2_STALL_MS) and frame log (EDEN_NX2_FRAME_LOG) ----------------------

const auto g_process_start = std::chrono::steady_clock::now();
std::atomic<long long> g_last_present_ns{0};

constexpr std::array<const char*, 8> StallGroups{"GPU",     "GPUBack", "VulkanWorker", "CPUCores",
                                                 "Workers", "Service", "Audio",        "Others"};

struct StallTarget {
    HANDLE handle;
    int group;
};

struct StallCounters {
    std::uint64_t astc;
    std::uint64_t pipelines;
    std::uint64_t pipeline_us;
    std::uint64_t sparse_binds;
    std::uint64_t sparse_us;
    std::uint64_t sparse_call_us;
    std::uint64_t created;
    std::uint64_t deleted;
    std::uint64_t gc_us;
    std::uint64_t del_gc;
    std::uint64_t del_gc_download;
    std::uint64_t del_unmap;
    std::uint64_t del_join;
    std::uint64_t created_replace;
    std::uint64_t gc_finish_us;
    std::uint64_t list_waits;
    std::uint64_t list_wait_us;
    std::uint64_t audio_underruns;
    std::uint64_t audio_late_ticks;
};

StallCounters ReadCounters() {
    constexpr auto relaxed = std::memory_order_relaxed;
    return {g_astc_decodes.load(relaxed),     g_pipelines_built.load(relaxed),
            g_pipeline_build_us.load(relaxed), g_sparse_binds.load(relaxed),
            g_sparse_bind_us.load(relaxed),   g_sparse_call_us.load(relaxed),
            g_images_created.load(relaxed),   g_images_deleted.load(relaxed),
            g_gc_us.load(relaxed),            g_del_gc.load(relaxed),
            g_del_gc_download.load(relaxed),  g_del_unmap.load(relaxed),
            g_del_join.load(relaxed),         g_created_replace.load(relaxed),
            g_gc_finish_us.load(relaxed),     g_list_waits.load(relaxed),
            g_list_wait_us.load(relaxed),     g_audio_underruns.load(relaxed),
            g_audio_late_ticks.load(relaxed)};
}

int StallGroupOf(const std::string& name) {
    if (name == "GPU") {
        return 0;
    }
    if (name == "GPUBack") {
        return 1;
    }
    if (name == "VulkanWorker") {
        return 2;
    }
    if (name.starts_with("CPUCore_")) {
        return 3;
    }
    if (name.starts_with("VkPipelineBuilder") || name.starts_with("TextureDecoder") ||
        name.starts_with("nx2")) {
        return 4;
    }
    if (name.starts_with("Service") || name.starts_with("HLE") || name.starts_with("Hle")) {
        return 5;
    }
    if (name == "audio" || name.starts_with("DSP_") || name.starts_with("Audio")) {
        return 6;
    }
    // EDEN_NX2_STALL_OTHERS=1: every other thread too, the graphics driver's among them.
    static const bool others = [] {
        const char* value = std::getenv("EDEN_NX2_STALL_OTHERS");
        return value != nullptr && *value != '\0' && *value != '0';
    }();
    return others ? 7 : -1;
}

void CloseStallTargets(std::vector<StallTarget>& targets) {
    for (const StallTarget& target : targets) {
        CloseHandle(target.handle);
    }
    targets.clear();
}

// Opens the threads of the groups above (by their names) and returns every name it saw.
std::string CollectStallTargets(std::vector<StallTarget>& targets) {
    CloseStallTargets(targets);
    using GetThreadDescriptionFn = HRESULT(WINAPI*)(HANDLE, PWSTR*);
    const auto get_description = reinterpret_cast<GetThreadDescriptionFn>(
        GetProcAddress(GetModuleHandleA("KernelBase.dll"), "GetThreadDescription"));
    if (get_description == nullptr) {
        return "GetThreadDescription is not available";
    }
    std::string seen;
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return seen;
    }
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    const DWORD self = GetCurrentThreadId();
    for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
        if (entry.th32OwnerProcessID != GetCurrentProcessId() || entry.th32ThreadID == self) {
            continue;
        }
        const HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT |
                                             THREAD_QUERY_LIMITED_INFORMATION,
                                         FALSE, entry.th32ThreadID);
        if (handle == nullptr) {
            continue;
        }
        PWSTR wide{};
        std::string name;
        if (SUCCEEDED(get_description(handle, &wide)) && wide != nullptr) {
            char buffer[256]{};
            WideCharToMultiByte(CP_UTF8, 0, wide, -1, buffer, sizeof(buffer) - 1, nullptr, nullptr);
            name = buffer;
            LocalFree(wide);
        }
        const int group = StallGroupOf(name);
        if (!name.empty()) {
            seen += name + " ";
        }
        if (group < 0) {
            CloseHandle(handle);
        } else {
            targets.push_back({handle, group});
        }
    }
    CloseHandle(snapshot);
    return seen;
}

// One pass over the targets. Nothing may allocate while a thread is paused (it could hold the
// heap lock): the sample vector is reserved by the caller and never grows here.
void SampleStallRound(const std::vector<StallTarget>& targets, std::vector<Sample>& samples) {
    for (unsigned i = 0; i < targets.size(); ++i) {
        if (samples.size() == samples.capacity()) {
            return;
        }
        const HANDLE handle = targets[i].handle;
        if (SuspendThread(handle) == static_cast<DWORD>(-1)) {
            continue;
        }
        CONTEXT context{};
        context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
        if (GetThreadContext(handle, &context)) {
            Sample& sample = samples.emplace_back();
            sample.thread = i;
            sample.depth = Walk(context, sample.frames);
        }
        ResumeThread(handle);
    }
}

void ReportStall(unsigned index, double at_seconds, long long gap_ms, unsigned rounds,
                 const std::vector<int>& groups, const std::vector<Sample>& samples,
                 const StallCounters& before, const StallCounters& after,
                 const std::string& threads) {
    LOG_INFO(Common,
             "nx2 stall #{} at {:.1f} s: {} ms between presents, {} sampling rounds; ASTC decodes "
             "+{}, pipelines built +{} ({:.1f} ms), sparse binds +{} ({:.1f} ms, of which the bind call {:.1f} ms), images +{} -{}, "
             "texture GC {:.1f} ms",
             index, at_seconds, gap_ms, rounds, after.astc - before.astc,
             after.pipelines - before.pipelines, (after.pipeline_us - before.pipeline_us) / 1000.0,
             after.sparse_binds - before.sparse_binds, (after.sparse_us - before.sparse_us) / 1000.0,
             (after.sparse_call_us - before.sparse_call_us) / 1000.0,
             after.created - before.created, after.deleted - before.deleted,
             (after.gc_us - before.gc_us) / 1000.0);
    LOG_INFO(Common,
             "nx2 stall #{} cache: deleted by collector {} ({} downloaded, GPU waits {:.1f} ms), by "
             "unmap {}, replaced {}; new images replacing others {}; command list waits {} ({:.1f} "
             "ms); audio underruns {}, late DSP ticks {}; texture cache {} MiB of {} / {} MiB, "
             "collector mode {}",
             index, after.del_gc - before.del_gc, after.del_gc_download - before.del_gc_download,
             (after.gc_finish_us - before.gc_finish_us) / 1000.0, after.del_unmap - before.del_unmap,
             after.del_join - before.del_join, after.created_replace - before.created_replace,
             after.list_waits - before.list_waits, (after.list_wait_us - before.list_wait_us) / 1000.0,
             after.audio_underruns - before.audio_underruns,
             after.audio_late_ticks - before.audio_late_ticks, g_tc_used_mib.load(std::memory_order_relaxed),
             g_tc_expected_mib.load(std::memory_order_relaxed),
             g_tc_critical_mib.load(std::memory_order_relaxed),
             g_tc_mode.load(std::memory_order_relaxed));
    if (!threads.empty()) {
        LOG_INFO(Common, "nx2 stall: threads seen: {}", threads);
    }
    Symbolizer& symbolizer = Symbolizer::Shared();
    const std::scoped_lock symbol_lock{symbolizer.mutex};
    for (size_t group = 0; group < StallGroups.size(); ++group) {
        struct Entry {
            unsigned count{};
            DWORD64 address{};
        };
        std::map<std::string, Entry> signatures;
        unsigned total = 0;
        for (const Sample& sample : samples) {
            if (sample.depth == 0 || groups[sample.thread] != static_cast<int>(group)) {
                continue;
            }
            ++total;
            std::string signature;
            const std::string* previous = nullptr;
            unsigned names = 0;
            for (unsigned i = 0; i < (std::min)(sample.depth, 14U) && names < 7; ++i) {
                const std::string& name = symbolizer.Name(sample.frames[i]);
                if (previous != nullptr && *previous == name) {
                    continue;
                }
                previous = &name;
                ++names;
                if (!signature.empty()) {
                    signature += " <- ";
                }
                signature += name.substr(0, 90);
            }
            Entry& entry = signatures[signature];
            ++entry.count;
            entry.address = sample.frames[0];
        }
        if (total == 0) {
            continue;
        }
        std::vector<std::pair<unsigned, std::string>> ranked;
        std::map<std::string, DWORD64> addresses;
        for (const auto& [signature, entry] : signatures) {
            ranked.emplace_back(entry.count, signature);
            addresses[signature] = entry.address;
        }
        std::sort(ranked.rbegin(), ranked.rend());
        const size_t shown = group <= 2 || group == 6 ? 3 : group == 7 ? 8 : 2;
        for (size_t i = 0; i < (std::min)(shown, ranked.size()); ++i) {
            LOG_INFO(Common, "  {} {:.0f}% of {}: {}  [{}]", StallGroups[group],
                     100.0 * ranked[i].first / total, total, ranked[i].second,
                     symbolizer.Line(addresses[ranked[i].second]));
        }
    }
}

// Reports are written by a thread of their own: looking up symbols takes long enough (tens of ms
// for the first lookups) that the next stall began before the watchdog was back.
struct StallJob {
    unsigned index{};
    double at_seconds{};
    long long gap_ms{};
    unsigned rounds{};
    std::vector<int> groups;
    std::vector<Sample> samples;
    StallCounters before{};
    StallCounters after{};
    std::string threads;
};

class StallReporter {
public:
    StallReporter() {
        std::thread{[this] { Run(); }}.detach();
    }

    void Push(StallJob&& job) {
        {
            const std::scoped_lock lock{mutex};
            if (queue.size() >= 8) {
                return;
            }
            queue.push_back(std::move(job));
        }
        wake.notify_one();
    }

private:
    void Run() {
        Common::SetCurrentThreadName("Nx2StallLog");
        for (;;) {
            StallJob job;
            {
                std::unique_lock lock{mutex};
                wake.wait(lock, [this] { return !queue.empty(); });
                job = std::move(queue.front());
                queue.pop_front();
            }
            ReportStall(job.index, job.at_seconds, job.gap_ms, job.rounds, job.groups, job.samples,
                        job.before, job.after, job.threads);
        }
    }

    std::mutex mutex;
    std::condition_variable wake;
    std::deque<StallJob> queue;
};

void StallLoop(unsigned threshold_ms) {
    Common::SetCurrentThreadName("Nx2Stall");
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    {
        // The first lookup reads this module's whole PDB: do it now, in the background.
        Symbolizer& symbolizer = Symbolizer::Shared();
        const std::scoped_lock symbol_lock{symbolizer.mutex};
        symbolizer.Name(reinterpret_cast<DWORD64>(&StallLoop));
    }
    LOG_INFO(Common, "nx2 stall watchdog: stacks are taken when presents are {} ms or more apart",
             threshold_ms);
    static StallReporter* const reporter = new StallReporter; // lives as long as the process
    constexpr unsigned MaxReports = 400;
    constexpr unsigned MaxRounds = 600;
    std::vector<StallTarget> targets;
    std::vector<Sample> samples;
    std::string threads;
    bool listed = false;
    bool stalling = false;
    long long stall_from = 0;
    unsigned rounds = 0;
    unsigned reports = 0;
    StallCounters before{};
    std::array<std::pair<long long, StallCounters>, 64> recent{};
    size_t recent_next = 0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds{stalling ? 2 : 4});
        const long long last = g_last_present_ns.load(std::memory_order_relaxed);
        if (last == 0) {
            continue;
        }
        const long long now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now().time_since_epoch())
                                  .count();
        if (!stalling) {
            // The counters as they were at the last present, found in what was read while waiting.
            recent[recent_next++ % recent.size()] = {now, ReadCounters()};
            if (now - last < static_cast<long long>(threshold_ms) * 1'000'000LL ||
                reports >= MaxReports) {
                continue;
            }
            stalling = true;
            stall_from = last;
            rounds = 0;
            samples.clear();
            threads = CollectStallTargets(targets);
            if (listed) {
                threads.clear();
            }
            listed = true;
            samples.reserve(size_t{MaxRounds} * (targets.size() + 1));
            before = ReadCounters();
            long long best = 0;
            for (const auto& [time, counters] : recent) {
                if (time != 0 && time <= last && time > best) {
                    best = time;
                    before = counters;
                }
            }
            continue;
        }
        if (last != stall_from) {
            ++reports;
            StallJob job;
            job.index = reports;
            job.at_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - g_process_start)
                    .count();
            job.gap_ms = (last - stall_from) / 1'000'000;
            job.rounds = rounds;
            for (const StallTarget& target : targets) {
                job.groups.push_back(target.group);
            }
            job.samples = std::move(samples);
            samples = {};
            job.before = before;
            job.after = ReadCounters();
            job.threads = std::move(threads);
            threads = {};
            reporter->Push(std::move(job));
            stalling = false;
            continue;
        }
        if (rounds < MaxRounds) {
            SampleStallRound(targets, samples);
            ++rounds;
        }
    }
}

struct FrameLog {
    ~FrameLog() {
        Flush();
    }
    void Flush() {
        if (file.is_open() && !pending.empty()) {
            file << pending;
            file.flush();
            pending.clear();
        }
    }
    std::mutex mutex;
    std::ofstream file;
    std::string pending;
    unsigned lines{};
    bool opened{};
};

/// The printable runs (4 characters or more) of a buffer, quoted, for thread names.
std::string PrintableRuns(const unsigned char* data, size_t size, unsigned limit) {
    std::string text;
    unsigned found = 0;
    size_t i = 0;
    while (i < size && found < limit) {
        size_t j = i;
        while (j < size && data[j] >= 0x20 && data[j] < 0x7f) {
            ++j;
        }
        if (j - i >= 4) {
            text += fmt::format("{}\"{}\"", found != 0 ? ", " : "",
                                std::string(reinterpret_cast<const char*>(data) + i,
                                            (std::min)(j - i, size_t{48})));
            ++found;
        }
        i = j + 1;
    }
    return text;
}

void DumpGuestCode(std::string& code, std::uint64_t start, size_t size, const std::string& title) {
    std::vector<unsigned char> bytes(size);
    if (!ReadGuest(start, bytes.data(), bytes.size())) {
        return;
    }
    code += fmt::format("# {}\n", title);
    for (size_t offset = 0; offset < size; offset += 32) {
        code += fmt::format("{:#x}: ", start + offset);
        for (size_t i = 0; i < 32 && offset + i < size; ++i) {
            code += fmt::format("{:02x}", bytes[offset + i]);
        }
        code += '\n';
    }
}

// <path>.guest_threads.txt: the guest threads of the game (not the emulated cores): when each was
// created and from where, what its ThreadType holds (names, code pointers), how many GPU
// submissions it made, and where it spent its samples (code blocks, call chains, return
// addresses). <path>.guest_threads_code.txt has the code around the hottest places of the busiest
// threads and around their most frequent return addresses.
void GuestThreadReport(const std::vector<Target>& targets, const std::vector<Sample>& samples,
                       const std::string& path,
                       const std::map<std::uint64_t, SubmitCount>& submits_before) {
    struct Agg {
        unsigned count{};
        std::map<int, unsigned> cores;
        std::map<std::uint64_t, unsigned> blocks;
        std::map<std::vector<std::uint64_t>, unsigned> stacks;
        std::map<std::uint64_t, unsigned> inclusive;
    };
    std::map<std::uint64_t, Agg> threads;
    std::vector<unsigned> per_target(targets.size());
    unsigned total = 0;
    for (const Sample& sample : samples) {
        if (!sample.has_guest || sample.thread >= targets.size()) {
            continue;
        }
        ++per_target[sample.thread];
        ++total;
        Agg& agg = threads[sample.guest.thread_id];
        ++agg.count;
        ++agg.cores[GuestCoreOf(targets[sample.thread].name)];
        const std::uint64_t block = sample.guest.pc & ~std::uint64_t{0xff};
        ++agg.blocks[block];
        std::vector<std::uint64_t> stack{block};
        std::set<std::uint64_t> seen;
        for (std::uint32_t i = 0; i < sample.guest.chain_depth; ++i) {
            if (i < 6) {
                stack.push_back(sample.guest.chain[i]);
            }
            if (seen.insert(sample.guest.chain[i]).second) {
                ++agg.inclusive[sample.guest.chain[i]];
            }
        }
        ++agg.stacks[stack];
    }
    if (total == 0) {
        return;
    }
    const unsigned window = *std::max_element(per_target.begin(), per_target.end());
    std::map<std::uint64_t, GuestThreadInfo> registry;
    std::map<std::uint64_t, SubmitCount> submits;
    {
        const std::scoped_lock threads_lock{g_threads_mutex};
        registry = g_guest_threads;
        submits = g_gpu_submits;
    }
    for (auto& [id, count] : submits) {
        if (const auto it = submits_before.find(id); it != submits_before.end()) {
            count.submits -= it->second.submits;
            count.entries -= it->second.entries;
        }
    }
    const auto top = [](const auto& counts, size_t limit) {
        std::vector<std::pair<unsigned, typename std::decay_t<decltype(counts)>::key_type>> sorted;
        for (const auto& [key, value] : counts) {
            sorted.emplace_back(value, key);
        }
        std::sort(sorted.rbegin(), sorted.rend());
        sorted.resize((std::min)(limit, sorted.size()));
        return sorted;
    };
    std::vector<std::pair<unsigned, std::uint64_t>> order;
    for (const auto& [id, agg] : threads) {
        order.emplace_back(agg.count, id);
    }
    std::sort(order.rbegin(), order.rend());
    std::set<std::uint64_t> shown;
    for (size_t i = 0; i < (std::min)(size_t{24}, order.size()); ++i) {
        shown.insert(order[i].second);
    }
    for (const auto& [id, count] : submits) {
        if (count.submits != 0) {
            shown.insert(id);
        }
    }

    std::string text = fmt::format(
        "{} guest samples over {} cores; a window is {} samples, so 1.00 below is one core "
        "busy for the whole window\n\n",
        total, targets.size(), window);
    text += "-- guest threads created by the game (id: creator, created at, entry, arg, priority, "
            "core | samples, GPU submits in the window)\n";
    for (const auto& [id, info] : registry) {
        const auto found = threads.find(id);
        const auto sub = submits.find(id);
        text += fmt::format(
            "  thread {:>4}: by {:>4}, at {:7.2f} s, entry {:#x}, arg {:#x}, prio {:>2}, core {} | "
            "{} samples, {} submits\n",
            id, info.creator,
            std::chrono::duration<double>(info.created - g_process_start).count(), info.entry,
            info.arg, info.priority, info.core, found != threads.end() ? found->second.count : 0U,
            sub != submits.end() ? sub->second.submits : 0U);
    }
    text += "\n-- GPU submissions in the window (thread id: submits, pushbuffer entries)\n";
    for (const auto& [id, count] : submits) {
        if (count.submits != 0) {
            text += fmt::format("  thread {:>4}: {} submits, {} entries\n", id, count.submits,
                                count.entries);
        }
    }

    std::string code;
    unsigned code_threads = 0;
    for (const auto& entry : order) {
        const std::uint64_t id = entry.second;
        if (shown.find(id) == shown.end()) {
            continue;
        }
        const Agg& agg = threads[id];
        const bool dump_code = code_threads++ < 8;
        if (dump_code) {
            code += fmt::format("=== thread {}\n", id);
        }
        text += fmt::format("\n=== guest thread {}: {} samples = {:.2f} of a core ({:.1f}% of the "
                            "guest samples)\n",
                            id, agg.count, static_cast<double>(agg.count) / (std::max)(window, 1U),
                            100.0 * agg.count / total);
        text += "  cores:";
        for (const auto& [core, n] : agg.cores) {
            text += fmt::format(" CPUCore_{} {}", core, n);
        }
        text += '\n';
        if (const auto sub = submits.find(id); sub != submits.end() && sub->second.submits != 0) {
            text += fmt::format("  GPU submits in the window: {} ({} entries)\n", sub->second.submits,
                                sub->second.entries);
        }
        if (const auto it = registry.find(id); it != registry.end()) {
            const GuestThreadInfo& info = it->second;
            text += fmt::format("  created by thread {} with entry {:#x}, arg {:#x}, priority {}, "
                                "core {}, tls {:#x}\n",
                                info.creator, info.entry, info.arg, info.priority, info.core,
                                info.tls);
            // The argument of a Nintendo thread is its ThreadType (name, function, stack...).
            std::array<unsigned char, 0x200> type{};
            if (ReadGuest(info.arg, type.data(), type.size())) {
                text += fmt::format("  arg strings: {}\n",
                                    PrintableRuns(type.data(), type.size(), 6));
                text += "  arg words:";
                unsigned printed = 0;
                for (size_t offset = 0; offset + 8 <= type.size() && printed < 40; offset += 8) {
                    std::uint64_t value{};
                    std::memcpy(&value, type.data() + offset, sizeof(value));
                    if (value >= 0x10000 && value < (std::uint64_t{1} << 40)) {
                        text += fmt::format(" +{:#x}={:#x}", offset, value);
                        ++printed;
                    }
                }
                text += '\n';
            }
            std::uint64_t type_pointer{};
            std::array<unsigned char, 0x200> tls_type{};
            if (info.tls != 0 && ReadGuest(info.tls + 0x1f8, &type_pointer, sizeof(type_pointer)) &&
                type_pointer != info.arg && ReadGuest(type_pointer, tls_type.data(), tls_type.size())) {
                text += fmt::format("  TLS ThreadType {:#x} strings: {}\n", type_pointer,
                                    PrintableRuns(tls_type.data(), tls_type.size(), 6));
            }
            if (dump_code) {
                DumpGuestCode(code, info.entry & ~std::uint64_t{0x1f}, 128,
                              fmt::format("thread {} entry {:#x}", id, info.entry));
            }
        } else {
            text += "  not created by svcCreateThread while the sampler ran (main thread?)\n";
        }
        text += "  -- 256-byte blocks of guest code (address: samples)\n";
        for (const auto& [n, block] : top(agg.blocks, 12)) {
            text += fmt::format("    {:#014x} {:>7} {:5.1f}%\n", block, n, 100.0 * n / agg.count);
        }
        text += "  -- call chains (code block <- return addresses, innermost first)\n";
        for (const auto& [n, stack] : top(agg.stacks, 12)) {
            text += fmt::format("    {:>7} {:5.1f}%  {:#x}", n, 100.0 * n / agg.count, stack[0]);
            for (size_t i = 1; i < stack.size(); ++i) {
                text += fmt::format(" <- {:#x}", stack[i]);
            }
            text += '\n';
        }
        text += "  -- return addresses in the chains (address: samples that had it)\n";
        const auto inclusive = top(agg.inclusive, 24);
        for (const auto& [n, address] : inclusive) {
            text += fmt::format("    {:#014x} {:>7} {:5.1f}%\n", address, n, 100.0 * n / agg.count);
        }
        if (dump_code) {
            for (const auto& [n, block] : top(agg.blocks, 4)) {
                DumpGuestCode(code, block >= 0x80 ? block - 0x80 : 0, 512,
                              fmt::format("block {:#x}: {} samples", block, n));
            }
            for (size_t i = 0; i < (std::min)(size_t{16}, inclusive.size()); ++i) {
                const std::uint64_t address = inclusive[i].second;
                DumpGuestCode(code, (address - 96) & ~std::uint64_t{0x1f}, 192,
                              fmt::format("return address {:#x}: {} samples", address,
                                          inclusive[i].first));
            }
        }
    }
    if (std::FILE* const file = std::fopen((path + ".guest_threads.txt").c_str(), "w")) {
        std::fputs(text.c_str(), file);
        std::fclose(file);
    }
    if (std::FILE* const file = std::fopen((path + ".guest_threads_code.txt").c_str(), "w")) {
        std::fputs(code.c_str(), file);
        std::fclose(file);
    }
}

std::string SampleThreads(const std::vector<std::string>& patterns, double seconds,
                          const std::string& out) {
    // EDEN_NX2_SAMPLE_BEEP=0 silences the tones that mark the start (two rising) and the end (one
    // long low) of the sampling.
    const bool beep = EnvSeconds("EDEN_NX2_SAMPLE_BEEP", 1.0) != 0.0;
    std::error_code ignored;
    std::filesystem::create_directories(std::filesystem::path{out}.parent_path(), ignored);
    std::string listing;
    const std::vector<Target> targets = FindThreads(patterns, listing);
    std::fprintf(stderr, "nx2 sampler: %zu threads match, sampling %.0f s\n", targets.size(),
                 seconds);
    std::string chosen;
    for (const Target& target : targets) {
        chosen += target.name + "; ";
    }
    LOG_INFO(Common, "nx2 sampler: {} threads sampled for {:.0f} s: {}", targets.size(), seconds,
             chosen);
    LOG_INFO(Common, "nx2 sampler: threads of the process: {}", listing);
    if (targets.empty()) {
        const std::string text = "no thread matches\nthreads of the process: " + listing + "\n";
        if (std::FILE* const file = std::fopen(out.c_str(), "w")) {
            std::fputs(text.c_str(), file);
            std::fclose(file);
        }
        if (beep) {
            Tone(300, 1500, 400);
        }
        return text;
    }
    if (beep) {
        Tone(800, 500, 400);
        Tone(1100, 500, 0);
    }
    std::vector<Sample> samples;
    samples.reserve(static_cast<size_t>(seconds * 1100.0) * targets.size() + 16);
    // The guest core of each target (CPUCore_<n>), for the guest addresses of .guest.txt. The probe
    // is called under its mutex, so the kernel can take it away (SetGuestProbe({})) at shutdown
    // without a call still running; it never waits for the paused thread.
    const auto guest_probe = [](unsigned core, GuestSample& out) {
        const std::scoped_lock probe_lock{g_guest_probe_mutex};
        if (!g_guest_probe || !g_guest_probe(core, out)) {
            return false;
        }
        UnwindGuest(out);
        return true;
    };
    std::map<std::uint64_t, SubmitCount> submits_before;
    {
        const std::scoped_lock threads_lock{g_threads_mutex};
        submits_before = g_gpu_submits;
    }
    std::vector<int> guest_core(targets.size(), -1);
    for (unsigned i = 0; i < targets.size(); ++i) {
        guest_core[i] = GuestCoreOf(targets[i].name);
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < deadline &&
           samples.size() + targets.size() <= samples.capacity()) {
        for (unsigned i = 0; i < targets.size(); ++i) {
            const HANDLE handle = targets[i].handle;
            if (SuspendThread(handle) == static_cast<DWORD>(-1)) {
                continue;
            }
            CONTEXT context{};
            context.ContextFlags = CONTEXT_FULL;
            if (GetThreadContext(handle, &context)) {
                Sample& sample = samples.emplace_back();
                sample.thread = i;
                sample.depth = Walk(context, sample.frames);
                if (guest_core[i] >= 0) {
                    sample.has_guest = guest_probe(static_cast<unsigned>(guest_core[i]), sample.guest);
                }
            }
            ResumeThread(handle);
        }
        Sleep(1);
    }
    std::string text = Report(targets, samples, out.c_str());
    GuestReport(targets, samples, out);
    GuestThreadReport(targets, samples, out, submits_before);
    std::fprintf(stderr, "nx2 sampler: %zu samples written to %s\n", samples.size(), out.c_str());
    LOG_INFO(Common, "nx2 sampler: {} samples written to {}", samples.size(), out);
    for (const auto& target : targets) {
        CloseHandle(target.handle);
    }
    if (beep) {
        Tone(500, 1200, 400);
    }
    return text;
}

} // Anonymous namespace

std::string Sample(const char* names, double seconds, const std::string& out) {
    return SampleThreads(SplitNames(names), seconds, out);
}

void NoteGuestThread(std::uint64_t id, std::uint64_t creator, std::uint64_t entry,
                     std::uint64_t arg, std::int32_t priority, std::int32_t core, std::uint64_t tls) {
    if (!g_thread_stats.load(std::memory_order_relaxed)) {
        return;
    }
    const std::scoped_lock threads_lock{g_threads_mutex};
    g_guest_threads[id] = GuestThreadInfo{creator, entry, arg, priority, core, tls,
                                          std::chrono::steady_clock::now()};
}

void SetClientThread(std::uint64_t id) {
    t_client_thread = id;
}

void NoteGpuSubmit(std::uint32_t entries) {
    if (!g_thread_stats.load(std::memory_order_relaxed)) {
        return;
    }
    const std::scoped_lock threads_lock{g_threads_mutex};
    SubmitCount& count = g_gpu_submits[t_client_thread];
    ++count.submits;
    count.entries += entries;
}

void SetGuestProbe(std::function<bool(unsigned, GuestSample&)> probe) {
    const std::scoped_lock probe_lock{g_guest_probe_mutex};
    g_guest_probe = std::move(probe);
    if (!g_guest_probe) {
        g_guest_reader = nullptr;
    }
}

void SetGuestMemoryReader(std::function<bool(std::uint64_t, void*, std::size_t)> reader) {
    const std::scoped_lock probe_lock{g_guest_probe_mutex};
    g_guest_reader = std::move(reader);
}

void NoteAudioUnderrun(const char* stream, std::uint32_t queued) {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_AUDIO_LOG");
        return value == nullptr || *value != '0';
    }();
    const auto count = g_audio_underruns.fetch_add(1, std::memory_order_relaxed) + 1;
    if (enabled && count <= 600) {
        LOG_INFO(Common, "nx2 audio underrun #{} at {:.3f} s on {} ({} buffers queued)", count,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - g_process_start)
                     .count(),
                 stream, queued);
    }
}

void NoteAudioRenderGap(double gap_ms, double process_ms) {
    static const double late_ms = [] {
        const char* value = std::getenv("EDEN_NX2_AUDIO_LATE_MS");
        return value != nullptr && *value != '\0' ? std::atof(value) : 20.0;
    }();
    if (gap_ms < late_ms && process_ms < late_ms) {
        return;
    }
    const auto count = g_audio_late_ticks.fetch_add(1, std::memory_order_relaxed) + 1;
    if (count <= 600) {
        LOG_INFO(Common,
                 "nx2 late DSP render tick #{} at {:.3f} s: {:.1f} ms since the previous tick, the "
                 "previous one took {:.1f} ms",
                 count,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - g_process_start)
                     .count(),
                 gap_ms, process_ms);
    }
}

void StartFromEnvironment() {
    if (const char* stall = std::getenv("EDEN_NX2_STALL_MS"); stall != nullptr && *stall != '\0') {
        const auto threshold = static_cast<unsigned>((std::max)(20L, std::strtol(stall, nullptr, 10)));
        std::thread{[threshold] { StallLoop(threshold); }}.detach();
    }
    const char* names = std::getenv("EDEN_NX2_SAMPLE");
    if (names == nullptr || *names == '\0') {
        return;
    }
    g_thread_stats.store(true, std::memory_order_relaxed);
    const char* out = std::getenv("EDEN_NX2_SAMPLE_OUT");
    // EDEN_NX2_SAMPLE_TRIGGER=<file>: the measurement starts when that file appears (it is
    // deleted at once), instead of EDEN_NX2_SAMPLE_START seconds after the launch, so a player
    // can ask for it when the game is where it must be measured.
    const char* trigger = std::getenv("EDEN_NX2_SAMPLE_TRIGGER");
    std::thread{[patterns = SplitNames(names), start = EnvSeconds("EDEN_NX2_SAMPLE_START", 60.0),
                 seconds = EnvSeconds("EDEN_NX2_SAMPLE_SECONDS", 30.0),
                 out = std::string{out != nullptr ? out : "nx2_samples.txt"},
                 trigger = std::string{trigger != nullptr ? trigger : ""}] {
        if (trigger.empty()) {
            std::this_thread::sleep_for(std::chrono::duration<double>(start));
        } else {
            std::error_code error;
            while (!std::filesystem::exists(trigger, error)) {
                std::this_thread::sleep_for(std::chrono::milliseconds{200});
            }
            std::filesystem::remove(trigger, error);
        }
        SampleThreads(patterns, seconds, out);
    }}.detach();
}

void NotePresent(long long frame_us, long long fence_us, long long composite_us, long long pace_us) {
    const auto now = std::chrono::steady_clock::now();
    g_last_present_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count(),
                            std::memory_order_relaxed);
    static const bool log_frames = std::getenv("EDEN_NX2_FRAME_LOG") != nullptr;
    if (!log_frames) {
        return;
    }
    static FrameLog frames;
    const std::scoped_lock lock{frames.mutex};
    if (!frames.opened) {
        frames.opened = true;
        const auto path = Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) / "nx2_frames.csv";
        frames.file.open(path, std::ios::out | std::ios::trunc);
        frames.pending =
            "t_s,frame_ms,fence_ms,composite_ms,pace_ms,astc_decodes,pipelines_built,pipeline_ms,"
            "sparse_binds,sparse_ms,sparse_call_ms,images_created,images_deleted,gc_ms,"
            "tc_used_mib,tc_expected_mib,tc_critical_mib,gc_mode,del_gc,del_gc_download,del_unmap,"
            "del_join,created_replace,gc_finish_ms,list_waits,list_wait_ms,recr_30,recr_150,recr_600,recr_old,ab,draws,indirect,dispatches,writer_waits,writer_wait_ms,drains,drain_ms,back_idles,back_idle_ms,trust_skips,trust_violations,sem_drains,sem_drain_ms,dma_drains,dma_drain_ms,acq_checks,acq_skips,acq_same,acq_needed,acq_waits,acq_wrong,acq_written,acq_exec,resume_polls,resume_skips,audio_underruns,audio_late_ticks\n";
    }
    if (!frames.file.is_open()) {
        return;
    }
    const StallCounters counters = ReadCounters();
    frames.pending += fmt::format("{:.3f},{:.2f},{:.2f},{:.2f},{:.2f},{},{},{:.1f},{},{:.1f},{:.1f},{},{},{:.1f},"
                                  "{},{},{},{},{},{},{},{},{},{:.1f},{},{:.1f},{},{},{},{},{},{},{},{},{},{:.1f},{},{:.1f},{},{:.1f},{},{},{},{:.1f},{},{:.1f},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                                  std::chrono::duration<double>(now - g_process_start).count(),
                                  frame_us / 1000.0, fence_us / 1000.0, composite_us / 1000.0,
                                  pace_us / 1000.0, counters.astc, counters.pipelines,
                                  counters.pipeline_us / 1000.0, counters.sparse_binds,
                                  counters.sparse_us / 1000.0, counters.sparse_call_us / 1000.0, counters.created,
                                  counters.deleted, counters.gc_us / 1000.0,
                                  g_tc_used_mib.load(std::memory_order_relaxed),
                                  g_tc_expected_mib.load(std::memory_order_relaxed),
                                  g_tc_critical_mib.load(std::memory_order_relaxed),
                                  g_tc_mode.load(std::memory_order_relaxed), counters.del_gc,
                                  counters.del_gc_download, counters.del_unmap, counters.del_join,
                                  counters.created_replace, counters.gc_finish_us / 1000.0,
                                  counters.list_waits, counters.list_wait_us / 1000.0,
                                  g_recreated_30.load(std::memory_order_relaxed),
                                  g_recreated_150.load(std::memory_order_relaxed),
                                  g_recreated_600.load(std::memory_order_relaxed),
                                  g_recreated_old.load(std::memory_order_relaxed), AbLabel(),
                                  g_draws.load(std::memory_order_relaxed),
                                  g_indirect_draws.load(std::memory_order_relaxed),
                                  g_dispatches.load(std::memory_order_relaxed),
                                  g_writer_waits.load(std::memory_order_relaxed),
                                  g_writer_wait_us.load(std::memory_order_relaxed) / 1000.0,
                                  g_drains.load(std::memory_order_relaxed),
                                  g_drain_us.load(std::memory_order_relaxed) / 1000.0,
                                  g_back_idles.load(std::memory_order_relaxed),
                                  g_back_idle_us.load(std::memory_order_relaxed) / 1000.0,
                                  g_trust_skips.load(std::memory_order_relaxed),
                                  g_clean_violations.load(std::memory_order_relaxed),
                                  g_sem_drains.load(std::memory_order_relaxed),
                                  g_sem_drain_us.load(std::memory_order_relaxed) / 1000.0,
                                  g_dma_drains.load(std::memory_order_relaxed),
                                  g_dma_drain_us.load(std::memory_order_relaxed) / 1000.0,
                                  g_acq_checks.load(std::memory_order_relaxed),
                                  g_acq_skips.load(std::memory_order_relaxed),
                                  g_acq_same.load(std::memory_order_relaxed),
                                  g_acq_needed.load(std::memory_order_relaxed),
                                  g_acq_waits.load(std::memory_order_relaxed),
                                  g_acq_wrong.load(std::memory_order_relaxed),
                                  g_acq_written.load(std::memory_order_relaxed),
                                  g_acq_exec.load(std::memory_order_relaxed),
                                  g_resume_polls.load(std::memory_order_relaxed),
                                  g_resume_skips.load(std::memory_order_relaxed),
                                  g_audio_underruns.load(std::memory_order_relaxed),
                                  g_audio_late_ticks.load(std::memory_order_relaxed));
    if (++frames.lines % 150 == 0) {
        frames.Flush();
    }
}

} // namespace Nx2Sampler

#else

namespace Nx2Sampler {
void SetGuestProbe(std::function<bool(unsigned, GuestSample&)>) {}
void SetGuestMemoryReader(std::function<bool(std::uint64_t, void*, std::size_t)>) {}
void NoteGuestThread(std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::int32_t,
                     std::int32_t, std::uint64_t) {}
void SetClientThread(std::uint64_t) {}
void NoteGpuSubmit(std::uint32_t) {}
void StartFromEnvironment() {}
void NoteAudioUnderrun(const char*, std::uint32_t) {}
void NoteAudioRenderGap(double, double) {}
std::string Sample(const char*, double, const std::string&) {
    return "the sampler is Windows only";
}
void NotePresent(long long, long long, long long, long long) {}
} // namespace Nx2Sampler

#endif
