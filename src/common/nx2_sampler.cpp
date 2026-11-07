// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/nx2_sampler.h"

#ifdef _WIN32

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"

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
};

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
// could be paused holding the heap lock.
unsigned Walk(CONTEXT context, std::array<DWORD64, MaxDepth>& frames) {
    unsigned depth = 0;
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
    return depth;
}

struct Symbolizer {
    using SymInitializeFn = BOOL(WINAPI*)(HANDLE, PCSTR, BOOL);
    using SymSetOptionsFn = DWORD(WINAPI*)(DWORD);
    using SymFromAddrFn = BOOL(WINAPI*)(HANDLE, DWORD64, PDWORD64, PSYMBOL_INFO);

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
        if (set_options == nullptr || initialize == nullptr || from_address == nullptr) {
            from_address = nullptr;
            return;
        }
        set_options(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
        if (!initialize(GetCurrentProcess(), nullptr, TRUE)) {
            from_address = nullptr;
        }
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

    SymFromAddrFn from_address{};
    std::unordered_map<DWORD64, std::string> cache;
};

std::string Report(const std::vector<Target>& targets, const std::vector<Sample>& samples,
                   const char* path) {
    Symbolizer symbolizer;
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
            }
            ResumeThread(handle);
        }
        Sleep(1);
    }
    std::string text = Report(targets, samples, out.c_str());
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

void StartFromEnvironment() {
    const char* names = std::getenv("EDEN_NX2_SAMPLE");
    if (names == nullptr || *names == '\0') {
        return;
    }
    const char* out = std::getenv("EDEN_NX2_SAMPLE_OUT");
    std::thread{[patterns = SplitNames(names), start = EnvSeconds("EDEN_NX2_SAMPLE_START", 60.0),
                 seconds = EnvSeconds("EDEN_NX2_SAMPLE_SECONDS", 30.0),
                 out = std::string{out != nullptr ? out : "nx2_samples.txt"}] {
        std::this_thread::sleep_for(std::chrono::duration<double>(start));
        SampleThreads(patterns, seconds, out);
    }}.detach();
}

} // namespace Nx2Sampler

#else

namespace Nx2Sampler {
void StartFromEnvironment() {}
std::string Sample(const char*, double, const std::string&) {
    return "the sampler is Windows only";
}
} // namespace Nx2Sampler

#endif
