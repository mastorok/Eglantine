// SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// FRINT* on half-precision vectors goes through a C++ fallback on the arm64 backend. The same
// block must give the same, correct lanes every time it runs.

#include <array>
#include <cstdint>
#include <optional>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/exclusive_monitor.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace {

using namespace Dynarmic;

constexpr u64 CodeStart = 0x1000;
constexpr u32 FrintnV15V5 = 0x0E7988AF; // frintn v15.4h, v5.4h
constexpr u32 Svc0 = 0xD4000001;

class RoundIntEnv final : public A64::UserCallbacks {
public:
    A64::Jit* jit = nullptr;

    std::optional<std::uint32_t> MemoryReadCode(u64 vaddr) override {
        return vaddr == CodeStart ? FrintnV15V5 : Svc0;
    }
    u64 MemoryRead(u64, size_t) override { return 0; }
    A64::Vector MemoryRead128(u64) override { return {}; }
    void MemoryWrite(u64, u64, size_t) override {}
    void MemoryWrite128(u64, A64::Vector) override {}
    void CallSVC(std::uint32_t) override { jit->HaltExecution(); }
    void ExceptionRaised(u64, A64::Exception) override {}
    void AddTicks(u64) override {}
    u64 GetTicksRemaining() override { return 0; }
    u64 GetCNTPCT() override { return 0; }
};

} // Anonymous namespace

TEST_CASE("JIT: FRINTN on half-precision lanes gives the same result every run", "[jit][fp16]") {
    RoundIntEnv env;
    ExclusiveMonitor monitor{1};
    A64::UserConfig config{};
    config.callbacks = &env;
    config.global_monitor = &monitor;
    config.enable_cycle_counting = false;
    A64::Jit jit{config};
    env.jit = &jit;

    // 1.5, 2.5, -1.5 and 0.4 round to 2, 2, -2 and 0 with ties to even.
    const A64::Vector input{0x3666'BE00'4100'3E00, 0};
    const A64::Vector expected{0x0000'C000'4000'4000, 0};
    for (int run = 0; run < 8; ++run) {
        if (run % 2 == 0) {
            jit.ClearCache();
        }
        std::array<A64::Vector, 32> vectors{};
        for (size_t i = 0; i < vectors.size(); ++i) {
            vectors[i] = {0x5A5A'5A5A'0000'0000 | i, 0xA5A5'A5A5'0000'0000 | i};
        }
        vectors[5] = input;
        jit.SetVectors(vectors);
        jit.SetPC(CodeStart);
        jit.SetFpcr(0);
        jit.SetFpsr(0);
        jit.Run();
        INFO("run " << run);
        CHECK(jit.GetVectors()[15] == expected);
    }
}
