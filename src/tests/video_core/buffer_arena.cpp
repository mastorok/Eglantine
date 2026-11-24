// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/buffer_cache/command_arena.h"

namespace {
struct Small {
    u8 value;
};
struct Wide {
    u64 a;
    u32 b;
};
} // Anonymous namespace

TEST_CASE("Command arena plays mixed records in order", "[buffer_arena]") {
    VideoCommon::CommandArena arena;
    for (int round = 0; round < 2; ++round) {
        for (u32 i = 0; i < 10000; ++i) {
            switch (i % 3) {
            case 0:
                arena.Push(0, Small{static_cast<u8>(i)});
                break;
            case 1:
                arena.Push(1, Wide{i * 3ULL, i});
                break;
            default: {
                const std::vector<u32> words(i % 7, i);
                arena.Push(2, Small{static_cast<u8>(i)}, std::span<const u32>{words});
                break;
            }
            }
        }
        REQUIRE(arena.Records() == 10000);
        REQUIRE(arena.Bytes() % 8 == 0);
        u32 i = 0;
        bool ok = true;
        arena.ForEach([&](u16 kind, const u8* record) {
            ok = ok && kind == i % 3;
            switch (kind) {
            case 0:
                ok = ok && VideoCommon::CommandArena::Body<Small>(record).value == u8(i);
                break;
            case 1: {
                const auto wide = VideoCommon::CommandArena::Body<Wide>(record);
                ok = ok && wide.a == i * 3ULL && wide.b == i;
                break;
            }
            default: {
                const auto words = VideoCommon::CommandArena::Array<Small, u32>(record);
                ok = ok && words.size() == i % 7;
                for (const u32 word : words) {
                    ok = ok && word == i;
                }
                break;
            }
            }
            ++i;
        });
        REQUIRE(ok);
        REQUIRE(i == 10000);
        const size_t bytes = arena.Bytes();
        VideoCommon::CommandArena played;
        played.Swap(arena);
        REQUIRE(arena.Empty());
        REQUIRE(played.Bytes() == bytes);
        played.Clear();
        REQUIRE(played.Empty());
        REQUIRE(played.Records() == 0);
    }
}

TEST_CASE("Arena clock frees a stamp only once its mapped tick is done", "[buffer_arena]") {
    VideoCommon::ArenaClock clock;
    u64 done_through = 0;
    const auto done = [&](u64 tick) { return tick <= done_through; };
    const u64 first = clock.Now();
    REQUIRE_FALSE(clock.IsFree(first, done));
    clock.Map(5);
    REQUIRE(clock.Now() == first + 1);
    REQUIRE_FALSE(clock.IsFree(first, done));
    done_through = 5;
    REQUIRE(clock.IsFree(first, done));
    REQUIRE_FALSE(clock.IsFree(clock.Now(), done));

    // A stamp far behind the window answers with the oldest tick kept, which is never older.
    for (u64 i = 0; i < VideoCommon::ArenaClock::Window * 3; ++i) {
        clock.Map(6 + i / 100);
    }
    done_through = 6;
    REQUIRE_FALSE(clock.IsFree(first, done));
    done_through = 1000;
    REQUIRE(clock.IsFree(first, done));
    REQUIRE(clock.IsFree(clock.Now() - 1, done));
}

TEST_CASE("Arena clock names the tick of a closed stamp once its player maps it",
          "[buffer_arena]") {
    // EDEN_NX2_RENDER_LIST_WORDS_E: an owed word noted with Now() waits for the tick its posted
    // copies were played in.
    VideoCommon::ArenaClock clock;
    const u64 noted = clock.Now();
    const u64 closed = clock.Close();
    REQUIRE(closed == noted);
    REQUIRE_FALSE(clock.TickOf(closed).has_value());
    clock.Map(closed, 7);
    REQUIRE(clock.TickOf(closed) == 7U);
    REQUIRE_FALSE(clock.TickOf(clock.Now()).has_value());
}

TEST_CASE("Arena clock stamps map from another thread", "[buffer_arena]") {
    VideoCommon::ArenaClock clock;
    constexpr u64 Maps = 100000;
    std::jthread mapper{[&] {
        for (u64 i = 1; i <= Maps; ++i) {
            clock.Map(i);
        }
    }};
    bool ok = true;
    while (clock.Now() <= Maps) {
        const u64 stamp = clock.Now() - 1;
        u64 seen = 0;
        ok = ok && clock.IsFree(stamp, [&](u64 tick) {
            seen = tick;
            return true;
        });
        ok = ok && seen >= stamp;
        ok = ok && !clock.IsFree(Maps + 1, [](u64) { return true; });
    }
    mapper.join();
    REQUIRE(ok);
}
