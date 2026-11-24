// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <latch>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/nx2_render_follower.h"

namespace {

using VideoCommon::Nx2RenderFollower;

void PlayUntilStop(Nx2RenderFollower& follower) {
    while (follower.PlayNext() != Nx2RenderFollower::Kind::Stop) {
    }
}

} // Anonymous namespace

TEST_CASE("Render follower keeps a dequeued packet pending until it plays",
          "[nx2_render_follower]") {
    Nx2RenderFollower follower;
    std::latch executing{1};
    std::latch release{1};
    std::vector<size_t> played;
    std::jthread render{[&] { PlayUntilStop(follower); }};
    const auto engine_id = std::this_thread::get_id();
    bool on_render = false;
    follower.Post(0, [&] {
        on_render = std::this_thread::get_id() != engine_id;
        executing.count_down();
        release.wait();
        played.push_back(0);
    });
    executing.wait();
    const bool pending_while_executing = follower.Pending();
    release.count_down();
    follower.Wait();
    const bool pending_after_wait = follower.Pending();
    for (size_t sequence = 1; sequence <= 256; ++sequence) {
        follower.Post(sequence, [&, sequence] { played.push_back(sequence); });
    }
    follower.Stop();
    render.join();

    REQUIRE(on_render);
    REQUIRE(pending_while_executing);
    REQUIRE_FALSE(pending_after_wait);
    REQUIRE(played.size() == 257);
    for (size_t sequence = 0; sequence < played.size(); ++sequence) {
        CHECK(played[sequence] == sequence);
    }
}

TEST_CASE("Render follower hands back-to-back packets over without parking while it polls",
          "[nx2_render_follower]") {
    Nx2RenderFollower follower{std::chrono::seconds{1}};
    std::vector<size_t> played;
    std::jthread render{[&] { PlayUntilStop(follower); }};
    size_t parked = 0;
    for (size_t sequence = 0; sequence < 4096; ++sequence) {
        follower.Post(sequence, [&, sequence] { played.push_back(sequence); });
        parked += follower.Wait() ? 1 : 0;
    }
    follower.Stop();
    render.join();

    CHECK(parked == 0);
    REQUIRE(played.size() == 4096);
    for (size_t sequence = 0; sequence < played.size(); ++sequence) {
        CHECK(played[sequence] == sequence);
    }
}

TEST_CASE("Render follower wakes a parked peer on both sides", "[nx2_render_follower]") {
    Nx2RenderFollower follower{std::chrono::nanoseconds{0}};
    std::latch executing{1};
    std::latch release{1};
    size_t played = 0;
    std::jthread render{[&] { PlayUntilStop(follower); }};
    follower.Post(0, [&] {
        executing.count_down();
        release.wait();
        ++played;
    });
    executing.wait();
    std::jthread releaser{[&] {
        std::this_thread::sleep_for(std::chrono::milliseconds{20});
        release.count_down();
    }};
    const bool parked = follower.Wait();
    for (size_t sequence = 0; sequence < 1024; ++sequence) {
        if (sequence % 64 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds{200});
        }
        follower.Post(sequence, [&] { ++played; });
        follower.Wait();
    }
    follower.Stop();
    render.join();

    CHECK(parked);
    CHECK(played == 1025);
}

TEST_CASE("Render follower ring keeps 10k mixed packets in order across wraps",
          "[nx2_render_follower]") {
    // A 4 KiB ring: the packets below wrap it hundreds of times, with many in flight at once.
    Nx2RenderFollower follower{std::chrono::microseconds{5}, true, 4096};
    std::vector<u64> played;
    played.reserve(10000);
    size_t alive = 0;
    size_t lends = 0;
    size_t lends_in_order = 0;
    struct Small {
        std::vector<u64>* out;
        u64 sequence;
        void operator()() const {
            out->push_back(sequence);
        }
    };
    struct Large {
        std::vector<u64>* out;
        size_t* alive;
        u64 sequence;
        std::array<u64, 40> words;
        std::unique_ptr<u64> owned;
        void operator()() const {
            bool intact = *owned == sequence;
            for (const u64 word : words) {
                intact = intact && word == sequence * 3;
            }
            out->push_back(intact ? sequence : ~u64{0});
            --*alive;
        }
    };
    std::jthread render{[&] { PlayUntilStop(follower); }};
    for (u64 sequence = 0; sequence < 10000; ++sequence) {
        switch (sequence % 7) {
        case 3: {
            Large large{&played, &alive, sequence, {}, std::make_unique<u64>(sequence)};
            large.words.fill(sequence * 3);
            ++alive;
            follower.Post(sequence, std::move(large));
            break;
        }
        case 6: {
            struct Probe {
                std::vector<u64>* out;
                u64 sequence;
                size_t* in_order;
            } probe{&played, sequence, &lends_in_order};
            follower.Lend(
                sequence,
                [](void* argument) {
                    auto& lent = *static_cast<Probe*>(argument);
                    *lent.in_order += lent.out->size() == lent.sequence ? 1 : 0;
                    lent.out->push_back(lent.sequence);
                },
                &probe);
            ++lends;
            break;
        }
        default:
            follower.Post(sequence, Small{&played, sequence});
            break;
        }
    }
    follower.Stop();
    render.join();

    REQUIRE(played.size() == 10000);
    for (u64 sequence = 0; sequence < played.size(); ++sequence) {
        CHECK(played[sequence] == sequence);
    }
    CHECK(alive == 0);
    CHECK(lends_in_order == lends);
    CHECK(follower.Lent() == lends);
    CHECK(follower.OutOfOrder() == 0);
    CHECK(follower.Wraps() > 100);
    CHECK(follower.MaxDepth() > 1);
}

TEST_CASE("Render follower lends a call to GPURender and returns its result and exception",
          "[nx2_render_follower]") {
    Nx2RenderFollower follower;
    std::jthread render{[&] { PlayUntilStop(follower); }};
    struct Call {
        std::thread::id thread;
        int value;
    } call{{}, 1};
    follower.Lend(
        0,
        [](void* argument) {
            auto& lent = *static_cast<Call*>(argument);
            lent.thread = std::this_thread::get_id();
            lent.value *= 42;
        },
        &call);
    const bool pending_after = follower.Pending();
    bool thrown = false;
    try {
        follower.Lend(1, [](void*) { throw std::runtime_error{"lent"}; }, nullptr);
    } catch (const std::runtime_error&) {
        thrown = true;
    }
    const auto render_id = render.get_id();
    follower.Stop();
    render.join();

    CHECK(call.thread == render_id);
    CHECK(call.thread != std::this_thread::get_id());
    CHECK(call.value == 42);
    CHECK_FALSE(pending_after);
    CHECK(thrown);
}

TEST_CASE("Render follower floor is the oldest unplayed packet's position",
          "[nx2_render_follower]") {
    Nx2RenderFollower follower{Nx2RenderFollower::DefaultSpin, true};
    std::latch executing{1};
    std::latch release{1};
    const u64 empty_floor = follower.Floor();
    std::jthread render{[&] { PlayUntilStop(follower); }};
    follower.Post(100, [&] {
        executing.count_down();
        release.wait();
    });
    follower.Post(200, [] {});
    executing.wait();
    const u64 blocked_floor = follower.Floor();
    release.count_down();
    follower.Wait();
    const u64 played_floor = follower.Floor();
    follower.Stop();
    render.join();

    CHECK(empty_floor == ~u64{0});
    CHECK(blocked_floor == 100);
    CHECK(played_floor == ~u64{0});
}
