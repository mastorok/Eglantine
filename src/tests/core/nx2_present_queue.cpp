// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "core/core_timing.h"
#include "core/hle/service/nx2_present_queue.h"

namespace {
using namespace std::chrono_literals;
using Service::Nx2PresentReleaseQueue;
using Service::Nx2PresentReleases;
constexpr u32 ReleaseId = 1;

struct Fixture {
    Fixture() {
        timing.SetMulticore(false);
        timing.Initialize([] {});
    }

    /// Moves single-core CoreTiming past every deadline these tests use and runs due events.
    void Advance() {
        timing.AddTicks(1'000'000'000);
        timing.Advance();
    }

    u32 Released() const {
        return syncpoints.GetGuestSyncpointValue(ReleaseId);
    }

    Core::Timing::CoreTiming timing;
    Tegra::Host1x::SyncpointManager syncpoints;
};
} // Anonymous namespace

TEST_CASE("A present release needs both its deadline and its composite", "[nx2_present]") {
    Fixture f;
    const auto queue = std::make_shared<Nx2PresentReleaseQueue>(f.timing, f.syncpoints, ReleaseId);
    REQUIRE(queue->Queue(f.timing.GetGlobalTimeNs() + 10ms) == 1U);

    SECTION("Composited before the deadline") {
        queue->Composited(1);
        f.timing.Advance();
        REQUIRE(f.Released() == 0);
        f.Advance();
    }
    SECTION("Deadline passes before the composite") {
        f.Advance();
        REQUIRE(f.Released() == 0);
        // A reached GPU fence alone does not release: only the composite does.
        f.Advance();
        REQUIRE(f.Released() == 0);
        queue->Composited(1);
    }
    REQUIRE(f.Released() == 1);
    REQUIRE_FALSE(f.timing.Advance().has_value());
}

TEST_CASE("Present releases stay in present order", "[nx2_present]") {
    Fixture f;
    const auto queue = std::make_shared<Nx2PresentReleaseQueue>(f.timing, f.syncpoints, ReleaseId);
    const auto now = f.timing.GetGlobalTimeNs();
    REQUIRE(queue->Queue(now) == 1U);
    REQUIRE(queue->Queue(now) == 2U);
    queue->Composited(2);
    REQUIRE(f.Released() == 0);
    queue->Composited(1);
    REQUIRE(f.Released() == 2);
    // Values no pending release has are ignored.
    queue->Composited(7);
    REQUIRE(f.Released() == 2);
}

TEST_CASE("Present releases signal in order while another thread composites",
          "[nx2_present]") {
    Fixture f;
    const auto queue = std::make_shared<Nx2PresentReleaseQueue>(f.timing, f.syncpoints, ReleaseId);
    const auto now = f.timing.GetGlobalTimeNs();
    REQUIRE(queue->Queue(now) == 1U);
    REQUIRE(queue->Queue(now) == 2U);
    // A guest action due at release 1 runs inside its increment. The GPU thread composites
    // release 2 meanwhile: it must leave the signal to the thread already signalling 1.
    u32 after_join = 0;
    f.syncpoints.RegisterGuestAction(ReleaseId, 1, [&] {
        std::thread gpu{[&] { queue->Composited(2); }};
        gpu.join();
        after_join = f.Released();
    });
    queue->Composited(1);
    REQUIRE(after_join == 1U);
    REQUIRE(f.Released() == 2U);
}

TEST_CASE("A guest action on the signalling thread may composite the next release",
          "[nx2_present]") {
    Fixture f;
    const auto queue = std::make_shared<Nx2PresentReleaseQueue>(f.timing, f.syncpoints, ReleaseId);
    const auto now = f.timing.GetGlobalTimeNs();
    REQUIRE(queue->Queue(now) == 1U);
    REQUIRE(queue->Queue(now) == 2U);
    f.syncpoints.RegisterGuestAction(ReleaseId, 1, [&] { queue->Composited(2); });
    queue->Composited(1);
    REQUIRE(f.Released() == 2U);
}

TEST_CASE("Present releases hold no more presents than the queue has slots", "[nx2_present]") {
    Fixture f;
    const auto queue = std::make_shared<Nx2PresentReleaseQueue>(f.timing, f.syncpoints, ReleaseId);
    const auto now = f.timing.GetGlobalTimeNs();
    for (u32 value = 1; value <= Nx2PresentReleaseQueue::Capacity; ++value) {
        REQUIRE(queue->Queue(now) == value);
    }
    REQUIRE_FALSE(queue->Queue(now).has_value());
    for (u32 value = 1; value <= Nx2PresentReleaseQueue::Capacity; ++value) {
        queue->Composited(value);
    }
    REQUIRE(f.Released() == Nx2PresentReleaseQueue::Capacity);
    REQUIRE(queue->Queue(now) == 16U);
}

TEST_CASE("A present holds until the newest release's deadline", "[nx2_present]") {
    Fixture f;
    const auto queue = std::make_shared<Nx2PresentReleaseQueue>(f.timing, f.syncpoints, ReleaseId);
    REQUIRE(queue->RunAhead() == 0ns);
    REQUIRE(queue->Queue(f.timing.GetGlobalTimeNs() + 10ms) == 1U);
    REQUIRE(queue->RunAhead() == 10ms);
    f.Advance();
    // Past its deadline, an uncomposited release no longer holds the next present.
    REQUIRE(queue->RunAhead() == 0ns);
}

TEST_CASE("Stopping a present queue wakes only the releases still pending", "[nx2_present]") {
    Fixture f;
    u32 expected = 2;
    std::function<void()> late_composite;
    {
        const auto queue =
            std::make_shared<Nx2PresentReleaseQueue>(f.timing, f.syncpoints, ReleaseId);
        const auto deadline = f.timing.GetGlobalTimeNs() + 10ms;
        REQUIRE(queue->Queue(deadline) == 1U);
        REQUIRE(queue->Queue(deadline) == 2U);
        queue->Composited(1);
        late_composite = queue->OnComposited(2);
        SECTION("Outstanding releases are signalled") {}
        SECTION("Counter already reached the target") {
            f.syncpoints.IncrementGuest(ReleaseId);
            f.syncpoints.IncrementGuest(ReleaseId);
        }
        SECTION("Counter already passed the target") {
            expected = 3;
            for (int i = 0; i < 3; ++i) {
                f.syncpoints.IncrementGuest(ReleaseId);
            }
        }
        queue->Stop();
        REQUIRE(f.Released() == expected);
    }
    // The renderer may composite after the session is gone.
    late_composite();
    REQUIRE_FALSE(f.timing.Advance().has_value());
    f.Advance();
    REQUIRE(f.Released() == expected);
}

TEST_CASE("A present queue continues an existing syncpoint", "[nx2_present]") {
    Fixture f;
    for (u32 i = 0; i < 7; ++i) {
        f.syncpoints.IncrementGuest(ReleaseId);
    }
    const auto queue = std::make_shared<Nx2PresentReleaseQueue>(f.timing, f.syncpoints, ReleaseId);
    REQUIRE(queue->Queue(f.timing.GetGlobalTimeNs()) == 8U);
    queue->Composited(8);
    REQUIRE(f.Released() == 8U);
}

namespace {
struct Session : Fixture {
    Nx2PresentReleases Releases() {
        return Nx2PresentReleases{
            timing, syncpoints,
            [this]() -> std::optional<u32> {
                ++allocations;
                return allocatable ? std::optional<u32>{ReleaseId} : std::nullopt;
            },
            [this](u32 id) {
                freed.push_back(id);
                released_when_freed = Released();
            }};
    }

    bool allocatable = true;
    int allocations = 0;
    std::vector<u32> freed;
    u32 released_when_freed = 0;
};
} // Anonymous namespace

TEST_CASE("A session keeps blocking pacing while no release syncpoint is free", "[nx2_present]") {
    Session s;
    s.allocatable = false;
    {
        auto releases = s.Releases();
        REQUIRE(releases.RunAhead() == 0ns);
        REQUIRE_FALSE(releases.Queue(0ns).has_value());
        REQUIRE_FALSE(releases.Queue(0ns).has_value());
        REQUIRE(s.allocations == 2);

        // A syncpoint freed later is taken on the next present.
        s.allocatable = true;
        const auto release = releases.Queue(0ns);
        REQUIRE(release.has_value());
        REQUIRE(release->id == ReleaseId);
        REQUIRE(release->value == 1U);
        REQUIRE(s.allocations == 3);
        release->on_composited();
        REQUIRE(s.Released() == 1U);
    }
    REQUIRE(s.freed == std::vector<u32>{ReleaseId});
}

TEST_CASE("A session releases a zero or late deadline once composited", "[nx2_present]") {
    Session s;
    auto releases = s.Releases();
    std::chrono::nanoseconds wait{};
    SECTION("Zero deadline") {}
    SECTION("Late deadline") {
        wait = -5ms;
    }
    const auto release = releases.Queue(wait);
    REQUIRE(release.has_value());
    REQUIRE(releases.RunAhead() == 0ns);
    REQUIRE(s.Released() == 0U);
    release->on_composited();
    REQUIRE(s.Released() == 1U);
}

TEST_CASE("A session holds a future deadline after the composite", "[nx2_present]") {
    Session s;
    auto releases = s.Releases();
    const auto release = releases.Queue(10ms);
    REQUIRE(release.has_value());
    REQUIRE(releases.RunAhead() == 10ms);
    release->on_composited();
    s.timing.Advance();
    REQUIRE(s.Released() == 0U);
    s.Advance();
    REQUIRE(s.Released() == 1U);
}

TEST_CASE("A session falls back to blocking pacing while its release queue is full",
          "[nx2_present]") {
    Session s;
    auto releases = s.Releases();
    std::vector<Nx2PresentReleases::Release> queued;
    for (size_t i = 0; i < Nx2PresentReleaseQueue::Capacity; ++i) {
        auto release = releases.Queue(0ns);
        REQUIRE(release.has_value());
        queued.push_back(std::move(*release));
    }
    REQUIRE_FALSE(releases.Queue(0ns).has_value());
    REQUIRE(s.allocations == 1);
    for (auto& release : queued) {
        release.on_composited();
    }
    REQUIRE(s.Released() == Nx2PresentReleaseQueue::Capacity);
    REQUIRE(releases.Queue(0ns).has_value());
}

TEST_CASE("A closed session wakes its releases before freeing their syncpoint", "[nx2_present]") {
    Session s;
    std::function<void()> late_composite;
    {
        auto releases = s.Releases();
        const auto release = releases.Queue(10ms);
        REQUIRE(release.has_value());
        late_composite = release->on_composited;
    }
    REQUIRE(s.freed == std::vector<u32>{ReleaseId});
    REQUIRE(s.released_when_freed == 1U);
    late_composite();
    REQUIRE(s.Released() == 1U);
}
