// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <memory>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "common/scope_exit.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/service/nvdrv/core/syncpoint_manager.h"
#include "tests/core/system_test_context.h"
#include "video_core/host1x/host1x.h"

// Adapted from Eglantine PR #695, 87d9d59aec88aedeec475becfc39c4ede537fb45.
TEST_CASE("Syncpoint exhaustion keeps ID zero free and permits safe reuse", "[syncpoint_allocator]") {
    ScopedSystemTestContext context;
    Core::System system;
    SCOPE_EXIT {
        system.CoreTiming().Reset();
    };
    system.Initialize();
    auto host = std::make_unique<Tegra::Host1x::Host1x>(system);
    Service::Nvidia::NvCore::SyncpointManager allocator{*host};
    std::vector<u32> allocated;
    const u32 required_id = allocator.AllocateSyncpoint(false);
    REQUIRE(required_id != 0);
    REQUIRE(allocator.IsSyncpointAllocated(required_id));
    allocated.push_back(required_id);
    while (const auto id = allocator.TryAllocateSyncpoint(true)) {
        REQUIRE(*id != 0);
        allocated.push_back(*id);
    }
    REQUIRE_FALSE(allocator.TryAllocateSyncpoint(true).has_value());
    REQUIRE_FALSE(allocator.IsSyncpointAllocated(0));
    allocator.FreeSyncpoint(allocated.back());
    REQUIRE(allocator.TryAllocateSyncpoint(true) == allocated.back());
    REQUIRE_FALSE(allocator.TryAllocateSyncpoint(true).has_value());
    REQUIRE_FALSE(allocator.IsSyncpointAllocated(0));
}
