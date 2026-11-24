// SPDX-FileCopyrightText: Copyright 2026 Lichii contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "core/core.h"
#include "core/hle/service/am/window_system.h"
#include "tests/core/system_test_context.h"

TEST_CASE("An empty window system requests exit once per session", "[core][am][window_system]") {
    ScopedSystemTestContext environment;
    Core::System system;
    int exit_requests = 0;
    system.RegisterExitCallback([&] {
        ++exit_requests;
        system.SetShuttingDown(true);
    });

    for (int session = 1; session <= 2; ++session) {
        CAPTURE(session);
        system.SetShuttingDown(false);
        Service::AM::WindowSystem window_system{system};
        window_system.Update();
        REQUIRE(system.IsShuttingDown());
        CHECK(exit_requests == session);

        for (int update = 0; update < 4096; ++update) {
            window_system.Update();
        }
        CHECK(exit_requests == session);
    }
}
