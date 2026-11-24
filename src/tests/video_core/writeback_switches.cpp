// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/texture_cache/writeback_switches.h"

TEST_CASE("Overlap write-backs stay on only for the titles that need them",
          "[video_core][texture_cache][writeback]") {
    using namespace VideoCommon::WritebackSwitches;
    REQUIRE(WritebackDefault(DuskbloodsNetworkTestProgramId));
    REQUIRE(WritebackDefault(KirbyForgottenLandNs2eProgramId));
    REQUIRE_FALSE(WritebackDefault(0));
    REQUIRE_FALSE(WritebackDefault(0x01004D300C5AE000ULL));
    REQUIRE_FALSE(WritebackDefault(0x04007EF01F8AE000ULL));
    REQUIRE(WritebackDefault(OcarinaOfTimeProgramId));
    REQUIRE_FALSE(WritebackDefault(0x0400C3F00006E000ULL));
}
