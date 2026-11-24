// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>

#include "common/common_types.h"
#include "common/settings.h"

namespace VideoCommon::WritebackSwitches {

inline constexpr u64 DuskbloodsNetworkTestProgramId = 0x0400D1D02399A000ULL;
inline constexpr u64 KirbyForgottenLandNs2eProgramId = 0x04004D300C5AE001ULL;
inline constexpr u64 OcarinaOfTimeProgramId = 0x0400486024DEC000ULL;

/// Titles whose collector and join write-backs and stale-page uploads are on: they sample
/// GPU-written images the collector would drop, which otherwise show recycled host memory
/// (The Duskbloods' rainbow-noise characters, Kirby's garbled loading text, #2422).
/// Not Ocarina of Time: it allocates about nine render targets a frame at shifting offsets of a
/// transient heap, every one overlapping older ones, so the write-backs downloaded and swizzled
/// multi-MiB targets nobody samples again (build 12: 44 -> 32 fps median, black textures still
/// there). EDEN_NX2_WRITEBACK=1 turns them on for a test.
[[nodiscard]] constexpr bool WritebackDefault(u64 program_id) {
    return program_id == DuskbloodsNetworkTestProgramId ||
           program_id == KirbyForgottenLandNs2eProgramId;
}

/// EDEN_NX2_WRITEBACK=0 or 1 forces the write-backs off or on for every title.
[[nodiscard]] inline bool WritebackDefault() {
    static const int forced = [] {
        const char* const value = std::getenv("EDEN_NX2_WRITEBACK");
        return value == nullptr || value[0] == '\0' ? -1 : (value[0] == '0' ? 0 : 1);
    }();
    if (forced >= 0) {
        return forced == 1;
    }
    return WritebackDefault(Settings::GetCurrentProgramID());
}

[[nodiscard]] inline bool AnyWritebackOn() {
    return WritebackDefault();
}

[[nodiscard]] inline bool UploadStale() {
    return WritebackDefault();
}

} // namespace VideoCommon::WritebackSwitches
