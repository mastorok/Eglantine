// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "common/settings.h"

namespace VideoCommon::WritebackSwitches {

inline constexpr u64 DuskbloodsNetworkTestProgramId = 0x0400D1D02399A000ULL;
inline constexpr u64 KirbyForgottenLandNs2eProgramId = 0x04004D300C5AE001ULL;

/// Titles whose collector and join write-backs and stale-page uploads are on: they sample
/// GPU-written images the collector would drop, which otherwise show recycled host memory
/// (The Duskbloods' rainbow-noise characters, Kirby's garbled loading text, #2422).
[[nodiscard]] constexpr bool WritebackDefault(u64 program_id) {
    return program_id == DuskbloodsNetworkTestProgramId ||
           program_id == KirbyForgottenLandNs2eProgramId;
}

[[nodiscard]] inline bool WritebackDefault() {
    return WritebackDefault(Settings::GetCurrentProgramID());
}

[[nodiscard]] inline bool AnyWritebackOn() {
    return WritebackDefault();
}

[[nodiscard]] inline bool UploadStale() {
    return WritebackDefault();
}

} // namespace VideoCommon::WritebackSwitches
