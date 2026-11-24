// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// What the settings need of FlyHigh frame generation without its library's header: where its
// model lives and which files it is.

#pragma once

#include <array>
#include <filesystem>
#include <string_view>

#include "common/common_types.h"

namespace VideoCore::FrameGen {

/// The model, downloaded on request from the engine's repository at a pinned release.
namespace FlyHighModel {
inline constexpr std::string_view DOWNLOAD_HOST = "https://raw.githubusercontent.com";
/// flyhigh-engine at tag v0.3.7.
inline constexpr std::string_view DOWNLOAD_PATH =
    "/getflyhigh/flyhigh-engine/18b20146c03d45f888937bd40251c95af91bbc90/models/";
inline constexpr std::string_view RELEASE_PAGE =
    "https://github.com/getflyhigh/flyhigh-engine/tree/v0.3.7/models";
inline constexpr std::string_view LICENSE_FILE = "WEIGHTS_LICENSE.md";

struct File {
    std::string_view name;
    u64 size;
    std::string_view sha256; ///< Lower-case hex.
};

inline constexpr std::array FILES{
    File{"graph.json", 14642, "5d367b59abf2a086c8529b3c8b7ba3990e11189a4b2150b362c4597c75d87d12"},
    File{"flyhighfg.json", 7410,
         "ab8be2b2e275656bf26be3448bfba6759be8ec365c7294dbdc8e0b9f7e80bf44"},
    File{"flyhighfg.bin", 1551622,
         "660fce3db712fa17484a4981a45c08f549e93610cbaea05a266a4c36985cb487"},
};
} // namespace FlyHighModel

/// Where the model files (graph.json, flyhighfg.json, flyhighfg.bin) are kept.
[[nodiscard]] std::filesystem::path FlyHighModelDir();

/// Where the engine keeps its kernel timings, measured once per GPU, driver and frame size.
[[nodiscard]] std::filesystem::path FlyHighTuneDir();

/// Whether every model file is in FlyHighModelDir() with its pinned size (not hashed).
[[nodiscard]] bool FlyHighModelInstalled();

/// Whether the library beside the executable loads and has everything this Lichii calls.
[[nodiscard]] bool FlyHighLibraryPresent();

} // namespace VideoCore::FrameGen
