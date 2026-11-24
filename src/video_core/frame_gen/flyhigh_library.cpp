// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <optional>

#ifdef _WIN32
#include <windows.h>
#endif

#include "common/dynamic_library.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "video_core/frame_gen/flyhigh_library.h"

namespace VideoCore::FrameGen {

namespace {

#ifdef _WIN32
constexpr const char* LIBRARY_NAME = "flyhigh.dll";
#else
constexpr const char* LIBRARY_NAME = "libflyhigh.so";
#endif

struct Loaded {
    Common::DynamicLibrary library;
    FlyHighLibrary functions;
};

std::optional<Loaded> Load() {
    Loaded loaded;
#ifdef _WIN32
    const std::filesystem::path path = Common::FS::GetExeDirectory() / LIBRARY_NAME;
    if (path.is_absolute()) {
        loaded.library = Common::DynamicLibrary{LoadLibraryExW(
            path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32)};
    }
#else
    std::error_code ec;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (ec) {
        return std::nullopt;
    }
    const auto path = executable.parent_path() / LIBRARY_NAME;
    if (path.is_absolute()) {
        (void)loaded.library.Open(Common::FS::PathToUTF8String(path).c_str());
    }
#endif
    if (!loaded.library.IsOpen()) {
        LOG_WARNING(Render_Vulkan, "FlyHigh: {} not found beside Lichii", LIBRARY_NAME);
        return std::nullopt;
    }
    bool complete = true;
#define FLYHIGH_SYMBOL(name)                                                                       \
    if (!loaded.library.GetSymbol(#name, &loaded.functions.name)) {                                \
        LOG_WARNING(Render_Vulkan, "FlyHigh: {} has no {}; it is older than this Lichii needs",    \
                    LIBRARY_NAME, #name);                                                          \
        complete = false;                                                                          \
    }
    FLYHIGH_FUNCTIONS(FLYHIGH_SYMBOL)
#undef FLYHIGH_SYMBOL
    if (!complete) {
        return std::nullopt;
    }
    return loaded;
}

} // Anonymous namespace

const FlyHighLibrary* LoadFlyHighLibrary() {
    static const auto* loaded = []() -> Loaded* {
        auto result = Load();
        return result ? new Loaded(std::move(*result)) : nullptr;
    }();
    return loaded != nullptr ? &loaded->functions : nullptr;
}

bool FlyHighLibraryPresent() {
    return LoadFlyHighLibrary() != nullptr;
}

std::filesystem::path FlyHighModelDir() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::FlyHighDir) / "model";
}

std::filesystem::path FlyHighTuneDir() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::FlyHighDir) / "tune";
}

bool FlyHighModelInstalled() {
    const std::filesystem::path dir = FlyHighModelDir();
    for (const FlyHighModel::File& file : FlyHighModel::FILES) {
        const std::filesystem::path path = dir / file.name;
        if (!Common::FS::IsFile(path) || Common::FS::GetSize(path) != file.size) {
            return false;
        }
    }
    return true;
}

} // namespace VideoCore::FrameGen
