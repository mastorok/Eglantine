// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// FlyHigh frame generation (https://github.com/getflyhigh/flyhigh-engine) runs in its own library,
// flyhigh.dll / libflyhigh.so beside Lichii, loaded when the frame generation type is chosen. The
// library is Apache-2.0; its model is CC BY-NC 4.0 and is kept in the user directory.

#pragma once

#include <string>

#include <flyhigh.h>

#include "video_core/frame_gen/flyhigh_model.h"

namespace VideoCore::FrameGen {

#define FLYHIGH_FUNCTIONS(X)                                                                       \
    X(fh_last_error)                                                                               \
    X(fh_engine_options_default)                                                                   \
    X(fh_engine_create)                                                                            \
    X(fh_engine_destroy)                                                                           \
    X(fh_engine_info)                                                                              \
    X(fh_model_load)                                                                               \
    X(fh_model_destroy)                                                                            \
    X(fh_session_options_default)                                                                  \
    X(fh_stream_options_default)                                                                   \
    X(fh_stream_create)                                                                            \
    X(fh_stream_destroy)                                                                           \
    X(fh_stream_next_input)                                                                        \
    X(fh_stream_push)                                                                              \
    X(fh_stream_generate)                                                                          \
    X(fh_stream_warm_up)                                                                           \
    X(fh_stream_write_rgba)                                                                        \
    X(fh_stream_read_rgba)

struct FlyHighLibrary {
#define FLYHIGH_POINTER(name) decltype(&::name) name{};
    FLYHIGH_FUNCTIONS(FLYHIGH_POINTER)
#undef FLYHIGH_POINTER

    [[nodiscard]] std::string LastError() const {
        return fh_last_error ? std::string{fh_last_error()} : std::string{};
    }
};

/// The library beside the executable, loaded on the first call. Null when it is missing or is a
/// version without every function above; the log says which.
[[nodiscard]] const FlyHighLibrary* LoadFlyHighLibrary();

} // namespace VideoCore::FrameGen
