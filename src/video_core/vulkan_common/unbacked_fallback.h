// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>

#include "common/common_types.h"

namespace Vulkan {

/// Images over both limits are never backed (title 1's sparse texture arrays reach 4.5G texels,
/// ~23 GB): VMA would quietly place them in system memory once VRAM is full. Texels alone also
/// caught Metroid Prime 4's 1408x256x1008 BC1 and BC6H light volumes (173 and 346 MiB), which
/// then sampled as zero and drew the world black.
constexpr u64 MaxBackedImageTexels = u64{1} << 28;
constexpr u64 MaxBackedImageBytes = u64{1} << 30;

constexpr bool IsTooLargeToBack(u64 texels, u64 bytes) {
    return texels > MaxBackedImageTexels && bytes > MaxBackedImageBytes;
}

constexpr u64 Bc6hMipTailBytes(u32 width, u32 height, u32 levels, u32 layers, u32 first_level) {
    u64 bytes = 0;
    for (u32 level = first_level; level < levels; ++level) {
        const u32 blocks_x = ((std::max)(width >> level, 1U) + 3) / 4;
        const u32 blocks_y = ((std::max)(height >> level, 1U) + 3) / 4;
        bytes += u64{blocks_x} * blocks_y * 16 * layers;
    }
    return bytes;
}

constexpr u32 CompactBc6hFirstMip(u32 width, u32 height, u32 levels, u32 layers) {
    constexpr u64 limit = u64{128} << 20;
    u32 first_level = 1;
    while (first_level + 1 < levels &&
           Bc6hMipTailBytes(width, height, levels, layers, first_level) > limit) {
        ++first_level;
    }
    return first_level;
}

} // namespace Vulkan
