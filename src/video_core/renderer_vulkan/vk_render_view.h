// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "video_core/engines/maxwell_3d.h"

namespace Vulkan {

struct DynamicFeatures;

/// EDEN_NX2_RENDER_VIEW_VERIFY: what a draw builds from GPURender's view (nx2_render_view.h)
/// against what it builds from the engine: the pipeline key, render targets, image views,
/// descriptors and dynamic state. The view must be quiet.
void VerifyRenderViewBuild(Tegra::Engines::Maxwell3D& view, Tegra::Engines::Maxwell3D& engine,
                           const DynamicFeatures& features);

} // namespace Vulkan
