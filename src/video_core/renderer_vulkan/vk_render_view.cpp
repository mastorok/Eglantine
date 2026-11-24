// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>

#include "video_core/nx2_render_view.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_render_view.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"

namespace Vulkan {

namespace {

using Tegra::Engines::Maxwell3D;
using Maxwell = Maxwell3D::Regs;
using VideoCommon::Nx2RenderView::Check;
using VideoCommon::Nx2RenderView::Note;

template <typename T>
bool Same(const T& a, const T& b) {
    static_assert(std::is_trivially_copyable_v<T>);
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

/// The key as a full refresh gives it: every part Refresh only rebuilds on a dirty flag.
FixedPipelineState FullKey(Maxwell3D& engine, DynamicFeatures features) {
    auto& flags = engine.dirty.flags;
    const auto saved = flags;
    flags[Dirty::VertexInput] = true;
    flags[Dirty::ViewportSwizzles] = true;
    flags[Dirty::Blending] = true;
    FixedPipelineState key{};
    key.Refresh(engine, features);
    flags = saved;
    return key;
}

/// The dynamic state UpdateDynamicStates records, as the key packs it with no dynamic state.
FixedPipelineState::DynamicState DynamicKey(const Maxwell3D& engine,
                                            const DynamicFeatures& features) {
    FixedPipelineState::DynamicState state{};
    state.raw1 = 0;
    state.raw2 = 0;
    state.Refresh(engine.regs);
    state.Refresh2(engine.regs, engine.draw_manager.draw_state.topology,
                   features.has_extended_dynamic_state_2);
    state.Refresh3(engine.regs, features);
    return state;
}

bool SameBindings(const Maxwell3D& view, const Maxwell3D& engine) {
    for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
        for (size_t slot = 0; slot < Maxwell::MaxConstBuffers; ++slot) {
            const auto& a = view.state.shader_stages[stage].const_buffers[slot];
            const auto& b = engine.state.shader_stages[stage].const_buffers[slot];
            if (a.address != b.address || a.size != b.size || a.enabled != b.enabled) {
                return false;
            }
        }
    }
    return true;
}

/// The registers UpdateDynamicStates reads beyond what DynamicKey packs.
bool SameDynamicRegisters(const Maxwell& v, const Maxwell& e) {
    return Same(v.viewport_transform, e.viewport_transform) && Same(v.viewports, e.viewports) &&
           Same(v.scissor_test, e.scissor_test) && Same(v.surface_clip, e.surface_clip) &&
           Same(v.window_origin, e.window_origin) &&
           Same(v.viewport_clip_control, e.viewport_clip_control) &&
           Same(v.viewport_scale_offset_enabled, e.viewport_scale_offset_enabled) &&
           Same(v.depth_bias, e.depth_bias) &&
           Same(v.slope_scale_depth_bias, e.slope_scale_depth_bias) &&
           Same(v.depth_bias_clamp, e.depth_bias_clamp) && Same(v.blend_color, e.blend_color) &&
           Same(v.depth_bounds, e.depth_bounds) &&
           Same(v.stencil_two_side_enable, e.stencil_two_side_enable) &&
           Same(v.stencil_front_ref, e.stencil_front_ref) &&
           Same(v.stencil_front_mask, e.stencil_front_mask) &&
           Same(v.stencil_front_func_mask, e.stencil_front_func_mask) &&
           Same(v.stencil_back_ref, e.stencil_back_ref) &&
           Same(v.stencil_back_mask, e.stencil_back_mask) &&
           Same(v.stencil_back_func_mask, e.stencil_back_func_mask) &&
           Same(v.line_width_smooth, e.line_width_smooth) &&
           Same(v.line_width_aliased, e.line_width_aliased) && Same(v.blend, e.blend) &&
           Same(v.blend_per_target_enabled, e.blend_per_target_enabled) &&
           Same(v.blend_per_target, e.blend_per_target) && Same(v.color_mask, e.color_mask) &&
           Same(v.color_mask_common, e.color_mask_common) && Same(v.logic_op, e.logic_op) &&
           Same(v.vertex_attrib_format, e.vertex_attrib_format) &&
           Same(v.vertex_streams, e.vertex_streams) &&
           Same(v.vertex_stream_instances, e.vertex_stream_instances);
}

} // Anonymous namespace

void VerifyRenderViewBuild(Maxwell3D& view, Maxwell3D& engine, const DynamicFeatures& features) {
    const Maxwell& v = view.regs;
    const Maxwell& e = engine.regs;
    if (!(FullKey(view, features) == FullKey(engine, features))) {
        Note(Check::BuildPipeline, "key");
    } else if (!Same(v.pipelines, e.pipelines) || !Same(v.program_region, e.program_region)) {
        Note(Check::BuildPipeline, "programs");
    }
    if (!Same(v.rt, e.rt) || !Same(v.rt_control, e.rt_control) || !Same(v.zeta, e.zeta) ||
        v.zeta_enable != e.zeta_enable || !Same(v.zeta_size, e.zeta_size) ||
        !Same(v.surface_clip, e.surface_clip) ||
        v.anti_alias_samples_mode != e.anti_alias_samples_mode) {
        Note(Check::BuildRenderTargets, "registers");
    }
    const bool bindings = SameBindings(view, engine);
    if (!Same(v.tex_header, e.tex_header) ||
        v.bindless_texture_const_buffer_slot != e.bindless_texture_const_buffer_slot) {
        Note(Check::BuildImageViews, "TIC pool or bindless slot");
    } else if (!bindings) {
        Note(Check::BuildImageViews, "constant buffer bindings");
    }
    if (!Same(v.tex_sampler, e.tex_sampler) || v.sampler_binding != e.sampler_binding) {
        Note(Check::BuildDescriptors, "TSC pool or sampler binding");
    } else if (!bindings) {
        Note(Check::BuildDescriptors, "constant buffer bindings");
    }
    const auto dynamic_view = DynamicKey(view, features);
    const auto dynamic_engine = DynamicKey(engine, features);
    if (dynamic_view.raw1 != dynamic_engine.raw1 || dynamic_view.raw2 != dynamic_engine.raw2) {
        Note(Check::BuildDynamic, "dynamic state");
    } else if (!SameDynamicRegisters(v, e)) {
        Note(Check::BuildDynamic, "registers");
    }
}

} // namespace Vulkan
