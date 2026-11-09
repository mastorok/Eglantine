// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;

/// Resources shared by the draws of task and mesh programs emulated with compute shaders on
/// devices without VK_EXT_mesh_shader (Profile::emulate_mesh_with_compute, MeshEmulation in
/// shader_info.h): one scratch buffer every such draw works in (a barrier orders one draw's reads
/// before the next one's writes), its descriptor set, and the pass that turns the task workgroups'
/// mesh group counts into the arguments of the mesh dispatch and of the draw.
class MeshEmulationResources {
public:
    /// Bytes of the scratch buffer, from which a draw's regions are laid out.
    static constexpr VkDeviceSize ScratchBytes{VkDeviceSize{64} << 20};

    /// EDEN_NX2_MESH_EMU=1 turns the emulation on (off by default until it is validated).
    [[nodiscard]] static bool Requested();

    explicit MeshEmulationResources(const Device& device, const MemoryAllocator& allocator);
    ~MeshEmulationResources();

    [[nodiscard]] VkDescriptorSetLayout ScratchSetLayout() const noexcept {
        return *scratch_set_layout;
    }
    /// Stands in for set 0 of a pipeline whose stages use no descriptors.
    [[nodiscard]] VkDescriptorSetLayout EmptySetLayout() const noexcept {
        return *empty_set_layout;
    }
    [[nodiscard]] VkDescriptorSet ScratchSet() const noexcept {
        return scratch_sets[0];
    }
    [[nodiscard]] VkBuffer Scratch() const noexcept {
        return *scratch;
    }
    [[nodiscard]] VkPipelineLayout PrefixLayout() const noexcept {
        return *prefix_layout;
    }
    [[nodiscard]] VkPipeline PrefixPipeline() const noexcept {
        return *prefix_pipeline;
    }

private:
    vk::Buffer scratch;
    vk::DescriptorSetLayout empty_set_layout;
    vk::DescriptorSetLayout scratch_set_layout;
    vk::DescriptorPool descriptor_pool;
    vk::DescriptorSets scratch_sets;
    vk::PipelineLayout prefix_layout;
    vk::ShaderModule prefix_module;
    vk::Pipeline prefix_pipeline;
};

} // namespace Vulkan
