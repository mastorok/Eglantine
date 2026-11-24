// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// Frame generation without engine motion vectors: coarse-to-fine block matching on a luma
/// pyramid finds per-pixel motion between the last two frames (at half resolution, tracking camera
/// pans up to 192 pixels per frame), and bilateral interpolation warps both frames along it.
class OpticalFlowFrameGen {
public:
    explicit OpticalFlowFrameGen(const Device& device, MemoryAllocator& memory_allocator,
                                 VkExtent2D extent, VkFormat format);
    ~OpticalFlowFrameGen();

    void Process(const Device& device, Scheduler& scheduler, VkImage current_frame,
                 VkImageView current_view, u64 frame_count);

    void Generate(const Device& device, Scheduler& scheduler, VkImage destination,
                  VkImageView destination_storage_view, float blend_factor, float flow_scale);

    [[nodiscard]] bool IsReady() const {
        return m_initialized && m_has_history && m_flow_valid;
    }

private:
    void CreateResources(const Device& device);
    void CreateShaders(const Device& device);
    void CreatePipelines(const Device& device);
    void WriteFlowDescriptors(const Device& device);
    void RecordFlow(vk::CommandBuffer cmdbuf, u32 parity, bool search) const;

    static constexpr size_t FLOW_LEVELS = 5;

    /// One pyramid level: luma of the two most recent frames (indexed by parity) and the motion
    /// found at this level, in this level's pixels.
    struct FlowLevel {
        VkExtent2D extent{};
        std::array<vk::Image, 2> luma;
        std::array<vk::ImageView, 2> luma_view;
        vk::Image motion;
        vk::ImageView motion_view;
    };

    MemoryAllocator& m_allocator;
    VkExtent2D m_extent;
    VkFormat m_format;

    vk::Sampler m_linear_sampler;
    vk::ShaderModule m_downsample_shader;
    vk::ShaderModule m_search_shader;
    vk::ShaderModule m_interp_shader;

    vk::DescriptorSetLayout m_downsample_desc_layout;
    vk::PipelineLayout m_downsample_pipe_layout;
    vk::Pipeline m_downsample_pipeline;

    vk::DescriptorSetLayout m_search_desc_layout;
    vk::PipelineLayout m_search_pipe_layout;
    vk::Pipeline m_search_pipeline;

    vk::DescriptorSetLayout m_interp_desc_layout;
    vk::PipelineLayout m_interp_pipe_layout;
    vk::Pipeline m_interp_pipeline;

    vk::DescriptorPool m_desc_pool;
    /// Indexed by parity * FLOW_LEVELS + level.
    vk::DescriptorSets m_downsample_desc_sets;
    vk::DescriptorSets m_search_desc_sets;
    std::array<FlowLevel, FLOW_LEVELS> m_levels;
    u32 m_parity{};
    std::optional<DescriptorPool> m_interp_pool;
    DescriptorAllocator m_interp_allocator;

    // History and motion vector images
    vk::Image m_history_image;
    vk::ImageView m_history_view;

    // Store the last current-frame view for descriptor caching
    vk::Image m_current_image;
    vk::ImageView m_current_view;

    bool m_initialized{};
    bool m_has_history{};

    // Duplicate-frame detection: skip optical flow when the frame hasn't changed
    u64 m_last_frame_count{};
    bool m_flow_valid{};
};

} // namespace Vulkan
