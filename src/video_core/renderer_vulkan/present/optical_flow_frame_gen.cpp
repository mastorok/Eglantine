// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <vector>

#include "video_core/renderer_vulkan/present/optical_flow_frame_gen.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/vulkan_common/vulkan_device.h"

#include "video_core/host_shaders/frame_gen_interpolation_comp_spv.h"
#include "video_core/host_shaders/frame_gen_flow_search_comp_spv.h"
#include "video_core/host_shaders/frame_gen_luma_downsample_comp_spv.h"

namespace Vulkan {

namespace {

// ── Precise pipeline barriers ──────────────────────────────────────────

VkImageMemoryBarrier MakeBarrier(VkImage image, VkAccessFlags src_access,
                                 VkAccessFlags dst_access, VkImageLayout old_layout,
                                 VkImageLayout new_layout) {
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
}

void DiscardForComputeWrite(vk::CommandBuffer cmdbuf, VkImage image) {
    const auto barrier = MakeBarrier(image, 0, VK_ACCESS_SHADER_WRITE_BIT,
                                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, barrier);
}

// ── Batched dual-copy ──────────────────────────────────────────────────
// Copy one source image into TWO destinations with only 3 barrier calls
// instead of 6 (avoiding redundant source transitions).
void CopyImageToTwo(vk::CommandBuffer cmdbuf, VkImage source,
                    VkImage dest_a, VkImage dest_b, VkExtent2D extent) {
    const VkImageCopy region{
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffset = {0, 0, 0},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffset = {0, 0, 0},
        .extent = {extent.width, extent.height, 1},
    };

    // 1. Transition source → TRANSFER_SRC, both dests → TRANSFER_DST (batched)
    const std::array before{
        MakeBarrier(source, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
        MakeBarrier(dest_a, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
        MakeBarrier(dest_b, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {}, {}, before);

    // 2. Copy source → dest_a, then source → dest_b (source stays TRANSFER_SRC)
    cmdbuf.CopyImage(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dest_a,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);
    cmdbuf.CopyImage(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dest_b,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);

    // 3. Transition all three back to GENERAL (batched)
    const std::array after{
        MakeBarrier(source, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        MakeBarrier(dest_a, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        MakeBarrier(dest_b, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           0, {}, {}, after);
}

void CopyImageToImage(vk::CommandBuffer cmdbuf, VkImage source, VkImage destination,
                      VkExtent2D extent) {
    const VkImageCopy region{
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffset = {0, 0, 0},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffset = {0, 0, 0},
        .extent = {extent.width, extent.height, 1},
    };

    const std::array before{
        MakeBarrier(source, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
        MakeBarrier(destination, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {}, {}, before);

    cmdbuf.CopyImage(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);

    const std::array after{
        MakeBarrier(source, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        MakeBarrier(destination, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           0, {}, {}, after);
}

} // anonymous namespace

struct DownsamplePushConstants {
    s32 destination_width;
    s32 destination_height;
    s32 is_color;
};

struct SearchPushConstants {
    s32 width;
    s32 height;
    s32 radius;
    s32 has_coarse;
    s32 subpixel;
};

struct InterpPushConstants {
    float inv_render_w;
    float inv_render_h;
    float render_w;
    float render_h;
    float blend_factor;
    float flow_scale;
};

namespace {

/// Full search radius at the coarsest level, in its pixels (32 frame pixels each).
constexpr s32 COARSE_SEARCH_RADIUS = 6;

vk::DescriptorSetLayout CreateComputeSetLayout(const Device& device, u32 samplers) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (u32 binding = 0; binding <= samplers; ++binding) {
        bindings.push_back(VkDescriptorSetLayoutBinding{
            .binding = binding,
            .descriptorType = binding < samplers ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                 : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        });
    }
    return device.GetLogical().CreateDescriptorSetLayout(VkDescriptorSetLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    });
}

vk::PipelineLayout CreateComputePipelineLayout(const Device& device,
                                               const vk::DescriptorSetLayout& set_layout,
                                               u32 push_size) {
    const VkPushConstantRange range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_size,
    };
    return device.GetLogical().CreatePipelineLayout(VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = set_layout.address(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    });
}

vk::Pipeline CreateComputePipeline(const Device& device, const vk::PipelineLayout& layout,
                                   const vk::ShaderModule& shader) {
    return device.GetLogical().CreateComputePipeline(VkComputePipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = *shader,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
        .layout = *layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    });
}

void ComputeToComputeBarrier(vk::CommandBuffer cmdbuf) {
    static constexpr VkMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, barrier);
}

u32 GroupCount(u32 size) {
    return (size + 15) / 16;
}

} // Anonymous namespace

OpticalFlowFrameGen::OpticalFlowFrameGen(const Device& device, MemoryAllocator& memory_allocator,
                                         VkExtent2D extent, VkFormat format)
    : m_allocator{memory_allocator}, m_extent{extent}, m_format{format} {
    CreateResources(device);
    CreateShaders(device);
    CreatePipelines(device);
    WriteFlowDescriptors(device);
    m_initialized = true;
}

OpticalFlowFrameGen::~OpticalFlowFrameGen() = default;

void OpticalFlowFrameGen::CreateResources(const Device& device) {
    m_linear_sampler = CreateBilinearSampler(device);

    m_history_image = CreateWrappedImage(m_allocator, m_extent, m_format);
    m_history_view = CreateWrappedImageView(device, m_history_image, m_format);

    // Current-frame copy for interpolation shader
    m_current_image = m_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = m_format,
        .extent = {m_extent.width, m_extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                 VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    m_current_view = CreateWrappedImageView(device, m_current_image, m_format);

    VkExtent2D level_extent = m_extent;
    for (FlowLevel& level : m_levels) {
        level_extent = {(std::max)(1u, (level_extent.width + 1) / 2),
                        (std::max)(1u, (level_extent.height + 1) / 2)};
        level.extent = level_extent;
        for (size_t parity = 0; parity < 2; ++parity) {
            level.luma[parity] =
                CreateWrappedImage(m_allocator, level_extent, VK_FORMAT_R16_SFLOAT);
            level.luma_view[parity] =
                CreateWrappedImageView(device, level.luma[parity], VK_FORMAT_R16_SFLOAT);
        }
        level.motion = CreateWrappedImage(m_allocator, level_extent, VK_FORMAT_R16G16_SFLOAT);
        level.motion_view = CreateWrappedImageView(device, level.motion, VK_FORMAT_R16G16_SFLOAT);
    }

    constexpr u32 SETS = 2 * 2 * FLOW_LEVELS;
    const std::array pool_sizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * 4 * FLOW_LEVELS},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, SETS},
    };
    m_desc_pool = device.GetLogical().CreateDescriptorPool(VkDescriptorPoolCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .maxSets = SETS,
        .poolSizeCount = static_cast<u32>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    });
}

void OpticalFlowFrameGen::CreateShaders(const Device& device) {
    m_downsample_shader = BuildShader(device, FRAME_GEN_LUMA_DOWNSAMPLE_COMP_SPV);
    m_search_shader = BuildShader(device, FRAME_GEN_FLOW_SEARCH_COMP_SPV);
    m_interp_shader = BuildShader(device, FRAME_GEN_INTERPOLATION_COMP_SPV);
}

void OpticalFlowFrameGen::CreatePipelines(const Device& device) {
    m_downsample_desc_layout = CreateComputeSetLayout(device, 1);
    m_downsample_pipe_layout = CreateComputePipelineLayout(device, m_downsample_desc_layout,
                                                           sizeof(DownsamplePushConstants));
    m_downsample_pipeline =
        CreateComputePipeline(device, m_downsample_pipe_layout, m_downsample_shader);

    m_search_desc_layout = CreateComputeSetLayout(device, 3);
    m_search_pipe_layout = CreateComputePipelineLayout(device, m_search_desc_layout,
                                                       sizeof(SearchPushConstants));
    m_search_pipeline = CreateComputePipeline(device, m_search_pipe_layout, m_search_shader);

    m_interp_desc_layout = CreateComputeSetLayout(device, 3);
    m_interp_pipe_layout = CreateComputePipelineLayout(device, m_interp_desc_layout,
                                                       sizeof(InterpPushConstants));
    m_interp_pipeline = CreateComputePipeline(device, m_interp_pipe_layout, m_interp_shader);

    std::vector<VkDescriptorSetLayout> downsample_layouts(2 * FLOW_LEVELS,
                                                          *m_downsample_desc_layout);
    std::vector<VkDescriptorSetLayout> search_layouts(2 * FLOW_LEVELS, *m_search_desc_layout);
    m_downsample_desc_sets = CreateWrappedDescriptorSets(m_desc_pool, downsample_layouts);
    m_search_desc_sets = CreateWrappedDescriptorSets(m_desc_pool, search_layouts);
}

void OpticalFlowFrameGen::WriteFlowDescriptors(const Device& device) {
    std::vector<VkDescriptorImageInfo> image_infos;
    image_infos.reserve(2 * 2 * 4 * FLOW_LEVELS);
    std::vector<VkWriteDescriptorSet> updates;
    const auto storage = [&](VkImageView view, VkDescriptorSet set, u32 binding) {
        image_infos.push_back(VkDescriptorImageInfo{
            .sampler = VK_NULL_HANDLE,
            .imageView = view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        });
        updates.push_back(VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = binding,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &image_infos.back(),
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        });
    };
    const auto sampled = [&](VkImageView view, VkDescriptorSet set, u32 binding) {
        updates.push_back(
            CreateWriteDescriptorSet(image_infos, *m_linear_sampler, view, set, binding));
    };
    for (u32 parity = 0; parity < 2; ++parity) {
        for (size_t level = 0; level < FLOW_LEVELS; ++level) {
            const FlowLevel& here = m_levels[level];
            const VkDescriptorSet downsample = m_downsample_desc_sets[parity * FLOW_LEVELS + level];
            sampled(level == 0 ? *m_current_view : *m_levels[level - 1].luma_view[parity],
                    downsample, 0);
            storage(*here.luma_view[parity], downsample, 1);

            const VkDescriptorSet search = m_search_desc_sets[parity * FLOW_LEVELS + level];
            sampled(*here.luma_view[parity], search, 0);
            sampled(*here.luma_view[parity ^ 1], search, 1);
            // The coarsest level has no coarser motion; it binds the finest, which it never reads.
            sampled(level + 1 < FLOW_LEVELS ? *m_levels[level + 1].motion_view
                                            : *m_levels[0].motion_view,
                    search, 2);
            storage(*here.motion_view, search, 3);
        }
    }
    device.GetLogical().UpdateDescriptorSets(updates, {});
}

void OpticalFlowFrameGen::RecordFlow(vk::CommandBuffer cmdbuf, u32 parity, bool search) const {
    ComputeToComputeBarrier(cmdbuf);
    cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *m_downsample_pipeline);
    for (size_t level = 0; level < FLOW_LEVELS; ++level) {
        const VkExtent2D extent = m_levels[level].extent;
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *m_downsample_pipe_layout, 0,
                                  m_downsample_desc_sets[parity * FLOW_LEVELS + level], {});
        cmdbuf.PushConstants(*m_downsample_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                             DownsamplePushConstants{
                                 .destination_width = static_cast<s32>(extent.width),
                                 .destination_height = static_cast<s32>(extent.height),
                                 .is_color = level == 0 ? 1 : 0,
                             });
        cmdbuf.Dispatch(GroupCount(extent.width), GroupCount(extent.height), 1);
        ComputeToComputeBarrier(cmdbuf);
    }
    if (!search) {
        return;
    }
    cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *m_search_pipeline);
    for (size_t level = FLOW_LEVELS; level-- > 0;) {
        const VkExtent2D extent = m_levels[level].extent;
        const bool coarsest = level + 1 == FLOW_LEVELS;
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *m_search_pipe_layout, 0,
                                  m_search_desc_sets[parity * FLOW_LEVELS + level], {});
        cmdbuf.PushConstants(*m_search_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                             SearchPushConstants{
                                 .width = static_cast<s32>(extent.width),
                                 .height = static_cast<s32>(extent.height),
                                 .radius = coarsest ? COARSE_SEARCH_RADIUS : 1,
                                 .has_coarse = coarsest ? 0 : 1,
                                 .subpixel = level == 0 ? 1 : 0,
                             });
        cmdbuf.Dispatch(GroupCount(extent.width), GroupCount(extent.height), 1);
        ComputeToComputeBarrier(cmdbuf);
    }
}

void OpticalFlowFrameGen::Process(const Device& device, Scheduler& scheduler, VkImage current_frame,
                                  VkImageView, u64 frame_count) {
    if (frame_count == 0) {
        m_has_history = false;
        m_flow_valid = false;
    }
    const u32 parity = m_parity;
    m_parity ^= 1;
    if (!m_has_history) {
        // First frame: populate history, the current-frame copy and this frame's luma pyramid.
        scheduler.Record([this, current_frame, parity](vk::CommandBuffer cmdbuf) {
            CopyImageToTwo(cmdbuf, current_frame, *m_history_image, *m_current_image, m_extent);
            std::vector<VkImageMemoryBarrier> barriers;
            for (const FlowLevel& level : m_levels) {
                barriers.push_back(MakeBarrier(*level.luma[0], 0, VK_ACCESS_SHADER_WRITE_BIT,
                                               VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL));
                barriers.push_back(MakeBarrier(*level.luma[1], 0, VK_ACCESS_SHADER_WRITE_BIT,
                                               VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL));
                barriers.push_back(MakeBarrier(*level.motion, 0, VK_ACCESS_SHADER_WRITE_BIT,
                                               VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL));
            }
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, barriers);
            RecordFlow(cmdbuf, parity, false);
        });
        m_has_history = true;
        m_last_frame_count = frame_count;
        return;
    }

    if (frame_count == m_last_frame_count) {
        m_parity ^= 1;
        return;
    }
    m_last_frame_count = frame_count;

    scheduler.Record([this, current_frame, parity](vk::CommandBuffer cmdbuf) {
        CopyImageToImage(cmdbuf, *m_current_image, *m_history_image, m_extent);
        CopyImageToImage(cmdbuf, current_frame, *m_current_image, m_extent);
        RecordFlow(cmdbuf, parity, true);
    });

    m_flow_valid = true;
}

void OpticalFlowFrameGen::Generate(const Device& device, Scheduler& scheduler, VkImage destination,
                                   VkImageView destination_storage_view, float blend_factor,
                                   float flow_scale) {
    if (!m_has_history || !m_flow_valid) {
        return;
    }

    // ── Descriptor caching for interpolation ───────────────────────────
    if (!m_interp_pool) {
        m_interp_pool.emplace(device, scheduler);
        m_interp_allocator = m_interp_pool->Allocator(device, scheduler, *m_interp_desc_layout,
                                                      DescriptorBankInfo{.textures = 3, .images = 1});
    }
    const VkDescriptorSet set = m_interp_allocator.Commit();
    std::vector<VkDescriptorImageInfo> image_infos;
    image_infos.reserve(4);
    std::vector<VkWriteDescriptorSet> updates{
        CreateWriteDescriptorSet(image_infos, *m_linear_sampler, *m_current_view, set, 0),
        CreateWriteDescriptorSet(image_infos, *m_linear_sampler, *m_history_view, set, 1),
        CreateWriteDescriptorSet(image_infos, *m_linear_sampler, *m_levels[0].motion_view, set, 2),
    };

    const VkDescriptorImageInfo dest_info{
        .imageView = destination_storage_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    updates.push_back(VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = 3,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &dest_info,
    });
    device.GetLogical().UpdateDescriptorSets(updates, {});

    // Interpolation dispatches at full resolution
    const u32 out_w = m_extent.width;
    const u32 out_h = m_extent.height;
    const InterpPushConstants constants{
        .inv_render_w = 1.0f / static_cast<float>(out_w),
        .inv_render_h = 1.0f / static_cast<float>(out_h),
        .render_w = static_cast<float>(out_w),
        .render_h = static_cast<float>(out_h),
        .blend_factor = blend_factor,
        .flow_scale = flow_scale,
    };

    scheduler.Record([this, constants, out_w, out_h, set, destination](vk::CommandBuffer cmdbuf) {
        DiscardForComputeWrite(cmdbuf, destination);
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *m_interp_pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *m_interp_pipe_layout, 0,
                                  set, {});
        cmdbuf.PushConstants(*m_interp_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
        cmdbuf.Dispatch((out_w + 15) / 16, (out_h + 15) / 16, 1);
    });
}

} // namespace Vulkan
