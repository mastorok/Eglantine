// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <thread>

#include <ranges>
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

constexpr u64 FENCE_RESERVE_SIZE = 8;

MasterSemaphore::MasterSemaphore(const Device& device_) : device(device_) {
    if (!device.HasTimelineSemaphore()) {
        static constexpr VkFenceCreateInfo fence_ci{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .pNext = nullptr, .flags = 0};
        free_queue.resize(FENCE_RESERVE_SIZE);
        std::ranges::generate(free_queue,
                              [&] { return device.GetLogical().CreateFence(fence_ci); });
        wait_thread = std::jthread([this](std::stop_token token) { WaitThread(token); });
        return;
    }

    static constexpr VkSemaphoreTypeCreateInfo semaphore_type_ci{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = nullptr,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0,
    };
    static constexpr VkSemaphoreCreateInfo semaphore_ci{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &semaphore_type_ci,
        .flags = 0,
    };
    semaphore = device.GetLogical().CreateSemaphore(semaphore_ci);

    if (!Settings::values.renderer_debug) {
        return;
    }
    // Validation layers have a bug where they fail to track resource usage when using timeline
    // semaphores and synchronizing with GetSemaphoreCounterValue. To workaround this issue, have
    // a separate thread waiting for each timeline semaphore value.
    debug_thread = std::jthread([this](std::stop_token stop_token) {
        u64 counter = 0;
        while (!stop_token.stop_requested()) {
            if (semaphore.Wait(counter, 10'000'000)) {
                ++counter;
            }
        }
    });
}

MasterSemaphore::~MasterSemaphore() = default;

void MasterSemaphore::Refresh() {
    if (!semaphore) {
        // If we don't support timeline semaphores, there's nothing to refresh
        return;
    }

    u64 this_tick{};
    u64 counter{};
    do {
        this_tick = gpu_tick.load(std::memory_order_acquire);
        counter = semaphore.GetCounter();
        if (counter < this_tick) {
            return;
        }
    } while (!gpu_tick.compare_exchange_weak(this_tick, counter, std::memory_order_release,
                                             std::memory_order_relaxed));
}

void MasterSemaphore::Wait(u64 tick) {
    if (!semaphore) {
        // Fast check: already reached the requested tick?
        if (gpu_tick.load(std::memory_order_acquire) >= tick) {
            return;
        }

        u64 last_tick = gpu_tick.load(std::memory_order_relaxed);
        while (gpu_tick.load(std::memory_order_acquire) < tick) {
            gpu_tick.wait(last_tick, std::memory_order_acquire);
            last_tick = gpu_tick.load(std::memory_order_relaxed);
        }
        return;
    }

    // No need to wait if the GPU is ahead of the tick
    if (IsFree(tick)) {
        return;
    }

    // Update the GPU tick and try again
    Refresh();

    if (IsFree(tick)) {
        return;
    }

    // If none of the above is hit, fallback to a regular wait. It runs in slices so that a long one
    // (the multi-second stalls with the GPU idle) can say what the queue looked like.
    try {
        const auto begin = std::chrono::steady_clock::now();
        const auto since_epoch = [](std::chrono::steady_clock::time_point point) {
            return static_cast<s64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        point.time_since_epoch())
                                        .count());
        };
        static std::atomic<u32> logged{};
        auto next_report = begin + std::chrono::milliseconds{500};
        while (!semaphore.Wait(tick, 100'000'000)) {
            const auto now = std::chrono::steady_clock::now();
            if (now < next_report || logged.load(std::memory_order_relaxed) >= 300) {
                continue;
            }
            next_report = now + std::chrono::milliseconds{1000};
            logged.fetch_add(1, std::memory_order_relaxed);
            const s64 now_ns = since_epoch(now);
            const s64 begin_ns = submit_begin_ns.load(std::memory_order_relaxed);
            const s64 end_ns = submit_end_ns.load(std::memory_order_relaxed);
            LOG_WARNING(Render_Vulkan,
                        "Waiting {} ms for GPU tick {}: semaphore counter {}, known {}, current {}; "
                        "last queue submission was tick {}, {} ms ago, {} (queue submissions: {}, "
                        "longest {} us)",
                        std::chrono::duration_cast<std::chrono::milliseconds>(now - begin).count(),
                        tick, semaphore.GetCounter(), KnownGpuTick(), CurrentTick(),
                        submit_tick.load(std::memory_order_relaxed), (now_ns - begin_ns) / 1'000'000,
                        begin_ns > end_ns ? "still inside the driver"
                                          : "returned after its call",
                        submit_count.load(std::memory_order_relaxed),
                        submit_max_us.load(std::memory_order_relaxed));
        }
        const auto total = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - begin)
                               .count();
        if (total >= 500 && logged.load(std::memory_order_relaxed) < 300) {
            logged.fetch_add(1, std::memory_order_relaxed);
            LOG_WARNING(Render_Vulkan, "GPU tick {} reached after {} ms (counter {}, current {})", tick,
                        total, semaphore.GetCounter(), CurrentTick());
        }
    } catch (const vk::Exception& exception) {
        // The throw ends the process; say why first (a device loss reached nothing else here).
        LOG_CRITICAL(Render_Vulkan, "Waiting for GPU tick {} failed: {}", tick, exception.what());
        throw;
    }

    Refresh();
}

VkResult MasterSemaphore::SubmitQueue(vk::CommandBuffer& cmdbuf, vk::CommandBuffer& upload_cmdbuf,
                                      VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                                      u64 host_tick) {
    const auto begin = std::chrono::steady_clock::now();
    const auto since_epoch = [](std::chrono::steady_clock::time_point point) {
        return static_cast<s64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(point.time_since_epoch()).count());
    };
    submit_tick.store(host_tick, std::memory_order_relaxed);
    submit_begin_ns.store(since_epoch(begin), std::memory_order_release);
    const VkResult result =
        semaphore ? SubmitQueueTimeline(cmdbuf, upload_cmdbuf, signal_semaphore, wait_semaphore,
                                        host_tick)
                  : SubmitQueueFence(cmdbuf, upload_cmdbuf, signal_semaphore, wait_semaphore,
                                     host_tick);
    const auto end = std::chrono::steady_clock::now();
    submit_end_ns.store(since_epoch(end), std::memory_order_release);
    submit_count.fetch_add(1, std::memory_order_relaxed);
    const u64 micros =
        static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    if (micros > submit_max_us.load(std::memory_order_relaxed)) {
        submit_max_us.store(micros, std::memory_order_relaxed);
    }
    if (micros >= 50'000) {
        static std::atomic<u32> slow_logged{};
        if (slow_logged.fetch_add(1, std::memory_order_relaxed) < 100) {
            LOG_WARNING(Render_Vulkan, "vkQueueSubmit for tick {} took {} ms", host_tick,
                        micros / 1000);
        }
    }
    return result;
}

static constexpr VkPipelineStageFlags wait_stage_mask = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

VkResult MasterSemaphore::SubmitQueueTimeline(vk::CommandBuffer& cmdbuf,
                                              vk::CommandBuffer& upload_cmdbuf,
                                              VkSemaphore signal_semaphore,
                                              VkSemaphore wait_semaphore, u64 host_tick) {
    const VkSemaphore timeline_semaphore = *semaphore;

    if (device.HasSynchronization2()) {
        const std::array<VkCommandBufferSubmitInfo, 2> cmdbuffer_infos{{
            {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                .pNext = nullptr,
                .commandBuffer = *upload_cmdbuf,
                .deviceMask = 0,
            },
            {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                .pNext = nullptr,
                .commandBuffer = *cmdbuf,
                .deviceMask = 0,
            },
        }};

        std::array<VkSemaphoreSubmitInfo, 2> signal_infos{{
            {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext = nullptr,
                .semaphore = timeline_semaphore,
                .value = host_tick,
                .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .deviceIndex = 0,
            },
            {},
        }};
        u32 num_signal_semaphores = 1;
        if (signal_semaphore) {
            signal_infos[1] = VkSemaphoreSubmitInfo{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext = nullptr,
                .semaphore = signal_semaphore,
                .value = 0,
                .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .deviceIndex = 0,
            };
            num_signal_semaphores = 2;
        }

        const u32 num_wait_semaphores = wait_semaphore ? 1 : 0;
        const VkSemaphoreSubmitInfo wait_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = wait_semaphore,
            .value = 0,
            .stageMask = static_cast<VkPipelineStageFlags2>(wait_stage_mask),
            .deviceIndex = 0,
        };

        const VkSubmitInfo2 submit_info2{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .pNext = nullptr,
            .flags = 0,
            .waitSemaphoreInfoCount = num_wait_semaphores,
            .pWaitSemaphoreInfos = num_wait_semaphores ? &wait_info : nullptr,
            .commandBufferInfoCount = static_cast<u32>(cmdbuffer_infos.size()),
            .pCommandBufferInfos = cmdbuffer_infos.data(),
            .signalSemaphoreInfoCount = num_signal_semaphores,
            .pSignalSemaphoreInfos = signal_infos.data(),
        };
        return device.GetGraphicsQueue().Submit2(submit_info2);
    }

    const u32 num_signal_semaphores = signal_semaphore ? 2 : 1;
    const std::array signal_values{host_tick, u64(0)};
    const std::array signal_semaphores{timeline_semaphore, signal_semaphore};

    const std::array cmdbuffers{*upload_cmdbuf, *cmdbuf};

    const u32 num_wait_semaphores = wait_semaphore ? 1 : 0;
    // Pointers must be null when the count is zero (best-practices)
    const VkSemaphore* p_wait_sems =
        (num_wait_semaphores > 0) ? &wait_semaphore : nullptr;
    const VkPipelineStageFlags* p_wait_masks =
        (num_wait_semaphores > 0) ? &wait_stage_mask : nullptr;
    const VkSemaphore* p_signal_sems =
        (num_signal_semaphores > 0) ? signal_semaphores.data() : nullptr;
    const u64 wait_zero = 0; // dummy for binary wait
    const VkTimelineSemaphoreSubmitInfo timeline_si{
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreValueCount = num_wait_semaphores,
        .pWaitSemaphoreValues    = num_wait_semaphores ? &wait_zero : nullptr,
        .signalSemaphoreValueCount = num_signal_semaphores,
        .pSignalSemaphoreValues = signal_values.data(),
    };
    const VkSubmitInfo submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = &timeline_si,
        .waitSemaphoreCount = num_wait_semaphores,
        .pWaitSemaphores = p_wait_sems,
        .pWaitDstStageMask = p_wait_masks,
        .commandBufferCount = static_cast<u32>(cmdbuffers.size()),
        .pCommandBuffers = cmdbuffers.data(),
        .signalSemaphoreCount = num_signal_semaphores,
        .pSignalSemaphores = p_signal_sems,
    };

    return device.GetGraphicsQueue().Submit(submit_info);
}

VkResult MasterSemaphore::SubmitQueueFence(vk::CommandBuffer& cmdbuf,
                                           vk::CommandBuffer& upload_cmdbuf,
                                           VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                                           u64 host_tick) {
    if (device.HasSynchronization2()) {
        const std::array<VkCommandBufferSubmitInfo, 2> cmdbuffer_infos{{
            {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                .pNext = nullptr,
                .commandBuffer = *upload_cmdbuf,
                .deviceMask = 0,
            },
            {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                .pNext = nullptr,
                .commandBuffer = *cmdbuf,
                .deviceMask = 0,
            },
        }};

        const u32 num_signal_semaphores = signal_semaphore ? 1 : 0;
        const VkSemaphoreSubmitInfo signal_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = signal_semaphore,
            .value = 0,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .deviceIndex = 0,
        };

        const u32 num_wait_semaphores = wait_semaphore ? 1 : 0;
        const VkSemaphoreSubmitInfo wait_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = wait_semaphore,
            .value = 0,
            .stageMask = static_cast<VkPipelineStageFlags2>(wait_stage_mask),
            .deviceIndex = 0,
        };

        const VkSubmitInfo2 submit_info2{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .pNext = nullptr,
            .flags = 0,
            .waitSemaphoreInfoCount = num_wait_semaphores,
            .pWaitSemaphoreInfos = num_wait_semaphores ? &wait_info : nullptr,
            .commandBufferInfoCount = static_cast<u32>(cmdbuffer_infos.size()),
            .pCommandBufferInfos = cmdbuffer_infos.data(),
            .signalSemaphoreInfoCount = num_signal_semaphores,
            .pSignalSemaphoreInfos = num_signal_semaphores ? &signal_info : nullptr,
        };

        auto fence = GetFreeFence();
        auto result = device.GetGraphicsQueue().Submit2(submit_info2, *fence);

        if (result == VK_SUCCESS) {
            std::scoped_lock lock{wait_mutex};
            wait_queue.emplace(host_tick, std::move(fence));
            wait_cv.notify_one();
        }

        return result;
    }

    const u32 num_signal_semaphores = signal_semaphore ? 1 : 0;
    const u32 num_wait_semaphores = wait_semaphore ? 1 : 0;

    const VkSemaphore* p_wait_sems =
            (num_wait_semaphores > 0) ? &wait_semaphore : nullptr;
    const VkPipelineStageFlags* p_wait_masks =
        (num_wait_semaphores > 0) ? &wait_stage_mask : nullptr;
    const VkSemaphore* p_signal_sems =
        (num_signal_semaphores > 0) ? &signal_semaphore : nullptr;
    const std::array cmdbuffers{*upload_cmdbuf, *cmdbuf};

    const VkSubmitInfo submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = num_wait_semaphores,
        .pWaitSemaphores = p_wait_sems,
        .pWaitDstStageMask = p_wait_masks,
        .commandBufferCount = static_cast<u32>(cmdbuffers.size()),
        .pCommandBuffers = cmdbuffers.data(),
        .signalSemaphoreCount = num_signal_semaphores,
        .pSignalSemaphores = p_signal_sems,
    };

    auto fence = GetFreeFence();
    auto result = device.GetGraphicsQueue().Submit(submit_info, *fence);

    if (result == VK_SUCCESS) {
        std::scoped_lock lock{wait_mutex};
        wait_queue.emplace(host_tick, std::move(fence));
        wait_cv.notify_one();
    }

    return result;
}

void MasterSemaphore::WaitThread(std::stop_token token) {
    while (!token.stop_requested()) {
        u64 host_tick;
        vk::Fence fence;
        {
            std::unique_lock lock{wait_mutex};
            wait_cv.wait(lock, token, [this] { return !wait_queue.empty(); });
            if (token.stop_requested()) {
                return;
            }
            std::tie(host_tick, fence) = std::move(wait_queue.front());
            wait_queue.pop();
        }

        fence.Wait();
        fence.Reset();

        {
            std::scoped_lock lock{free_mutex};
            free_queue.push_front(std::move(fence));
            gpu_tick.store(host_tick, std::memory_order_release);
        }
        gpu_tick.notify_one();
    }
}

vk::Fence MasterSemaphore::GetFreeFence() {
    std::scoped_lock lock{free_mutex};
    if (free_queue.empty()) {
        static constexpr VkFenceCreateInfo fence_ci{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .pNext = nullptr, .flags = 0};
        return device.GetLogical().CreateFence(fence_ci);
    }

    auto fence = std::move(free_queue.back());
    free_queue.pop_back();
    return fence;
}

} // namespace Vulkan
