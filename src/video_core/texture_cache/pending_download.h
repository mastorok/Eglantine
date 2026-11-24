// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <memory>
#include <unordered_set>
#include <cstddef>
#include <utility>
#include "common/common_types.h"
#include "common/slot_vector.h"

namespace VideoCommon {

template <class Image>
struct PendingImageDownload {
    bool is_swizzle;
    size_t async_buffer_id;
    Common::SlotId object_id;
    std::shared_ptr<Image> retired_image{};
    /// The image's modification tick when its contents were staged (asynchronous downloads).
    u64 tick{};
    /// The GPU submission tick at deletion (0 on OpenGL).
    u64 retired_tick{};

    Image& GetImage(Common::SlotVector<Image>& images) const {
        return retired_image ? *retired_image : images[object_id];
    }
};

// Whole-image copies are recorded together at commit time. Repeated requests
// for the same live image therefore observe the same contents. Keep the first
// occurrence so reverse CPU writeback order remains unchanged for aliases.
template <class Image, class Downloads>
size_t DeduplicatePendingImageDownloads(Downloads& downloads, Common::SlotVector<Image>& images) {
    std::unordered_set<const Image*> seen;
    const size_t previous_size = downloads.size();
    const auto end = std::remove_if(downloads.begin(), downloads.end(), [&](const auto& download) {
        return download.is_swizzle && !seen.insert(&download.GetImage(images)).second;
    });
    downloads.erase(end, downloads.end());
    return previous_size - downloads.size();
}

// Keep the original image past slot reuse. The caller holds the returned reference until
// the delete tick completes, since the download's fence may precede other uses of the image.
template <class Image, class Downloads, class Batches>
std::shared_ptr<Image> RetainPendingImageDownloads(Common::SlotId id, Image& image,
                                                  Downloads& pending, Batches& submitted,
                                                  u64 tick) {
    std::shared_ptr<Image> retained;
    const auto retain = [&](auto& downloads) {
        for (auto& download : downloads) {
            if (!download.is_swizzle || download.object_id != id || download.retired_image) continue;
            if (!retained) retained = std::make_shared<Image>(std::move(image));
            download.retired_image = retained;
            download.retired_tick = tick;
        }
    };
    retain(pending);
    for (auto& downloads : submitted) retain(downloads);
    return retained;
}

} // namespace VideoCommon
