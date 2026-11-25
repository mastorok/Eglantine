// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <filesystem>
#include <fstream>

#include <catch2/catch_test_macros.hpp>

#include "tests/common/temp_dir.h"
#include "video_core/renderer_vulkan/vk_frame_dump.h"

namespace {
using Vulkan::FrameDumpSchedule;
using Step = FrameDumpSchedule::Step;
} // Anonymous namespace

TEST_CASE("Frame dump first pass accepts only a decimal pass number", "[frame_dump_schedule]") {
    CHECK(Vulkan::ParseFrameDumpFirstPass("0") == 0U);
    CHECK(Vulkan::ParseFrameDumpFirstPass("220") == 220U);
    CHECK_FALSE(Vulkan::ParseFrameDumpFirstPass(""));
    CHECK_FALSE(Vulkan::ParseFrameDumpFirstPass("-1"));
    CHECK_FALSE(Vulkan::ParseFrameDumpFirstPass("220f"));
    CHECK_FALSE(Vulkan::ParseFrameDumpFirstPass("4294967296"));
}

TEST_CASE("Frame dump schedule is idle until configured or armed", "[frame_dump_schedule]") {
    FrameDumpSchedule schedule;
    REQUIRE(schedule.Tick(100.0) == Step::None);
    REQUIRE_FALSE(schedule.Capturing());
    REQUIRE_FALSE(Vulkan::RecordingSampledTextures());
}

TEST_CASE("Frame dump schedule records sampled textures for an armed sampled capture",
          "[frame_dump_schedule]") {
    FrameDumpSchedule schedule;
    REQUIRE(schedule.Arm(true, false, 7));
    REQUIRE_FALSE(Vulkan::RecordingSampledTextures());
    // The pipelines read the flag while the captured frame is drawn, between Start and Finish.
    REQUIRE(schedule.Tick(0.0) == Step::Start);
    REQUIRE(schedule.Capturing());
    REQUIRE(schedule.Sampled());
    REQUIRE_FALSE(schedule.Buffers());
    REQUIRE(schedule.CaptureId() == 7);
    REQUIRE(Vulkan::RecordingSampledTextures());
    REQUIRE_FALSE(schedule.Arm(false, false, 8));
    REQUIRE(schedule.Tick(0.0) == Step::Finish);
    REQUIRE_FALSE(schedule.Capturing());
    REQUIRE_FALSE(Vulkan::RecordingSampledTextures());
    REQUIRE(schedule.Tick(0.0) == Step::None);

    // The next request decides again: this one does not read sampled textures back.
    REQUIRE(schedule.Arm(false, false, 8));
    REQUIRE(schedule.Tick(0.0) == Step::Start);
    REQUIRE(schedule.CaptureId() == 8);
    REQUIRE_FALSE(Vulkan::RecordingSampledTextures());
    REQUIRE(schedule.Tick(0.0) == Step::Finish);

    REQUIRE(schedule.Arm(true, false, 9));
    REQUIRE(schedule.Tick(0.0) == Step::Start);
    REQUIRE(Vulkan::RecordingSampledTextures());
    schedule.Abort();
    REQUIRE_FALSE(schedule.Capturing());
    REQUIRE_FALSE(Vulkan::RecordingSampledTextures());
    REQUIRE(schedule.Tick(0.0) == Step::None);
}

TEST_CASE("Frame dump schedule records buffers independently of sampled textures",
          "[frame_dump_schedule]") {
    FrameDumpSchedule schedule;
    REQUIRE(schedule.Arm(false, true, 3));
    REQUIRE(schedule.Tick(0.0) == Step::Start);
    REQUIRE(schedule.Buffers());
    REQUIRE_FALSE(schedule.Sampled());
    // Buffers come from the buffer cache's bindings, not from the pipelines' sampled list.
    REQUIRE_FALSE(Vulkan::RecordingSampledTextures());
    REQUIRE(schedule.Tick(0.0) == Step::Finish);
    REQUIRE(schedule.Arm(true, true, 4));
    REQUIRE(schedule.Tick(0.0) == Step::Start);
    REQUIRE(schedule.Buffers());
    REQUIRE(Vulkan::RecordingSampledTextures());
    REQUIRE(schedule.Tick(0.0) == Step::Finish);
}

TEST_CASE("Frame dump schedule follows EDEN_NX2_FRAME_DUMP_AT", "[frame_dump_schedule]") {
    FrameDumpSchedule by_frame;
    by_frame.Configure("3f", {}, true);
    REQUIRE(by_frame.Tick(1000.0) == Step::None);
    REQUIRE(by_frame.Tick(1000.0) == Step::None);
    REQUIRE(by_frame.Tick(1000.0) == Step::Start);
    REQUIRE(by_frame.Frame() == 3);
    REQUIRE(by_frame.Sampled());
    REQUIRE(by_frame.Buffers());
    REQUIRE(Vulkan::RecordingSampledTextures());
    REQUIRE(by_frame.Tick(1000.0) == Step::Finish);
    REQUIRE(by_frame.Tick(1000.0) == Step::None);

    FrameDumpSchedule by_seconds;
    by_seconds.Configure("0.5", {}, false);
    REQUIRE(by_seconds.Tick(0.4) == Step::None);
    REQUIRE(by_seconds.Tick(0.6) == Step::Start);
    REQUIRE_FALSE(Vulkan::RecordingSampledTextures());
    REQUIRE(by_seconds.Tick(0.7) == Step::Finish);

    const Tests::TempDir directory{"lichii-frame-dump-schedule"};
    const auto trigger = directory / "capture.now";
    FrameDumpSchedule by_file;
    by_file.Configure("file", trigger, false);
    for (int frame = 1; frame <= 60; ++frame) {
        REQUIRE(by_file.Tick(1000.0) == Step::None);
    }
    std::ofstream{trigger}.put('\n');
    for (int frame = 61; frame < 90; ++frame) {
        REQUIRE(by_file.Tick(1000.0) == Step::None);
    }
    REQUIRE(by_file.Tick(1000.0) == Step::Start);
    REQUIRE(by_file.Frame() == 90);
}

TEST_CASE("Frame dump schedule captures consecutive frames and the buffers alone",
          "[frame_dump_schedule]") {
    FrameDumpSchedule schedule;
    schedule.Configure("2f", {}, false, true, 3);
    REQUIRE(schedule.Tick(1.0) == Step::None);
    REQUIRE(schedule.Tick(1.0) == Step::Start);
    REQUIRE(schedule.Frame() == 2);
    REQUIRE(schedule.Buffers());
    REQUIRE_FALSE(schedule.Sampled());
    REQUIRE_FALSE(Vulkan::RecordingSampledTextures());
    // Each frame ends the tick the next one starts: no frame is left out between them.
    REQUIRE(schedule.Tick(1.0) == Step::Finish);
    REQUIRE(schedule.MoreFramesFollow());
    REQUIRE(schedule.Capturing());
    REQUIRE(schedule.Frame() == 3);
    REQUIRE(schedule.Tick(1.0) == Step::Finish);
    REQUIRE(schedule.MoreFramesFollow());
    REQUIRE(schedule.Capturing());
    REQUIRE(schedule.Tick(1.0) == Step::Finish);
    REQUIRE_FALSE(schedule.MoreFramesFollow());
    REQUIRE_FALSE(schedule.Capturing());
    REQUIRE(schedule.Frame() == 5);
    REQUIRE(schedule.Tick(1.0) == Step::None);
    REQUIRE_FALSE(schedule.MoreFramesFollow());

    // Sampled textures stay recorded through the whole run of frames.
    FrameDumpSchedule sampled;
    sampled.Configure("1f", {}, true, false, 2);
    REQUIRE(sampled.Tick(1.0) == Step::Start);
    REQUIRE(Vulkan::RecordingSampledTextures());
    REQUIRE(sampled.Tick(1.0) == Step::Finish);
    REQUIRE(sampled.MoreFramesFollow());
    REQUIRE(Vulkan::RecordingSampledTextures());
    REQUIRE(sampled.Tick(1.0) == Step::Finish);
    REQUIRE_FALSE(sampled.MoreFramesFollow());
    REQUIRE_FALSE(Vulkan::RecordingSampledTextures());

    // A debug server request is one frame whatever EDEN_NX2_FRAME_DUMP_FRAMES says.
    FrameDumpSchedule configured;
    configured.Configure("never", {}, false, true, 4);
    REQUIRE(configured.Arm(false, true, 11));
    REQUIRE(configured.Tick(0.0) == Step::Start);
    REQUIRE(configured.Tick(0.0) == Step::Finish);
    REQUIRE_FALSE(configured.MoreFramesFollow());
    REQUIRE(configured.Tick(0.0) == Step::None);
}
