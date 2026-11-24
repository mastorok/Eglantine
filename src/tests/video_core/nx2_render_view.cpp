// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <cstdio>
#include <functional>
#include <optional>

#include <catch2/catch_test_macros.hpp>

#include "core/core.h"
#include "core/device_memory.h"
#include "tests/core/system_test_context.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_render_view.h"
#include "video_core/rasterizer_interface.h"

namespace {

using Tegra::Engines::Maxwell3D;
namespace View = VideoCommon::Nx2RenderView;
using View::Check;

constexpr u32 ViewportScaleX = MAXWELL3D_REG_INDEX(viewport_transform);
constexpr u32 ScissorEnable = MAXWELL3D_REG_INDEX(scissor_test);
constexpr u8 ViewportFlag = VideoCommon::Dirty::LastCommonEntry;
constexpr u8 ScissorFlag = VideoCommon::Dirty::LastCommonEntry + 1;

class NoDma final : public Tegra::Engines::AccelerateDMAInterface {
public:
    bool BufferCopy(GPUVAddr, GPUVAddr, u64) override { return true; }
    bool BufferClear(GPUVAddr, u64, u32) override { return true; }
    bool ImageToBuffer(const Tegra::DMA::ImageCopy&, const Tegra::DMA::ImageOperand&,
                       const Tegra::DMA::BufferOperand&) override { return true; }
    bool BufferToImage(const Tegra::DMA::ImageCopy&, const Tegra::DMA::BufferOperand&,
                       const Tegra::DMA::ImageOperand&) override { return true; }
};

/// Each draw runs the check as PrepareDraw does, then `consume` as the draw's own work.
class ViewRasterizer final : public VideoCore::RasterizerInterface {
public:
    Maxwell3D* engine{};
    std::function<void(Maxwell3D&)> consume;
    u32 draws{};

    void Draw(bool, u32) override {
        const View::DrawCheck check{*engine};
        ++draws;
        if (consume) {
            consume(*engine);
        }
    }
    void DrawTexture() override {}
    void Clear(u32) override {}
    void DispatchCompute() override {}
    void ResetCounter(VideoCommon::QueryType) override {}
    void Query(GPUVAddr, VideoCommon::QueryType, VideoCommon::QueryPropertiesFlags, u32,
               u32) override {}
    void BindGraphicsUniformBuffer(size_t, u32, GPUVAddr, u32) override {}
    void DisableGraphicsUniformBuffer(size_t, u32) override {}
    void SignalFence(std::function<void()>&& func) override { func(); }
    void SyncOperation(std::function<void()>&& func) override { func(); }
    void SignalSyncPoint(u32) override {}
    void SignalReference() override {}
    void ReleaseFences(bool) override {}
    void FlushAll() override {}
    void FlushRegion(DAddr, u64, VideoCommon::CacheType) override {}
    bool MustFlushRegion(DAddr, u64, VideoCommon::CacheType) override { return false; }
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override {
        return {addr, addr + size, false};
    }
    void InvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void OnCacheInvalidation(PAddr, u64) override {}
    bool OnCPUWrite(PAddr, u64) override { return false; }
    void InvalidateGPUCache() override {}
    void UnmapMemory(DAddr, u64) override {}
    void ModifyGPUMemory(size_t, GPUVAddr, u64) override {}
    void FlushAndInvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void WaitForIdle() override {}
    void FragmentBarrier() override {}
    void TiledCacheBarrier() override {}
    void FlushCommands() override {}
    void TickFrame() override {}
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override { return dma; }
    void AccelerateInlineToMemory(GPUVAddr, size_t, std::span<const u8>) override {}

private:
    NoDma dma;
};

struct ViewFixture {
    ViewFixture() : memory{system, device_memory, 32, 1ULL << 31}, maxwell3d{memory} {
        View::SetEnabledForTest(true);
        View::ResetForTest();
        rasterizer.engine = &maxwell3d;
        maxwell3d.BindRasterizer(&rasterizer);
        maxwell3d.dirty.tables[0][ViewportScaleX] = ViewportFlag;
        maxwell3d.dirty.tables[0][ScissorEnable] = ScissorFlag;
        maxwell3d.dirty.flags.reset();
    }
    ~ViewFixture() {
        View::SetEnabledForTest(std::nullopt);
        View::ResetForTest();
    }

    void Write(u32 method, u32 value) {
        maxwell3d.CallMethod(system, method, value, true);
    }

    void DrawArrays(u32 first, u32 count) {
        Write(MAXWELL3D_REG_INDEX(vertex_buffer.first), first);
        Write(MAXWELL3D_REG_INDEX(vertex_buffer.count), count);
        Write(MAXWELL3D_REG_INDEX(draw.begin), 0x4);
        Write(MAXWELL3D_REG_INDEX(draw.end), 0);
    }

    void Bind(u32 stage, u32 slot, u32 address, u32 size) {
        Write(MAXWELL3D_REG_INDEX(const_buffer.size), size);
        Write(MAXWELL3D_REG_INDEX(const_buffer.address_high), 0);
        Write(MAXWELL3D_REG_INDEX(const_buffer.address_low), address);
        Write(MAXWELL3D_REG_INDEX(bind_groups[0].raw_config) + stage * 8, (slot << 4) | 1);
    }

    static u64 AllNoted() {
        u64 sum = 0;
        for (u32 i = 0; i < static_cast<u32>(Check::Count); ++i) {
            sum += View::Noted(static_cast<Check>(i));
        }
        return sum;
    }

    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory;
    ViewRasterizer rasterizer;
    Maxwell3D maxwell3d;
};

} // namespace

TEST_CASE("Render view follows register writes, binds and flags to every draw",
          "[nx2_render_view]") {
    ViewFixture f;
    f.rasterizer.consume = [](Maxwell3D& engine) { engine.dirty.flags[ViewportFlag] = false; };
    f.DrawArrays(0, 3);
    f.Write(ViewportScaleX, 0x3f800000);
    f.Bind(0, 3, 0x10000, 0x100);
    f.DrawArrays(3, 3);
    f.Write(ScissorEnable, 1);
    f.Write(ViewportScaleX, 0x40000000);
    f.DrawArrays(6, 3);
    // A flag cleared between draws, as a clear consumes the render targets' flags.
    f.Write(ScissorEnable, 0);
    f.maxwell3d.dirty.flags[ScissorFlag] = false;
    f.maxwell3d.engine_state = Maxwell3D::EngineHint::OnHLEMacro;
    f.DrawArrays(9, 3);
    CHECK(f.rasterizer.draws == 4);
    CHECK(View::Compared() == 4);
    CHECK(ViewFixture::AllNoted() == 0);
    auto& view = f.maxwell3d.render_view_writes->View();
    CHECK(view.regs.reg_array[ViewportScaleX] == 0x40000000);
    CHECK(view.state.shader_stages[0].const_buffers[3].address == 0x10000);
    CHECK(view.engine_state == Maxwell3D::EngineHint::OnHLEMacro);
}

TEST_CASE("Render view counts a register written past the notes", "[nx2_render_view]") {
    ViewFixture f;
    f.DrawArrays(0, 3);
    f.maxwell3d.regs.reg_array[ViewportScaleX] = 0x1234;
    f.DrawArrays(3, 3);
    CHECK(View::Noted(Check::ViewWords) == 1);
    f.Write(ViewportScaleX, 0x5678);
    // The HLE draw macros' direct stores are compared at each packet instead.
    f.maxwell3d.regs.vertex_id_base = 7;
    f.DrawArrays(6, 3);
    CHECK(View::Noted(Check::ViewWords) == 1);
    CHECK(f.maxwell3d.render_view_writes->View().regs.vertex_id_base == 7);
}

TEST_CASE("Render view counts a binding made past the notes", "[nx2_render_view]") {
    ViewFixture f;
    f.DrawArrays(0, 3);
    f.maxwell3d.state.shader_stages[4].const_buffers[1].address = 0x2000;
    f.DrawArrays(3, 3);
    CHECK(View::Noted(Check::ViewValues) == 1);
}

TEST_CASE("Render view packets seed once, then carry what changed", "[nx2_render_view]") {
    ViewFixture f;
    View::Writes writes{f.maxwell3d};
    const View::Packet seed = writes.Pack(f.maxwell3d);
    CHECK(seed.seed.size() == View::NumRegs);
    writes.DrawDone(f.maxwell3d);
    writes.Note(ViewportScaleX);
    writes.Note(MAXWELL3D_REG_INDEX(const_buffer.offset));
    writes.Note(ViewportScaleX);
    f.maxwell3d.dirty.flags[ScissorFlag] = true;
    const View::Packet delta = writes.Pack(f.maxwell3d);
    CHECK(delta.seed.empty());
    REQUIRE(delta.words.size() == 1);
    CHECK(delta.words[0].index == ViewportScaleX);
    CHECK(delta.raised.count() == 1);
    CHECK(delta.raised.test(ScissorFlag));
    CHECK(!delta.has_values);
    f.maxwell3d.dirty.flags[ScissorFlag] = false;
    writes.DrawDone(f.maxwell3d);
    const View::Packet next = writes.Pack(f.maxwell3d);
    CHECK(next.words.empty());
    CHECK(next.consumed.count() == 1);
    CHECK(next.consumed.test(ScissorFlag));
    CHECK(View::Forwarded().test(VideoCommon::Dirty::VertexBuffers));
    CHECK(!View::Forwarded().test(VideoCommon::Dirty::IndexBuffer));
    CHECK(!View::Watched(MAXWELL3D_REG_INDEX(const_buffer.buffer)));
}

TEST_CASE("Render view keeps the pipeline key while only draw parameters change",
          "[nx2_render_view]") {
    ViewFixture f;
    View::Writes writes{f.maxwell3d};
    CHECK(writes.KeyWordsWritten(f.maxwell3d));
    (void)writes.Pack(f.maxwell3d, true);
    CHECK(writes.KeyWordsWritten(f.maxwell3d));
    writes.KeyWordsSeen();
    CHECK(!writes.KeyWordsWritten(f.maxwell3d));
    writes.Note(MAXWELL3D_REG_INDEX(index_buffer.count));
    writes.Note(MAXWELL3D_REG_INDEX(vertex_buffer.first));
    CHECK(!writes.KeyWordsWritten(f.maxwell3d));
    (void)writes.Pack(f.maxwell3d, true);
    CHECK(!writes.KeyWordsWritten(f.maxwell3d));
    writes.Note(ViewportScaleX);
    CHECK(writes.KeyWordsWritten(f.maxwell3d));
    (void)writes.Pack(f.maxwell3d, true);
    CHECK(writes.KeyWordsWritten(f.maxwell3d));
    writes.KeyWordsSeen();
    CHECK(!writes.KeyWordsWritten(f.maxwell3d));
    ++f.maxwell3d.regs.reg_array[MAXWELL3D_REG_INDEX(draw_auto_stride)];
    CHECK(writes.KeyWordsWritten(f.maxwell3d));
}

TEST_CASE("Render view packets carry each noted word once, at its final value",
          "[nx2_render_view]") {
    ViewFixture f;
    View::Writes writes{f.maxwell3d};
    (void)writes.Pack(f.maxwell3d);
    writes.DrawDone(f.maxwell3d);
    u32 last = View::NumRegs - 1;
    while (!View::Watched(last)) {
        --last;
    }
    auto& regs = f.maxwell3d.regs.reg_array;
    for (const u32 index : {64U, 63U, last, 64U}) {
        regs[index] = index + 1;
        writes.Note(index);
    }
    regs[64] = 0x1234;
    writes.Note(View::NumRegs);
    writes.Note(MAXWELL3D_REG_INDEX(shadow_scratch));
    const View::Packet packet = writes.Pack(f.maxwell3d);
    REQUIRE(packet.words.size() == 3);
    CHECK(packet.words[0].index == 63);
    CHECK(packet.words[0].value == 64);
    CHECK(packet.words[1].index == 64);
    CHECK(packet.words[1].value == 0x1234);
    CHECK(packet.words[2].index == last);
    CHECK(packet.words[2].value == last + 1);
    writes.DrawDone(f.maxwell3d);
    CHECK(writes.Pack(f.maxwell3d).words.empty());
}

TEST_CASE("Render view tracker cost per register write", "[.][nx2_render_view_bench]") {
    ViewFixture f;
    View::Writes writes{f.maxwell3d};
    (void)writes.Pack(f.maxwell3d);
    writes.DrawDone(f.maxwell3d);
    // One draw's writes: half constant buffer data (not watched), the rest a hot set of state
    // words written again and again, and a few spread over the register file.
    constexpr size_t WritesPerDraw = 64;
    std::array<u32, WritesPerDraw * 16> indices{};
    u32 seed = 12345;
    for (u32& index : indices) {
        seed = seed * 1664525 + 1013904223;
        const u32 pick = seed >> 8;
        if (pick % 10 < 5) {
            index = MAXWELL3D_REG_INDEX(const_buffer.buffer) + pick % 16;
        } else if (pick % 10 < 8) {
            index = ViewportScaleX + pick % 32;
        } else {
            index = 0x200 + pick % 0xb00;
        }
    }
    constexpr u32 Draws = 200000;
    using Clock = std::chrono::steady_clock;
    u64 checksum = 0;
    Clock::duration note_time{};
    Clock::duration empty_time{};
    // Notes alone: every index between two packs, so the clock reads cost little per write.
    constexpr u32 NoteRounds = Draws / 16;
    for (u32 round = 0; round < NoteRounds; ++round) {
        const auto start = Clock::now();
        for (const u32 index : indices) {
            writes.Note(index);
        }
        const auto middle = Clock::now();
        const auto end = Clock::now();
        note_time += middle - start;
        empty_time += end - middle;
        checksum += writes.Pack(f.maxwell3d).words.size();
        writes.DrawDone(f.maxwell3d);
    }
    const auto pack_start = Clock::now();
    for (u32 draw = 0; draw < Draws; ++draw) {
        const u32* const batch = &indices[(draw % 16) * WritesPerDraw];
        for (size_t i = 0; i < WritesPerDraw; ++i) {
            writes.Note(batch[i]);
        }
        checksum += writes.Pack(f.maxwell3d).words.size();
        writes.DrawDone(f.maxwell3d);
    }
    const auto pack_time = Clock::now() - pack_start;
    const auto ns = [](Clock::duration d) {
        return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
    };
    std::printf("tracker: %.2f ns per write (Note), %.1f ns per draw (64 notes and a pack), "
                "checksum %llu\n",
                (ns(note_time) - ns(empty_time)) / (NoteRounds * indices.size()),
                ns(pack_time) / Draws, static_cast<unsigned long long>(checksum));
    CHECK(checksum != 0);
}
