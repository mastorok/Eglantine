// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <utility>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/dirty_flags.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_render_view.h"

namespace VideoCommon::Nx2RenderView {

namespace {

using Regs = Maxwell3D::Regs;
constexpr u32 Slots = Regs::MaxConstBuffers;

struct Partition {
    Flags forward;

    Partition() {
        forward.set();
        forward.reset(Dirty::NullEntry);
        forward.reset(Dirty::IndexBuffer);
        for (u32 i = 0; i < Regs::NumVertexArrays; ++i) {
            forward.reset(Dirty::VertexBuffer0 + i);
        }
    }
};

const Partition& Parts() {
    static const Partition parts;
    return parts;
}

// The HLE draw macros (macro.cpp) store these without ProcessDirtyRegisters: compared at each
// packet instead of noted.
constexpr std::array HleWords{
    static_cast<u32>(MAXWELL3D_REG_INDEX(vertex_id_base)),
    static_cast<u32>(MAXWELL3D_REG_INDEX(global_base_vertex_index)),
    static_cast<u32>(MAXWELL3D_REG_INDEX(global_base_instance_index)),
    static_cast<u32>(MAXWELL3D_REG_INDEX(clear_surface)),
    static_cast<u32>(MAXWELL3D_REG_INDEX(draw.begin)),
    static_cast<u32>(MAXWELL3D_REG_INDEX(draw_auto_stride)),
    static_cast<u32>(MAXWELL3D_REG_INDEX(draw_auto_byte_count)),
};

constexpr std::array<std::string_view, static_cast<size_t>(Check::Count)> CheckNames{
    "ViewWords",     "ViewFlags",       "ViewDraw",     "ViewPools",        "ViewValues",
    "BuildPipeline", "BuildImageViews", "BuildDynamic", "BuildDescriptors", "BuildRenderTargets",
};

std::array<std::atomic<u64>, static_cast<size_t>(Check::Count)> g_noted{};
std::atomic<u64> g_compared{};
std::array<u64, 256> g_cleared_between{};
u64 g_words{};
u64 g_binds{};
u64 g_raised{};
u64 g_consumed{};
std::optional<bool> g_test_enabled;

template <typename T>
bool SameBytes(const T& a, const T& b) {
    static_assert(std::has_unique_object_representations_v<T> || std::is_trivially_copyable_v<T>);
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

DrawValues DrawOf(const Maxwell3D& engine) {
    const auto& state = engine.draw_manager.draw_state;
    return {
        .topology = state.topology,
        .draw_mode = state.draw_mode,
        .draw_indexed = state.draw_indexed ? 1U : 0U,
        .draw_element_count = state.draw_element_count,
        .base_index = state.base_index,
        .vertex_buffer = state.vertex_buffer,
        .index_buffer = state.index_buffer,
        .base_instance = state.base_instance,
        .instance_count = state.instance_count,
        .drawn_instances = state.drawn_instances,
    };
}

bool SameDraw(const DrawValues& a, const DrawValues& b) {
    return a.topology == b.topology && a.draw_mode == b.draw_mode &&
           a.draw_indexed == b.draw_indexed && a.draw_element_count == b.draw_element_count &&
           a.base_index == b.base_index && SameBytes(a.vertex_buffer, b.vertex_buffer) &&
           SameBytes(a.index_buffer, b.index_buffer) && a.base_instance == b.base_instance &&
           a.instance_count == b.instance_count && a.drawn_instances == b.drawn_instances;
}

EngineValues ValuesOf(Maxwell3D& engine) {
    return {engine.engine_state, engine.GetNativeMacroEngine().native_turing};
}

BindValue BindOf(const Maxwell3D& engine, u32 stage, u32 slot) {
    const auto& info = engine.state.shader_stages[stage].const_buffers[slot];
    return {
        .stage = static_cast<u8>(stage),
        .slot = static_cast<u8>(slot),
        .enabled = static_cast<u8>(info.enabled ? 1 : 0),
        .size = info.size,
        .address = info.address,
    };
}

u32 FirstFlag(const Flags& flags) {
    u32 flag = 0;
    while (flag < flags.size() && !flags.test(flag)) {
        ++flag;
    }
    return flag;
}

} // Anonymous namespace

bool VerifyAll() {
    static const bool env = [] {
        const char* const value = std::getenv("EDEN_NX2_RENDER_VIEW_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    return g_test_enabled.value_or(env);
}

bool Enabled() {
    return VerifyAll() || GpuSplit::UnlendEnabled();
}

void SetEnabledForTest(std::optional<bool> on) {
    g_test_enabled = on;
}

void Note(Check check, std::string_view detail) {
    const u64 count = g_noted[static_cast<size_t>(check)].fetch_add(1, std::memory_order_relaxed);
    if (count < 8) {
        LOG_ERROR(HW_GPU, "Render view verify: {} differs: {}", CheckNames[static_cast<size_t>(check)],
                  detail);
    }
}

u64 Noted(Check check) {
    return g_noted[static_cast<size_t>(check)].load(std::memory_order_relaxed);
}

u64 Compared() {
    return g_compared.load(std::memory_order_relaxed);
}

void ResetForTest() {
    for (auto& count : g_noted) {
        count.store(0, std::memory_order_relaxed);
    }
    g_compared.store(0, std::memory_order_relaxed);
    g_cleared_between = {};
    g_words = g_binds = g_raised = g_consumed = 0;
}

bool Watched(u32 index) {
    return index < NumRegs && ((WatchMask[index / 64] >> (index % 64)) & 1) != 0;
}

const Flags& Forwarded() {
    return Parts().forward;
}

void Apply(Maxwell3D& view, const Packet& packet) {
    const Flags& forward = Forwarded();
    if (!packet.seed.empty()) {
        std::memcpy(view.regs.reg_array.data(), packet.seed.data(), sizeof(view.regs));
        view.dirty.flags = (view.dirty.flags & ~forward) | packet.raised;
    } else {
        for (const RegPair& pair : packet.words) {
            view.regs.reg_array[pair.index] = pair.value;
        }
        view.dirty.flags =
            ((view.dirty.flags & ~packet.consumed) | packet.raised) & ~packet.cleared;
    }
    for (const BindValue& bind : packet.binds) {
        auto& info = view.state.shader_stages[bind.stage].const_buffers[bind.slot];
        info.address = bind.address;
        info.size = bind.size;
        info.enabled = bind.enabled != 0;
    }
    if (packet.has_values) {
        view.engine_state = packet.values.engine_state;
        view.SetNativeTuringMacroMode(packet.values.native);
    }
    auto& state = view.draw_manager.draw_state;
    const DrawValues& draw = packet.draw;
    state.topology = draw.topology;
    state.draw_mode = draw.draw_mode;
    state.draw_indexed = draw.draw_indexed != 0;
    state.draw_element_count = draw.draw_element_count;
    state.base_index = draw.base_index;
    state.vertex_buffer = draw.vertex_buffer;
    state.index_buffer = draw.index_buffer;
    state.base_instance = draw.base_instance;
    state.instance_count = draw.instance_count;
    state.drawn_instances = draw.drawn_instances;
}

Writes::Writes(Maxwell3D& engine)
    : view{std::make_unique<Maxwell3D>(engine.GetMemoryManager())}, hle_sent(HleWords.size()) {}

Writes::~Writes() {
    if (seeded) {
        LogCounts("engine released");
    }
}

void NoteWrite(Writes& writes, u32 index) noexcept {
    writes.Note(index);
}

void Writes::NoteBind(size_t stage, u32 slot) noexcept {
    if (stage >= Regs::MaxShaderStage || slot >= Slots) {
        return;
    }
    const size_t bit = stage * Slots + slot;
    binds[bit / 64] |= u64{1} << (bit % 64);
}

Packet Writes::Pack(Maxwell3D& engine, bool events) {
    Packet packet;
    const Flags now = engine.dirty.flags & Forwarded();
    const EngineValues values = ValuesOf(engine);
    packet.draw = DrawOf(engine);
    const auto& reg_array = engine.regs.reg_array;
    if (!seeded) {
        seeded = true;
        packet.seed.assign(reg_array.begin(), reg_array.end());
        for (u32 stage = 0; stage < Regs::MaxShaderStage; ++stage) {
            for (u32 slot = 0; slot < Slots; ++slot) {
                packet.binds.push_back(BindOf(engine, stage, slot));
            }
        }
        packet.raised = now;
        packet.values = values;
        packet.has_values = true;
        key_words_written = true;
        key_dirty = 0;
        dirty = {};
        binds = {};
        consumed.reset();
        after = now;
        sent = values;
        std::ranges::transform(HleWords, hle_sent.begin(),
                               [&](u32 index) { return reg_array[index]; });
        return packet;
    }
    for (size_t i = 0; i < HleWords.size(); ++i) {
        if (reg_array[HleWords[i]] != hle_sent[i]) {
            Note(HleWords[i]);
        }
    }
    key_words_written |= key_dirty != 0;
    key_dirty = 0;
    for (u32 word = 0; word < dirty.size(); ++word) {
        for (u64 bits = std::exchange(dirty[word], 0); bits != 0; bits &= bits - 1) {
            const u32 index = word * 64 + static_cast<u32>(std::countr_zero(bits));
            packet.words.push_back({index, reg_array[index]});
        }
    }
    std::ranges::transform(HleWords, hle_sent.begin(), [&](u32 index) { return reg_array[index]; });
    if ((binds[0] | binds[1]) != 0) {
        packet.binds.reserve(
            static_cast<size_t>(std::popcount(binds[0]) + std::popcount(binds[1])));
    }
    for (u32 word = 0; word < binds.size(); ++word) {
        for (u64 bits = binds[word]; bits != 0; bits &= bits - 1) {
            const u32 bit = word * 64 + static_cast<u32>(std::countr_zero(bits));
            packet.binds.push_back(BindOf(engine, bit / Slots, bit % Slots));
        }
    }
    binds = {};
    if (values != sent) {
        packet.values = values;
        packet.has_values = true;
        sent = values;
        key_words_written = true;
    }
    if (events) {
        packet.raised = now;
        g_words += packet.words.size();
        g_binds += packet.binds.size();
        g_raised += packet.raised.count();
        return packet;
    }
    packet.consumed = consumed;
    consumed.reset();
    packet.raised = now & ~after;
    packet.cleared = after & ~now;
    for (u32 flag = 0; packet.cleared.any() && flag < packet.cleared.size(); ++flag) {
        g_cleared_between[flag] += packet.cleared.test(flag) ? 1 : 0;
    }
    after = now;
    g_words += packet.words.size();
    g_binds += packet.binds.size();
    g_raised += packet.raised.count();
    g_consumed += packet.consumed.count();
    return packet;
}

bool Writes::KeyWordsWritten(Maxwell3D& engine) const {
    if (key_words_written || key_dirty != 0 || !seeded || ValuesOf(engine) != sent) {
        return true;
    }
    for (size_t i = 0; i < HleWords.size(); ++i) {
        const u32 index = HleWords[i];
        if (((KeyMask[index / 64] >> (index % 64)) & 1) != 0 &&
            engine.regs.reg_array[index] != hle_sent[i]) {
            return true;
        }
    }
    return false;
}

void Writes::DrawDone(const Maxwell3D& engine) {
    const Flags exit = engine.dirty.flags & Forwarded();
    consumed |= after & ~exit;
    after &= exit;
}

bool Compare(Maxwell3D& view, Maxwell3D& engine) {
    g_compared.fetch_add(1, std::memory_order_relaxed);
    bool differs = false;
    const auto& v = view.regs.reg_array;
    const auto& e = engine.regs.reg_array;
    u32 differing = 0;
    u32 first = NumRegs;
    constexpr u32 Block = 64;
    for (u32 begin = 0; begin < NumRegs; begin += Block) {
        const u32 words = (std::min)(Block, NumRegs - begin);
        if (std::memcmp(&v[begin], &e[begin], words * sizeof(u32)) == 0) {
            continue;
        }
        for (u32 i = begin; i < begin + words; ++i) {
            if (Watched(i) && v[i] != e[i]) {
                first = first == NumRegs ? i : first;
                ++differing;
            }
        }
    }
    if (differing != 0) {
        differs = true;
        Note(Check::ViewWords, fmt::format("{} words, first {:#x}: view {:#x}, engine {:#x}",
                                           differing, first, v[first], e[first]));
    }
    const Flags flags =
        GpuSplit::UnlendEnabled() ? Flags{} : (view.dirty.flags ^ engine.dirty.flags) & Forwarded();
    if (flags.any()) {
        const u32 flag = FirstFlag(flags);
        differs = true;
        Note(Check::ViewFlags, fmt::format("{} flags, first {}: view {}, engine {}", flags.count(),
                                           flag, view.dirty.flags.test(flag),
                                           engine.dirty.flags.test(flag)));
    }
    if (!SameDraw(DrawOf(view), DrawOf(engine))) {
        differs = true;
        Note(Check::ViewDraw, "the draw values against the draw manager");
    }
    const auto& vr = view.regs;
    const auto& er = engine.regs;
    if (!SameBytes(vr.tex_header, er.tex_header) || !SameBytes(vr.tex_sampler, er.tex_sampler) ||
        vr.sampler_binding != er.sampler_binding) {
        differs = true;
        Note(Check::ViewPools, "the TIC/TSC pools or the sampler binding");
    }
    if (ValuesOf(view) != ValuesOf(engine)) {
        differs = true;
        Note(Check::ViewValues, fmt::format("engine hint {} native {}, engine {} native {}",
                                            static_cast<u32>(view.engine_state),
                                            ValuesOf(view).native,
                                            static_cast<u32>(engine.engine_state),
                                            ValuesOf(engine).native));
    }
    for (u32 stage = 0; stage < Regs::MaxShaderStage; ++stage) {
        for (u32 slot = 0; slot < Slots; ++slot) {
            const auto& a = view.state.shader_stages[stage].const_buffers[slot];
            const auto& b = engine.state.shader_stages[stage].const_buffers[slot];
            if (a.address != b.address || a.size != b.size || a.enabled != b.enabled) {
                Note(Check::ViewValues,
                     fmt::format("constant buffer {} of stage {}: view {:#x}+{:#x} {}, engine "
                                 "{:#x}+{:#x} {}",
                                 slot, stage, a.address, a.size, a.enabled, b.address, b.size,
                                 b.enabled));
                return true;
            }
        }
    }
    return differs;
}

DrawCheck::DrawCheck(Maxwell3D& engine_, bool verify) {
    if (!Enabled()) {
        return;
    }
    const bool unlend = GpuSplit::UnlendEnabled();
    engine = &engine_;
    if (!engine_.render_view_writes) {
        engine_.render_view_writes = std::make_unique<Writes>(engine_);
    }
    writes = engine_.render_view_writes.get();
    Maxwell3D* const view = &writes->View();
    GpuSplit::Get().LendRender(
        [view, packet = writes->Pack(engine_, unlend)] { Apply(*view, packet); });
    compared = !unlend || verify || VerifyAll();
    if (!compared) {
        return;
    }
    differed = Compare(*view, engine_);
    static u32 draws{};
    static auto next = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    if (++draws % 1024 == 0 && std::chrono::steady_clock::now() >= next) {
        next = std::chrono::steady_clock::now() + std::chrono::seconds{10};
        LogCounts("10 s");
    }
}

DrawCheck::~DrawCheck() {
    if (writes != nullptr && !GpuSplit::UnlendEnabled()) {
        writes->DrawDone(*engine);
    }
}

void LogCounts(std::string_view when) {
    std::string checks;
    for (size_t i = 0; i < CheckNames.size(); ++i) {
        checks += fmt::format("{}{} {}", i == 0 ? "" : ", ", CheckNames[i],
                              g_noted[i].load(std::memory_order_relaxed));
    }
    std::array<u32, 256> order;
    for (u32 i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::ranges::stable_sort(order, [](u32 a, u32 b) {
        return g_cleared_between[a] > g_cleared_between[b];
    });
    std::string cleared;
    for (size_t i = 0; i < 8 && g_cleared_between[order[i]] != 0; ++i) {
        cleared += fmt::format(" {}:{}", order[i], g_cleared_between[order[i]]);
    }
    LOG_INFO(HW_GPU,
             "Render view verify ({}): {} draws compared; must be 0: {}; carried {} words, {} "
             "bindings, {} flag raises, {} flags the previous draw cleared; forwarded flags the "
             "back cleared between draws (flag:times):{}",
             when, Compared(), checks, g_words, g_binds, g_raised, g_consumed,
             cleared.empty() ? " none" : cleared);
}

} // namespace VideoCommon::Nx2RenderView
