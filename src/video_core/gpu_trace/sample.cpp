// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <initializer_list>
#include <span>
#include <vector>

#include <nlohmann/json.hpp>

#include "shader_recompiler/program_header.h"
#include "video_core/dma_pusher.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_trace/sample.h"
#include "video_core/gpu_trace/trace_file.h"
#include "video_core/pte_kind.h"
#include "video_core/textures/decoders.h"

namespace VideoCommon::GpuTrace {
namespace {
using Tegra::Engines::Maxwell3D;

constexpr u64 DeviceBase = 0x40000000;
constexpr u64 PhysicalBase = 0x40000000;
constexpr u64 GpuBase = 0x200000000;
constexpr u64 Size = 16ULL << 20;
constexpr u64 PushOffset = 0;
constexpr u64 ImageOffset = 0x100000;
constexpr u64 TargetOffset = 0x200000;
constexpr u32 Width = 256;
constexpr u32 Height = 144;
constexpr s32 Channel = 1;

struct Rect {
    u32 x0, y0, x1, y1;
};

class Commands {
public:
    void Method(u32 method, std::initializer_list<u32> arguments) {
        Tegra::CommandHeader header{};
        header.method.Assign(method);
        header.subchannel.Assign(0);
        header.arg_count.Assign(static_cast<u32>(arguments.size()));
        header.mode.Assign(Tegra::SubmissionMode::Increasing);
        words.push_back(header.argument);
        words.insert(words.end(), arguments);
    }

    void Clear(std::array<f32, 4> color, const Rect* scissor) {
        if (scissor != nullptr) {
            Method(MAXWELL3D_REG_INDEX(scissor_test),
                   {1, scissor->x0 | (scissor->x1 << 16), scissor->y0 | (scissor->y1 << 16)});
        }
        Method(MAXWELL3D_REG_INDEX(clear_control), {scissor != nullptr ? 0x100u : 0u});
        Method(MAXWELL3D_REG_INDEX(clear_color),
               {std::bit_cast<u32>(color[0]), std::bit_cast<u32>(color[1]),
                std::bit_cast<u32>(color[2]), std::bit_cast<u32>(color[3])});
        // R, G, B and A of render target 0, layer 0.
        Method(MAXWELL3D_REG_INDEX(clear_surface), {0x3c});
        if (scissor != nullptr) {
            Method(MAXWELL3D_REG_INDEX(scissor_test), {0});
        }
    }

    std::vector<u32> words;
};

std::span<const u8> AsBytes(const auto& values) {
    return {reinterpret_cast<const u8*>(values.data()), values.size() * sizeof(values[0])};
}

/// Presented images and render targets are block linear with 16-GOB blocks, as a game's are.
constexpr u32 BlockHeightLog2 = 4;

std::vector<u32> Image(u32 seed) {
    std::vector<u32> pixels(Width * Height);
    for (u32 y = 0; y < Height; ++y) {
        for (u32 x = 0; x < Width; ++x) {
            const u32 r = (x * 255 / (Width - 1)) ^ (seed * 0x3f);
            const u32 g = (y * 255 / (Height - 1));
            const u32 b = ((x / 16 + y / 16 + seed) % 2) * 0xff;
            pixels[y * Width + x] = (r & 0xff) | (g << 8) | (b << 16) | 0xff000000;
        }
    }
    std::vector<u32> tiled(Tegra::Texture::CalculateSize(true, 4, Width, Height, 1,
                                                         BlockHeightLog2, 0) /
                           sizeof(u32));
    Tegra::Texture::SwizzleTexture(
        std::span<u8>{reinterpret_cast<u8*>(tiled.data()), tiled.size() * sizeof(u32)},
        AsBytes(pixels), 4, Width, Height, 1, BlockHeightLog2, 0, 1, true);
    return tiled;
}

void Present(Writer& writer, u64 offset, u64 time_ns) {
    PresentLayer layer{};
    layer.address = DeviceBase + offset;
    layer.width = Width;
    layer.height = Height;
    layer.stride = Width;
    layer.pixel_format = 1; // Rgba8888
    layer.crop_right = static_cast<s32>(Width);
    layer.crop_bottom = static_cast<s32>(Height);
    layer.layer_stack_mask = 1;
    layer.native_tiling = 1;
    const PresentRecord header{time_ns, 1, 0};
    const std::array<std::span<const u8>, 2> parts{
        std::span<const u8>{reinterpret_cast<const u8*>(&header), sizeof(header)},
        std::span<const u8>{reinterpret_cast<const u8*>(&layer), sizeof(layer)}};
    writer.Write(RecordType::Present, parts);
}

/// The commands go to memory first (a CPU write), as a game writes them.
void Submit(Writer& writer, u64 offset, const Commands& commands) {
    writer.WriteMemory(PhysicalBase + offset, AsBytes(commands.words));
    Tegra::CommandListHeader entry{};
    entry.addr.Assign(GpuBase + offset);
    entry.size.Assign(commands.words.size());
    const SubmitRecord header{Channel, 1, 0, 0, 0, 0};
    const std::array<std::span<const u8>, 2> parts{
        std::span<const u8>{reinterpret_cast<const u8*>(&header), sizeof(header)},
        std::span<const u8>{reinterpret_cast<const u8*>(&entry.raw), sizeof(entry.raw)}};
    writer.Write(RecordType::Submit, parts);
}

/// The config, one address space mapping the whole sample memory and one channel.
void WriteSampleSnapshot(Writer& writer) {
    const nlohmann::json config{
        {"program_id", "0400000000000001"}, {"switch2", true},  {"synthetic", true},
        {"memory_layout", 0},               {"width", Width},   {"height", Height},
    };
    const std::string text = config.dump();
    writer.Write(RecordType::Config, AsBytes(text));
    writer.WritePod(RecordType::AddressSpace, AddressSpaceRecord{0, 40, 1ULL << 34, 16, 12});
    writer.WritePod(RecordType::SmmuMap, SmmuMapRecord{DeviceBase, PhysicalBase, Size});
    writer.WritePod(RecordType::GmmuMap,
                    GmmuMapRecord{0, static_cast<u32>(Tegra::PTEKind::PITCH), GpuBase, DeviceBase,
                                  Size, 0, 0});
    writer.WritePod(RecordType::Channel, ChannelRecord{Channel, 0, 0x0400000000000001ULL});
    writer.Write(RecordType::SnapshotEnd, std::span<const u8>{});
}

Commands BindTarget() {
    Commands commands;
    const u64 address = GpuBase + TargetOffset;
    commands.Method(MAXWELL3D_REG_INDEX(rt),
                    {static_cast<u32>(address >> 32), static_cast<u32>(address), Width, Height,
                     static_cast<u32>(Tegra::RenderTargetFormat::A8B8G8R8_UNORM),
                     BlockHeightLog2 << 4, 1, 0, 0});
    commands.Method(MAXWELL3D_REG_INDEX(rt_control), {1});
    commands.Method(MAXWELL3D_REG_INDEX(surface_clip), {Width << 16, Height << 16});
    return commands;
}

constexpr u64 ProgramOffset = 0x100000;
constexpr u64 VertexProgramOffset = ProgramOffset;
constexpr u64 PixelProgramOffset = ProgramOffset + 0x1000;
constexpr u64 VertexOffset = 0x110000;
constexpr u64 IndexOffset = 0x111000;
constexpr u64 UniformOffset = 0x120000;
constexpr u64 FramePushSize = 0x10000;
constexpr u32 UniformSize = 0x100;
constexpr u32 UniformBank = 3;
constexpr f32 FloatWidth = static_cast<f32>(Width);
constexpr f32 FloatHeight = static_cast<f32>(Height);
constexpr u32 DrawFrames = 4;
constexpr u32 DrawsPerFrame = 300;
/// Switch 2 programs start after a 128-byte header whose first 0x50 bytes are the SPH.
constexpr size_t NativeHeaderSize = 128;

using Instruction = std::array<u32, 4>;

/// c[3][offset] as an Ampere constant buffer operand.
constexpr u32 Uniform(u32 offset) {
    return (offset << 6) | (UniformBank << 22);
}

constexpr Instruction Exit{0x0000794d, 0, 0x03800000, 0};

/// Position = attribute 0 (x, y) * c[3][0x0, 0x4] + c[3][0x8, 0xc], z = 0, w = 1.
constexpr std::array VertexProgram{
    Instruction{0xff007321, 0x000080ff, 0x00000400, 0},   // ALD.64 R0, a[0x80]
    Instruction{0x00007a20, Uniform(0x0), 0x00400000, 0}, // FMUL R0, R0, c[3][0x0]
    Instruction{0x01017a20, Uniform(0x4), 0x00400000, 0}, // FMUL R1, R1, c[3][0x4]
    Instruction{0x00007a21, Uniform(0x8), 0, 0},          // FADD R0, R0, c[3][0x8]
    Instruction{0x01017a21, Uniform(0xc), 0, 0},          // FADD R1, R1, c[3][0xc]
    Instruction{0x00027802, 0, 0x00000f00, 0},            // MOV R2, 0
    Instruction{0x00037802, 0x3f800000, 0x00000f00, 0},   // MOV R3, 1.0
    Instruction{0xff007322, 0x00007000, 0x00000cff, 0},   // AST.128 a[0x70], R0
    Exit,
};

/// Render target 0 = c[3][0x10..0x1c].
constexpr std::array PixelProgram{
    Instruction{0x00007a02, Uniform(0x10), 0x00000f00, 0}, // MOV R0, c[3][0x10]
    Instruction{0x00017a02, Uniform(0x14), 0x00000f00, 0}, // MOV R1, c[3][0x14]
    Instruction{0x00027a02, Uniform(0x18), 0x00000f00, 0}, // MOV R2, c[3][0x18]
    Instruction{0x00037a02, Uniform(0x1c), 0x00000f00, 0}, // MOV R3, c[3][0x1c]
    Exit,
};

std::vector<u8> Program(std::span<const Instruction> code, bool pixel) {
    Shader::ProgramHeader sph;
    std::memset(&sph, 0, sizeof(sph));
    sph.common0.sph_type.Assign(pixel ? 2 : 1);
    sph.common0.version.Assign(3);
    sph.common0.shader_type.Assign(static_cast<u32>(pixel ? Maxwell3D::Regs::ShaderType::Pixel
                                                          : Maxwell3D::Regs::ShaderType::VertexB));
    if (pixel) {
        sph.ps.omap.target = 0xf;
    } else {
        sph.vtg.imap_generic_vector[0] = 0x3;
        sph.vtg.omap_systemb.raw = 0xf0;
    }
    std::vector<u8> bytes(NativeHeaderSize + code.size_bytes());
    std::memcpy(bytes.data(), &sph, sizeof(sph));
    std::memcpy(bytes.data() + NativeHeaderSize, code.data(), code.size_bytes());
    return bytes;
}

/// A unit quad as a triangle list (vertices 0-5), then as four corners for the index buffer.
constexpr std::array<f32, 20> Vertices{0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1,
                                       0, 0, 1, 0, 0, 1, 1, 1};
constexpr std::array<u16, 12> Indices{6, 7, 8, 8, 7, 9, 9, 8, 7, 7, 8, 6};

void Viewport(Commands& commands, f32 x, f32 y, f32 width, f32 height) {
    commands.Method(MAXWELL3D_REG_INDEX(viewport_transform),
                    {std::bit_cast<u32>(width / 2), std::bit_cast<u32>(height / 2),
                     std::bit_cast<u32>(0.5f), std::bit_cast<u32>(x + width / 2),
                     std::bit_cast<u32>(y + height / 2), std::bit_cast<u32>(0.5f)});
}

Commands DrawSetup() {
    Commands commands = BindTarget();
    using Pipeline = Maxwell3D::Regs::Pipeline;
    using ShaderType = Maxwell3D::Regs::ShaderType;
    // Switch 2 programs are bound by address, in the pipeline's reserved words.
    const auto bind_program = [&commands](ShaderType type, u64 offset) {
        const u32 index = static_cast<u32>(type);
        const u32 method = static_cast<u32>(MAXWELL3D_REG_INDEX(pipelines) +
                                            index * sizeof(Pipeline) / sizeof(u32));
        const u64 address = GpuBase + offset;
        commands.Method(method, {1 | (index << 4)});
        commands.Method(method + offsetof(Pipeline, reserved) / sizeof(u32),
                        {static_cast<u32>(address >> 32), static_cast<u32>(address)});
    };
    bind_program(ShaderType::VertexB, VertexProgramOffset);
    bind_program(ShaderType::Pixel, PixelProgramOffset);

    Maxwell3D::Regs::VertexAttribute attribute{};
    attribute.size.Assign(Maxwell3D::Regs::VertexAttribute::Size::Size_R32_G32);
    attribute.type.Assign(Maxwell3D::Regs::VertexAttribute::Type::Float);
    commands.Method(MAXWELL3D_REG_INDEX(vertex_attrib_format), {attribute.hex});
    const u64 vertices = GpuBase + VertexOffset;
    // SET_VERTEX_STREAM_SIZE_A/B of stream 0 (NativeVertexStreamSize).
    commands.Method(0x600 / sizeof(u32), {0, static_cast<u32>(sizeof(Vertices))});
    commands.Method(MAXWELL3D_REG_INDEX(vertex_streams),
                    {(1u << 12) | (2 * sizeof(f32)), static_cast<u32>(vertices >> 32),
                     static_cast<u32>(vertices)});
    const u64 indices = GpuBase + IndexOffset;
    const u64 indices_end = indices + sizeof(Indices);
    commands.Method(MAXWELL3D_REG_INDEX(index_buffer),
                    {static_cast<u32>(indices >> 32), static_cast<u32>(indices),
                     static_cast<u32>(indices_end >> 32), static_cast<u32>(indices_end),
                     static_cast<u32>(Maxwell3D::Regs::IndexFormat::UnsignedShort)});

    commands.Method(MAXWELL3D_REG_INDEX(viewport_scale_offset_enabled), {1});
    Viewport(commands, 0, 0, FloatWidth, FloatHeight);
    commands.Method(MAXWELL3D_REG_INDEX(viewports),
                    {Width << 16, Height << 16, std::bit_cast<u32>(0.0f),
                     std::bit_cast<u32>(1.0f)});

    const u64 uniforms = GpuBase + UniformOffset;
    commands.Method(MAXWELL3D_REG_INDEX(const_buffer),
                    {UniformSize, static_cast<u32>(uniforms >> 32), static_cast<u32>(uniforms)});
    commands.Method(MAXWELL3D_REG_INDEX(bind_groups[0].raw_config), {1 | (UniformBank << 4)});
    commands.Method(MAXWELL3D_REG_INDEX(bind_groups[4].raw_config), {1 | (UniformBank << 4)});
    commands.Method(MAXWELL3D_REG_INDEX(blend.color_source),
                    {static_cast<u32>(Maxwell3D::Regs::Blend::Factor::SourceAlpha_D3D),
                     static_cast<u32>(Maxwell3D::Regs::Blend::Factor::OneMinusSourceAlpha_D3D)});
    return commands;
}

/// Draws a quad at pixel rectangle (x, y, width, height) in `color`, written to the constant
/// buffer just before the draw.
void Quad(Commands& commands, u32 x, u32 y, u32 width, u32 height, std::array<f32, 4> color,
          bool indexed, u32 first) {
    const f32 scale_x = 2.0f * static_cast<f32>(width) / FloatWidth;
    const f32 scale_y = 2.0f * static_cast<f32>(height) / FloatHeight;
    const f32 offset_x = 2.0f * static_cast<f32>(x) / FloatWidth - 1.0f;
    const f32 offset_y = 2.0f * static_cast<f32>(y) / FloatHeight - 1.0f;
    commands.Method(MAXWELL3D_REG_INDEX(const_buffer.offset), {0});
    commands.Method(MAXWELL3D_REG_INDEX(const_buffer.buffer),
                    {std::bit_cast<u32>(scale_x), std::bit_cast<u32>(scale_y),
                     std::bit_cast<u32>(offset_x), std::bit_cast<u32>(offset_y),
                     std::bit_cast<u32>(color[0]), std::bit_cast<u32>(color[1]),
                     std::bit_cast<u32>(color[2]), std::bit_cast<u32>(color[3])});
    if (indexed) {
        commands.Method(MAXWELL3D_REG_INDEX(index_buffer.first), {first, 6});
    } else {
        commands.Method(MAXWELL3D_REG_INDEX(vertex_buffer.first), {first, 6});
    }
    commands.Method(MAXWELL3D_REG_INDEX(draw.begin),
                    {static_cast<u32>(Maxwell3D::Regs::PrimitiveTopology::Triangles)});
    commands.Method(MAXWELL3D_REG_INDEX(draw.end), {0});
}

/// Draws [first, last) of one frame's overlapping quads with state changes and clears between
/// them. The seed carries the frame's quads from one part to the next.
Commands DrawFrame(u32 frame, u32 first, u32 last, u32& seed) {
    Commands commands = frame == 0 && first == 0 ? DrawSetup() : Commands{};
    if (first == 0) {
        commands.Method(MAXWELL3D_REG_INDEX(depth_test_enable), {0});
        commands.Clear({0.1f * static_cast<f32>(frame), 0.1f, 0.2f, 1.0f}, nullptr);
    }
    const auto next = [&seed] {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        return seed;
    };
    const Rect scissor{32 + frame * 8, 24, 200, 120};
    const Rect dark{184, 88, 248, 136};
    const Rect light{8, 8, 72, 56};
    for (u32 draw = first; draw < last; ++draw) {
        switch (draw) {
        case 80:
            commands.Clear({0.0f, 0.0f, 0.0f, 1.0f}, &dark);
            break;
        case 100:
            commands.Method(MAXWELL3D_REG_INDEX(scissor_test),
                            {1, scissor.x0 | (scissor.x1 << 16), scissor.y0 | (scissor.y1 << 16)});
            break;
        case 150:
            commands.Method(MAXWELL3D_REG_INDEX(scissor_test), {0});
            commands.Method(MAXWELL3D_REG_INDEX(blend.enable), {1});
            break;
        case 200:
            commands.Method(MAXWELL3D_REG_INDEX(blend.enable), {0});
            commands.Method(MAXWELL3D_REG_INDEX(color_mask), {0x0101});
            break;
        case 230:
            commands.Method(MAXWELL3D_REG_INDEX(color_mask), {0x1111});
            Viewport(commands, FloatWidth / 4, FloatHeight / 4, FloatWidth / 2, FloatHeight / 2);
            break;
        case 260:
            Viewport(commands, 0, 0, FloatWidth, FloatHeight);
            commands.Clear({1.0f, 1.0f, 1.0f, 1.0f}, &light);
            break;
        default:
            break;
        }
        const u32 width = 8 + next() % 40;
        const u32 height = 8 + next() % 32;
        const u32 x = next() % (Width - width);
        const u32 y = next() % (Height - height);
        const u32 rgb = next();
        const std::array<f32, 4> color{
            static_cast<f32>(rgb & 0xff) / 255.0f, static_cast<f32>((rgb >> 8) & 0xff) / 255.0f,
            static_cast<f32>((rgb >> 16) & 0xff) / 255.0f, draw >= 150 && draw < 200 ? 0.5f : 1.0f};
        const bool indexed = draw % 3 == 0;
        Quad(commands, x, y, width, height, color, indexed, indexed ? (draw % 2) * 6 : 0);
    }
    return commands;
}

Commands DrawFrame(u32 frame) {
    u32 seed = 0x9e3779b9u * (frame + 1);
    return DrawFrame(frame, 0, DrawsPerFrame, seed);
}
} // Anonymous namespace

bool WriteSampleTrace(const std::filesystem::path& path) {
    Writer writer{path};
    if (!writer.IsOpen()) {
        return false;
    }
    WriteSampleSnapshot(writer);

    constexpr u64 Frame = 16'666'667;
    writer.WriteMemory(PhysicalBase + ImageOffset, AsBytes(Image(0)));
    Present(writer, ImageOffset, Frame);

    Commands first = BindTarget();
    const Rect square{64, 36, 192, 108};
    first.Clear({1.0f, 0.0f, 0.0f, 1.0f}, nullptr);
    first.Clear({0.0f, 1.0f, 0.0f, 1.0f}, &square);
    Submit(writer, PushOffset, first);
    Present(writer, TargetOffset, Frame * 2);

    writer.WriteMemory(PhysicalBase + ImageOffset, AsBytes(Image(1)));
    Present(writer, ImageOffset, Frame * 3);

    Commands second;
    const Rect left{16, 16, 96, 128};
    const Rect right{160, 16, 240, 128};
    second.Clear({0.0f, 0.0f, 1.0f, 1.0f}, nullptr);
    second.Clear({1.0f, 1.0f, 0.0f, 1.0f}, &left);
    second.Clear({1.0f, 0.0f, 1.0f, 1.0f}, &right);
    Submit(writer, PushOffset + 0x8000, second);
    Present(writer, TargetOffset, Frame * 4);
    return writer.Finish();
}

bool WriteDrawSampleTrace(const std::filesystem::path& path) {
    Writer writer{path};
    if (!writer.IsOpen()) {
        return false;
    }
    WriteSampleSnapshot(writer);
    writer.WriteMemory(PhysicalBase + VertexProgramOffset, Program(VertexProgram, false));
    writer.WriteMemory(PhysicalBase + PixelProgramOffset, Program(PixelProgram, true));
    writer.WriteMemory(PhysicalBase + VertexOffset, AsBytes(Vertices));
    writer.WriteMemory(PhysicalBase + IndexOffset, AsBytes(Indices));
    constexpr u64 Frame = 16'666'667;
    for (u32 frame = 0; frame < DrawFrames; ++frame) {
        Submit(writer, PushOffset + frame * FramePushSize, DrawFrame(frame));
        Present(writer, TargetOffset, Frame * (frame + 1));
    }
    return writer.Finish();
}

bool WriteUploadSampleTrace(const std::filesystem::path& path) {
    Writer writer{path};
    if (!writer.IsOpen()) {
        return false;
    }
    WriteSampleSnapshot(writer);
    writer.WriteMemory(PhysicalBase + VertexProgramOffset, Program(VertexProgram, false));
    writer.WriteMemory(PhysicalBase + PixelProgramOffset, Program(PixelProgram, true));
    constexpr u32 Parts = 8;
    constexpr u64 PartPushSize = FramePushSize / Parts;
    constexpr u64 Frame = 16'666'667;
    for (u32 frame = 0; frame < DrawFrames; ++frame) {
        u32 seed = 0x9e3779b9u * (frame + 1);
        for (u32 part = 0; part < Parts; ++part) {
            // New vertex and index data before every part: uploads between the draws.
            const u32 step = frame * Parts + part;
            std::array<f32, Vertices.size()> vertices = Vertices;
            for (f32& coordinate : vertices) {
                coordinate *= 1.0f - 0.03f * static_cast<f32>(step % 9);
            }
            std::array<u16, Indices.size()> indices = Indices;
            std::rotate(indices.begin(), indices.begin() + 3 * (step % 4), indices.end());
            writer.WriteMemory(PhysicalBase + VertexOffset, AsBytes(vertices));
            writer.WriteMemory(PhysicalBase + IndexOffset, AsBytes(indices));
            const u32 first = DrawsPerFrame * part / Parts;
            const u32 last = DrawsPerFrame * (part + 1) / Parts;
            Submit(writer, PushOffset + frame * FramePushSize + part * PartPushSize,
                   DrawFrame(frame, first, last, seed));
        }
        Present(writer, TargetOffset, Frame * (frame + 1));
    }
    return writer.Finish();
}

} // namespace VideoCommon::GpuTrace
