// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The vertex shader of a mesh program emulated with compute (Profile::emulate_mesh_with_compute).
// The task and mesh compute shaders leave one record per mesh workgroup in the draw's scratch
// buffer (MeshInfo::RecordWords: primitive count, attribute region, map bytes). The draw issues
// `PrimitiveVertices() * max_primitives` vertices per record; this shader finds the record, the
// primitive and the corner from gl_VertexIndex, reads the corner's vertex from the attribute
// region and writes the same outputs the mesh program would have. Primitives past the record's
// count collapse to one point (zero area), which the rasterizer drops.

#include <algorithm>
#include <array>
#include <vector>

#include <sirit/sirit.h>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/attribute.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/shader_info.h"

namespace Shader::Backend::SPIRV {
namespace {
using Sirit::Id;
} // Anonymous namespace

bool CanEmulateMeshVertex(const MeshInfo& mesh) {
    // Per-primitive values (layer, viewport, primitive id, per-primitive generics) have no
    // vertex shader equivalent.
    return mesh.primitive_lines.empty() && !mesh.vertex_lines.empty() && mesh.max_primitives != 0;
}

std::vector<u32> EmitMeshEmulationVertexShader(const Profile& profile, const RuntimeInfo& runtime_info,
                                               const MeshInfo& mesh) {
    Sirit::Module m{(std::max)(profile.supported_spirv, 0x00010300U)};
    m.SetMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);
    m.AddCapability(spv::Capability::Shader);

    const Id void_type{m.TypeVoid()};
    const Id u32_type{m.TypeInt(32, false)};
    const Id f32_type{m.TypeFloat(32)};
    const Id bool_type{m.TypeBool()};
    const Id vec4_type{m.TypeVector(f32_type, 4)};
    const auto u32c = [&](u32 value) { return m.Constant(u32_type, value); };
    const auto f32c = [&](f32 value) { return m.Constant(f32_type, value); };

    // The scratch buffer, read only.
    const Id scratch_array{m.TypeRuntimeArray(u32_type)};
    m.Decorate(scratch_array, spv::Decoration::ArrayStride, 4U);
    const Id scratch_struct{m.TypeStruct(scratch_array)};
    m.Decorate(scratch_struct, spv::Decoration::Block);
    m.MemberDecorate(scratch_struct, 0, spv::Decoration::Offset, 0U);
    m.MemberDecorate(scratch_struct, 0, spv::Decoration::NonWritable);
    const Id scratch{m.AddGlobalVariable(m.TypePointer(spv::StorageClass::StorageBuffer, scratch_struct),
                                         spv::StorageClass::StorageBuffer)};
    m.Decorate(scratch, spv::Decoration::DescriptorSet, MeshEmulation::ScratchSet);
    m.Decorate(scratch, spv::Decoration::Binding, MeshEmulation::ScratchBinding);
    m.Name(scratch, "mesh_scratch");
    const Id scratch_u32{m.TypePointer(spv::StorageClass::StorageBuffer, u32_type)};

    std::vector<Id> interfaces;
    if (profile.supported_spirv >= 0x00010400) {
        interfaces.push_back(scratch);
    }

    const Id vertex_index{m.AddGlobalVariable(m.TypePointer(spv::StorageClass::Input, u32_type),
                                              spv::StorageClass::Input)};
    m.Decorate(vertex_index, spv::Decoration::BuiltIn, spv::BuiltIn::VertexIndex);
    interfaces.push_back(vertex_index);

    // Outputs the lines feed: position, point size, and one vec4 per generic attribute.
    const Id output_vec4{m.TypePointer(spv::StorageClass::Output, vec4_type)};
    const Id output_f32{m.TypePointer(spv::StorageClass::Output, f32_type)};
    const Id position{m.AddGlobalVariable(output_vec4, spv::StorageClass::Output)};
    m.Decorate(position, spv::Decoration::BuiltIn, spv::BuiltIn::Position);
    interfaces.push_back(position);
    const bool points{mesh.PrimitiveVertices() == 1};
    const bool writes_point_size{std::ranges::find(mesh.vertex_lines,
                                                   static_cast<u32>(IR::Attribute::PointSize)) !=
                                 mesh.vertex_lines.end()};
    Id point_size{};
    if (points) {
        point_size = m.AddGlobalVariable(output_f32, spv::StorageClass::Output);
        m.Decorate(point_size, spv::Decoration::BuiltIn, spv::BuiltIn::PointSize);
        interfaces.push_back(point_size);
    }
    std::array<Id, IR::NUM_GENERICS> generic_outputs{};
    for (const u32 line : mesh.vertex_lines) {
        const IR::Attribute attribute{line};
        if (!IR::IsGeneric(attribute)) {
            continue;
        }
        const u32 index{IR::GenericAttributeIndex(attribute)};
        if (!Sirit::ValidId(generic_outputs[index])) {
            generic_outputs[index] = m.AddGlobalVariable(output_vec4, spv::StorageClass::Output);
            m.Decorate(generic_outputs[index], spv::Decoration::Location, index);
            interfaces.push_back(generic_outputs[index]);
        }
    }

    const Id main{m.OpFunction(void_type, spv::FunctionControlMask::MaskNone, m.TypeFunction(void_type))};
    m.AddLabel(m.OpLabel());

    const u32 primitive_vertices{mesh.PrimitiveVertices()};
    const u32 num_lines{static_cast<u32>(mesh.vertex_lines.size())};
    const u32 attribute_words{mesh.AttributeWords()};
    const auto add = [&](Id a, Id b) { return m.OpIAdd(u32_type, a, b); };
    const auto mul = [&](Id a, Id b) { return m.OpIMul(u32_type, a, b); };
    const auto load_scratch = [&](Id index) {
        return m.OpLoad(u32_type, m.OpAccessChain(scratch_u32, scratch, u32c(0), index));
    };

    const Id vid{m.OpLoad(u32_type, vertex_index)};
    const Id primitive_global{m.OpUDiv(u32_type, vid, u32c(primitive_vertices))};
    const Id corner{m.OpUMod(u32_type, vid, u32c(primitive_vertices))};
    const Id group{m.OpUDiv(u32_type, primitive_global, u32c(mesh.max_primitives))};
    const Id primitive{m.OpUMod(u32_type, primitive_global, u32c(mesh.max_primitives))};
    const Id record{add(load_scratch(u32c(MeshEmulation::ParamRecordsBase)),
                        mul(group, u32c(mesh.RecordWords())))};
    const Id count{load_scratch(record)};
    const Id valid{m.OpULessThan(bool_type, primitive, count)};

    // The vertex index: byte 4 + primitive_vertices * primitive + corner of the map.
    const Id map_byte{add(u32c(4), add(mul(primitive, u32c(primitive_vertices)), corner))};
    const Id map_word{load_scratch(add(record, add(u32c(1 + attribute_words),
                                                   m.OpShiftRightLogical(u32_type, map_byte, u32c(2)))))};
    const Id shift{m.OpShiftLeftLogical(u32_type, m.OpBitwiseAnd(u32_type, map_byte, u32c(3)), u32c(3))};
    const Id raw_vertex{m.OpBitwiseAnd(u32_type, m.OpShiftRightLogical(u32_type, map_word, shift), u32c(0xff))};
    const Id vertex{m.OpUMin(u32_type, raw_vertex, u32c(mesh.max_vertices - 1))};
    // Word of line 0 of this vertex: record + 1 + (vertex / 32) * lines * 32 + vertex % 32.
    const Id first{add(add(record, u32c(1)),
                       add(mul(m.OpShiftRightLogical(u32_type, vertex, u32c(5)), u32c(num_lines * 32)),
                           m.OpBitwiseAnd(u32_type, vertex, u32c(31))))};

    std::array<std::array<Id, 4>, IR::NUM_GENERICS> generic_values{};
    std::array<Id, 4> position_values{f32c(0.0f), f32c(0.0f), f32c(0.0f), f32c(1.0f)};
    Id point_size_value{f32c(runtime_info.fixed_state_point_size.value_or(1.0f))};
    for (u32 line = 0; line < num_lines; ++line) {
        const IR::Attribute attribute{mesh.vertex_lines[line]};
        const bool is_position{attribute >= IR::Attribute::PositionX &&
                               attribute <= IR::Attribute::PositionW};
        const bool is_point_size{attribute == IR::Attribute::PointSize};
        if (!IR::IsGeneric(attribute) && !is_position && !is_point_size) {
            continue;
        }
        const Id value{m.OpBitcast(f32_type, load_scratch(add(first, u32c(line * 32))))};
        if (is_position) {
            position_values[static_cast<u32>(attribute) - static_cast<u32>(IR::Attribute::PositionX)] =
                value;
        } else if (is_point_size) {
            point_size_value = value;
        } else {
            generic_values[IR::GenericAttributeIndex(attribute)][IR::GenericAttributeElement(attribute)] =
                value;
        }
    }
    // Primitives past the count: one point, so they have no area.
    const Id zero{f32c(0.0f)};
    const std::array<Id, 4> collapsed{zero, zero, zero, f32c(1.0f)};
    std::array<Id, 4> selected;
    for (u32 component = 0; component < 4; ++component) {
        selected[component] =
            m.OpSelect(f32_type, valid, position_values[component], collapsed[component]);
    }
    m.OpStore(position, m.OpCompositeConstruct(vec4_type, selected[0], selected[1], selected[2], selected[3]));
    if (points) {
        m.OpStore(point_size, writes_point_size ? point_size_value
                                                : f32c(runtime_info.fixed_state_point_size.value_or(1.0f)));
    }
    for (u32 index = 0; index < IR::NUM_GENERICS; ++index) {
        if (!Sirit::ValidId(generic_outputs[index])) {
            continue;
        }
        std::array<Id, 4> components;
        for (u32 element = 0; element < 4; ++element) {
            components[element] = Sirit::ValidId(generic_values[index][element])
                                      ? generic_values[index][element]
                                      : zero;
        }
        m.OpStore(generic_outputs[index], m.OpCompositeConstruct(vec4_type, components[0], components[1],
                                                                 components[2], components[3]));
    }
    m.OpReturn();
    m.OpFunctionEnd();
    m.AddEntryPoint(spv::ExecutionModel::Vertex, main, "main", std::span<const Id>(interfaces));
    return m.Assemble();
}

} // namespace Shader::Backend::SPIRV
