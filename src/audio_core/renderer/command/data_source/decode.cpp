// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include "audio_core/renderer/command/data_source/decode.h"
#include "audio_core/renderer/command/resample/resample.h"
#include "common/fixed_point.h"
#include "common/logging.h"
#include "common/nx2_sampler.h"
#include "common/scratch_buffer.h"
#include "core/guest_memory.h"
#include "core/memory.h"

namespace AudioCore::Renderer {

constexpr u32 TempBufferSize = 0x3F00;

namespace {

// EDEN_NX2_AUDIO_DUMP=<folder> also writes voices.csv: a line each time a voice starts a render tick at
// the same wave buffer position it started the tick before last, but not the tick right before
// (the X Y X Y replay heard as a 100 Hz buzz). It names the voice, its wave buffer and loop settings,
// so a replayed guest ring and a DSP state fault can be told apart.
struct VoiceReplayTrace {
    struct Entry {
        VAddr buffer{};
        u32 position{};
        bool valid{};
    };
    std::array<Entry, 3> history{};
    u32 ticks{};
    u32 events{};
};

std::FILE* OpenVoiceTrace() {
    static std::FILE* file = [] () -> std::FILE* {
        const char* folder = std::getenv("EDEN_NX2_AUDIO_DUMP");
        if (folder == nullptr || *folder == '\0') {
            return nullptr;
        }
        const std::string path = std::string(folder) + "/voices.csv";
        std::FILE* f = std::fopen(path.c_str(), "w");
        if (f != nullptr) {
            std::fputs("wall_ms,voice,channel,buffer,position,previous_position,loop,loop_count,"
                       "voice_loop_count,loop_start,loop_end,start,end,index,consumed,played,"
                       "format,ticks,event\n",
                       f);
        }
        return f;
    }();
    return file;
}

} // namespace

/// @brief Decode PCM data. Only s16 or f32 is supported.
/// @tparam T         - Type to decode. Only s16 and f32 are supported.
/// @param memory     - Core memory for reading samples.
/// @param out_buffer - Output mix buffer to receive the samples.
/// @param req        - Information for how to decode.
/// @return Number of samples decoded.
template <typename T>
static u32 DecodePcm(Core::Memory::Memory& memory, std::span<s16> out_buffer, const DecodeArg& req) {
    constexpr s32 min{(std::numeric_limits<s16>::min)()};
    constexpr s32 max{(std::numeric_limits<s16>::max)()};

    if (req.buffer == 0 || req.buffer_size == 0) {
        return 0;
    }

    if (req.start_offset >= req.end_offset) {
        return 0;
    }

    auto samples_to_decode{
        (std::min)(req.samples_to_read, req.end_offset - req.start_offset - req.offset)};
    u32 channel_count{static_cast<u32>(req.channel_count)};

    switch (req.channel_count) {
    default: {
        const VAddr source{req.buffer +
                           (((req.start_offset + req.offset) * channel_count) * sizeof(T))};
        const u64 size{channel_count * samples_to_decode};

        Core::Memory::CpuGuestMemory<T, Core::Memory::GuestMemoryFlags::UnsafeRead> samples(
            memory, source, size);
        if constexpr (std::is_floating_point_v<T>) {
            for (u32 i = 0; i < samples_to_decode; i++) {
                auto sample{static_cast<s32>(samples[i * channel_count + req.target_channel] *
                                             (std::numeric_limits<s16>::max)())};
                out_buffer[i] = static_cast<s16>(std::clamp(sample, min, max));
            }
        } else {
            for (u32 i = 0; i < samples_to_decode; i++) {
                out_buffer[i] = samples[i * channel_count + req.target_channel];
            }
        }
    } break;

    case 1:
        if (req.target_channel != 0) {
            LOG_ERROR(Service_Audio, "Invalid target channel, expected 0, got {}",
                      req.target_channel);
            return 0;
        }

        const VAddr source{req.buffer + ((req.start_offset + req.offset) * sizeof(T))};
        Core::Memory::CpuGuestMemory<T, Core::Memory::GuestMemoryFlags::UnsafeRead> samples(
            memory, source, samples_to_decode);

        if constexpr (std::is_floating_point_v<T>) {
            for (u32 i = 0; i < samples_to_decode; i++) {
                auto sample{static_cast<s32>(samples[i * channel_count + req.target_channel] *
                                             (std::numeric_limits<s16>::max)())};
                out_buffer[i] = static_cast<s16>(std::clamp(sample, min, max));
            }
        } else {
            std::memcpy(out_buffer.data(), samples.data(), samples_to_decode * sizeof(s16));
        }
        break;
    }

    return samples_to_decode;
}

/// @brief Decode ADPCM data.
/// @param memory     - Core memory for reading samples.
/// @param out_buffer - Output mix buffer to receive the samples.
/// @param req        - Information for how to decode.
/// @return Number of samples decoded.
static u32 DecodeAdpcm(Core::Memory::Memory& memory, std::span<s16> out_buffer, const DecodeArg& req) {
    constexpr u32 SamplesPerFrame{14};
    constexpr u32 NibblesPerFrame{16};

    if (req.buffer == 0 || req.buffer_size == 0) {
        return 0;
    }

    if (req.end_offset < req.start_offset) {
        return 0;
    }

    auto end{(req.end_offset % SamplesPerFrame) + NibblesPerFrame * (req.end_offset / SamplesPerFrame)};
    if (req.end_offset % SamplesPerFrame) {
        end += 3;
    } else {
        end += 1;
    }

    if (req.buffer_size < end / 2) {
        return 0;
    }

    auto start_pos{req.start_offset + req.offset};
    auto samples_to_process{(std::min)(req.end_offset - start_pos, req.samples_to_read)};
    if (samples_to_process == 0) {
        return 0;
    }

    auto samples_to_read = samples_to_process;
    auto samples_remaining_in_frame = start_pos % SamplesPerFrame;
    auto position_in_frame = (start_pos / SamplesPerFrame) * NibblesPerFrame + samples_remaining_in_frame;
    if (samples_remaining_in_frame) {
        position_in_frame += 2;
    }

    const auto size{(std::max)((samples_to_process / 8U) * SamplesPerFrame, 8U)};
    Core::Memory::CpuGuestMemory<u8, Core::Memory::GuestMemoryFlags::UnsafeRead> wavebuffer(memory, req.buffer + position_in_frame / 2, size);

    auto context = req.adpcm_context;
    auto header = context->header;
    u8 scale = u8(header & 0xfU);
    u8 coeff_index = u8((header >> 4U) & 0x7u);
    s32 coeff0 = req.coefficients[coeff_index * 2 + 0];
    s32 coeff1 = req.coefficients[coeff_index * 2 + 1];

    auto yn0 = context->yn0;
    auto yn1 = context->yn1;
    auto const get_step = [](u32 index) {
        // Emulates the following table
        // 0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
        // (a set bit in steps_sign marks the negative half, nibbles 8..15)
        constexpr u64 steps_table = 0x1234567876543210ull;
        constexpr u64 steps_sign = 0b1111111100000000ull;
        auto const r = s32((steps_table >> (index * 4)) & 0xf);
        return ((steps_sign >> index) & 1) == 0 ? r : -r;
    };
    auto const decode_sample = [&](const s32 code) -> s16 {
        const auto xn = code * (1 << scale);
        const auto prediction = coeff0 * yn0 + coeff1 * yn1;
        const auto sample = ((xn << 11) + 0x400 + prediction) >> 11;
        const auto saturated = std::clamp<s32>(sample, -0x8000, 0x7FFF);
        yn1 = yn0;
        return yn0 = s16(saturated);
    };

    u32 read_index = 0;
    for (u32 write_index = 0; samples_to_read > 0 && write_index < out_buffer.size(); ) {
        // Are we at a new frame?
        if ((position_in_frame % NibblesPerFrame) == 0) {
            header = wavebuffer[read_index++];
            scale = header & 0xFu;
            coeff_index = (header >> 4) & 0x7u;
            coeff0 = req.coefficients[coeff_index * 2 + 0];
            coeff1 = req.coefficients[coeff_index * 2 + 1];
            position_in_frame += 2;

            // Can we consume all of this frame's samples?
            if (samples_to_read >= SamplesPerFrame) {
                // Can grab all samples until the next header
                for (u32 i = 0; i < SamplesPerFrame / 2; i++) {
                    auto code0 = get_step((wavebuffer[read_index + i] >> 4) & 0xF);
                    auto code1 = get_step(wavebuffer[read_index + i] & 0xF);
                    out_buffer[write_index++] = decode_sample(code0);
                    out_buffer[write_index++] = decode_sample(code1);
                }
                read_index += SamplesPerFrame / 2;
                position_in_frame += SamplesPerFrame;
                samples_to_read -= SamplesPerFrame;
                continue;
            }
        }

        // Decode a single sample
        auto code = wavebuffer[read_index];
        if (position_in_frame & 1) {
            code &= 0xF;
            read_index++;
        } else {
            code >>= 4;
        }
        out_buffer[write_index++] = decode_sample(get_step(code));

        position_in_frame++;
        samples_to_read--;
    }

    context->header = header;
    context->yn0 = yn0;
    context->yn1 = yn1;
    return samples_to_process;
}

/// @brief Decode implementation.
/// Decode wavebuffers according to the given args.
///
/// @param memory - Core memory to read data from.
/// @param args   - The wavebuffer data, and information for how to decode it.
void DecodeFromWaveBuffers(Core::Memory::Memory& memory, const DecodeFromWaveBuffersArgs& args) {
    constexpr auto EndWaveBuffer = [](auto& voice_state, auto& wavebuffer, auto& index, auto& played_samples, auto& consumed) -> void {
        voice_state.wave_buffer_valid[index] = false;
        voice_state.loop_count = 0;
        if (wavebuffer.stream_ended) {
            played_samples = 0;
        }
        index = (index + 1) % MaxWaveBuffers;
        consumed++;
    };
    auto& voice_state{*args.voice_state};
    auto remaining_sample_count{args.sample_count};
    auto fraction{voice_state.fraction};

    std::FILE* const trace_file = OpenVoiceTrace();
    VoiceReplayTrace::Entry trace_start{};
    if (trace_file != nullptr) {
        const auto first{voice_state.wave_buffer_index % MaxWaveBuffers};
        const auto& first_buffer{args.wave_buffers[first]};
        trace_start.valid = voice_state.wave_buffer_valid[first];
        trace_start.buffer = first_buffer.buffer;
        trace_start.position = first_buffer.start_offset + voice_state.offset;
    }

    const auto sample_rate_ratio{Common::FixedPoint<49, 15>(
        (f32)args.source_sample_rate / (f32)args.target_sample_rate * (f32)args.pitch)};
    const auto size_required{fraction + remaining_sample_count * sample_rate_ratio};

    if (size_required < 0) {
        return;
    }

    // 0 -> 4, 1 -> 8, 2 -> 4
    auto pitch = u32((0x040804ul >> (u32(args.src_quality) * 8)) & 0xfful);
    if (u32(pitch + size_required.to_int_floor()) > TempBufferSize) {
        return;
    }

    auto max_remaining_sample_count{
        ((Common::FixedPoint<17, 15>(TempBufferSize) - fraction) / sample_rate_ratio)
            .to_uint_floor()};
    max_remaining_sample_count = (std::min)(max_remaining_sample_count, remaining_sample_count);

    auto wavebuffers_consumed{voice_state.wave_buffers_consumed};
    auto wavebuffer_index{voice_state.wave_buffer_index};
    auto played_sample_count{voice_state.played_sample_count};

    bool is_buffer_starved{false};
    u32 offset{voice_state.offset};

    auto output_buffer = args.output;
    std::array<s16, TempBufferSize> temp_buffer{};

    while (remaining_sample_count > 0) {
        const auto samples_to_write{(std::min)(remaining_sample_count, max_remaining_sample_count)};
        const auto samples_to_read{
            (fraction + samples_to_write * sample_rate_ratio).to_uint_floor()};

        u32 temp_buffer_pos{0};

        if (!args.IsVoicePitchAndSrcSkippedSupported) {
            for (u32 i = 0; i < pitch; i++) {
                temp_buffer[i] = voice_state.sample_history[i];
            }
            temp_buffer_pos = pitch;
        }

        u32 samples_read{0};
        while (samples_read < samples_to_read) {
            if (wavebuffer_index >= MaxWaveBuffers) {
                LOG_ERROR(Service_Audio, "Invalid wavebuffer index! {}", wavebuffer_index);
                wavebuffer_index = 0;
                wavebuffers_consumed = MaxWaveBuffers;
                voice_state.wave_buffer_valid.fill(false);
            }

            if (!voice_state.wave_buffer_valid[wavebuffer_index]) {
                is_buffer_starved = true;
                break;
            }

            auto& wavebuffer = args.wave_buffers[wavebuffer_index];
            if (offset == 0 && args.sample_format == SampleFormat::Adpcm && wavebuffer.context != 0) {
                memory.ReadBlockUnsafe(wavebuffer.context, &voice_state.adpcm_context, wavebuffer.context_size);
            }

            auto start_offset{wavebuffer.start_offset};
            auto end_offset{wavebuffer.end_offset};

            if (wavebuffer.loop && voice_state.loop_count > 0 &&
                wavebuffer.loop_start_offset <= wavebuffer.loop_end_offset) {
                start_offset = wavebuffer.loop_start_offset;
                end_offset = wavebuffer.loop_end_offset;
            }

            DecodeArg decode_arg{
                .buffer{wavebuffer.buffer},
                .buffer_size{wavebuffer.buffer_size},
                .start_offset{start_offset},
                .end_offset{end_offset},
                .channel_count{args.channel_count},
                .coefficients{},
                .adpcm_context{nullptr},
                .target_channel{args.channel},
                .offset{offset},
                .samples_to_read{samples_to_read - samples_read},
            };

            s32 samples_decoded{0};

            switch (args.sample_format) {
            case SampleFormat::PcmInt16:
                samples_decoded = DecodePcm<s16>(
                    memory, {&temp_buffer[temp_buffer_pos], TempBufferSize - temp_buffer_pos},
                    decode_arg);
                break;

            case SampleFormat::PcmFloat:
                samples_decoded = DecodePcm<f32>(
                    memory, {&temp_buffer[temp_buffer_pos], TempBufferSize - temp_buffer_pos},
                    decode_arg);
                break;

            case SampleFormat::Adpcm: {
                decode_arg.adpcm_context = &voice_state.adpcm_context;
                memory.ReadBlockUnsafe(args.data_address, &decode_arg.coefficients, args.data_size);
                samples_decoded = DecodeAdpcm( memory, {&temp_buffer[temp_buffer_pos], TempBufferSize - temp_buffer_pos}, decode_arg);
            } break;

            default:
                LOG_ERROR(Service_Audio, "Invalid sample format to decode {}",
                          static_cast<u32>(args.sample_format));
                samples_decoded = 0;
                break;
            }

            played_sample_count += samples_decoded;
            samples_read += samples_decoded;
            temp_buffer_pos += samples_decoded;
            offset += samples_decoded;

            if (samples_decoded && offset < end_offset - start_offset) {
                continue;
            }

            offset = 0;
            if (wavebuffer.loop) {
                voice_state.loop_count++;
                if (wavebuffer.loop_count >= 0 &&
                    (voice_state.loop_count > wavebuffer.loop_count || samples_decoded == 0)) {
                    EndWaveBuffer(voice_state, wavebuffer, wavebuffer_index, played_sample_count,
                                  wavebuffers_consumed);
                }

                if (samples_decoded == 0) {
                    is_buffer_starved = true;
                    break;
                }

                if (args.IsVoicePlayedSampleCountResetAtLoopPointSupported) {
                    played_sample_count = 0;
                }
            } else {
                EndWaveBuffer(voice_state, wavebuffer, wavebuffer_index, played_sample_count,
                              wavebuffers_consumed);
            }
        }

        if (args.IsVoicePitchAndSrcSkippedSupported) {
            if (samples_read > output_buffer.size()) {
                LOG_ERROR(Service_Audio, "Attempting to write past the end of output buffer!");
            }
            for (u32 i = 0; i < samples_read; i++) {
                output_buffer[i] = temp_buffer[i];
            }
        } else {
            std::memset(&temp_buffer[temp_buffer_pos], 0,
                        (samples_to_read - samples_read) * sizeof(s16));

            Resample(output_buffer, temp_buffer, sample_rate_ratio, fraction, samples_to_write,
                     args.src_quality);

            std::memcpy(voice_state.sample_history.data(), &temp_buffer[samples_to_read],
                        pitch * sizeof(s16));
        }

        remaining_sample_count -= samples_to_write;
        if (remaining_sample_count != 0 && is_buffer_starved) {
            LOG_ERROR(Service_Audio, "Samples remaining but buffer is starving??");
            break;
        }

        output_buffer = output_buffer.subspan(samples_to_write);
    }

    if (trace_file != nullptr) {
        static std::unordered_map<const void*, VoiceReplayTrace> traces;
        static const auto trace_origin = std::chrono::steady_clock::now();
        auto& trace = traces[&voice_state];
        trace.history[0] = trace.history[1];
        trace.history[1] = trace.history[2];
        trace.history[2] = trace_start;
        trace.ticks++;
        const auto& now = trace.history[2];
        const auto& last = trace.history[1];
        const auto& before = trace.history[0];
        if (trace.ticks >= 3 && now.valid && before.valid && trace.events < 400 &&
            now.buffer == before.buffer && now.position == before.position &&
            (now.buffer != last.buffer || now.position != last.position)) {
            trace.events++;
            const auto first{voice_state.wave_buffer_index % MaxWaveBuffers};
            const auto& wb{args.wave_buffers[first]};
            const double wall_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - trace_origin)
                                       .count();
            std::fprintf(trace_file,
                         "%.1f,%p,%d,%llx,%u,%u,%d,%d,%d,%u,%u,%u,%u,%u,%u,%llu,%u,%u,replay\n",
                         wall_ms, static_cast<const void*>(&voice_state),
                         static_cast<int>(args.channel),
                         static_cast<unsigned long long>(now.buffer),
                         static_cast<unsigned>(now.position),
                         static_cast<unsigned>(last.position), wb.loop ? 1 : 0,
                         static_cast<int>(wb.loop_count), static_cast<int>(voice_state.loop_count),
                         static_cast<unsigned>(wb.loop_start_offset),
                         static_cast<unsigned>(wb.loop_end_offset),
                         static_cast<unsigned>(wb.start_offset),
                         static_cast<unsigned>(wb.end_offset), static_cast<unsigned>(first),
                         static_cast<unsigned>(voice_state.wave_buffers_consumed),
                         static_cast<unsigned long long>(voice_state.played_sample_count),
                         static_cast<unsigned>(args.sample_format),
                         static_cast<unsigned>(trace.ticks));
            std::fflush(trace_file);
        }
    }

    voice_state.wave_buffers_consumed = wavebuffers_consumed;
    voice_state.played_sample_count = played_sample_count;
    voice_state.wave_buffer_index = wavebuffer_index;
    voice_state.offset = offset;
    voice_state.fraction = fraction;

    if (is_buffer_starved) {
        // A voice that ran out of wave buffers before the game flagged the end of its stream.
        const auto previous{(wavebuffer_index + MaxWaveBuffers - 1) % MaxWaveBuffers};
        Nx2Sampler::NoteVoiceStarved(!args.wave_buffers[previous].stream_ended);
    }
}

} // namespace AudioCore::Renderer
