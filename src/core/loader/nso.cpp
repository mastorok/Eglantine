// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "common/common_funcs.h"
#include "common/game_presets.h"
#include "common/hex_util.h"
#include "common/logging.h"
#include "common/lz4_compression.h"
#include "common/zbic_compression.h"
#include "common/settings.h"
#include "common/swap.h"
#include "core/core.h"
#include "core/file_sys/patch_manager.h"
#include "core/hle/kernel/code_set.h"
#include "core/hle/kernel/k_page_table.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/loader/nso.h"
#include "core/memory.h"

#ifdef HAS_NCE
#include "core/arm/nce/patcher.h"
#endif

namespace Loader {
namespace {
struct MODHeader {
    u32_le magic;
    u32_le dynamic_offset;
    u32_le bss_start_offset;
    u32_le bss_end_offset;
    u32_le eh_frame_hdr_start_offset;
    u32_le eh_frame_hdr_end_offset;
    u32_le module_offset; // Offset to runtime-generated module object. typically equal to .bss base
};
static_assert(sizeof(MODHeader) == 0x1c, "MODHeader has incorrect size.");

constexpr u32 PageAlignSize(u32 size) {
    return static_cast<u32>((size + Core::Memory::YUZU_PAGEMASK) & ~Core::Memory::YUZU_PAGEMASK);
}
} // Anonymous namespace

bool NSOHeader::IsSegmentCompressed(size_t segment_num) const {
    ASSERT_MSG(segment_num < 3, "Invalid segment {}", segment_num);
    return ((flags >> segment_num) & 1) != 0;
}

AppLoader_NSO::AppLoader_NSO(FileSys::VirtualFile file_) : AppLoader(std::move(file_)) {}

FileType AppLoader_NSO::IdentifyType(const FileSys::VirtualFile& in_file) {
    u32 magic = 0;
    if (in_file->ReadObject(&magic) != sizeof(magic)) {
        return FileType::Error;
    }

    if (Common::MakeMagic('N', 'S', 'O', '0') != magic) {
        return FileType::Error;
    }

    return FileType::NSO;
}

std::optional<VAddr> AppLoader_NSO::LoadModule(Kernel::KProcess& process, Core::System& system, const FileSys::VfsFile& nso_file, VAddr load_base, bool should_pass_arguments, bool load_into_process, std::optional<FileSys::PatchManager> pm, std::vector<Core::NCE::Patcher>* patches, s32 patch_index) {
    if (nso_file.GetSize() < sizeof(NSOHeader))
        return std::nullopt;
    NSOHeader nso_header{};
    if (sizeof(NSOHeader) != nso_file.ReadObject(&nso_header))
        return std::nullopt;
    if (nso_header.magic != Common::MakeMagic('N', 'S', 'O', '0'))
        return std::nullopt;
    if (nso_header.segments.empty())
        return std::nullopt;

    // Allocate some space at the beginning if we are patching in PreText mode.
    const size_t module_start = [&]() -> size_t {
#ifdef HAS_NCE
        if (patches && load_into_process) {
            auto* patch = &patches->operator[](patch_index);
            if (patch->GetPatchMode() == Core::NCE::PatchMode::PreText) {
                return patch->GetSectionSize();
            } else if (patch->GetPatchMode() == Core::NCE::PatchMode::Split) {
                return patch->GetPreSectionSize();
            }
        }
#endif
        return 0;
    }();

    auto const last_segment_it = &nso_header.segments[nso_header.segments.size() - 1];
    // Build program image directly in codeset memory :)
    Kernel::CodeSet codeset;
    codeset.memory.resize(module_start + last_segment_it->location + last_segment_it->size);
    {
        std::vector<u8> compressed_data(*std::ranges::max_element(nso_header.segments_compressed_size));
        std::vector<u8> decompressed_size(std::ranges::max_element(nso_header.segments, [](auto const& a, auto const& b) {
            return a.size < b.size;
        })->size);
        for (std::size_t i = 0; i < nso_header.segments.size(); ++i) {
            nso_file.Read(compressed_data.data(), nso_header.segments_compressed_size[i], nso_header.segments[i].offset);
            if (nso_header.IsSegmentCompressed(i)) {
                if (nso_header.IsZBICCompressed()) {
                    // ZBIC compression
                    const int r = Common::Compression::DecompressDataZBIC(
                        std::span<u8>{decompressed_size}.first(nso_header.segments[i].size),
                        std::span<const u8>{compressed_data}.first(nso_header.segments_compressed_size[i])
                    );
                    ASSERT(r > 0);
                } else {
                    // LZ4 compression
                    int r = Common::Compression::DecompressDataLZ4(
                        decompressed_size.data(),
                        nso_header.segments[i].size,
                        compressed_data.data(),
                        nso_header.segments_compressed_size[i]
                    );
                    ASSERT(r == int(nso_header.segments[i].size));
                }

                std::memcpy(
                    codeset.memory.data() + module_start + nso_header.segments[i].location,
                    decompressed_size.data(),
                    nso_header.segments[i].size
                );
            } else {
                // Not compressed
                std::memcpy(
                    codeset.memory.data() + module_start + nso_header.segments[i].location,
                    compressed_data.data(),
                    nso_header.segments[i].size
                );
            }
            codeset.segments[i].addr = module_start + nso_header.segments[i].location;
            codeset.segments[i].offset = module_start + nso_header.segments[i].location;
            codeset.segments[i].size = nso_header.segments[i].size;
        }
    }

    if (should_pass_arguments && !Settings::values.program_args.GetValue().empty()) {
        const auto arg_data{Settings::values.program_args.GetValue()};

        codeset.DataSegment().size += NSO_ARGUMENT_DATA_ALLOCATION_SIZE;
        NSOArgumentHeader args_header{NSO_ARGUMENT_DATA_ALLOCATION_SIZE, static_cast<u32_le>(arg_data.size()), {}};
        const auto end_offset = codeset.memory.size();
        codeset.memory.resize(u32(codeset.memory.size()) + NSO_ARGUMENT_DATA_ALLOCATION_SIZE);
        std::memcpy(codeset.memory.data() + end_offset, &args_header, sizeof(NSOArgumentHeader));
        std::memcpy(codeset.memory.data() + end_offset + sizeof(NSOArgumentHeader), arg_data.data(), arg_data.size());
    }

    codeset.DataSegment().size += nso_header.segments[2].bss_size;
    u32 image_size = PageAlignSize(u32(codeset.memory.size()) + nso_header.segments[2].bss_size);
    codeset.memory.resize(image_size);

    for (std::size_t i = 0; i < nso_header.segments.size(); ++i) {
        codeset.segments[i].size = PageAlignSize(codeset.segments[i].size);
    }

    // Apply patches if necessary
    const auto name = nso_file.GetName();
    if (pm && (pm->HasNSOPatch(nso_header.build_id, name) || Settings::values.dump_nso)) {
        std::span<u8> patchable_section(codeset.memory.data() + module_start, codeset.memory.size() - module_start);
        std::vector<u8> pi_header(sizeof(NSOHeader) + patchable_section.size());
        std::memcpy(pi_header.data(), &nso_header, sizeof(NSOHeader));
        std::memcpy(pi_header.data() + sizeof(NSOHeader), patchable_section.data(),
                    patchable_section.size());

        pi_header = pm->PatchNSO(pi_header, name);

        std::copy(pi_header.begin() + sizeof(NSOHeader), pi_header.end(), patchable_section.data());
    }

    // EDEN_NX2_DUMP_MODULES=<dir>: write each module's image as loaded (text, rodata, data) to
    // <dir>/<program id>-<name>.bin, to study a game's code offline.
    if (const char* const dump_dir = std::getenv("EDEN_NX2_DUMP_MODULES");
        dump_dir != nullptr && dump_dir[0] != 0) {
        const std::string dump_path{
            fmt::format("{}/{:016X}-{}.bin", dump_dir, process.GetProgramId(), name)};
        const size_t dump_size{codeset.memory.size() - module_start};
        std::ofstream dump_file{dump_path, std::ios::binary};
        dump_file.write(reinterpret_cast<const char*>(codeset.memory.data() + module_start),
                        static_cast<std::streamsize>(dump_size));
        if (dump_file) {
            LOG_INFO(Loader,
                     "Module image written to {} ({} bytes; text {:#x}+{:#x}, rodata {:#x}+{:#x}, "
                     "data {:#x}+{:#x})",
                     dump_path, dump_size, static_cast<u32>(nso_header.segments[0].location),
                     codeset.segments[0].size, static_cast<u32>(nso_header.segments[1].location),
                     codeset.segments[1].size, static_cast<u32>(nso_header.segments[2].location),
                     codeset.segments[2].size);
        } else {
            LOG_WARNING(Loader, "Cannot write the module image to {}", dump_path);
        }
    }

    // Star Fox output pacing for the tested build. The engine's explicit-rate path
    // updates its render delta as well; simulation timing and the guest clock stay unchanged.
    if (name == "main" && process.GetProgramId() == 0x0400E88025B1A000ULL) {
        constexpr u32 Fps = 60;
        constexpr size_t Offset = 0xB444ADC;
        constexpr std::array<u32, 3> Original{0xBD4223E1, 0x1E202028, 0x540000ED};
        constexpr std::array<u8, 20> BuildId{
            0xA7, 0x73, 0x0D, 0x0F, 0x74, 0xD1, 0x68, 0xD9, 0x04, 0x22,
            0x35, 0x4C, 0x83, 0x17, 0xF7, 0x6F, 0xF7, 0x8A, 0xD9, 0x38};
        const auto& text = nso_header.segments[0];
        if (std::equal(BuildId.begin(), BuildId.end(), nso_header.build_id.begin()) &&
            Offset >= text.location && Offset + sizeof(Original) <= u64{text.location} + text.size &&
            module_start + Offset + sizeof(Original) <= codeset.memory.size() &&
            std::memcmp(codeset.memory.data() + module_start + Offset, Original.data(),
                        sizeof(Original)) == 0) {
            constexpr std::array<u32, 3> replacement{
                0x52800019U | (Fps << 5), // mov w25, #60
                0x1E2E1000,              // fmov s0, #1.0
                0x14000035};             // b main+0xB444BB8 (explicit output-rate path)
            std::memcpy(codeset.memory.data() + module_start + Offset, replacement.data(),
                        sizeof(replacement));
            LOG_INFO(Loader, "Star Fox output interval: {} FPS; guest clock unchanged", Fps);
        } else {
            LOG_WARNING(Loader,
                        "Star Fox output patch skipped: unsupported build or instructions");
        }
    }

    // Mario Kart World selects a 1- or 2-vblank frame mode in its main loop. Force its native
    // 30 FPS mode, or let it choose dynamically using the whole frame time (CPU + GPU wait).
    // Match the instruction sequence instead of relying on a title-version-specific address.
    const bool mkw_dynamic = Settings::values.mkw_dynamic_fps.GetValue();
    bool mkw_dynamic_fps_patch_applied = false;
    if ((mkw_dynamic || Settings::values.mkw_30fps.GetValue()) && name == "main" &&
        process.GetProgramId() == Settings::MarioKartWorldProgramId) {
        constexpr u32 ModeLoadInsn = 0xB9421268; // ldr w8, [x19, #0x210]
        const u32 mode = mkw_dynamic ? 0U : 2U;
        const u32 mode_instruction = 0x52800008U | (mode << 5); // mov w8, #mode
        const auto& text = nso_header.segments[0];
        u8* const code = codeset.memory.data() + module_start + text.location;
        const size_t count = text.size / sizeof(u32);
        const auto word = [code](size_t index) {
            u32 value;
            std::memcpy(&value, code + index * sizeof(u32), sizeof(value));
            return value;
        };
        size_t mode_site = 0;
        size_t mode_matches = 0;
        for (size_t i = 2; i + 8 < count; ++i) {
            if (word(i) != ModeLoadInsn || word(i - 2) != 0x6B01013F || // cmp w9, w1
                (word(i - 1) & 0xFF00001F) != 0x54000001 ||             // b.ne
                (word(i + 1) & 0xFF00001F) != 0x35000008) {             // cbnz w8
                continue;
            }
            for (size_t j = i + 2; j < i + 8; ++j) {
                if (word(j) == 0x52A84E08) { // mov w8, #60.0f
                    mode_site = i;
                    ++mode_matches;
                    break;
                }
            }
        }

        size_t frame_time_site = 0;
        size_t frame_time_matches = 0;
        if (mode_matches == 1 && mkw_dynamic) {
            for (size_t i = mode_site + 1; i < mode_site + 32 && i < count; ++i) {
                if ((word(i) & 0xFFFFFFE0) == 0xF9409660) { // ldr xN, [x19, #0x128]
                    frame_time_site = i;
                    ++frame_time_matches;
                }
            }
        }

        if (mode_matches == 1 && (!mkw_dynamic || frame_time_matches == 1)) {
            std::memcpy(code + mode_site * sizeof(u32), &mode_instruction,
                        sizeof(mode_instruction));
            if (mkw_dynamic) {
                const u32 whole_frame_time = 0xF9407660 | (word(frame_time_site) & 0x1F);
                std::memcpy(code + frame_time_site * sizeof(u32), &whole_frame_time,
                            sizeof(whole_frame_time));
                mkw_dynamic_fps_patch_applied = true;
            }
            LOG_INFO(Loader, "Mario Kart World: {} FPS mode enabled (main+{:#x})",
                     mkw_dynamic ? "dynamic" : "30",
                     text.location + mode_site * sizeof(u32));
        } else {
            LOG_WARNING(Loader,
                        "Mario Kart World FPS patch skipped (mode matches {}, frame-time matches {})",
                        mode_matches, frame_time_matches);
        }
    }

    // Ocarina of Time (Switch 2): real-time frame delta.
    // The game derives its frame delta from an integer "frames per second" its own frame
    // governor picks (60 / interval): delta = 60 / fps, in 60 Hz frames. The governor only sees
    // the time its own thread spends, not the time Eden's present path holds the game, so it
    // keeps choosing 60 fps while frames really take 30-50 ms and the whole game runs in slow
    // motion. The two places that turn that integer into a delta (the main frame update and the
    // frame-begin virtual that publishes the global delta) are pointed at a float cell in the
    // text segment's page padding that the present path of nx2:vi2 refreshes with the measured
    // frame rate (nx2_gsv.h, UpdateOotDynamicDt). EDEN_NX2_OOT_DYNAMIC_DT=0 leaves the code alone.
    size_t oot_dt_mailbox_offset = 0;
    if (name == "main" && process.GetProgramId() == 0x0400486024DEC000ULL) {
        const char* const oot_dt_env = std::getenv("EDEN_NX2_OOT_DYNAMIC_DT");
        constexpr std::array<u8, 20> BuildId{0xE0, 0xF3, 0xCC, 0xFA, 0xF5, 0x00, 0x65,
                                             0xAE, 0x4E, 0xA0, 0x1E, 0xB5, 0xAE, 0xB1,
                                             0x3B, 0x1A, 0x45, 0xDA, 0x33, 0x48};
        // Sites: scvtf s0, s0 (fps integer to float) and the instruction that follows it.
        constexpr u32 ScvtfInsn = 0x5E21D800;
        struct Site {
            size_t offset;
            u32 next;
        };
        constexpr std::array<Site, 2> sites{{{0x3B4EAF4, 0x1E201820},   // fdiv s0, s1, s0
                                             {0x6E6CA8, 0x7200011F}}}; // tst w8, #1
        // Cave: ldr s0, [pc, #8]; ret; .float fps; nop. Needs 16 zero bytes in the page padding
        // after the text segment, and the fps float starts at 60 (the unpatched behaviour).
        constexpr size_t CaveOffset = 0x3C46200;
        constexpr std::array<u32, 4> cave{0x1C000040U, 0xD65F03C0U, 0x42700000U, 0xD503201FU};
        const auto& text = nso_header.segments[0];
        const auto& rodata = nso_header.segments[1];
        u8* const image = codeset.memory.data() + module_start;
        const auto read_word = [image](size_t offset) {
            u32 value;
            std::memcpy(&value, image + offset, sizeof(value));
            return value;
        };
        bool match = (oot_dt_env == nullptr || oot_dt_env[0] != '0') &&
                     std::equal(BuildId.begin(), BuildId.end(), nso_header.build_id.begin()) &&
                     CaveOffset >= u64{text.location} + text.size &&
                     CaveOffset + sizeof(cave) <= u64{rodata.location} &&
                     module_start + CaveOffset + sizeof(cave) <= codeset.memory.size();
        for (size_t i = 0; match && i < sizeof(cave) / sizeof(u32); ++i) {
            match = read_word(CaveOffset + i * sizeof(u32)) == 0;
        }
        for (const Site& site : sites) {
            match = match && site.offset + 2 * sizeof(u32) <= text.size &&
                    module_start + site.offset + 2 * sizeof(u32) <= codeset.memory.size() &&
                    read_word(site.offset) == ScvtfInsn && read_word(site.offset + 4) == site.next;
        }
        if (match) {
            std::memcpy(image + CaveOffset, cave.data(), sizeof(cave));
            for (const Site& site : sites) {
                const s64 words = (static_cast<s64>(CaveOffset) - static_cast<s64>(site.offset)) / 4;
                const u32 call = 0x94000000U | (static_cast<u32>(words) & 0x03FFFFFFU); // bl cave
                std::memcpy(image + site.offset, &call, sizeof(call));
            }
            oot_dt_mailbox_offset = CaveOffset + 2 * sizeof(u32);
            LOG_INFO(Loader, "Ocarina of Time: real-time frame delta patch applied (main+{:#x}, "
                             "main+{:#x}; fps cell at main+{:#x})",
                     sites[0].offset, sites[1].offset, oot_dt_mailbox_offset);
        } else if (oot_dt_env == nullptr || oot_dt_env[0] != '0') {
            LOG_WARNING(Loader, "Ocarina of Time real-time frame delta patch skipped "
                                "(unsupported build or instructions)");
        } else {
            LOG_INFO(Loader, "Ocarina of Time real-time frame delta patch disabled "
                             "(EDEN_NX2_OOT_DYNAMIC_DT=0)");
        }
    }

#ifdef HAS_NCE
    // If we are computing the process code layout and using nce backend, patch.
    const auto& code = codeset.CodeSegment();
    auto* patch = patches ? &patches->operator[](patch_index) : nullptr;
    if (patch && !load_into_process) {
        //Set module ID using build_id from the NSO header
        patch->SetModuleID(nso_header.build_id);
        // Patch SVCs and MRS calls in the guest code
        while (!patch->PatchText(codeset.memory, code)) {
            patch = &patches->emplace_back();
            patch->SetModuleID(nso_header.build_id);  // In case the patcher is changed for big modules, the new patcher should also have the build_id
        }
    } else if (patch) {
        // Relocate code patch and copy to the program image.
        // Save size before RelocateAndCopy (which may resize)
        const size_t size_before_relocate = codeset.memory.size();
        if (patch->RelocateAndCopy(load_base, code, codeset.memory, &process.GetPostHandlers())) {
            // Update patch section.
            auto& patch_segment = codeset.PatchSegment();
            auto& post_patch_segment = codeset.PostPatchSegment();
            const auto patch_mode = patch->GetPatchMode();
            if (patch_mode == Core::NCE::PatchMode::PreText) {
                patch_segment.addr = 0;
                patch_segment.size = static_cast<u32>(patch->GetSectionSize());
            } else if (patch_mode == Core::NCE::PatchMode::Split) {
                // For Split-mode, we are using pre-patch buffer at start, post-patch buffer at end
                patch_segment.addr = 0;
                patch_segment.size = static_cast<u32>(patch->GetPreSectionSize());
                post_patch_segment.addr = size_before_relocate;
                post_patch_segment.size = static_cast<u32>(patch->GetSectionSize());
            } else {
                patch_segment.addr = image_size;
                patch_segment.size = static_cast<u32>(patch->GetSectionSize());
            }
        }

        // Refresh image_size to take account the patch section if it was added by RelocateAndCopy
        image_size = static_cast<u32>(codeset.memory.size());
    }
#endif

    // If we aren't actually loading (i.e. just computing the process code layout), we are done
    if (!load_into_process) {
#ifdef HAS_NCE
        // Ok, so for Split mode, we need to account for pre-patch and post-patch space
        // which will be added during RelocateAndCopy in the second pass. Where it crashed
        // in Android Studio at PreText. May be a better way. Works for now.
        if (patch && patch->GetPatchMode() == Core::NCE::PatchMode::Split) {
            return load_base + patch->GetPreSectionSize() + image_size + patch->GetSectionSize();
        } else if (patch && patch->GetPatchMode() == Core::NCE::PatchMode::PreText) {
            return load_base + patch->GetSectionSize() + image_size;
        } else if (patch && patch->GetPatchMode() == Core::NCE::PatchMode::PostData) {
            return load_base + image_size + patch->GetSectionSize();
        }
#endif
        return load_base + image_size;
    }

    // Apply cheats if they exist and the program has a valid title ID
    if (pm) {
        // TODO(Maufeat): Check if there is a better way to check
        if (name == "main")
            system.SetApplicationProcessBuildID(nso_header.build_id);

        const auto cheats = pm->CreateCheatList(nso_header.build_id);
        if (!cheats.empty()) {
            system.RegisterCheatList(cheats, nso_header.build_id, load_base, image_size);
        }
    }

    // Metroid Prime 4 (Switch 1 build): the game advances by a fixed frame time it picks from a
    // {1/60, 1/30} table in rodata, so it runs in slow motion whenever a frame takes longer. The
    // GPU front end rewrites that table with the measured frame time (video_core/gpu.cpp).
    // EDEN_NX2_MP4_DYNAMIC_DT=0 leaves the table alone.
    size_t mp4_frame_time_table_offset = 0;
    if (name == "main" && process.GetProgramId() == 0x010019A01E2F2000ULL) {
        constexpr size_t TableOffset = 0x13bf400;
        constexpr std::array<u32, 2> Expected{0x3c888889, 0x3d088889};
        const char* const mp4_env = std::getenv("EDEN_NX2_MP4_DYNAMIC_DT");
        std::array<u32, 2> found{};
        if (module_start + TableOffset + sizeof(found) <= codeset.memory.size()) {
            std::memcpy(found.data(), codeset.memory.data() + module_start + TableOffset,
                        sizeof(found));
        }
        if (mp4_env != nullptr && mp4_env[0] == '0') {
            LOG_INFO(Loader, "Metroid Prime 4: real-time frame time disabled by the environment");
        } else if (found == Expected) {
            mp4_frame_time_table_offset = TableOffset;
            LOG_INFO(Loader, "Metroid Prime 4: frame-time table found at main+{:#x}", TableOffset);
        } else {
            LOG_WARNING(Loader,
                        "Metroid Prime 4: frame-time table not found at main+{:#x} ({:#x}, {:#x}): "
                        "another build of the game",
                        TableOffset, found[0], found[1]);
        }
    }

    // Load codeset for current process
    LOG_INFO(Loader, "Module {} loaded at {:#x} (size {:#x})", nso_file.GetName(), load_base, image_size);
    process.LoadModule(system.Kernel(), std::move(codeset), load_base);
    if (name == "main" && process.GetProgramId() == Settings::MarioKartWorldProgramId) {
        process.SetMkwDynamicFpsPatchApplied(mkw_dynamic_fps_patch_applied);
    }
    if (oot_dt_mailbox_offset != 0) {
        process.SetOotDynamicDtMailbox(load_base + module_start + oot_dt_mailbox_offset);
    }
    if (mp4_frame_time_table_offset != 0) {
        process.SetFrameTimeTable(load_base + module_start + mp4_frame_time_table_offset);
    }
    return load_base + image_size;
}

AppLoader_NSO::LoadResult AppLoader_NSO::Load(Kernel::KProcess& process, Core::System& system) {
    if (is_loaded) {
        return {ResultStatus::ErrorAlreadyLoaded, {}};
    }

    modules.clear();

    // Load module
    const VAddr base_address = GetInteger(process.GetEntryPoint());
    if (!LoadModule(process, system, *file, base_address, true, true)) {
        return {ResultStatus::ErrorLoadingNSO, {}};
    }

    modules.insert_or_assign(base_address, file->GetName());
    LOG_DEBUG(Loader, "loaded module {} @ {:#x}", file->GetName(), base_address);

    is_loaded = true;
    return {ResultStatus::Success, LoadParameters{Kernel::KThread::DefaultThreadPriority,
                                                  Core::Memory::DEFAULT_STACK_SIZE}};
}

ResultStatus AppLoader_NSO::ReadNSOModules(Modules& out_modules) {
    out_modules = this->modules;
    return ResultStatus::Success;
}

} // namespace Loader
