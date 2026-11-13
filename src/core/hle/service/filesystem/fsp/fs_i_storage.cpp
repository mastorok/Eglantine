// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>

#include "core/file_sys/errors.h"
#include "core/hle/service/cmif_serialization.h"
#include "core/hle/service/filesystem/fsp/fs_i_storage.h"

namespace Service::FileSystem {

IStorage::IStorage(Core::System& system_, FileSys::VirtualFile backend_)
    : ServiceFramework{system_, "IStorage"}, backend(std::move(backend_)) {
    static const FunctionInfo functions[] = {
        {0, D<&IStorage::Read>, "Read"},
        {1, nullptr, "Write"},
        {2, nullptr, "Flush"},
        {3, nullptr, "SetSize"},
        {4, D<&IStorage::GetSize>, "GetSize"},
        {5, nullptr, "OperateRange"},
    };
    RegisterHandlers(functions);
}

Result IStorage::Read(
    OutBuffer<BufferAttr_HipcMapAlias | BufferAttr_HipcMapTransferAllowsNonSecure> out_bytes,
    s64 offset, s64 length) {
    LOG_DEBUG(Service_FS, "called, offset={:#x}, length={}", offset, length);

    R_UNLESS(length >= 0, FileSys::ResultInvalidSize);
    R_UNLESS(offset >= 0, FileSys::ResultInvalidOffset);

    // EDEN_NX2_READ_THROTTLE=0 (boot): no pacing of game file reads. Upstream added the pacing for
    // video freezes in other games (Eden #4525), but it holds every read back to one per 1.6 ms or
    // more on the one file system thread: area loads and streaming wait for it (Fritchii measured
    // 36-151 ms of sleep at boot and area loads in Ocarina of Time). Default unchanged.
    static const bool throttle = [] {
        const char* const value = std::getenv("EDEN_NX2_READ_THROTTLE");
        return value == nullptr || value[0] != '0';
    }();
    const auto now = std::chrono::steady_clock::now();
    if (throttle && now < next_read_tick) {
        std::this_thread::sleep_for(next_read_tick - now);
    }
    next_read_tick = std::chrono::steady_clock::now() +
                     std::chrono::nanoseconds{std::min(length, MaxChargedReadSize) *
                                              1'000'000'000 / ReadBytesPerSecond};

    // Read the data from the Storage backend
    backend->Read(out_bytes.data(), length, offset);

    R_SUCCEED();
}

Result IStorage::GetSize(Out<u64> out_size) {
    *out_size = backend->GetSize();

    LOG_DEBUG(Service_FS, "called, size={}", *out_size);

    R_SUCCEED();
}

} // namespace Service::FileSystem
