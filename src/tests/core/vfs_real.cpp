// SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "common/fs/file.h"
#include "common/fs/path_util.h"
#include "core/file_sys/errors.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "tests/common/temp_dir.h"

TEST_CASE("Real VFS can replace a file while its old reference is live", "[core][vfs]") {
    const Tests::TempDir temp{"lichii-vfs"};
    const auto path = Common::FS::PathToUTF8String(temp / "trial.ghost");
    FileSys::RealVfsFilesystem fs;
    const auto ghost = fs.CreateFile(path);
    REQUIRE(ghost);
    REQUIRE(ghost->WriteByte(0x42));
    REQUIRE(ghost->ReadByte(0) == 0x42);

    REQUIRE(fs.DeleteFile(path));
    REQUIRE_FALSE(fs.OpenFile(path));
    const auto replacement = fs.CreateFile(path);
    REQUIRE(replacement);
    REQUIRE(replacement->WriteByte(0x24));
    CHECK(replacement->ReadByte(0) == 0x24);
#ifdef _WIN32
    CHECK_FALSE(ghost->ReadByte(0));
    CHECK_FALSE(ghost->WriteByte(0x55));
#else
    CHECK(ghost->ReadByte(0) == 0x42);
    REQUIRE(ghost->WriteByte(0x55));
#endif
    CHECK(replacement->ReadByte(0) == 0x24);
}

TEST_CASE("Real VFS can rename a file while its reference is live", "[core][vfs]") {
    const Tests::TempDir temp{"lichii-vfs"};
    const auto old_path = Common::FS::PathToUTF8String(temp / "old.ghost");
    const auto new_path = Common::FS::PathToUTF8String(temp / "new.ghost");
    FileSys::RealVfsFilesystem fs;
    const auto ghost = fs.CreateFile(old_path);
    REQUIRE(ghost);
    REQUIRE(ghost->WriteByte(0x42));

    {
        const auto other = fs.CreateFile(new_path);
        REQUIRE(other);
        REQUIRE(other->WriteByte(0x24));
        CHECK_FALSE(ghost->Rename("new.ghost"));
        CHECK(ghost->ReadByte(0) == 0x42);
        CHECK(other->ReadByte(0) == 0x24);
        REQUIRE(fs.DeleteFile(new_path));
    }
    REQUIRE(ghost->Rename("new.ghost"));
    REQUIRE_FALSE(fs.OpenFile(old_path));
    const auto renamed = fs.OpenFile(new_path);
    REQUIRE(renamed);
    CHECK(renamed == ghost);
    CHECK(renamed->GetSize() == 1);
    CHECK(renamed->ReadAllBytes() == std::vector<u8>{0x42});
    const auto replacement = fs.CreateFile(old_path);
    REQUIRE(replacement);
    REQUIRE(replacement->Resize(2));
    CHECK(renamed->GetSize() == 1);
    CHECK(renamed->ReadByte(0) == 0x42);
}

#ifdef _WIN32
TEST_CASE("Real VFS reports a genuinely locked delete", "[core][vfs]") {
    const Tests::TempDir temp{"lichii-vfs"};
    const auto path = temp / "trial.ghost";
    FileSys::RealVfsFilesystem fs;
    const auto dir = fs.OpenDirectory(Common::FS::PathToUTF8String(temp.Path()));
    {
        const auto seed = fs.CreateFile(Common::FS::PathToUTF8String(path));
        REQUIRE(seed);
        REQUIRE(seed->WriteByte('a'));
    }
    const auto ghost = dir->GetFile("trial.ghost");
    REQUIRE(ghost);
    REQUIRE(ghost->ReadByte(0) == 'a');
    Common::FS::IOFile external{path, Common::FS::FileAccessMode::Read};
    REQUIRE(external.IsOpen());

    CHECK(dir->DeleteFileWithResult("trial.ghost") == FileSys::ResultTargetLocked);
    CHECK(std::filesystem::exists(path));
    CHECK(dir->GetFile("trial.ghost") == ghost);
    CHECK(ghost->ReadByte(0) == 'a');
    external.Close();
    CHECK(dir->DeleteFileWithResult("trial.ghost") == ResultSuccess);
    CHECK_FALSE(std::filesystem::exists(path));
}
#endif
