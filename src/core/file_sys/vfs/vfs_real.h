// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <map>
#include <mutex>
#include <optional>
#include <string_view>
#include "common/intrusive_list.h"
#include "core/file_sys/fs_filesystem.h"
#include "core/file_sys/vfs/vfs.h"

namespace Common::FS {
class IOFile;
}

namespace FileSys {

struct FileReference : public Common::IntrusiveListBaseNode<FileReference> {
    std::string path;
    std::optional<u64> size;
    std::shared_ptr<Common::FS::IOFile> file{};
    bool detached{};
};

class RealVfsFile;
class RealVfsDirectory;

class RealVfsFilesystem : public VfsFilesystem {
public:
    RealVfsFilesystem();
    ~RealVfsFilesystem() override;

    std::string GetName() const override;
    bool IsReadable() const override;
    bool IsWritable() const override;
    VfsEntryType GetEntryType(std::string_view path) const override;
    VirtualFile OpenFile(std::string_view path, OpenMode perms = OpenMode::Read) override;
    VirtualFile CreateFile(std::string_view path, OpenMode perms = OpenMode::ReadWrite) override;
    VirtualFile CopyFile(std::string_view old_path, std::string_view new_path) override;
    VirtualFile MoveFile(std::string_view old_path, std::string_view new_path) override;
    bool DeleteFile(std::string_view path) override;
    VirtualDir OpenDirectory(std::string_view path, OpenMode perms = OpenMode::Read) override;
    VirtualDir CreateDirectory(std::string_view path,
                               OpenMode perms = OpenMode::ReadWrite) override;
    VirtualDir CopyDirectory(std::string_view old_path, std::string_view new_path) override;
    VirtualDir MoveDirectory(std::string_view old_path, std::string_view new_path) override;
    bool DeleteDirectory(std::string_view path) override;

private:
    using ReferenceListType = Common::IntrusiveListBaseTraits<FileReference>::ListType;
    struct CacheEntry {
        std::weak_ptr<VfsFile> file;
        FileReference* reference;
    };
    using FileCache = std::map<std::string, CacheEntry, std::less<>>;
    FileCache cache;
    ReferenceListType open_references;
    ReferenceListType closed_references;
    std::mutex list_lock;
    size_t num_open_files{};
    // TODO: Workaround for improper dtor() ordering on clang + FreeBSD
    bool in_dtor = false;

private:
    friend class RealVfsFile;
    void RefreshReferenceLocked(FileReference& reference, OpenMode perms);
    void DropReference(FileReference& reference);

private:
    friend class RealVfsDirectory;
    VirtualFile OpenFileFromEntry(std::string_view path, std::optional<u64> size,
                                  OpenMode perms = OpenMode::Read);

private:
    VirtualFile OpenFileLocked(const std::string& path, std::optional<u64> size, OpenMode perms);
    bool RenameFileLocked(const std::string& old_path, const std::string& new_path);
    Result DeleteFileWithResult(std::string_view path);
    void DetachReferenceLocked(FileCache::iterator entry);
    void CloseReferenceLocked(FileReference& reference);
    void EvictSingleReferenceLocked();
    void InsertReferenceIntoListLocked(FileReference& reference);
    void RemoveReferenceFromListLocked(FileReference& reference);
};

// An implementation of VfsFile that represents a file on the user's computer.
class RealVfsFile : public VfsFile {
    friend class RealVfsDirectory;
    friend class RealVfsFilesystem;

public:
    RealVfsFile(RealVfsFilesystem& base, std::unique_ptr<FileReference> reference, OpenMode perms);
    ~RealVfsFile() override;

    std::string GetName() const override;
    std::size_t GetSize() const override;
    bool Resize(std::size_t new_size) override;
    VirtualDir GetContainingDirectory() const override;
    bool IsWritable() const override;
    bool IsReadable() const override;
    std::size_t Read(u8* data, std::size_t length, std::size_t offset) const override;
    std::size_t Write(const u8* data, std::size_t length, std::size_t offset) override;
    bool Rename(std::string_view name) override;

private:

    RealVfsFilesystem& base;
    std::unique_ptr<FileReference> reference;
    OpenMode perms;
};

// An implementation of VfsDirectory that represents a directory on the user's computer.
class RealVfsDirectory : public VfsDirectory {
    friend class RealVfsFilesystem;

public:
    RealVfsDirectory(RealVfsFilesystem& base, const std::string& path,
                     OpenMode perms = OpenMode::Read);
    ~RealVfsDirectory() override;

    VirtualFile GetFileRelative(std::string_view relative_path) const override;
    VirtualDir GetDirectoryRelative(std::string_view relative_path) const override;
    VirtualFile GetFile(std::string_view name) const override;
    VirtualDir GetSubdirectory(std::string_view name) const override;
    VirtualFile CreateFileRelative(std::string_view relative_path) override;
    VirtualDir CreateDirectoryRelative(std::string_view relative_path) override;
    bool DeleteSubdirectoryRecursive(std::string_view name) override;
    std::vector<VirtualFile> GetFiles() const override;
    FileTimeStampRaw GetFileTimeStamp(std::string_view path) const override;
    std::vector<VirtualDir> GetSubdirectories() const override;
    bool IsWritable() const override;
    bool IsReadable() const override;
    std::string GetName() const override;
    VirtualDir GetParentDirectory() const override;
    VirtualDir CreateSubdirectory(std::string_view name) override;
    VirtualFile CreateFile(std::string_view name) override;
    bool DeleteSubdirectory(std::string_view name) override;
    bool DeleteFile(std::string_view name) override;
    Result DeleteFileWithResult(std::string_view name) override;
    bool Rename(std::string_view name) override;
    std::string GetFullPath() const override;
    std::map<std::string, VfsEntryType, std::less<>> GetEntries() const override;

private:
    template <typename T, typename R>
    std::vector<std::shared_ptr<R>> IterateEntries() const;

    RealVfsFilesystem& base;
    std::string path;
    std::string parent_path;
    std::vector<std::string> path_components;
    OpenMode perms;
};

} // namespace FileSys
