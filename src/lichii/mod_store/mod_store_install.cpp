// SPDX-FileCopyrightText: Copyright 2026 Lichii Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "lichii/mod_store/mod_store_install.h"

#include <algorithm>
#include <array>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>

#include <QBuffer>
#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <QScopeGuard>
#include <QSet>
#include <QStringList>
#include <QTemporaryDir>
#include <QTemporaryFile>
#include <QtEndian>
#include <quazip.h>
#include <quazipfile.h>
#include <zlib.h>

namespace ModStore {
namespace {

using namespace Qt::StringLiterals;

constexpr qint64 MaxArchiveBytes = 64 * 1024 * 1024;
constexpr qint64 MaxPatchBytes = 2 * 1024 * 1024;
constexpr quint64 MaxExpandedBytes = 256 * 1024 * 1024;
constexpr int MaxEntries = 4096;
constexpr qint64 MaxManifestBytes = 16 * 1024;
constexpr qint64 ChunkSize = 64 * 1024;
const QString ManifestName = u".lichii-mod-store.json"_s;
const QString StagingTemplate = u".lichii-mod-store-XXXXXX"_s;

std::mutex install_mutex;

bool IsTitleId(const QString& title_id) {
    static const QRegularExpression pattern(u"\\A[0-9a-fA-F]{16}\\z"_s);
    return pattern.match(title_id).hasMatch();
}

bool IsReservedName(const QString& component) {
    static const QRegularExpression pattern(
        u"\\A(?:CON|PRN|AUX|NUL|CLOCK\\$|CONIN\\$|CONOUT\\$|(?:COM|LPT)[0-9¹²³])\\z"_s);
    return pattern.match(component.section(u'.', 0, 0).trimmed().toUpper()).hasMatch();
}

bool IsModId(const QString& mod_id) {
    static const QRegularExpression pattern(u"\\A[a-z0-9][a-z0-9-]{0,62}\\z"_s);
    return pattern.match(mod_id).hasMatch() && !IsReservedName(mod_id);
}

bool IsDisplayMetadata(const QString& name, const QString& version) {
    return !name.trimmed().isEmpty() && name.size() <= 512 && !version.trimmed().isEmpty() &&
           version.size() <= 256;
}

bool IsInstallInfo(const QString& root, const QString& title_id, const QString& mod_id,
                   const QString& name, const QString& version, const QByteArray& sha256) {
    return !root.isEmpty() && IsTitleId(title_id) && IsModId(mod_id) &&
           IsDisplayMetadata(name, version) && sha256.size() == 32;
}

bool IsLink(const QFileInfo& info) {
    return info.isSymLink() || info.isJunction();
}

bool IsDirectory(const QString& path) {
    const QFileInfo info(path);
    return info.isDir() && !IsLink(info);
}

bool IsRegularFile(const QString& path) {
    const QFileInfo info(path);
    return info.isFile() && !IsLink(info);
}

bool IsAbsent(const QString& path) {
    const QFileInfo info(path);
    return !info.exists() && !IsLink(info);
}

bool HasNameConflict(const QDir& parent, const QString& name) {
    const auto entries =
        parent.entryList(QDir::AllEntries | QDir::Hidden | QDir::System | QDir::NoDotAndDotDot);
    return std::ranges::any_of(entries, [&](const QString& entry) {
        return entry != name && entry.compare(name, Qt::CaseInsensitive) == 0;
    });
}

std::optional<InstalledMod> ReadOwnedMod(const QString& path, const QString& title_id,
                                         const QString& mod_id) {
    const QString manifest_path = QDir(path).filePath(ManifestName);
    QFile file(manifest_path);
    if (!IsDirectory(path) || !IsRegularFile(manifest_path) || !file.open(QIODevice::ReadOnly) ||
        file.size() > MaxManifestBytes) {
        return std::nullopt;
    }
    const QByteArray bytes = file.read(MaxManifestBytes + 1);
    if (file.error() != QFileDevice::NoError || bytes.size() > MaxManifestBytes) {
        return std::nullopt;
    }
    const QJsonObject object = QJsonDocument::fromJson(bytes).object();
    InstalledMod mod{
        .id = object[u"id"_s].toString(),
        .version = object[u"version"_s].toString(),
        .name = object[u"name"_s].toString(),
        .sha256 = {},
    };
    if (object[u"schema"_s] != QJsonValue(1) || object[u"title_id"_s].toString() != title_id ||
        !IsModId(mod.id) || (!mod_id.isEmpty() && mod.id != mod_id) ||
        !IsDisplayMetadata(mod.name, mod.version)) {
        return std::nullopt;
    }
    if (const QJsonValue digest = object.value(u"sha256"_s); !digest.isUndefined()) {
        static const QRegularExpression pattern(u"\\A[0-9a-f]{64}\\z"_s);
        const QString hex = digest.toString();
        if (!pattern.match(hex).hasMatch()) {
            return std::nullopt;
        }
        mod.sha256 = QByteArray::fromHex(hex.toLatin1());
    }
    return mod;
}

bool HasSafeContents(const QString& path) {
    QDirIterator entries(path,
                         QDir::AllEntries | QDir::Hidden | QDir::System | QDir::NoDotAndDotDot,
                         QDirIterator::Subdirectories);
    while (entries.hasNext()) {
        entries.next();
        const QFileInfo info = entries.fileInfo();
        if (IsLink(info) || (!info.isDir() && !info.isFile())) {
            return false;
        }
    }
    return true;
}

void RestoreInterruptedUpdates(const QDir& mods_dir, const QDir& title_dir, const QString& title) {
    const QString pattern = StagingTemplate.left(StagingTemplate.size() - 6) + u"??????"_s;
    const auto staged = mods_dir.entryList({pattern}, QDir::Dirs | QDir::Hidden | QDir::NoSymLinks);
    for (const auto& entry : staged) {
        const QDir transaction(mods_dir.filePath(entry));
        if (!IsDirectory(transaction.path())) {
            continue;
        }
        const QString backup = transaction.filePath(u"previous"_s);
        const QString pending = transaction.filePath(u"new"_s);
        const auto mod = ReadOwnedMod(backup, title, {});
        if (!mod || !ReadOwnedMod(pending, title, mod->id) || HasNameConflict(title_dir, mod->id) ||
            !HasSafeContents(backup)) {
            continue;
        }
        const auto destination = title_dir.filePath(mod->id);
        if (IsAbsent(destination) && QDir().rename(backup, destination)) {
            if (HasSafeContents(pending)) {
                QDir(pending).removeRecursively();
            }
            QDir().rmdir(transaction.path());
        }
    }
}

struct ZipEntry {
    QStringList parts;
    QString relative_path;
    quint64 compressed_size;
    quint64 expanded_size;
    quint32 crc;
    quint16 method;
    bool directory;
};

int ReadDirectoryCount(QTemporaryFile& file) {
    constexpr int footer_size = 22;
    constexpr int header_size = 46;
    constexpr qint64 max_directory_bytes = 4 * 1024 * 1024;
    const qint64 end = file.size() - footer_size;
    if (end < 0 || !file.seek(0) || file.read(4) != QByteArrayLiteral("PK\x03\x04") ||
        !file.seek(end)) {
        return 0;
    }
    const QByteArray footer = file.read(footer_size);
    if (footer.size() != footer_size || !footer.startsWith("PK\x05\x06") ||
        qFromLittleEndian<quint32>(footer.constData() + 4) != 0 ||
        qFromLittleEndian<quint16>(footer.constData() + 20) != 0) {
        return 0;
    }
    const int count = qFromLittleEndian<quint16>(footer.constData() + 10);
    const qint64 size = qFromLittleEndian<quint32>(footer.constData() + 12);
    const qint64 offset = qFromLittleEndian<quint32>(footer.constData() + 16);
    if (count < 1 || count > MaxEntries || size > max_directory_bytes || offset + size != end ||
        qFromLittleEndian<quint16>(footer.constData() + 8) != count) {
        return 0;
    }
    qint64 pos = offset;
    QSet<quint32> local_offsets;
    for (int i = 0; i < count; ++i) {
        if (end - pos < header_size || !file.seek(pos)) {
            return 0;
        }
        const QByteArray header = file.read(header_size);
        if (header.size() != header_size || !header.startsWith("PK\x01\x02")) {
            return 0;
        }
        const quint32 local = qFromLittleEndian<quint32>(header.constData() + 42);
        const quint32 compressed = qFromLittleEndian<quint32>(header.constData() + 20);
        if (offset - local - 30 < compressed || local_offsets.contains(local)) {
            return 0;
        }
        local_offsets.insert(local);
        pos += header_size + qFromLittleEndian<quint16>(header.constData() + 28) +
               qFromLittleEndian<quint16>(header.constData() + 30) +
               qFromLittleEndian<quint16>(header.constData() + 32);
    }
    return pos == end && file.seek(0) ? count : 0;
}

bool IsPortableComponent(const QString& component, bool external) {
    static const QRegularExpression server_name(u"\\A[A-Za-z0-9][A-Za-z0-9._ -]{0,119}\\z"_s);
    static const QRegularExpression forbidden(u"[\\p{Cc}\\p{Cf}\\x{fffd}<>:\"|?*~\\\\]"_s);
    return !component.isEmpty() && component.toUtf8().size() <= 255 && !component.endsWith(u'.') &&
           !component.endsWith(u' ') && !IsReservedName(component) &&
           (external ? !component.contains(forbidden) : server_name.match(component).hasMatch());
}

bool HasSafeExtraFields(QByteArrayView extra) {
    while (!extra.empty()) {
        if (extra.size() < 4) {
            return false;
        }
        const quint16 tag = qFromLittleEndian<quint16>(extra.data());
        const quint16 size = qFromLittleEndian<quint16>(extra.data() + 2);
        if (size > extra.size() - 4 || tag == 0x0001 || tag == 0x000d || tag == 0x756e) {
            return false;
        }
        extra = extra.sliced(size + 4);
    }
    return true;
}

QList<ZipEntry> ReadEntries(QuaZip& zip, int count, quint64 archive_bytes, bool external) {
    struct PathNode {
        QString spelling;
        bool directory;
        bool explicit_entry;
    };
    QHash<QString, PathNode> paths;
    QList<ZipEntry> entries;
    entries.reserve(count);
    quint64 expanded = 0;
    quint64 compressed = 0;
    for (bool more = zip.goToFirstFile(); more; more = zip.goToNextFile()) {
        QuaZipFileInfo64 info;
        if (entries.size() >= count || !zip.getCurrentFileInfo(&info) ||
            info.diskNumberStart != 0 || (info.flags & ~0x080e) != 0 ||
            (info.method != 0 && info.method != Z_DEFLATED) ||
            (info.method == 0 && info.compressedSize != info.uncompressedSize) ||
            info.compressedSize > archive_bytes - compressed ||
            info.uncompressedSize > MaxExpandedBytes - expanded ||
            !HasSafeExtraFields(info.extra)) {
            return {};
        }

        const bool directory = info.name.endsWith(u'/');
        const quint32 mode_type = info.externalAttr >> 16 & 0170000;
        const QString path = directory ? info.name.chopped(1) : info.name;
        QStringList parts = path.split(u'/');
        if ((mode_type != 0 && mode_type != (directory ? 0040000U : 0100000U)) ||
            (info.externalAttr & 0x0408) != 0 ||
            (!directory && (info.externalAttr & 0x0010) != 0) ||
            (directory && info.uncompressedSize != 0) ||
            path.toUtf8().size() > (external ? 512 : 220) || parts.size() > (external ? 24 : 16)) {
            return {};
        }

        QString prefix;
        for (qsizetype i = 0; i < parts.size(); ++i) {
            const QString& component = parts[i];
            if (!IsPortableComponent(component, external)) {
                return {};
            }
            prefix = prefix.isEmpty() ? component : prefix + u'/' + component;
            const QString key = prefix.normalized(QString::NormalizationForm_C).toCaseFolded();
            const bool last = i + 1 == parts.size();
            const bool node_is_directory = directory || !last;
            auto existing = paths.find(key);
            if (existing == paths.end()) {
                paths.insert(key, {prefix, node_is_directory, last});
            } else if (existing->spelling != prefix || existing->directory != node_is_directory ||
                       (last && existing->explicit_entry)) {
                return {};
            } else if (last) {
                existing->explicit_entry = true;
            }
        }

        expanded += info.uncompressedSize;
        compressed += info.compressedSize;
        entries.append({std::move(parts),
                        {},
                        info.compressedSize,
                        info.uncompressedSize,
                        info.crc,
                        info.method,
                        directory});
    }
    if (zip.getZipError() != 0 || entries.size() != count) {
        return {};
    }
    return entries;
}

bool IsPayloadRoot(const QString& name) {
    return name == u"exefs" || name == u"romfs" || name == u"cheats" || name == u"romfslite" ||
           name == u"romfs_ext";
}

bool IsDocumentation(const QString& name) {
    static const QRegularExpression pattern(
        u"\\A(?:readme|license|licence)(?:\\.(?:txt|md|rst|markdown))?\\z"_s,
        QRegularExpression::CaseInsensitiveOption);
    return pattern.match(name).hasMatch();
}

std::optional<QStringList> FindPackageRoot(const QStringList& path, const QString& title_id,
                                           bool external) {
    qsizetype depth = 0;
    if (external) {
        const QString first = path.front().toLower();
        if (!IsPayloadRoot(first) && first != u"atmosphere") {
            if (first == u"contents" || IsTitleId(first)) {
                return std::nullopt;
            }
            ++depth;
        }
        if (path.size() > depth && path[depth].compare(u"atmosphere"_s, Qt::CaseInsensitive) == 0) {
            if (path.size() < depth + 5 ||
                path[depth + 1].compare(u"contents"_s, Qt::CaseInsensitive) != 0 ||
                path[depth + 2].toUpper() != title_id) {
                return std::nullopt;
            }
            depth += 3;
        }
    }
    if (path.size() > depth + 1 && IsPayloadRoot(external ? path[depth].toLower() : path[depth])) {
        return path.mid(0, depth);
    }
    return std::nullopt;
}

bool PlanExtraction(QList<ZipEntry>& entries, const QString& title_id, bool external) {
    std::optional<QStringList> package_root;
    for (const ZipEntry& entry : entries) {
        if (!entry.directory) {
            if (auto root = FindPackageRoot(entry.parts, title_id, external)) {
                if (package_root && *package_root != *root) {
                    return false;
                }
                package_root = std::move(root);
            }
        }
    }
    if (!package_root) {
        return false;
    }

    for (ZipEntry& entry : entries) {
        const qsizetype depth = package_root->size();
        const qsizetype length = entry.parts.size();
        if (entry.directory && length <= depth && package_root->mid(0, length) == entry.parts) {
            continue;
        }
        if (!entry.directory && IsDocumentation(entry.parts.back()) && length <= depth &&
            package_root->mid(0, length - 1) == entry.parts.mid(0, length - 1)) {
            continue;
        }
        if (length <= depth || entry.parts.mid(0, depth) != *package_root) {
            return false;
        }
        QStringList relative = entry.parts.mid(depth);
        if (external) {
            relative.front() = relative.front().toLower();
        }
        if (IsPayloadRoot(relative.front()) && (entry.directory || relative.size() > 1)) {
            entry.relative_path = relative.join(u'/');
        } else if (!entry.directory && relative.size() == 1 && IsDocumentation(relative.front())) {
            entry.relative_path = relative.front();
        } else {
            return false;
        }
    }
    return true;
}

bool ExtractEntry(QuaZip& zip, const ZipEntry& entry, QIODevice* output) {
    QuaZipFile input(&zip);
    int method = 0;
    if (!input.open(QIODevice::ReadOnly, &method, nullptr, true) || method != entry.method ||
        !HasSafeExtraFields(input.getLocalExtraField()) || input.getZipError() != 0) {
        return false;
    }

    quint64 remaining = entry.compressed_size;
    quint64 written = 0;
    uLong crc = crc32(0, Z_NULL, 0);
    const auto write = [&](const Bytef* bytes, uInt size) {
        if (size > entry.expanded_size - written ||
            (output && output->write(reinterpret_cast<const char*>(bytes), size) != size)) {
            return false;
        }
        written += size;
        crc = crc32(crc, bytes, size);
        return true;
    };

    std::array<Bytef, ChunkSize> in;
    const auto read = [&] {
        const qint64 bytes =
            input.read(reinterpret_cast<char*>(in.data()), std::min<quint64>(remaining, in.size()));
        if (bytes <= 0) {
            return uInt{0};
        }
        remaining -= bytes;
        return static_cast<uInt>(bytes);
    };

    if (method == 0) {
        while (remaining != 0) {
            const uInt bytes = read();
            if (bytes == 0 || !write(in.data(), bytes)) {
                return false;
            }
        }
    } else {
        z_stream stream{};
        if (inflateInit2(&stream, -MAX_WBITS) != Z_OK) {
            return false;
        }
        const auto cleanup = qScopeGuard([&] { inflateEnd(&stream); });
        std::array<Bytef, ChunkSize> out;
        for (;;) {
            if (stream.avail_in == 0 && remaining != 0) {
                stream.next_in = in.data();
                stream.avail_in = read();
                if (stream.avail_in == 0) {
                    return false;
                }
            }
            const uInt capacity = std::min<quint64>(out.size(), entry.expanded_size - written + 1);
            stream.next_out = out.data();
            stream.avail_out = capacity;
            const int status = inflate(&stream, Z_NO_FLUSH);
            if ((status != Z_OK && status != Z_STREAM_END) ||
                !write(out.data(), capacity - stream.avail_out)) {
                return false;
            }
            if (status == Z_STREAM_END) {
                if (remaining != 0 || stream.avail_in != 0) {
                    return false;
                }
                break;
            }
        }
    }

    input.close();
    return input.getZipError() == 0 && written == entry.expanded_size && crc == entry.crc;
}

bool ExtractArchive(QuaZip& zip, const QList<ZipEntry>& entries, const QDir& stage) {
    if (!zip.goToFirstFile()) {
        return false;
    }
    QSet<QString> directories;
    for (const ZipEntry& entry : entries) {
        if (!zip.hasCurrentFile()) {
            return false;
        }
        QFile output;
        if (!entry.relative_path.isEmpty()) {
            for (qsizetype end = 0; end < entry.relative_path.size();) {
                end = entry.relative_path.indexOf(u'/', end);
                if (end < 0 && !entry.directory) {
                    output.setFileName(stage.filePath(entry.relative_path));
                    if (!output.open(QIODevice::WriteOnly | QIODevice::NewOnly)) {
                        return false;
                    }
                    break;
                }
                if (end < 0) {
                    end = entry.relative_path.size();
                }
                const QString dir = entry.relative_path.left(end++);
                if (!directories.contains(dir)) {
                    if (!stage.mkdir(dir)) {
                        return false;
                    }
                    directories.insert(dir);
                }
            }
        }
        if (!ExtractEntry(zip, entry, output.isOpen() ? &output : nullptr)) {
            return false;
        }
        if (output.isOpen()) {
            output.close();
            if (output.error() != QFileDevice::NoError) {
                return false;
            }
        }
        zip.goToNextFile();
    }
    return !zip.hasCurrentFile() && zip.getZipError() == 0;
}

QString SnapshotSource(QTemporaryFile& snapshot, const QString& path, const QByteArray& sha256,
                       bool raw_patch) {
    const qint64 limit = raw_patch ? MaxPatchBytes : MaxArchiveBytes;
    const QString kind = raw_patch ? u"patch"_s : u"ZIP"_s;
    const QString limit_error =
        u"The downloaded %1 exceeds the %2 MiB limit."_s.arg(kind).arg(limit / (1024 * 1024));
    QFile source(path);
    if (!IsRegularFile(path) || !source.open(QIODevice::ReadOnly) || !snapshot.open()) {
        return u"Could not open the downloaded %1."_s.arg(kind);
    }
    if (source.size() > limit) {
        return limit_error;
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    for (qint64 total = 0;;) {
        const QByteArray block = source.read(ChunkSize);
        if (source.error() != QFileDevice::NoError) {
            return u"Could not read the downloaded %1."_s.arg(kind);
        }
        if (block.isEmpty()) {
            break;
        }
        total += block.size();
        if (total > limit) {
            return limit_error;
        }
        hash.addData(block);
        if (snapshot.write(block) != block.size()) {
            return u"Could not stage the downloaded %1."_s.arg(kind);
        }
    }
    if (hash.result() != sha256) {
        return u"The downloaded %1 does not match its expected SHA-256."_s.arg(kind);
    }
    if (!snapshot.flush() || !snapshot.seek(0)) {
        return u"Could not stage the downloaded %1."_s.arg(kind);
    }
    return {};
}

QList<ZipEntry> ReadArchive(QTemporaryFile& snapshot, QuaZip& zip, bool external) {
    const int count = ReadDirectoryCount(snapshot);
    zip.setAutoClose(false);
    zip.setFileNameCodec("UTF-8");
    if (count == 0 || !zip.open(QuaZip::mdUnzip) || zip.getEntriesCount() != count) {
        return {};
    }
    return ReadEntries(zip, count, snapshot.size(), external);
}

QString NormalizedBuildId(const QString& id, bool cheat) {
    static const QRegularExpression hex(u"\\A[0-9a-fA-F]+\\z"_s);
    const auto length = id.size();
    if ((length != 16 && (cheat || (length != 32 && length != 40 && length != 64))) ||
        !hex.match(id).hasMatch()) {
        return {};
    }
    return id.toUpper().leftJustified(cheat ? 16 : 64, u'0');
}

bool IsPatchMember(const ZipEntry& entry, const GitHubPatch& patch, const QString& build_id,
                   const QString& title) {
    if (entry.directory || entry.expanded_size > MaxPatchBytes) {
        return false;
    }
    for (qsizetype i = 0; i + 2 < entry.parts.size(); ++i) {
        if (entry.parts[i].compare(u"contents"_s, Qt::CaseInsensitive) == 0 &&
            IsTitleId(entry.parts[i + 1]) && entry.parts[i + 1].toUpper() != title) {
            return false;
        }
    }
    const QString& filename = entry.parts.back();
    const qsizetype dot = filename.lastIndexOf(u'.');
    if (dot <= 0) {
        return false;
    }
    const auto extension = QStringView(filename).sliced(dot + 1);
    if (patch.kind == GitHubPatchKind::Pchtxt) {
        return extension.compare(u"pchtxt", Qt::CaseInsensitive) == 0;
    }
    if (patch.kind == GitHubPatchKind::Cheat) {
        return extension.compare(u"txt", Qt::CaseInsensitive) == 0 &&
               NormalizedBuildId(filename.left(dot), true) == build_id;
    }
    return MatchesGitHubIpsFilename(filename, build_id);
}

bool ReadPatchMember(QuaZip& zip, const QList<ZipEntry>& entries, const GitHubPatch& patch,
                     const QString& build_id, const QString& title, QByteArray& bytes) {
    if (entries.isEmpty() || !zip.goToFirstFile()) {
        return false;
    }
    QBuffer output(&bytes);
    if (!output.open(QIODevice::WriteOnly)) {
        return false;
    }
    bool found = false;
    for (const auto& entry : entries) {
        const bool selected = entry.parts.join(u'/') == patch.member;
        if (!zip.hasCurrentFile() ||
            (selected && (found || !IsPatchMember(entry, patch, build_id, title))) ||
            !ExtractEntry(zip, entry, selected ? &output : nullptr)) {
            return false;
        }
        found |= selected;
        zip.goToNextFile();
    }
    return found && !zip.hasCurrentFile() && zip.getZipError() == 0;
}

std::string_view TrimPatchLine(std::string_view line) {
    constexpr std::string_view space = " \t\v\f";
    const auto start = line.find_first_not_of(space);
    if (start == std::string_view::npos) {
        return {};
    }
    return line.substr(start, line.find_last_not_of(space) - start + 1);
}

std::string_view TakePatchLine(std::string_view& bytes) {
    const auto end = bytes.find_first_of("\r\n");
    const auto line = bytes.substr(0, end);
    bytes.remove_prefix(end == std::string_view::npos ? bytes.size() : end + 1);
    return TrimPatchLine(line);
}

bool MatchesPchtxt(QByteArrayView patch, const QString& build_id) {
    std::string_view bytes(patch.data(), patch.size());
    if (bytes.find('\0') != std::string_view::npos) {
        return false;
    }
    bool matched = false;
    while (!bytes.empty()) {
        const auto raw_line = TakePatchLine(bytes);
        const auto line = TrimPatchLine(raw_line.substr(0, raw_line.find_first_of("/#")));
        if (line.size() < 3 || line[0] != '@') {
            continue;
        }
        if (line[1] == 's' || line[1] == 'S') {
            break;
        }
        if (line[1] != 'n' && line[1] != 'N') {
            continue;
        }
        if (matched || line.size() <= 8 || line.size() > 8 + 64) {
            return false;
        }
        const QString directive = QString::fromLatin1(line.data(), line.size());
        if (!directive.startsWith(u"@nsobid-"_s, Qt::CaseInsensitive) ||
            NormalizedBuildId(directive.sliced(8), false) != build_id) {
            return false;
        }
        matched = true;
    }
    return matched;
}

bool IsIps(QByteArrayView bytes, bool require_ips32) {
    const bool ips32 = bytes.startsWith("IPS32");
    if (!ips32 && (require_ips32 || !bytes.startsWith("PATCH"))) {
        return false;
    }
    const QByteArrayView end = ips32 ? "EEOF" : "EOF";
    const qsizetype offset_size = end.size();
    bytes = bytes.sliced(5);
    while (bytes.size() >= offset_size) {
        if (bytes.startsWith(end)) {
            return bytes.size() == offset_size;
        }
        if (bytes.size() < offset_size + 2) {
            return false;
        }
        const quint32 offset = ips32 ? qFromBigEndian<quint32>(bytes.data())
                                    : (quint32(static_cast<quint8>(bytes[0])) << 16) |
                                          qFromBigEndian<quint16>(bytes.data() + 1);
        bytes = bytes.sliced(offset_size);
        quint16 count = qFromBigEndian<quint16>(bytes.data());
        bytes = bytes.sliced(2);
        if (count == 0) {
            if (bytes.size() < 3) {
                return false;
            }
            count = qFromBigEndian<quint16>(bytes.data());
            bytes = bytes.sliced(3);
        } else {
            if (bytes.size() < count) {
                return false;
            }
            bytes = bytes.sliced(count);
        }
        if (count == 0 || count > std::numeric_limits<quint32>::max() - offset) {
            return false;
        }
    }
    return false;
}

bool ScopeCheat(QByteArray& patch, const QString& tag) {
    static const QRegularExpression master(
        u"\\A(?:master\\z|master\\s+code\\b|must\\s+be\\s+on\\b)"_s,
        QRegularExpression::CaseInsensitiveOption);
    constexpr int max_opcodes = 0x100;
    QSet<QString> names;
    QByteArray scoped;
    std::string_view bytes(patch.constData(), patch.size());
    bool selected = false;
    int opcodes = 0;
    bool selected_has_code = false;
    while (!bytes.empty()) {
        auto line = TakePatchLine(bytes);
        if (line.empty()) {
            continue;
        }
        if (line.front() == '[' && line.back() == ']') {
            const auto inner = line.substr(1, line.size() - 2);
            const QString name = QString::fromUtf8(inner.data(), inner.size());
            if (!IsCheatTag(name) || master.match(name.trimmed()).hasMatch() || names.contains(name)) {
                return false;
            }
            names.insert(name);
            selected = !tag.isEmpty() && name == tag;
            opcodes = 0;
            if (selected) {
                scoped.append(line.data(), line.size());
                scoped.append('\n');
            }
            continue;
        }
        if (names.isEmpty()) {
            return false;
        }
        if (selected) {
            scoped.append(line.data(), line.size());
            scoped.append('\n');
        }
        while (!line.empty()) {
            const auto end = line.find_first_of(" \t");
            const auto word = line.substr(0, end);
            if (word.size() != 8 || word.find_first_not_of("0123456789abcdefABCDEF") !=
                                        std::string_view::npos ||
                ++opcodes > max_opcodes) {
                return false;
            }
            const auto next = line.find_first_not_of(" \t", word.size());
            line = next == std::string_view::npos ? std::string_view{} : line.substr(next);
        }
        selected_has_code |= selected;
    }
    if (tag.isEmpty()) {
        return names.size() == 1 && opcodes > 0;
    }
    if (!selected_has_code) {
        return false;
    }
    patch = std::move(scoped);
    return true;
}

bool WriteNewFile(const QString& path, const QByteArray& bytes) {
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::NewOnly) || file.write(bytes) != bytes.size()) {
        return false;
    }
    file.close();
    return file.error() == QFileDevice::NoError;
}

template <typename StagePayload>
QString InstallPreparedMod(const QString& root, const QString& title, const QString& mod_id,
                           const QString& name, const QString& version, const QByteArray& sha256,
                           const QString& stage_error, StagePayload stage_payload) {
    QDir mods_dir(QDir(root).absolutePath());
    const QDir title_dir(mods_dir.filePath(title));
    const QString destination = title_dir.filePath(mod_id);
    const QString root_error = u"The mods directory is unavailable or is a link."_s;
    const QString conflict_error =
        u"An existing mod folder is not owned by the Lichii Mod Store."_s;
    if (!IsDirectory(mods_dir.path()) &&
        (!IsAbsent(mods_dir.path()) || !QDir().mkpath(mods_dir.path()) ||
         !IsDirectory(mods_dir.path()))) {
        return root_error;
    }
    QTemporaryDir transaction(mods_dir.filePath(StagingTemplate));
    if (!transaction.isValid()) {
        return u"Could not create a staging folder in the mods directory."_s;
    }
    const QDir stage(transaction.filePath(u"new"_s));
    if (!QDir().mkdir(stage.path()) || !stage_payload(stage)) {
        return stage_error;
    }

    const QJsonObject object{
        {u"schema"_s, 1},
        {u"title_id"_s, title},
        {u"id"_s, mod_id},
        {u"name"_s, name},
        {u"version"_s, version},
        {u"sha256"_s, QString::fromLatin1(sha256.toHex())},
    };
    if (!WriteNewFile(stage.filePath(ManifestName),
                      QJsonDocument(object).toJson(QJsonDocument::Compact))) {
        return u"Could not write the mod ownership manifest."_s;
    }

    const std::scoped_lock lock(install_mutex);
    if (!IsDirectory(mods_dir.path()) || HasNameConflict(mods_dir, title) ||
        (!IsDirectory(title_dir.path()) && (!IsAbsent(title_dir.path()) || !mods_dir.mkdir(title) ||
                                            !IsDirectory(title_dir.path())))) {
        return root_error;
    }
    RestoreInterruptedUpdates(mods_dir, title_dir, title);
    if (HasNameConflict(title_dir, mod_id)) {
        return conflict_error;
    }
    const bool updating = !IsAbsent(destination);
    const QString backup = transaction.filePath(u"previous"_s);
    if (updating) {
        if (!ReadOwnedMod(destination, title, mod_id) || !HasSafeContents(destination)) {
            return conflict_error;
        }
        if (!mods_dir.rename(destination, backup)) {
            return u"Could not move the previous mod out of the way."_s;
        }
    }
    if (!mods_dir.rename(stage.path(), destination)) {
        if (updating && !mods_dir.rename(backup, destination)) {
            transaction.setAutoRemove(false);
            return u"Could not install or restore the mod. The previous files are in %1."_s.arg(
                backup);
        }
        return u"Could not move the staged mod into the mods directory."_s;
    }
    return {};
}

} // namespace

bool IsCheatTag(const QString& tag) {
    static const QRegularExpression forbidden(u"[\\p{Cc}\\p{Cf}\\p{Cs}\\x{fffd}\\[\\]{}]"_s);
    return tag.size() <= 160 && !tag.trimmed().isEmpty() && !tag.contains(forbidden);
}

bool MatchesGitHubIpsFilename(const QString& filename, QStringView build_id) {
    static const QRegularExpression ips_id{u"\\A[0-9A-F]{16,64}\\z"_s};
    const auto stem = filename.section(u'.', 0, 0).toUpper();
    return (filename.endsWith(u".ips"_s, Qt::CaseInsensitive) ||
            filename.endsWith(u".ips32"_s, Qt::CaseInsensitive)) &&
           ips_id.match(stem).hasMatch() && stem.leftJustified(64, u'0') == build_id;
}

QString InstallMod(const QString& root, const QString& title_id, const QString& mod_id,
                   const QString& name, const QString& version, const QString& archive_path,
                   const QByteArray& sha256, bool external) {
    if (!IsInstallInfo(root, title_id, mod_id, name, version, sha256)) {
        return u"The mod installation information is invalid."_s;
    }
    QTemporaryFile snapshot;
    if (const QString error = SnapshotSource(snapshot, archive_path, sha256, false);
        !error.isEmpty()) {
        return error;
    }
    QuaZip zip(&snapshot);
    QList<ZipEntry> entries = ReadArchive(snapshot, zip, external);
    if (entries.isEmpty()) {
        return u"The ZIP is malformed or contains unsafe entries."_s;
    }
    const QString title = title_id.toUpper();
    if (!PlanExtraction(entries, title, external)) {
        return u"The ZIP does not contain a single supported mod layout for this title."_s;
    }
    return InstallPreparedMod(root, title, mod_id, name, version, sha256,
                              u"Could not extract the mod ZIP safely."_s,
                              [&](const QDir& stage) { return ExtractArchive(zip, entries, stage); });
}

QString InstallGitHubPatch(const QString& root, const QString& title_id, const QString& mod_id,
                           const QString& name, const QString& version, const QString& archive_path,
                           const QByteArray& sha256, const GitHubPatch& patch) {
    const bool cheat = patch.kind == GitHubPatchKind::Cheat;
    const QString build_id = NormalizedBuildId(patch.build_id, cheat);
    if (!IsInstallInfo(root, title_id, mod_id, name, version, sha256) || build_id.isEmpty() ||
        patch.member.size() > 512 || patch.member.toUtf8().size() > 512 ||
        (cheat && !patch.member.isEmpty() && patch.tag.isEmpty()) ||
        (!patch.tag.isEmpty() && (!cheat || !IsCheatTag(patch.tag)))) {
        return u"The mod installation information is invalid."_s;
    }

    const QString title = title_id.toUpper();
    const bool raw_patch = patch.member.isEmpty();
    QTemporaryFile snapshot;
    if (const QString error = SnapshotSource(snapshot, archive_path, sha256, raw_patch);
        !error.isEmpty()) {
        return error;
    }
    QByteArray bytes;
    if (raw_patch) {
        bytes = snapshot.readAll();
        if (snapshot.error() != QFileDevice::NoError || bytes.size() != snapshot.size()) {
            return u"Could not read the downloaded patch."_s;
        }
    } else {
        QuaZip zip(&snapshot);
        const auto entries = ReadArchive(snapshot, zip, true);
        if (!ReadPatchMember(zip, entries, patch, build_id, title, bytes)) {
            return u"The ZIP does not contain the expected safe patch member."_s;
        }
    }

    const QString invalid_patch = u"The downloaded patch is malformed or targets another build."_s;
    QString extension;
    switch (patch.kind) {
    case GitHubPatchKind::Pchtxt:
        if (!MatchesPchtxt(bytes, build_id)) {
            return invalid_patch;
        }
        extension = u".pchtxt"_s;
        break;
    case GitHubPatchKind::Ips:
        if (!IsIps(bytes, patch.member.endsWith(u".ips32"_s, Qt::CaseInsensitive))) {
            return invalid_patch;
        }
        extension = u".ips"_s;
        break;
    case GitHubPatchKind::Cheat:
        if (!ScopeCheat(bytes, patch.tag)) {
            return u"The downloaded cheat has no unambiguous standalone section for this mod."_s;
        }
        extension = u".txt"_s;
        break;
    default:
        return u"The mod installation information is invalid."_s;
    }

    const QString folder = cheat ? u"cheats"_s : u"exefs"_s;
    const QString filename = patch.build_id.toUpper() + extension;
    return InstallPreparedMod(root, title, mod_id, name, version, sha256,
                              u"Could not stage the downloaded patch safely."_s,
                              [&](const QDir& stage) {
                                  return stage.mkdir(folder) &&
                                         WriteNewFile(stage.filePath(folder + u'/' + filename),
                                                      bytes);
                              });
}

QString UninstallMod(const QString& root, const QString& title_id, const QString& mod_id) {
    if (root.isEmpty() || !IsTitleId(title_id) || !IsModId(mod_id)) {
        return u"The mod installation information is invalid."_s;
    }
    std::unique_lock lock(install_mutex);
    QDir mods_dir(QDir(root).absolutePath());
    const QString title = title_id.toUpper();
    const QDir title_dir(mods_dir.filePath(title));
    if (IsAbsent(mods_dir.path())) {
        return {};
    }
    if (!IsDirectory(mods_dir.path()) || HasNameConflict(mods_dir, title)) {
        return u"The mods directory is unavailable or is a link."_s;
    }
    if (IsAbsent(title_dir.path())) {
        return {};
    }
    if (!IsDirectory(title_dir.path())) {
        return u"The mods directory is unavailable or is a link."_s;
    }
    RestoreInterruptedUpdates(mods_dir, title_dir, title);
    const QString destination = title_dir.filePath(mod_id);
    if (HasNameConflict(title_dir, mod_id) ||
        (!IsAbsent(destination) &&
         (!ReadOwnedMod(destination, title, mod_id) || !HasSafeContents(destination)))) {
        return u"An existing mod folder is not owned by the Lichii Mod Store."_s;
    }
    if (IsAbsent(destination)) {
        return {};
    }

    QTemporaryDir transaction(mods_dir.filePath(StagingTemplate));
    if (!transaction.isValid() ||
        !mods_dir.rename(destination, transaction.filePath(u"removed"_s))) {
        return u"Could not remove the mod from the mods directory."_s;
    }
    lock.unlock();
    return {};
}

QList<InstalledMod> ReadInstalledMods(const QString& root, const QString& title_id) {
    if (root.isEmpty() || !IsTitleId(title_id)) {
        return {};
    }
    const std::scoped_lock lock(install_mutex);
    const QDir mods_dir(QDir(root).absolutePath());
    const QString title = title_id.toUpper();
    const QDir title_dir(mods_dir.filePath(title));
    if (!IsDirectory(mods_dir.path()) || HasNameConflict(mods_dir, title) ||
        !IsDirectory(title_dir.path())) {
        return {};
    }
    RestoreInterruptedUpdates(mods_dir, title_dir, title);
    QList<InstalledMod> installed;
    for (const QString& id : title_dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name)) {
        if (IsModId(id)) {
            if (auto mod = ReadOwnedMod(title_dir.filePath(id), title, id)) {
                installed.append(std::move(*mod));
            }
        }
    }
    return installed;
}

} // namespace ModStore
