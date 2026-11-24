// SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "lichii/mod_store/mod_store_client.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>
#include <QBuffer>
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkProxy>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPointer>
#include <QRegularExpression>
#include <QTimer>
#include <QUrlQuery>

namespace ModStore {
namespace {

using namespace Qt::StringLiterals;
using namespace std::chrono_literals;

constexpr qint64 MaxJsonBytes = 1024 * 1024;
constexpr qint64 MaxPreviewBytes = 2 * 1024 * 1024;
constexpr qint64 MaxArchiveBytes = 64 * 1024 * 1024;
constexpr qint64 MaxUnpackedBytes = 256 * 1024 * 1024;
constexpr qint64 ChunkBytes = 64 * 1024;
constexpr qint64 MaxGitHubPatchBytes = 2 * 1024 * 1024;
constexpr int MaxPreviewSide = 2048;
constexpr int MaxOffset = 1'000'000;
constexpr int MaxRequests = 8;
constexpr int MaxRedirects = 4;

const QRegularExpression ModId{u"\\A[a-z0-9][a-z0-9-]{0,62}\\z"_s};
const QRegularExpression VersionId{u"\\A[a-z0-9][a-z0-9._-]{0,62}\\z"_s};
const QRegularExpression TitleId{u"\\A[0-9A-F]{16}\\z"_s};
const QRegularExpression BuildId{u"\\A[0-9A-F]{64}\\z"_s};
const QRegularExpression EvidenceBuildId{u"\\A(?:[0-9A-F]{16}|[0-9A-F]{64})\\z"_s};
const QRegularExpression BuildTag{u"\\Abuild-[1-9][0-9]{0,19}\\z"_s};
const QRegularExpression Sha256{u"\\A[0-9a-f]{64}\\z"_s};

Error MakeError(ErrorCode code, int status = 0) {
    using enum ErrorCode;
    switch (code) {
    case InvalidUrl:
        return {code,
                QCoreApplication::translate("ModStoreClient", "The store URL is not allowed.")};
    case InvalidInput:
        return {code,
                QCoreApplication::translate("ModStoreClient", "The store request is invalid.")};
    case InvalidResponse:
        return {code, QCoreApplication::translate("ModStoreClient",
                                                  "The store sent an invalid response.")};
    case InvalidImage:
        return {code, QCoreApplication::translate("ModStoreClient",
                                                  "The preview image is unsupported or damaged.")};
    case MissingSha256:
        return {code, QCoreApplication::translate("ModStoreClient",
                                                  "The catalog has no checksum for this file.")};
    case SizeMismatch:
        return {code, QCoreApplication::translate("ModStoreClient",
                                                  "The file does not match the catalog size.")};
    case IntegrityMismatch:
        return {code, QCoreApplication::translate("ModStoreClient",
                                                  "The file does not match the catalog checksum.")};
    case TooLarge:
        return {code, QCoreApplication::translate("ModStoreClient",
                                                  "The response exceeds the size limit.")};
    case Http:
        return {code,
                QCoreApplication::translate("ModStoreClient", "The request failed (HTTP %1).")
                    .arg(status),
                status};
    case Network:
        return {code,
                QCoreApplication::translate(
                    "ModStoreClient", "Could not connect to the server or complete the request.")};
    case File:
        return {code, QCoreApplication::translate("ModStoreClient",
                                                  "Could not save the temporary download.")};
    case TimedOut:
        return {code,
                QCoreApplication::translate("ModStoreClient", "The store request timed out.")};
    case Cancelled:
        return {code,
                QCoreApplication::translate("ModStoreClient", "The store request was cancelled.")};
    case Busy:
        return {code, QCoreApplication::translate("ModStoreClient",
                                                  "Too many store requests are already running.")};
    }
    return {code, {}};
}

bool IsCredentialFree(const QUrl& url) {
    return url.isValid() && !url.host().isEmpty() && !url.hasFragment() &&
           !url.authority(QUrl::FullyEncoded).contains(u'@') && url.toEncoded().size() <= 2048;
}

QUrl ParseBaseUrl(const QString& text) {
    const QUrl url(text.trimmed(), QUrl::StrictMode);
    if (!IsCredentialFree(url) || url.hasQuery() || (!url.path().isEmpty() && url.path() != u"/") ||
        url.port() == 0) {
        return {};
    }
    const bool loopback = url.host() == u"127.0.0.1" || url.host() == u"::1";
    if (url.scheme() == u"https" || (url.scheme() == u"http" && loopback && url.port() > 0)) {
        return url;
    }
    return {};
}

QUrl Endpoint(const QUrl& base_url, const QString& path) {
    if (base_url.isEmpty()) {
        return {};
    }
    QUrl url = base_url;
    url.setPath(path);
    return url;
}

QUrl GameBananaUrl(QLatin1StringView category, qint64 id) {
    return QUrl(u"https://gamebanana.com/%1/%2"_s.arg(category).arg(id), QUrl::StrictMode);
}

bool IsGameBananaPreview(const QUrl& url) {
    static const QRegularExpression path{
        u"\\A/img/ss/mods/[A-Za-z0-9_.-]{1,240}\\.(?i:jpg|jpeg|png|webp|gif)\\z"_s};
    return IsCredentialFree(url) && url.scheme() == u"https" && url.port() == -1 &&
           url.host() == u"images.gamebanana.com" && !url.hasQuery() &&
           path.match(url.path(QUrl::FullyEncoded)).hasMatch();
}

bool IsGameBananaCdn(const QUrl& url) {
    static const QRegularExpression host{
        u"\\A(?:files2?|archive|filecache[0-9]{1,3})\\.gamebanana\\.com\\z"_s};
    if (!IsCredentialFree(url) || url.scheme() != u"https" || url.port() != -1 ||
        !host.match(url.host()).hasMatch()) {
        return false;
    }
    const QString path = url.path(QUrl::FullyDecoded);
    if (!path.startsWith(u"/mods/") || path.size() <= 6 || path.contains(u'\\') ||
        std::ranges::any_of(path, [](QChar ch) { return ch.category() == QChar::Other_Control; })) {
        return false;
    }
    const auto parts = path.split(u'/', Qt::SkipEmptyParts);
    return std::ranges::none_of(parts, [](const QString& part) {
        const QString lower = part.toLower();
        return lower == u"." || lower == u".." || lower == u"login" || lower == u"log-in" ||
               lower == u"signin" || lower == u"sign-in" || lower == u"authenticate";
    });
}

QString PreviewPath(const QString& id) {
    return u"/api/v1/mods/%1/preview"_s.arg(id);
}

QString DownloadPath(const QString& mod_id, const QString& version_id) {
    return u"/api/v1/mods/%1/versions/%2/download"_s.arg(mod_id, version_id);
}

bool IsLocalDownload(const Version& version) {
    static const QRegularExpression path{
        u"\\A/api/v1/mods/([a-z0-9][a-z0-9-]{0,62})/versions/([a-z0-9][a-z0-9._-]{0,62})/download\\z"_s};
    const auto match = path.match(QString::fromLatin1(version.download_url.toEncoded()));
    return match.hasMatch() && match.capturedView(2) == version.id;
}

QStringView TypeName(ModType type) {
    switch (type) {
    case ModType::Graphics:
        return u"graphics";
    case ModType::Fps:
        return u"fps";
    case ModType::Cheats:
        return u"cheats";
    case ModType::Translation:
        return u"translation";
    case ModType::Gameplay:
        return u"gameplay";
    }
    return {};
}

bool IsText(const QString& text, qsizetype max_length, bool multiline = false) {
    return text.size() <= max_length * 2 && !text.trimmed().isEmpty() &&
           std::ranges::none_of(text,
                                [multiline](QChar ch) {
                                    return ch.category() == QChar::Other_Control &&
                                           !(multiline && (ch == u'\n' || ch == u'\t'));
                                }) &&
           std::ranges::count_if(text, [](QChar ch) { return !ch.isLowSurrogate(); }) <= max_length;
}

QDateTime DateTime(const QJsonValue& value) {
    const QDateTime date = QDateTime::fromString(value.toString(), Qt::ISODateWithMs);
    return date.timeSpec() == Qt::LocalTime ? QDateTime{} : date.toUTC();
}

QByteArray ParseHash(const QJsonValue& value) {
    const QString hex = value.toString();
    return Sha256.match(hex).hasMatch() ? QByteArray::fromHex(hex.toLatin1()) : QByteArray{};
}

std::optional<QStringList> StringList(const QJsonValue& value, qsizetype max_count,
                                      const QRegularExpression& pattern) {
    const auto items = value.toArray();
    if (!value.isArray() || items.size() > max_count) {
        return std::nullopt;
    }
    QStringList out;
    out.reserve(items.size());
    for (const auto item : items) {
        const QString text = item.toString();
        if (!pattern.match(text).hasMatch()) {
            return std::nullopt;
        }
        out.append(text);
    }
    return out;
}

std::optional<Game> ParseGame(const QJsonObject& row) {
    Game game{row.value("title_id"_L1).toString(), row.value("name"_L1).toString(),
              row.value("mod_count"_L1).toInt(-1)};
    if (!TitleId.match(game.title_id).hasMatch() || !IsText(game.name, 120) || game.mod_count < 1) {
        return std::nullopt;
    }
    return game;
}

std::optional<Version> ParseVersion(const QJsonObject& row, const Mod& mod) {
    Version version{
        .id = row.value("id"_L1).toString(),
        .title_id = row.value("title_id"_L1).toString(),
        .game_version = row.value("game_version"_L1).toString(),
        .build_id = row.value("build_id"_L1).toString(),
        .min_lichii_build = row.value("min_lichii_build"_L1).toString(),
        .download_url = QUrl(row.value("download_url"_L1).toString(), QUrl::StrictMode),
        .sha256 = ParseHash(row.value("sha256"_L1)),
        .size_bytes = row.value("size_bytes"_L1).toInteger(-1),
        .unpacked_size_bytes = row.value("unpacked_size_bytes"_L1).toInteger(-1),
    };
    bool valid_build{};
    if (BuildTag.match(version.min_lichii_build).hasMatch()) {
        version.min_lichii_build.sliced(6).toULongLong(&valid_build);
    }
    if (!VersionId.match(version.id).hasMatch() || !mod.title_ids.contains(version.title_id) ||
        !IsText(version.game_version, 32) || !BuildId.match(version.build_id).hasMatch() ||
        !valid_build ||
        version.download_url != QUrl(DownloadPath(mod.id, version.id), QUrl::StrictMode) ||
        version.sha256.isEmpty() || version.size_bytes <= 0 ||
        version.size_bytes > MaxArchiveBytes || version.unpacked_size_bytes < 0 ||
        version.unpacked_size_bytes > MaxUnpackedBytes) {
        return std::nullopt;
    }
    return version;
}

std::optional<ExternalFile> ParseExternalFile(const QJsonObject& row) {
    const auto ambiguous_layout = row.value("ambiguous_layout"_L1);
    ExternalFile file{
        .id = row.value("id"_L1).toInteger(-1),
        .name = row.value("name"_L1).toString(),
        .size_bytes = row.value("size_bytes"_L1).toInteger(-1),
        .uploaded_at = DateTime(row.value("uploaded_at"_L1)),
        .download_url = QUrl(row.value("download_url"_L1).toString(), QUrl::StrictMode),
        .sha256 = ParseHash(row.value("sha256"_L1)),
        .build_ids = {},
        .ambiguous_layout = ambiguous_layout.toBool(),
    };
    auto build_ids = StringList(row.value("build_ids"_L1), 32, EvidenceBuildId);
    if (file.id <= 0 || !IsText(file.name, 240) || file.size_bytes <= 0 ||
        !file.uploaded_at.isValid() || file.download_url != GameBananaUrl("dl"_L1, file.id) ||
        !build_ids || (!ambiguous_layout.isUndefined() && !ambiguous_layout.isBool()) ||
        (file.sha256.isEmpty() && (!row.value("sha256"_L1).isNull() || !build_ids->isEmpty()))) {
        return std::nullopt;
    }
    file.build_ids = std::move(*build_ids);
    return file;
}

std::optional<External> ParseExternal(const QJsonObject& row, const Mod& mod) {
    External external;
    external.id = row.value("id"_L1).toInteger(-1);
    external.category = row.value("category"_L1).toString();
    const auto credits = row.value("credits"_L1).toArray();
    const auto previews = row.value("preview_urls"_L1).toArray();
    const auto files = row.value("files"_L1).toArray();
    if (external.id <= 0 || mod.id != u"gb-%1"_s.arg(external.id) ||
        !IsText(external.category, 160) ||
        row.value("compatibility"_L1).toString() != u"unverified" || credits.isEmpty() ||
        credits.size() > 8 || !row.value("preview_urls"_L1).isArray() || previews.size() > 8 ||
        files.isEmpty() || files.size() > 8) {
        return std::nullopt;
    }
    static const QRegularExpression member_url{
        u"\\Ahttps://gamebanana\\.com/members/[0-9]{1,19}\\z"_s};
    for (const auto value : credits) {
        const auto item = value.toObject();
        const QString url = item.value("url"_L1).toString();
        Credit credit{item.value("name"_L1).toString(), item.value("role"_L1).toString(), {}};
        if (!IsText(credit.name, 100) || !IsText(credit.role, 100) ||
            (!url.isEmpty() && !member_url.match(url).hasMatch())) {
            return std::nullopt;
        }
        credit.url = QUrl(url, QUrl::StrictMode);
        external.credits.append(std::move(credit));
    }
    for (const auto value : previews) {
        const QUrl url(value.toString(), QUrl::StrictMode);
        if (IsGameBananaPreview(url)) {
            external.preview_urls.append(url);
        }
    }
    for (const auto value : files) {
        auto file = ParseExternalFile(value.toObject());
        if (!file) {
            return std::nullopt;
        }
        external.files.append(std::move(*file));
    }
    return external;
}

bool IsGitHubPath(const QString& path) {
    if (path.isEmpty() || path.toUtf8().size() > 512 || path.contains(u'\\') || path.contains(u':') ||
        path.contains(u'%') || path.contains(u'?') || path.contains(u'#')) {
        return false;
    }
    return std::ranges::all_of(path.split(u'/'), [](const QString& part) {
        return !part.isEmpty() && part != u"." && part != u".." &&
               std::ranges::none_of(part, [](QChar ch) { return ch.category() == QChar::Other_Control; });
    });
}

bool IsRawGitHubUrl(const QUrl& url) {
    return IsCredentialFree(url) && url.scheme() == u"https" && url.port() == -1 &&
           url.host() == u"raw.githubusercontent.com" && !url.hasQuery();
}

std::optional<GitHubDownload> ParseGitHubDownload(const QJsonObject& row,
                                                 const GitHubListing& listing, const QString& id) {
    const auto path = row.value("path"_L1).toString();
    const auto member = row.value("member"_L1);
    const auto tag = row.value("tag"_L1);
    const auto kind = row.value("kind"_L1).toString();
    const auto leaf = member.isUndefined() ? path : member.toString();
    const bool archive = !member.isUndefined();
    if (!IsGitHubPath(path) ||
        (archive && (!IsGitHubPath(leaf) || !path.endsWith(u".zip", Qt::CaseInsensitive)))) {
        return std::nullopt;
    }
    const auto has_suffix = [&](QStringView suffix) {
        return leaf.endsWith(suffix, Qt::CaseInsensitive);
    };
    GitHubPatchKind patch_kind;
    if (kind == u"pchtxt" && id.size() == 64 && has_suffix(u".pchtxt")) {
        patch_kind = GitHubPatchKind::Pchtxt;
    } else if (kind == u"ips" && id.size() == 64 &&
               MatchesGitHubIpsFilename(leaf.section(u'/', -1), id)) {
        patch_kind = GitHubPatchKind::Ips;
    } else if (kind == u"cheat" && id.size() == 16 && (!archive || !tag.isUndefined()) &&
               has_suffix(u".txt") &&
               leaf.section(u'/', -1).chopped(4).compare(id, Qt::CaseInsensitive) == 0) {
        patch_kind = GitHubPatchKind::Cheat;
    } else {
        return std::nullopt;
    }
    const QString tag_text = tag.toString();
    if (!tag.isUndefined() && (patch_kind != GitHubPatchKind::Cheat || !IsCheatTag(tag_text))) {
        return std::nullopt;
    }
    const auto size = row.value("size_bytes"_L1).toInteger(-1);
    const auto sha256 = ParseHash(row.value("sha256"_L1));
    if (size <= 0 || size > (archive ? MaxArchiveBytes : MaxGitHubPatchBytes) || sha256.size() != 32) {
        return std::nullopt;
    }
    QUrl url(u"https://raw.githubusercontent.com"_s, QUrl::StrictMode);
    url.setPath(u"/%1/%2/%3"_s.arg(listing.repository, listing.commit, path), QUrl::DecodedMode);
    if (!IsRawGitHubUrl(url)) {
        return std::nullopt;
    }
    return GitHubDownload{url, sha256, size, {patch_kind, id, member.toString(), tag_text}};
}

std::optional<GitHubListing> ParseGitHub(const QJsonObject& row, const QUrl& source_url) {
    GitHubListing listing{
        .repository = row.value("repository"_L1).toString(),
        .commit = row.value("commit"_L1).toString(),
        .builds = {},
    };
    static const QRegularExpression commit{u"\\A[0-9a-f]{40}\\z"_s};
    const auto expected =
        QUrl(u"https://github.com/%1/tree/%2"_s.arg(listing.repository, listing.commit),
             QUrl::StrictMode);
    const auto builds_value = row.value("builds"_L1);
    const auto builds = builds_value.toArray();
    if (!commit.match(listing.commit).hasMatch() || source_url != expected ||
        row.value("license"_L1).toString() != u"NOASSERTION" || !builds_value.isArray() ||
        builds.size() > 128) {
        return std::nullopt;
    }
    QSet<QString> seen;
    for (const auto value : builds) {
        const auto build = value.toObject();
        const auto id = build.value("id"_L1).toString();
        const auto target = build.value("target"_L1).toString();
        if (!EvidenceBuildId.match(id).hasMatch() || seen.contains(id) ||
            std::ranges::all_of(id, [](QChar ch) { return ch == u'0'; }) ||
            (target != u"main" && target != u"unknown")) {
            return std::nullopt;
        }
        seen.insert(id);
        GitHubBuild result{id, target == u"main", {}};
        if (const auto download = build.value("download"_L1); !download.isUndefined()) {
            result.download = ParseGitHubDownload(download.toObject(), listing, id);
            if (!result.download ||
                (result.download->patch.kind == GitHubPatchKind::Cheat && !result.main)) {
                return std::nullopt;
            }
        }
        listing.builds.append(std::move(result));
    }
    return listing;
}

std::optional<Mod> ParseMod(const QJsonObject& row, bool detail = false) {
    Mod mod;
    mod.id = row.value("id"_L1).toString();
    mod.name = row.value("name"_L1).toString();
    mod.summary = row.value("summary"_L1).toString();
    mod.author = row.value("author"_L1).toString();
    mod.updated_at = DateTime(row.value("updated_at"_L1));
    auto titles = StringList(row.value("title_ids"_L1), 128, TitleId);
    if (!ModId.match(mod.id).hasMatch() || !titles || titles->isEmpty() || !IsText(mod.name, 120) ||
        !IsText(mod.summary, 240) || !IsText(mod.author, 100) || !mod.updated_at.isValid() ||
        !row.value("demo"_L1).isBool()) {
        return std::nullopt;
    }
    mod.title_ids = std::move(*titles);
    mod.demo = row.value("demo"_L1).toBool();
    const auto type = row.value("type"_L1).toString();
    constexpr ModType types[]{ModType::Graphics, ModType::Fps, ModType::Cheats,
                              ModType::Translation, ModType::Gameplay};
    const auto* type_it = std::ranges::find(types, type, TypeName);
    if (type_it == std::end(types)) {
        return std::nullopt;
    }
    mod.type = *type_it;
    const auto source = row.value("source"_L1).toString();
    if (source == u"demo" && mod.demo) {
        mod.source = Source::Demo;
    } else if (source == u"submission" && !mod.demo) {
        mod.source = Source::Submission;
    } else if (source == u"gamebanana" && !mod.demo && mod.title_ids.size() == 1) {
        mod.source = Source::GameBanana;
        bool valid{};
        const qint64 id = mod.id.mid(3).toLongLong(&valid);
        mod.source_url = QUrl(row.value("source_url"_L1).toString(), QUrl::StrictMode);
        if (!valid || id <= 0 || mod.id != u"gb-%1"_s.arg(id) ||
            mod.source_url != GameBananaUrl("mods"_L1, id)) {
            return std::nullopt;
        }
    } else if (source == u"github" && !mod.demo && mod.title_ids.size() == 1 &&
               mod.id.startsWith(u"gh-")) {
        static const QRegularExpression url{
            u"\\Ahttps://github\\.com/[A-Za-z0-9-]{1,39}/(?!\\.{1,2}/)[A-Za-z0-9_.-]{1,100}"
            u"/tree/[0-9a-f]{40}\\z"_s};
        const auto source_url = row.value("source_url"_L1).toString();
        if (!url.match(source_url).hasMatch() || !row.value("preview_url"_L1).isNull()) {
            return std::nullopt;
        }
        mod.source_url = QUrl(source_url, QUrl::StrictMode);
        mod.source = Source::GitHub;
    } else {
        return std::nullopt;
    }
    if (const auto preview = row.value("preview_url"_L1); !preview.isNull()) {
        const QUrl url(preview.toString(), QUrl::StrictMode);
        if (mod.source == Source::GameBanana) {
            if (IsGameBananaPreview(url)) {
                mod.preview_url = url;
            }
        } else if (url != QUrl(PreviewPath(mod.id))) {
            return std::nullopt;
        } else {
            mod.preview_url = url;
        }
    }
    if (!detail) {
        return mod;
    }

    mod.description = row.value("description"_L1).toString();
    const auto value = row.value("versions"_L1);
    const auto versions = value.toArray();
    if (!IsText(mod.description, 4000, true) || !value.isArray() || versions.size() > 128) {
        return std::nullopt;
    }
    if (mod.source == Source::GameBanana) {
        mod.external = ParseExternal(row.value("external"_L1).toObject(), mod);
        if (!versions.isEmpty() || !mod.external) {
            return std::nullopt;
        }
        return mod;
    }
    if (mod.source == Source::GitHub) {
        mod.github = ParseGitHub(row.value("github"_L1).toObject(), mod.source_url);
        if (!versions.isEmpty() || !mod.github) {
            return std::nullopt;
        }
        return mod;
    }
    if (versions.isEmpty()) {
        return std::nullopt;
    }
    for (const auto item : versions) {
        auto version = ParseVersion(item.toObject(), mod);
        if (!version) {
            return std::nullopt;
        }
        mod.versions.append(std::move(*version));
    }
    return mod;
}

bool IsPageRange(int offset, int limit) {
    return offset >= 0 && offset <= MaxOffset && limit >= 1 && limit <= 100;
}

template <typename T, typename Parse>
std::optional<Page<T>> ParsePage(const QJsonObject& row, QLatin1StringView key, int offset,
                                 int limit, Parse parse) {
    const auto value = row.value(key);
    const auto items = value.toArray();
    const auto next_offset = row.value("next_offset"_L1);
    Page<T> page;
    if (!value.isArray() || items.size() > limit) {
        return std::nullopt;
    }
    if (!next_offset.isNull()) {
        const int next = next_offset.toInt(-1);
        if (next <= offset || next > MaxOffset || items.isEmpty()) {
            return std::nullopt;
        }
        page.next_offset = next;
    }
    page.items.reserve(items.size());
    for (const auto item : items) {
        auto parsed = parse(item.toObject());
        if (!parsed) {
            return std::nullopt;
        }
        page.items.append(std::move(*parsed));
    }
    return page;
}

template <typename T, typename Parse>
auto JsonCallback(Callback<T> done, Parse parse) {
    return [callback = std::move(done), parse_object = std::move(parse)](auto result) {
        if (auto* error = std::get_if<Error>(&result)) {
            callback(std::move(*error));
        } else if (const auto document = QJsonDocument::fromJson(std::get<QByteArray>(result));
                   document.isObject()) {
            if (auto parsed = parse_object(document.object())) {
                callback(std::move(*parsed));
            } else {
                callback(MakeError(ErrorCode::InvalidResponse));
            }
        } else {
            callback(MakeError(ErrorCode::InvalidResponse));
        }
    };
}

QImage DecodePreview(QByteArray bytes, bool gamebanana) {
    QByteArray format;
    if (bytes.startsWith("\x89PNG\r\n\x1a\n")) {
        format = "png";
    } else if (bytes.startsWith("\xff\xd8\xff")) {
        format = "jpeg";
    } else if (gamebanana && (bytes.startsWith("GIF87a") || bytes.startsWith("GIF89a"))) {
        format = "gif";
    } else if (gamebanana && bytes.size() >= 12 && bytes.startsWith("RIFF") &&
               QByteArrayView(bytes).sliced(8, 4) == "WEBP") {
        format = "webp";
    } else {
        return {};
    }
    QBuffer buffer(&bytes);
    buffer.open(QIODevice::ReadOnly);
    QImageReader reader(&buffer, format);
    reader.setAutoDetectImageFormat(false);
    const QSize size = reader.size();
    if (!size.isValid() || size.isEmpty() || size.width() > MaxPreviewSide ||
        size.height() > MaxPreviewSide) {
        return {};
    }
    QImage image = reader.read();
    return image.width() <= MaxPreviewSide && image.height() <= MaxPreviewSide ? image : QImage{};
}

} // namespace

struct Request::Transfer {
    QUrl url;
    Client::TransferKind kind{};
    qint64 limit{};
    qint64 received{};
    QByteArray expected_hash;
    std::function<void(Result)> done;
    Progress progress;
    std::optional<Error> error;
    QByteArray bytes;
    std::unique_ptr<QTemporaryFile> file;
    std::optional<QCryptographicHash> hash;
    QPointer<QNetworkReply> reply;
    QTimer* deadline{};
    int redirects{};
    bool counted{};
    bool headers_checked{};
    bool archive{};
};

Request::~Request() {
    ReleaseReply();
}

void Request::Cancel() {
    Finish(MakeError(ErrorCode::Cancelled));
}

void Request::Begin() {
    if (!transfer) {
        return;
    }
    if (transfer->error) {
        Finish(std::move(*transfer->error));
        return;
    }
    if (transfer->archive) {
        transfer->file =
            std::make_unique<QTemporaryFile>(QDir::temp().filePath(u"lichii-mod-XXXXXX"_s));
        if (!transfer->file->open()) {
            Finish(MakeError(ErrorCode::File));
            return;
        }
        transfer->hash.emplace(QCryptographicHash::Sha256);
    }
    transfer->deadline = new QTimer(this);
    transfer->deadline->setSingleShot(true);
    connect(transfer->deadline, &QTimer::timeout, this,
            [this] { Finish(MakeError(ErrorCode::TimedOut)); });
    transfer->deadline->start(transfer->archive ? 5min : 25s);
    Get(transfer->url);
}

void Request::Get(const QUrl& url) {
    QNetworkRequest request(url);
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                         QNetworkRequest::ManualRedirectPolicy);
    request.setAttribute(QNetworkRequest::AuthenticationReuseAttribute, QNetworkRequest::Manual);
    request.setAttribute(QNetworkRequest::CookieLoadControlAttribute, QNetworkRequest::Manual);
    request.setAttribute(QNetworkRequest::CookieSaveControlAttribute, QNetworkRequest::Manual);
    request.setAttribute(QNetworkRequest::CacheLoadControlAttribute,
                         QNetworkRequest::AlwaysNetwork);
    request.setAttribute(QNetworkRequest::CacheSaveControlAttribute, false);
    request.setRawHeader("Accept-Encoding", "identity");
    request.setRawHeader("Cache-Control", "no-store");
    switch (transfer->kind) {
    case Client::TransferKind::Json:
        request.setRawHeader("Accept", "application/json");
        request.setRawHeader("X-Lichii-Catalog-Features", "github-v1");
        break;
    case Client::TransferKind::Preview:
        request.setRawHeader("Accept", "image/png, image/jpeg, image/webp, image/gif");
        break;
    case Client::TransferKind::GitHubPatch:
        request.setRawHeader("Accept", "application/octet-stream, text/plain, application/zip");
        break;
    default:
        request.setRawHeader("Accept", "application/zip, application/octet-stream, */*");
        break;
    }
    request.setTransferTimeout(15'000);
    auto* const reply = static_cast<Client*>(parent())->network->get(request);
    transfer->reply = reply;
    transfer->headers_checked = false;
    reply->setReadBufferSize(ChunkBytes);
    connect(reply, &QNetworkReply::metaDataChanged, this, [this] { CheckHeaders(); });
    connect(reply, &QIODevice::readyRead, this, &Request::Drain);
    connect(reply, &QNetworkReply::finished, this, &Request::Complete);
}

bool Request::CheckHeaders() {
    if (transfer->headers_checked) {
        return true;
    }
    auto* const reply = transfer->reply.data();
    const int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    if (status == 0) {
        return false;
    }
    if (status == 301 || status == 302 || status == 303 || status == 307 || status == 308) {
        if (transfer->kind == Client::TransferKind::GameBananaArchive &&
            transfer->redirects < MaxRedirects) {
            const auto location = reply->rawHeader("Location");
            if (!location.isEmpty() && location.size() <= 2048) {
                const QUrl target = QUrl::fromEncoded(location, QUrl::StrictMode);
                const QUrl next = reply->url().resolved(target);
                if (target.isValid() && IsGameBananaCdn(next)) {
                    ++transfer->redirects;
                    ReleaseReply();
                    Get(next);
                    return false;
                }
            }
        }
        Finish(MakeError(ErrorCode::InvalidUrl));
        return false;
    }
    if (status != 200) {
        Finish(MakeError(ErrorCode::Http, status));
        return false;
    }
    const auto encoding = reply->rawHeader("Content-Encoding").trimmed().toLower();
    const auto mime = reply->header(QNetworkRequest::ContentTypeHeader).toByteArray().toLower();
    const bool from_gamebanana = transfer->kind == Client::TransferKind::GameBananaArchive;
    const bool from_github = transfer->kind == Client::TransferKind::GitHubPatch;
    if ((!encoding.isEmpty() && encoding != "identity") ||
        (from_gamebanana && !IsGameBananaCdn(reply->url())) ||
        (from_github && !IsRawGitHubUrl(reply->url())) ||
        ((from_gamebanana || from_github) &&
         (mime.startsWith("text/html") || mime.startsWith("application/xhtml+xml")))) {
        Finish(MakeError(ErrorCode::InvalidResponse));
        return false;
    }
    bool known_size{};
    const qint64 size = reply->header(QNetworkRequest::ContentLengthHeader).toLongLong(&known_size);
    if (transfer->archive && known_size && size != transfer->limit) {
        Finish(MakeError(ErrorCode::SizeMismatch));
        return false;
    }
    if (known_size && size > transfer->limit) {
        Finish(MakeError(ErrorCode::TooLarge));
        return false;
    }
    transfer->headers_checked = true;
    return true;
}

void Request::Drain() {
    if (!transfer || !CheckHeaders()) {
        return;
    }
    const qint64 previous = transfer->received;
    auto* const reply = transfer->reply.data();
    while (reply->bytesAvailable() > 0) {
        const QByteArray chunk =
            reply->read(std::min(ChunkBytes, transfer->limit - transfer->received + 1));
        if (chunk.isEmpty()) {
            break;
        }
        transfer->received += chunk.size();
        if (transfer->received > transfer->limit) {
            Finish(MakeError(transfer->archive ? ErrorCode::SizeMismatch : ErrorCode::TooLarge));
            return;
        }
        if (transfer->archive) {
            if (transfer->file->write(chunk) != chunk.size()) {
                Finish(MakeError(ErrorCode::File));
                return;
            }
            transfer->hash->addData(chunk);
        } else {
            transfer->bytes.append(chunk);
        }
    }
    if (transfer->progress && transfer->received != previous) {
        const auto progress = transfer->progress;
        progress(transfer->received, transfer->limit);
    }
}

void Request::Complete() {
    if (!transfer) {
        return;
    }
    auto* const reply = transfer->reply.data();
    const QPointer guard(this);
    Drain();
    if (!guard || !transfer || transfer->reply != reply) {
        return;
    }
    if (reply->error() != QNetworkReply::NoError) {
        const auto code = reply->error();
        Finish(MakeError(code == QNetworkReply::TimeoutError ||
                                 code == QNetworkReply::OperationCanceledError
                             ? ErrorCode::TimedOut
                             : ErrorCode::Network));
    } else if (!transfer->headers_checked) {
        Finish(MakeError(ErrorCode::InvalidResponse));
    } else if (!transfer->archive) {
        Finish(std::move(transfer->bytes));
    } else if (transfer->received != transfer->limit) {
        Finish(MakeError(ErrorCode::SizeMismatch));
    } else if (QByteArray sha256 = transfer->hash->result();
               !transfer->expected_hash.isEmpty() && sha256 != transfer->expected_hash) {
        Finish(MakeError(ErrorCode::IntegrityMismatch));
    } else if (!transfer->file->flush()) {
        Finish(MakeError(ErrorCode::File));
    } else {
        transfer->file->close();
        Finish(DownloadedFile{std::move(transfer->file), std::move(sha256)});
    }
}

void Request::Finish(Result result) {
    if (!transfer) {
        return;
    }
    ReleaseReply();
    if (transfer->deadline) {
        transfer->deadline->stop();
    }
    if (transfer->counted) {
        auto* const client = static_cast<Client*>(parent());
        --client->active_requests;
        if (transfer->archive) {
            client->active_download = false;
        }
    }
    auto done = std::move(transfer->done);
    transfer.reset();
    QTimer::singleShot(
        0, this, [this, callback = std::move(done), completed = std::move(result)]() mutable {
            deleteLater();
            callback(std::move(completed));
        });
}

void Request::ReleaseReply() {
    if (transfer && transfer->reply) {
        auto* const reply = transfer->reply.data();
        transfer->reply.clear();
        disconnect(reply, nullptr, this, nullptr);
        if (!reply->isFinished()) {
            reply->abort();
        }
        reply->deleteLater();
    }
}

Client::Client(const QString& url, QObject* parent)
    : QObject(parent), base_url(ParseBaseUrl(url)), network(new QNetworkAccessManager(this)) {
    network->setProxy(QNetworkProxy::NoProxy);
}

bool Client::IsValidBaseUrl(const QString& base_url) {
    return !ParseBaseUrl(base_url).isEmpty();
}

Request* Client::Start(const QUrl& url, TransferKind kind,
                       std::function<void(Request::Result)> done, std::optional<Error> error,
                       qint64 size, QByteArray hash, Progress progress) {
    auto* const request = new Request(this);
    request->transfer = std::make_unique<Request::Transfer>();
    auto& transfer = *request->transfer;
    transfer.url = url;
    transfer.kind = kind;
    transfer.archive =
        kind == TransferKind::LocalArchive || kind == TransferKind::GameBananaArchive ||
        kind == TransferKind::GitHubPatch;
    transfer.limit =
        transfer.archive ? size : (kind == TransferKind::Json ? MaxJsonBytes : MaxPreviewBytes);
    transfer.expected_hash = std::move(hash);
    transfer.done = std::move(done);
    transfer.progress = std::move(progress);
    transfer.error = std::move(error);
    if (!transfer.error) {
        if (base_url.isEmpty() || url.isEmpty()) {
            transfer.error = MakeError(ErrorCode::InvalidUrl);
        } else if (active_requests >= MaxRequests || (transfer.archive && active_download)) {
            transfer.error = MakeError(ErrorCode::Busy);
        } else {
            ++active_requests;
            active_download |= transfer.archive;
            transfer.counted = true;
        }
    }
    QTimer::singleShot(0, request, &Request::Begin);
    return request;
}

Request* Client::GetHealth(Callback<bool> done) {
    return Start(
        Endpoint(base_url, u"/healthz"_s), TransferKind::Json,
        JsonCallback<bool>(std::move(done), [](const QJsonObject& row) -> std::optional<bool> {
            return row.value("status"_L1).toString() == u"ok" ? std::optional{true} : std::nullopt;
        }));
}

Request* Client::GetGames(int offset, int limit, Callback<Page<Game>> done) {
    QUrl url = Endpoint(base_url, u"/api/v1/games"_s);
    url.setQuery(u"offset=%1&limit=%2"_s.arg(offset).arg(limit));
    return Start(url, TransferKind::Json,
                 JsonCallback<Page<Game>>(std::move(done),
                                          [offset, limit](const QJsonObject& row) {
                                              return ParsePage<Game>(row, "games"_L1, offset, limit,
                                                                     ParseGame);
                                          }),
                 IsPageRange(offset, limit) ? std::nullopt
                                            : std::optional{MakeError(ErrorCode::InvalidInput)});
}

Request* Client::GetMods(const ModFilter& filter, int offset, int limit, Callback<Page<Mod>> done) {
    QUrl url = Endpoint(base_url, u"/api/v1/mods"_s);
    const QString title = filter.title_id.toUpper();
    const QString query = filter.query.trimmed();
    const QStringView type = filter.type ? TypeName(*filter.type) : QStringView{};
    const bool valid =
        IsPageRange(offset, limit) && (title.isEmpty() || TitleId.match(title).hasMatch()) &&
        (query.isEmpty() || IsText(query, 128)) && (!filter.type || !type.isEmpty()) &&
        (filter.sort == Sort::Name || filter.sort == Sort::Updated);
    if (valid) {
        QUrlQuery params;
        params.addQueryItem(u"offset"_s, QString::number(offset));
        params.addQueryItem(u"limit"_s, QString::number(limit));
        params.addQueryItem(u"sort"_s, filter.sort == Sort::Name ? u"name"_s : u"updated"_s);
        if (!title.isEmpty()) {
            params.addQueryItem(u"title_id"_s, title);
        }
        if (!query.isEmpty()) {
            params.addQueryItem(u"q"_s, QString::fromLatin1(QUrl::toPercentEncoding(query)));
        }
        if (!type.isEmpty()) {
            params.addQueryItem(u"type"_s, type.toString());
        }
        url.setQuery(params);
    }
    return Start(url, TransferKind::Json,
                 JsonCallback<Page<Mod>>(std::move(done),
                                         [offset, limit](const QJsonObject& row) {
                                             return ParsePage<Mod>(row, "mods"_L1, offset, limit,
                                                                   [](const QJsonObject& mod) {
                                                                       return ParseMod(mod);
                                                                   });
                                         }),
                 valid ? std::nullopt : std::optional{MakeError(ErrorCode::InvalidInput)});
}

Request* Client::GetMod(const QString& id, Callback<Mod> done) {
    const bool valid = ModId.match(id).hasMatch();
    const QUrl url = valid ? Endpoint(base_url, u"/api/v1/mods/"_s + id) : QUrl{};
    return Start(url, TransferKind::Json,
                 JsonCallback<Mod>(std::move(done),
                                   [id](const QJsonObject& row) -> std::optional<Mod> {
                                       auto mod = ParseMod(row, true);
                                       return mod && mod->id == id ? std::move(mod) : std::nullopt;
                                   }),
                 valid ? std::nullopt : std::optional{MakeError(ErrorCode::InvalidInput)});
}

Request* Client::GetPreview(const Mod& mod, Callback<QImage> done) {
    const bool gamebanana = mod.source == Source::GameBanana;
    QUrl url;
    if (gamebanana && IsGameBananaPreview(mod.preview_url)) {
        url = mod.preview_url;
    } else if (!gamebanana && ModId.match(mod.id).hasMatch() &&
               mod.preview_url == QUrl(PreviewPath(mod.id))) {
        url = Endpoint(base_url, PreviewPath(mod.id));
    }
    return Start(
        url, TransferKind::Preview, [callback = std::move(done), gamebanana](Request::Result result) {
            if (auto* error = std::get_if<Error>(&result)) {
                callback(std::move(*error));
            } else if (QImage image =
                           DecodePreview(std::move(std::get<QByteArray>(result)), gamebanana);
                       !image.isNull()) {
                callback(std::move(image));
            } else {
                callback(MakeError(ErrorCode::InvalidImage));
            }
        });
}

Request* Client::Download(const Version& version, Callback<DownloadedFile> done,
                          Progress progress) {
    const QUrl url =
        IsLocalDownload(version) ? Endpoint(base_url, version.download_url.path()) : QUrl{};
    return DownloadFile(url, TransferKind::LocalArchive, version.size_bytes, version.sha256,
                        std::move(done),
                        std::move(progress));
}

Request* Client::Download(const ExternalFile& file, Callback<DownloadedFile> done,
                          Progress progress) {
    const QUrl url = !file.ambiguous_layout && file.name.endsWith(u".zip", Qt::CaseInsensitive) &&
                             file.id > 0 && file.download_url == GameBananaUrl("dl"_L1, file.id)
                         ? file.download_url
                         : QUrl{};
    return DownloadFile(url, TransferKind::GameBananaArchive, file.size_bytes, file.sha256,
                        std::move(done),
                        std::move(progress));
}

Request* Client::Download(const GitHubDownload& file, Callback<DownloadedFile> done,
                          Progress progress) {
    const bool valid = IsRawGitHubUrl(file.url) &&
                       (!file.patch.member.isEmpty() || file.size_bytes <= MaxGitHubPatchBytes);
    return DownloadFile(valid ? file.url : QUrl{}, TransferKind::GitHubPatch, file.size_bytes,
                        file.sha256, std::move(done), std::move(progress));
}

Request* Client::DownloadFile(const QUrl& url, TransferKind kind, qint64 size, const QByteArray& hash,
                              Callback<DownloadedFile> done, Progress progress) {
    std::optional<Error> error;
    if ((kind != TransferKind::GameBananaArchive || !hash.isEmpty()) &&
        hash.size() != QCryptographicHash::hashLength(QCryptographicHash::Sha256)) {
        error = MakeError(ErrorCode::MissingSha256);
    } else if (size <= 0) {
        error = MakeError(ErrorCode::InvalidInput);
    } else if (size > MaxArchiveBytes) {
        error = MakeError(ErrorCode::TooLarge);
    }
    return Start(
        url, kind,
        [callback = std::move(done)](Request::Result result) {
            if (auto* failure = std::get_if<Error>(&result)) {
                callback(std::move(*failure));
            } else {
                callback(std::move(std::get<DownloadedFile>(result)));
            }
        },
        std::move(error), size, hash, std::move(progress));
}

} // namespace ModStore
