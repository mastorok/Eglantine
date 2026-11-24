// SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <variant>
#include <QByteArray>
#include <QDateTime>
#include <QImage>
#include <QList>
#include <QObject>
#include <QString>
#include <QStringList>
#include <QTemporaryFile>
#include <QUrl>

#include "lichii/mod_store/mod_store_install.h"

class QNetworkAccessManager;

namespace ModStore {

enum class ModType { Graphics, Fps, Cheats, Translation, Gameplay };
enum class Sort { Updated, Name };
enum class Source { Demo, Submission, GameBanana, GitHub };

struct Game {
    QString title_id;
    QString name;
    int mod_count{};
};

struct Version {
    QString id;
    QString title_id;
    QString game_version;
    QString build_id;
    QString min_lichii_build;
    QUrl download_url;
    QByteArray sha256;
    qint64 size_bytes{};
    qint64 unpacked_size_bytes{};
};

struct ExternalFile {
    qint64 id{};
    QString name;
    qint64 size_bytes{};
    QDateTime uploaded_at;
    QUrl download_url;
    QByteArray sha256;
    QStringList build_ids;
    bool ambiguous_layout{};
};

struct Credit {
    QString name;
    QString role;
    QUrl url;
};

struct External {
    qint64 id{};
    QString category;
    QList<Credit> credits;
    QList<QUrl> preview_urls;
    QList<ExternalFile> files;
};

struct GitHubDownload {
    QUrl url;
    QByteArray sha256;
    qint64 size_bytes;
    GitHubPatch patch;
};

struct GitHubBuild {
    QString id;
    bool main{};
    std::optional<GitHubDownload> download;
};

struct GitHubListing {
    QString repository;
    QString commit;
    QList<GitHubBuild> builds;
};

struct Mod {
    QString id;
    QStringList title_ids;
    QString name;
    QString summary;
    QString description;
    QString author;
    ModType type{};
    Source source{};
    bool demo{};
    QUrl source_url;
    QUrl preview_url;
    QDateTime updated_at;
    QList<Version> versions;
    std::optional<External> external;
    std::optional<GitHubListing> github;
};

struct ModFilter {
    QString title_id;
    QString query;
    std::optional<ModType> type;
    Sort sort = Sort::Updated;
};

template <typename T>
struct Page {
    QList<T> items;
    std::optional<int> next_offset;
};

enum class ErrorCode {
    InvalidUrl,
    InvalidInput,
    InvalidResponse,
    InvalidImage,
    MissingSha256,
    SizeMismatch,
    IntegrityMismatch,
    TooLarge,
    Http,
    Network,
    File,
    TimedOut,
    Cancelled,
    Busy,
};

struct Error {
    ErrorCode code;
    QString message;
    int http_status{};
};

template <typename T>
using Callback = std::function<void(std::variant<T, Error>)>;

using Progress = std::function<void(qint64 received, qint64 total)>;

struct DownloadedFile {
    std::unique_ptr<QTemporaryFile> file;
    QByteArray sha256;
};

class Client;

class Request final : public QObject {
public:
    void Cancel();

private:
    struct Transfer;
    using Result = std::variant<QByteArray, DownloadedFile, Error>;
    using QObject::QObject;
    ~Request() override;

    void Begin();
    void Get(const QUrl& url);
    bool CheckHeaders();
    void Drain();
    void Complete();
    void Finish(Result result);
    void ReleaseReply();

    std::unique_ptr<Transfer> transfer;
    friend class Client;
};

class Client final : public QObject {
public:
    explicit Client(const QString& base_url, QObject* parent = nullptr);
    static bool IsValidBaseUrl(const QString& base_url);

    Request* GetHealth(Callback<bool> done);
    Request* GetGames(int offset, int limit, Callback<Page<Game>> done);
    Request* GetMods(const ModFilter& filter, int offset, int limit, Callback<Page<Mod>> done);
    Request* GetMod(const QString& id, Callback<Mod> done);
    Request* GetPreview(const Mod& mod, Callback<QImage> done);
    Request* Download(const Version& version, Callback<DownloadedFile> done,
                      Progress progress = {});
    Request* Download(const ExternalFile& file, Callback<DownloadedFile> done,
                      Progress progress = {});
    Request* Download(const GitHubDownload& file, Callback<DownloadedFile> done,
                      Progress progress = {});

private:
    enum class TransferKind { Json, Preview, LocalArchive, GameBananaArchive, GitHubPatch };

    Request* Start(const QUrl& url, TransferKind kind, std::function<void(Request::Result)> done,
                   std::optional<Error> error = {}, qint64 size = 0, QByteArray hash = {},
                   Progress progress = {});
    Request* DownloadFile(const QUrl& url, TransferKind kind, qint64 size, const QByteArray& hash,
                          Callback<DownloadedFile> done, Progress progress);

    const QUrl base_url;
    QNetworkAccessManager* const network;
    int active_requests{};
    bool active_download{};
    friend class Request;
};

} // namespace ModStore
