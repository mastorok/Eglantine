// SPDX-FileCopyrightText: Copyright 2026 Lichii Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <QByteArray>
#include <QList>
#include <QString>
#include <QStringView>

namespace ModStore {

struct InstalledMod {
    QString id;
    QString version;
    QString name;
    QByteArray sha256;
};

enum class GitHubPatchKind { Pchtxt, Ips, Cheat };

struct GitHubPatch {
    GitHubPatchKind kind;
    QString build_id;
    QString member;
    QString tag;
};

[[nodiscard]] bool MatchesGitHubIpsFilename(const QString& filename, QStringView build_id);
[[nodiscard]] bool IsCheatTag(const QString& tag);

[[nodiscard]] QString InstallMod(const QString& root, const QString& title_id,
                                 const QString& mod_id, const QString& name, const QString& version,
                                 const QString& archive_path, const QByteArray& sha256,
                                 bool external);
[[nodiscard]] QString InstallGitHubPatch(const QString& root, const QString& title_id,
                                         const QString& mod_id, const QString& name,
                                         const QString& version, const QString& archive_path,
                                         const QByteArray& sha256, const GitHubPatch& patch);
[[nodiscard]] QString UninstallMod(const QString& root, const QString& title_id,
                                   const QString& mod_id);
[[nodiscard]] QList<InstalledMod> ReadInstalledMods(const QString& root, const QString& title_id);

} // namespace ModStore
