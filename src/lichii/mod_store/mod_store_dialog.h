// SPDX-FileCopyrightText: Copyright 2026 Lichii Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <QString>

class QWidget;

namespace ModStore {

struct GameIdentity {
    bool installed{};
    QString display_version;
    QString build_id;
};

struct DialogContext {
    QString base_url;
    QString initial_title_id;
    QString mods_root;
    std::function<GameIdentity(const QString&)> game_identity;
    std::function<bool(const QString&, const QString&)> is_enabled;
    std::function<bool(const QString&, const QString&, bool)> set_enabled;
};

bool ShowDialog(QWidget* parent, DialogContext context);

} // namespace ModStore
