// SPDX-FileCopyrightText: Copyright 2026 Lichii Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "lichii/mod_store/mod_store_dialog.h"

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

#include <QComboBox>
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDesktopServices>
#include <QDialog>
#include <QDir>
#include <QEvent>
#include <QFutureWatcher>
#include <QGridLayout>
#include <QHash>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPointer>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QScrollBar>
#include <QSignalBlocker>
#include <QSplitter>
#include <QTemporaryFile>
#include <QTimer>
#include <QToolButton>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QtConcurrentRun>

#include "common/scm_rev.h"
#include "lichii/mod_store/mod_store_client.h"
#include "lichii/mod_store/mod_store_install.h"

namespace ModStore {
namespace {

constexpr int PageSize = 50;
constexpr qint64 MaxArchiveBytes = 64 * 1024 * 1024;
constexpr int IdRole = Qt::UserRole;
constexpr int TitleRole = Qt::UserRole + 1;
const QString PrimaryButtonStyle = QStringLiteral(
    "QPushButton { background: #d52c5b; border: 1px solid #d52c5b; border-radius: 7px;"
    " color: white; font-weight: 600; padding: 0 12px; min-height: 30px; }"
    "QPushButton:hover { background: #bc1f4b; border-color: #bc1f4b; }"
    "QPushButton:pressed { background: #a61942; border-color: #a61942; }"
    "QPushButton:focus { border-color: palette(window-text); }");
const QString DarkPrimaryButtonStyle = QStringLiteral(
    "QPushButton { background: #f0386b; border-color: #f0386b; color: white; font-weight: 600;"
    " border-width: 1px; border-style: solid; border-radius: 7px; padding: 0 12px;"
    " min-height: 30px; }"
    "QPushButton:hover { background: #f45a85; border-color: #f45a85; }"
    "QPushButton:pressed { background: #d42c5a; border-color: #d42c5a; }"
    "QPushButton:focus { border-color: white; }");

enum class Compatibility { Match, Unknown, Mismatch };

std::optional<qulonglong> BuildOrdinal(const QString& tag) {
    static const QRegularExpression format(QStringLiteral("^build-[1-9][0-9]*$"));
    bool ok = false;
    const auto number = tag.mid(6).toULongLong(&ok);
    if (!ok || !format.match(tag).hasMatch()) {
        return std::nullopt;
    }
    return number;
}

bool KnownBuild(const QString& build) {
    static const QRegularExpression format(QStringLiteral("^[0-9A-Fa-f]{64}$"));
    return format.match(build).hasMatch() &&
           std::ranges::any_of(build, [](QChar digit) { return digit != QLatin1Char('0'); });
}

Compatibility MatchVersion(const QString& selected_title, const GameIdentity& game,
                           const Version& version) {
    if (selected_title != version.title_id ||
        (!game.display_version.isEmpty() && game.display_version != version.game_version) ||
        (KnownBuild(game.build_id) &&
         game.build_id.compare(version.build_id, Qt::CaseInsensitive) != 0)) {
        return Compatibility::Mismatch;
    }
    const auto running = BuildOrdinal(QString::fromLatin1(Common::g_build_release_tag));
    const auto required = BuildOrdinal(version.min_lichii_build);
    if (running && required && *running < *required) {
        return Compatibility::Mismatch;
    }
    return game.installed && !game.display_version.isEmpty() && KnownBuild(game.build_id) &&
                   running && required
               ? Compatibility::Match
               : Compatibility::Unknown;
}

Compatibility MatchGitHubBuild(const GameIdentity& game, const GitHubBuild& build) {
    if (!game.installed || !KnownBuild(game.build_id)) {
        return Compatibility::Unknown;
    }
    if (build.main) {
        return game.build_id.startsWith(build.id, Qt::CaseInsensitive) ? Compatibility::Match
                                                                      : Compatibility::Mismatch;
    }
    return build.id.size() == 64 && game.build_id.compare(build.id, Qt::CaseInsensitive) == 0
               ? Compatibility::Match
               : Compatibility::Unknown;
}

struct Offer {
    std::variant<Version, ExternalFile, GitHubBuild> file;
    QString version;
    Compatibility compatibility;
    bool supported;
};

QString ExternalVersionPrefix(qint64 id) {
    return QStringLiteral("%1-").arg(id);
}

QString ExternalVersion(qint64 id, const QByteArray& sha256) {
    return ExternalVersionPrefix(id) + QString::fromLatin1(sha256.toHex().left(16));
}

QLabel* PlainLabel(QWidget* parent, bool wrap = false) {
    auto* label = new QLabel(parent);
    label->setTextFormat(Qt::PlainText);
    label->setWordWrap(wrap);
    label->setTextInteractionFlags(Qt::TextSelectableByMouse);
    return label;
}

class ModResultsTree final : public QTreeWidget {
public:
    using QTreeWidget::QTreeWidget;

protected:
    bool viewportEvent(QEvent* event) override {
        switch (event->type()) {
        case QEvent::MouseButtonPress:
        case QEvent::MouseButtonDblClick:
        case QEvent::MouseMove: {
            const bool auto_scroll = hasAutoScroll();
            setAutoScroll(false);
            const bool handled = QTreeWidget::viewportEvent(event);
            setAutoScroll(auto_scroll);
            return handled;
        }
        default:
            return QTreeWidget::viewportEvent(event);
        }
    }
};

class Dialog final : public QDialog {
    Q_DECLARE_TR_FUNCTIONS(ModStore::Dialog)

public:
    Dialog(QWidget* parent, DialogContext ctx)
        : QDialog(parent), context(std::move(ctx)), client(context.base_url, this) {
        setWindowTitle(tr("Mod Store"));
        resize(1050, 760);
        setMinimumSize(730, 520);

        const auto border = dark ? QStringLiteral("#2a2b30") : QStringLiteral("#cbd0d6");
        const auto tree_style = QStringLiteral("QTreeWidget { border: 1px solid %1; }").arg(border);
        auto* root = new QVBoxLayout(this);
        root->setSpacing(8);
        auto* games_row = new QHBoxLayout;
        auto* toolbar = new QHBoxLayout;
        game_picker = new QComboBox(this);
        game_picker->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
        game_picker->setMinimumContentsLength(22);
        refresh = new QPushButton(tr("Refresh"), this);
        all_games = new QPushButton(tr("All games"), this);
        installed_mode = new QPushButton(tr("Installed"), this);
        installed_mode->setCheckable(true);
        games_row->addWidget(new QLabel(tr("Game:"), this));
        games_row->addWidget(game_picker);
        games_row->addWidget(all_games);
        games_row->addStretch();
        games_row->addWidget(refresh);
        games_row->addWidget(installed_mode);

        search = new QLineEdit(this);
        search->setPlaceholderText(tr("Enter a name, author, or description"));
        search->setClearButtonEnabled(true);
        search->setMaxLength(128);
        type = new QComboBox(this);
        type->addItem(tr("All types"));
        for (const auto category : {ModType::Graphics, ModType::Fps, ModType::Cheats,
                                    ModType::Translation, ModType::Gameplay}) {
            type->addItem(TypeName(category), static_cast<int>(category));
        }
        sort = new QComboBox(this);
        sort->addItem(tr("Recently updated"), static_cast<int>(Sort::Updated));
        sort->addItem(tr("Name A–Z"), static_cast<int>(Sort::Name));
        toolbar->addWidget(new QLabel(tr("Filter:"), this));
        toolbar->addWidget(search, 1);
        toolbar->addWidget(new QLabel(tr("Type:"), this));
        toolbar->addWidget(type);
        toolbar->addWidget(new QLabel(tr("Sort by:"), this));
        toolbar->addWidget(sort);
        root->addLayout(games_row);
        root->addLayout(toolbar);

        auto* split = new QSplitter(this);
        game_tree = new QTreeWidget(split);
        game_tree->setHeaderLabels({tr("Games"), tr("Mods")});
        game_tree->setRootIsDecorated(false);
        game_tree->setUniformRowHeights(true);
        game_tree->setStyleSheet(tree_style);
        game_tree->header()->setMinimumHeight(row_height);
        game_tree->header()->setStretchLastSection(false);
        game_tree->header()->setSectionResizeMode(0, QHeaderView::Stretch);
        game_tree->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);

        auto* middle = new QWidget(split);
        auto* middle_layout = new QVBoxLayout(middle);
        middle_layout->setContentsMargins(0, 0, 0, 0);
        results = new ModResultsTree(middle);
        results->setHeaderLabels({tr("Mod name"), tr("Type")});
        results->setRootIsDecorated(false);
        results->setUniformRowHeights(true);
        results->setAlternatingRowColors(true);
        results->setStyleSheet(tree_style);
        results->header()->setMinimumHeight(row_height);
        results->header()->setStretchLastSection(false);
        results->header()->setSectionResizeMode(0, QHeaderView::Stretch);
        results->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
        more = new QPushButton(tr("Load more"), middle);
        more->hide();
        middle_layout->addWidget(results, 1);
        middle_layout->addWidget(more);

        auto* inspector_frame = new QFrame(split);
        inspector_frame->setObjectName(QStringLiteral("modStoreInspector"));
        inspector_frame->setStyleSheet(
            QStringLiteral("QFrame#modStoreInspector { border: 1px solid %1; }").arg(border));
        auto* inspector_layout = new QVBoxLayout(inspector_frame);
        inspector_layout->setContentsMargins(1, 1, 1, 1);
        inspector_layout->setSpacing(0);
        inspector = new QScrollArea(inspector_frame);
        inspector->setFrameShape(QFrame::NoFrame);
        inspector->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        inspector->setWidgetResizable(true);
        inspector_layout->addWidget(inspector, 1);
        auto* pane = new QWidget(inspector);
        auto* detail = new QVBoxLayout(pane);
        detail->setContentsMargins(8, 8, 8, 8);
        detail->setSpacing(6);
        name = PlainLabel(pane, true);
        QFont heading = name->font();
        heading.setBold(true);
        heading.setPointSizeF(heading.pointSizeF() + 2);
        name->setFont(heading);
        byline = PlainLabel(pane, true);
        description = PlainLabel(pane, true);
        description_toggle = new QToolButton(pane);
        description_toggle->setCheckable(true);
        description_toggle->setAutoRaise(true);
        preview = PlainLabel(pane);
        preview->setAlignment(Qt::AlignCenter);
        preview->setFixedHeight(172);
        preview->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
        preview->setToolTip(tr("Mod preview"));
        preview_select = new QComboBox(pane);
        preview_select->hide();
        detail->addWidget(name);
        detail->addWidget(byline);
        detail->addWidget(description);
        detail->addWidget(description_toggle, 0, Qt::AlignLeft);
        target_picker = new QComboBox(pane);
        target_picker->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
        target_picker->setMinimumContentsLength(20);
        detail->addWidget(target_picker);
        release = new QComboBox(pane);
        release->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
        release->setMinimumContentsLength(20);
        auto* media_options = new QHBoxLayout;
        media_options->addWidget(release, 1);
        media_options->addWidget(preview_select);
        detail->addLayout(media_options);
        detail->addWidget(preview);

        compatibility_card = new QFrame(pane);
        compatibility_card->setObjectName(QStringLiteral("modStoreCompatibility"));
        compatibility_card->setStyleSheet(
            QStringLiteral("QFrame#modStoreCompatibility { background: %1; border: 1px solid %2;"
                           " border-radius: 6px; }")
                .arg(dark ? QStringLiteral("#1b1c20") : QStringLiteral("#f7f7f8"),
                     dark ? QStringLiteral("#34363c") : QStringLiteral("#cbd0d6")));
        auto* compatibility_layout = new QGridLayout(compatibility_card);
        compatibility_layout->setContentsMargins(9, 8, 9, 8);
        compatibility_layout->setHorizontalSpacing(12);
        compatibility_layout->setVerticalSpacing(4);
        compatibility_layout->setColumnStretch(1, 1);
        auto* caption = PlainLabel(compatibility_card);
        caption->setText(tr("Compatibility"));
        auto caption_font = caption->font();
        caption_font.setBold(true);
        caption->setFont(caption_font);
        compatibility_state = PlainLabel(compatibility_card);
        compatibility_state->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        compatibility_layout->addWidget(caption, 0, 0);
        compatibility_layout->addWidget(compatibility_state, 0, 1);
        const auto divider = [&](int row) {
            auto* rule = new QFrame(compatibility_card);
            rule->setFixedHeight(1);
            rule->setStyleSheet(dark ? QStringLiteral("background: #34363c;")
                                     : QStringLiteral("background: #cbd0d6;"));
            compatibility_layout->addWidget(rule, row, 0, 1, 2);
        };
        divider(1);
        for (size_t i = 0; i < compatibility_rows.size(); ++i) {
            auto& [key, value] = compatibility_rows[i];
            key = PlainLabel(compatibility_card, true);
            key->setStyleSheet(dark ? QStringLiteral("color: #9b9da4;")
                                    : QStringLiteral("color: #555c67;"));
            value = PlainLabel(compatibility_card, true);
            const auto row = static_cast<int>(i) + (i < 5 ? 2 : 3);
            compatibility_layout->addWidget(key, row, 0, Qt::AlignTop);
            compatibility_layout->addWidget(value, row, 1, Qt::AlignTop);
        }
        divider(7);
        catalog_note = PlainLabel(compatibility_card, true);
        catalog_note->setText(tr("Not verified by the catalog"));
        compatibility_layout->addWidget(catalog_note, 11, 0, 1, 2);
        github_provenance = PlainLabel(compatibility_card, true);
        compatibility_layout->addWidget(github_provenance, 12, 0, 1, 2);
        detail->addWidget(compatibility_card);

        detail->addStretch();
        inspector->setWidget(pane);

        auto* actions = new QWidget(inspector_frame);
        auto* actions_layout = new QVBoxLayout(actions);
        actions_layout->setContentsMargins(8, 4, 8, 8);
        actions_layout->setSpacing(6);
        inspector_layout->addWidget(actions);
        hint = PlainLabel(actions, true);
        mod_actions = new QHBoxLayout;
        install = new QPushButton(tr("Install"), actions);
        source = new QPushButton(actions);
        mod_actions->addWidget(install);
        mod_actions->addWidget(source);
        mod_actions->addStretch();
        actions_layout->addLayout(mod_actions);
        auto* installed_actions = new QHBoxLayout;
        enable = new QPushButton(actions);
        uninstall = new QPushButton(tr("Uninstall"), actions);
        installed_actions->addWidget(enable);
        installed_actions->addWidget(uninstall);
        installed_actions->addStretch();
        actions_layout->addLayout(installed_actions);
        actions_layout->addWidget(hint);
        split->setChildrenCollapsible(false);
        split->setSizes({208, 330, 500});
        split->setStretchFactor(2, 1);
        root->addWidget(split, 1);

        auto* footer = new QHBoxLayout;
        status = PlainLabel(this, true);
        close_button = new QPushButton(tr("Close"), this);
        footer->addWidget(status, 1);
        footer->addWidget(close_button);
        root->addLayout(footer);
        progress = new QDialog(this);
        progress->setWindowTitle(tr("Installing mod"));
        progress->setWindowModality(Qt::WindowModal);
        progress->setMinimumWidth(360);
        auto* progress_layout = new QVBoxLayout(progress);
        progress_label = PlainLabel(progress, true);
        progress_catalog_note = PlainLabel(progress, true);
        progress_catalog_note->setText(catalog_note->text());
        progress_bar = new QProgressBar(progress);
        cancel_button = new QPushButton(tr("Cancel"), progress);
        progress_layout->addWidget(progress_label);
        progress_layout->addWidget(progress_catalog_note);
        progress_layout->addWidget(progress_bar);
        progress_layout->addWidget(cancel_button, 0, Qt::AlignRight);
        for (auto* button : findChildren<QPushButton*>()) {
            button->setAutoDefault(false);
        }
        connect(cancel_button, &QPushButton::clicked, this, [this] { CancelDownload(); });
        connect(progress, &QDialog::rejected, this, [this] { CancelDownload(); });
        debounce.setSingleShot(true);
        debounce.setInterval(240);
        connect(&debounce, &QTimer::timeout, this, [this] { ReloadMods(); });
        connect(search, &QLineEdit::textChanged, this, [this] {
            more->hide();
            debounce.start();
        });
        const auto filters_changed = [this] {
            debounce.stop();
            ReloadMods();
        };
        connect(type, &QComboBox::currentIndexChanged, this, filters_changed);
        connect(sort, &QComboBox::currentIndexChanged, this, filters_changed);
        connect(refresh, &QPushButton::clicked, this, [this] {
            identities.clear();
            if (!installed_mode->isChecked()) {
                LoadGames();
            }
            ReloadMods();
        });
        connect(all_games, &QPushButton::clicked, this, [this] { SelectGame({}); });
        connect(installed_mode, &QPushButton::toggled, this, [this](bool installed) {
            type->setEnabled(!installed);
            sort->setEnabled(!installed);
            ReloadMods();
        });
        connect(game_picker, &QComboBox::currentIndexChanged, this,
                [this] { SelectGame(game_picker->currentData().toString()); });
        connect(game_tree, &QTreeWidget::currentItemChanged, this, [this](QTreeWidgetItem* item) {
            if (item) {
                SelectGame(item->data(0, IdRole).toString());
            }
        });
        connect(results, &QTreeWidget::currentItemChanged, this,
                [this](QTreeWidgetItem* item) { SelectMod(item); });
        connect(more, &QPushButton::clicked, this, [this] {
            if (next && !debounce.isActive() && !reload_after_install) {
                FetchMods(*next, list_epoch);
            }
        });
        connect(release, &QComboBox::currentIndexChanged, this, [this] { UpdateOffer(); });
        connect(target_picker, &QComboBox::currentIndexChanged, this, [this] {
            local_title = target_picker->currentData().toString();
            ReadLocal();
            BuildOffers();
        });
        connect(preview_select, &QComboBox::currentIndexChanged, this, [this] { LoadPreview(); });
        connect(description_toggle, &QToolButton::toggled, this, [this](bool expanded) {
            if (selected) {
                description_toggle->setText(expanded ? tr("Show less")
                                                     : tr("Show full description"));
                description->setText(expanded ? selected->description : description_excerpt);
            }
        });
        connect(source, &QPushButton::clicked, this, [this] { OpenSource(); });
        connect(install, &QPushButton::clicked, this, [this] { StartInstall(); });
        connect(enable, &QPushButton::clicked, this, [this] { ToggleEnabled(); });
        connect(uninstall, &QPushButton::clicked, this, [this] { RemoveInstalled(); });
        connect(close_button, &QPushButton::clicked, this, &Dialog::reject);

        title = context.initial_title_id;
        PopulateGames();
        ClearDetail();
        QTimer::singleShot(0, this, [this] {
            LoadGames();
            ReloadMods();
        });
    }

    bool Changed() const {
        return changed;
    }

    void reject() override {
        if (applying) {
            return;
        }
        if (busy) {
            CancelDownload();
            return;
        }
        QDialog::reject();
    }

private:
    static QString TypeName(ModType value) {
        switch (value) {
        case ModType::Graphics:
            return tr("Graphics");
        case ModType::Fps:
            return tr("FPS");
        case ModType::Cheats:
            return tr("Cheats");
        case ModType::Translation:
            return tr("Translation");
        case ModType::Gameplay:
            return tr("Gameplay");
        }
        return {};
    }

    static QString CompatibilityName(Compatibility value) {
        switch (value) {
        case Compatibility::Match:
            return tr("Match");
        case Compatibility::Unknown:
            return tr("Unverified");
        case Compatibility::Mismatch:
            return tr("Mismatch");
        }
        return {};
    }

    void Status(const QString& text) {
        status->setText(text);
    }

    GameIdentity Identity(const QString& title_id) {
        const auto found = identities.constFind(title_id);
        if (found != identities.cend()) {
            return found.value();
        }
        const auto identity = !title_id.isEmpty() && context.game_identity
                                  ? context.game_identity(title_id)
                                  : GameIdentity{};
        identities.insert(title_id, identity);
        return identity;
    }

    void PopulateGames() {
        const QSignalBlocker block_picker(game_picker);
        const QSignalBlocker block_tree(game_tree);
        game_picker->clear();
        game_tree->clear();
        const auto append = [&](const QString& id, const QString& game, const QString& count) {
            game_picker->addItem(game, id);
            auto* node = new QTreeWidgetItem(game_tree, {game, count});
            node->setSizeHint(0, QSize(0, row_height));
            node->setData(0, IdRole, id);
            node->setToolTip(0, game);
            node->setTextAlignment(1, Qt::AlignRight | Qt::AlignVCenter);
            if (id == title) {
                game_tree->setCurrentItem(node);
                game_picker->setCurrentIndex(game_picker->count() - 1);
            }
        };
        append({}, tr("All games"), {});
        for (const auto& game : games) {
            append(game.title_id, game.name, QString::number(game.mod_count));
        }
        if (!title.isEmpty() && game_picker->findData(title) < 0) {
            append(title, title, {});
        }
    }

    void LoadGames(int offset = 0) {
        if (offset == 0) {
            if (games_request) {
                games_request->Cancel();
            }
            ++game_epoch;
            incoming_games.clear();
        }
        const auto epoch = game_epoch;
        games_request = client.GetGames(offset, PageSize, [this, epoch, offset](auto response) {
            if (epoch != game_epoch) {
                return;
            }
            if (const auto* page = std::get_if<Page<Game>>(&response)) {
                incoming_games.append(page->items);
                if (page->next_offset && *page->next_offset > offset &&
                    incoming_games.size() < 1000) {
                    LoadGames(*page->next_offset);
                    return;
                }
                games = std::move(incoming_games);
                PopulateGames();
            }
        });
    }

    void SelectGame(const QString& id) {
        if (title == id) {
            return;
        }
        title = id;
        PopulateGames();
        ReloadMods();
    }

    void ReloadMods() {
        debounce.stop();
        ++list_epoch;
        if (list_request) {
            list_request->Cancel();
        }
        next.reset();
        more->hide();
        results->clear();
        catalog.clear();
        ClearDetail();
        if (installed_mode->isChecked()) {
            ShowInstalled();
            return;
        }
        list_filter = {title, search->text().trimmed(),
                       type->currentIndex() == 0
                           ? std::nullopt
                           : std::optional{static_cast<ModType>(type->currentData().toInt())},
                       static_cast<Sort>(sort->currentData().toInt())};
        FetchMods(0, list_epoch);
    }

    void FetchMods(int offset, int epoch) {
        more->setEnabled(false);
        Status(tr("Loading mods…"));
        list_request =
            client.GetMods(list_filter, offset, PageSize, [this, offset, epoch](auto response) {
                if (epoch != list_epoch) {
                    return;
                }
                if (const auto* error = std::get_if<Error>(&response)) {
                    Status(tr("Could not load the catalog: %1. Retry with Refresh; installed mods "
                              "are still available.")
                               .arg(error->message));
                    more->setEnabled(true);
                    return;
                }
                auto page = std::get<Page<Mod>>(std::move(response));
                for (auto& mod : page.items) {
                    if (catalog.contains(mod.id)) {
                        continue;
                    }
                    auto* item = new QTreeWidgetItem(results, {mod.name, TypeName(mod.type)});
                    item->setSizeHint(0, QSize(0, row_height));
                    item->setData(0, IdRole, mod.id);
                    item->setToolTip(0, mod.summary);
                    const auto key = mod.id;
                    catalog.insert(key, std::move(mod));
                }
                next = page.next_offset && *page.next_offset > offset ? page.next_offset
                                                                      : std::nullopt;
                more->setVisible(next.has_value() && !debounce.isActive() && !reload_after_install);
                more->setEnabled(true);
                Status(catalog.isEmpty()
                           ? tr("No matching mods. Change the game, type, or search filter.")
                           : tr("%n result(s)", nullptr, static_cast<int>(catalog.size())));
            });
    }

    void ShowInstalled() {
        QStringList titles{title};
        if (title.isEmpty()) {
            titles = QDir(context.mods_root).entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        }
        for (const auto& id : titles) {
            for (const auto& installed : ReadInstalledMods(context.mods_root, id)) {
                if (!search->text().trimmed().isEmpty() &&
                    !installed.name.contains(search->text().trimmed(), Qt::CaseInsensitive)) {
                    continue;
                }
                const auto game = std::ranges::find(games, id, &Game::title_id);
                auto* item = new QTreeWidgetItem(
                    results, {installed.name, game != games.cend() ? game->name : id});
                item->setSizeHint(0, QSize(0, row_height));
                item->setData(0, IdRole, installed.id);
                item->setData(0, TitleRole, id);
                item->setToolTip(0, installed.version);
            }
        }
        results->headerItem()->setText(1, tr("Game"));
        Status(results->topLevelItemCount() == 0 ? tr("No installed store mods match this filter.")
                                                 : tr("Installed mods are available offline."));
    }

    void ClearDetail() {
        ++mod_epoch;
        ++preview_epoch;
        if (mod_request) {
            mod_request->Cancel();
        }
        if (preview_request) {
            preview_request->Cancel();
        }
        selected.reset();
        offer_list.clear();
        local.reset();
        local_id.clear();
        local_title.clear();
        name->setText(tr("Select a mod to see its details"));
        byline->clear();
        description->clear();
        description_toggle->hide();
        inspector->verticalScrollBar()->setValue(0);
        preview->clear();
        preview->hide();
        preview_select->hide();
        compatibility_card->hide();
        target_picker->hide();
        release->hide();
        hint->clear();
        hint->hide();
        source->hide();
        install->hide();
        enable->hide();
        uninstall->hide();
        results->headerItem()->setText(1, installed_mode->isChecked() ? tr("Game") : tr("Type"));
    }

    void SelectMod(QTreeWidgetItem* item) {
        ClearDetail();
        if (!item) {
            return;
        }
        const auto id = item->data(0, IdRole).toString();
        local_title = item->data(0, TitleRole).toString();
        local_id = id;
        if (installed_mode->isChecked()) {
            ReadLocal();
            if (local) {
                name->setText(local->name);
                byline->setText(tr("Installed release: %1").arg(local->version));
                UpdateButtons();
            }
        } else {
            const auto& entry = catalog[id];
            name->setText(entry.name);
            byline->setText(entry.author);
            description->setText(entry.summary);
        }
        FetchDetail(id);
    }

    void FetchDetail(const QString& mod_id) {
        const auto epoch = mod_epoch;
        mod_request = client.GetMod(mod_id, [this, epoch](auto response) {
            if (epoch != mod_epoch) {
                return;
            }
            if (const auto* error = std::get_if<Error>(&response)) {
                hint->setText(tr("Could not load details: %1").arg(error->message));
                hint->show();
                return;
            }
            selected = std::get<Mod>(std::move(response));
            if (local_title.isEmpty()) {
                local_title = title;
                if (local_title.isEmpty()) {
                    const auto found =
                        std::ranges::find_if(selected->title_ids, [this](const QString& id) {
                            return Identity(id).installed;
                        });
                    local_title =
                        found == selected->title_ids.cend() ? selected->title_ids.value(0) : *found;
                }
            }
            {
                const QSignalBlocker block(target_picker);
                target_picker->clear();
                for (const auto& id : selected->title_ids) {
                    const auto game = std::ranges::find(games, id, &Game::title_id);
                    target_picker->addItem(game == games.cend() ? id : game->name, id);
                }
                target_picker->setCurrentIndex(target_picker->findData(local_title));
                target_picker->setVisible(title.isEmpty() && !installed_mode->isChecked() &&
                                          target_picker->count() > 1);
            }
            ReadLocal();
            name->setText(selected->name);
            const auto source_name = selected->source == Source::GameBanana ? tr("GameBanana")
                                     : selected->source == Source::GitHub   ? tr("GitHub")
                                     : selected->demo                       ? tr("Sample")
                                                                            : tr("Lichii catalog");
            byline->setText(QStringLiteral("%1 · %2 · %3")
                                .arg(selected->author, TypeName(selected->type), source_name));
            const auto full = selected->description.trimmed();
            description_excerpt = full.section(QLatin1Char('\n'), 0, 1).simplified();
            if (description_excerpt.size() > 200) {
                description_excerpt.truncate(200);
                if (description_excerpt.back().isHighSurrogate()) {
                    description_excerpt.chop(1);
                }
            }
            const bool truncated = description_excerpt != full.simplified();
            if (truncated) {
                description_excerpt = description_excerpt.trimmed() + QStringLiteral("…");
            }
            const QSignalBlocker description_signals(description_toggle);
            description_toggle->setChecked(false);
            description_toggle->setText(tr("Show full description"));
            description_toggle->setVisible(truncated);
            description->setText(truncated ? description_excerpt : full);
            source->setText(selected->github ? tr("View on GitHub") : tr("View on GameBanana"));
            source->setVisible(!selected->source_url.isEmpty());
            {
                const QSignalBlocker block(preview_select);
                preview_select->clear();
                if (selected->external) {
                    for (qsizetype i = 0; i < selected->external->preview_urls.size(); ++i) {
                        preview_select->addItem(tr("Preview %1").arg(i + 1));
                    }
                }
                preview_select->setVisible(preview_select->count() > 1);
            }
            LoadPreview();
            BuildOffers();
        });
    }

    void LoadPreview() {
        const auto epoch = ++preview_epoch;
        if (preview_request) {
            preview_request->Cancel();
        }
        if (!selected) {
            return;
        }
        auto mod = *selected;
        if (mod.external && preview_select->currentIndex() >= 0) {
            mod.preview_url = mod.external->preview_urls.value(preview_select->currentIndex());
        }
        preview->clear();
        preview->setVisible(mod.preview_url.isValid() && !mod.preview_url.isEmpty());
        if (preview->isHidden()) {
            return;
        }
        preview_request = client.GetPreview(mod, [this, epoch](auto response) {
            if (epoch != preview_epoch) {
                return;
            }
            if (const auto* image = std::get_if<QImage>(&response)) {
                preview->setPixmap(QPixmap::fromImage(
                    image->scaled(qMax(1, preview->width()), preview->height(), Qt::KeepAspectRatio,
                                  Qt::SmoothTransformation)));
            } else {
                preview->setText(tr("Preview unavailable"));
            }
        });
    }

    void BuildOffers() {
        const auto identity =
            selected->title_ids.contains(local_title) ? Identity(local_title) : GameIdentity{};
        const QSignalBlocker block(release);
        offer_list.clear();
        release->clear();
        int first_match = -1;
        if (selected->github) {
            int first_installable = -1;
            int first_unknown = -1;
            for (const auto& build : selected->github->builds) {
                const auto match = MatchGitHubBuild(identity, build);
                QString version;
                if (build.download) {
                    const auto& build_file = *build.download;
                    const QJsonArray selection{QString::fromLatin1(build_file.sha256.toHex()),
                                               static_cast<int>(build_file.patch.kind),
                                               build_file.patch.member, build_file.patch.tag};
                    const auto fingerprint = QCryptographicHash::hash(
                        QJsonDocument(selection).toJson(QJsonDocument::Compact),
                        QCryptographicHash::Sha256);
                    version = build.id.toLower() + u'-' + QString::fromLatin1(fingerprint.toHex());
                }
                offer_list.append({build, version, match, build.download.has_value()});
                const auto display = build.id.size() == 16
                                         ? build.id
                                         : QStringLiteral("%1…%2").arg(build.id.left(12),
                                                                      build.id.right(8));
                release->addItem(tr("%1 · %2").arg(display, CompatibilityName(match)));
                const auto index = release->count() - 1;
                release->setItemData(index, build.id, Qt::ToolTipRole);
                if (first_match < 0 && match == Compatibility::Match && build.download) {
                    first_match = index;
                } else if (first_installable < 0 && CanInstall(offer_list.back())) {
                    first_installable = index;
                }
                if (first_unknown < 0 && match != Compatibility::Mismatch) {
                    first_unknown = index;
                }
            }
            if (first_match < 0) {
                first_match = first_installable < 0 ? first_unknown : first_installable;
            }
        } else if (selected->external) {
            bool first_match_has_checksum = false;
            for (const auto& file : selected->external->files) {
                offer_list.append(
                    {file, {}, Compatibility::Unknown,
                     file.name.endsWith(QStringLiteral(".zip"), Qt::CaseInsensitive) &&
                         file.size_bytes <= MaxArchiveBytes});
                const auto entry =
                    tr("%1 · %2 · %3")
                        .arg(file.name, locale().formattedDataSize(file.size_bytes),
                             locale().toString(file.uploaded_at.date(), QLocale::ShortFormat));
                release->addItem(entry);
                release->setItemData(release->count() - 1, entry, Qt::ToolTipRole);
                if ((first_match < 0 || (!first_match_has_checksum && !file.sha256.isEmpty())) &&
                    CanInstall(offer_list.back())) {
                    first_match = release->count() - 1;
                    first_match_has_checksum = !file.sha256.isEmpty();
                }
            }
        } else {
            for (const auto& version : selected->versions) {
                const auto match = MatchVersion(local_title, identity, version);
                offer_list.append({version, version.id, match, true});
                release->addItem(
                    tr("%1 · %2 · %3")
                        .arg(version.id, version.game_version, CompatibilityName(match)));
                if (first_match < 0 && match == Compatibility::Match) {
                    first_match = release->count() - 1;
                }
            }
        }
        release->setCurrentIndex(first_match < 0 ? 0 : first_match);
        release->setVisible(release->count() > 0);
        UpdateOffer();
    }

    const Offer* CurrentOffer() const {
        const auto index = release->currentIndex();
        return index >= 0 && index < offer_list.size() ? &offer_list[index] : nullptr;
    }

    bool CanInstall(const Offer& offer) {
        if (!selected || selected->demo || !selected->title_ids.contains(local_title) ||
            !offer.supported || !Identity(local_title).installed ||
            offer.compatibility == Compatibility::Mismatch) {
            return false;
        }
        if (const auto* file = std::get_if<ExternalFile>(&offer.file)) {
            return !file->ambiguous_layout && (file->sha256.isEmpty() || file->sha256.size() == 32);
        }
        if (const auto* github = std::get_if<GitHubBuild>(&offer.file)) {
            return github->download.has_value();
        }
        return offer.compatibility == Compatibility::Match &&
               std::get<Version>(offer.file).sha256.size() == 32;
    }

    void UpdateOffer() {
        if (!selected) {
            return;
        }
        const auto identity =
            selected->title_ids.contains(local_title) ? Identity(local_title) : GameIdentity{};
        const auto* offer = CurrentOffer();
        const auto unknown = tr("Unknown");
        const auto short_build = [](const QString& build) {
            return build.size() > 16 ? QStringLiteral("%1…%2").arg(build.left(12), build.right(8))
                                     : build;
        };
        const auto set_row = [&](size_t row, const QString& key, const QString& value,
                                 const QString& tooltip = {}) {
            const auto [left, right] = compatibility_rows[row];
            left->setText(key);
            right->setText(value);
            right->setToolTip(tooltip);
            left->setVisible(!key.isEmpty());
            right->setVisible(!key.isEmpty());
        };
        const auto game = std::ranges::find(games, local_title, &Game::title_id);
        set_row(0, tr("Catalog game"), game == games.cend() ? unknown : game->name);
        set_row(1, tr("Title ID"), local_title);
        set_row(2, tr("Target game version"), unknown);
        set_row(3, tr("Target build ID"), unknown);
        set_row(4, tr("Minimum Lichii"), unknown);
        set_row(5, tr("Installed version"),
                identity.display_version.isEmpty()
                    ? (identity.installed ? unknown : tr("Not in your library"))
                    : identity.display_version);
        set_row(6, tr("Installed build ID"),
                KnownBuild(identity.build_id) ? short_build(identity.build_id) : unknown,
                KnownBuild(identity.build_id) ? identity.build_id : QString{});
        set_row(7, tr("Running Lichii"),
                Common::g_build_release_tag[0] ? QString::fromLatin1(Common::g_build_release_tag)
                                               : tr("Release number unavailable"));
        const auto* external = offer ? std::get_if<ExternalFile>(&offer->file) : nullptr;
        catalog_note->setVisible(external && external->sha256.isEmpty());
        auto match = offer ? offer->compatibility : Compatibility::Unknown;
        github_provenance->setVisible(selected->github.has_value());
        if (selected->github) {
            const auto& github = *selected->github;
            set_row(2, {}, {});
            set_row(4, {}, {});
            if (const auto* selected_build = offer ? std::get_if<GitHubBuild>(&offer->file) : nullptr) {
                const auto& build = *selected_build;
                set_row(3, build.main && build.id.size() == 16 ? tr("Main build ID prefix")
                           : build.main                       ? tr("Main build ID")
                                                              : tr("Patch build ID"),
                        short_build(build.id), build.id);
            }
            github_provenance->setText(
                tr("Source: %1 @ %2\nLicense: Not declared")
                    .arg(github.repository, github.commit.left(12)));
            github_provenance->setToolTip(github.commit);
        }
        if (offer || selected->github) {
            const auto color = match == Compatibility::Match
                                   ? (dark ? QStringLiteral("#79d798") : QStringLiteral("#25623b"))
                               : match == Compatibility::Mismatch
                                   ? (dark ? QStringLiteral("#ff9595") : QStringLiteral("#a12828"))
                                   : (dark ? QStringLiteral("#f1bf64") : QStringLiteral("#795100"));
            compatibility_state->setStyleSheet(QStringLiteral("color: %1;").arg(color));
            compatibility_state->setText(
                QStringLiteral("●  %1").arg(CompatibilityName(match)));
        }
        if (offer) {
            if (const auto* version = std::get_if<Version>(&offer->file)) {
                set_row(1, tr("Title ID"), version->title_id);
                set_row(2, tr("Target game version"), version->game_version);
                set_row(3, tr("Target build ID"), short_build(version->build_id),
                        version->build_id);
                set_row(4, tr("Minimum Lichii"), version->min_lichii_build);
            } else if (const auto* file = std::get_if<ExternalFile>(&offer->file)) {
                QString visible_builds = unknown;
                if (!file->build_ids.isEmpty()) {
                    visible_builds = short_build(file->build_ids.front());
                    if (file->build_ids.size() > 1) {
                        visible_builds = tr("%1 and %n more", nullptr,
                                            static_cast<int>(file->build_ids.size() - 1))
                                             .arg(visible_builds);
                    }
                }
                set_row(3, tr("Patch build IDs"), visible_builds,
                        file->build_ids.join(QLatin1Char('\n')));
            }
        }
        compatibility_state->setVisible(offer != nullptr || selected->github.has_value());
        compatibility_card->show();
        UpdateButtons();
    }

    void ReadLocal() {
        local.reset();
        if (local_title.isEmpty() || local_id.isEmpty()) {
            return;
        }
        const auto installed = ReadInstalledMods(context.mods_root, local_title);
        const auto it = std::ranges::find(installed, local_id, &InstalledMod::id);
        if (it != installed.cend()) {
            local = *it;
        }
    }

    void UpdateButtons() {
        const auto* offer = CurrentOffer();
        const auto* external = offer ? std::get_if<ExternalFile>(&offer->file) : nullptr;
        const bool installed = local.has_value();
        const bool same_file =
            external && installed && local->version.startsWith(ExternalVersionPrefix(external->id));
        const bool reinstall = same_file && external->sha256.isEmpty();
        const bool current =
            offer && installed &&
            (external ? same_file && !reinstall &&
                            (local->sha256.isEmpty()
                                 ? local->version == ExternalVersion(external->id, external->sha256)
                                 : local->sha256 == external->sha256)
                      : local->version == offer->version);
        install->setVisible(offer && (!selected->github || offer->supported));
        install->setText(reinstall ? tr("Reinstall") : installed ? tr("Update") : tr("Install"));
        install->setEnabled(!busy && offer && !current && CanInstall(*offer));
        source->setEnabled(!busy);
        const auto& primary = dark ? DarkPrimaryButtonStyle : PrimaryButtonStyle;
        const bool install_primary = install->isEnabled();
        const bool source_primary = !install_primary && !source->isHidden() && !busy;
        install->setStyleSheet(install_primary ? primary : QString{});
        source->setStyleSheet(source_primary ? primary : QString{});
        if (mod_actions->indexOf(source_primary ? source : install) != 0) {
            auto* first = source_primary ? source : install;
            mod_actions->removeWidget(first);
            mod_actions->insertWidget(0, first);
        }
        enable->setVisible(installed);
        uninstall->setVisible(installed);
        enable->setEnabled(!busy && static_cast<bool>(context.set_enabled));
        uninstall->setEnabled(!busy);
        const bool enabled =
            installed && (!context.is_enabled || context.is_enabled(local_title, local_id));
        enable->setText(enabled ? tr("Disable") : tr("Enable"));
        hint->clear();
        if (selected && selected->github && (!offer || !offer->supported)) {
            hint->setText(tr("No verified download is available for this patch. "
                             "Open GitHub for the files and instructions."));
        } else if (offer && selected) {
            if (selected->demo) {
                hint->setText(tr("This sample is not available for installation."));
            } else if (external && external->ambiguous_layout) {
                hint->setText(tr("This ZIP requires choosing components. Open GameBanana to "
                                 "install it manually."));
            } else if (!selected->title_ids.contains(local_title)) {
                hint->setText(tr("This listing does not target the selected game."));
            } else if (offer->compatibility == Compatibility::Mismatch) {
                hint->setText(
                    tr("This release does not match the installed game or Lichii build."));
            } else if (!offer->supported) {
                hint->setText(tr("Only ZIP files up to 64 MiB can be installed. Open GameBanana "
                                 "for this file."));
            } else if (!Identity(local_title).installed) {
                hint->setText(tr("Add this game to your library to install its mods."));
            } else if (!CanInstall(*offer)) {
                hint->setText(tr("The installed game or Lichii build could not be verified."));
            } else if (current || reinstall) {
                hint->setText(enabled ? tr("Installed and enabled") : tr("Installed and disabled"));
            } else {
                hint->setText(installed ? tr("A different release is available.") : QString{});
            }
        } else if (installed) {
            hint->setText(enabled ? tr("Installed and enabled") : tr("Installed and disabled"));
        }
        hint->setVisible(!hint->text().isEmpty());
    }

    void OpenSource() {
        if (selected && !selected->source_url.isEmpty()) {
            QDesktopServices::openUrl(selected->source_url);
        }
    }

    void InstallFailed(const QString& error) {
        Status(error);
        if (!selected || selected->source_url.isEmpty()) {
            return;
        }
        const QUrl source_url = selected->source_url;
        QMessageBox prompt(QMessageBox::Warning, tr("Could not install the mod"), error,
                           QMessageBox::Close, this);
        auto* open = prompt.addButton(source->text(), QMessageBox::ActionRole);
        prompt.exec();
        if (prompt.clickedButton() == open && !source_url.isEmpty()) {
            QDesktopServices::openUrl(source_url);
        }
    }

    void SetProgress(const QString& text, int percent = -1, bool can_cancel = false) {
        if (text.isEmpty()) {
            progress->hide();
            return;
        }
        progress_label->setText(text);
        cancel_button->setVisible(can_cancel);
        progress_bar->setRange(0, percent < 0 ? 0 : 100);
        if (percent >= 0) {
            progress_bar->setValue(percent);
        }
        progress->show();
    }

    void SetBusy(bool value) {
        busy = value;
        for (QWidget* widget :
             {static_cast<QWidget*>(game_picker), static_cast<QWidget*>(refresh),
              static_cast<QWidget*>(all_games), static_cast<QWidget*>(preview_select),
              static_cast<QWidget*>(installed_mode), static_cast<QWidget*>(search),
              static_cast<QWidget*>(game_tree), static_cast<QWidget*>(results),
              static_cast<QWidget*>(more), static_cast<QWidget*>(release),
              static_cast<QWidget*>(target_picker), static_cast<QWidget*>(close_button)}) {
            widget->setEnabled(!value);
        }
        type->setEnabled(!value && !installed_mode->isChecked());
        sort->setEnabled(!value && !installed_mode->isChecked());
        UpdateButtons();
    }

    void StartInstall() {
        const auto* offer = CurrentOffer();
        if (!offer || !CanInstall(*offer) || busy) {
            return;
        }
        const auto chosen = *offer;
        const auto* external_file = std::get_if<ExternalFile>(&chosen.file);
        const auto* github = std::get_if<GitHubBuild>(&chosen.file);
        const auto patch = github ? std::optional{github->download->patch} : std::nullopt;
        const bool external = external_file != nullptr;
        const bool unverified = external && external_file->sha256.isEmpty();
        const auto label = selected->name;
        const auto epoch = mod_epoch;
        if ((external || patch) && !unverified && chosen.compatibility == Compatibility::Unknown &&
            QMessageBox::question(
                this, tr("Compatibility unverified"),
                tr("Lichii could not confirm that this file works with your game version. "
                   "It will check the download's SHA-256 before installing. Continue?"),
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No) != QMessageBox::Yes) {
            return;
        }
        if (epoch != mod_epoch) {
            return;
        }
        reload_after_install = debounce.isActive();
        debounce.stop();
        SetBusy(true);
        progress_catalog_note->setVisible(unverified);
        SetProgress(tr("Downloading %1…").arg(label), 0, true);
        const auto generation = ++install_epoch;
        const auto version = chosen.version;
        const auto file_id = external ? external_file->id : 0;
        auto receive = [this, generation, version, file_id, label, external, patch](auto response) {
            if (generation != install_epoch) {
                return;
            }
            if (const auto* error = std::get_if<Error>(&response)) {
                FinishOperation();
                InstallFailed(tr("Could not download the mod: %1").arg(error->message));
                return;
            }
            auto received = std::get<DownloadedFile>(std::move(response));
            auto owned = std::shared_ptr<QTemporaryFile>(std::move(received.file));
            ApplyDownloaded(owned, external ? ExternalVersion(file_id, received.sha256) : version,
                            received.sha256, label, external, patch);
        };
        auto report = [this, generation, unverified](qint64 received, qint64 total) {
            if (generation != install_epoch || total <= 0) {
                return;
            }
            if (received >= total) {
                SetProgress(unverified ? tr("Checking the download…") : tr("Verifying SHA-256…"));
            } else {
                SetProgress(tr("Downloading %1 of %2 KiB…").arg(received / 1024).arg(total / 1024),
                            static_cast<int>(100 * received / total), true);
            }
        };
        download = std::visit(
            [&](const auto& file) {
                if constexpr (std::is_same_v<std::decay_t<decltype(file)>, GitHubBuild>) {
                    return client.Download(*file.download, std::move(receive), std::move(report));
                } else {
                    return client.Download(file, std::move(receive), std::move(report));
                }
            },
            chosen.file);
    }

    void ApplyDownloaded(const std::shared_ptr<QTemporaryFile>& file, const QString& version,
                         const QByteArray& digest, const QString& label, bool external,
                         const std::optional<GitHubPatch>& patch) {
        SetProgress(patch ? tr("Checking and installing the patch…")
                          : tr("Checking the ZIP and installing the mod…"));
        applying = true;
        const auto root = context.mods_root;
        const auto target = local_title;
        const auto id = local_id;
        auto* install_task = new QFutureWatcher<QString>(this);
        connect(install_task, &QFutureWatcher<QString>::finished, this, [this, install_task] {
            const auto error = install_task->result();
            install_task->deleteLater();
            if (error.isEmpty()) {
                changed = true;
                ReadLocal();
            }
            FinishOperation();
            if (error.isEmpty()) {
                Status(tr("Installation complete. The mod is available in Add-Ons."));
            } else {
                InstallFailed(error);
            }
        });
        install_task->setFuture(QtConcurrent::run([root, target, id, label, version, file, digest,
                                                   external, patch] {
            if (patch) {
                return InstallGitHubPatch(root, target, id, label, version, file->fileName(), digest,
                                          *patch);
            }
            return InstallMod(root, target, id, label, version, file->fileName(), digest, external);
        }));
    }

    void FinishOperation() {
        applying = false;
        SetProgress({});
        SetBusy(false);
        if (reload_after_install) {
            reload_after_install = false;
            debounce.start();
        }
    }

    void CancelDownload() {
        if (applying || !busy) {
            return;
        }
        ++install_epoch;
        if (download) {
            download->Cancel();
        }
        FinishOperation();
        Status(tr("Download canceled. No add-ons were changed."));
    }

    void ToggleEnabled() {
        if (!local || !context.set_enabled || busy) {
            return;
        }
        const bool before = context.is_enabled && context.is_enabled(local_title, local_id);
        if (!context.set_enabled(local_title, local_id, !before)) {
            Status(tr("Could not save the change to this mod."));
            return;
        }
        changed = true;
        UpdateButtons();
        Status(before ? tr("Mod disabled.") : tr("Mod enabled."));
    }

    void RemoveInstalled() {
        if (!local || busy) {
            return;
        }
        const auto epoch = mod_epoch;
        if (QMessageBox::question(this, tr("Uninstall mod"), tr("Uninstall %1?").arg(local->name),
                                  QMessageBox::Yes | QMessageBox::No,
                                  QMessageBox::No) != QMessageBox::Yes ||
            epoch != mod_epoch) {
            return;
        }
        const auto error = UninstallMod(context.mods_root, local_title, local_id);
        if (!error.isEmpty()) {
            Status(error);
            return;
        }
        const bool preference_reset =
            !context.set_enabled || context.set_enabled(local_title, local_id, true);
        changed = true;
        local.reset();
        if (installed_mode->isChecked()) {
            ReloadMods();
        } else {
            UpdateButtons();
        }
        Status(preference_reset
                   ? tr("Mod uninstalled.")
                   : tr("Mod uninstalled, but Lichii could not clear its saved disabled state."));
    }

    DialogContext context;
    Client client;
    QTimer debounce;
    QList<Game> games;
    QList<Game> incoming_games;
    QHash<QString, Mod> catalog;
    QHash<QString, GameIdentity> identities;
    QList<Offer> offer_list;
    ModFilter list_filter{};
    std::optional<int> next;
    std::optional<Mod> selected;
    std::optional<InstalledMod> local;
    QString description_excerpt;
    QString title;
    QString local_title;
    QString local_id;
    QPointer<Request> games_request;
    QPointer<Request> list_request;
    QPointer<Request> mod_request;
    QPointer<Request> preview_request;
    QPointer<Request> download;
    int game_epoch{};
    int list_epoch{};
    int mod_epoch{};
    int preview_epoch{};
    int install_epoch{};
    bool changed{};
    bool busy{};
    bool applying{};
    bool reload_after_install{};
    const bool dark = palette().color(QPalette::Window).lightness() < 128;
    const int row_height = qMax(28, fontMetrics().height() + 10);
    QComboBox* game_picker{};
    QComboBox* type{};
    QComboBox* sort{};
    QComboBox* release{};
    QComboBox* target_picker{};
    QComboBox* preview_select{};
    QLineEdit* search{};
    QTreeWidget* game_tree{};
    QTreeWidget* results{};
    QScrollArea* inspector{};
    QLabel* name{};
    QLabel* byline{};
    QLabel* description{};
    QToolButton* description_toggle{};
    QLabel* preview{};
    QLabel* compatibility_state{};
    QLabel* catalog_note{};
    QLabel* github_provenance{};
    QLabel* hint{};
    QLabel* status{};
    QFrame* compatibility_card{};
    std::array<std::pair<QLabel*, QLabel*>, 8> compatibility_rows{};
    QHBoxLayout* mod_actions{};
    QDialog* progress{};
    QLabel* progress_label{};
    QLabel* progress_catalog_note{};
    QProgressBar* progress_bar{};
    QPushButton* cancel_button{};
    QPushButton* all_games{};
    QPushButton* refresh{};
    QPushButton* installed_mode{};
    QPushButton* more{};
    QPushButton* install{};
    QPushButton* source{};
    QPushButton* enable{};
    QPushButton* uninstall{};
    QPushButton* close_button{};
};

} // namespace

bool ShowDialog(QWidget* parent, DialogContext context) {
    Dialog dialog(parent, std::move(context));
    dialog.exec();
    return dialog.Changed();
}

} // namespace ModStore
