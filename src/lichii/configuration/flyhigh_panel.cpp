// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/httplib.h"

#include <array>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <QApplication>
#include <QCryptographicHash>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressDialog>
#include <QPushButton>
#include <QSaveFile>
#include <QScopedValueRollback>
#include <QUrl>
#include <QVBoxLayout>
#include <fmt/format.h>

#include "common/common_types.h"
#include "common/fs/fs_util.h"
#include "common/logging.h"
#include "common/net/ca_store.h"
#include "lichii/configuration/flyhigh_panel.h"
#include "video_core/frame_gen/flyhigh_model.h"

namespace {

namespace FrameGen = VideoCore::FrameGen;
namespace Model = VideoCore::FrameGen::FlyHighModel;

constexpr std::string_view WEIGHTS_LICENSE = R"license(# FlyHighFG model weights: license

The model weights in this directory, `flyhighfg.bin` (f16 blob) and its `flyhighfg.json` manifest (`graph.json` is
the engine's op graph), are dual licensed.

## Noncommercial use

**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**:
<https://creativecommons.org/licenses/by-nc/4.0/>

Free to use, share and adapt for noncommercial purposes with attribution. This covers personal use, research,
education, open-source projects that are not sold and products that are not monetised.

## Commercial use

A product that ships these weights, or a service that runs them, and that is sold, monetised or used in a commercial
offering needs a commercial license. It is granted by agreement with the maintainers: open an issue in the repository
or write to the maintainers through the GitHub organisation. The engine code needs no such license.

## The code

The engine **code** is licensed separately under the Apache License 2.0 (see `../LICENSE`), commercially too; that
license does not cover these weights. A model you trained or converted yourself is yours to license as you like.
)license";

QString Tr(const char* text) {
    return QCoreApplication::translate("FlyHighPanel", text);
}

QString PathText(const std::filesystem::path& path) {
    return QString::fromStdString(Common::FS::PathToUTF8String(path));
}

QString Text(std::string_view text) {
    return QString::fromUtf8(text.data(), static_cast<qsizetype>(text.size()));
}

u64 DownloadSize() {
    u64 total = 0;
    for (const Model::File& file : Model::FILES) {
        total += file.size;
    }
    return total;
}

std::optional<std::string> Fetch(const Model::File& file, std::string& error,
                                 const std::function<bool(u64)>& progress) {
    httplib::Client client{std::string{Model::DOWNLOAD_HOST}};
    client.set_connection_timeout(30);
    client.set_read_timeout(30);
    client.set_max_timeout(60'000);
    client.set_payload_max_length(static_cast<size_t>(file.size));
    client.set_decompress(false);
    Common::Net::ConfigureCaStore(client);
    std::string body;
    body.reserve(static_cast<size_t>(file.size));
    const std::string mismatch = fmt::format("{} doesn't match the pinned model", file.name);
    const auto result = client.Get(
        fmt::format("{}{}", Model::DOWNLOAD_PATH, file.name), {{"Accept-Encoding", "identity"}},
        [&](const httplib::Response& response) {
            if (response.status != 200) {
                error = fmt::format("{}: HTTP {}", file.name, response.status);
                return false;
            }
            if (response.has_header("Content-Length") &&
                response.get_header_value_u64("Content-Length") != file.size) {
                error = mismatch;
                return false;
            }
            return true;
        },
        [&](const char* data, size_t length) {
            if (length > file.size - body.size()) {
                error = mismatch;
                return false;
            }
            body.append(data, length);
            return progress(body.size());
        });
    if (!result) {
        if (error.empty()) {
            error = fmt::format("{}: {}", file.name, httplib::to_string(result.error()));
        }
        return std::nullopt;
    }
    if (result->status != 200) {
        error = fmt::format("{}: HTTP {}", file.name, result->status);
        return std::nullopt;
    }
    if (body.size() != file.size ||
        QCryptographicHash::hash(QByteArrayView{body}, QCryptographicHash::Sha256).toHex() !=
            QByteArrayView{file.sha256}) {
        error = mismatch;
        return std::nullopt;
    }
    return body;
}

bool Save(const std::filesystem::path& path, std::string_view contents, std::string& error) {
    QSaveFile file(PathText(path));
    const auto size = static_cast<qint64>(contents.size());
    if (!file.open(QIODevice::WriteOnly) || file.write(contents.data(), size) != size ||
        !file.commit()) {
        error = fmt::format("could not save {}: {}", Common::FS::PathToUTF8String(path),
                            file.errorString().toStdString());
        return false;
    }
    return true;
}

} // Anonymous namespace

FlyHighPanel::FlyHighPanel(QWidget* parent) : QWidget(parent) {
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(24, 0, 0, 8);
    layout->setSpacing(6);

    QHBoxLayout* status_row = new QHBoxLayout;
    status_row->setSpacing(8);
    icon_label = new QLabel(this);
    icon_label->setTextFormat(Qt::RichText);
    icon_label->setAlignment(Qt::AlignTop | Qt::AlignHCenter);
    icon_label->setFixedWidth(16);
    status_label = new QLabel(this);
    status_label->setTextFormat(Qt::RichText);
    status_label->setWordWrap(true);
    status_row->addWidget(icon_label);
    status_row->addWidget(status_label, 1);

    QHBoxLayout* button_row = new QHBoxLayout;
    button_row->setContentsMargins(24, 0, 0, 0);
    download_button = new QPushButton(this);
    folder_button = new QPushButton(Tr("Open Folder"), this);
    QLabel* about = new QLabel(QStringLiteral("<a href=\"%1\">%2</a>")
                                   .arg(Text(Model::RELEASE_PAGE), Tr("About FlyHigh")),
                               this);
    about->setOpenExternalLinks(true);
    about->setToolTip(Tr("The FlyHigh engine on GitHub. The engine is Apache-2.0. The weights "
                         "are CC BY-NC 4.0 for noncommercial use and require a separate licence "
                         "for commercial use."));
    button_row->addWidget(download_button);
    button_row->addWidget(folder_button);
    button_row->addStretch(1);
    button_row->addWidget(about);

    layout->addLayout(status_row);
    layout->addLayout(button_row);

    connect(download_button, &QPushButton::clicked, this, [this] { Download(); });
    connect(folder_button, &QPushButton::clicked, this, [] {
        const std::filesystem::path folder = FrameGen::FlyHighModelDir();
        std::error_code ec;
        std::filesystem::create_directories(folder, ec);
        QDesktopServices::openUrl(QUrl::fromLocalFile(PathText(folder)));
    });
}

void FlyHighPanel::showEvent(QShowEvent* event) {
    Refresh();
    QWidget::showEvent(event);
}

void FlyHighPanel::Refresh() {
    const bool library = FrameGen::FlyHighLibraryPresent();
    const bool model = FrameGen::FlyHighModelInstalled();
    const bool ready = library && model;
    icon_label->setText(ready ? QStringLiteral("<span style=\"color:#2ea043\">&#10004;</span>")
                              : QStringLiteral("<span style=\"color:#d29922\">&#9888;</span>"));
    if (!library) {
        status_label->setText(
            Tr("<b>Library missing:</b> flyhigh.dll (libflyhigh.so on Linux) must be beside "
               "Lichii. Lichii's release archives include it."));
    } else if (!model) {
        status_label->setText(Tr("<b>Model needed:</b> FlyHigh's network weights, about %1 MB, "
                                 "are downloaded once into the user folder.")
                                  .arg((DownloadSize() + (1 << 19)) >> 20));
    } else {
        status_label->setText(Tr("<b>Ready.</b> FlyHigh runs on its own Vulkan device on the "
                                 "same GPU; the first start at a new resolution times its "
                                 "kernels for a few seconds."));
    }
    status_label->setToolTip(Tr("Folder: %1").arg(PathText(FrameGen::FlyHighModelDir())));
    download_button->setText(Tr("Download Model"));
    download_button->setVisible(library && !model);
}

void FlyHighPanel::Download() {
    DownloadFlyHighModel(this);
    Refresh();
}

void OfferFlyHighModel(QWidget* parent) {
    if (FrameGen::FlyHighModelInstalled() || !FrameGen::FlyHighLibraryPresent()) {
        return;
    }
    DownloadFlyHighModel(parent);
}

bool DownloadFlyHighModel(QWidget* parent) {
    static bool downloading = false;
    if (downloading) {
        return false;
    }
    const QScopedValueRollback download_guard{downloading, true};
    const std::filesystem::path folder = FrameGen::FlyHighModelDir();
    const auto fail = [parent](const std::string& error) {
        LOG_ERROR(Frontend, "FlyHigh model download failed: {}", error);
        QMessageBox::warning(parent, Tr("FlyHigh"),
                             Tr("Could not download the FlyHigh model: %1. Optical Flow "
                                "Interpolation runs until it is there.")
                                 .arg(QString::fromStdString(error)));
        return false;
    };

    QDialog dialog(parent);
    dialog.setWindowTitle(Tr("Download the FlyHigh Model"));
    QVBoxLayout* layout = new QVBoxLayout(&dialog);
    QLabel* intro = new QLabel(
        Tr("If you accept, Lichii will download the FlyHigh model (graph.json, flyhighfg.json "
           "and flyhighfg.bin) from GitHub into %1.\n\n"
           "The weights are available for noncommercial use with attribution under CC BY-NC "
           "4.0. Commercial use needs a separate licence from the FlyHigh maintainers. "
           "The Apache-2.0 engine licence does not cover the weights.")
            .arg(PathText(folder)),
        &dialog);
    intro->setTextFormat(Qt::PlainText);
    intro->setWordWrap(true);
    QPlainTextEdit* licence_text = new QPlainTextEdit(Text(WEIGHTS_LICENSE), &dialog);
    licence_text->setReadOnly(true);
    licence_text->setMinimumSize(640, 380);
    QDialogButtonBox* buttons = new QDialogButtonBox(&dialog);
    QPushButton* accept_button =
        buttons->addButton(Tr("Accept and Download"), QDialogButtonBox::AcceptRole);
    accept_button->setAutoDefault(false);
    buttons->addButton(QDialogButtonBox::Cancel)->setDefault(true);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(intro);
    layout->addWidget(licence_text, 1);
    layout->addWidget(buttons);
    if (dialog.exec() != QDialog::Accepted) {
        return false;
    }

    QProgressDialog progress(Tr("Downloading the FlyHigh model..."), Tr("Cancel"), 0, 1000,
                             parent);
    progress.setWindowTitle(Tr("FlyHigh"));
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);
    progress.setValue(0);
    QApplication::processEvents();
    std::string error;
    const u64 total = DownloadSize();
    u64 done = 0;
    std::array<std::string, Model::FILES.size()> downloads;
    for (size_t i = 0; i < downloads.size(); ++i) {
        if (progress.wasCanceled()) {
            return false;
        }
        const Model::File& file = Model::FILES[i];
        auto contents = Fetch(file, error, [&](u64 received) {
            progress.setValue(static_cast<int>((done + received) * 999 / total));
            QApplication::processEvents();
            return !progress.wasCanceled();
        });
        if (progress.wasCanceled()) {
            return false;
        }
        if (!contents) {
            progress.close();
            return fail(error);
        }
        done += file.size;
        downloads[i] = std::move(*contents);
    }

    std::error_code ec;
    std::filesystem::create_directories(folder, ec);
    if (ec) {
        progress.close();
        return fail(fmt::format("could not create {}: {}", Common::FS::PathToUTF8String(folder),
                                ec.message()));
    }
    if (!Save(folder / Model::LICENSE_FILE, WEIGHTS_LICENSE, error)) {
        progress.close();
        return fail(error);
    }
    for (size_t i = 0; i < downloads.size(); ++i) {
        if (!Save(folder / Model::FILES[i].name, downloads[i], error)) {
            progress.close();
            return fail(error);
        }
    }
    LOG_INFO(Frontend, "Downloaded the FlyHigh model into {}", Common::FS::PathToUTF8String(folder));
    progress.setValue(1000);
    progress.close();
    return true;
}
