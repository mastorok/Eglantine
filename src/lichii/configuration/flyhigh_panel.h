// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Shown under the frame generation type while FlyHigh is chosen: whether its library is beside
// Lichii and its model in the user folder, and a button that downloads the model from the
// FlyHigh engine's repository at a pinned release.

#pragma once

#include <QWidget>

class QLabel;
class QPushButton;

/// Shows the model's licence and downloads it on Accept. True when the model is there afterwards.
bool DownloadFlyHighModel(QWidget* parent);

/// Turning FlyHigh on: offers the download when the model is missing. Until it is there, Optical
/// Flow runs in FlyHigh's place.
void OfferFlyHighModel(QWidget* parent);

class FlyHighPanel : public QWidget {
public:
    explicit FlyHighPanel(QWidget* parent);

    void Refresh();

protected:
    void showEvent(QShowEvent* event) override;

private:
    void Download();

    QLabel* icon_label;
    QLabel* status_label;
    QPushButton* download_button;
    QPushButton* folder_button;
};
