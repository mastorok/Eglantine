// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// FlyHigh's own settings in one place, opened from the gear beside FlyHigh in the frame
// generation chip's menu: the model, the multiplier, the target frame rate and the HUD.

#pragma once

#include <QDialog>

class QCheckBox;
class QComboBox;
class QSpinBox;

class FlyHighDialog : public QDialog {
    Q_OBJECT

public:
    explicit FlyHighDialog(QWidget* parent);

    /// Writes the choices to the settings and makes FlyHigh the running frame generation.
    void Apply();

private:
    QComboBox* multiplier;
    QSpinBox* target_rate;
    QCheckBox* protect_hud;
};
