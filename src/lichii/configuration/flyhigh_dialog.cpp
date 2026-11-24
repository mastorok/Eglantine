// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

#include "common/settings.h"
#include "lichii/configuration/flyhigh_dialog.h"
#include "lichii/configuration/flyhigh_panel.h"

FlyHighDialog::FlyHighDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("FlyHigh Frame Generation"));
    QVBoxLayout* layout = new QVBoxLayout(this);

    QLabel* about = new QLabel(
        tr("A neural network that makes the frames between the game's own, on the same GPU."),
        this);
    about->setWordWrap(true);
    layout->addWidget(about);
    layout->addWidget(new FlyHighPanel(this));

    QFormLayout* form = new QFormLayout;
    multiplier = new QComboBox(this);
    for (u32 factor = Settings::MIN_FRAME_GEN_MULTIPLIER;
         factor <= Settings::MAX_FRAME_GEN_MULTIPLIER; ++factor) {
        multiplier->addItem(tr("%1x (%2 generated per game frame)").arg(factor).arg(factor - 1),
                            factor);
    }
    multiplier->setCurrentIndex(multiplier->findData(Settings::FrameGenMultiplier()));
    form->addRow(tr("Multiplier"), multiplier);

    target_rate = new QSpinBox(this);
    target_rate->setRange(0, 240);
    target_rate->setSuffix(tr(" FPS"));
    target_rate->setSpecialValueText(tr("Off (use the multiplier)"));
    target_rate->setValue(static_cast<int>(Settings::values.frame_gen_target_rate.GetValue()));
    target_rate->setToolTip(tr("Generates up to %1 frames per game frame, as many as reach this "
                               "frame rate.")
                                .arg(Settings::MAX_FRAME_GEN_MULTIPLIER - 1));
    form->addRow(tr("Target frame rate"), target_rate);
    const auto update_multiplier = [this] { multiplier->setEnabled(target_rate->value() == 0); };
    update_multiplier();
    connect(target_rate, &QSpinBox::valueChanged, this, update_multiplier);

    protect_hud = new QCheckBox(tr("Keep HUD still"), this);
    protect_hud->setChecked(Settings::values.frame_gen_flyhigh_protect_hud.GetValue());
    protect_hud->setToolTip(tr("Pixels that have not changed for several frames, such as a HUD, "
                               "are shown as they are instead of being interpolated."));
    form->addRow(QString{}, protect_hud);
    layout->addLayout(form);

    QDialogButtonBox* buttons = new QDialogButtonBox(this);
    buttons->addButton(tr("Use FlyHigh"), QDialogButtonBox::AcceptRole);
    buttons->addButton(QDialogButtonBox::Cancel);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    layout->addWidget(buttons);
}

void FlyHighDialog::Apply() {
    Settings::values.frame_gen_multiplier.SetValue(multiplier->currentData().toUInt());
    Settings::values.frame_gen_target_rate.SetValue(static_cast<u32>(target_rate->value()));
    Settings::values.frame_gen_flyhigh_protect_hud.SetValue(protect_hud->isChecked());
    Settings::values.frame_gen_type.SetValue(Settings::FrameGenType::FlyHigh);
    Settings::values.frame_gen.SetValue(true);
    OfferFlyHighModel(parentWidget());
}
