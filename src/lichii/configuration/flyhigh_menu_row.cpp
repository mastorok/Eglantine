// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <QCoreApplication>
#include <QIcon>
#include <QMouseEvent>
#include <QPainter>
#include <QStyleOptionMenuItem>
#include <QTimer>

#include "lichii/configuration/flyhigh_menu_row.h"

namespace {
constexpr int GAP = 4;
constexpr int ICON_SIZE = 16;

/// The icon Lichii's menus give their Configure entries, grey and white when hovered.
QIcon ConfigureIcon() {
    QIcon icon;
    icon.addFile(QStringLiteral(":/lichii/icons/menu_sliders.png"));
    icon.addFile(QStringLiteral(":/lichii/icons/menu_sliders_white.png"), {}, QIcon::Active);
    return icon;
}
} // Anonymous namespace

void FlyHighTypeMenu::SetGear(QAction* action, std::function<void()> on_configure_) {
    gear_action = action;
    on_configure = std::move(on_configure_);
    setMouseTracking(true);
}

QRect FlyHighTypeMenu::GearRect() const {
    if (gear_action == nullptr) {
        return {};
    }
    const QRect row = actionGeometry(gear_action);
    const int width = row.height() + ICON_SIZE;
    return {row.right() - width + 1, row.top(), width, row.height()};
}

bool FlyHighTypeMenu::OverGear(const QPoint& point) const {
    return gear_action != nullptr && GearRect().contains(point);
}

void FlyHighTypeMenu::paintEvent(QPaintEvent* event) {
    QMenu::paintEvent(event);
    if (gear_action == nullptr) {
        return;
    }
    const QRect row = actionGeometry(gear_action);
    const QRect gear = GearRect();
    const QRect name{row.left(), row.top(), gear.left() - GAP - row.left(), row.height()};
    const bool row_active = activeAction() == gear_action;

    QPainter painter(this);
    painter.setClipRect(row);
    QStyleOption panel;
    panel.initFrom(this);
    panel.rect = rect();
    style()->drawPrimitive(QStyle::PE_PanelMenu, &panel, &painter, this);

    QStyleOptionMenuItem item;
    initStyleOption(&item, gear_action);
    item.rect = name;
    if (!row_active || gear_hovered) {
        item.state &= ~QStyle::State_Selected;
    }
    style()->drawControl(QStyle::CE_MenuItem, &item, &painter, this);

    QStyleOptionMenuItem gear_item;
    initStyleOption(&gear_item, gear_action);
    gear_item.rect = gear;
    gear_item.text.clear();
    gear_item.icon = {};
    if (row_active && gear_hovered) {
        gear_item.state |= QStyle::State_Selected;
    } else {
        gear_item.state &= ~QStyle::State_Selected;
    }
    style()->drawControl(QStyle::CE_MenuItem, &gear_item, &painter, this);
    const bool selected = (gear_item.state & QStyle::State_Selected) != 0;
    const QRect icon_rect{gear.center().x() - ICON_SIZE / 2 + 1,
                          gear.center().y() - ICON_SIZE / 2 + 1, ICON_SIZE, ICON_SIZE};
    ConfigureIcon().paint(&painter, icon_rect, Qt::AlignCenter,
                          selected ? QIcon::Active : QIcon::Normal);
}

void FlyHighTypeMenu::mouseMoveEvent(QMouseEvent* event) {
    QMenu::mouseMoveEvent(event);
    const bool over = OverGear(event->position().toPoint());
    if (over != gear_hovered) {
        gear_hovered = over;
        setToolTip(over ? tr("FlyHigh settings") : QString{});
    }
    if (gear_action != nullptr) {
        update(actionGeometry(gear_action));
    }
}

void FlyHighTypeMenu::leaveEvent(QEvent* event) {
    gear_hovered = false;
    QMenu::leaveEvent(event);
}

void FlyHighTypeMenu::mouseReleaseEvent(QMouseEvent* event) {
    if (!OverGear(event->position().toPoint())) {
        QMenu::mouseReleaseEvent(event);
        return;
    }
    close();
    // After the menu's own event loop has ended: the settings are a modal dialog.
    QTimer::singleShot(0, qApp, on_configure);
}
