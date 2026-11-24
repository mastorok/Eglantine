// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>

#include <QMenu>

/// The frame generation chip's menu, where FlyHigh's own entry ends in a settings icon that opens its
/// settings. The entry stays an ordinary action drawn by the menu; the icon is drawn over its right
/// end the way the menu draws an item, and hovered and clicked apart from the rest of the entry.
class FlyHighTypeMenu : public QMenu {
public:
    /// `action` is FlyHigh's entry, already added; `on_configure` runs once the menu has closed.
    void SetGear(QAction* action, std::function<void()> on_configure);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void leaveEvent(QEvent* event) override;

private:
    [[nodiscard]] QRect GearRect() const;
    [[nodiscard]] bool OverGear(const QPoint& point) const;

    QAction* gear_action{};
    std::function<void()> on_configure;
    bool gear_hovered{};
};
