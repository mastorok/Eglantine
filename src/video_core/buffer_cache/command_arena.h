// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstring>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

#include <boost/container/small_vector.hpp>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common_types.h"

namespace VideoCommon {

/// Commands recorded by value, played in order later (EDEN_NX2_BUFFER_ARENA). Each record is a
/// header, a trivially copyable body and an optional array, 8-byte aligned. Clear keeps the
/// storage, so a steady stream of records allocates nothing.
class CommandArena {
public:
    struct Header {
        u16 kind;
        u16 count; ///< Elements of the trailing array
        u32 bytes; ///< Whole record, header included
    };

    template <typename T>
    void Push(u16 kind, const T& body) {
        Push<T, u8>(kind, body, {});
    }

    template <typename T, typename E>
    void Push(u16 kind, const T& body, std::span<const E> array) {
        static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_copyable_v<E>);
        static_assert(alignof(T) <= Align && alignof(E) <= Align);
        ASSERT(array.size() <= 0xffff);
        const size_t array_offset = ArrayOffset<T, E>();
        const size_t bytes = Common::AlignUp(array_offset + array.size_bytes(), Align);
        const size_t start = data.size();
        data.resize(start + bytes, boost::container::default_init);
        u8* const record = data.data() + start;
        const Header header{kind, static_cast<u16>(array.size()), static_cast<u32>(bytes)};
        std::memcpy(record, &header, sizeof(header));
        std::memcpy(record + sizeof(Header), &body, sizeof(T));
        if (!array.empty()) {
            std::memcpy(record + array_offset, array.data(), array.size_bytes());
        }
        ++records;
    }

    /// Calls visit(kind, record) in recording order; Body and Array read a record.
    template <typename Visit>
    void ForEach(Visit&& visit) const {
        for (size_t offset = 0; offset < data.size();) {
            const u8* const record = data.data() + offset;
            Header header;
            std::memcpy(&header, record, sizeof(header));
            visit(header.kind, record);
            offset += header.bytes;
        }
    }

    template <typename T>
    [[nodiscard]] static T Body(const u8* record) {
        T body;
        std::memcpy(&body, record + sizeof(Header), sizeof(T));
        return body;
    }

    template <typename T, typename E>
    [[nodiscard]] static std::span<const E> Array(const u8* record) {
        Header header;
        std::memcpy(&header, record, sizeof(header));
        return {reinterpret_cast<const E*>(record + ArrayOffset<T, E>()), header.count};
    }

    [[nodiscard]] bool Empty() const noexcept {
        return data.empty();
    }

    [[nodiscard]] size_t Records() const noexcept {
        return records;
    }

    [[nodiscard]] size_t Bytes() const noexcept {
        return data.size();
    }

    void Clear() noexcept {
        data.clear();
        records = 0;
    }

    void Swap(CommandArena& other) noexcept {
        data.swap(other.data);
        std::swap(records, other.records);
    }

private:
    static constexpr size_t Align = 8;

    template <typename T, typename E>
    static constexpr size_t ArrayOffset() {
        return Common::AlignUp(sizeof(Header) + sizeof(T), alignof(E));
    }

    boost::container::small_vector<u8, 128> data;
    size_t records{};
};

/// The buffer side's clock. The recording thread stamps what it records (staging memory, buffer
/// usage) with Now(); whoever plays the commands maps that stamp to the scheduler tick they go
/// in (Map), so the recording thread never reads the tick of a submission it does not make. A
/// stamp is free once it is mapped and its tick is complete. One thread maps, any thread reads.
class ArenaClock {
public:
    static constexpr u64 Window = 4096;

    /// The stamp of the commands recorded now, played at the next Map.
    [[nodiscard]] u64 Now() const noexcept {
        return next.load(std::memory_order_acquire);
    }

    /// Ends the batch stamped Now(), for a player that maps it later: what is recorded after it
    /// gets the next stamp. Returns the batch's stamp.
    u64 Close() noexcept {
        return next.fetch_add(1, std::memory_order_acq_rel);
    }

    /// The batch `stamp` goes in tick. Batches are mapped in order, and ticks never decrease.
    void Map(u64 stamp, u64 tick) noexcept {
        ticks[stamp % Window].store(tick, std::memory_order_relaxed);
        mapped.store(stamp, std::memory_order_release);
    }

    /// The commands stamped Now() go in tick, mapped before the next stamp opens. Stamps are
    /// taken (here and by Close) by one thread at a time: the buffer cache's owner.
    void Map(u64 tick) noexcept {
        const u64 stamp = next.load(std::memory_order_relaxed);
        Map(stamp, tick);
        next.store(stamp + 1, std::memory_order_release);
    }

    /// Whether the host GPU completed a stamp; tick_done(tick) answers for a tick.
    template <typename TickDone>
    [[nodiscard]] bool IsFree(u64 stamp, TickDone&& tick_done) const {
        const std::optional<u64> tick = TickOf(stamp);
        return tick && tick_done(*tick);
    }

    /// The tick a stamp went in, or a later one; nullopt while it is not mapped.
    [[nodiscard]] std::optional<u64> TickOf(u64 stamp) const {
        const u64 last = mapped.load(std::memory_order_acquire);
        if (stamp > last) {
            return std::nullopt;
        }
        if (last - stamp > Window - 2) {
            // Older than the window: its tick is at most the oldest one kept.
            stamp = last - (Window - 2);
        }
        return ticks[stamp % Window].load(std::memory_order_relaxed);
    }

private:
    std::atomic<u64> next{1};
    std::atomic<u64> mapped{};
    std::array<std::atomic<u64>, Window> ticks{};
};

} // namespace VideoCommon
