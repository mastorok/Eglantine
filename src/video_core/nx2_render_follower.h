// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "common/assert.h"
#include "common/common_types.h"
#include "common/scope_exit.h"

namespace VideoCommon {

/// Packets from the GPU back (E) to GPURender (R) in order: an SPSC byte ring of a header and a
/// typed payload each, built in place, so a packet allocates nothing (port of Fritchii's follower
/// ring, ead877857). A packet never straddles the ring's end: a pad fills the tail. A lent packet
/// runs a call on R while E waits for it (Lend). Both sides poll for a while before they park: a
/// futex wake-up costs more than a packet. Each side wakes only a parked peer, with a full fence
/// between its store and its look at the peer's flag (Fritchii 3cc04f7e6, the lost wake-up).
class Nx2RenderFollower {
public:
    static constexpr std::chrono::nanoseconds DefaultSpin = std::chrono::microseconds{30};
    static constexpr size_t DefaultBytes = size_t{1} << 20;
    static constexpr size_t Align = 16;

    enum class Kind : u8 { Pad, Packet, Lent, Stop };

    struct alignas(Align) Header {
        u32 bytes;
        Kind kind;
        u64 number;
        u64 position;
        void (*play)(void*);
    };

    /// A packet's size with a payload of "payload" bytes: whole headers, so a pad at the ring's
    /// end always has room for its own header.
    static constexpr size_t PacketBytes(size_t payload) {
        return (sizeof(Header) + payload + sizeof(Header) - 1) / sizeof(Header) * sizeof(Header);
    }

    /// With "ahead" E may post while earlier packets are unplayed; without it a post first waits
    /// for the packet before it, as the single slot did.
    explicit Nx2RenderFollower(std::chrono::nanoseconds spin_ = DefaultSpin, bool ahead_ = false,
                               size_t bytes = DefaultBytes)
        : capacity{bytes}, slots{bytes / sizeof(Header) + 1}, spin{spin_}, ahead{ahead_},
          ring{std::make_unique<Block[]>(bytes / Align)},
          positions{std::make_unique<u64[]>(slots)} {
        ASSERT(bytes % sizeof(Header) == 0 && bytes >= 2 * sizeof(Header));
    }

    // E's side.

    [[nodiscard]] bool Pending() const {
        return pushed != played.load(std::memory_order_acquire);
    }

    /// Waits until R played every packet. Returns whether E parked.
    bool Wait() {
        return EngineWait([&] { return played.load(std::memory_order_acquire) == pushed; }) ==
               Waited::Parked;
    }

    /// Posts a packet for the ring position "position": R calls it once, then destroys it.
    /// Returns whether E had to wait for room or for the packet before it.
    template <typename Packet>
    bool Post(u64 position, Packet&& packet) {
        using T = std::remove_cvref_t<Packet>;
        static_assert(alignof(T) <= Align);
        constexpr u32 bytes = static_cast<u32>(PacketBytes(sizeof(T)));
        std::byte* at;
        const bool waited = Reserve(bytes, at);
        ::new (static_cast<void*>(at + sizeof(Header))) T(std::forward<Packet>(packet));
        Commit(at, bytes, Kind::Packet, &PlayPacket<T>, position);
        return waited;
    }

    /// Runs call(argument) on R once R played every earlier packet, E waiting meanwhile. An
    /// exception the call throws is rethrown here. Returns whether E parked.
    bool Lend(u64 position, void (*call)(void*), void* argument) {
        constexpr u32 bytes = static_cast<u32>(PacketBytes(sizeof(void*)));
        std::byte* at;
        Reserve(bytes, at);
        std::memcpy(at + sizeof(Header), &argument, sizeof(argument));
        Commit(at, bytes, Kind::Lent, call, position);
        const u64 number = pushed;
        const bool parked = EngineWait([&] {
                                return played.load(std::memory_order_acquire) >= number;
                            }) == Waited::Parked;
        if (lent_error) {
            std::rethrow_exception(std::exchange(lent_error, nullptr));
        }
        return parked;
    }

    /// The last packet: PlayNext returns Kind::Stop for it. It holds no position (Floor).
    void Stop() {
        constexpr u32 bytes = static_cast<u32>(sizeof(Header));
        std::byte* at;
        Reserve(bytes, at);
        Commit(at, bytes, Kind::Stop, nullptr, ~u64{0});
    }

    // R's side.

    /// Takes the next packet (waiting for one), plays it and retires it. Returns its kind.
    Kind PlayNext() {
        for (;;) {
            if (sent.load(std::memory_order_acquire) == read_at) {
                RenderWait();
                continue;
            }
            std::byte* const at = Base() + read_at % capacity;
            Header header;
            std::memcpy(&header, at, sizeof(header));
            if (header.kind == Kind::Pad) {
                read_at += header.bytes;
                freed.store(read_at, std::memory_order_release);
                WakeEngine();
                continue;
            }
            out_of_order += header.number != played.load(std::memory_order_relaxed) + 1 ? 1 : 0;
            void* const payload = at + sizeof(Header);
            if (header.kind == Kind::Packet) {
                header.play(payload);
            } else if (header.kind == Kind::Lent) {
                void* argument;
                std::memcpy(&argument, payload, sizeof(argument));
                ++lent;
                try {
                    header.play(argument);
                } catch (...) {
                    lent_error = std::current_exception();
                }
            }
            read_at += header.bytes;
            freed.store(read_at, std::memory_order_release);
            played.store(header.number, std::memory_order_seq_cst);
            WakeEngine();
            return header.kind;
        }
    }

    // Either side.

    /// The ring position of the oldest packet R has not finished, ~0 when it finished them all.
    [[nodiscard]] u64 Floor() const {
        const u64 done = played.load(std::memory_order_acquire);
        const u64 count = sent_packets.load(std::memory_order_acquire);
        return done < count ? positions[(done + 1) % slots] : ~u64{0};
    }

    // Counters: E's are read on E or once it stopped, R's on R or once it stopped.

    /// E: the number of the last packet posted.
    [[nodiscard]] u64 Pushed() const {
        return pushed;
    }
    /// Either side: the number of the last packet R finished.
    [[nodiscard]] u64 Played() const {
        return played.load(std::memory_order_acquire);
    }

    [[nodiscard]] u64 MaxDepth() const {
        return max_depth;
    }
    [[nodiscard]] u64 RoomWaits() const {
        return room_waits;
    }
    [[nodiscard]] u64 Wraps() const {
        return wraps;
    }
    [[nodiscard]] u64 Lent() const {
        return lent;
    }
    /// Must be 0: packets R took with another number than the one after the last.
    [[nodiscard]] u64 OutOfOrder() const {
        return out_of_order;
    }

private:
    struct alignas(Align) Block {
        std::byte bytes[Align];
    };

    enum class Waited : u8 { No, Polled, Parked };

    template <typename T>
    static void PlayPacket(void* payload) {
        T* const packet = std::launder(static_cast<T*>(payload));
        SCOPE_EXIT {
            std::destroy_at(packet);
        };
        (*packet)();
    }

    std::byte* Base() const {
        return ring[0].bytes;
    }

    /// E: room for a packet of "bytes" at the write position, after a pad when the tail is short.
    bool Reserve(u32 bytes, std::byte*& at) {
        ASSERT(bytes <= capacity);
        bool waited = false;
        const u64 offset = write % capacity;
        if (offset + bytes > capacity) {
            const u64 pad = capacity - offset;
            waited |= RoomWait(
                [&] { return write + pad - freed.load(std::memory_order_acquire) <= capacity; });
            const Header header{static_cast<u32>(pad), Kind::Pad, 0, 0, nullptr};
            std::memcpy(Base() + offset, &header, sizeof(header));
            write += pad;
            sent.store(write, std::memory_order_release);
            WakeRender();
            ++wraps;
        }
        if (!ahead) {
            waited |= EngineWait([&] {
                          return played.load(std::memory_order_acquire) == pushed;
                      }) != Waited::No;
        }
        waited |= RoomWait(
            [&] { return write + bytes - freed.load(std::memory_order_acquire) <= capacity; });
        at = Base() + write % capacity;
        return waited;
    }

    void Commit(std::byte* at, u32 bytes, Kind kind, void (*play)(void*), u64 position) {
        const u64 number = ++pushed;
        const Header header{bytes, kind, number, position, play};
        std::memcpy(at, &header, sizeof(header));
        positions[number % slots] = position;
        write += bytes;
        const u64 depth = number - played.load(std::memory_order_relaxed);
        max_depth = depth > max_depth ? depth : max_depth;
        // The position before the count: Floor reads it after an acquire of the count.
        sent_packets.store(number, std::memory_order_release);
        sent.store(write, std::memory_order_release);
        WakeRender();
    }

    template <typename Ready>
    bool RoomWait(Ready&& ready) {
        const bool waited = EngineWait(ready) != Waited::No;
        room_waits += waited ? 1 : 0;
        return waited;
    }

    template <typename Ready>
    Waited EngineWait(Ready&& ready) {
        if (ready()) {
            return Waited::No;
        }
        if (Poll(ready)) {
            return Waited::Polled;
        }
        e_waiting.store(true, std::memory_order_seq_cst);
        for (;;) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
            const u32 seen = e_wake.load(std::memory_order_seq_cst);
            if (ready()) {
                break;
            }
            e_wake.wait(seen, std::memory_order_seq_cst);
        }
        e_waiting.store(false, std::memory_order_relaxed);
        return Waited::Parked;
    }

    void RenderWait() {
        const auto ready = [&] { return sent.load(std::memory_order_acquire) != read_at; };
        if (Poll(ready)) {
            return;
        }
        r_sleeping.store(true, std::memory_order_seq_cst);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const u32 seen = r_wake.load(std::memory_order_seq_cst);
        if (!ready()) {
            r_wake.wait(seen, std::memory_order_seq_cst);
        }
        r_sleeping.store(false, std::memory_order_relaxed);
    }

    void WakeRender() {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (r_sleeping.load(std::memory_order_seq_cst)) {
            r_wake.fetch_add(1, std::memory_order_seq_cst);
            r_wake.notify_one();
        }
    }

    void WakeEngine() {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (e_waiting.load(std::memory_order_seq_cst)) {
            e_wake.fetch_add(1, std::memory_order_seq_cst);
            e_wake.notify_one();
        }
    }

    static void Relax() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        asm volatile("yield");
#else
        std::this_thread::yield();
#endif
    }

    /// Whether "ready" turned true within the spin.
    template <typename Ready>
    bool Poll(Ready&& ready) const {
        if (spin.count() <= 0) {
            return ready();
        }
        const auto deadline = std::chrono::steady_clock::now() + spin;
        for (;;) {
            for (u32 i = 0; i < 64; ++i) {
                if (ready()) {
                    return true;
                }
                Relax();
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                return ready();
            }
        }
    }

    const size_t capacity;
    const size_t slots;
    const std::chrono::nanoseconds spin;
    const bool ahead;
    std::unique_ptr<Block[]> ring;
    /// The ring position of packet n at n % slots: fewer than "slots" packets fit in the ring.
    std::unique_ptr<u64[]> positions;

    // E's side.
    alignas(64) u64 write{};
    u64 pushed{};
    u64 max_depth{};
    u64 room_waits{};
    u64 wraps{};
    std::atomic<u64> sent{};
    std::atomic<u64> sent_packets{};

    // R's side.
    alignas(64) u64 read_at{};
    u64 lent{};
    u64 out_of_order{};
    std::exception_ptr lent_error;
    std::atomic<u64> freed{};
    std::atomic<u64> played{};

    alignas(64) std::atomic<u32> r_wake{};
    std::atomic<bool> r_sleeping{};
    alignas(64) std::atomic<u32> e_wake{};
    std::atomic<bool> e_waiting{};
};

} // namespace VideoCommon
