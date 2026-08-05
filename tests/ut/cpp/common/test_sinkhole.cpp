/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/*
 * Standalone test for the self-notify sinkhole.
 *
 * Covers three layers:
 *   1. Registry (register / unregister / is_self_notify_sinkhole)
 *   2. TNOTIFY integration (self-window write dropped, remote write proceeds)
 *   3. Edge cases: zero-size, double-unregister, no-windows-registered
 *
 * Build & run (inside the sim Docker image):
 *   g++ -std=c++17 -o test_sinkhole test_sinkhole.cpp -ldl -lpthread
 *   LD_PRELOAD=<path>/libcpu_sim_context.so ./test_sinkhole
 *
 * This test does NOT #include pto-isa headers because they pull in the full
 * NPU instruction set (aicore attributes, etc.) that only the device compiler
 * understands.  Instead it replicates the ~10-line sinkhole check inline:
 * the dlsym(RTLD_DEFAULT, "pto_sim_is_self_notify_sinkhole") pattern is
 * bit-identical to what TNotify.hpp does at runtime.
 */

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

// ---------------------------------------------------------------------------
// Dynamically resolve the three registry hooks from libcpu_sim_context.so.
// This mirrors exactly how pto-isa TNotify.hpp resolves the sinkhole hook,
// and how comm_sim.cpp resolves register/unregister hooks.
// ---------------------------------------------------------------------------

using RegisterFn   = void (*)(uint64_t, uint64_t);
using UnregisterFn = void (*)(uint64_t);
using SinkholeFn   = bool (*)(uint64_t);

struct SinkholeApi {
    RegisterFn   reg;
    UnregisterFn unreg;
    SinkholeFn   is_sink;
};

static SinkholeApi load_sinkhole_api() {
    SinkholeApi api{};
    api.reg     = reinterpret_cast<RegisterFn>(dlsym(RTLD_DEFAULT, "pto_sim_register_self_window"));
    api.unreg   = reinterpret_cast<UnregisterFn>(dlsym(RTLD_DEFAULT, "pto_sim_unregister_self_window"));
    api.is_sink = reinterpret_cast<SinkholeFn>(dlsym(RTLD_DEFAULT, "pto_sim_is_self_notify_sinkhole"));
    if (!api.reg || !api.unreg || !api.is_sink) {
        std::fprintf(stderr, "FATAL: sinkhole API not found — did you LD_PRELOAD libcpu_sim_context.so?\n");
        std::exit(1);
    }
    return api;
}

// ---------------------------------------------------------------------------
// Minimal TNOTIFY reimplementation — matches the pto-isa TNotify.hpp code
// path for CPU-sim.  We inline it here so this test builds with a standard
// C++ compiler (no aicore attributes needed).
// ---------------------------------------------------------------------------

// Mirror of pto::comm::Signal — lightweight wrapper around a typed pointer
// so the call site looks identical to real kernel code.
template <typename T>
struct Signal {
    using DType = T;
    T *ptr;
    explicit Signal(T *p) : ptr(p) {}
    T *get() const { return ptr; }
};

enum class NotifyOp : int32_t { Set = 0, AtomicAdd = 1 };

// Inline IsSelfNotifySinkhole — copied from pto-isa TNotify.hpp:detail::
inline bool IsSelfNotifySinkhole(uint64_t addr) {
    static auto hook =
        reinterpret_cast<bool (*)(uint64_t)>(dlsym(RTLD_DEFAULT, "pto_sim_is_self_notify_sinkhole"));
    return hook != nullptr && hook(addr);
}

template <typename SignalType>
void TNotify_Impl(typename SignalType::DType *dstSignalData, int32_t value, NotifyOp op) {
    using DType = typename SignalType::DType;
    if (IsSelfNotifySinkhole(reinterpret_cast<uint64_t>(dstSignalData))) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr,
                         "[cpu-sim] TNOTIFY to self window dropped (addr=%p): "
                         "onboard NPU self-notify is a no-op; "
                         "use a local store for self-destination data\n",
                         static_cast<void *>(dstSignalData));
        }
        return;
    }
    auto *atomicPtr = reinterpret_cast<std::atomic<DType> *>(dstSignalData);
    switch (op) {
        case NotifyOp::Set:
            atomicPtr->store(value, std::memory_order_release);
            break;
        case NotifyOp::AtomicAdd:
            atomicPtr->fetch_add(value, std::memory_order_acq_rel);
            break;
    }
}

// Convenience overloads matching the pto-isa TNOTIFY macro expansion.
template <typename Sig, typename... Extra>
void TNOTIFY(Sig signal, int32_t value, NotifyOp op) {
    TNotify_Impl<Sig>(signal.get(), value, op);
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static int g_failures = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: " fmt "\n", ##__VA_ARGS__); \
        ++g_failures; \
    } else { \
        std::fprintf(stdout, "  OK: " fmt "\n", ##__VA_ARGS__); \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Test 1: Registry basics
// ---------------------------------------------------------------------------
static void test_registry_basics(const SinkholeApi &api) {
    std::fprintf(stdout, "\n=== Test 1: Registry basics ===\n");

    // Single window
    api.reg(0x1000, 0x100);
    CHECK( api.is_sink(0x1000),       "addr at window base is sinkhole");
    CHECK( api.is_sink(0x10FF),       "addr at window end-1 is sinkhole");
    CHECK(!api.is_sink(0x0FFF),       "addr just before window is NOT sinkhole");
    CHECK(!api.is_sink(0x1100),       "addr just after window is NOT sinkhole");
    CHECK(!api.is_sink(0x0000),       "null addr is NOT sinkhole");

    // Unregister
    api.unreg(0x1000);
    CHECK(!api.is_sink(0x1000),       "unregistered window is NOT sinkhole");
    CHECK(!api.is_sink(0x10FF),       "unregistered window end is NOT sinkhole");

    // Re-register
    api.reg(0x1000, 0x100);
    CHECK( api.is_sink(0x1000),       "re-registered window is sinkhole");
    api.unreg(0x1000);
}

// ---------------------------------------------------------------------------
// Test 2: Multi-window registration
// ---------------------------------------------------------------------------
static void test_multi_window(const SinkholeApi &api) {
    std::fprintf(stdout, "\n=== Test 2: Multi-window ===\n");

    api.reg(0x1000, 0x100);
    api.reg(0x3000, 0x200);

    CHECK( api.is_sink(0x1000),       "window 1 base is sinkhole");
    CHECK( api.is_sink(0x10FF),       "window 1 end is sinkhole");
    CHECK( api.is_sink(0x3000),       "window 2 base is sinkhole");
    CHECK( api.is_sink(0x31FF),       "window 2 end is sinkhole");
    CHECK(!api.is_sink(0x2000),       "gap between windows is NOT sinkhole");
    CHECK(!api.is_sink(0x3200),       "after window 2 is NOT sinkhole");

    // Unregister first — second should remain
    api.unreg(0x1000);
    CHECK(!api.is_sink(0x1000),       "window 1 unregistered");
    CHECK( api.is_sink(0x3000),       "window 2 still sinkhole after 1 removed");

    // Unregister second — all clean
    api.unreg(0x3000);
    CHECK(!api.is_sink(0x3000),       "window 2 unregistered");
    CHECK(!api.is_sink(0x1000),       "window 1 still gone");
}

// ---------------------------------------------------------------------------
// Test 3: Zero-size window (no-op)
// ---------------------------------------------------------------------------
static void test_zero_size(const SinkholeApi &api) {
    std::fprintf(stdout, "\n=== Test 3: Zero-size window ===\n");

    api.reg(0x5000, 0);
    CHECK(!api.is_sink(0x5000),       "zero-size window is NOT registered");
    api.unreg(0x5000);                // harmless
}

// ---------------------------------------------------------------------------
// Test 4: No double-unregister crash
// ---------------------------------------------------------------------------
static void test_double_unregister(const SinkholeApi &api) {
    std::fprintf(stdout, "\n=== Test 4: Double unregister safety ===\n");

    api.reg(0x6000, 0x80);
    CHECK( api.is_sink(0x6000),       "registered");
    api.unreg(0x6000);
    CHECK(!api.is_sink(0x6000),       "unregistered");
    api.unreg(0x6000);                // second unregister — must not crash
    CHECK(!api.is_sink(0x6000),       "still gone after double unregister");
}

// ---------------------------------------------------------------------------
// Test 5: TNOTIFY self-window write is dropped (Set + AtomicAdd)
// ---------------------------------------------------------------------------
static void test_tnotify_self_dropped(const SinkholeApi &api) {
    std::fprintf(stdout, "\n=== Test 5: TNOTIFY self-window write dropped ===\n");

    constexpr uint64_t kWindowSize = 64;
    alignas(64) int32_t buf[kWindowSize] = {};
    uint64_t base = reinterpret_cast<uint64_t>(&buf[0]);

    api.reg(base, kWindowSize * sizeof(int32_t));

    // TNOTIFY(Set) to self-window → dropped
    int32_t before = buf[0];
    Signal<int32_t> sig_set(&buf[0]);
    TNOTIFY(sig_set, 42, NotifyOp::Set);
    int32_t after = buf[0];
    CHECK(before == after,
          "TNOTIFY Set to self-window DROPPED (before=%d after=%d)", before, after);

    // TNOTIFY(AtomicAdd) to self-window → dropped
    before = buf[4];
    Signal<int32_t> sig_add(&buf[4]);
    TNOTIFY(sig_add, 7, NotifyOp::AtomicAdd);
    after = buf[4];
    CHECK(before == after,
          "TNOTIFY AtomicAdd to self-window DROPPED (before=%d after=%d)", before, after);

    api.unreg(base);
}

// ---------------------------------------------------------------------------
// Test 6: TNOTIFY remote write proceeds normally (Set + AtomicAdd)
// ---------------------------------------------------------------------------
static void test_tnotify_remote_proceeds(const SinkholeApi &api) {
    std::fprintf(stdout, "\n=== Test 6: TNOTIFY remote write proceeds ===\n");

    constexpr uint64_t kWindowSize = 64;
    alignas(64) int32_t buf[kWindowSize] = {};
    uint64_t base = reinterpret_cast<uint64_t>(&buf[0]);

    // Register a DIFFERENT range — buf is NOT self
    api.reg(base + 0x10000, kWindowSize * sizeof(int32_t));

    // TNOTIFY(Set) to non-self → proceeds
    int32_t before = buf[0];
    Signal<int32_t> sig_set(&buf[0]);
    TNOTIFY(sig_set, 99, NotifyOp::Set);
    int32_t after = buf[0];
    CHECK(after == 99,
          "TNOTIFY Set to remote window PROCEEDS (before=%d after=%d)", before, after);

    // TNOTIFY(AtomicAdd) to non-self → proceeds
    buf[4] = 10;
    before = buf[4];
    Signal<int32_t> sig_add(&buf[4]);
    TNOTIFY(sig_add, 5, NotifyOp::AtomicAdd);
    after = buf[4];
    CHECK(after == 15,
          "TNOTIFY AtomicAdd to remote PROCEEDS (10+5=%d)", after);

    api.unreg(base + 0x10000);
}

// ---------------------------------------------------------------------------
// Test 7: No windows registered — all writes proceed
// ---------------------------------------------------------------------------
static void test_tnotify_no_windows(const SinkholeApi &api) {
    (void)api;
    std::fprintf(stdout, "\n=== Test 7: TNOTIFY with no windows registered ===\n");

    alignas(64) int32_t buf[4] = {};
    Signal<int32_t> sig(&buf[0]);
    TNOTIFY(sig, 77, NotifyOp::Set);
    CHECK(buf[0] == 77,
          "TNOTIFY Set with no windows PROCEEDS (got %d)", buf[0]);
}

// ---------------------------------------------------------------------------
// Test 8: dlsym(RTLD_DEFAULT) contract — symbols visible when
//         libcpu_sim_context.so is LD_PRELOADed.
// ---------------------------------------------------------------------------
static void test_dlsym_contract() {
    std::fprintf(stdout, "\n=== Test 8: dlsym(RTLD_DEFAULT) contract ===\n");

    void *reg_sym = dlsym(RTLD_DEFAULT, "pto_sim_register_self_window");
    void *unreg_sym = dlsym(RTLD_DEFAULT, "pto_sim_unregister_self_window");
    void *sink_sym = dlsym(RTLD_DEFAULT, "pto_sim_is_self_notify_sinkhole");

    CHECK(reg_sym != nullptr,   "pto_sim_register_self_window found via RTLD_DEFAULT");
    CHECK(unreg_sym != nullptr, "pto_sim_unregister_self_window found via RTLD_DEFAULT");
    CHECK(sink_sym != nullptr,  "pto_sim_is_self_notify_sinkhole found via RTLD_DEFAULT");
}

// ===========================================================================
int main() {
    std::fprintf(stdout, "=== Self-Notify Sinkhole Test Suite ===\n");

    auto api = load_sinkhole_api();
    std::fprintf(stdout, "Sinkhole API resolved: reg=%p unreg=%p is_sink=%p\n",
                 reinterpret_cast<void*>(api.reg),
                 reinterpret_cast<void*>(api.unreg),
                 reinterpret_cast<void*>(api.is_sink));

    test_dlsym_contract();
    test_registry_basics(api);
    test_multi_window(api);
    test_zero_size(api);
    test_double_unregister(api);
    test_tnotify_self_dropped(api);
    test_tnotify_remote_proceeds(api);
    test_tnotify_no_windows(api);

    std::fprintf(stdout, "\n=== Results: %d failures ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
