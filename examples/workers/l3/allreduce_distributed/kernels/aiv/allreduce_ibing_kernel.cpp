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
/**
 * IBing Interleaved Bidirectional Ring AllReduce — faithful implementation
 * of the algorithm by Zong et al., ACM TACO 2025.
 *
 * P-1 bidirectional rounds with mixed AtomicAdd / AtomicNone phases.
 * Verified for P=2 (no forward phase).
 *
 * All bulk TLOAD/TSTORE/TPUT operations are tiled in TILE_CAP blocks to
 * keep UB usage under 256 KB regardless of ALLREDUCE_COUNT.
 *
 * Scratch layout (per rank, in HCCL window):
 *   [0 .. P*chunk_elems)                   P working chunks
 *   [P*chunk .. (P+1)*chunk)               exchange_right (snapshot buffer)
 *   [(P+1)*chunk .. (P+2)*chunk)          exchange_left  (snapshot buffer)
 *   tail                   (2*(P-1)+1) * kMaxSupportedRanks int32 signals
 *
 * Divisibility: requires ALLREDUCE_COUNT % nranks == 0.
 */
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "platform_comm/comm_context.h"
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static constexpr size_t ALLREDUCE_COUNT = 16777216;
static constexpr int kMaxSupportedRanks = 16;
static constexpr size_t TILE_CAP = 4096;

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommContext *ctx, __gm__ T *localPtr, int pe) {
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T *)(ctx->windowsIn[pe] + offset);
}

AICORE inline void RoundBarrier(__gm__ CommContext *ctx, __gm__ int32_t *signal_row, int my_rank, int nranks) {
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        __gm__ int32_t *remote_signal = CommRemotePtr(ctx, signal_row + my_rank, peer);
        pto::comm::Signal sig(remote_signal);
        pto::comm::TNOTIFY(sig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
    }
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        pto::comm::Signal sig(signal_row + peer);
        pto::comm::TWAIT(sig, (int32_t)1, pto::comm::WaitCmp::GE);
    }
    pipe_barrier(PIPE_ALL);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *input_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *output_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *scratch_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    int nranks = static_cast<int>(args[3]);
    __gm__ CommContext *commCtx = reinterpret_cast<__gm__ CommContext *>(args[4]);

    __gm__ float *input = reinterpret_cast<__gm__ float *>(input_tensor->buffer.addr) + input_tensor->start_offset;
    __gm__ float *output = reinterpret_cast<__gm__ float *>(output_tensor->buffer.addr) + output_tensor->start_offset;
    __gm__ float *scratch =
        reinterpret_cast<__gm__ float *>(scratch_tensor->buffer.addr) + scratch_tensor->start_offset;

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GT = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;

    int my_rank = static_cast<int>(commCtx->rankId);

    if (nranks <= 1 || nranks > kMaxSupportedRanks || (ALLREDUCE_COUNT % static_cast<size_t>(nranks)) != 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const int chunk_elems = static_cast<int>(ALLREDUCE_COUNT / static_cast<size_t>(nranks));
    __gm__ float *chunks = scratch;
    __gm__ float *exchange_right = scratch + static_cast<size_t>(nranks * chunk_elems);
    __gm__ float *exchange_left = exchange_right + static_cast<size_t>(chunk_elems);
    __gm__ int32_t *signal_base = reinterpret_cast<__gm__ int32_t *>(exchange_left + static_cast<size_t>(chunk_elems));

    using TileSub = pto::Tile<pto::TileType::Vec, float, 1, TILE_CAP, pto::BLayout::RowMajor, -1, -1>;
    TileSub stageTile(1, TILE_CAP);
    TASSIGN(stageTile, 0x0);
#ifndef __CPU_SIM
    TileSub pushTile(1, TILE_CAP);
    TASSIGN(pushTile, 0x10000);
#endif

    // ------------------------------------------------------------------
    // Phase 1: stage-in — copy P chunks from input to scratch.
    // Tiled: each chunk broken into TILE_CAP blocks.
    // ------------------------------------------------------------------
    for (int c = 0; c < nranks; ++c) {
        __gm__ float *base_dst = chunks + c * chunk_elems;
        __gm__ float *base_src = input + c * chunk_elems;
        for (int blk = 0; blk < chunk_elems; blk += (int)TILE_CAP) {
            int cur = ((int)TILE_CAP < chunk_elems - blk) ? (int)TILE_CAP : (chunk_elems - blk);
            __gm__ float *dst = base_dst + blk;
            __gm__ float *src = base_src + blk;
            ShapeDyn shape(1, 1, 1, 1, cur);
            StrideDyn stride(cur, cur, cur, cur, 1);
            GT srcG(src, shape, stride);
            GT dstG(dst, shape, stride);
            TLOAD(stageTile, srcG);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstG, stageTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // Phase 2: IBing interleaved RS+AG — P−1 rounds.
    // Tiled: snapshot and both pushes done in TILE_CAP blocks.
    // ------------------------------------------------------------------
    const int left = (my_rank - 1 + nranks) % nranks;
    const int right = (my_rank + 1) % nranks;
    const int reduce_steps = nranks / 2;

    for (int step = 1; step < nranks; ++step) {
        // Barrier A: prior writes globally visible.
        RoundBarrier(commCtx, signal_base + (2 * (step - 1)) * kMaxSupportedRanks, my_rank, nranks);

        const int idx_r = (my_rank - step + nranks) % nranks;
        const int idx_l = (my_rank + step + nranks + 1) % nranks;
        const bool fwd = (step > reduce_steps);

        // Snapshot source chunks into exchange buffers — tiled.
#ifdef __CPU_SIM
        for (int blk = 0; blk < chunk_elems; blk += (int)TILE_CAP) {
            int cur = ((int)TILE_CAP < chunk_elems - blk) ? (int)TILE_CAP : (chunk_elems - blk);
            __gm__ float *src_r = chunks + static_cast<size_t>(idx_r * chunk_elems + blk);
            __gm__ float *src_l = chunks + static_cast<size_t>(idx_l * chunk_elems + blk);
            for (int i = 0; i < cur; ++i) exchange_right[blk + i] = src_r[i];
            for (int i = 0; i < cur; ++i) exchange_left[blk + i] = src_l[i];
        }
#else
        for (int blk = 0; blk < chunk_elems; blk += (int)TILE_CAP) {
            int cur = ((int)TILE_CAP < chunk_elems - blk) ? (int)TILE_CAP : (chunk_elems - blk);
            ShapeDyn shape(1, 1, 1, 1, cur);
            StrideDyn stride(cur, cur, cur, cur, 1);

            // exchange_right ← chunks[idx_r]
            {
                __gm__ float *src_r = chunks + static_cast<size_t>(idx_r * chunk_elems + blk);
                __gm__ float *dst_r = exchange_right + blk;
                GT srcG(src_r, shape, stride);
                GT dstG(dst_r, shape, stride);
                TLOAD(stageTile, srcG);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                TSTORE(dstG, stageTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }
            // exchange_left ← chunks[idx_l]
            {
                __gm__ float *src_l = chunks + static_cast<size_t>(idx_l * chunk_elems + blk);
                __gm__ float *dst_l = exchange_left + blk;
                GT srcG(src_l, shape, stride);
                GT dstG(dst_l, shape, stride);
                TLOAD(stageTile, srcG);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
                TSTORE(dstG, stageTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
            }
        }
        pipe_barrier(PIPE_ALL);
#endif

        // Barrier B: all snapshots complete.
        RoundBarrier(commCtx, signal_base + (2 * (step - 1) + 1) * kMaxSupportedRanks, my_rank, nranks);

        // Right-bound push — tiled.
        for (int blk = 0; blk < chunk_elems; blk += (int)TILE_CAP) {
            int cur = ((int)TILE_CAP < chunk_elems - blk) ? (int)TILE_CAP : (chunk_elems - blk);
            ShapeDyn shape(1, 1, 1, 1, cur);
            StrideDyn stride(cur, cur, cur, cur, 1);
#ifdef __CPU_SIM
            __gm__ float *src = exchange_right + blk;
            __gm__ float *dst = CommRemotePtr(commCtx, chunks + static_cast<size_t>(idx_r * chunk_elems + blk), right);
            if (fwd) {
                for (int i = 0; i < cur; ++i) dst[i] = src[i];
            } else {
                for (int i = 0; i < cur; ++i) dst[i] += src[i];
            }
#else
            __gm__ float *src = exchange_right + blk;
            __gm__ float *dst = CommRemotePtr(commCtx, chunks + static_cast<size_t>(idx_r * chunk_elems + blk), right);
            GT srcG(src, shape, stride);
            GT dstG(dst, shape, stride);
            if (fwd) pto::comm::TPUT<pto::AtomicType::AtomicNone>(dstG, srcG, pushTile);
            else pto::comm::TPUT<pto::AtomicType::AtomicAdd>(dstG, srcG, pushTile);
#endif
        }

        // Left-bound push — tiled.
        for (int blk = 0; blk < chunk_elems; blk += (int)TILE_CAP) {
            int cur = ((int)TILE_CAP < chunk_elems - blk) ? (int)TILE_CAP : (chunk_elems - blk);
            ShapeDyn shape(1, 1, 1, 1, cur);
            StrideDyn stride(cur, cur, cur, cur, 1);
#ifdef __CPU_SIM
            __gm__ float *src = exchange_left + blk;
            __gm__ float *dst = CommRemotePtr(commCtx, chunks + static_cast<size_t>(idx_l * chunk_elems + blk), left);
            if (fwd) {
                for (int i = 0; i < cur; ++i) dst[i] = src[i];
            } else {
                for (int i = 0; i < cur; ++i) dst[i] += src[i];
            }
#else
            __gm__ float *src = exchange_left + blk;
            __gm__ float *dst = CommRemotePtr(commCtx, chunks + static_cast<size_t>(idx_l * chunk_elems + blk), left);
            GT srcG(src, shape, stride);
            GT dstG(dst, shape, stride);
            if (fwd) pto::comm::TPUT<pto::AtomicType::AtomicNone>(dstG, srcG, pushTile);
            else pto::comm::TPUT<pto::AtomicType::AtomicAdd>(dstG, srcG, pushTile);
#endif
        }

        pipe_barrier(PIPE_ALL);
    }

    // Final sync barrier.
    if (nranks > 1) {
        RoundBarrier(commCtx, signal_base + (2 * (nranks - 1)) * kMaxSupportedRanks, my_rank, nranks);
    }

    // ------------------------------------------------------------------
    // Phase 3: stage-out — copy P fully-reduced chunks to output.
    // Tiled same as stage-in.
    // ------------------------------------------------------------------
    for (int c = 0; c < nranks; ++c) {
        __gm__ float *base_dst = output + c * chunk_elems;
        __gm__ float *base_src = chunks + c * chunk_elems;
        for (int blk = 0; blk < chunk_elems; blk += (int)TILE_CAP) {
            int cur = ((int)TILE_CAP < chunk_elems - blk) ? (int)TILE_CAP : (chunk_elems - blk);
            __gm__ float *dst = base_dst + blk;
            __gm__ float *src = base_src + blk;
            ShapeDyn shape(1, 1, 1, 1, cur);
            StrideDyn stride(cur, cur, cur, cur, 1);
            GT srcG(src, shape, stride);
            GT dstG(dst, shape, stride);
            TLOAD(stageTile, srcG);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstG, stageTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
    pipe_barrier(PIPE_ALL);
}
