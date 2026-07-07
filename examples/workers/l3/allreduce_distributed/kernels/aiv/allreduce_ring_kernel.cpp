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
 * Ring AllReduce kernel — chunked reduce-scatter + allgather, HCCL-window scratch.
 *
 * Phase 1 (stage-in):       partition input → P chunk slots in window
 * Phase 2 (reduce-scatter): (P-1) ring steps; rank r owns reduced chunk r
 * Phase 3 (allgather):      (P-1) ring steps; collect all reduced chunks
 * Phase 4 (stage-out):      chunks → output
 *
 * All bulk TLOAD/TSTORE/TADD operations are tiled in TILE_CAP blocks to
 * keep UB usage under 256 KB regardless of ALLREDUCE_COUNT.
 *
 * args layout (passed as Tensor arg slots — see allreduce_ring_orch.cpp):
 *   tensor(0) = input    (host-backed, framework-supplied device addr)
 *   tensor(1) = output   (host-backed, framework-supplied device addr)
 *   tensor(2) = scratch  (HCCL window slot, cross-rank addressable)
 *   scalar(0) = nranks
 *   scalar(1) = CommContext device pointer
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
// Max elements per single TLOAD/TSTORE — keeps UB usage well under 256 KB
// even when multiple tiles coexist (e.g. chunkTile + recvTile = 2 × 16 KB).
static constexpr size_t TILE_CAP = 4096;

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommContext *ctx, __gm__ T *localPtr, int pe) {
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T *)(ctx->windowsIn[pe] + offset);
}

// Per-round barrier row: used exactly once (AtomicAdd 0→1, TWAIT GE 1).
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
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, float, 1, TILE_CAP, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(commCtx->rankId);

    if (nranks <= 1 || nranks > kMaxSupportedRanks || (ALLREDUCE_COUNT % static_cast<size_t>(nranks)) != 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const int chunk_elems = static_cast<int>(ALLREDUCE_COUNT / static_cast<size_t>(nranks));
    __gm__ float *chunks = scratch;
    __gm__ int32_t *signal_base =
        reinterpret_cast<__gm__ int32_t *>(scratch + static_cast<size_t>(nranks * chunk_elems));

    // Two tiles: one for local chunk ops, one for remote chunk receive.
    TileData chunkTile(1, TILE_CAP);
    TileData recvTile(1, TILE_CAP);
    TASSIGN(chunkTile, 0x0);
    TASSIGN(recvTile, 0x10000);  // 64 KB offset — total 2×16 KB = 32 KB < 256 KB

    // ------------------------------------------------------------------
    // Phase 1: stage-in — partition local input into P chunk slots.
    // Tiled: each TLOAD/TSTORE loads at most TILE_CAP elements per block.
    // ------------------------------------------------------------------
    for (int chunk = 0; chunk < nranks; ++chunk) {
        for (int blk = 0; blk < chunk_elems; blk += (int)TILE_CAP) {
            int cur = ((int)TILE_CAP < chunk_elems - blk) ? (int)TILE_CAP : (chunk_elems - blk);
            __gm__ float *dst = chunks + static_cast<size_t>(chunk * chunk_elems + blk);
            __gm__ float *src = input + static_cast<size_t>(chunk * chunk_elems + blk);
            ShapeDyn shape(1, 1, 1, 1, cur);
            StrideDyn stride(cur, cur, cur, cur, 1);
            Global srcG(src, shape, stride);
            Global dstG(dst, shape, stride);
            TLOAD(chunkTile, srcG);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstG, chunkTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
    pipe_barrier(PIPE_ALL);

    int round = 0;

    // ------------------------------------------------------------------
    // Phase 2: reduce-scatter — (P-1) ring steps.
    // Barrier, then TLOAD remote chunk + TLOAD local chunk + TADD + TSTORE.
    // Each bulk operation tiled in TILE_CAP blocks.
    // ------------------------------------------------------------------
    for (int step = 1; step < nranks; ++step) {
        const int recv_add_idx = (my_rank - step - 1 + nranks) % nranks;

        RoundBarrier(commCtx, signal_base + round * kMaxSupportedRanks, my_rank, nranks);
        ++round;

        const int left = (my_rank - 1 + nranks) % nranks;
        const int left_send_idx = (left - step + nranks) % nranks;

        for (int blk = 0; blk < chunk_elems; blk += (int)TILE_CAP) {
            int cur = ((int)TILE_CAP < chunk_elems - blk) ? (int)TILE_CAP : (chunk_elems - blk);
            ShapeDyn shape(1, 1, 1, 1, cur);
            StrideDyn stride(cur, cur, cur, cur, 1);

            // TLOAD remote chunk block.
            {
                __gm__ float *remote_chunk =
                    CommRemotePtr(commCtx, chunks + static_cast<size_t>(left_send_idx * chunk_elems + blk), left);
                Global remoteG(remote_chunk, shape, stride);
                TLOAD(recvTile, remoteG);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            }

            // TLOAD local chunk block, TADD, TSTORE back.
            {
                __gm__ float *local_ptr = chunks + static_cast<size_t>(recv_add_idx * chunk_elems + blk);
                Global localG(local_ptr, shape, stride);
                TLOAD(chunkTile, localG);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                TADD(chunkTile, chunkTile, recvTile);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TSTORE(localG, chunkTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }
        }
        pipe_barrier(PIPE_ALL);
    }

    // ------------------------------------------------------------------
    // Phase 3: allgather — (P-1) ring steps.
    // Barrier, then TLOAD remote chunk + TSTORE local. Tiled same way.
    // ------------------------------------------------------------------
    for (int step = 1; step < nranks; ++step) {
        const int recv_idx = (my_rank - step + nranks) % nranks;

        RoundBarrier(commCtx, signal_base + round * kMaxSupportedRanks, my_rank, nranks);
        ++round;

        const int left = (my_rank - 1 + nranks) % nranks;
        const int left_send_idx = (left - step + 1 + nranks) % nranks;

        for (int blk = 0; blk < chunk_elems; blk += (int)TILE_CAP) {
            int cur = ((int)TILE_CAP < chunk_elems - blk) ? (int)TILE_CAP : (chunk_elems - blk);
            ShapeDyn shape(1, 1, 1, 1, cur);
            StrideDyn stride(cur, cur, cur, cur, 1);

            // TLOAD remote chunk block.
            {
                __gm__ float *remote_chunk =
                    CommRemotePtr(commCtx, chunks + static_cast<size_t>(left_send_idx * chunk_elems + blk), left);
                Global remoteG(remote_chunk, shape, stride);
                TLOAD(recvTile, remoteG);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            }

            // TSTORE into local chunk slot.
            {
                __gm__ float *dst = chunks + static_cast<size_t>(recv_idx * chunk_elems + blk);
                Global dstG(dst, shape, stride);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TSTORE(dstG, recvTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }
        }
        pipe_barrier(PIPE_ALL);
    }

    // ------------------------------------------------------------------
    // Phase 4: stage-out — write concatenated chunks into local output.
    // Tiled same way as stage-in.
    // ------------------------------------------------------------------
    for (int chunk = 0; chunk < nranks; ++chunk) {
        for (int blk = 0; blk < chunk_elems; blk += (int)TILE_CAP) {
            int cur = ((int)TILE_CAP < chunk_elems - blk) ? (int)TILE_CAP : (chunk_elems - blk);
            __gm__ float *dst = output + static_cast<size_t>(chunk * chunk_elems + blk);
            __gm__ float *src = chunks + static_cast<size_t>(chunk * chunk_elems + blk);
            ShapeDyn shape(1, 1, 1, 1, cur);
            StrideDyn stride(cur, cur, cur, cur, 1);
            Global srcG(src, shape, stride);
            Global dstG(dst, shape, stride);
            TLOAD(chunkTile, srcG);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstG, chunkTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
    pipe_barrier(PIPE_ALL);
}
