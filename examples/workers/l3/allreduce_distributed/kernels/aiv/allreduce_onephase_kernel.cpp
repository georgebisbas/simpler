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
 * One-Phase Mesh AllReduce kernel — symmetric full-vector sum across peers.
 *
 * Phase 1 (stage-in):   input → my scratch slot (in window)
 * Phase 2 (barrier):    signal matrix + TWAIT cross-rank sync
 * Phase 3 (compute):    for peer in nranks: TLOAD(peer_scratch), TADD(acc)
 * Phase 4 (stage-out):  TSTORE(output, acc)
 *
 * All bulk TLOAD/TSTORE/TADD operations are tiled in TILE_CAP blocks to
 * keep UB usage under 256 KB regardless of ALLREDUCE_COUNT.
 *
 * args layout (passed as ContinuousTensor arg slots):
 *   tensor(0) = input    (host-backed)
 *   tensor(1) = output   (host-backed)
 *   tensor(2) = scratch  (HCCL window slot)
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

static constexpr size_t ALLREDUCE_COUNT = 1048576;
static constexpr int kMaxSupportedRanks = 16;
static constexpr size_t TILE_CAP = 4096;

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommContext *ctx, __gm__ T *localPtr, int pe) {
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T *)(ctx->windowsIn[pe] + offset);
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

    __gm__ int32_t *signal_base = reinterpret_cast<__gm__ int32_t *>(scratch + ALLREDUCE_COUNT);

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, float, 1, TILE_CAP, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(commCtx->rankId);

    if (nranks <= 0 || nranks > kMaxSupportedRanks) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    TileData stageTile(1, TILE_CAP);
    TileData accTile(1, TILE_CAP);
    TileData recvTile(1, TILE_CAP);
    TASSIGN(stageTile, 0x0);
    TASSIGN(accTile, 0x4000);
    TASSIGN(recvTile, 0x8000);

    // ------------------------------------------------------------------
    // Phase 1: stage-in — copy local input into scratch slot.
    // Tiled: ALLREDUCE_COUNT elements broken into TILE_CAP blocks.
    // ------------------------------------------------------------------
    for (int blk = 0; blk < (int)ALLREDUCE_COUNT; blk += (int)TILE_CAP) {
        int cur = ((int)TILE_CAP < (int)ALLREDUCE_COUNT - blk) ? (int)TILE_CAP : ((int)ALLREDUCE_COUNT - blk);
        ShapeDyn shape(1, 1, 1, 1, cur);
        StrideDyn stride(cur, cur, cur, cur, 1);
        __gm__ float *src = input + blk;
        __gm__ float *dst = scratch + blk;
        Global srcG(src, shape, stride);
        Global dstG(dst, shape, stride);
        TLOAD(stageTile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstG, stageTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // Phase 2: device barrier — notify all peers that stage-in is visible.
    // ------------------------------------------------------------------
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        __gm__ int32_t *remote_signal = CommRemotePtr(commCtx, signal_base + my_rank, peer);
        pto::comm::Signal sig(remote_signal);
        pto::comm::TNOTIFY(sig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
    }
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        pto::comm::Signal sig(signal_base + peer);
        pto::comm::TWAIT(sig, (int32_t)1, pto::comm::WaitCmp::GE);
    }
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // Phase 3: compute — sum every rank's scratch into accTile, then TSTORE
    // back to scratch.
    //
    // For each block: TLOAD my local data into accTile, then TADD each peer.
    // ------------------------------------------------------------------
    {
        // First block: load my local scratch into accTile.
        int first_blk = 0;
        int cur = ((int)TILE_CAP < (int)ALLREDUCE_COUNT) ? (int)TILE_CAP : (int)ALLREDUCE_COUNT;
        ShapeDyn shape(1, 1, 1, 1, cur);
        StrideDyn stride(cur, cur, cur, cur, 1);
        Global scratchG(scratch + first_blk, shape, stride);
        TLOAD(accTile, scratchG);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Add each peer.
        for (int peer = 0; peer < nranks; ++peer) {
            if (peer == my_rank) continue;
            __gm__ float *remote_scratch = CommRemotePtr(commCtx, scratch + first_blk, peer);
            Global remoteG(remote_scratch, shape, stride);
            TLOAD(recvTile, remoteG);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            TADD(accTile, accTile, recvTile);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }

        // Write first block to output.
        {
            Global outputG(output + first_blk, shape, stride);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(outputG, accTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }

    // Remaining blocks.
    for (int blk = (int)TILE_CAP; blk < (int)ALLREDUCE_COUNT; blk += (int)TILE_CAP) {
        int cur = ((int)TILE_CAP < (int)ALLREDUCE_COUNT - blk) ? (int)TILE_CAP : ((int)ALLREDUCE_COUNT - blk);
        ShapeDyn shape(1, 1, 1, 1, cur);
        StrideDyn stride(cur, cur, cur, cur, 1);

        // Load my scratch block into accTile.
        Global scratchG(scratch + blk, shape, stride);
        TLOAD(accTile, scratchG);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Add each peer's block.
        for (int peer = 0; peer < nranks; ++peer) {
            if (peer == my_rank) continue;
            __gm__ float *remote_scratch = CommRemotePtr(commCtx, scratch + blk, peer);
            Global remoteG(remote_scratch, shape, stride);
            TLOAD(recvTile, remoteG);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            TADD(accTile, accTile, recvTile);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }

        // Write block to output.
        Global outputG(output + blk, shape, stride);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(outputG, accTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }

    pipe_barrier(PIPE_ALL);
}
