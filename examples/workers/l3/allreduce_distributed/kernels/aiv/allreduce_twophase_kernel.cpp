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
 * Two-Phase Mesh AllReduce kernel — reduce-scatter + allgather with mesh barriers.
 *
 * Phase 1 (stage-in):   partition input → P chunk slots in window
 * Phase 2 (RS barrier): mesh barrier (all-to-all notify/wait)
 * Phase 3 (reduce):     acc[my_rank] = sum over peers of peer.scratch[my_rank]
 * Phase 4 (AG barrier): mesh barrier
 * Phase 5 (gather):     for r in P: read peer[r].scratch[r] → output[r*C]
 *
 * All bulk TLOAD/TSTORE/TADD operations tiled in TILE_CAP blocks (256 KB UB).
 *
 * args layout:
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

AICORE inline void MeshBarrier(__gm__ CommContext *ctx, __gm__ int32_t *signal_row, int my_rank, int nranks) {
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

    int my_rank = static_cast<int>(commCtx->rankId);

    if (nranks <= 1 || nranks > kMaxSupportedRanks || (ALLREDUCE_COUNT % static_cast<size_t>(nranks)) != 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const int chunk_elems = static_cast<int>(ALLREDUCE_COUNT / static_cast<size_t>(nranks));

    __gm__ int32_t *signal_rs = reinterpret_cast<__gm__ int32_t *>(scratch + nranks * chunk_elems);
    __gm__ int32_t *signal_ag = signal_rs + kMaxSupportedRanks;

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, float, 1, TILE_CAP, pto::BLayout::RowMajor, -1, -1>;

    TileData chunkTile(1, TILE_CAP);
    TileData accTile(1, TILE_CAP);
    TileData recvTile(1, TILE_CAP);
    TileData stageTile(1, TILE_CAP);
    TASSIGN(chunkTile, 0x0);
    TASSIGN(accTile, 0x4000);
    TASSIGN(recvTile, 0x8000);
    TASSIGN(stageTile, 0xc000);

    // Phase 1: stage-in, tiled per chunk.
    for (int chunk = 0; chunk < nranks; ++chunk) {
        __gm__ float *base_dst = scratch + chunk * chunk_elems;
        __gm__ float *base_src = input + chunk * chunk_elems;
        for (int blk = 0; blk < chunk_elems; blk += (int)TILE_CAP) {
            int cur = ((int)TILE_CAP < chunk_elems - blk) ? (int)TILE_CAP : (chunk_elems - blk);
            __gm__ float *dst = base_dst + blk;
            __gm__ float *src = base_src + blk;
            ShapeDyn shape(1, 1, 1, 1, cur);
            StrideDyn stride(cur, cur, cur, cur, 1);
            Global srcG(src, shape, stride);
            Global dstG(dst, shape, stride);
            TLOAD(stageTile, srcG);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstG, stageTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
    pipe_barrier(PIPE_ALL);

    // Phase 2: RS barrier.
    MeshBarrier(commCtx, signal_rs, my_rank, nranks);

    // Phase 3: reduce-scatter.
    // Per block: load my local → accTile, TADD each peer's block, store back.
    for (int blk = 0; blk < chunk_elems; blk += (int)TILE_CAP) {
        int cur = ((int)TILE_CAP < chunk_elems - blk) ? (int)TILE_CAP : (chunk_elems - blk);
        ShapeDyn shape(1, 1, 1, 1, cur);
        StrideDyn stride(cur, cur, cur, cur, 1);
        {
            __gm__ float *local_ptr = scratch + my_rank * chunk_elems + blk;
            Global localG(local_ptr, shape, stride);
            TLOAD(accTile, localG);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        }
        for (int peer = 0; peer < nranks; ++peer) {
            if (peer == my_rank) continue;
            __gm__ float *remote_chunk = CommRemotePtr(commCtx, scratch + my_rank * chunk_elems + blk, peer);
            Global remoteG(remote_chunk, shape, stride);
            TLOAD(recvTile, remoteG);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            TADD(accTile, accTile, recvTile);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }
        {
            __gm__ float *local_ptr = scratch + my_rank * chunk_elems + blk;
            Global localG(local_ptr, shape, stride);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(localG, accTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
    pipe_barrier(PIPE_ALL);

    // Phase 4: AG barrier.
    MeshBarrier(commCtx, signal_ag, my_rank, nranks);

    // Phase 5: allgather, tiled per chunk.
    for (int r = 0; r < nranks; ++r) {
        __gm__ float *base_remote = CommRemotePtr(commCtx, scratch + r * chunk_elems, r);
        __gm__ float *base_output = output + r * chunk_elems;
        for (int blk = 0; blk < chunk_elems; blk += (int)TILE_CAP) {
            int cur = ((int)TILE_CAP < chunk_elems - blk) ? (int)TILE_CAP : (chunk_elems - blk);
            ShapeDyn shape(1, 1, 1, 1, cur);
            StrideDyn stride(cur, cur, cur, cur, 1);
            __gm__ float *remote_ptr = base_remote + blk;
            __gm__ float *out_ptr = base_output + blk;
            Global remoteG(remote_ptr, shape, stride);
            Global outputG(out_ptr, shape, stride);
            TLOAD(stageTile, remoteG);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(outputG, stageTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
    pipe_barrier(PIPE_ALL);
}
