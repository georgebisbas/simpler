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
 * Bidirectional Ring AllReduce — two parallel unidirectional rings on
 * disjoint halves of the data, sharing RoundBarrier per step.
 *
 * Ring 0 (clockwise, →right neighbour): first half of each chunk's elements.
 * Ring 1 (counter-clockwise, →left neighbour): second half of each chunk's elements.
 *
 * Both rings run reduce-scatter (P−1 barrier rounds) then allgather (P−1
 * barrier rounds). Total: 2(P−1) barrier rounds.
 *
 * All bulk TLOAD/TSTORE/TPUT operations are tiled in TILE_CAP blocks to
 * keep UB usage under 256 KB regardless of ALLREDUCE_COUNT.
 *
 * On sim (POSIX shared memory): raw float* arithmetic into shared memory.
 * On NPU hardware: TPUT<AtomicAdd/AtomicNone> for remote DMA writes.
 *
 * Scratch layout (per rank, in HCCL window):
 *   [0 .. P*subchunk)              ring0 chunks (clockwise ring)
 *   [P*subchunk .. 2*P*subchunk)   ring1 chunks (counter-clockwise ring)
 *   tail                           (2*(P−1)+1)*kMaxSupportedRanks int32 barrier rows
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

    if (nranks <= 1 || nranks > kMaxSupportedRanks || (ALLREDUCE_COUNT % (2ULL * static_cast<size_t>(nranks))) != 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const int subchunk_elems = static_cast<int>(ALLREDUCE_COUNT / (2 * static_cast<size_t>(nranks)));
    const int chunk_elems = 2 * subchunk_elems;
    const int ring_stride = nranks * subchunk_elems;

    __gm__ float *ring0 = scratch;
    __gm__ float *ring1 = scratch + ring_stride;
    __gm__ int32_t *signal_base = reinterpret_cast<__gm__ int32_t *>(scratch + static_cast<size_t>(2 * ring_stride));

    using TileSub = pto::Tile<pto::TileType::Vec, float, 1, TILE_CAP, pto::BLayout::RowMajor, -1, -1>;
    TileSub stageTile(1, TILE_CAP);
    TASSIGN(stageTile, 0x0);
#ifndef __CPU_SIM
    TileSub pushTile(1, TILE_CAP);
    TASSIGN(pushTile, 0x10000);
#endif

    // ------------------------------------------------------------------
    // Phase 1: stage-in — split each logical chunk into ring0 / ring1.
    // Tiled: each subchunk block loaded/stored separately.
    // ------------------------------------------------------------------
    for (int c = 0; c < nranks; ++c) {
        for (int blk = 0; blk < subchunk_elems; blk += (int)TILE_CAP) {
            int cur = ((int)TILE_CAP < subchunk_elems - blk) ? (int)TILE_CAP : (subchunk_elems - blk);
            ShapeDyn shape(1, 1, 1, 1, cur);
            StrideDyn stride(cur, cur, cur, cur, 1);

            // ring0: first half of chunk c.
            {
                __gm__ float *dst = ring0 + static_cast<size_t>(c * subchunk_elems + blk);
                __gm__ float *src = input + static_cast<size_t>(c * chunk_elems + blk);
                GT srcG(src, shape, stride);
                GT dstG(dst, shape, stride);
                TLOAD(stageTile, srcG);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                TSTORE(dstG, stageTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }
            // ring1: second half of chunk c.
            {
                __gm__ float *dst = ring1 + static_cast<size_t>(c * subchunk_elems + blk);
                __gm__ float *src = input + static_cast<size_t>(c * chunk_elems + subchunk_elems + blk);
                GT srcG(src, shape, stride);
                GT dstG(dst, shape, stride);
                TLOAD(stageTile, srcG);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
                TSTORE(dstG, stageTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
            }
        }
    }
    pipe_barrier(PIPE_ALL);

    const int left = (my_rank - 1 + nranks) % nranks;
    const int right = (my_rank + 1) % nranks;

    // ------------------------------------------------------------------
    // Phase 2: reduce-scatter — P−1 barrier rounds.
    // Ring0 (cw→right): push ring0[(r-step+P)%P] → right via AtomicAdd.
    // Ring1 (ccw→left): push ring1[(r+step+P)%P] → left via AtomicAdd.
    // Each TPUT tiled: subchunk broken into TILE_CAP blocks.
    // ------------------------------------------------------------------
    for (int step = 1; step < nranks; ++step) {
        RoundBarrier(commCtx, signal_base + (step - 1) * kMaxSupportedRanks, my_rank, nranks);

        // Ring0 reduce-scatter.
        {
            const int idx = (my_rank - step + nranks) % nranks;
            __gm__ float *base_src = ring0 + static_cast<size_t>(idx * subchunk_elems);
            __gm__ float *base_dst = CommRemotePtr(commCtx, base_src, right);
            for (int blk = 0; blk < subchunk_elems; blk += (int)TILE_CAP) {
                int cur = ((int)TILE_CAP < subchunk_elems - blk) ? (int)TILE_CAP : (subchunk_elems - blk);
                ShapeDyn shape(1, 1, 1, 1, cur);
                StrideDyn stride(cur, cur, cur, cur, 1);
                __gm__ float *src = base_src + blk;
                __gm__ float *dst = base_dst + blk;
#if defined(__CPU_SIM)
                for (int i = 0; i < cur; ++i) dst[i] += src[i];
#else
                GT srcG(src, shape, stride);
                GT dstG(dst, shape, stride);
                pto::comm::TPUT<pto::AtomicType::AtomicAdd>(dstG, srcG, pushTile);
#endif
            }
        }

        // Ring1 reduce-scatter.
        {
            const int idx = (my_rank + step + nranks) % nranks;
            __gm__ float *base_src = ring1 + static_cast<size_t>(idx * subchunk_elems);
            __gm__ float *base_dst = CommRemotePtr(commCtx, base_src, left);
            for (int blk = 0; blk < subchunk_elems; blk += (int)TILE_CAP) {
                int cur = ((int)TILE_CAP < subchunk_elems - blk) ? (int)TILE_CAP : (subchunk_elems - blk);
                ShapeDyn shape(1, 1, 1, 1, cur);
                StrideDyn stride(cur, cur, cur, cur, 1);
                __gm__ float *src = base_src + blk;
                __gm__ float *dst = base_dst + blk;
#if defined(__CPU_SIM)
                for (int i = 0; i < cur; ++i) dst[i] += src[i];
#else
                GT srcG(src, shape, stride);
                GT dstG(dst, shape, stride);
                pto::comm::TPUT<pto::AtomicType::AtomicAdd>(dstG, srcG, pushTile);
#endif
            }
        }

        pipe_barrier(PIPE_ALL);
    }

    // ------------------------------------------------------------------
    // Phase 3: allgather — P−1 barrier rounds.
    // Ring0: TPUT<AtomicNone> from ring0[idx] → right.
    // Ring1: TPUT<AtomicNone> from ring1[idx] → left.
    // Tiled same way — subchunk blocks.
    // ------------------------------------------------------------------
    for (int step = 1; step < nranks; ++step) {
        const int rs_rounds = nranks - 1;
        RoundBarrier(commCtx, signal_base + (rs_rounds + step - 1) * kMaxSupportedRanks, my_rank, nranks);

        // Ring0 AG.
        {
            const int idx = (my_rank - step + 1 + nranks) % nranks;
            __gm__ float *base_src = ring0 + static_cast<size_t>(idx * subchunk_elems);
            __gm__ float *base_dst = CommRemotePtr(commCtx, base_src, right);
            for (int blk = 0; blk < subchunk_elems; blk += (int)TILE_CAP) {
                int cur = ((int)TILE_CAP < subchunk_elems - blk) ? (int)TILE_CAP : (subchunk_elems - blk);
                ShapeDyn shape(1, 1, 1, 1, cur);
                StrideDyn stride(cur, cur, cur, cur, 1);
                __gm__ float *src = base_src + blk;
                __gm__ float *dst = base_dst + blk;
#if defined(__CPU_SIM)
                for (int i = 0; i < cur; ++i) dst[i] = src[i];
#else
                GT srcG(src, shape, stride);
                GT dstG(dst, shape, stride);
                pto::comm::TPUT<pto::AtomicType::AtomicNone>(dstG, srcG, pushTile);
#endif
            }
        }

        // Ring1 AG.
        {
            const int idx = (my_rank + step - 1 + nranks) % nranks;
            __gm__ float *base_src = ring1 + static_cast<size_t>(idx * subchunk_elems);
            __gm__ float *base_dst = CommRemotePtr(commCtx, base_src, left);
            for (int blk = 0; blk < subchunk_elems; blk += (int)TILE_CAP) {
                int cur = ((int)TILE_CAP < subchunk_elems - blk) ? (int)TILE_CAP : (subchunk_elems - blk);
                ShapeDyn shape(1, 1, 1, 1, cur);
                StrideDyn stride(cur, cur, cur, cur, 1);
                __gm__ float *src = base_src + blk;
                __gm__ float *dst = base_dst + blk;
#if defined(__CPU_SIM)
                for (int i = 0; i < cur; ++i) dst[i] = src[i];
#else
                GT srcG(src, shape, stride);
                GT dstG(dst, shape, stride);
                pto::comm::TPUT<pto::AtomicType::AtomicNone>(dstG, srcG, pushTile);
#endif
            }
        }

        pipe_barrier(PIPE_ALL);
    }

    // Final sync barrier.
    if (nranks > 1) {
        RoundBarrier(commCtx, signal_base + (2 * (nranks - 1)) * kMaxSupportedRanks, my_rank, nranks);
    }

    // ------------------------------------------------------------------
    // Phase 4: stage-out — recombine ring0/ring1 halves into output.
    // Tiled: each subchunk block loaded/stored separately.
    // ------------------------------------------------------------------
    for (int c = 0; c < nranks; ++c) {
        for (int blk = 0; blk < subchunk_elems; blk += (int)TILE_CAP) {
            int cur = ((int)TILE_CAP < subchunk_elems - blk) ? (int)TILE_CAP : (subchunk_elems - blk);
            ShapeDyn shape(1, 1, 1, 1, cur);
            StrideDyn stride(cur, cur, cur, cur, 1);

            // First half of chunk c ← ring0.
            {
                __gm__ float *dst = output + static_cast<size_t>(c * chunk_elems + blk);
                __gm__ float *src = ring0 + static_cast<size_t>(c * subchunk_elems + blk);
                GT srcG(src, shape, stride);
                GT dstG(dst, shape, stride);
                TLOAD(stageTile, srcG);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                TSTORE(dstG, stageTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }
            // Second half of chunk c ← ring1.
            {
                __gm__ float *dst = output + static_cast<size_t>(c * chunk_elems + subchunk_elems + blk);
                __gm__ float *src = ring1 + static_cast<size_t>(c * subchunk_elems + blk);
                GT srcG(src, shape, stride);
                GT dstG(dst, shape, stride);
                TLOAD(stageTile, srcG);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
                TSTORE(dstG, stageTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
            }
        }
    }
    pipe_barrier(PIPE_ALL);
}
