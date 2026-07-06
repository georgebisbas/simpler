/*
 * Minimal HCCL comm kernel (simpler runtime) — strips the AllScan down to just
 * the ring notify/wait handshake over the symmetric HCCL window. NO data
 * movement, NO tiles, NO compute:
 *
 *   rank p (p>0)      : TWAIT its own signal >= epoch
 *   rank p (p<last)   : remote TNOTIFY (AtomicAdd 1) rank p+1's signal
 *
 * If THIS hangs, the pto/simpler comm primitive (TNOTIFY/TWAIT over the
 * allocate_domain window / CommContext) is the culprit — independent of any
 * ZeCO/GLA computation.
 *
 * args: [0]=scratch tensor (>=1 int32 signal slot in the HCCL window) INOUT,
 *       scalar[0]=nranks, scalar[1]=epoch (1-based run index),
 *       scalar[2]=CommContext device pointer.
 */

#include <cstdint>

#include <pto/pto-inst.hpp>
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "platform_comm/comm_context.h"
#include "tensor.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommContext *ctx, __gm__ T *localPtr, int pe) {
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(localPtr) - localBase;
    return reinterpret_cast<__gm__ T *>(ctx->windowsIn[pe] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *scratch_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    int nranks = static_cast<int>(args[1]);
    int32_t epoch = static_cast<int32_t>(args[2]);
    __gm__ CommContext *commCtx = reinterpret_cast<__gm__ CommContext *>(args[3]);

    __gm__ int32_t *signal =
        reinterpret_cast<__gm__ int32_t *>(scratch_tensor->buffer.addr) + scratch_tensor->start_offset;

    int my_rank = static_cast<int>(commCtx->rankId);
    int last_rank = nranks - 1;

    if (my_rank != 0) {
        pto::comm::Signal sig(signal);
        pto::comm::TWAIT(sig, epoch, pto::comm::WaitCmp::GE);
        pipe_barrier(PIPE_ALL);
    }

    if (my_rank != last_rank) {
        int peer = my_rank + 1;
        __gm__ int32_t *peer_signal = CommRemotePtr(commCtx, signal, peer);
        pto::comm::Signal nsig(peer_signal);
        pto::comm::TNOTIFY(nsig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
    }
    pipe_barrier(PIPE_ALL);
}
