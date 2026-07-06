/*
 * Minimal comm orchestration shim — one AIV task per chip.
 *   tensor(0) scratch INOUT (HCCL window signal slot)
 *   scalar(0) nranks   scalar(1) epoch   scalar(2) CommContext ptr
 */
#include <stdint.h>
#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
comm_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 4};  // 1 tensor + 3 scalars
}

__attribute__((visibility("default"))) void comm_orchestration(const L2TaskArgs &orch_args) {
    const Tensor &scratch = orch_args.tensor(0).ref();
    L0TaskArgs params;
    params.add_inout(scratch);
    params.add_scalar(orch_args.scalar(0));  // nranks
    params.add_scalar(orch_args.scalar(1));  // epoch
    params.add_scalar(orch_args.scalar(2));  // CommContext
    rt_submit_aiv_task(0, params);
}

}  // extern "C"
