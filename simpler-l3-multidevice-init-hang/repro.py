#!/usr/bin/env python3
"""Minimal HCCL reproducer for the simpler-runtime multi-device hang.

Strips out ALL ZeCO/GLA computation. Just stands up a 2+-rank simpler L3 Worker
and exercises the pto comm path in stages, so we can localise where it hangs:

    stage=init    Worker(level=3, N devices) + register + init()   [HCCL worker init]
    stage=domain  + worker.run(orch that ONLY allocate_domain())   [HCCL window rendezvous]
    stage=comm    + submit the minimal ring notify/wait kernel      [TNOTIFY/TWAIT over window]

Each stage either PRINTS "STAGE <x> OK" (completes) or HANGS (no output) — run
the stages in order to find the first one that hangs. Passes on the simulator.
Self-contained: depends only on the installed ``simpler`` / ``simpler_setup``
packages and the two kernels in ``kernels/`` next to this file.

Usage:
    python repro.py -p a2a3sim -d 0-1 -s comm     # sim: should PASS (init + comm)
    python repro.py -p a2a3   -d 2,7  -s init     # HW: hangs at worker init
    python repro.py -p a2a3   -d 2,7  -s domain
    python repro.py -p a2a3   -d 2,7  -s comm
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

RUNTIME = "tensormap_and_ringbuffer"
KDIR = os.path.join(HERE, "kernels")


def build_chip_callable(platform):
    from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable
    from simpler_setup.elf_parser import extract_text_section
    from simpler_setup.kernel_compiler import KernelCompiler
    from simpler_setup.pto_isa import ensure_pto_isa_root

    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    inc = list(kc.get_orchestration_include_dirs(RUNTIME)) + [str(kc.project_root / "src" / "common")]
    kbytes = kc.compile_incore(source_path=os.path.join(KDIR, "comm_kernel.cpp"),
                               core_type="aiv", pto_isa_root=pto_isa_root, extra_include_dirs=inc)
    if not platform.endswith("sim"):
        kbytes = extract_text_section(kbytes)
    obytes = kc.compile_orchestration(runtime_name=RUNTIME, source_path=os.path.join(KDIR, "comm_orch.cpp"))
    core = CoreCallable.build(signature=[ArgDirection.INOUT], binary=kbytes)
    return ChipCallable.build(signature=[ArgDirection.INOUT], func_name="comm_orchestration",
                              config_name="comm_orchestration_config", binary=obytes,
                              children=[(0, core)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--platform", default="a2a3")
    ap.add_argument("-d", "--device", default="0-1")
    ap.add_argument("-s", "--stage", default="comm", choices=["init", "domain", "comm"])
    cli = ap.parse_args()

    spec = cli.device
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-")); devs = list(range(lo, hi + 1))
    else:
        devs = [int(x) for x in spec.split(",") if x != ""]
    P = len(devs)
    print(f"[repro] platform={cli.platform} devices={devs} P={P} stage={cli.stage}", flush=True)

    from simpler.task_interface import (CallConfig, CommBufferSpec, DataType,
                                        TaskArgs, TensorArgType)
    from simpler.task_interface import Tensor as ContinuousTensor
    from simpler.worker import Worker
    from simpler_setup.torch_interop import make_tensor_arg  # noqa: F401 (parity)

    NSLOTS = 8  # int32 signal slots (>=1 needed); 8 for alignment
    cc = build_chip_callable(cli.platform)
    worker = Worker(level=3, platform=cli.platform, runtime=RUNTIME, device_ids=devs, num_sub_workers=0)
    cid = worker.register(cc)
    worker.init()
    print("STAGE init OK", flush=True)
    if cli.stage == "init":
        worker.close(); return 0

    def domain(orch, n_slots):
        return orch.allocate_domain(
            name="minrepro", workers=list(range(P)), window_size=4 * 1024,
            buffers=[CommBufferSpec(name="scratch", dtype="int32", count=n_slots, nbytes=n_slots * 4)])

    if cli.stage == "domain":
        def orch_fn(orch, _a, cfg):
            with domain(orch, NSLOTS):
                pass  # allocate + free the HCCL window, submit nothing
        worker.run(orch_fn, args=None, config=CallConfig())
        print("STAGE domain OK", flush=True)
        worker.close(); return 0

    # stage == comm: allocate domain + submit the minimal ring notify/wait
    def orch_fn(orch, _a, cfg):
        # Tell the parent we're entering the device-critical section
        import os; os.write(1, b"ENTER_KERNEL\n")
        with domain(orch, NSLOTS) as handle:
            for i in range(P):
                dom = handle[i]
                a = TaskArgs()
                a.add_tensor(ContinuousTensor.make(
                    data=dom.buffer_ptrs["scratch"], shapes=(NSLOTS,),
                    dtype=DataType.INT32, child_memory=True), TensorArgType.INOUT)
                a.add_scalar(P)          # nranks
                a.add_scalar(1)          # epoch (window zeroed at alloc)
                a.add_scalar(dom.device_ctx)
                orch.submit_next_level(cid, a, cfg, worker=i)
    worker.run(orch_fn, args=None, config=CallConfig())
    print("STAGE comm OK", flush=True)
    worker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
