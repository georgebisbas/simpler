# [Bug] Simpler L3 multi-device `Worker.init()` hangs forever on a2a3 HW — parent busy-spins with no timeout waiting for a chip child that never reaches `_INIT_DONE`

<!-- Draft for GitHub issue (template: bug_report.yml). Fill dropdowns as noted. -->

**Component:** Runtime (simpler — L3 multi-device Worker bootstrap / chip-child init)
**NPU Kind:** Ascend 910B2 (a2a3)
**Host Platform:** Linux (aarch64)
**Runtime:** `/opt/pypto/runtime/python/simpler/worker.py`, `tensormap_and_ringbuffer`

## Description

Standing up a **multi-device** simpler L3 `Worker` on a2a3 hardware **hangs
indefinitely inside `Worker.init()`** — before any HCCL communicator is created,
before any comm domain is allocated, and before any kernel runs. The hang is in
the **chip-child bootstrap barrier**, not in any collective or in user compute.

The reproducer (`repro.py`, attached) strips out **all** ZeCO/GLA computation: it
just builds a 2-rank L3 `Worker` and (optionally) submits a trivial ring
notify/wait kernel. Three stages localise the failure:

| stage    | what it does                                         | a2a3sim | a2a3 HW |
|----------|------------------------------------------------------|:-------:|:-------:|
| `init`   | `Worker(level=3, device_ids=[a,b]).register().init()`|  PASS   | **HANG**|
| `domain` | + `orch.allocate_domain()` (HCCL window), no tasks    |  PASS   | (n/a — never reached) |
| `comm`   | + submit minimal ring `TNOTIFY`/`TWAIT`              |  PASS   | (n/a — never reached) |

It **hangs at the very first stage (`init`)** on hardware, and **passes all
stages on the simulator** — so this is not our kernel logic and not HCCL data
transfer; it is the simpler runtime's per-device chip-child bootstrap on HW.

## Steps to Reproduce

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh

# Simulator — passes:
python repro.py -p a2a3sim -d 0-1 -s comm
#   [repro] platform=a2a3sim devices=[0, 1] P=2 stage=comm
#   STAGE init OK
#   STAGE comm OK

# Real HW — hangs at worker init (no output; must be SIGKILLed):
python repro.py -p a2a3 -d 2,7 -s init
#   [repro] platform=a2a3 devices=[2, 7] P=2 stage=init
#   (hangs forever — never prints "STAGE init OK")
```

Run on devices confirmed free in `npu-smi info` (see operational note below).

## Root cause (localized in the runtime)

`Worker(level=3, device_ids=[...]).init()` forks one **chip child per device**,
then the parent runs a **cross-chip init barrier** that **busy-spins with no
timeout** until every child publishes `_INIT_DONE`:

`worker.py` (~line 3000):
```python
for shm in self._chip_shms:
    addr = _buffer_field_addr(shm.buf, _OFF_STATE)
    while _mailbox_load_i32(addr) != _INIT_DONE:   # <-- busy-spin, NO timeout
        pass
    _mailbox_store_i32(addr, _IDLE)
```

Each child, in `_chip_process_loop` (~line 1123), calls
`ChipWorker.init(device_id, ...)` — which attaches the process to the NPU
(`aclrtSetDevice` / stream creation). If that device call **hangs or raises** on
HW, the child never publishes `_INIT_DONE`, so the parent spins forever.

The failure path is acknowledged as broken in-code (`_chip_process_loop`, ~line
1130):
> `# State handshake for this init-time failure is broken — see KNOWN_ISSUES.md`

i.e. if `ChipWorker.init` **raises**, the child writes an error and `return`s
**without** setting `_INIT_DONE` → the parent's spin never terminates.

`_BOOTSTRAP_WAIT_TIMEOUT_S = 120.0` is *defined* (worker.py ~line 122) but is
**not applied** to this barrier loop — so instead of failing loudly after 120 s,
the parent hangs indefinitely.

## Expected Behavior

`Worker.init()` should either come up, or **fail loudly with a catchable error**
within a bounded time (e.g. `_BOOTSTRAP_WAIT_TIMEOUT_S`). A silent, un-killable
infinite spin is never acceptable — and a child whose `ChipWorker.init` raises
must publish a failed state the parent observes.

## Actual Behavior

`repro.py -p a2a3 -d 2,7 -s init` prints the banner and then **hangs with no
further output**. `SIGTERM` is ignored (the child is in an uninterruptible device
call); the process must be `SIGKILL`ed. The orphaned chip child leaves the NPU
**pinned at AICore 100 % with no owning process** (`npu-smi info`), which
progressively degrades the box (see operational note).

## Additional Context

- **Corroborating evidence** that this is chip-child *device* init, not comm:
  - Sim passes (`a2a3sim`) — chip "devices" attach instantly, barrier clears.
  - Single-device L3 workers (our GLA kernels, no comm) work **when the target
    NPU is clean**, but the *same* single-device init **hangs once that NPU is in
    a bad state** — nothing comm/HCCL-specific is required to trigger it.
  - The `ptoisa` ZeCO backend's `torch.distributed` HCCL path
    (`init_process_group("hccl")`, a completely different init) works at P=2/4 on
    the same box — so the HCCL fabric itself is fine.
- **Discovered** building the simpler-runtime ZeCO / GLA sequence-parallel
  operator. All single-device compute (three GLA kernels + the P=1 end-to-end
  ZeCO) validates on this HW; only the multi-device `Worker` bootstrap (needed for
  the AllScan boundary collective) hangs, which blocks P≥2.
- **Suggested fixes (runtime side):**
  1. Apply `_BOOTSTRAP_WAIT_TIMEOUT_S` to the init-barrier spin so a stuck child
     fails loudly instead of hanging forever.
  2. Fix the init-time failure handshake so a child whose `ChipWorker.init` raises
     publishes an observable failed state (the "broken handshake" comment).
  3. Separately: investigate why `ChipWorker.init(device)` stalls on a2a3 HW here
     — it correlates with NPUs left in a bad state by prior killed runs; a device
     reset (`aclrtResetDeviceForce` / `npu-smi ... reset`) between runs may be
     required.

## Operational note (shared box)

Each hung run leaves a chip child stuck in an uninterruptible device call
(D-state); `SIGKILL` orphans it and the NPU stays pinned (`npu-smi` AICore 100 %,
no proc). After a few hung multi-device attempts even single-device init begins to
hang. Before a run: ensure no stray `chip_process` workers, delete stale
`/tmp/barrier_pto_multi_comm_*`, and pick devices showing AICore 0 % in
`npu-smi info`; a full device reset may be needed to recover.
