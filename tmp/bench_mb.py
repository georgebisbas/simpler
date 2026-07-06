"""Benchmark allreduce modes at 1MB/rank (ALLREDUCE_COUNT=1048576). P=2,4,8."""
from __future__ import annotations
import math, os, sys, torch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
sys.path.insert(0,"/opt/simpler/examples/workers/l3/allreduce_distributed")
from main import ALLREDUCE_COUNT, build_chip_callable, compute_scratch_params, expected_output
from simpler.task_interface import CallConfig, CommBufferSpec, DataType, TaskArgs, Tensor, TensorArgType
from simpler.worker import Worker
from simpler_setup.torch_interop import make_tensor_arg

def bench(mode, devs, W, R):
    n=len(devs); fe,sn,ws=compute_scratch_params(mode,n)
    ins=[torch.tensor([float(i+r*100) for i in range(ALLREDUCE_COUNT)],dtype=torch.float32).share_memory_() for r in range(n)]
    outs=[torch.zeros(ALLREDUCE_COUNT,dtype=torch.float32).share_memory_() for _ in range(n)]
    golden=torch.tensor(expected_output(n),dtype=torch.float32)
    cc=build_chip_callable("a2a3",mode,None)
    w=Worker(level=3,platform="a2a3",runtime="tensormap_and_ringbuffer",device_ids=devs,num_sub_workers=0)
    h=w.register(cc); w.init()
    ias=[make_tensor_arg(t) for t in ins]; oas=[make_tensor_arg(t) for t in outs]
    def orch(orch,_,cfg):
        with orch.allocate_domain(name="d",workers=list(range(n)),window_size=ws,
            buffers=[CommBufferSpec(name="s",dtype="float32",count=fe,nbytes=sn)]) as hnd:
            for i in range(n):
                d=hnd[i]; a=TaskArgs()
                a.add_tensor(ias[i],TensorArgType.INPUT); a.add_tensor(oas[i],TensorArgType.OUTPUT_EXISTING)
                a.add_tensor(Tensor.make(data=d.buffer_ptrs["s"],shapes=(fe,),dtype=DataType.FLOAT32,child_memory=True),TensorArgType.INOUT)
                a.add_scalar(d.domain_size); a.add_scalar(d.device_ctx); orch.submit_next_level(h,a,cfg,worker=i)
    ht=[]
    for rnd in range(W+R):
        rt=w.run(orch,args=None,config=CallConfig())
        if rnd>=W: ht.append(rt.host_wall_us)
    ok=all(float(torch.max(torch.abs(outs[i]-golden)))<=1e-3 for i in range(n))
    w.close()
    return ht, ok

def fmt(vals):
    n=len(vals); m=sum(vals)/n; s=sorted(vals); p50=s[n//2]
    return m, p50, min(vals), max(vals)

MB=ALLREDUCE_COUNT*4//(1024*1024)
W,R=3,10

for P in [2,4,8]:
    devs=list(range(P))
    modes=["onephase","twophase","ring","bidirectional_ring"]
    if P==2: modes.append("ibing")
    print(f"\n=== P={P} ({MB}MB/rank) devices={devs} ===")
    res={}
    for m in modes:
        print(f"  [{m}] ",end="",flush=True)
        try:
            ht,ok=bench(m,devs,W,R)
            avg,p50,mn,mx=fmt(ht)
            print(f"avg={avg:.0f} p50={p50:.0f} mn={mn:.0f} mx={mx:.0f} ok={'Y' if ok else 'N'}")
            res[m]=(avg,p50)
        except Exception as e: print(f"FAILED: {e}")
    if "ring" in res:
        ra=res["ring"][0]
        print(f"  {'Mode':24s} {'avg(us)':>10s} {'vs ring':>10s}")
        for m in modes:
            if m in res:
                a,_=res[m]; d=f"{(a-ra)/ra*100:+.1f}%" if m!="ring" else "(baseline)"
                print(f"  {m:24s} {a:10.0f} {d:>10s}")
print("\nDone")
