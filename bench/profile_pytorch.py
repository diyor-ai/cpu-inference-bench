"""Stage 3a: torch.profiler (CPU) for ResNet-18 FP32 / INT8 (FX x86). Profiled runs only - NOT timing results."""
import json, sys, time
from pathlib import Path
from collections import defaultdict
import torch
from torch.profiler import profile, ProfilerActivity
from common import common_args, get_model, get_input, save

args = common_args()
torch.set_num_threads(args.threads)
x = torch.from_numpy(get_input(args.batch))
if args.precision == "fp32":
    model = get_model(pretrained=True)
else:
    sys.path.insert(0, str(Path(__file__).parent / "quant"))
    from pt_int8 import build
    model, _, _ = build()
    assert torch.backends.quantized.engine == "x86"

def group(name):
    n = name.lower()
    if "quantize_per" in n or "dequantize" in n:   # aten::quantize_per_tensor, aten::dequantize, ...
        return "quantize/dequantize"
    if "conv" in n:
        return "conv"
    if "linear" in n or "addmm" in n or n in ("aten::mm", "aten::matmul"):
        return "linear"
    return "other"

with torch.inference_mode():
    for _ in range(10):
        model(x)
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(args.iters):
            model(x)
    wall_ms = (time.perf_counter() - t0) * 1000 / args.iters

ops = []
for e in prof.key_averages():
    ops.append({"name": e.key, "count": e.count, "self_cpu_us_total": e.self_cpu_time_total,
                "cpu_us_total": e.cpu_time_total, "group": group(e.key)})
ops.sort(key=lambda o: -o["self_cpu_us_total"])
tot = sum(o["self_cpu_us_total"] for o in ops)
groups = defaultdict(float)
for o in ops:
    groups[o["group"]] += o["self_cpu_us_total"]
res = {"iters": args.iters, "profiled_wall_ms_per_iter": round(wall_ms, 3),
       "sum_self_cpu_ms_per_iter": round(tot / 1000 / args.iters, 3),
       "group_self_cpu_ms_per_iter": {k: round(v / 1000 / args.iters, 3) for k, v in groups.items()},
       "group_share_pct": {k: round(100 * v / tot, 1) for k, v in groups.items()}, "ops": ops}
print(f"pytorch {args.precision} b{args.batch}: profiled wall {wall_ms:.2f} ms/iter; shares {res['group_share_pct']}")
for o in ops[:8]:
    print(f"   {o['name']:45s} n={o['count']:5d} self {o['self_cpu_us_total']/1000/args.iters:9.3f} ms/iter  [{o['group']}]")
save(f"pytorch_{args.precision}_b{args.batch}", {"framework": "pytorch", "version": torch.__version__,
     "precision": args.precision, "batch": args.batch, "threads": args.threads, "PROFILED_NOT_TIMING": True, "result": res},
     out=f"{args.out}/raw")
