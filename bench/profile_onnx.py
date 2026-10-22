"""Stage 3a: ONNX Runtime built-in profiler (enable_profiling) for ResNet-18 FP32 / INT8 (QDQ MinMax). Profiled runs only."""
import json, time
from collections import defaultdict
import onnxruntime as ort
from common import common_args, get_input, save, onnx_path, TEMP

args = common_args()
path = onnx_path(pretrained=True) if args.precision == "fp32" else TEMP / "resnet18_pretrained_int8_ort_minmax.onnx"
so = ort.SessionOptions()
so.intra_op_num_threads = args.threads
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.enable_profiling = True
so.profile_file_prefix = f"{args.out}/raw/ort_profile_{args.precision}_b{args.batch}"
sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
x = get_input(args.batch)
WARM = 10
for _ in range(WARM + args.iters):
    sess.run(None, {"input": x})
pf = sess.end_profiling()
ev = json.load(open(pf))
runs = sorted((e for e in ev if e.get("name") == "model_run"), key=lambda e: e["ts"])
assert len(runs) == WARM + args.iters, len(runs)
t_cut = runs[WARM]["ts"]
meas_runs = runs[WARM:]
nodes = [e for e in ev if e.get("cat") == "Node" and e["name"].endswith("_kernel_time") and e["ts"] >= t_cut]

def group(op):
    o = op.lower()
    if "quantizelinear" in o or "dequantizelinear" in o or "dynamicquantize" in o:
        return "quantize/dequantize"
    if "conv" in o:
        return "conv"
    if "gemm" in o or "matmul" in o:
        return "linear"
    return "other"

by_op = defaultdict(lambda: {"count": 0, "dur_us": 0})
for e in nodes:
    k = (e["args"]["op_name"], e["args"].get("provider"))
    by_op[k]["count"] += 1
    by_op[k]["dur_us"] += e["dur"]
tot = sum(v["dur_us"] for v in by_op.values())
ops = sorted(({"op_type": k[0], "provider": k[1], "count": v["count"], "dur_us_total": v["dur_us"],
               "group": group(k[0])} for k, v in by_op.items()), key=lambda o: -o["dur_us_total"])
groups = defaultdict(float)
for o in ops:
    groups[o["group"]] += o["dur_us_total"]
run_ms = sum(r["dur"] for r in meas_runs) / 1000 / args.iters
res = {"iters": args.iters, "profiled_model_run_ms_per_iter": round(run_ms, 3),
       "sum_node_kernel_ms_per_iter": round(tot / 1000 / args.iters, 3), "n_node_events": len(nodes),
       "group_ms_per_iter": {k: round(v / 1000 / args.iters, 3) for k, v in groups.items()},
       "group_share_pct": {k: round(100 * v / tot, 1) for k, v in groups.items()}, "ops": ops}
print(f"onnx {args.precision} b{args.batch}: profiled model_run {run_ms:.2f} ms/iter; node-sum {tot/1000/args.iters:.2f}; shares {res['group_share_pct']}")
for o in ops[:8]:
    print(f"   {o['op_type']:22s} n={o['count']:5d} {o['dur_us_total']/1000/args.iters:9.3f} ms/iter  [{o['group']}]")
save(f"onnx_{args.precision}_b{args.batch}", {"framework": "onnxruntime", "version": ort.__version__,
     "precision": args.precision, "batch": args.batch, "threads": args.threads, "PROFILED_NOT_TIMING": True, "result": res},
     out=f"{args.out}/raw")
