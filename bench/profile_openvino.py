"""Stage 3a: OpenVINO PERF_COUNT per-layer profiling for ResNet-18 FP32 / INT8 (NNCF). Profiled runs only."""
import statistics
from collections import defaultdict
import numpy as np
import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
from common import common_args, get_input, save, onnx_path, TEMP

args = common_args()
config = {hints.performance_mode: hints.PerformanceMode.LATENCY, props.inference_num_threads: args.threads,
          props.num_streams: 1, hints.enable_hyper_threading: False, props.enable_profiling: True}
core = ov.Core()
src = onnx_path(pretrained=True) if args.precision == "fp32" else TEMP / "resnet18_pretrained_int8_ov.xml"
model = core.read_model(str(src))
model.reshape({"input": [args.batch, 3, 224, 224]})
compiled = core.compile_model(model, "CPU", config)
req = compiled.create_infer_request()
x = get_input(args.batch)
for _ in range(10):
    req.infer({0: x})
per = defaultdict(list); meta = {}; order = []
for _ in range(args.iters):
    req.infer({0: x})
    for i in req.get_profiling_info():
        if i.node_name not in meta:
            order.append(i.node_name)
        meta[i.node_name] = (i.node_type, i.exec_type, str(i.status))
        per[i.node_name].append(i.real_time.total_seconds() * 1e6)

def group(t):
    t = t.lower()
    if "convolution" in t:
        return "conv"
    if "fullyconnected" in t or "matmul" in t or "gemm" in t:
        return "linear"
    if "quantize" in t or "reorder" in t or "convert" in t:
        return "quantize/dequantize/reorder"
    return "other"

layers = [{"name": n, "node_type": meta[n][0], "exec_type": meta[n][1], "status": meta[n][2],
           "real_us_median": statistics.median(per[n]), "real_us_mean": statistics.fmean(per[n]),
           "group": group(meta[n][0])} for n in order]
tot = sum(l["real_us_median"] for l in layers)
groups = defaultdict(float)
for l in layers:
    groups[l["group"]] += l["real_us_median"]
res = {"iters": args.iters, "sum_layer_real_ms_median": round(tot / 1000, 3),
       "group_ms": {k: round(v / 1000, 3) for k, v in groups.items()},
       "group_share_pct": {k: round(100 * v / tot, 1) for k, v in groups.items()}, "layers": layers}
print(f"openvino {args.precision} b{args.batch}: sum of layer real_time {tot/1000:.2f} ms; shares {res['group_share_pct']}")
conv_exec = defaultdict(int)
for l in layers:
    if l["group"] == "conv":
        conv_exec[l["exec_type"]] += 1
print("   conv exec_types:", dict(conv_exec))
save(f"openvino_{args.precision}_b{args.batch}", {"framework": "openvino", "version": ov.__version__,
     "precision": args.precision, "batch": args.batch, "threads": args.threads, "PROFILED_NOT_TIMING": True, "result": res},
     out=f"{args.out}/raw")
