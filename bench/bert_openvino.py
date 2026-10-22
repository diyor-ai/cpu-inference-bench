"""Stage 3b: OpenVINO FP32 (ONNX) / INT8 (NNCF TRANSFORMER IR, per model+seq), BERT SST-2. One process = one repeat."""
from collections import Counter
import numpy as np
import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
from bert_common import bert_args, make_inputs, reference_logits, onnx_fp32_path, MODELS
from common import run_repeat, save, max_abs_diff, TEMP

args = bert_args()
names = MODELS[args.model]["inputs"]
config = {hints.performance_mode: hints.PerformanceMode.LATENCY, props.inference_num_threads: args.threads,
          props.num_streams: 1, hints.enable_hyper_threading: args.threads > 6}
core = ov.Core()
src = onnx_fp32_path(args.model) if args.precision == "fp32" else TEMP / f"bert_{args.model}_L{args.seq}_int8_ov.xml"
model = core.read_model(str(src))
model.reshape({k: [args.batch, args.seq] for k in names})
compiled = core.compile_model(model, "CPU", config)
req = compiled.create_infer_request()

def prop(n):
    try:
        return str(compiled.get_property(n))
    except Exception as e:
        return f"unavailable ({type(e).__name__})"

by_type = Counter()
for op in compiled.get_runtime_model().get_ordered_ops():
    i = op.get_rt_info()
    typ = i["layerType"].astype(str) if "layerType" in i else op.get_type_name()
    prec = i["runtimePrecision"].astype(str) if "runtimePrecision" in i else "n/a"
    by_type[f"{typ}:{prec}"] += 1
mm = {k: v for k, v in by_type.items() if any(s in k for s in ("MatMul", "FullyConnected", "Gemm"))}
extra = {"model_file": src.name, "INFERENCE_NUM_THREADS": prop("INFERENCE_NUM_THREADS"), "NUM_STREAMS": prop("NUM_STREAMS"),
         "PERFORMANCE_HINT": prop("PERFORMANCE_HINT"), "ENABLE_HYPER_THREADING": prop("ENABLE_HYPER_THREADING"),
         "INFERENCE_PRECISION_HINT": prop("INFERENCE_PRECISION_HINT"), "EXECUTION_MODE_HINT": prop("EXECUTION_MODE_HINT"),
         "runtime_layers_by_type_precision": dict(by_type), "runtime_matmul_fc_by_precision": mm}
x = make_inputs(args.model, args.seq, args.batch)
row = run_repeat(lambda: req.infer(x), args.batch, iters=args.iters)
out = np.array(req.infer(x)[compiled.output(0)])
row["max_abs_diff_vs_pytorch_fp32"] = max_abs_diff(out, reference_logits(args.model, x))
print(f"openvino {args.model} {args.precision} L{args.seq} b{args.batch} r{args.repeat}: p50 {row['p50_ms']} ms, "
      f"{row['telemetry']['freq_mean_mhz']} MHz, {row['telemetry']['temp_end_c']} C, throttled={row['telemetry']['throttled']}, "
      f"threads={extra['INFERENCE_NUM_THREADS']} mm/fc={mm}")
save(f"openvino_{args.model}_{args.precision}_L{args.seq}_t{args.threads}_b{args.batch}_r{args.repeat}", {
    "framework": "openvino", "version": ov.__version__, "precision": args.precision, "model": args.model, "seq": args.seq,
    "threads": args.threads, "repeat": args.repeat, "execution_check": extra,
    "config": {"performance_hint": "LATENCY", "num_streams": 1, "inference_num_threads": args.threads},
    "result": row}, out=f"{args.out}/raw")
