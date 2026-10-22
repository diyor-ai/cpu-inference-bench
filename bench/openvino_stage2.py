"""Stage 2: OpenVINO FP32 (ONNX) / INT8 (NNCF IR), pretrained weights. One process = one repeat."""
from collections import Counter
import numpy as np
import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
from common import (common_args, run_repeat, save, get_model, get_input, quant_info,
                    reference_output, max_abs_diff, onnx_path, TEMP)

args = common_args()
smt = args.threads > 6
config = {hints.performance_mode: hints.PerformanceMode.LATENCY, props.inference_num_threads: args.threads,
          props.num_streams: 1, hints.enable_hyper_threading: smt}
core = ov.Core()
src = onnx_path(pretrained=True) if args.precision == "fp32" else TEMP / "resnet18_pretrained_int8_ov.xml"
model = core.read_model(str(src))
model.reshape({"input": [args.batch, 3, 224, 224]})
compiled = core.compile_model(model, "CPU", config)
req = compiled.create_infer_request()

def prop(name):
    try:
        return str(compiled.get_property(name))
    except Exception as e:                      # property not exposed by this build
        return f"unavailable ({type(e).__name__})"

by_prec, by_type = Counter(), Counter()
for op in compiled.get_runtime_model().get_ordered_ops():
    info = op.get_rt_info()
    prec = info["runtimePrecision"].astype(str) if "runtimePrecision" in info else "n/a"
    typ = info["layerType"].astype(str) if "layerType" in info else op.get_type_name()
    by_prec[prec] += 1; by_type[f"{typ}:{prec}"] += 1
extra = {"model_file": src.name,
         "INFERENCE_NUM_THREADS": prop("INFERENCE_NUM_THREADS"), "NUM_STREAMS": prop("NUM_STREAMS"),
         "PERFORMANCE_HINT": prop("PERFORMANCE_HINT"), "ENABLE_HYPER_THREADING": prop("ENABLE_HYPER_THREADING"),
         "INFERENCE_PRECISION_HINT": prop("INFERENCE_PRECISION_HINT"), "EXECUTION_MODE_HINT": prop("EXECUTION_MODE_HINT"),
         "runtime_layers_by_precision": dict(by_prec),
         "runtime_layers_by_type_precision": dict(by_type),
         "runtime_conv_by_precision": {k: v for k, v in by_type.items() if "onvolution" in k}}
if args.precision == "int8":
    extra["quantization"] = quant_info("resnet18_pretrained_int8_ov")

x = get_input(args.batch)
row = run_repeat(lambda: req.infer({0: x}), args.batch, iters=args.iters)
out = np.array(req.infer({0: x})[compiled.output(0)])
row["max_abs_diff_vs_pytorch_fp32"] = max_abs_diff(out, reference_output(get_model(pretrained=True), x))
print(f"openvino {args.precision} t{args.threads} b{args.batch} r{args.repeat}: p50 {row['p50_ms']} ms, "
      f"{row['telemetry']['freq_mean_mhz']} MHz, {row['telemetry']['temp_end_c']} C, throttled={row['telemetry']['throttled']}, "
      f"threads={extra['INFERENCE_NUM_THREADS']} conv={extra['runtime_conv_by_precision']}")
save(f"openvino_{args.precision}_t{args.threads}_b{args.batch}_r{args.repeat}", {
    "framework": "openvino", "version": ov.__version__, "precision": args.precision, "weights": "IMAGENET1K_V1",
    "model": "resnet18", "threads": args.threads, "repeat": args.repeat, "execution_check": extra,
    "config": {"performance_hint": "LATENCY", "num_streams": 1, "inference_num_threads": args.threads,
               "enable_hyper_threading": smt},
    "result": row,
}, out=f"{args.out}/raw")
