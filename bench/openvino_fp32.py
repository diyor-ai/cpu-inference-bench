import numpy as np
import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
from common import (common_args, run_repeat, save, get_model, get_input,
                    reference_output, max_abs_diff, onnx_path)

args = common_args()
# LATENCY hint + 1 stream: one request at a time, all threads on it
# (matches the synchronous loop used for the other frameworks).
smt = args.threads > 6
config = {
    hints.performance_mode: hints.PerformanceMode.LATENCY,
    props.inference_num_threads: args.threads,
    props.num_streams: 1,
    hints.enable_hyper_threading: smt,
}
core = ov.Core()
ov_model = core.read_model(str(onnx_path()))  # same ONNX file as ONNX Runtime
ov_model.reshape({"input": [args.batch, 3, 224, 224]})
compiled = core.compile_model(ov_model, "CPU", config)
req = compiled.create_infer_request()
x = get_input(args.batch)

row = run_repeat(lambda: req.infer({0: x}), args.batch, iters=args.iters)
out = np.array(req.infer({0: x})[compiled.output(0)])
row["max_abs_diff_vs_pytorch"] = max_abs_diff(out, reference_output(get_model(), x))
print(f"openvino t{args.threads} b{args.batch} r{args.repeat}: p50 {row['p50_ms']} ms, "
      f"{row['telemetry']['freq_mean_mhz']} MHz, {row['telemetry']['temp_end_c']} C, "
      f"throttled={row['telemetry']['throttled']}, max|diff| {row['max_abs_diff_vs_pytorch']:.2e}")

save(f"openvino_fp32_t{args.threads}_b{args.batch}_r{args.repeat}", {
    "framework": "openvino", "version": ov.__version__, "precision": "fp32",
    "model": "resnet18", "threads": args.threads, "repeat": args.repeat,
    "config": {"performance_hint": "LATENCY", "num_streams": 1,
               "inference_num_threads": args.threads, "enable_hyper_threading": smt},
    "result": row,
}, out=f"{args.out}/raw")
