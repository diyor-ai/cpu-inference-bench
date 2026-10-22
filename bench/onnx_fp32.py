import onnxruntime as ort
from common import (common_args, run_repeat, save, get_model, get_input,
                    reference_output, max_abs_diff, onnx_path)

args = common_args()
so = ort.SessionOptions()
so.intra_op_num_threads = args.threads
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(str(onnx_path()), so, providers=["CPUExecutionProvider"])

x = get_input(args.batch)
row = run_repeat(lambda: sess.run(None, {"input": x}), args.batch, iters=args.iters)
row["max_abs_diff_vs_pytorch"] = max_abs_diff(
    sess.run(None, {"input": x})[0], reference_output(get_model(), x))
print(f"onnx t{args.threads} b{args.batch} r{args.repeat}: p50 {row['p50_ms']} ms, "
      f"{row['telemetry']['freq_mean_mhz']} MHz, {row['telemetry']['temp_end_c']} C, "
      f"throttled={row['telemetry']['throttled']}, max|diff| {row['max_abs_diff_vs_pytorch']:.2e}")

save(f"onnx_fp32_t{args.threads}_b{args.batch}_r{args.repeat}", {
    "framework": "onnxruntime", "version": ort.__version__, "precision": "fp32",
    "model": "resnet18", "threads": args.threads, "repeat": args.repeat, "result": row,
}, out=f"{args.out}/raw")
