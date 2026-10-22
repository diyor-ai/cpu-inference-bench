"""Stage 2: ONNX Runtime FP32 / INT8 (QDQ, MinMax = primary), pretrained weights. One process = one repeat."""
import onnx
import onnxruntime as ort
from collections import Counter
from common import (common_args, run_repeat, save, get_model, get_input, quant_info,
                    reference_output, max_abs_diff, onnx_path, TEMP)

args = common_args()
path = onnx_path(pretrained=True) if args.precision == "fp32" else TEMP / "resnet18_pretrained_int8_ort_minmax.onnx"
opt_path = TEMP / f"ort_optimized_{args.precision}_b{args.batch}.onnx"
so = ort.SessionOptions()
so.intra_op_num_threads = args.threads
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.optimized_model_filepath = str(opt_path)          # dump the graph ORT actually executes (after QDQ fusion etc.)
sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

ops = Counter(n.op_type for n in onnx.load(str(opt_path)).graph.node)
extra = {"model_file": path.name, "optimized_op_counts": dict(ops),
         "n_int8_conv": ops.get("QLinearConv", 0) + ops.get("ConvInteger", 0),
         "n_int8_matmul": ops.get("QLinearMatMul", 0) + ops.get("MatMulInteger", 0) + ops.get("QGemm", 0),
         "n_float_conv": sum(v for k, v in ops.items() if k.endswith("Conv") and not k.startswith("QLinear")),
         "n_quantize_dequantize": ops.get("QuantizeLinear", 0) + ops.get("DequantizeLinear", 0),
         "intra_op_num_threads": args.threads, "providers": sess.get_providers(), "onnx": onnx.__version__}
if args.precision == "int8":
    extra["quantization"] = quant_info("resnet18_pretrained_int8_ort_minmax")

x = get_input(args.batch)
row = run_repeat(lambda: sess.run(None, {"input": x}), args.batch, iters=args.iters)
row["max_abs_diff_vs_pytorch_fp32"] = max_abs_diff(sess.run(None, {"input": x})[0],
                                                   reference_output(get_model(pretrained=True), x))
print(f"onnx {args.precision} t{args.threads} b{args.batch} r{args.repeat}: p50 {row['p50_ms']} ms, "
      f"{row['telemetry']['freq_mean_mhz']} MHz, {row['telemetry']['temp_end_c']} C, throttled={row['telemetry']['throttled']}, "
      f"int8 conv {extra['n_int8_conv']} / float conv {extra['n_float_conv']}")
save(f"onnx_{args.precision}_t{args.threads}_b{args.batch}_r{args.repeat}", {
    "framework": "onnxruntime", "version": ort.__version__, "precision": args.precision, "weights": "IMAGENET1K_V1",
    "model": "resnet18", "threads": args.threads, "repeat": args.repeat, "execution_check": extra, "result": row,
}, out=f"{args.out}/raw")
