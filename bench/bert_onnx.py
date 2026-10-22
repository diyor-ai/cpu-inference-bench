"""Stage 3b: ONNX Runtime FP32 / dynamic INT8 (quantize_dynamic, per-channel), BERT SST-2. One process = one repeat."""
from collections import Counter
import onnx, onnxruntime as ort
from bert_common import bert_args, make_inputs, reference_logits, onnx_fp32_path, MODELS
from common import run_repeat, save, max_abs_diff, TEMP

args = bert_args()
path = onnx_fp32_path(args.model) if args.precision == "fp32" else TEMP / f"bert_{args.model}_int8_ort_dyn.onnx"
opt_path = TEMP / f"ort_optimized_bert_{args.model}_{args.precision}_L{args.seq}_b{args.batch}.onnx"
so = ort.SessionOptions()
so.intra_op_num_threads = args.threads
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.optimized_model_filepath = str(opt_path)        # dump the graph ORT actually executes
sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
ops = Counter(n.op_type for n in onnx.load(str(opt_path)).graph.node)
extra = {"model_file": path.name, "optimized_op_counts": dict(ops),
         # ORT_ENABLE_ALL fuses MatMulInteger(+scales) into MatMulIntegerToFloat / DynamicQuantizeMatMul: count all three as INT8 matmuls
         "n_int8_matmul": sum(ops.get(k, 0) for k in ("MatMulInteger", "MatMulIntegerToFloat", "DynamicQuantizeMatMul")),
         "n_matmul_integer_unfused": ops.get("MatMulInteger", 0), "n_dynamic_quantize_linear": ops.get("DynamicQuantizeLinear", 0),
         "n_float_matmul_gemm": sum(ops.get(k, 0) for k in ("MatMul", "Gemm", "FusedMatMul", "FusedGemm", "Attention", "MultiHeadAttention")),
         "intra_op_num_threads": args.threads, "providers": sess.get_providers(), "onnx": onnx.__version__}
x = make_inputs(args.model, args.seq, args.batch)
row = run_repeat(lambda: sess.run(None, x), args.batch, iters=args.iters)
row["max_abs_diff_vs_pytorch_fp32"] = max_abs_diff(sess.run(None, x)[0], reference_logits(args.model, x))
print(f"onnx {args.model} {args.precision} L{args.seq} b{args.batch} r{args.repeat}: p50 {row['p50_ms']} ms, "
      f"{row['telemetry']['freq_mean_mhz']} MHz, {row['telemetry']['temp_end_c']} C, throttled={row['telemetry']['throttled']}, "
      f"int8 matmul={extra['n_int8_matmul']} float matmul/gemm={extra['n_float_matmul_gemm']}")
save(f"onnx_{args.model}_{args.precision}_L{args.seq}_t{args.threads}_b{args.batch}_r{args.repeat}", {
    "framework": "onnxruntime", "version": ort.__version__, "precision": args.precision, "model": args.model, "seq": args.seq,
    "threads": args.threads, "repeat": args.repeat, "execution_check": extra, "result": row}, out=f"{args.out}/raw")
