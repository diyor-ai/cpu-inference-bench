"""ONNX Runtime DYNAMIC INT8 (primary for BERT): quantize_dynamic, QInt8 per-channel weights, other args default. No calibration.
Usage: bert_ort_int8.py <A|B>   (main venv)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import onnx, onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from bert_common import onnx_fp32_path, TEMP

n = sys.argv[1]
pre = TEMP / f"bert_{n}_fp32_preproc.onnx"
out = TEMP / f"bert_{n}_int8_ort_dyn.onnx"
quant_pre_process(str(onnx_fp32_path(n)), str(pre), skip_symbolic_shape=True)
quantize_dynamic(str(pre), str(out), weight_type=QuantType.QInt8, per_channel=True)
ops = {}
for nd in onnx.load(str(out)).graph.node:
    ops[nd.op_type] = ops.get(nd.op_type, 0) + 1
(TEMP / f"bert_{n}_int8_ort_dyn.json").write_text(json.dumps({
    "onnxruntime": ort.__version__, "onnx": onnx.__version__, "method": "quantize_dynamic(weight_type=QInt8, per_channel=True), defaults otherwise",
    "preprocess": "quant_pre_process(skip_symbolic_shape=True)", "op_counts": ops}, indent=1))
print(n, "saved", out.name, {k: v for k, v in ops.items() if "Quant" in k or "MatMul" in k or k == "Gemm"})
