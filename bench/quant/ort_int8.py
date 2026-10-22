"""ONNX Runtime static PTQ (QDQ, per-channel QInt8 weights, QUInt8 activations).
Usage: ort_int8.py <minmax|percentile>.  Calibration: data/calib.npz (300 images, disjoint from eval)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnx, onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from common import load_split, normalize, onnx_path, TEMP

method = sys.argv[1]
CM = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile}[method]
x_u8, _ = load_split("calib")


class Reader(CalibrationDataReader):
    def __init__(self):
        self.it = iter(range(len(x_u8)))
    def get_next(self):
        i = next(self.it, None)
        return None if i is None else {"input": normalize(x_u8[i:i + 1])}


pre = TEMP / "resnet18_pretrained_fp32_preproc.onnx"
quant_pre_process(str(onnx_path(pretrained=True)), str(pre))
out = TEMP / f"resnet18_pretrained_int8_ort_{method}.onnx"
quantize_static(str(pre), str(out), Reader(), quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                per_channel=True, calibrate_method=CM,
                extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
ops = {}
for n in onnx.load(str(out)).graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
(TEMP / f"resnet18_pretrained_int8_ort_{method}.json").write_text(json.dumps({
    "onnxruntime": ort.__version__, "onnx": onnx.__version__, "calibration": method, "calib_images": len(x_u8),
    "format": "QDQ", "weights": "QInt8 per-channel symmetric", "activations": "QUInt8 asymmetric", "op_counts": ops}, indent=1))
print("saved", out.name, "| QuantizeLinear:", ops.get("QuantizeLinear"), "DequantizeLinear:", ops.get("DequantizeLinear"))
