"""OpenVINO NNCF post-training quantization (default preset). Runs in .venv-quant. Calibration: data/calib.npz."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nncf, numpy as np, openvino as ov
from common import load_split, normalize, TEMP

x_u8, _ = load_split("calib")
core = ov.Core()
model = core.read_model(str(TEMP / "resnet18_pretrained_fp32.onnx"))
model.reshape({"input": [1, 3, 224, 224]})
ds = nncf.Dataset([normalize(x_u8[i:i + 1]) for i in range(len(x_u8))])
q = nncf.quantize(model, ds, subset_size=len(x_u8))   # default preset, target_device=ANY->CPU, no accuracy control
ov.save_model(q, str(TEMP / "resnet18_pretrained_int8_ov.xml"))
(TEMP / "resnet18_pretrained_int8_ov.json").write_text(json.dumps({
    "nncf": nncf.__version__, "openvino": ov.__version__, "calib_images": len(x_u8),
    "call": "nncf.quantize(model, Dataset, subset_size=300) - default preset/params"}, indent=1))
print("saved resnet18_pretrained_int8_ov.xml")
