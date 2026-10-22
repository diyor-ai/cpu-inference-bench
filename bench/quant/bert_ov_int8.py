"""OpenVINO NNCF PTQ for BERT: nncf.quantize(model_type=TRANSFORMER, default preset, subset_size=300). Runs in .venv-quant.
Usage: bert_ov_int8.py <A|B> <seq>.  Calibration: 300 SST-2 TRAIN sentences, padded to <seq>, batch 1 (npz made by bert_prep.py)."""
import json, sys
from pathlib import Path
import nncf, numpy as np, openvino as ov
ROOT = Path(__file__).resolve().parents[2]
TEMP, BERT = ROOT / "temp", ROOT / "data" / "bert"
n, L = sys.argv[1], int(sys.argv[2])
core = ov.Core()
model = core.read_model(str(TEMP / f"bert_{n}_fp32.onnx"))
names = [i.get_any_name() for i in model.inputs]
model.reshape({k: [1, L] for k in names})
d = np.load(BERT / f"{n}_calib_L{L}.npz")
assert set(names) == set(d.files), (names, d.files)
ds = nncf.Dataset([{k: d[k][i:i + 1] for k in names} for i in range(len(d[names[0]]))])
q = nncf.quantize(model, ds, model_type=nncf.ModelType.TRANSFORMER, subset_size=len(d[names[0]]))
out = TEMP / f"bert_{n}_L{L}_int8_ov.xml"
ov.save_model(q, str(out))
(TEMP / f"bert_{n}_L{L}_int8_ov.json").write_text(json.dumps({
    "nncf": nncf.__version__, "openvino": ov.__version__, "calib_sentences": len(d[names[0]]), "seq": L,
    "call": "nncf.quantize(model, Dataset, model_type=TRANSFORMER, subset_size=300) - default preset"}, indent=1))
print("saved", out.name)
