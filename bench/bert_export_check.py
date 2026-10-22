"""Export check: FP32 ONNX Runtime vs PyTorch FP32 logits on 64 SST-2 validation sentences at every timing-grid shape
(batch 1 and 8; seq 32 and 128). Writes results/exec_check_bert/export_check.json. Not a timing run.
Usage: bert_export_check.py"""
import json, sys
from pathlib import Path
import numpy as np, torch, onnxruntime as ort
sys.path.insert(0, str(Path(__file__).parent / "quant"))
from bert_common import BERT, MODELS, onnx_fp32_path, ACC_OUT
from bert_pt_int8 import fp32
from common import save

out = {"n_sentences": 64, "what": "max |logit diff| ONNX Runtime FP32 vs PyTorch FP32", "cases": []}
for n in "AB":
    m = fp32(n); s = ort.InferenceSession(str(onnx_fp32_path(n)), providers=["CPUExecutionProvider"])
    for L in (32, 128):
        d = np.load(BERT / f"{n}_val_L{L}.npz"); X = {k: d[k] for k in MODELS[n]["inputs"]}
        for b in (1, 8):
            mx = 0.0
            for i in range(0, 64, b):
                x = {k: v[i:i + b] for k, v in X.items()}
                with torch.inference_mode():
                    r = m(*[torch.from_numpy(x[k]) for k in MODELS[n]["inputs"]]).numpy()
                mx = max(mx, float(np.abs(r - s.run(None, x)[0]).max()))
            out["cases"].append({"model": n, "seq": L, "batch": b, "max_abs_diff": mx})
            print(n, L, b, f"{mx:.2e}")
out["max_abs_diff_A"] = max(c["max_abs_diff"] for c in out["cases"] if c["model"] == "A")
out["max_abs_diff_B"] = max(c["max_abs_diff"] for c in out["cases"] if c["model"] == "B")
save("export_check", out, out=ACC_OUT.parent / "exec_check_bert")
