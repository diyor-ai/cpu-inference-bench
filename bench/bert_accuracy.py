"""SST-2 validation accuracy (872). Usage: bert_accuracy.py <A|B> <pytorch|onnx|openvino> <fp32|int8> <seq>
Writes results/accuracy_bert/<model>_<fw>_<prec>_L<seq>.json (per-sentence correctness for paired tests)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent / "quant"))
from bert_common import *
from common import save

name, fw, prec, L = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
d = np.load(BERT / f"{name}_val_L{L}.npz")
names = MODELS[name]["inputs"]
y = d["labels"]; X = {k: d[k] for k in names}
BS = 8; info = {}

def run(predict):
    return np.concatenate([predict({k: v[i:i + BS] for k, v in X.items()}) for i in range(0, len(y), BS)])

if fw == "pytorch":
    import torch
    torch.set_num_threads(6); info["version"] = torch.__version__
    if prec == "fp32":
        from bert_pt_int8 import fp32; m = fp32(name)
    else:
        from bert_pt_int8 import build; m = build(name); info["quantized_engine"] = torch.backends.quantized.engine
    with torch.inference_mode():
        logits = run(lambda b: m(*[torch.from_numpy(b[k]) for k in names]).numpy())
elif fw == "onnx":
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 6; info["version"] = ort.__version__
    path = onnx_fp32_path(name) if prec == "fp32" else TEMP / f"bert_{name}_int8_ort_dyn.onnx"
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    logits = run(lambda b: s.run(None, b)[0])
else:
    import openvino as ov
    info["version"] = ov.__version__
    core = ov.Core()
    path = onnx_fp32_path(name) if prec == "fp32" else TEMP / f"bert_{name}_L{L}_int8_ov.xml"
    mdl = core.read_model(str(path)); mdl.reshape({k: [BS, L] for k in names})
    c = core.compile_model(mdl, "CPU", {"INFERENCE_NUM_THREADS": 6, "NUM_STREAMS": 1})
    logits = run(lambda b: np.array(c(b)[c.output(0)]))

ok = (logits.argmax(1) == y)
res = {"model": name, "framework": fw, "precision": prec, "seq": L, **info, "n": len(y),
       "acc": round(float(ok.mean()) * 100, 2), "correct": ok.astype(int).tolist(), "n_truncated": json.loads((BERT / "data_info.json").read_text())[f"{name}_val_L{L}"]["truncated"]}
print(f"{name} {fw} {prec} L{L}: acc {res['acc']}% ({int(ok.sum())}/{len(y)}), truncated sentences: {res['n_truncated']}")
if prec == "fp32":
    diff = res["acc"] - MODELS[name]["card_acc"]
    print(f"   PROVENANCE vs card {MODELS[name]['card_acc']}%: diff {diff:+.2f} pt -> {'OK' if abs(diff) <= 0.5 else 'OUTSIDE +-0.5 pt: STOP'}")
save(f"{name}_{fw}_{prec}_L{L}", res, out=ACC_OUT)
np.save(ACC_OUT / f"{name}_{fw}_{prec}_L{L}_logits.npy", logits)
