"""Top-1/top-5 on the shared eval set. Usage: accuracy.py <pytorch|onnx|openvino> <fp32|int8> [variant]"""
import json, sys
import numpy as np
from common import (load_split, normalize, get_model, onnx_path, TEMP, save, WEIGHTS)

fw, prec = sys.argv[1], sys.argv[2]
variant = sys.argv[3] if len(sys.argv) > 3 else ""
x_u8, y = load_split("eval")
BS = 50
info = {}


def run(predict):
    out = []
    for i in range(0, len(y), BS):
        out.append(predict(normalize(x_u8[i:i + BS])))
    return np.concatenate(out)


if fw == "pytorch":
    import torch
    torch.set_num_threads(6); info["version"] = torch.__version__
    if prec == "fp32":
        m = get_model(pretrained=True)
    else:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "quant"))
        from pt_int8 import build
        m = build()[0]
    with torch.inference_mode():
        logits = run(lambda a: m(torch.from_numpy(a)).numpy())
elif fw == "onnx":
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 6; info["version"] = ort.__version__
    path = onnx_path(pretrained=True) if prec == "fp32" else TEMP / f"resnet18_pretrained_int8_ort{variant}.onnx"
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    logits = run(lambda a: s.run(None, {"input": a})[0])
else:
    import openvino as ov
    info["version"] = ov.__version__
    core = ov.Core()
    path = onnx_path(pretrained=True) if prec == "fp32" else TEMP / "resnet18_pretrained_int8_ov.xml"
    mdl = core.read_model(str(path)); mdl.reshape({"input": [BS, 3, 224, 224]})
    c = core.compile_model(mdl, "CPU", {"INFERENCE_NUM_THREADS": 6, "NUM_STREAMS": 1})
    logits = run(lambda a: np.array(c(a)[c.output(0)]))

top5 = np.argsort(-logits, axis=1)[:, :5]
top1_ok = (top5[:, 0] == y); top5_ok = (top5 == y[:, None]).any(axis=1)
res = {"framework": fw, "precision": prec, "variant": variant, **info, "n": len(y),
       "top1": round(float(top1_ok.mean()) * 100, 2), "top5": round(float(top5_ok.mean()) * 100, 2),
       "top1_correct": top1_ok.astype(int).tolist(), "top5_correct": top5_ok.astype(int).tolist(),
       "eval_distinct_classes": int(len(set(y.tolist())))}
print(f"{fw} {prec}{(' ' + variant) if variant else ''}: top-1 {res['top1']}%  top-5 {res['top5']}%  (n={len(y)})")
save(f"{fw}_{prec}{variant}", res, out="results/accuracy")
