"""INT8 execution check for BERT (like ResNet stage 2) + short latency probe for the runtime ESTIMATE only (20 iters, not a result).
Usage: bert_exec_check.py <A|B> <seq>"""
import json, sys, time, statistics
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent / "quant"))
from bert_common import *
from common import save
name, L = sys.argv[1], int(sys.argv[2])
names = MODELS[name]["inputs"]; out = {"model": name, "seq": L}
probe = {}
def lat(fn, n=20):
    for _ in range(3): fn()
    t = []
    for _ in range(n):
        s = time.perf_counter(); fn(); t.append((time.perf_counter() - s) * 1000)
    return round(statistics.median(t), 2)
import onnx, onnxruntime as ort, openvino as ov, torch
torch.set_num_threads(6)
# ORT
ops = Counter(n.op_type for n in onnx.load(str(TEMP / f"bert_{name}_int8_ort_dyn.onnx")).graph.node)
fops = Counter(n.op_type for n in onnx.load(str(onnx_fp32_path(name))).graph.node)
out["ort_int8_ops"] = {k: v for k, v in ops.items() if k in ("MatMulInteger", "DynamicQuantizeLinear", "MatMul", "Gemm", "Attention", "QAttention")}
out["ort_fp32_ops"] = {k: v for k, v in fops.items() if k in ("MatMul", "Gemm", "Attention")}
# PyTorch
from bert_pt_int8 import build, fp32
q = build(name); mods = Counter(f"{type(m).__module__}.{type(m).__name__}" for m in q.modules())
out["pt_int8_modules"] = {k: v for k, v in mods.items() if "quantized" in k or k.endswith("Linear")}
# OpenVINO
core = ov.Core()
def ov_layers(path, B):
    m = core.read_model(str(path)); m.reshape({k: [B, L] for k in names})
    c = core.compile_model(m, "CPU", {"INFERENCE_NUM_THREADS": 6, "NUM_STREAMS": 1, "PERFORMANCE_HINT": "LATENCY"})
    cnt = Counter()
    for op in c.get_runtime_model().get_ordered_ops():
        i = op.get_rt_info(); cnt[f"{i['layerType'].astype(str) if 'layerType' in i else op.get_type_name()}:{i['runtimePrecision'].astype(str) if 'runtimePrecision' in i else 'n/a'}"] += 1
    return c, cnt
for B in (1, 8):
    x = make_inputs(name, L, B)
    c8, cnt8 = ov_layers(TEMP / f"bert_{name}_L{L}_int8_ov.xml", B)
    if B == 1: out["ov_int8_layers"] = {k: v for k, v in cnt8.items() if any(s in k for s in ("MatMul", "FullyConnected", "Convolution", "Gemm"))}
    cf, cntf = ov_layers(onnx_fp32_path(name), B)
    if B == 1: out["ov_fp32_layers"] = {k: v for k, v in cntf.items() if any(s in k for s in ("MatMul", "FullyConnected", "Gemm"))}
    r8, rf = c8.create_infer_request(), cf.create_infer_request()
    probe[f"b{B}"] = {"ov_fp32": lat(lambda: rf.infer(x)), "ov_int8": lat(lambda: r8.infer(x))}
    for p, path in (("fp32", onnx_fp32_path(name)), ("int8", TEMP / f"bert_{name}_int8_ort_dyn.onnx")):
        so = ort.SessionOptions(); so.intra_op_num_threads = 6; so.inter_op_num_threads = 1
        s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        probe[f"b{B}"][f"ort_{p}"] = lat(lambda: s.run(None, x))
    xt = [torch.from_numpy(x[k]) for k in names]
    with torch.inference_mode():
        probe[f"b{B}"]["pt_fp32"] = lat(lambda m=fp32(name): m(*xt)); probe[f"b{B}"]["pt_int8"] = lat(lambda: q(*xt))
out["PROBE_NOT_A_RESULT_ms_p50_20iters_unrepeated"] = probe
print(json.dumps(out, indent=1))
save(f"exec_check_{name}_L{L}", out, out=ACC_OUT.parent / "exec_check_bert")
