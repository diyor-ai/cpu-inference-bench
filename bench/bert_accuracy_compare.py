"""Paired FP32 vs INT8 on the identical 872 SST-2 validation sentences (McNemar exact + 95% paired bootstrap CI) + FP32 provenance."""
import json, math
from pathlib import Path
import numpy as np
D = Path(__file__).resolve().parent.parent / "results" / "accuracy_bert"
card = {"A": 91.06, "B": 84.74}
rng = np.random.default_rng(0)
def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
def load(n, fw, p, L):
    f = D / f"{n}_{fw}_{p}_L{L}.json"
    return json.loads(f.read_text()) if f.exists() else None
print("Provenance (FP32, seq128, n=872): " + "; ".join(
    f"{n}: " + "/".join(f"{load(n, fw, 'fp32', 128)['acc']}" for fw in ('pytorch', 'onnx', 'openvino')) + f" (card {card[n]})" for n in "AB"))
print(f"\n{'model':<6}{'seq':>4}{'framework':<12}{'method':<28}{'FP32':>7}{'INT8':>7}{'diff':>7}{'95% CI (paired bootstrap)':>27}{'fp32-only/int8-only':>21}{'McNemar p':>10}")
METHOD = {"pytorch": "quantize_dynamic Linear", "onnx": "quantize_dynamic per-channel", "openvino": "NNCF TRANSFORMER default"}
for n, L in (("A", 128), ("A", 32), ("B", 128)):
    for fw in ("pytorch", "onnx", "openvino"):
        f, q = load(n, fw, "fp32", L), load(n, fw, "int8", L)
        if not (f and q): continue
        a, b = np.array(f["correct"]), np.array(q["correct"]); d = b - a
        bs = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(5000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        o_f, o_i = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        print(f"{n:<6}{L:>4}{fw:<12}{METHOD[fw]:<28}{f['acc']:>7.2f}{q['acc']:>7.2f}{d.mean()*100:>+7.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>27}{f'{o_f}/{o_i}':>21}{mcnemar_exact(o_f, o_i):>10.3f}")
print("\nseq32 truncates 208/872 validation sentences (same for FP32 and INT8). Methods were fixed before any INT8 accuracy was computed (notes/2026-10-22.md).")
