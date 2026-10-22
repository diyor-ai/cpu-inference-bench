"""Paired FP32 vs INT8 accuracy on identical images + FP32 provenance check."""
import json, math
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent.parent / "results" / "accuracy"
rng = np.random.default_rng(0)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(name):
    p = D / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


fp = load("pytorch_fp32")
if fp:
    p = fp["top1"] / 100; se = math.sqrt(p * (1 - p) / fp["n"]) * 100
    print(f"Provenance: PyTorch FP32 top-1 {fp['top1']}% (n={fp['n']}, binomial SE {se:.2f} pt); torchvision full-val 69.758%; "
          f"±3 pt band [66.8, 72.8] -> {'INSIDE' if 66.8 <= fp['top1'] <= 72.8 else 'OUTSIDE (investigate)'}; distinct eval classes: {fp['eval_distinct_classes']}")
ROLE = {"_percentile": "secondary"}   # ORT MinMax (primary) was fixed in advance; Percentile is a secondary variant, not selected on eval accuracy
print(f"\n{'framework':<12}{'role':<11}{'int8 variant':<14}{'FP32 top1':>10}{'INT8 top1':>10}{'diff':>7}{'95% CI (paired bootstrap)':>28}"
      f"{'fp32-only/int8-only':>21}{'McNemar p':>10}{'top5 diff':>10}")
for q in sorted(D.glob("*_int8*.json")):
    r = json.loads(q.read_text()); f = load(f"{r['framework']}_fp32")
    if not f: continue
    a = np.array(f["top1_correct"]); b_ = np.array(r["top1_correct"])
    d = (b_ - a)
    boots = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    only_fp, only_i8 = int(((a == 1) & (b_ == 0)).sum()), int(((a == 0) & (b_ == 1)).sum())
    t5 = (np.array(r["top5_correct"]) - np.array(f["top5_correct"])).mean() * 100
    print(f"{r['framework']:<12}{ROLE.get(r['variant'], 'primary'):<11}{r['variant'] or '-':<14}{f['top1']:>10.2f}{r['top1']:>10.2f}{d.mean()*100:>+7.2f}"
          f"{f'[{lo:+.2f}, {hi:+.2f}]':>28}{f'{only_fp}/{only_i8}':>21}{mcnemar_exact(only_fp, only_i8):>10.3f}{t5:>+10.2f}")
print("\nPrimary = pre-registered method per framework (ORT: MinMax; OpenVINO: NNCF defaults; PyTorch: default x86 qconfig). Secondary variants are reported, never substituted for the primary.")
print("McNemar = exact binomial test on discordant pairs; CI = paired bootstrap (5000 resamples, seed 0) of the top-1 difference.")
print("Note: offset sampling is mildly size-biased; the paired FP32-vs-INT8 difference on identical images is the main result.")
