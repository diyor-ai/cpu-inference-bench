"""Single source of truth for README tables, figures and the number list. Reads ONLY results/**/*.json (+ summary.json files).
Every number is registered with its source file via Reg.add(); build_readme.py renders the README from these registrations
and refuses to build if the README text contains a number that was not registered (or whitelisted)."""
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
FW = ["pytorch", "onnxruntime", "openvino"]
FWKEY = {"pytorch": "pytorch", "onnx": "onnxruntime", "onnxruntime": "onnxruntime", "openvino": "openvino"}
FWLABEL = {"pytorch": "PyTorch", "onnxruntime": "ONNX Runtime", "openvino": "OpenVINO"}


def jload(p):
    return json.loads((ROOT / p).read_text()) if not isinstance(p, Path) else json.loads(p.read_text())


class Reg:
    """Registry: key -> (value, formatted text, source). Cells and prose placeholders both go through add()."""
    def __init__(self):
        self.items = {}

    def add(self, key, value, fmt, src):
        if isinstance(value, str):
            txt = value
        elif fmt == "i":
            txt = f"{int(round(value)):,}".replace(",", "")
        else:
            txt = format(value, fmt)
            if txt.startswith("-"):
                txt = "−" + txt[1:]          # typographic minus everywhere
        self.items[key] = {"value": value, "text": txt, "src": src}
        return txt

    def get(self, key):
        return self.items[key]["text"]


def stable_ratio(rows, a, b):
    """p50(a)/p50(b) only if both stable, else None."""
    return round(a["p50_ms"] / b["p50_ms"], 2) if a["stable"] and b["stable"] else None


# ---------- accuracy with paired bootstrap / McNemar (same algorithm + RNG order as accuracy_compare.py / bert_accuracy_compare.py)
def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired(rng, fp, i8):
    a, b = np.array(fp), np.array(i8)
    d = b - a
    boots = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    of, oi = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return {"diff": d.mean() * 100, "lo": float(lo), "hi": float(hi), "fp32_only": of, "int8_only": oi, "p": mcnemar_exact(of, oi)}


def resnet_accuracy():
    rng = np.random.default_rng(0)
    out = []
    D = R / "accuracy"
    for q in sorted(D.glob("*_int8*.json")):          # same sorted order as accuracy_compare.py (single RNG stream)
        r = json.loads(q.read_text()); f = json.loads((D / f"{r['framework']}_fp32.json").read_text())
        s = paired(rng, f["top1_correct"], r["top1_correct"])
        s.update(framework=FWKEY[r["framework"]], variant=r["variant"] or "-", fp32=f["top1"], int8=r["top1"], n=r["n"], src=f"results/accuracy/{q.name}")
        t5 = (np.array(r["top5_correct"]) - np.array(f["top5_correct"])).mean() * 100
        s["top5_diff"] = float(t5)
        out.append(s)
    return out


def bert_accuracy():
    rng = np.random.default_rng(0)
    D = R / "accuracy_bert"
    out = []
    for n, L in (("A", 128), ("A", 32), ("B", 128)):   # same order as bert_accuracy_compare.py
        for fw in ("pytorch", "onnx", "openvino"):
            f = json.loads((D / f"{n}_{fw}_fp32_L{L}.json").read_text()); q = json.loads((D / f"{n}_{fw}_int8_L{L}.json").read_text())
            s = paired(rng, f["correct"], q["correct"])
            s.update(model=n, seq=L, framework=FWKEY[fw], fp32=f["acc"], int8=q["acc"], n=f["n"], truncated=f["n_truncated"],
                     src=f"results/accuracy_bert/{n}_{fw}_int8_L{L}.json")
            s["p_bonf"] = min(1.0, s["p"] * 9)
            out.append(s)
    return out


def load_all():
    d = {}
    d["p1"] = jload("results/phase1/summary.json")
    d["p2"] = jload("results/phase2/summary.json")
    d["s2"] = jload("results/stage2/summary.json")
    d["bert"] = jload("results/bert/summary.json")
    d["racc"] = resnet_accuracy()
    d["bacc"] = bert_accuracy()
    prof = {}
    for f in sorted((R / "profile" / "raw").glob("*_b*.json")):
        if f.name.startswith("ort_profile"):
            continue   # raw ORT traces (git-ignored); per-op totals are in onnx_*_b*.json
        j = json.loads(f.read_text())
        prof[(j["framework"], j["precision"], j["batch"])] = (j, f"results/profile/raw/{f.name}")
    d["prof"] = prof
    d["env"] = jload("results/stage2/raw/pytorch_int8_t6_b1_r0.json")["env"]
    d["rawcounts"] = {k: len(list((R / k / "raw").glob("*.json"))) for k in ("phase1", "phase2", "stage2", "bert")}
    return d


def row(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    raise KeyError(kw)
