import json
import statistics
import sys
from pathlib import Path
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
SPREAD_LIMIT_PCT = 5.0
MIN_REPEATS = 5


def exec_summary(fw, e):
    """One-line proof of what actually executed (from the JSON execution_check block)."""
    if fw == "onnxruntime":
        return f"int8conv={e['n_int8_conv']} floatconv={e['n_float_conv']}"
    if fw == "openvino":
        return f"conv={e['runtime_conv_by_precision']} thr={e['INFERENCE_NUM_THREADS']}"
    return f"engine={e.get('quantized_engine', '-')} qconv/linear={e.get('n_quantized_conv_linear', '-')}"


def aggregate(out_dir):
    groups, execs = {}, {}
    for p in sorted((out_dir / "raw").glob("*.json")):
        d = json.loads(p.read_text())
        key = (d["framework"], d["precision"], d["threads"], d["result"]["batch"])
        groups.setdefault(key, []).append(d["result"])
        if "execution_check" in d:
            execs[key] = exec_summary(d["framework"], d["execution_check"])
    rows = []
    for (fw, prec, thr, b), reps in sorted(groups.items()):
        p50s = [r["p50_ms"] for r in reps]
        med = statistics.median(p50s)
        tel = [r["telemetry"] for r in reps]
        spread = (max(p50s) - min(p50s)) / med * 100
        throttled = any(t["throttled"] for t in tel)
        rows.append({
            "framework": fw, "precision": prec, "threads": thr, "batch": b,
            "repeats": len(reps), "p50_ms": round(med, 3),
            "min_ms": min(r["min_ms"] for r in reps),
            "p95_ms": round(statistics.median(r["p95_ms"] for r in reps), 3),
            "img_s": round(b / (med / 1000), 2),
            "spread_%": round(spread, 1),
            "freq_mean_mhz": round(statistics.fmean(t["freq_mean_mhz"] for t in tel)),
            "freq_min_mhz": min(t["freq_min_mhz"] for t in tel),
            "temp_max_c": max(t["temp_end_c"] for t in tel),
            "cooldown_timeouts": sum(t["cooldown"]["timed_out"] for t in tel),
            "throttled": throttled,
            "max_abs_diff": max((r.get("max_abs_diff_vs_pytorch", r.get("max_abs_diff_vs_pytorch_fp32")) for r in reps
                                 if r.get("max_abs_diff_vs_pytorch", r.get("max_abs_diff_vs_pytorch_fp32")) is not None), default=None),
            "executed": execs.get((fw, prec, thr, b), ""),
            "stable": spread < SPREAD_LIMIT_PCT and not throttled and len(reps) >= MIN_REPEATS,
        })
    return pd.DataFrame(rows)


def ratios(df):
    """Speedup vs PyTorch; only where both rows are stable."""
    out = []
    for (prec, thr, b), g in df.groupby(["precision", "threads", "batch"]):
        base = g[g.framework == "pytorch"]
        for _, r in g[g.framework != "pytorch"].iterrows():
            ok = len(base) and base.iloc[0].stable and r.stable
            out.append({"framework": r.framework, "precision": prec, "threads": thr, "batch": b,
                        "speedup_vs_pytorch": round(base.iloc[0].p50_ms / r.p50_ms, 2) if ok else "n/a (unstable)"})
    return pd.DataFrame(out)


def int8_vs_fp32(df):
    """INT8 speedup over FP32 within each framework; only where both rows are stable."""
    out = []
    for (fw, thr, b), g in df.groupby(["framework", "threads", "batch"]):
        f, i = g[g.precision == "fp32"], g[g.precision == "int8"]
        if len(f) and len(i):
            ok = f.iloc[0].stable and i.iloc[0].stable
            out.append({"framework": fw, "threads": thr, "batch": b, "fp32_p50_ms": f.iloc[0].p50_ms, "int8_p50_ms": i.iloc[0].p50_ms,
                        "int8_speedup_vs_fp32": round(f.iloc[0].p50_ms / i.iloc[0].p50_ms, 2) if ok else "n/a (unstable)"})
    return pd.DataFrame(out)


if __name__ == "__main__":
    out_dir = RESULTS / (sys.argv[1] if len(sys.argv) > 1 else "phase1")
    df = aggregate(out_dir)
    df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(df.to_json(orient="records", indent=2))
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    print(f"\nstable = spread < {SPREAD_LIMIT_PCT}% across repeats, not throttled, >= {MIN_REPEATS} repeats")
    print(f"{int((~df.stable).sum())} of {len(df)} rows UNSTABLE (flagged, not hidden)\n")
    print(ratios(df).to_string(index=False))
    if (df.precision == "int8").any():
        print("\nINT8 vs FP32 (same framework):")
        print(int8_vs_fp32(df).to_string(index=False))
