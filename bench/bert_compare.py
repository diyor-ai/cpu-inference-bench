"""Stage 3b summary: results/<dir>/raw -> table. Same stability rules as compare.py (spread <5%, not throttled, >=5 repeats)."""
import json, statistics, sys
from pathlib import Path
import pandas as pd
from compare import SPREAD_LIMIT_PCT, MIN_REPEATS
R = Path(__file__).resolve().parent.parent / "results"

def executed(fw, e):
    if fw == "onnxruntime": return (f"int8mm={e['n_int8_matmul']} DynQ={e['n_dynamic_quantize_linear']} floatMM/Gemm={e['n_float_matmul_gemm']}" if "n_int8_matmul" in e else f"[smoke schema] MatMulInteger={e['n_matmul_integer']} (fused ops not counted) floatMM/Gemm={e['n_float_matmul_gemm_attention']}")
    if fw == "openvino": return f"mm/fc={e['runtime_matmul_fc_by_precision']} thr={e['INFERENCE_NUM_THREADS']}"
    return f"engine={e.get('quantized_engine', '-')} dynQLinear={e.get('dynamic_quantized_linear', '-')} floatLinear={e.get('float_linear_left', '-')}"

def aggregate(d):
    g, ex = {}, {}
    for p in sorted((d / "raw").glob("*.json")):
        j = json.loads(p.read_text())
        k = (j["model"], j["seq"], j["framework"], j["precision"], j["threads"], j["result"]["batch"])
        g.setdefault(k, []).append(j["result"]); ex[k] = executed(j["framework"], j["execution_check"])
    rows = []
    for k, reps in sorted(g.items()):
        p50 = [r["p50_ms"] for r in reps]; med = statistics.median(p50); tel = [r["telemetry"] for r in reps]
        spread = (max(p50) - min(p50)) / med * 100; thr = any(t["throttled"] for t in tel)
        rows.append({"model": k[0], "seq": k[1], "framework": k[2], "precision": k[3], "threads": k[4], "batch": k[5], "repeats": len(reps),
                     "p50_ms": round(med, 3), "min_ms": min(r["min_ms"] for r in reps),
                     "p95_ms": round(statistics.median(r["p95_ms"] for r in reps), 3), "sent_s": round(k[5] / (med / 1000), 2),
                     "spread_%": round(spread, 1), "freq_mean_mhz": round(statistics.fmean(t["freq_mean_mhz"] for t in tel)),
                     "freq_min_mhz": min(t["freq_min_mhz"] for t in tel), "temp_max_c": max(t["temp_end_c"] for t in tel),
                     "cooldown_timeouts": sum(t["cooldown"]["timed_out"] for t in tel), "throttled": thr,
                     "max_abs_diff_fp32": max(r["max_abs_diff_vs_pytorch_fp32"] for r in reps), "executed": ex[k],
                     "stable": spread < SPREAD_LIMIT_PCT and not thr and len(reps) >= MIN_REPEATS})
    return pd.DataFrame(rows)

def int8_vs_fp32(df):
    out = []
    for (m, s, fw, t, b), g in df.groupby(["model", "seq", "framework", "threads", "batch"]):
        f, i = g[g.precision == "fp32"], g[g.precision == "int8"]
        if len(f) and len(i):
            ok = f.iloc[0].stable and i.iloc[0].stable
            out.append({"model": m, "seq": s, "framework": fw, "batch": b, "fp32_p50_ms": f.iloc[0].p50_ms, "int8_p50_ms": i.iloc[0].p50_ms,
                        "int8_speedup_vs_fp32": round(f.iloc[0].p50_ms / i.iloc[0].p50_ms, 2) if ok else "n/a (unstable)"})
    return pd.DataFrame(out)

def vs_pytorch(df):
    out = []
    for (m, s, p, t, b), g in df.groupby(["model", "seq", "precision", "threads", "batch"]):
        base = g[g.framework == "pytorch"]
        for _, r in g[g.framework != "pytorch"].iterrows():
            ok = len(base) and base.iloc[0].stable and r.stable
            out.append({"model": m, "seq": s, "framework": r.framework, "precision": p, "batch": b,
                        "speedup_vs_pytorch": round(base.iloc[0].p50_ms / r.p50_ms, 2) if ok else "n/a (unstable)"})
    return pd.DataFrame(out)

if __name__ == "__main__":
    d = R / (sys.argv[1] if len(sys.argv) > 1 else "bert")
    df = aggregate(d); df.to_csv(d / "summary.csv", index=False); (d / "summary.json").write_text(df.to_json(orient="records", indent=2))
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 80)
    print(df.to_string(index=False))
    print(f"\nstable = spread < {SPREAD_LIMIT_PCT}% across repeats, not throttled, >= {MIN_REPEATS} repeats\n{int((~df.stable).sum())} of {len(df)} rows UNSTABLE (flagged, not hidden)\n")
    print("INT8 vs FP32 (same framework):"); print(int8_vs_fp32(df).to_string(index=False))
    print("\nSame-precision ratio vs PyTorch:"); print(vs_pytorch(df).to_string(index=False))
