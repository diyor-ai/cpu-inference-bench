"""Stage 3a summary: reads results/profile/raw/*.json -> results/profile/summary.txt (profiled runs, NOT timing results)."""
import json
from pathlib import Path
from collections import defaultdict
R = Path(__file__).resolve().parent.parent / "results" / "profile"
out = []
def P(s=""): out.append(s); print(s)
D = {}
for f in sorted((R / "raw").glob("*_b*.json")):
    if f.name.startswith("ort_profile"): continue
    d = json.loads(f.read_text()); D[(d["framework"], d["precision"], d["batch"])] = d
P("== Share of profiled time by group (%) and total profiled ms/iter (profiler ON; not comparable with timing tables) ==")
P(f"{'framework':12s} {'prec':5s} {'batch':>5s} {'conv%':>6s} {'linear%':>8s} {'q/dq%':>6s} {'other%':>7s} {'conv ms':>9s} {'total ms':>9s}")
for (fw, pr, b), d in sorted(D.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
    r = d["result"]; sh = defaultdict(float)
    for k, v in r["group_share_pct"].items():
        sh["q" if k.startswith("quantize") else k] += v
    if fw == "pytorch":   conv, tot = r["group_self_cpu_ms_per_iter"].get("conv", 0), r["sum_self_cpu_ms_per_iter"]
    elif fw == "onnxruntime": conv, tot = r["group_ms_per_iter"].get("conv", 0), r["sum_node_kernel_ms_per_iter"]
    else: conv, tot = r["group_ms"].get("conv", 0), r["sum_layer_real_ms_median"]
    P(f"{fw:12s} {pr:5s} {b:5d} {sh['conv']:6.1f} {sh['linear']:8.1f} {sh['q']:6.1f} {sh['other']:7.1f} {conv:9.2f} {tot:9.2f}")
P()
P("== Kernel / implementation names actually reported ==")
for (fw, pr, b), d in sorted(D.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
    r = d["result"]
    if fw == "pytorch":
        top = [(o["name"], o["count"] // r["iters"]) for o in r["ops"] if o["group"] in ("conv", "linear", "quantize/dequantize")][:5]
        P(f"{fw} {pr} b{b}: conv/linear/q ops (name, calls per iter): {top}")
    elif fw == "onnxruntime":
        top = [(o["op_type"], o["count"] // r["iters"]) for o in r["ops"]][:6]
        P(f"{fw} {pr} b{b}: node op types (calls per iter): {top}")
    else:
        c = defaultdict(lambda: [0, 0.0])
        for l in r["layers"]:
            k = f"{l['node_type']}|{l['exec_type']}"; c[k][0] += 1; c[k][1] += l["real_us_median"]
        top = sorted(c.items(), key=lambda kv: -kv[1][1])[:7]
        P(f"{fw} {pr} b{b}: " + "; ".join(f"{k} x{n} ({us/1000:.2f} ms)" for k, (n, us) in top))
(R / "summary.txt").write_text("\n".join(out) + "\n")
