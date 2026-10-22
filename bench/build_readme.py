"""Render README.md from docs/README.template.md. Every number comes from results/**/*.json through report_data.Reg:
 - {{key}} placeholders and generated tables (cells are registered too) -> docs/readme_numbers.tsv (key, text, source file)
 - any other number in the template prose must be listed in LITERAL_OK (with the reason) or the build FAILS.
Usage: .venv/bin/python bench/build_readme.py [--check]   (--check: render in memory, verify README.md on disk is identical)"""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import report_data as rd
from report_data import FW, FWLABEL, Reg, row

D = rd.load_all()
REG = Reg()
ENV = D["env"]

# ---------------------------------------------------------------- numbers that are facts of the setup, taken from JSON
S2 = "results/stage2/summary.json"; P1 = "results/phase1/summary.json"; P2 = "results/phase2/summary.json"; BT = "results/bert/summary.json"
REG.add("hw.cpu", ENV["cpu"], "", "results/stage2/raw/*.json env.cpu")
REG.add("hw.cap_mhz", int(ENV["scaling_max_khz"]["0"]) / 1000, "i", "results/stage2/raw/*.json env.scaling_max_khz")
REG.add("hw.epp", ENV["epp"], "", "results/stage2/raw/*.json env.epp")
REG.add("hw.pl1_w", int(ENV["rapl_pl1_uw"]) / 1e6, "i", "results/stage2/raw/*.json env.rapl_pl1_uw")
REG.add("hw.pl2_w", int(ENV["rapl_pl2_uw"]) / 1e6, "i", "results/stage2/raw/*.json env.rapl_pl2_uw")
REG.add("hw.python", ENV["python"], "", "results/stage2/raw/*.json env.python")
for fwk, f in (("pytorch", "results/stage2/raw/pytorch_fp32_t6_b1_r0.json"), ("onnxruntime", "results/stage2/raw/onnx_fp32_t6_b1_r0.json"), ("openvino", "results/stage2/raw/openvino_fp32_t6_b1_r0.json")):
    REG.add(f"ver.{fwk}", rd.jload(f)["version"].split("-")[0], "", f)   # build suffix dropped
REG.add("m.threads", D["s2"][0]["threads"], "i", S2)
REG.add("m.repeats", D["s2"][0]["repeats"], "i", S2)
REG.add("m.iters", rd.jload("results/stage2/raw/pytorch_fp32_t6_b1_r0.json")["result"]["iters"], "i", "results/stage2/raw/pytorch_fp32_t6_b1_r0.json result.iters")
REG.add("m.spread_limit", 5, "i", "bench/compare.py SPREAD_LIMIT_PCT (stability rule)")
REG.add("m.bert_procs", D["rawcounts"]["bert"], "i", "results/bert/raw/*.json (count)")
REG.add("m.s2_procs", D["rawcounts"]["stage2"], "i", "results/stage2/raw/*.json (count)")
REG.add("m.p1_procs", D["rawcounts"]["phase1"], "i", "results/phase1/raw/*.json (count)")
REG.add("m.p2_procs", D["rawcounts"]["phase2"], "i", "results/phase2/raw/*.json (count)")
bacc, racc = D["bacc"], D["racc"]
REG.add("acc.rn_n", racc[0]["n"], "i", "results/accuracy/pytorch_int8.json n")
REG.add("acc.bert_n", bacc[0]["n"], "i", "results/accuracy_bert/A_pytorch_fp32_L128.json n")
REG.add("acc.bert_trunc32", bacc[3]["truncated"], "i", "results/accuracy_bert/A_pytorch_fp32_L32.json n_truncated")
REG.add("acc.bert_trunc32_pct", 100 * bacc[3]["truncated"] / bacc[3]["n"], ".0f", "results/accuracy_bert/A_pytorch_fp32_L32.json n_truncated / n")
REG.add("acc.bert_ntests", 9, "i", "bench/bert_accuracy_compare.py (3 frameworks x {A seq128, A seq32, B seq128})")
REG.add("acc.boot", 5000, "i", "bench/accuracy_compare.py / bert_accuracy_compare.py (bootstrap resamples, seed 0)")
REG.add("acc.rn_fp32", racc[0]["fp32"], ".1f", "results/accuracy/*_fp32.json top1")
REG.add("acc.bertA_fp32_128", bacc[0]["fp32"], ".2f", "results/accuracy_bert/A_*_fp32_L128.json acc")
REG.add("acc.bertB_fp32_128", bacc[6]["fp32"], ".2f", "results/accuracy_bert/B_*_fp32_L128.json acc")
REG.add("acc.bertA_fp32_32", bacc[3]["fp32"], ".2f", "results/accuracy_bert/A_*_fp32_L32.json acc")
REG.add("acc.bertA_card", 91.06, ".2f", "model card (distilbert-base-uncased-finetuned-sst-2-english); compare results/accuracy_bert/A_*_fp32_L128.json")
REG.add("acc.bertB_card", 84.74, ".2f", "model card (M-FAC/bert-mini-finetuned-sst2); compare results/accuracy_bert/B_*_fp32_L128.json")


def f1(x): return format(x, ".1f")
def sign(x, nd=2): return format(x, f"+.{nd}f").replace("-", "−")


# ---------------------------------------------------------------- tables (cells registered with sources)
def cell(key, v, fmt, src):
    return REG.add(key, v, fmt, src)


def t_fp32_p1():
    out = ["| batch | framework | p50 (ms) | p95 (ms) | throughput (img/s) | spread (%) | speedup vs PyTorch | stable |", "|---|---|---|---|---|---|---|---|"]
    for b in (1, 8, 32):
        base = row(D["p1"], framework="pytorch", batch=b)
        for fw in FW:
            r = row(D["p1"], framework=fw, batch=b); k = f"p1.{fw}.b{b}"
            ratio = "—" if fw == "pytorch" else (cell(k + ".ratio", base["p50_ms"] / r["p50_ms"], ".2f", P1) + "×" if r["stable"] and base["stable"] else "n/a (unstable)")
            out.append(f"| {cell(k+'.b', b, 'i', P1)} | {FWLABEL[fw]} | {cell(k+'.p50', r['p50_ms'], '.1f', P1)} | {cell(k+'.p95', r['p95_ms'], '.1f', P1)} | "
                       f"{cell(k+'.ips', r['img_s'], '.1f', P1)} | {cell(k+'.spread', r['spread_%'], '.1f', P1)} | {ratio} | {'yes' if r['stable'] else '**UNSTABLE**'} |")
    return "\n".join(out)


def pair(rows, **kw):
    f = row(rows, precision="fp32", **kw); i = row(rows, precision="int8", **kw)
    return f, i


def t_resnet_s2():
    out = ["| batch | framework | FP32 p50 (ms) | INT8 p50 (ms) | INT8 speedup | INT8 p50 vs PyTorch INT8 | FP32 spread / INT8 spread (%) | stable (FP32/INT8) |", "|---|---|---|---|---|---|---|---|"]
    for b in (1, 8, 32):
        pf, pi = pair(D["s2"], framework="pytorch", batch=b)
        for fw in FW:
            f, i = pair(D["s2"], framework=fw, batch=b); k = f"s2.{fw}.b{b}"
            sp = cell(k + ".sp", f["p50_ms"] / i["p50_ms"], ".2f", S2) + "×" if f["stable"] and i["stable"] else "n/a (unstable)"
            vs = "—" if fw == "pytorch" else (cell(k + ".vs", pi["p50_ms"] / i["p50_ms"], ".2f", S2) + "×" if i["stable"] and pi["stable"] else "n/a (unstable)")
            fl = lambda r: "yes" if r["stable"] else "**NO**"
            out.append(f"| {cell(k+'.b', b, 'i', S2)} | {FWLABEL[fw]} | {cell(k+'.f', f['p50_ms'], '.1f', S2)} | {cell(k+'.i', i['p50_ms'], '.1f', S2)} | {sp} | {vs} | "
                       f"{cell(k+'.fs', f['spread_%'], '.1f', S2)} / {cell(k+'.is', i['spread_%'], '.1f', S2)} | {fl(f)} / {fl(i)} |")
    return "\n".join(out)


def t_bert(model):
    out = ["| seq | batch | framework | FP32 p50 (ms) | INT8 p50 (ms) | INT8 speedup | INT8 vs PyTorch INT8 | stable (FP32/INT8) |", "|---|---|---|---|---|---|---|---|"]
    rows = [r for r in D["bert"] if r["model"] == model]
    for seq in sorted({r["seq"] for r in rows}):
        for b in (1, 8):
            pf, pi = pair(rows, framework="pytorch", seq=seq, batch=b)
            for fw in FW:
                f, i = pair(rows, framework=fw, seq=seq, batch=b); k = f"bert.{model}.{seq}.{fw}.b{b}"
                sp = cell(k + ".sp", f["p50_ms"] / i["p50_ms"], ".2f", BT) + "×" if f["stable"] and i["stable"] else "n/a (unstable)"
                vs = "—" if fw == "pytorch" else (cell(k + ".vs", pi["p50_ms"] / i["p50_ms"], ".2f", BT) + "×" if i["stable"] and pi["stable"] else "n/a (unstable)")
                fl = lambda r: "yes" if r["stable"] else "**NO**"
                out.append(f"| {cell(k+'.seq', seq, 'i', BT)} | {cell(k+'.b', b, 'i', BT)} | {FWLABEL[fw]} | {cell(k+'.f', f['p50_ms'], '.1f', BT)} | {cell(k+'.i', i['p50_ms'], '.1f', BT)} | {sp} | {vs} | {fl(f)} / {fl(i)} |")
    return "\n".join(out)


def t_racc():
    out = ["| framework | INT8 method | FP32 top-1 (%) | INT8 top-1 (%) | diff (pt) | 95% CI (pt) | FP32-only / INT8-only correct | McNemar p |", "|---|---|---|---|---|---|---|---|"]
    meth = {"onnxruntime": "static QDQ, MinMax (primary)", "openvino": "NNCF PTQ, default", "pytorch": "FX graph mode, x86 qconfig"}
    for s in racc:
        k = f"racc.{s['framework']}{s['variant']}"; src = s["src"]
        m = "static QDQ, Percentile (secondary)" if s["variant"] == "_percentile" else meth[s["framework"]]
        out.append(f"| {FWLABEL[s['framework']]} | {m} | {cell(k+'.f', s['fp32'], '.1f', src)} | {cell(k+'.i', s['int8'], '.1f', src)} | {cell(k+'.d', s['diff'], '+.1f', src).replace('-', '−')} | "
                   f"[{cell(k+'.lo', s['lo'], '+.1f', src).replace('-', '−')}, {cell(k+'.hi', s['hi'], '+.1f', src).replace('-', '−')}] | "
                   f"{cell(k+'.of', s['fp32_only'], 'i', src)} / {cell(k+'.oi', s['int8_only'], 'i', src)} | {cell(k+'.p', s['p'], '.3f', src)} |")
    return "\n".join(out)


def t_bacc():
    out = ["| model | seq | framework | FP32 acc (%) | INT8 acc (%) | diff (pt) | 95% CI (pt) | FP32-only / INT8-only | McNemar p (raw) | p (Bonferroni ×9) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in bacc:
        k = f"bacc.{s['model']}.{s['seq']}.{s['framework']}"; src = s["src"]
        out.append(f"| {s['model']} | {cell(k+'.seq', s['seq'], 'i', src)} | {FWLABEL[s['framework']]} | {cell(k+'.f', s['fp32'], '.2f', src)} | {cell(k+'.i', s['int8'], '.2f', src)} | "
                   f"{cell(k+'.d', s['diff'], '+.2f', src).replace('-', '−')} | [{cell(k+'.lo', s['lo'], '+.2f', src).replace('-', '−')}, {cell(k+'.hi', s['hi'], '+.2f', src).replace('-', '−')}] | "
                   f"{cell(k+'.of', s['fp32_only'], 'i', src)} / {cell(k+'.oi', s['int8_only'], 'i', src)} | {cell(k+'.p', s['p'], '.3f', src)} | {cell(k+'.pb', s['p_bonf'], '.2f', src)} |")
    return "\n".join(out)


def t_profile():
    out = ["| framework | precision | batch | conv time (ms) | conv share of profiled time (%) | quantize/dequantize share (%) | profiled total (ms) |", "|---|---|---|---|---|---|---|"]
    for (fw, prec, b), (j, src) in sorted(D["prof"].items(), key=lambda kv: (kv[0][2], FW.index(kv[0][0]), kv[0][1])):
        r = j["result"]; k = f"prof.{fw}.{prec}.b{b}"
        tot = r.get("sum_self_cpu_ms_per_iter") or r.get("sum_node_kernel_ms_per_iter") or r.get("sum_layer_real_ms_median")
        gms = r.get("group_self_cpu_ms_per_iter") or r.get("group_ms_per_iter") or r.get("group_ms")
        qdq = sum(v for g, v in r["group_share_pct"].items() if g.startswith("quantize"))
        out.append(f"| {FWLABEL[fw]} | {prec.upper()} | {cell(k+'.b', b, 'i', src)} | {cell(k+'.convms', gms['conv'], '.1f', src)} | {cell(k+'.conv', r['group_share_pct']['conv'], '.1f', src)} | "
                   f"{cell(k+'.qdq', qdq, '.1f', src)} | {cell(k+'.tot', tot, '.1f', src)} |")
    return "\n".join(out)


def ov_kernels(prec):
    j, src = D["prof"][("openvino", prec, 1)]
    from collections import Counter
    c = Counter(l["exec_type"] for l in j["result"]["layers"] if l["node_type"] == "Convolution")
    return ", ".join(f"`{k}` ×{cell('kern.' + prec + '.' + k, v, 'i', src)}" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))


def t_smt():
    out = ["| framework | batch | 6 threads p50 (ms) | 12 threads (SMT on) p50 (ms) |", "|---|---|---|---|"]
    for fw in FW:
        for b in (1, 8, 32):
            a, c = row(D["p1"], framework=fw, batch=b), row(D["p2"], framework=fw, batch=b)
            if a["stable"] and c["stable"]:
                k = f"smt.{fw}.b{b}"
                out.append(f"| {FWLABEL[fw]} | {cell(k+'.b', b, 'i', P1)} | {cell(k+'.a', a['p50_ms'], '.1f', P1)} | {cell(k+'.c', c['p50_ms'], '.1f', P2)} |")
    return "\n".join(out)


# ---------------------------------------------------------------- headline ranges (stable pairs only)
def sp_range(rows, src, key):
    v = [(r_f["p50_ms"] / r_i["p50_ms"], r_f, r_i) for r_f, r_i in
         ((row(rows, precision="fp32", **{k: r[k] for k in ("framework", "batch") + (("model", "seq") if "model" in r else ())}), r)
          for r in rows if r["precision"] == "int8") if r_f["stable"] and r_i["stable"]]
    lo, hi = min(v, key=lambda t: t[0]), max(v, key=lambda t: t[0])
    cell(key + ".min", lo[0], ".2f", src); cell(key + ".max", hi[0], ".2f", src); cell(key + ".n", len(v), "i", src)
    return lo, hi


sp_range(D["s2"], S2, "head.rn")
sp_range(D["bert"], BT, "head.bert")
REG.add("head.rn_unstable", sum(not r["stable"] for r in D["s2"]), "i", S2)
REG.add("head.rn_rows", len(D["s2"]), "i", S2)
REG.add("head.bert_unstable", sum(not r["stable"] for r in D["bert"]), "i", BT)
REG.add("head.bert_rows", len(D["bert"]), "i", BT)
REG.add("head.p1_unstable", sum(not r["stable"] for r in D["p1"]), "i", P1)
REG.add("head.p1_rows", len(D["p1"]), "i", P1)
REG.add("head.p2_unstable", sum(not r["stable"] for r in D["p2"]), "i", P2)
REG.add("head.p2_rows", len(D["p2"]), "i", P2)
rn_sig = [s for s in racc]
REG.add("head.rn_diff_min", min(s["diff"] for s in racc), ".1f", "results/accuracy/*_int8*.json")
REG.add("head.rn_diff_max", max(s["diff"] for s in racc), ".1f", "results/accuracy/*_int8*.json")
REG.add("head.rn_ci_all_include_zero", "all", "", "results/accuracy/*_int8*.json (paired bootstrap)")
REG.add("head.bertA128pt.diff", bacc[0]["diff"], ".2f", bacc[0]["src"])
REG.add("head.bertA128pt.p", bacc[0]["p"], ".3f", bacc[0]["src"])
REG.add("head.bertA128pt.pb", bacc[0]["p_bonf"], ".2f", bacc[0]["src"])
REG.add("head.bert_other_min", min(s["diff"] for s in bacc[1:]), ".2f", "results/accuracy_bert/*_int8_*.json")
REG.add("head.bert_other_max", max(s["diff"] for s in bacc[1:]), ".2f", "results/accuracy_bert/*_int8_*.json")
REG.add("head.ratio_min_note", "", "", "")
for fw in FW:
    for b in (1, 32):
        f, i = pair(D["s2"], framework=fw, batch=b)
        REG.add(f"hd.s2.{fw}.b{b}.sp", f["p50_ms"] / i["p50_ms"], ".2f", S2)

# archived (invalid / superseded) runs: numbers derived from the archived JSON, never edited
_pairs = []
for _f in sorted((rd.R / "archive_noisy_first_attempt").glob("*_run2.json")):
    _a = rd.jload(Path(str(_f).replace("_run2", "")))["results"]; _b = rd.jload(_f)["results"]
    _pairs += [(x["p50_ms"], y["p50_ms"]) for x, y in zip(_a, _b)]
REG.add("arch.noisy_pairs", len(_pairs), "i", "results/archive_noisy_first_attempt/*.json (run vs run2 per framework/threads/batch)")
REG.add("arch.noisy_over", sum((max(a, b) - min(a, b)) / min(a, b) > 0.05 for a, b in _pairs), "i", "results/archive_noisy_first_attempt/*.json ((max-min)/min > 5%)")
_p3 = rd.jload("results/archive_pilot_3000mhz/summary.json")
REG.add("arch.p3000_rows", len(_p3), "i", "results/archive_pilot_3000mhz/summary.json")
REG.add("arch.p3000_throttled", sum(r["throttled"] for r in _p3), "i", "results/archive_pilot_3000mhz/summary.json (throttled flag)")
REG.add("arch.p3000_mhz", 3000, "i", "results/archive_pilot_3000mhz/summary.json (freq_mean_mhz ~2995; cap 3000000 kHz)")
_bug = [rd.jload(f"results/archive_bert_tokenizer_bug/{n}_{fw}_fp32_L128.json")["acc"] for n in "AB" for fw in ("pytorch", "onnx", "openvino")]
REG.add("arch.bugacc", _bug[0], ".2f", "results/archive_bert_tokenizer_bug/*.json (all six identical)")
assert len(set(_bug)) == 1

# ---- claims made in the prose are asserted against the data (build fails if the data contradicts the text)
def _pairs_all(rows, extra=()):
    for r in rows:
        if r["precision"] == "int8":
            f = row(rows, precision="fp32", framework=r["framework"], batch=r["batch"], **{k: r[k] for k in extra})
            yield f, r
assert all(f["p50_ms"] > i["p50_ms"] for f, i in _pairs_all(D["s2"]) if f["stable"] and i["stable"]), "INT8 not faster in a stable ResNet pair"
assert all(f["p50_ms"] > i["p50_ms"] for f, i in _pairs_all(D["bert"], ("model", "seq")) if f["stable"] and i["stable"]), "INT8 not faster in a stable BERT pair"
assert all(s["hi"] >= 0 for s in racc), "a ResNet CI lies entirely below zero"
assert sum(s["p"] < 0.05 for s in bacc) == 1 and bacc[0]["p"] < 0.05, "expected exactly one nominally significant BERT McNemar test (A seq128 pytorch)"
assert all(s["p_bonf"] >= 0.05 for s in bacc), "a BERT test survives Bonferroni"
for _n, _rows in (("phase1", D["p1"]), ("phase2", D["p2"]), ("stage2", D["s2"]), ("bert", D["bert"])):
    assert all(r["cooldown_timeouts"] == 0 for r in _rows), f"cooldown timeout in {_n}"
_ec = rd.jload("results/exec_check_bert/export_check.json")
_td = max(r["max_abs_diff_fp32"] for r in D["bert"] if r["precision"] == "fp32")
REG.add("bert.fp32_maxdiff_timing", _td, ".1e", BT + " max_abs_diff_fp32 (FP32 rows: ORT and OpenVINO vs PyTorch, first `batch` validation sentences, timing run)")
REG.add("exp.n", _ec["n_sentences"], "i", "results/exec_check_bert/export_check.json n_sentences")
REG.add("bert.maxdiff_A", _ec["max_abs_diff_A"], ".1e", "results/exec_check_bert/export_check.json max_abs_diff_A (ORT vs PyTorch, 64 sentences, batch 1/8, seq 32/128)")
REG.add("bert.maxdiff_B", _ec["max_abs_diff_B"], ".1e", "results/exec_check_bert/export_check.json max_abs_diff_B (same check)")
REG.add("bert.maxdiff_all", max(_td, _ec["max_abs_diff_A"], _ec["max_abs_diff_B"]), ".1e", "max over the timing-run rows and results/exec_check_bert/export_check.json")
REG.add("rn.fp32_maxdiff", max(r["max_abs_diff"] for r in D["s2"] if r["precision"] == "fp32" and r["max_abs_diff"] == r["max_abs_diff"]), ".1e", S2 + " max_abs_diff (FP32 rows)")
_bo = [s for s in bacc if s["model"] == "B" and s["framework"] == "openvino"][0]
assert _bo["lo"] > 0 and _bo["p"] > 0.05
REG.add("bo.lo", _bo["lo"], "+.2f", _bo["src"]); REG.add("bo.hi", _bo["hi"], "+.2f", _bo["src"]); REG.add("bo.p", _bo["p"], ".3f", _bo["src"])
REG.add("bo.of", _bo["fp32_only"], "i", _bo["src"]); REG.add("bo.oi", _bo["int8_only"], "i", _bo["src"]); REG.add("bo.d", _bo["diff"], "+.2f", _bo["src"])

# ---- takeaway assertions
def _int8_groups():
    g = {}
    for r in D["s2"]:
        if r["precision"] == "int8": g.setdefault(("rn", r["batch"]), []).append(r)
    for r in D["bert"]:
        if r["precision"] == "int8": g.setdefault((r["model"], r["seq"], r["batch"]), []).append(r)
    return g
_g = _int8_groups()
for _k, _rs in _g.items():
    _ov = [r for r in _rs if r["framework"] == "openvino"][0]
    assert _ov["stable"], f"OpenVINO INT8 row unstable in {_k}"
    assert all(_ov["p50_ms"] < r["p50_ms"] for r in _rs if r["framework"] != "openvino" and r["stable"]), f"OpenVINO INT8 not lowest in {_k}"
REG.add("tk.ov_groups", len(_g), "i", S2 + " + " + BT + " (INT8 configurations compared)")
_lo, _hi = sp_range(D["s2"], S2, "head.rn"), None
_rlo, _rhi = sp_range(D["s2"], S2, "head.rn"); _blo, _bhi = sp_range(D["bert"], BT, "head.bert")
assert _rlo[1]["framework"] == "pytorch" and _rhi[1]["framework"] == "openvino", "ResNet speedup extremes changed"
assert _blo[1]["framework"] == "pytorch" and _bhi[1]["framework"] == "openvino", "BERT speedup extremes changed"
_smt = [(a, c) for fw in FW for b in (1, 8, 32) for a, c in [(row(D["p1"], framework=fw, batch=b), row(D["p2"], framework=fw, batch=b))] if a["stable"] and c["stable"]]
assert all(c["p50_ms"] <= a["p50_ms"] for a, c in _smt), "SMT slower in a stable pair"
_g_smt = [100 * (a["p50_ms"] - c["p50_ms"]) / a["p50_ms"] for a, c in _smt]
REG.add("tk.smt_n", len(_smt), "i", P1 + " + " + P2 + " (pairs stable in both)")
REG.add("tk.smt_min", min(_g_smt), ".1f", P1 + " + " + P2 + " (p50 reduction, 12 vs 6 threads)")
REG.add("tk.smt_max", max(_g_smt), ".1f", P1 + " + " + P2 + " (p50 reduction, 12 vs 6 threads)")
_mm = [s_ for s_ in racc if s_["framework"] == "onnxruntime" and s_["variant"] == "_minmax"][0]
assert _mm["hi"] == 0.0, f"ORT MinMax upper bound is {_mm['hi']!r}, text says exactly 0"
assert all(s_["lo"] < 0 <= s_["hi"] for s_ in racc), "a ResNet interval does not include zero"
REG.add("rn.minmax_hi", _mm["hi"], "+.1f", _mm["src"])

# ---- 'Results at a glance'
def t_glance():
    def sp(rows, **kw):
        f, i = pair(rows, **kw); return f, i
    out = ["| | PyTorch | ONNX Runtime | OpenVINO |", "|---|---|---|---|"]
    cells = {}
    for fw in FW:
        f, i = pair(D["s2"], framework=fw, batch=1); k = f"gl.rn.{fw}"
        cells[fw] = {"time": f"{cell(k+'.f', f['p50_ms'], '.1f', S2)} → {cell(k+'.i', i['p50_ms'], '.1f', S2)} ms ({cell(k+'.sp', f['p50_ms']/i['p50_ms'], '.2f', S2)}×)",
                     "int8": (cell(k+'.ims', i["p50_ms"], ".1f", S2), cell(k+'.iips', i["img_s"], ".0f", S2))}
        a = [x for x in racc if x["framework"] == fw and x["variant"] in ("-", "_minmax")][0]
        cells[fw]["acc"] = f"{cell(k+'.ad', a['diff'], '+.1f', a['src'])} [{cell(k+'.alo', a['lo'], '+.1f', a['src'])}, {cell(k+'.ahi', a['hi'], '+.1f', a['src'])}]"
        base = row(D["p1"], framework="pytorch", batch=1); r1 = row(D["p1"], framework=fw, batch=1)
        cells[fw]["fp32vs"] = "—" if fw == "pytorch" else cell(k+".p1ratio", base["p50_ms"] / r1["p50_ms"], ".2f", P1) + "×"
        bf, bi = pair([r for r in D["bert"] if r["model"] == "A"], framework=fw, seq=128, batch=1)
        cells[fw]["bsp"] = (cell(k+".bsp", bf["p50_ms"] / bi["p50_ms"], ".2f", BT) + "×") if bf["stable"] and bi["stable"] else f"n/a (FP32 row unstable; INT8 {cell(k+'.bi', bi['p50_ms'], '.1f', BT)} ms)"
        b_ = [x for x in bacc if x["model"] == "A" and x["seq"] == 128 and x["framework"] == fw][0]
        cells[fw]["bacc"] = f"{cell(k+'.bd', b_['diff'], '+.2f', b_['src'])} [{cell(k+'.blo', b_['lo'], '+.2f', b_['src'])}, {cell(k+'.bhi', b_['hi'], '+.2f', b_['src'])}]"
    best = min(FW, key=lambda f: float(REG.items[f"gl.rn.{f}.ims"]["value"]))
    assert best == "openvino"
    def ibest(fw):
        a, b = cells[fw]["int8"]; t = f"{a} ms ({b} img/s)"; return f"**{t}**" if fw == best else t
    out.append("| ResNet-18, batch 1: FP32 → INT8 p50 and speedup | " + " | ".join(cells[f]["time"] for f in FW) + " |")
    out.append("| ResNet-18 top-1 change INT8 − FP32 (pt, 95% CI) | " + " | ".join(cells[f]["acc"] for f in FW) + " |")
    out.append("| ResNet-18, batch 1: INT8 p50 (lowest in bold) | " + " | ".join(ibest(f) for f in FW) + " |")
    out.append("| ResNet-18 FP32, batch 1: speedup vs PyTorch | " + " | ".join(cells[f]["fp32vs"] for f in FW) + " |")
    out.append("| BERT A, seq 128, batch 1: INT8 speedup vs FP32 | " + " | ".join(cells[f]["bsp"] for f in FW) + " |")
    out.append("| BERT A, seq 128 top-1 change INT8 − FP32 (pt, 95% CI) | " + " | ".join(cells[f]["bacc"] for f in FW) + " |")
    return "\n".join(out)


TABLES = {"glance": t_glance, "fp32_p1": t_fp32_p1, "resnet_s2": t_resnet_s2, "bert_A": lambda: t_bert("A"), "bert_B": lambda: t_bert("B"),
          "racc": t_racc, "bacc": t_bacc, "profile": t_profile, "smt": t_smt}

# literals allowed in template prose (not data results): explained so the checker output is auditable
LITERAL_OK = {
    "10": "warmup iterations (bench/common.py run_repeat warmup=10)",
    "60": "cooldown target in degrees C (bench/common.py cooldown 60.0); also seconds timeouts not used",
    "180": "cooldown max wait in seconds (bench/common.py)",
    "250": "frequency sampling interval in ms (bench/common.py FreqSampler)",
    "300": "calibration images/sentences (bench/quant/*.py, bench/bert_common.py N_CALIB)",
    "1000": "ResNet evaluation sample size (data prep; also results/accuracy n)",
    "256": "ImageNet resize size in preprocessing (bench/common.py)",
    "224": "ImageNet crop size / input resolution",
    "32": "sequence length / batch size labels", "128": "sequence length label", "8": "batch size label", "1": "batch size label",
    "2": "list/ordinal words", "3": "frameworks count word", "6": "P-core count / threads (hardware fact, = m.threads)", "12": "logical CPUs on the 6 P-cores with SMT (hardware fact)",
    "16": "logical CPUs reported by lscpu (hardware fact)", "4": "E-core count (hardware fact)", "0": "zero", "5": "(unused)", "9": "(unused)",
    "95": "confidence level of the bootstrap interval (percentiles 2.5 and 97.5 in bench/accuracy_compare.py, bench/bert_accuracy_compare.py)",
    "2026": "year",
}


def render(template):
    parts = re.split(r"(\{\{[^}]+\}\})", template)
    out, lits = [], []
    for p in parts:
        m = re.fullmatch(r"\{\{([^}]+)\}\}", p)
        if not m:
            out.append(p); lits.append(p); continue
        key = m.group(1).strip()
        if key.startswith("table:"):
            out.append(TABLES[key[6:]]())
        elif key.startswith("ovk:"):
            out.append(ov_kernels(key[4:]))
        else:
            out.append(REG.get(key))
    return "".join(out), "".join(lits)


NUM = re.compile(r"(?<![\w.])(?<![A-Za-z]-)[-+−]?\d[\d.,]*\d|(?<![\w.])(?<![A-Za-z]-)\d")


def literal_numbers(lit):
    txt = re.sub(r"```.*?```", "", lit, flags=re.S)           # code fences (commands) are not result claims
    txt = re.sub(r"\]\([^)]*\)", "]", txt)                      # link targets
    txt = re.sub(r"`[^`]*`", "", txt)                           # inline code
    txt = re.sub(r"<!--.*?-->", "", txt, flags=re.S)
    return [m.group(0) for m in NUM.finditer(txt)]


def main():
    tpl = (rd.ROOT / "docs" / "README.template.md").read_text()
    for _t in TABLES.values():
        _t()                      # pre-register table cells so prose placeholders can reference them
    text, lit = render(tpl)
    bad = sorted({n for n in literal_numbers(lit) if n.lstrip("+-−") not in LITERAL_OK})
    # tokens inside rendered text that are not placeholders are in `lit`; anything not allow-listed is an error
    if bad:
        print("ERROR: numbers in README prose that are not placeholders and not allow-listed:", bad); sys.exit(1)
    reg_txt = {v["text"].lstrip("+-−") for v in REG.items.values()} | {n.lstrip("+-−") for v in REG.items.values() for n in NUM.findall(v["text"])}   # also numeric tokens inside registered strings (CPU name, versions)
    rendered = literal_numbers(text)
    unknown = sorted({n for n in rendered if n.lstrip("+-−") not in reg_txt and n.lstrip("+-−") not in LITERAL_OK})
    if unknown:
        print("ERROR: numbers in rendered README that are neither registered nor allow-listed:", unknown); sys.exit(1)
    if "--check" in sys.argv:
        cur = (rd.ROOT / "README.md").read_text()
        print("README.md matches regenerated text" if cur == text else "MISMATCH: README.md differs from regenerated text"); sys.exit(0 if cur == text else 1)
    (rd.ROOT / "README.md").write_text(text)
    tsv = ["key\ttext\tsource"] + [f"{k}\t{v['text']}\t{v['src']}" for k, v in REG.items.items()]
    lits = sorted({n for n in literal_numbers(lit)}, key=lambda s: (len(s), s))
    tsv += ["", "# literal numbers in prose (allow-listed, not result values)", "literal\treason"] + [f"{n}\t{LITERAL_OK[n.lstrip('+-−')]}" for n in lits]
    (rd.ROOT / "docs" / "readme_numbers.tsv").write_text("\n".join(tsv) + "\n")
    used = sorted(set(re.findall(r"\{\{([^}]+)\}\}", tpl)))
    print(f"README.md written; {len(REG.items)} registered numbers; {len(used)} placeholders; {len(lits)} allow-listed literals")


if __name__ == "__main__":
    main()
