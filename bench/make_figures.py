"""Stage 4 figures -> docs/figures/*.png. Data only from results/ via report_data.py. Run with the main venv (.venv).
Colour-blind-safe (Okabe-Ito); FP32 = light tint, INT8 = full colour; UNSTABLE rows are hatched and labelled, never dropped."""
import sys, textwrap
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import report_data as rd

OUT = rd.ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
COL = {"pytorch": "#E69F00", "onnxruntime": "#0072B2", "openvino": "#009E73"}   # Okabe-Ito
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "hatch.linewidth": 0.8})
D = rd.load_all()
env = D["env"]
cap_mhz = int(env["scaling_max_khz"]["0"]) // 1000
COND = (f"{env['cpu']} | 6 P-cores pinned, SMT off, 6 threads, fixed {cap_mhz} MHz, EPP {env['epp']} | 5 repeats x 100 iterations, "
        "bar = median of repeat p50s | hatched + 'unstable' = repeat spread >= 5% or thermal-throttle counter rose")


def bars(ax, groups, rows_of, unit_note=None):
    """groups: list of (label, [(fw, prec, row)]) -> grouped bars; FP32 light, INT8 solid; unstable hatched."""
    w = 0.38
    xt, xl = [], []
    for gi, (label, items) in enumerate(groups):
        for fi, fw in enumerate(rd.FW):
            for pi, prec in enumerate(("fp32", "int8")):
                r = items.get((fw, prec))
                if r is None:
                    continue
                x = gi * 4 + fi + (pi - 0.5) * w * 1.05
                st = r["stable"]
                ax.bar(x, r["p50_ms"], w, color=COL[fw], alpha=0.35 if prec == "fp32" else 1.0, edgecolor="black" if not st else "#444",
                       linewidth=0.5, hatch=None if st else "////")
                ax.text(x, r["p50_ms"], "%.1f" % r["p50_ms"], ha="center", va="bottom", fontsize=6.5, rotation=90)
                if not st:
                    ax.text(x, r["p50_ms"] / 2, "unstable", ha="center", va="center", fontsize=6.5, rotation=90,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#444", lw=0.5))
        xt.append(gi * 4 + 1); xl.append(label)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("p50 latency per batch (ms)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)


def foot(fig, text, width=150):
    fig.text(0.01, 0.005, "\n".join(textwrap.wrap(text, width)), fontsize=6, va="bottom")


def legend(fig):
    h = [Patch(color=COL[f], label=rd.FWLABEL[f]) for f in rd.FW]
    h += [Patch(facecolor="#888", alpha=0.35, label="FP32"), Patch(facecolor="#888", label="INT8"),
          Patch(facecolor="white", edgecolor="#444", hatch="////", label="unstable row")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=6, frameon=False, fontsize=8)


def index(rows, **fixed):
    out = {}
    for r in rows:
        if all(r.get(k) == v for k, v in fixed.items()):
            out[(r["framework"], r["precision"])] = r
    return out


def fig_a():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, b in zip(axes, (1, 32)):
        bars(ax, [(f"batch {b}", index(D["s2"], batch=b))], None)
        ax.set_xticks([1]); ax.set_xticklabels([f"ResNet-18, batch {b}"])
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels([rd.FWLABEL[f] for f in rd.FW])
    fig.suptitle("ResNet-18 (pretrained): FP32 vs INT8 latency", y=0.985)
    legend(fig); foot(fig, COND)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9)); fig.savefig(OUT / "resnet_fp32_vs_int8.png", dpi=170); plt.close(fig)


def fig_b():
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, (seq, b) in zip(axes.flat, ((32, 1), (32, 8), (128, 1), (128, 8))):
        bars(ax, [("", index([r for r in D["bert"] if r["model"] == "A"], seq=seq, batch=b))], None)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels([rd.FWLABEL[f] for f in rd.FW]); ax.set_title(f"seq {seq}, batch {b}", fontsize=9)
    fig.suptitle("BERT A (DistilBERT, SST-2): FP32 vs INT8 latency", y=0.985)
    legend(fig); foot(fig, COND + " | INT8 = each framework's standard INT8 path (not like-for-like)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.9)); fig.savefig(OUT / "bert_a_fp32_vs_int8.png", dpi=170); plt.close(fig)


def fig_c():
    rows = []
    for s in D["racc"]:
        rows.append((f"ResNet-18 {rd.FWLABEL[s['framework']]}{' (' + s['variant'].strip('_') + ')' if s['variant'] != '-' else ''}", s, False))
    for s in D["bacc"]:
        rows.append((f"BERT {s['model']} seq {s['seq']} {rd.FWLABEL[s['framework']]}", s, True))
    fig, ax = plt.subplots(figsize=(8, 6.2))
    for i, (lab, s, bert) in enumerate(rows):
        y = len(rows) - i
        c = COL[s["framework"]]
        ax.errorbar(s["diff"], y, xerr=[[s["diff"] - s["lo"]], [s["hi"] - s["diff"]]], fmt="o", color=c, ecolor=c, capsize=3, ms=5)
        ax.text(2.55, y, f"p={s['p']:.3f}" + (f" (Bonf. {s['p_bonf']:.2f})" if bert else ""), va="center", fontsize=6.5)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks([len(rows) - i for i in range(len(rows))]); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlim(-3.5, 4.8)
    ax.set_xlabel("top-1 accuracy: INT8 minus FP32 (percentage points), 95% paired-bootstrap CI")
    ax.set_title("INT8 accuracy change vs FP32 (paired, same samples)")
    foot(fig, "ResNet-18: 1000 ImageNet validation images (sampled); BERT: SST-2 validation, 872 sentences; seq 32 truncates 24% of sentences. "
             "p = exact McNemar (raw); Bonferroni over 9 BERT tests. Line at 0 = no change.", 130)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(OUT / "accuracy_int8_minus_fp32.png", dpi=170); plt.close(fig)


def fig_d():
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2))
    for ax, b in zip(axes, (1, 8, 32)):
        for fi, fw in enumerate(rd.FW):
            r = rd.row(D["p1"], framework=fw, batch=b)
            ax.bar(fi, r["p50_ms"], 0.6, color=COL[fw], edgecolor="#444", linewidth=0.5, hatch=None if r["stable"] else "////")
            ax.text(fi, r["p50_ms"], "%.1f" % r["p50_ms"], ha="center", va="bottom", fontsize=7)
            if not r["stable"]:
                ax.text(fi, r["p50_ms"] / 2, "unstable", ha="center", va="center", fontsize=7, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#444", lw=0.5))
        ax.set_xticks(range(3)); ax.set_xticklabels([rd.FWLABEL[f] for f in rd.FW], fontsize=8); ax.set_title(f"batch {b}", fontsize=9)
        ax.set_ylabel("p50 latency per batch (ms)"); ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    fig.suptitle("ResNet-18 FP32: framework comparison (random weights, phase 1)", y=0.995)
    foot(fig, COND, 160)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95)); fig.savefig(OUT / "resnet_fp32_frameworks.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    fig_a(); fig_b(); fig_c(); fig_d()
    print("figures:", *sorted(p.name for p in OUT.glob("*.png")))
