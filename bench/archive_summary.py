"""Derive the small numbers the README quotes about archived (invalid/superseded) runs and write results/archive_summary.json.
The archive_* directories stay local (git-ignored); this summary is tracked so the README can be rebuilt and checked from a clean clone (CI).
Run from the repo root: .venv/bin/python bench/archive_summary.py"""
import json
from pathlib import Path
R = Path(__file__).resolve().parent.parent / "results"

pairs = []
for f in sorted((R / "archive_noisy_first_attempt").glob("*_run2.json")):
    a = json.loads(Path(str(f).replace("_run2", "")).read_text())["results"]; b = json.loads(f.read_text())["results"]
    pairs += [(x["p50_ms"], y["p50_ms"]) for x, y in zip(a, b)]
p3 = json.loads((R / "archive_pilot_3000mhz" / "summary.json").read_text())
bug = [json.loads((R / "archive_bert_tokenizer_bug" / f"{n}_{fw}_fp32_L128.json").read_text())["acc"] for n in "AB" for fw in ("pytorch", "onnx", "openvino")]
assert len(set(bug)) == 1
out = {
    "noisy_first_attempt": {"source": "results/archive_noisy_first_attempt/*.json (local archive)", "pairs": len(pairs),
                            "pairs_over_5pct": sum((max(a, b) - min(a, b)) / min(a, b) > 0.05 for a, b in pairs),
                            "rule": "(max - min) / min of run vs run2 p50 > 5%"},
    "pilot_3000mhz": {"source": "results/archive_pilot_3000mhz/summary.json (local archive)", "rows": len(p3),
                      "throttled_rows": sum(r["throttled"] for r in p3), "cap_mhz": 3000},
    "bert_tokenizer_bug": {"source": "results/archive_bert_tokenizer_bug/*.json (local archive)", "runs": len(bug), "acc_all_runs": bug[0]},
}
(R / "archive_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
