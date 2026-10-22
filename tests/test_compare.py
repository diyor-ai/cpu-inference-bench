"""Stability rule and ratio gating of bench/compare.py (needs only pandas + stdlib; no torch/onnxruntime/openvino)."""
import json
import compare


def write_runs(tmp_path, fw, prec, batch, p50s, throttled=False):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    for r, p50 in enumerate(p50s):
        rec = {"framework": fw, "precision": prec, "threads": 6, "result": {
            "batch": batch, "p50_ms": p50, "min_ms": p50 * 0.99, "p95_ms": p50 * 1.02,
            "telemetry": {"freq_mean_mhz": 2600, "freq_min_mhz": 2590, "temp_end_c": 50,
                          "cooldown": {"timed_out": False}, "throttled": throttled and r == 0}}}
        (raw / f"{fw}_{prec}_b{batch}_r{r}.json").write_text(json.dumps(rec))


def stable(tmp_path, fw="pytorch", prec="fp32", batch=1):
    df = compare.aggregate(tmp_path)
    row = df[(df.framework == fw) & (df.precision == prec) & (df.batch == batch)].iloc[0]
    return bool(row.stable), float(row["spread_%"])


def test_clean_five_repeats_is_stable(tmp_path):
    write_runs(tmp_path, "pytorch", "fp32", 1, [100, 100, 100, 100, 100])
    assert stable(tmp_path) == (True, 0.0)


def test_spread_just_below_limit_is_stable(tmp_path):
    write_runs(tmp_path, "pytorch", "fp32", 1, [100, 100, 100, 100, 104.9])
    ok, spread = stable(tmp_path)
    assert ok and spread == 4.9


def test_spread_exactly_at_limit_is_unstable(tmp_path):
    """The rule is spread < 5.0%, so exactly 5.0% is unstable."""
    write_runs(tmp_path, "pytorch", "fp32", 1, [100, 100, 100, 100, 105])
    ok, spread = stable(tmp_path)
    assert spread == 5.0 and not ok


def test_spread_above_limit_is_unstable(tmp_path):
    write_runs(tmp_path, "pytorch", "fp32", 1, [100, 100, 100, 100, 105.1])
    assert stable(tmp_path)[0] is False


def test_throttled_repeat_makes_row_unstable_even_with_tiny_spread(tmp_path):
    write_runs(tmp_path, "pytorch", "fp32", 1, [100, 100, 100, 100, 100.1], throttled=True)
    ok, spread = stable(tmp_path)
    assert spread < compare.SPREAD_LIMIT_PCT and not ok


def test_four_repeats_is_unstable(tmp_path):
    write_runs(tmp_path, "pytorch", "fp32", 1, [100, 100, 100, 100])
    assert compare.MIN_REPEATS == 5 and stable(tmp_path)[0] is False


def test_int8_speedup_only_when_both_sides_stable(tmp_path):
    write_runs(tmp_path, "pytorch", "fp32", 1, [100, 100, 100, 100, 100])
    write_runs(tmp_path, "pytorch", "int8", 1, [50, 50, 50, 50, 50])
    out = compare.int8_vs_fp32(compare.aggregate(tmp_path))
    assert out.iloc[0]["int8_speedup_vs_fp32"] == 2.0


def test_int8_speedup_is_na_when_one_side_unstable(tmp_path):
    write_runs(tmp_path, "pytorch", "fp32", 1, [100, 100, 100, 100, 110])     # 10% spread -> unstable
    write_runs(tmp_path, "pytorch", "int8", 1, [50, 50, 50, 50, 50])
    out = compare.int8_vs_fp32(compare.aggregate(tmp_path))
    assert out.iloc[0]["int8_speedup_vs_fp32"] == "n/a (unstable)"
    write_runs(tmp_path, "pytorch", "fp32", 8, [100, 100, 100, 100, 100])
    write_runs(tmp_path, "pytorch", "int8", 8, [50, 50, 50, 50, 70])          # int8 side unstable
    out = compare.int8_vs_fp32(compare.aggregate(tmp_path))
    assert out[out.batch == 8].iloc[0]["int8_speedup_vs_fp32"] == "n/a (unstable)"


def test_cross_framework_ratio_is_na_when_unstable(tmp_path):
    write_runs(tmp_path, "pytorch", "fp32", 1, [100, 100, 100, 100, 100])
    write_runs(tmp_path, "onnxruntime", "fp32", 1, [50, 50, 50, 50, 50])
    write_runs(tmp_path, "openvino", "fp32", 1, [50, 50, 50, 50, 60])         # unstable
    r = compare.ratios(compare.aggregate(tmp_path)).set_index("framework")["speedup_vs_pytorch"]
    assert r["onnxruntime"] == 2.0 and r["openvino"] == "n/a (unstable)"
