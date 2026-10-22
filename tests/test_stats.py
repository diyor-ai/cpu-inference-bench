"""McNemar exact test and paired bootstrap helpers in bench/report_data.py (numpy + stdlib only)."""
import numpy as np
import pytest
import report_data as rd


def test_mcnemar_equal_discordant_pairs_gives_one():
    assert rd.mcnemar_exact(5, 5) == 1.0
    assert rd.mcnemar_exact(0, 0) == 1.0


def test_mcnemar_zero_vs_four_is_one_eighth():
    """2 * P(X <= 0 | n=4, p=0.5) = 2 / 16."""
    assert rd.mcnemar_exact(0, 4) == pytest.approx(0.125)
    assert rd.mcnemar_exact(4, 0) == pytest.approx(0.125)


def test_mcnemar_symmetric_and_capped():
    assert rd.mcnemar_exact(3, 9) == rd.mcnemar_exact(9, 3)
    assert 0.0 <= rd.mcnemar_exact(1, 0) <= 1.0


@pytest.mark.parametrize("b, c, expected", [
    (16, 7, 0.0931396484375),     # reference values cross-checked with scipy.stats.binomtest(c, b + c, 0.5)
    (27, 13, 0.03847730828420026),
])
def test_mcnemar_matches_scipy_reference(b, c, expected):
    assert rd.mcnemar_exact(b, c) == pytest.approx(expected, rel=1e-12)


def test_paired_counts_and_difference():
    s = rd.paired(np.random.default_rng(0), [1, 1, 0, 0], [1, 0, 1, 1])
    assert s["fp32_only"] == 1 and s["int8_only"] == 2
    assert s["diff"] == pytest.approx(25.0)
    assert s["lo"] <= s["diff"] <= s["hi"]
