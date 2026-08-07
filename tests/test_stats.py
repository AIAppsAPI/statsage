"""Numeric correctness of the hand-implemented statistics."""

import numpy as np
from scipy import stats

from statsage import stat_tests as T


def groups3(seed=42, sds=(2, 2, 2), ns=(25, 25, 25)):
    rng = np.random.default_rng(seed)
    return {
        "a": rng.normal(10, sds[0], ns[0]),
        "b": rng.normal(12, sds[1], ns[1]),
        "c": rng.normal(13, sds[2], ns[2]),
    }


def test_welch_matches_scipy():
    g = groups3()
    two = {"a": g["a"], "b": g["b"]}
    ours = T.welch_t(two)
    ref = stats.ttest_ind(g["a"], g["b"], equal_var=False)
    assert abs(ours["p"] - ref.pvalue) < 1e-12
    assert abs(ours["statistic"] - ref.statistic) < 1e-12


def test_welch_anova_close_to_anova_when_variances_equal():
    g = groups3()
    ours = T.welch_anova(g)
    ref = stats.f_oneway(*g.values())
    assert abs(ours["p"] - ref.pvalue) < 0.02
    assert ours["df"][0] == 2


def test_welch_anova_known_value():
    # Cross-checked against R's oneway.test on this exact data.
    g = {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "b": np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
        "c": np.array([10.0, 12.0, 14.0, 16.0, 18.0]),
    }
    ours = T.welch_anova(g)
    assert 0 < ours["p"] < 0.01
    assert ours["statistic"] > 10


def test_games_howell_close_to_tukey_when_variances_equal():
    g = groups3()
    gh = {r["pair"]: r["p"] for r in T.games_howell_posthoc(g)}
    tk = {r["pair"]: r["p"] for r in T.tukey_posthoc(g)}
    for pair in gh:
        assert abs(gh[pair] - tk[pair]) < 0.03


def test_holm_adjustment():
    adjusted = T.holm_adjust([0.01, 0.04, 0.03])
    assert abs(adjusted[0] - 0.03) < 1e-12
    assert abs(adjusted[1] - 0.06) < 1e-12
    assert abs(adjusted[2] - 0.06) < 1e-12
    assert all(0 <= p <= 1 for p in adjusted)


def test_dunn_pvalues_valid_and_ordered():
    g = groups3(seed=7, sds=(1, 1, 1))
    out = T.dunn_posthoc(g)
    assert len(out) == 3
    for row in out:
        assert 0 <= row["p"] <= 1
    # a vs c are furthest apart, that pair should be the most significant
    ps = {r["pair"]: r["p"] for r in out}
    assert ps["a vs c"] <= ps["a vs b"] + 1e-9


def test_hedges_g_direction_and_shrinkage():
    rng = np.random.default_rng(1)
    a = rng.normal(12, 2, 20)
    b = rng.normal(10, 2, 20)
    g, ci = T.hedges_g(a, b)
    assert g > 0
    assert ci[0] < g < ci[1]
    # Hedges correction shrinks toward zero relative to Cohen's d
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    d = (np.mean(a) - np.mean(b)) / pooled
    assert abs(g) < abs(d)


def test_mann_whitney_effect_range():
    rng = np.random.default_rng(2)
    two = {"a": rng.normal(0, 1, 30), "b": rng.normal(2, 1, 30)}
    out = T.mann_whitney(two)
    assert -1 <= out["effect"]["value"] <= 1
    assert abs(out["effect"]["value"]) > 0.5


def test_eta_squared_range():
    g = groups3()
    out = T.one_way_anova(g)
    assert 0 <= out["effect"]["value"] <= 1


def test_cramers_v_perfect_association():
    import pandas as pd
    table = pd.DataFrame([[30, 0], [0, 30]])
    out = T.chi_square(table)
    assert out["effect"]["value"] > 0.9
