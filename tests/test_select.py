"""Route synthetic data down every branch of the decision tree and check
the chosen test is the one the rules promise."""

import numpy as np
import pandas as pd

from statsage.detect import detect
from statsage.select import choose_and_run


def frame(groups):
    rows = []
    for name, values in groups.items():
        for v in values:
            rows.append({"grp": name, "value": float(v)})
    return pd.DataFrame(rows)


def run(df, **kwargs):
    info = detect(df, **kwargs)
    return choose_and_run(info)


def test_normal_two_groups_gets_welch():
    rng = np.random.default_rng(42)
    df = frame({"a": rng.normal(10, 2, 25), "b": rng.normal(12, 2, 25)})
    test, checks, trace, warnings = run(df, outcome="value", group="grp")
    assert test["name"] == "Welch's t-test"
    assert any("Welch" in step for step in trace)


def test_skewed_small_two_groups_gets_mann_whitney():
    rng = np.random.default_rng(3)
    df = frame({
        "a": np.exp(rng.normal(0, 1.5, 15)),
        "b": np.exp(rng.normal(1, 1.5, 15)),
    })
    test, checks, trace, warnings = run(df, outcome="value", group="grp")
    assert test["name"] == "Mann-Whitney U test"


def test_skewed_large_two_groups_keeps_welch_with_warning():
    rng = np.random.default_rng(5)
    df = frame({
        "a": np.exp(rng.normal(0, 1.2, 80)),
        "b": np.exp(rng.normal(0.5, 1.2, 80)),
    })
    test, checks, trace, warnings = run(df, outcome="value", group="grp")
    assert test["name"] == "Welch's t-test"
    assert any("sensitivity" in w for w in warnings)


def test_three_normal_equal_var_gets_anova_tukey():
    rng = np.random.default_rng(42)
    df = frame({
        "a": rng.normal(10, 2, 25),
        "b": rng.normal(12, 2, 25),
        "c": rng.normal(14, 2, 25),
    })
    test, checks, trace, warnings = run(df, outcome="value", group="grp")
    assert test["name"] == "One-way ANOVA"
    assert test["posthoc"][0]["method"] == "Tukey HSD"


def test_three_normal_unequal_var_gets_welch_anova():
    rng = np.random.default_rng(42)
    df = frame({
        "a": rng.normal(10, 1, 30),
        "b": rng.normal(12, 6, 30),
        "c": rng.normal(14, 12, 30),
    })
    test, checks, trace, warnings = run(df, outcome="value", group="grp")
    assert test["name"] == "Welch's ANOVA"
    assert test["posthoc"][0]["method"] == "Games-Howell"


def test_three_skewed_small_gets_kruskal_dunn():
    rng = np.random.default_rng(3)
    df = frame({
        "a": np.exp(rng.normal(0, 1.5, 12)),
        "b": np.exp(rng.normal(1, 1.5, 12)),
        "c": np.exp(rng.normal(2, 1.5, 12)),
    })
    test, checks, trace, warnings = run(df, outcome="value", group="grp")
    assert test["name"] == "Kruskal-Wallis H test"
    assert "Dunn" in test["posthoc"][0]["method"]


def test_paired_normal_gets_paired_t():
    rng = np.random.default_rng(42)
    base = rng.normal(10, 2, 20)
    rows = []
    for i, v in enumerate(base):
        rows.append({"subject": i, "grp": "pre", "value": v})
        rows.append({"subject": i, "grp": "post", "value": v + rng.normal(1, 0.5)})
    df = pd.DataFrame(rows)
    test, checks, trace, warnings = run(df, outcome="value", group="grp", paired="subject")
    assert test["name"] == "Paired t-test"


def test_paired_skewed_gets_wilcoxon():
    rng = np.random.default_rng(4)
    base = rng.normal(10, 2, 20)
    rows = []
    for i, v in enumerate(base):
        rows.append({"subject": i, "grp": "pre", "value": v})
        rows.append({"subject": i, "grp": "post", "value": v + np.exp(rng.normal(0, 1.6))})
    df = pd.DataFrame(rows)
    test, checks, trace, warnings = run(df, outcome="value", group="grp", paired="subject")
    assert test["name"] == "Wilcoxon signed-rank test"


def test_big_contingency_gets_chi_square():
    df = pd.DataFrame({
        "result": (["yes"] * 40 + ["no"] * 20) + (["yes"] * 20 + ["no"] * 40),
        "arm": ["drug"] * 60 + ["placebo"] * 60,
    })
    test, checks, trace, warnings = run(df, outcome="result", group="arm")
    assert "Chi-square" in test["name"]


def test_sparse_2x2_gets_fisher():
    df = pd.DataFrame({
        "result": ["yes", "yes", "yes", "no", "yes", "no", "no", "no", "no", "no"],
        "arm": ["drug"] * 5 + ["placebo"] * 5,
    })
    test, checks, trace, warnings = run(df, outcome="result", group="arm")
    assert test["name"] == "Fisher's exact test"


def test_normal_xy_gets_pearson():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 40)
    df = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(0, 0.5, 40)})
    test, checks, trace, warnings = run(df, x="x", y="y")
    assert test["name"] == "Pearson correlation"


def test_skewed_xy_gets_spearman():
    rng = np.random.default_rng(3)
    x = np.exp(rng.normal(0, 1.5, 40))
    df = pd.DataFrame({"x": x, "y": x ** 2 + np.exp(rng.normal(0, 1.0, 40))})
    test, checks, trace, warnings = run(df, x="x", y="y")
    assert test["name"] == "Spearman rank correlation"
