import numpy as np
import pandas as pd
import pytest

from statsage import InputError
from statsage.detect import detect


def make_df():
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "value": rng.normal(10, 2, 40),
        "grp": ["a"] * 20 + ["b"] * 20,
        "subject": list(range(20)) * 2,
    })


def test_two_group_detection():
    info = detect(make_df(), outcome="value", group="grp")
    assert info["kind"] == "two_group"
    assert set(info["groups"]) == {"a", "b"}


def test_missing_column_message():
    with pytest.raises(InputError, match="not in the data"):
        detect(make_df(), outcome="nope", group="grp")


def test_single_group_rejected():
    df = make_df()
    df["grp"] = "only"
    with pytest.raises(InputError, match="at least 2"):
        detect(df, outcome="value", group="grp")


def test_tiny_group_rejected():
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0], "grp": ["a", "a", "a", "b"]})
    with pytest.raises(InputError, match="at least 3 per group"):
        detect(df, outcome="value", group="grp")


def test_empty_df_rejected():
    with pytest.raises(InputError, match="empty"):
        detect(pd.DataFrame(), outcome="value", group="grp")


def test_paired_detection():
    info = detect(make_df(), outcome="value", group="grp", paired="subject")
    assert info["kind"] == "paired_two"
    assert len(info["pairs"][0]) == 20


def test_categorical_detection():
    df = pd.DataFrame({
        "result": ["yes", "no"] * 30,
        "arm": ["drug"] * 30 + ["placebo"] * 30,
    })
    info = detect(df, outcome="result", group="arm")
    assert info["kind"] == "categorical"
    assert info["table"].shape == (2, 2)


def test_correlation_detection():
    df = make_df()
    df["second"] = df["value"] * 2 + 1
    info = detect(df, x="value", y="second")
    assert info["kind"] == "correlation"


def test_correlation_needs_numeric():
    df = make_df()
    with pytest.raises(InputError, match="numeric"):
        detect(df, x="value", y="grp")


def test_nan_rows_dropped():
    df = make_df()
    df.loc[0, "value"] = np.nan
    info = detect(df, outcome="value", group="grp")
    assert info["dropped_rows"] == 1
