"""Input validation and data shape detection.

Decides which analysis family a request belongs to before any statistics
run: two_group, multi_group, paired_two, categorical, or correlation.
"""

import numpy as np
import pandas as pd

MAX_CATEGORICAL_LEVELS = 20


class InputError(ValueError):
    """Raised for problems the user can fix, always with a plain message."""


def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def detect(df, outcome=None, group=None, paired=None, x=None, y=None):
    """Return a dict describing the analysis shape.

    Keys: kind, and per kind the cleaned data. Group data arrives as an
    ordered dict of label -> numpy array with NaN rows dropped.
    """
    if not isinstance(df, pd.DataFrame):
        raise InputError("Expected a pandas DataFrame, got " + type(df).__name__ + ".")
    if len(df) == 0:
        raise InputError("The data table is empty.")

    if x is not None or y is not None:
        return _detect_correlation(df, x, y)
    if outcome is None or group is None:
        raise InputError(
            "Name the columns to analyze: outcome and group for comparisons, "
            "or x and y for correlation."
        )
    for col in (outcome, group, paired):
        if col is not None and col not in df.columns:
            raise InputError(
                "Column '" + str(col) + "' is not in the data. Available columns: "
                + ", ".join(str(c) for c in df.columns) + "."
            )

    if not is_numeric(df[outcome]):
        return _detect_categorical(df, outcome, group)
    return _detect_groups(df, outcome, group, paired)


def _detect_groups(df, outcome, group, paired):
    keep = [outcome, group] + ([paired] if paired else [])
    data = df[keep].dropna()
    dropped = len(df) - len(data)
    levels = data[group].astype(str)
    names = sorted(levels.unique())

    if len(names) < 2:
        raise InputError(
            "Column '" + str(group) + "' has " + str(len(names))
            + " group after removing missing rows, at least 2 are needed."
        )
    if len(names) > MAX_CATEGORICAL_LEVELS:
        raise InputError(
            "Column '" + str(group) + "' has " + str(len(names))
            + " groups, which looks like a continuous variable. "
            "Use x and y for correlation instead."
        )

    groups = {}
    for name in names:
        values = data.loc[levels == name, outcome].to_numpy(dtype=float)
        if len(values) < 3:
            raise InputError(
                "Group '" + name + "' has only " + str(len(values))
                + " values, at least 3 per group are needed."
            )
        if np.ptp(values) == 0 and len(names) == 1:
            raise InputError("Group '" + name + "' has no variation.")
        groups[name] = values

    info = {
        "outcome": outcome,
        "group": group,
        "groups": groups,
        "dropped_rows": dropped,
    }

    if paired:
        if len(names) != 2:
            raise InputError("Paired analysis needs exactly 2 groups, found " + str(len(names)) + ".")
        wide = data.pivot_table(index=paired, columns=group, values=outcome, aggfunc="first")
        wide = wide.dropna()
        if len(wide) < 3:
            raise InputError(
                "Fewer than 3 complete pairs found using '" + str(paired)
                + "' as the pairing column."
            )
        info["kind"] = "paired_two"
        info["paired"] = paired
        info["pairs"] = (
            wide[names[0]].to_numpy(dtype=float),
            wide[names[1]].to_numpy(dtype=float),
        )
        return info

    info["kind"] = "two_group" if len(names) == 2 else "multi_group"
    return info


def _detect_categorical(df, outcome, group):
    data = df[[outcome, group]].dropna()
    dropped = len(df) - len(data)
    out_levels = data[outcome].astype(str).nunique()
    grp_levels = data[group].astype(str).nunique()
    if out_levels < 2 or grp_levels < 2:
        raise InputError(
            "Both '" + str(outcome) + "' and '" + str(group)
            + "' need at least 2 distinct values for a categorical comparison."
        )
    if out_levels > MAX_CATEGORICAL_LEVELS or grp_levels > MAX_CATEGORICAL_LEVELS:
        raise InputError(
            "More than " + str(MAX_CATEGORICAL_LEVELS)
            + " distinct values in a categorical column, check the column choice."
        )
    table = pd.crosstab(data[outcome].astype(str), data[group].astype(str))
    return {
        "kind": "categorical",
        "outcome": outcome,
        "group": group,
        "table": table,
        "dropped_rows": dropped,
    }


def _detect_correlation(df, x, y):
    if x is None or y is None:
        raise InputError("Correlation needs both x and y column names.")
    for col in (x, y):
        if col not in df.columns:
            raise InputError(
                "Column '" + str(col) + "' is not in the data. Available columns: "
                + ", ".join(str(c) for c in df.columns) + "."
            )
        if not is_numeric(df[col]):
            raise InputError("Column '" + str(col) + "' must be numeric for correlation.")
    data = df[[x, y]].dropna()
    dropped = len(df) - len(data)
    if len(data) < 4:
        raise InputError("Correlation needs at least 4 complete rows.")
    xv = data[x].to_numpy(dtype=float)
    yv = data[y].to_numpy(dtype=float)
    if np.ptp(xv) == 0 or np.ptp(yv) == 0:
        raise InputError("One of the correlation columns has no variation.")
    return {
        "kind": "correlation",
        "x": x,
        "y": y,
        "xv": xv,
        "yv": yv,
        "dropped_rows": dropped,
    }
