"""Assumption checks.

Each check returns a dict with: name, passed (True, False, or None when
not applicable), detail (plain language), and any numbers used. The
report prints these verbatim, so wording stays readable to non
statisticians.
"""

import numpy as np
from scipy import stats

SMALL_GROUP_N = 30
SHAPIRO_MAX_N = 50
ALPHA_DEFAULT = 0.05


def normality(values, label, alpha=ALPHA_DEFAULT):
    n = len(values)
    if np.ptp(values) == 0:
        return {
            "name": "Normality (" + label + ")",
            "passed": False,
            "p": None,
            "detail": "All values are identical, normality cannot hold.",
        }
    if n < 8:
        return {
            "name": "Normality (" + label + ")",
            "passed": None,
            "p": None,
            "detail": "Only " + str(n) + " values, too few to test normality reliably. "
                      "Treated as unknown, small sample rules apply.",
        }
    if n <= SHAPIRO_MAX_N:
        stat, p = stats.shapiro(values)
        test_name = "Shapiro-Wilk"
    else:
        stat, p = stats.normaltest(values)
        test_name = "D'Agostino K-squared"
    passed = bool(p > alpha)
    verdict = "consistent with a normal distribution" if passed else "not normally distributed"
    return {
        "name": "Normality (" + label + ")",
        "passed": passed,
        "p": float(p),
        "detail": test_name + " p = " + format_p(p) + ", data look " + verdict + ".",
    }


def equal_variance(groups, alpha=ALPHA_DEFAULT):
    arrays = list(groups.values())
    stat, p = stats.levene(*arrays, center="median")
    passed = bool(p > alpha)
    verdict = "similar across groups" if passed else "different between groups"
    return {
        "name": "Equal variances",
        "passed": passed,
        "p": float(p),
        "detail": "Levene's test (median centered) p = " + format_p(p)
                  + ", variances look " + verdict + ".",
    }


def sample_sizes(groups):
    smallest = min(len(v) for v in groups.values())
    parts = [name + " n = " + str(len(v)) for name, v in groups.items()]
    if smallest >= SMALL_GROUP_N:
        detail = ", ".join(parts) + ". All groups reach n of 30, large sample behavior applies."
        passed = True
    else:
        detail = ", ".join(parts) + ". Smallest group is below 30, results lean on the normality checks."
        passed = None
    return {"name": "Sample sizes", "passed": passed, "p": None, "detail": detail}


def outliers(groups):
    flagged = []
    for name, values in groups.items():
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        count = int(np.sum((values < low) | (values > high)))
        if count > 0:
            flagged.append(name + " has " + str(count))
    if flagged:
        detail = ("Potential outliers by the 1.5 IQR rule: " + ", ".join(flagged)
                  + ". Worth a look at the raw data, they were kept in the analysis.")
        passed = None
    else:
        detail = "No outliers flagged by the 1.5 IQR rule."
        passed = True
    return {"name": "Outliers", "passed": passed, "p": None, "detail": detail}


def expected_counts(table):
    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    below5 = float(np.mean(expected < 5))
    minimum = float(expected.min())
    ok = below5 <= 0.2 and minimum >= 1
    if ok:
        detail = "All expected cell counts are adequate for chi-square."
    else:
        detail = (str(int(round(below5 * 100)))
                  + " percent of expected counts fall below 5 (minimum "
                  + format_num(minimum) + "), chi-square becomes unreliable.")
    return {
        "name": "Expected counts",
        "passed": ok,
        "p": None,
        "detail": detail,
        "min_expected": minimum,
    }


def run_group_checks(groups, alpha=ALPHA_DEFAULT):
    results = [sample_sizes(groups)]
    for name, values in groups.items():
        results.append(normality(values, name, alpha))
    if len(groups) >= 2:
        results.append(equal_variance(groups, alpha))
    results.append(outliers(groups))
    return results


def all_normal(check_results):
    """True when no normality check failed. Unknown (None) counts as pass,
    the sample size rule handles small groups separately."""
    for c in check_results:
        if c["name"].startswith("Normality") and c["passed"] is False:
            return False
    return True


def variances_equal(check_results):
    for c in check_results:
        if c["name"] == "Equal variances":
            return c["passed"] is not False
    return True


def smallest_n(groups):
    return min(len(v) for v in groups.values())


def format_p(p):
    if p is None:
        return "NA"
    if p < 0.001:
        return "< 0.001"
    return "{:.3f}".format(p)


def format_num(x, digits=3):
    if x is None:
        return "NA"
    if abs(x) >= 1000 or (abs(x) < 0.001 and x != 0):
        return "{:.2e}".format(x)
    return ("{:." + str(digits) + "g}").format(x)
