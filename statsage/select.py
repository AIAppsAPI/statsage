"""Test selection decision tree.

Takes the detected shape and the assumption checks, picks one test, runs
it, and records every step of the reasoning as plain language so the
report can show why this test and not another.
"""

from . import checks as C
from . import stat_tests as T


def choose_and_run(info, alpha=0.05):
    """Return (test_result, check_results, trace, warnings)."""
    kind = info["kind"]
    if kind == "two_group":
        return _two_group(info, alpha)
    if kind == "multi_group":
        return _multi_group(info, alpha)
    if kind == "paired_two":
        return _paired(info, alpha)
    if kind == "categorical":
        return _categorical(info)
    if kind == "correlation":
        return _correlation(info, alpha)
    raise ValueError("Unknown analysis kind: " + str(kind))


def _two_group(info, alpha):
    groups = info["groups"]
    results = C.run_group_checks(groups, alpha)
    trace = ["Outcome '" + info["outcome"] + "' is numeric and '" + info["group"]
             + "' has 2 independent groups, so this is a two group comparison."]
    warnings = []
    normal = C.all_normal(results)
    small = C.smallest_n(groups) < C.SMALL_GROUP_N
    if normal:
        trace.append("Normality checks passed in both groups.")
        trace.append("Welch's t-test was chosen. It does not assume equal variances, "
                     "which makes it the safer default over Student's t-test.")
        test = T.welch_t(groups)
    elif not small:
        trace.append("Normality failed in at least one group, but every group has "
                     "n of 30 or more, where the t-test is robust to non-normality.")
        trace.append("Welch's t-test was kept, with a robustness note.")
        warnings.append("Normality failed but sample sizes are large. If the data are "
                        "heavily skewed, compare with a Mann-Whitney U as a sensitivity check.")
        test = T.welch_t(groups)
    else:
        trace.append("Normality failed in at least one group and the smallest group "
                     "is under 30, so a rank based test is the safer choice.")
        trace.append("Mann-Whitney U test was chosen.")
        test = T.mann_whitney(groups)
    return test, results, trace, warnings


def _multi_group(info, alpha):
    groups = info["groups"]
    results = C.run_group_checks(groups, alpha)
    k = len(groups)
    trace = ["Outcome '" + info["outcome"] + "' is numeric and '" + info["group"]
             + "' has " + str(k) + " groups, so this is a multi group comparison."]
    warnings = []
    normal = C.all_normal(results)
    small = C.smallest_n(groups) < C.SMALL_GROUP_N
    equal_var = C.variances_equal(results)
    if normal or not small:
        if not normal:
            trace.append("Normality failed somewhere, but all groups have n of 30 "
                         "or more, where ANOVA is robust.")
            warnings.append("Normality failed but sample sizes are large. "
                            "A Kruskal-Wallis sensitivity check is reasonable.")
        else:
            trace.append("Normality checks passed in all groups.")
        if equal_var:
            trace.append("Variances look similar, one-way ANOVA with Tukey HSD post-hoc was chosen.")
            test = T.one_way_anova(groups)
        else:
            trace.append("Variances differ between groups, Welch's ANOVA with "
                         "Games-Howell post-hoc was chosen.")
            test = T.welch_anova(groups)
    else:
        trace.append("Normality failed and at least one group is under 30, "
                     "Kruskal-Wallis with Dunn's post-hoc (Holm adjusted) was chosen.")
        test = T.kruskal_wallis(groups)
    return test, results, trace, warnings


def _paired(info, alpha):
    a, b = info["pairs"]
    labels = list(info["groups"].keys())
    diff = a - b
    diff_check = C.normality(diff, "paired differences", alpha)
    results = [C.sample_sizes({"pairs": a}), diff_check,
               C.outliers({labels[0]: a, labels[1]: b})]
    trace = ["A pairing column was given, so the two groups are treated as "
             "repeated measurements and the test runs on the paired differences."]
    warnings = []
    if info["dropped_rows"] > 0:
        warnings.append(str(info["dropped_rows"]) + " rows without a complete pair were dropped.")
    if diff_check["passed"] is not False:
        trace.append("The paired differences look normal, paired t-test was chosen.")
        test = T.paired_t(a, b, labels)
    else:
        trace.append("The paired differences are not normal, "
                     "Wilcoxon signed-rank test was chosen.")
        test = T.wilcoxon_signed_rank(a, b, labels)
    return test, results, trace, warnings


def _categorical(info):
    table = info["table"]
    counts_check = C.expected_counts(table)
    results = [counts_check]
    trace = ["Both '" + info["outcome"] + "' and '" + info["group"]
             + "' are categorical, so this is a test of independence on a "
             + str(table.shape[0]) + " by " + str(table.shape[1]) + " table."]
    warnings = []
    if counts_check["passed"]:
        trace.append("Expected counts are adequate, chi-square test was chosen.")
        test = T.chi_square(table)
    elif table.shape == (2, 2):
        trace.append("Expected counts are low and the table is 2 by 2, "
                     "Fisher's exact test was chosen.")
        test = T.fisher_exact(table)
    else:
        trace.append("Expected counts are low on a table larger than 2 by 2. "
                     "Chi-square was run but treat the p-value with caution.")
        warnings.append("Low expected counts make chi-square unreliable here. "
                        "Consider pooling sparse categories.")
        test = T.chi_square(table)
    return test, results, trace, warnings


def _correlation(info, alpha):
    xv, yv = info["xv"], info["yv"]
    labels = (info["x"], info["y"])
    results = [C.normality(xv, info["x"], alpha), C.normality(yv, info["y"], alpha),
               C.outliers({info["x"]: xv, info["y"]: yv})]
    trace = ["Two numeric columns were given, so this is a correlation analysis."]
    warnings = []
    both_normal = all(c["passed"] is not False for c in results[:2])
    if both_normal:
        trace.append("Both variables look normal, Pearson correlation was chosen.")
        test = T.pearson(xv, yv, labels)
    else:
        trace.append("At least one variable is not normal, "
                     "Spearman rank correlation was chosen.")
        test = T.spearman(xv, yv, labels)
    return test, results, trace, warnings
