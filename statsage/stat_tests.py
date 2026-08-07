"""Statistical test runners and effect sizes.

Every runner returns a dict with the same shape so the report layer can
render any test:
    name, statistic_label, statistic, df, p, ci (may be None),
    effect: {name, value, interpretation}, descriptives, posthoc (list).

Only numpy and scipy are used. Welch's ANOVA, Games-Howell, and Dunn's
test are implemented here from their standard formulas because scipy
does not ship them.
"""

import itertools

import numpy as np
from scipy import stats

from .checks import format_num, format_p


def _interpret(value, small, medium, large, name):
    v = abs(value)
    if v < small:
        size = "negligible"
    elif v < medium:
        size = "small"
    elif v < large:
        size = "medium"
    else:
        size = "large"
    return "a " + size + " effect (" + name + " = " + format_num(value) + ")"


def _descriptives(groups):
    rows = []
    for name, values in groups.items():
        rows.append({
            "group": name,
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "median": float(np.median(values)),
        })
    return rows


def hedges_g(a, b):
    n1, n2 = len(a), len(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    dfp = n1 + n2 - 2
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / dfp)
    if pooled == 0:
        return 0.0, (0.0, 0.0)
    d = (np.mean(a) - np.mean(b)) / pooled
    j = 1.0 - 3.0 / (4.0 * dfp - 1.0)
    g = j * d
    se = np.sqrt((n1 + n2) / (n1 * n2) + g * g / (2.0 * (n1 + n2)))
    return float(g), (float(g - 1.96 * se), float(g + 1.96 * se))


# ---------------------------------------------------------------- two groups

def welch_t(groups):
    (na, a), (nb, b) = [(k, v) for k, v in groups.items()]
    res = stats.ttest_ind(a, b, equal_var=False)
    ci = res.confidence_interval(0.95)
    g, gci = hedges_g(a, b)
    return {
        "name": "Welch's t-test",
        "statistic_label": "t",
        "statistic": float(res.statistic),
        "df": float(res.df),
        "p": float(res.pvalue),
        "ci": {"label": "95% CI of mean difference", "low": float(ci.low), "high": float(ci.high)},
        "effect": {"name": "Hedges' g", "value": g, "ci": gci,
                   "interpretation": _interpret(g, 0.2, 0.5, 0.8, "g")},
        "descriptives": _descriptives(groups),
        "posthoc": [],
    }


def student_t(groups):
    (na, a), (nb, b) = [(k, v) for k, v in groups.items()]
    res = stats.ttest_ind(a, b, equal_var=True)
    ci = res.confidence_interval(0.95)
    g, gci = hedges_g(a, b)
    return {
        "name": "Student's t-test",
        "statistic_label": "t",
        "statistic": float(res.statistic),
        "df": float(res.df),
        "p": float(res.pvalue),
        "ci": {"label": "95% CI of mean difference", "low": float(ci.low), "high": float(ci.high)},
        "effect": {"name": "Hedges' g", "value": g, "ci": gci,
                   "interpretation": _interpret(g, 0.2, 0.5, 0.8, "g")},
        "descriptives": _descriptives(groups),
        "posthoc": [],
    }


def mann_whitney(groups):
    (na, a), (nb, b) = [(k, v) for k, v in groups.items()]
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    u = float(res.statistic)
    r = 1.0 - 2.0 * u / (len(a) * len(b))
    return {
        "name": "Mann-Whitney U test",
        "statistic_label": "U",
        "statistic": u,
        "df": None,
        "p": float(res.pvalue),
        "ci": None,
        "effect": {"name": "rank-biserial r", "value": float(r), "ci": None,
                   "interpretation": _interpret(r, 0.1, 0.3, 0.5, "r")},
        "descriptives": _descriptives(groups),
        "posthoc": [],
    }


def paired_t(a, b, labels):
    res = stats.ttest_rel(a, b)
    ci = res.confidence_interval(0.95)
    diff = a - b
    sd = np.std(diff, ddof=1)
    dz = float(np.mean(diff) / sd) if sd > 0 else 0.0
    groups = {labels[0]: a, labels[1]: b}
    return {
        "name": "Paired t-test",
        "statistic_label": "t",
        "statistic": float(res.statistic),
        "df": float(res.df),
        "p": float(res.pvalue),
        "ci": {"label": "95% CI of mean difference", "low": float(ci.low), "high": float(ci.high)},
        "effect": {"name": "Cohen's dz", "value": dz, "ci": None,
                   "interpretation": _interpret(dz, 0.2, 0.5, 0.8, "dz")},
        "descriptives": _descriptives(groups),
        "posthoc": [],
    }


def wilcoxon_signed_rank(a, b, labels):
    res = stats.wilcoxon(a, b)
    diff = a - b
    nz = diff[diff != 0]
    ranks = stats.rankdata(np.abs(nz))
    t_pos = float(ranks[nz > 0].sum())
    t_neg = float(ranks[nz < 0].sum())
    total = t_pos + t_neg
    r = (t_pos - t_neg) / total if total > 0 else 0.0
    groups = {labels[0]: a, labels[1]: b}
    return {
        "name": "Wilcoxon signed-rank test",
        "statistic_label": "W",
        "statistic": float(res.statistic),
        "df": None,
        "p": float(res.pvalue),
        "ci": None,
        "effect": {"name": "matched rank-biserial r", "value": float(r), "ci": None,
                   "interpretation": _interpret(r, 0.1, 0.3, 0.5, "r")},
        "descriptives": _descriptives(groups),
        "posthoc": [],
    }


# ------------------------------------------------------------- multi group

def _eta_squared(groups):
    allv = np.concatenate(list(groups.values()))
    grand = np.mean(allv)
    ss_between = sum(len(v) * (np.mean(v) - grand) ** 2 for v in groups.values())
    ss_total = float(np.sum((allv - grand) ** 2))
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def one_way_anova(groups):
    arrays = list(groups.values())
    res = stats.f_oneway(*arrays)
    eta = _eta_squared(groups)
    k = len(arrays)
    n = sum(len(v) for v in arrays)
    return {
        "name": "One-way ANOVA",
        "statistic_label": "F",
        "statistic": float(res.statistic),
        "df": (k - 1, n - k),
        "p": float(res.pvalue),
        "ci": None,
        "effect": {"name": "eta squared", "value": eta, "ci": None,
                   "interpretation": _interpret(eta, 0.01, 0.06, 0.14, "eta squared")},
        "descriptives": _descriptives(groups),
        "posthoc": tukey_posthoc(groups),
    }


def welch_anova(groups):
    means = {k: np.mean(v) for k, v in groups.items()}
    variances = {k: np.var(v, ddof=1) for k, v in groups.items()}
    ns = {k: len(v) for k, v in groups.items()}
    k = len(groups)
    w = {g: ns[g] / variances[g] for g in groups}
    w_sum = sum(w.values())
    grand = sum(w[g] * means[g] for g in groups) / w_sum
    a = sum(w[g] * (means[g] - grand) ** 2 for g in groups) / (k - 1)
    tail = sum((1 - w[g] / w_sum) ** 2 / (ns[g] - 1) for g in groups)
    b = 1 + (2.0 * (k - 2) / (k * k - 1.0)) * tail
    f = a / b
    df1 = k - 1
    df2 = (k * k - 1.0) / (3.0 * tail)
    p = float(stats.f.sf(f, df1, df2))
    eta = _eta_squared(groups)
    return {
        "name": "Welch's ANOVA",
        "statistic_label": "F",
        "statistic": float(f),
        "df": (df1, round(df2, 1)),
        "p": p,
        "ci": None,
        "effect": {"name": "eta squared", "value": eta, "ci": None,
                   "interpretation": _interpret(eta, 0.01, 0.06, 0.14, "eta squared")},
        "descriptives": _descriptives(groups),
        "posthoc": games_howell_posthoc(groups),
    }


def kruskal_wallis(groups):
    arrays = list(groups.values())
    res = stats.kruskal(*arrays)
    k = len(arrays)
    n = sum(len(v) for v in arrays)
    eps = float((res.statistic - k + 1) / (n - k)) if n > k else 0.0
    eps = max(0.0, eps)
    return {
        "name": "Kruskal-Wallis H test",
        "statistic_label": "H",
        "statistic": float(res.statistic),
        "df": k - 1,
        "p": float(res.pvalue),
        "ci": None,
        "effect": {"name": "epsilon squared", "value": eps, "ci": None,
                   "interpretation": _interpret(eps, 0.01, 0.06, 0.14, "epsilon squared")},
        "descriptives": _descriptives(groups),
        "posthoc": dunn_posthoc(groups),
    }


# ----------------------------------------------------------------- post-hoc

def tukey_posthoc(groups):
    names = list(groups.keys())
    res = stats.tukey_hsd(*[groups[g] for g in names])
    out = []
    for i, j in itertools.combinations(range(len(names)), 2):
        out.append({
            "pair": names[i] + " vs " + names[j],
            "p": float(res.pvalue[i, j]),
            "method": "Tukey HSD",
        })
    return out


def games_howell_posthoc(groups):
    names = list(groups.keys())
    k = len(names)
    out = []
    for i, j in itertools.combinations(range(k), 2):
        a, b = groups[names[i]], groups[names[j]]
        na, nb = len(a), len(b)
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        sa, sb = va / na, vb / nb
        se = np.sqrt(sa + sb)
        if se == 0:
            out.append({"pair": names[i] + " vs " + names[j], "p": 1.0, "method": "Games-Howell"})
            continue
        t = (np.mean(a) - np.mean(b)) / se
        df = (sa + sb) ** 2 / (sa * sa / (na - 1) + sb * sb / (nb - 1))
        p = float(stats.studentized_range.sf(abs(t) * np.sqrt(2.0), k, df))
        out.append({
            "pair": names[i] + " vs " + names[j],
            "p": min(1.0, p),
            "method": "Games-Howell",
        })
    return out


def dunn_posthoc(groups):
    names = list(groups.keys())
    arrays = [groups[g] for g in names]
    allv = np.concatenate(arrays)
    n = len(allv)
    ranks = stats.rankdata(allv)
    mean_ranks = []
    idx = 0
    for v in arrays:
        mean_ranks.append(np.mean(ranks[idx:idx + len(v)]))
        idx += len(v)
    _, counts = np.unique(allv, return_counts=True)
    tie_term = float(np.sum(counts ** 3 - counts)) / (12.0 * (n - 1))
    base_var = n * (n + 1) / 12.0 - tie_term
    raw = []
    for i, j in itertools.combinations(range(len(names)), 2):
        se = np.sqrt(base_var * (1.0 / len(arrays[i]) + 1.0 / len(arrays[j])))
        z = (mean_ranks[i] - mean_ranks[j]) / se if se > 0 else 0.0
        p = 2.0 * stats.norm.sf(abs(z))
        raw.append((names[i] + " vs " + names[j], p))
    adjusted = holm_adjust([p for _, p in raw])
    return [{"pair": pair, "p": float(p), "method": "Dunn (Holm adjusted)"}
            for (pair, _), p in zip(raw, adjusted)]


def holm_adjust(pvalues):
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * pvalues[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


# -------------------------------------------------------------- categorical

def cramers_v(chi2, table):
    n = table.to_numpy().sum()
    r, c = table.shape
    denom = n * (min(r, c) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def chi_square(table):
    observed = table.to_numpy(dtype=float)
    correction = observed.shape == (2, 2)
    chi2, p, dof, _ = stats.chi2_contingency(observed, correction=correction)
    v = cramers_v(chi2, table)
    name = "Chi-square test of independence"
    if correction:
        name += " (Yates corrected)"
    return {
        "name": name,
        "statistic_label": "chi-square",
        "statistic": float(chi2),
        "df": int(dof),
        "p": float(p),
        "ci": None,
        "effect": {"name": "Cramer's V", "value": v, "ci": None,
                   "interpretation": _interpret(v, 0.1, 0.3, 0.5, "V")},
        "descriptives": _table_rows(table),
        "posthoc": [],
    }


def fisher_exact(table):
    observed = table.to_numpy(dtype=float)
    odds, p = stats.fisher_exact(observed)
    chi2, _, _, _ = stats.chi2_contingency(observed, correction=False)
    v = cramers_v(chi2, table)
    return {
        "name": "Fisher's exact test",
        "statistic_label": "odds ratio",
        "statistic": float(odds),
        "df": None,
        "p": float(p),
        "ci": None,
        "effect": {"name": "Cramer's V", "value": v, "ci": None,
                   "interpretation": _interpret(v, 0.1, 0.3, 0.5, "V")},
        "descriptives": _table_rows(table),
        "posthoc": [],
    }


def _table_rows(table):
    rows = []
    for outcome_level, row in table.iterrows():
        for group_level, count in row.items():
            rows.append({
                "group": str(group_level) + " / " + str(outcome_level),
                "n": int(count),
                "mean": None, "sd": None, "median": None,
            })
    return rows


# -------------------------------------------------------------- correlation

def pearson(xv, yv, labels):
    res = stats.pearsonr(xv, yv)
    ci = res.confidence_interval(0.95)
    r = float(res.statistic)
    return {
        "name": "Pearson correlation",
        "statistic_label": "r",
        "statistic": r,
        "df": len(xv) - 2,
        "p": float(res.pvalue),
        "ci": {"label": "95% CI of r", "low": float(ci.low), "high": float(ci.high)},
        "effect": {"name": "r", "value": r, "ci": (float(ci.low), float(ci.high)),
                   "interpretation": _interpret(r, 0.1, 0.3, 0.5, "r")},
        "descriptives": _xy_descriptives(xv, yv, labels),
        "posthoc": [],
    }


def spearman(xv, yv, labels):
    res = stats.spearmanr(xv, yv)
    rho = float(res.statistic)
    n = len(xv)
    ci = None
    if n > 4 and abs(rho) < 1:
        z = np.arctanh(rho)
        se = 1.06 / np.sqrt(n - 3)
        ci = {"label": "95% CI of rho", "low": float(np.tanh(z - 1.96 * se)),
              "high": float(np.tanh(z + 1.96 * se))}
    return {
        "name": "Spearman rank correlation",
        "statistic_label": "rho",
        "statistic": rho,
        "df": n - 2,
        "p": float(res.pvalue),
        "ci": ci,
        "effect": {"name": "rho", "value": rho, "ci": None,
                   "interpretation": _interpret(rho, 0.1, 0.3, 0.5, "rho")},
        "descriptives": _xy_descriptives(xv, yv, labels),
        "posthoc": [],
    }


def _xy_descriptives(xv, yv, labels):
    return [
        {"group": labels[0], "n": len(xv), "mean": float(np.mean(xv)),
         "sd": float(np.std(xv, ddof=1)), "median": float(np.median(xv))},
        {"group": labels[1], "n": len(yv), "mean": float(np.mean(yv)),
         "sd": float(np.std(yv, ddof=1)), "median": float(np.median(yv))},
    ]
