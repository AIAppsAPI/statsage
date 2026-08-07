"""Methods and results text.

Deterministic templates always work. When an LLM provider is available
the templates are polished into journal prose, with the hard rule that
no number may change. If the LLM output drops any required number, the
template text is kept instead.
"""

import re

from . import providers
from .checks import format_num, format_p

SOFTWARE_SENTENCE = ("Statistical analysis was performed with statsage "
                     "(Python, SciPy based), with significance set at alpha = {alpha}.")


def _stat_string(test):
    label = test["statistic_label"]
    df = test["df"]
    if isinstance(df, tuple):
        df_text = "(" + ", ".join(str(d) for d in df) + ")"
    elif df is None:
        df_text = ""
    else:
        df_text = "(" + format_num(df, 4) + ")"
    return label + df_text + " = " + format_num(test["statistic"], 4)


def methods_text(kind, test, trace, alpha):
    reason = " ".join(trace[1:]) if len(trace) > 1 else ""
    parts = [SOFTWARE_SENTENCE.format(alpha=alpha)]
    parts.append("A " + test["name"] + " was used. " + reason)
    if test["posthoc"]:
        parts.append("Post-hoc pairwise comparisons used the "
                     + test["posthoc"][0]["method"] + " method.")
    return " ".join(p.strip() for p in parts if p.strip())


def results_text(kind, test):
    p = test["p"]
    sig = "a statistically significant" if p < 0.05 else "no statistically significant"
    desc = test["descriptives"]
    if kind in ("two_group", "paired_two") and desc and desc[0]["mean"] is not None:
        a, b = desc[0], desc[1]
        summary = (a["group"] + " (mean " + format_num(a["mean"]) + ", SD " + format_num(a["sd"])
                   + ", n = " + str(a["n"]) + ") and " + b["group"] + " (mean "
                   + format_num(b["mean"]) + ", SD " + format_num(b["sd"])
                   + ", n = " + str(b["n"]) + ")")
        text = ("The " + test["name"] + " showed " + sig + " difference between "
                + summary + ", " + _stat_string(test) + ", p " + _p_rel(p) + ".")
    elif kind == "multi_group":
        text = ("The " + test["name"] + " showed " + sig
                + " difference between groups, " + _stat_string(test)
                + ", p " + _p_rel(p) + ".")
        sig_pairs = [ph for ph in test["posthoc"] if ph["p"] < 0.05]
        if p < 0.05 and sig_pairs:
            pairs = "; ".join(ph["pair"] + " (p " + _p_rel(ph["p"]) + ")" for ph in sig_pairs)
            text += " Post-hoc comparisons were significant for: " + pairs + "."
        elif p < 0.05:
            text += " No individual pair reached significance in post-hoc testing."
    elif kind == "categorical":
        text = ("The " + test["name"] + " showed " + sig + " association, "
                + _stat_string(test) + ", p " + _p_rel(p) + ".")
    elif kind == "correlation":
        direction = "positive" if test["statistic"] > 0 else "negative"
        text = ("The " + test["name"] + " showed " + sig + " " + direction
                + " relationship, " + _stat_string(test) + ", p " + _p_rel(p) + ".")
    else:
        text = _stat_string(test) + ", p " + _p_rel(p) + "."
    effect = test.get("effect")
    if effect and effect.get("value") is not None:
        text += " This corresponds to " + effect["interpretation"] + "."
    return text


def _p_rel(p):
    if p < 0.001:
        return "< 0.001"
    return "= " + format_p(p)


def polish(methods, results, provider=None):
    """Ask an LLM to smooth the template text. Returns (methods, results),
    falling back to the inputs when no provider or the numbers changed."""
    name = providers.detect_provider(provider)
    if name is None:
        return methods, results, None
    prompt = (
        "Rewrite these two paragraphs for a scientific manuscript. Keep every "
        "number, statistic, and test name exactly as written, do not add new "
        "claims, do not use any character outside plain ASCII. Reply with the "
        "methods paragraph, then the line ---, then the results paragraph, "
        "nothing else.\n\nMETHODS:\n" + methods + "\n\nRESULTS:\n" + results
    )
    reply = providers.generate(prompt, provider=provider)
    if not reply or "---" not in reply:
        return methods, results, None
    new_methods, new_results = reply.split("---", 1)
    new_methods = new_methods.replace("METHODS:", "").strip()
    new_results = new_results.replace("RESULTS:", "").strip()
    if not new_methods or not new_results:
        return methods, results, None
    if not new_methods.isascii() or not new_results.isascii():
        return methods, results, None
    if _numbers_kept(methods, new_methods) and _numbers_kept(results, new_results):
        return new_methods, new_results, name
    return methods, results, None


def _numbers_kept(original, rewritten):
    wanted = set(re.findall(r"\d+\.\d+|\d+", original))
    have = set(re.findall(r"\d+\.\d+|\d+", rewritten))
    return wanted.issubset(have)
