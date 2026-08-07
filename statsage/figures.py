"""Publication figures.

One entry point, make_figure, returns PNG bytes at 300 dpi using the
Okabe-Ito colorblind safe palette. Uses the Agg backend so it works on
headless servers and inside cron jobs.
"""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
             "#56B4E9", "#D55E00", "#F0E442", "#000000"]

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "figure.dpi": 100,
    "savefig.dpi": 300,
}


def make_figure(info, test_result):
    kind = info["kind"]
    with plt.rc_context(RC):
        if kind in ("two_group", "multi_group"):
            fig = _group_plot(info, test_result)
        elif kind == "paired_two":
            fig = _paired_plot(info, test_result)
        elif kind == "categorical":
            fig = _count_plot(info)
        elif kind == "correlation":
            fig = _scatter_plot(info, test_result)
        else:
            return None
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()


def _p_text(p):
    if p < 0.001:
        return "p < 0.001"
    return "p = {:.3f}".format(p)


def _jitter(rng, n, center, width=0.08):
    return center + rng.uniform(-width, width, size=n)


def _group_plot(info, test_result):
    groups = info["groups"]
    names = list(groups.keys())
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(names), 4.2))
    rng = np.random.default_rng(7)
    positions = np.arange(len(names))

    box_data = [groups[n] for n in names]
    bp = ax.boxplot(box_data, positions=positions, widths=0.5, showfliers=False,
                    patch_artist=True, medianprops={"color": "black", "linewidth": 1.4})
    for patch, color in zip(bp["boxes"], OKABE_ITO):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor("black")
    for i, name in enumerate(names):
        values = groups[name]
        ax.scatter(_jitter(rng, len(values), positions[i]), values, s=18,
                   color=OKABE_ITO[i % len(OKABE_ITO)], edgecolor="white",
                   linewidth=0.4, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    ax.set_xlabel(info["group"])
    ax.set_ylabel(info["outcome"])

    if len(names) == 2:
        top = max(np.max(v) for v in box_data)
        span = top - min(np.min(v) for v in box_data)
        y = top + 0.08 * (span if span > 0 else abs(top) or 1.0)
        h = 0.02 * (span if span > 0 else abs(top) or 1.0)
        ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], color="black", linewidth=1.0)
        ax.text(0.5, y + 1.5 * h, _p_text(test_result["p"]), ha="center", va="bottom")
    else:
        ax.set_title(test_result["name"] + ", " + _p_text(test_result["p"]), fontsize=11)
    return fig


def _paired_plot(info, test_result):
    a, b = info["pairs"]
    names = list(info["groups"].keys())
    fig, ax = plt.subplots(figsize=(3.6, 4.2))
    for pa, pb in zip(a, b):
        ax.plot([0, 1], [pa, pb], color="#999999", linewidth=0.8, alpha=0.6, zorder=1)
    rng = np.random.default_rng(7)
    ax.scatter(_jitter(rng, len(a), 0, 0.03), a, s=22, color=OKABE_ITO[0],
               edgecolor="white", linewidth=0.4, zorder=3)
    ax.scatter(_jitter(rng, len(b), 1, 0.03), b, s=22, color=OKABE_ITO[1],
               edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(names)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(info["outcome"])
    ax.set_title(test_result["name"] + ", " + _p_text(test_result["p"]), fontsize=11)
    return fig


def _count_plot(info):
    table = info["table"]
    fig, ax = plt.subplots(figsize=(1.8 + 1.0 * table.shape[1], 4.2))
    n_out = table.shape[0]
    width = 0.8 / n_out
    x = np.arange(table.shape[1])
    for i, (outcome_level, row) in enumerate(table.iterrows()):
        ax.bar(x + i * width, row.to_numpy(), width=width * 0.92,
               label=str(outcome_level), color=OKABE_ITO[i % len(OKABE_ITO)],
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x + width * (n_out - 1) / 2.0)
    ax.set_xticklabels([str(c) for c in table.columns])
    ax.set_xlabel(info["group"])
    ax.set_ylabel("count")
    ax.legend(title=info["outcome"], frameon=False)
    return fig


def _scatter_plot(info, test_result):
    xv, yv = info["xv"], info["yv"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(xv, yv, s=24, color=OKABE_ITO[0], edgecolor="white",
               linewidth=0.4, zorder=3)
    slope, intercept = np.polyfit(xv, yv, 1)
    xs = np.linspace(np.min(xv), np.max(xv), 100)
    ax.plot(xs, slope * xs + intercept, color=OKABE_ITO[1], linewidth=1.6, zorder=2)
    ax.set_xlabel(info["x"])
    ax.set_ylabel(info["y"])
    label = test_result["statistic_label"] + " = " + "{:.2f}".format(test_result["statistic"])
    ax.set_title(label + ", " + _p_text(test_result["p"]), fontsize=11)
    return fig
