"""statsage, the AI statistician.

Give it a table and the columns to compare. It checks assumptions, picks
the right statistical test, runs it with effect sizes, draws a
publication figure, and writes the methods text. Works fully offline,
and polishes the prose with your own LLM (claude or codex CLI, or an
API key) when one is available.

    import statsage
    result = statsage.analyze(df, outcome="od600", group="strain")
    result.save_report("report.html")
"""

from . import detect, figures, narrative, report, select
from .detect import InputError

__version__ = "0.1.0"
__all__ = ["analyze", "AnalysisResult", "InputError", "__version__"]


class AnalysisResult(object):
    """Everything one analysis produced. Render with save_report(),
    markdown(), or html(), or read the fields directly."""

    def __init__(self, kind, test, checks, trace, warnings, figure_png,
                 methods_text, results_text, alpha, llm_used):
        self.kind = kind
        self.test = test
        self.checks = checks
        self.trace = trace
        self.warnings = warnings
        self.figure_png = figure_png
        self.methods_text = methods_text
        self.results_text = results_text
        self.alpha = alpha
        self.llm_used = llm_used
        self.generated = report.timestamp()

    @property
    def p(self):
        return self.test["p"]

    @property
    def significant(self):
        return self.test["p"] < self.alpha

    @property
    def test_name(self):
        return self.test["name"]

    def markdown(self):
        return report.build_markdown(self)

    def html(self):
        return report.build_html(self)

    def save_report(self, path):
        """Write the report. Extension decides the format, .html or .md."""
        if str(path).lower().endswith((".md", ".markdown", ".txt")):
            content = self.markdown()
        else:
            content = self.html()
        with open(path, "w", encoding="ascii") as handle:
            handle.write(content)
        return path

    def save_figure(self, path):
        if not self.figure_png:
            raise ValueError("No figure was produced for this analysis.")
        with open(path, "wb") as handle:
            handle.write(self.figure_png)
        return path

    def __repr__(self):
        from .checks import format_p
        return ("AnalysisResult(" + self.test["name"] + ", p = "
                + format_p(self.test["p"]) + ")")


def analyze(df, outcome=None, group=None, paired=None, x=None, y=None,
            alpha=0.05, llm="auto", figure=True):
    """Run the full statsage pipeline on a pandas DataFrame.

    outcome and group name the columns for a comparison, add paired for
    repeated measures. Use x and y instead for correlation. Set
    llm="off" to skip the prose polish, or name a provider: claude,
    codex, anthropic, openai.
    """
    info = detect.detect(df, outcome=outcome, group=group, paired=paired, x=x, y=y)
    test, checks_out, trace, warnings = select.choose_and_run(info, alpha=alpha)

    if info.get("dropped_rows"):
        note = str(info["dropped_rows"]) + " rows with missing values were dropped."
        if note not in " ".join(warnings):
            warnings.append(note)

    figure_png = figures.make_figure(info, test) if figure else None

    methods = narrative.methods_text(info["kind"], test, trace, alpha)
    results = narrative.results_text(info["kind"], test)
    llm_used = None
    provider = None if llm == "auto" else llm
    if llm != "off":
        methods, results, llm_used = narrative.polish(methods, results, provider=provider)

    return AnalysisResult(
        kind=info["kind"], test=test, checks=checks_out, trace=trace,
        warnings=warnings, figure_png=figure_png, methods_text=methods,
        results_text=results, alpha=alpha, llm_used=llm_used,
    )
