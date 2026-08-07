"""End to end: analyze() produces ASCII-only reports with every section."""

import numpy as np
import pandas as pd

import statsage


def sample_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "od600": np.concatenate([rng.normal(0.8, 0.1, 24), rng.normal(1.1, 0.1, 24)]),
        "strain": ["wt"] * 24 + ["mutant"] * 24,
    })


def test_analyze_end_to_end(tmp_path):
    result = statsage.analyze(sample_df(), outcome="od600", group="strain", llm="off")
    assert result.test_name == "Welch's t-test"
    assert result.significant
    assert result.figure_png[:8] == b"\x89PNG\r\n\x1a\n"

    html_path = tmp_path / "report.html"
    result.save_report(str(html_path))
    html = html_path.read_text(encoding="ascii")
    for section in ["Result", "Descriptives", "Assumption checks", "Why this test",
                    "Methods paragraph", "Results sentence", "aiappsapi.com",
                    "learnhowtoscience.com", "data:image/png;base64"]:
        assert section in html


def test_markdown_is_ascii_and_complete():
    result = statsage.analyze(sample_df(), outcome="od600", group="strain", llm="off")
    md = result.markdown()
    assert md.isascii()
    for section in ["## Result", "## Descriptives", "## Assumption checks",
                    "## Why this test", "## Methods paragraph", "## Results sentence"]:
        assert section in md


def test_html_is_ascii():
    result = statsage.analyze(sample_df(), outcome="od600", group="strain", llm="off")
    assert result.html().isascii()


def test_multi_group_report_has_posthoc(tmp_path):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "value": np.concatenate([rng.normal(10, 2, 25), rng.normal(12, 2, 25),
                                 rng.normal(14, 2, 25)]),
        "grp": ["a"] * 25 + ["b"] * 25 + ["c"] * 25,
    })
    result = statsage.analyze(df, outcome="value", group="grp", llm="off")
    md = result.markdown()
    assert "Post-hoc comparisons" in md
    assert "a vs c" in md


def test_methods_text_mentions_test_and_alpha():
    result = statsage.analyze(sample_df(), outcome="od600", group="strain", llm="off")
    assert "Welch's t-test" in result.methods_text
    assert "0.05" in result.methods_text
    assert result.results_text.isascii()
    assert "Hedges" in result.results_text or "g = " in result.results_text


def test_save_figure(tmp_path):
    result = statsage.analyze(sample_df(), outcome="od600", group="strain", llm="off")
    path = tmp_path / "fig.png"
    result.save_figure(str(path))
    assert path.read_bytes()[:4] == b"\x89PNG"
