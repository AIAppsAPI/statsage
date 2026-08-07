# statsage

The AI statistician. Give it a table and the columns to compare, it checks the assumptions, picks the correct statistical test, runs it with effect sizes and confidence intervals, draws a publication-quality figure, and writes the methods paragraph and results sentence you can paste into a manuscript.

It works completely offline with no account and no key. If you have the Claude or Codex CLI installed, or an Anthropic or OpenAI compatible API key, statsage uses your own model to polish the written narrative, and never lets it change a number.

## Why

Picking the wrong test is one of the most common statistical mistakes in published research. statsage makes the safe choice automatically and, more importantly, shows its reasoning: every report includes the decision path, like "normality failed in group B and n is under 30, so Mann-Whitney U was chosen". You learn the statistics while getting the answer.

## Install

```
pip install statsage
```

Excel input support: `pip install statsage[excel]`

## Use from Python

```python
import pandas as pd
import statsage

df = pd.read_csv("growth.csv")

result = statsage.analyze(df, outcome="od600", group="strain")
print(result.test_name, result.p, result.significant)
print(result.results_text)
result.save_report("report.html")
```

Paired designs and correlation:

```python
statsage.analyze(df, outcome="score", group="timepoint", paired="subject")
statsage.analyze(df, x="dose", y="response")
```

## Use from the command line

```
statsage growth.csv --outcome od600 --group strain
statsage growth.csv --outcome score --group timepoint --paired subject
statsage growth.csv --x dose --y response --out report.html
```

The report prints to the terminal as markdown and saves as a self-contained HTML file with the figure embedded, ready to share.

## What it covers

- Two groups: Welch's t-test (the default, it does not assume equal variances), Student's t-test, Mann-Whitney U
- Paired designs: paired t-test, Wilcoxon signed-rank
- Three or more groups: one-way ANOVA with Tukey HSD, Welch's ANOVA with Games-Howell, Kruskal-Wallis with Dunn's test (Holm adjusted)
- Categorical data: chi-square, Fisher's exact test
- Correlation: Pearson, Spearman
- Assumption checks: Shapiro-Wilk or D'Agostino normality, Levene's test for variances, sample size rules, IQR outlier flags, expected cell counts
- Effect sizes with plain-language interpretation: Hedges' g, Cohen's dz, rank-biserial r, eta squared, epsilon squared, Cramer's V

Every report shows descriptives, each assumption check with its verdict, the reasoning path that led to the chosen test, post-hoc comparisons when relevant, warnings when something deserves a second look, and copy-paste methods and results text.

## Bring your own model (optional)

statsage looks for a language model in this order and uses the first one it finds:

1. `claude` CLI (an active Claude Code login)
2. `codex` CLI
3. `ANTHROPIC_API_KEY`
4. `OPENAI_API_KEY` (set `OPENAI_BASE_URL` for any OpenAI compatible server)

Control it with `--llm off`, `--llm claude`, or the `STATSAGE_LLM` environment variable. The model only rewords the narrative text. If its rewrite drops or changes any number, statsage keeps the template text instead.

## Figures

Box plots with individual points for group comparisons, paired line plots for repeated measures, grouped bars for counts, scatter with a fit line for correlation. Colorblind-safe Okabe-Ito palette, 300 dpi, exported inside the HTML report or separately with `--figure out.png`.

## Author

Built by [Paul Crinigan, AI Apps API](https://www.aiappsapi.com). Full documentation lives on the [statsage page](https://www.learnhowtoscience.com/ai-science-tools/statsage.php), part of the [free AI science tools collection](https://www.learnhowtoscience.com/ai-science-tools/) at learnhowtoscience.com.

MIT licensed.
