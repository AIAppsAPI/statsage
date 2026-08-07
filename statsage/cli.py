"""Command line entry point.

    statsage data.csv --outcome od600 --group strain --out report.html
    statsage data.csv --x dose --y response
    statsage data.csv --outcome score --group timepoint --paired subject
"""

import argparse
import os
import sys

import pandas as pd

from . import InputError, __version__, analyze


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="statsage",
        description="Checks assumptions, picks the right statistical test, "
                    "runs it, and writes the report.",
    )
    parser.add_argument("data", help="path to a CSV, TSV, or Excel file")
    parser.add_argument("--outcome", help="outcome column for a group comparison")
    parser.add_argument("--group", help="grouping column")
    parser.add_argument("--paired", help="pairing column for repeated measures")
    parser.add_argument("--x", help="first numeric column for correlation")
    parser.add_argument("--y", help="second numeric column for correlation")
    parser.add_argument("--alpha", type=float, default=0.05, help="significance level, default 0.05")
    parser.add_argument("--llm", default="auto",
                        help="auto, off, claude, codex, anthropic, or openai")
    parser.add_argument("--out", help="report path, .html or .md, default <data>_report.html")
    parser.add_argument("--figure", help="also save the figure PNG to this path")
    parser.add_argument("--version", action="version", version="statsage " + __version__)
    args = parser.parse_args(argv)

    try:
        df = _load(args.data)
    except Exception as err:
        print("Could not read '" + args.data + "': " + str(err), file=sys.stderr)
        return 2

    try:
        result = analyze(df, outcome=args.outcome, group=args.group,
                         paired=args.paired, x=args.x, y=args.y,
                         alpha=args.alpha, llm=args.llm)
    except InputError as err:
        print("statsage: " + str(err), file=sys.stderr)
        return 2

    out = args.out or os.path.splitext(args.data)[0] + "_report.html"
    result.save_report(out)
    if args.figure:
        result.save_figure(args.figure)

    print(result.markdown())
    print("")
    print("Report saved to " + out)
    if result.llm_used:
        print("Narrative polished by " + result.llm_used + ".")
    return 0


def _load(path):
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if lower.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


if __name__ == "__main__":
    sys.exit(main())
