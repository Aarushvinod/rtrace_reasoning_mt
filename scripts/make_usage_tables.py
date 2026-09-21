"""
make_usage_tables.py
────────────────────
Emit the two example-utility paper tables for the active dataset arm
(RTRACE_DATASET) in the paper's table* / adjustbox layout: models down the
side, the four selection methods x k across the top.

  • usage_rate.tex — example utility: chrF recall of a sentence's k in-context
    examples within its reasoning trace, averaged over test sentences and
    target languages. The highest value across methods per (model, k) is bold.
  • usage_corr.tex — Pearson r between per-sentence example utility and
    per-sentence chrF++ (reasoning ON), pooled over all target languages
    after centering both variables within each language. * p<.05, ** p<.01.

Inputs:  trace_example_recall's sentence_recall_rates_all_methods.csv and
         chrfpp_per_sentence_analysis's regression_dataset_per_sentence.csv,
         located via the dataset registry (RTRACE_RECALL_RATES_CSV /
         RTRACE_MASTER_CSV override either path).
Outputs: <out_base>/<prefix>paper_tables/{usage_rate,usage_corr}.tex, also
         printed to stdout.

    RTRACE_DATASET=wmt24pp .venv-sent/bin/python scripts/make_usage_tables.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common.dataset_registry import get_dataset

DS = get_dataset()
RECALL_CSV = os.environ.get(
    "RTRACE_RECALL_RATES_CSV",
    os.path.join(DS.analysis_dir("trace_example_recall"), "sentence_recall_rates_all_methods.csv"),
)
MASTER_CSV = os.environ.get(
    "RTRACE_MASTER_CSV",
    os.path.join(DS.analysis_dir("chrfpp_per_sentence_analysis"), "regression_dataset_per_sentence.csv"),
)
OUT_DIR = DS.analysis_dir("paper_tables")

MODEL_GROUPS = [
    ["Ministral 8B", "Ministral 14B", "Magistral 24B"],
    ["Qwen3 8B", "Qwen3 14B", "Qwen3 32B"],
]
# (method_label in the analysis CSVs, column header in the paper)
METHODS = [
    ("RRF", "RRF"),
    ("Edit Distance", "Edit Distance"),
    ("Random", "Random"),
    ("Sentinel", "Translatability"),
]
K_LIST = [k for k in DS.k_list if k > 0]
N_LANGS_WORD = {5: "five", 7: "seven"}.get(len(DS.tgt_langs), str(len(DS.tgt_langs)))
LABEL_SUFFIX = "" if DS.key == "flores" else f"_{DS.key}"

RATE_CAPTION = (
    "Example utility scores measuring how many and how much of the provided k examples were "
    "used in the model's reasoning stage. Scores are averaged across the "
    f"{N_LANGS_WORD} target languages for each model and presented by the ensemble method "
    "and $k$ value. We compute this metric by measuring the recall component of chrF++ for "
    "each of a sentence's $k$ examples within that sentence's reasoning trace. To score a "
    "test sentence, we average the recall scores of all $k$ examples. For each model and "
    "$k$, the highest percentage across methods is bolded."
)
CORR_CAPTION = (
    "Pearson correlation between example utility and translation quality for each model, "
    "ensemble method, and $k$ value. For each test sentence, example utility is the average "
    "chrF++ recall of its $k$ examples within its reasoning trace, and translation quality is "
    "its sentence-level chrF++ score. Correlations pool the test sentences of all "
    f"{N_LANGS_WORD} target languages after centering both variables within each language, "
    "so they measure the association within a language rather than differences between "
    "languages. $^{*}p<0.05$, $^{**}p<0.01$."
)


def _header() -> list:
    nk = len(K_LIST)
    col_spec = "l|" + "|".join(["c" * nk] * len(METHODS))
    lines = [
        r"% Requires: \usepackage[table]{xcolor}",
        r"%           \usepackage{booktabs}",
        r"%           \usepackage{multirow}",
        r"%           \usepackage{adjustbox}",
        r"%           \usepackage{caption}",
        r"%           \usepackage{array}",
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\renewcommand{\arraystretch}{1.7}",
        r"\captionsetup{skip=9pt}",
        r"\begin{adjustbox}{width=\textwidth}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]
    groups = []
    for i, (_, head) in enumerate(METHODS):
        align = "c|" if i < len(METHODS) - 1 else "c"
        groups.append(r"\multicolumn{%d}{%s}{\textbf{%s}}" % (nk, align, head))
    lines.append(r"\multirow{2}{*}{\textbf{Model}} & " + " & ".join(groups) + r" \\")
    lines.append("".join(
        r"\cmidrule(lr){%d-%d}" % (2 + i * nk, 1 + (i + 1) * nk) for i in range(len(METHODS))
    ))
    ks = " & ".join(r"$k{=}%d$" % k for k in K_LIST)
    lines.append(" & " + " & ".join([ks] * len(METHODS)) + r" \\")
    lines.append(r"\midrule")
    return lines


def _body(cell) -> list:
    lines = []
    for gi, group in enumerate(MODEL_GROUPS):
        if gi:
            lines.append(r"\midrule")
        for md in group:
            cells = [cell(md, ml, k) for ml, _ in METHODS for k in K_LIST]
            lines.append(r"\textbf{%s} & " % md + " & ".join(cells) + r" \\")
    return lines


def _footer(caption: str, label: str) -> list:
    return [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\end{table*}",
    ]


def rate_table() -> str:
    rr = pd.read_csv(RECALL_CSV)
    val = rr.groupby(["model_display", "method_label", "k"])["mean_recall"].mean()

    def get(md, ml, k):
        return val.get((md, ml, k), np.nan)

    def cell(md, ml, k):
        x = get(md, ml, k)
        if pd.isna(x):
            return "--"
        s = f"{x:.1f}"
        best = max((get(md, m2, k) for m2, _ in METHODS if not pd.isna(get(md, m2, k))))
        return r"\textbf{%s}" % s if f"{best:.1f}" == s else s

    return "\n".join(_header() + _body(cell) + _footer(RATE_CAPTION, f"tab:usage_rate{LABEL_SUFFIX}"))


def _within_language_r(sub: pd.DataFrame):
    sub = sub.dropna(subset=["mean_recall", "chrfpp_sentence"])
    n, n_langs = len(sub), sub["tgt_lang"].nunique()
    dof = n - n_langs - 1
    if dof < 3:
        return np.nan, np.nan
    g = sub.groupby("tgt_lang")
    x = sub["mean_recall"] - g["mean_recall"].transform("mean")
    y = sub["chrfpp_sentence"] - g["chrfpp_sentence"].transform("mean")
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom == 0.0:
        return np.nan, np.nan
    r = float((x * y).sum()) / denom
    t = r * np.sqrt(dof / max(1e-12, 1.0 - r * r))
    return r, float(2.0 * stats.t.sf(abs(t), dof))


def corr_table() -> str:
    m = pd.read_csv(
        MASTER_CSV,
        usecols=["model_display", "reasoning_state", "method_label", "tgt_lang", "k",
                 "chrfpp_sentence", "mean_recall"],
    )
    m = m[m["reasoning_state"] == "On"]
    res = {key: _within_language_r(sub)
           for key, sub in m.groupby(["model_display", "method_label", "k"])}

    def cell(md, ml, k):
        r, p = res.get((md, ml, k), (np.nan, np.nan))
        if pd.isna(r):
            return "--"
        s = f"{r:.2f}"
        if s == "-0.00":
            s = "0.00"
        stars = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        return f"${s}^{{{stars}}}$" if stars else f"${s}$"

    return "\n".join(_header() + _body(cell) + _footer(CORR_CAPTION, f"tab:usage_corr{LABEL_SUFFIX}"))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, build in (("usage_rate", rate_table), ("usage_corr", corr_table)):
        try:
            tex = build()
        except FileNotFoundError as e:
            print(f"% !! skipped {name}: {e}\n")
            continue
        path = os.path.join(OUT_DIR, f"{name}.tex")
        with open(path, "w", encoding="utf-8") as f:
            f.write(tex + "\n")
        print(f"% ===== {path} =====")
        print(tex)
        print()


if __name__ == "__main__":
    main()
