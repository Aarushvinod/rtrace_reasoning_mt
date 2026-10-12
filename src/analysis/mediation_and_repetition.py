"""
mediation_and_repetition.py
───────────────────────────
Two follow-up analyses on the per-sentence outputs of
chrfpp_per_sentence_analysis.py, for the active dataset arm (RTRACE_DATASET):

  (1) Mediation LMMs, fit per model family, once per target language and once
      with all languages pooled (tgt_lang = "ALL"), on reasoning-ON sentences
      with k > 0:

        M1   chrF++  ~ method * k + model                                    total effect c
        M2a  utility ~ method * k + model                                    method → utility (a)
        M2b  chrF++  ~ utility_within + utility_mean_c + method * k + model  b, direct effect c'
        M3   chrF++  ~ tok_within + tok_mean_c + method * k + model
        random: (1 | sentence) per language; (1 | sentence) + (1 | sentence:language) pooled

      reporting, per method contrast against Random, the indirect effect a·b
      (Monte Carlo 95% CI, Holm-adjusted p across all per-language tests), the
      direct effect c', the total effect c and a·b / c.

  (2) n-gram repetition (rep-2 … rep-5) of every final translation and of its
      reference, as a baseline for natural repetition.

This is the repo version of the final tile of the analysis notebook; the two
should stay in sync.

Inputs:  regression_dataset_per_sentence.csv and per_sentence_chrfpp.csv from
         the chrF++ per-sentence analysis directory (RTRACE_CHRFPP_DIR, default
         from the dataset registry); the translation JSONLs they point to; the
         dataset's reference sentences via the registry loader.
Outputs: mediation_lmm_{coefs,summary}_{family}.csv and
         translation_repetition_{summary,per_sentence}.csv in the same
         directory, plus printed summary tables.

    RTRACE_DATASET=wmt24pp .venv-sent/bin/python -m src.analysis.mediation_and_repetition
"""

import json
import os
import sys
import traceback
import unicodedata
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.common.dataset_registry import filter_models, get_dataset

DS = get_dataset()
ROOT = os.environ.get("RTRACE_CHRFPP_DIR", DS.analysis_dir("chrfpp_per_sentence_analysis"))

MODEL_DISPLAY = {
    "ministral_8b": "Ministral 8B",
    "ministral_14b": "Ministral 14B",
    "magistral_small": "Magistral 24B",
    "qwen3_8b": "Qwen3 8B",
    "qwen3_14b": "Qwen3 14B",
    "qwen3_32b": "Qwen3 32B",
}
_SELECTED = filter_models(MODEL_DISPLAY)
FAMILIES = {
    fam: [m for m in models if m in _SELECTED]
    for fam, models in {
        "Mistral": ["ministral_8b", "ministral_14b", "magistral_small"],
        "Qwen": ["qwen3_8b", "qwen3_14b", "qwen3_32b"],
    }.items()
}
FAMILIES = {fam: models for fam, models in FAMILIES.items() if models}
DISPLAY_ORDER = [MODEL_DISPLAY[m] for m in MODEL_DISPLAY if m in _SELECTED]
pd.set_option("display.width", 220)


# ─────────────────────────────────────────────────────────────────────────────
# (1) Mediation LMMs: does a method win because its examples get used?
# ─────────────────────────────────────────────────────────────────────────────
#
# Method is treatment-coded against Random; k and model are sum-coded. So method
# effects are averaged over k, and each intercept is the Random-selection level
# averaged over k and the family's models. *_mean is a variable's average over
# the method × k runs of one (model, sentence, language) cell and *_within the
# deviation from it; *_mean_c is centred on the family-wide average, so the
# M2b/M3 intercepts are comparable across languages. a·b / c is only reported
# where c is significant, since it is unstable as c approaches zero.

METHOD = "C(method_label, Treatment(reference='Random'))"
FIXED = f"{METHOD} * C(k, Sum) + C(model_key, Sum)"
FORMULAS = {
    "M1": f"chrfpp_sentence ~ {FIXED}",
    "M2a": f"mean_recall ~ {FIXED}",
    "M2b": f"chrfpp_sentence ~ utility_within + utility_mean_c + {FIXED}",
    "M3": f"chrfpp_sentence ~ tok_within + tok_mean_c + {FIXED}",
}
CONTRASTS = ["RRF", "Edit Distance", "Sentinel"]
MC_DRAWS, MC_SEED = 100_000, 12345


def prepare_family(master: pd.DataFrame, models: list) -> pd.DataFrame:
    """
    Purpose: Build one family's reasoning-ON row set with the within/between
             splits of utility and log reasoning tokens.
    Inputs: Master per-sentence DataFrame, the family's model keys.
    Outputs: DataFrame of rows usable by all four models.
    """
    need = ["chrfpp_sentence", "mean_recall", "reasoning_tokens", "method_label",
            "k", "model_key", "tgt_lang", "sentence_idx"]
    d = master[(master["reasoning_state"] == "On") & master["model_key"].isin(models)]
    d = d.dropna(subset=need).copy()
    d["k"] = d["k"].astype(int)
    d = d[d["k"] > 0].copy()  # k = 0 has no examples, so utility is undefined
    d["sentence_idx"] = d["sentence_idx"].astype(int)
    cell = ["model_key", "sentence_idx", "tgt_lang"]
    d["utility_mean"] = d.groupby(cell)["mean_recall"].transform("mean")
    d["utility_within"] = d["mean_recall"] - d["utility_mean"]
    d["log_tok"] = np.log1p(d["reasoning_tokens"].astype(float))
    d["tok_mean"] = d.groupby(cell)["log_tok"].transform("mean")
    d["tok_within"] = d["log_tok"] - d["tok_mean"]
    d["utility_mean_c"] = d["utility_mean"] - d["utility_mean"].mean()
    d["tok_mean_c"] = d["tok_mean"] - d["tok_mean"].mean()
    return d


def fit(formula: str, d: pd.DataFrame, pooled: bool):
    """
    Purpose: Fit one mixed model, rejecting degenerate solutions.
    Inputs: Patsy formula, row set, whether languages are pooled (adds the
            sentence × language variance component).
    Outputs: MixedLMResults, or None if no optimizer gives a usable fit.
    """
    # lbfgs alone can stop at a degenerate solution — random-intercept variance
    # 0, intercept unidentified (standard error ~1e7) — while still reporting
    # convergence. Use statsmodels' default optimizer sequence, fall back to
    # derivative-free ones, and reject any fit whose intercept is unidentified.
    vc = {"vc_formula": {"sentence_language": "0 + C(tgt_lang)"}} if pooled else {}
    model = smf.mixedlm(formula, d, groups=d["sentence_idx"], re_formula="1", **vc)
    for method in (None, "powell", "nm"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                res = model.fit(method=method, reml=True, disp=False)
        except Exception as e:
            print(f"    ! fit failed ({method or 'default'}): {e}")
            continue
        se = float(res.bse_fe.get("Intercept", np.nan))
        if np.isfinite(se) and se < 100 and np.isfinite(res.fe_params).all():
            return res
        print(f"    ! degenerate fit with {method or 'default'} optimizer — retrying")
    print(f"    ! no usable fit for: {formula}")
    return None


def coef_rows(res, family: str, lang: str, name: str, d: pd.DataFrame) -> list:
    """
    Purpose: Flatten one fit into long-form coefficient rows.
    Inputs: Fitted result, family, language ("ALL" when pooled), model name, rows.
    Outputs: List of row dicts, one per fixed-effect coefficient.
    """
    conf = res.conf_int().loc[res.fe_params.index]
    vcomp = np.asarray(getattr(res, "vcomp", []), dtype=float).reshape(-1)
    rows = []
    for cn in res.fe_params.index:
        coef, se = float(res.fe_params[cn]), float(res.bse_fe[cn])
        rows.append({
            "family": family, "tgt_lang": lang, "model": name, "coef_name": cn,
            "coef": coef, "std_err": se, "z": coef / se if se > 0 else np.nan,
            "p_value": float(res.pvalues[cn]),
            "ci_low": float(conf.loc[cn, 0]), "ci_high": float(conf.loc[cn, 1]),
            "n_obs": int(res.nobs), "n_sentences": int(d["sentence_idx"].nunique()),
            "var_sentence": float(np.asarray(res.cov_re).reshape(-1)[0]),
            "var_sentence_language": float(vcomp[0]) if vcomp.size else np.nan,
            "resid_var": float(res.scale),
        })
    return rows


def mediation_rows(fits: dict, family: str, lang: str, n_obs: int, rng) -> list:
    """
    Purpose: Combine M1, M2a and M2b into per-method indirect, direct and total
             effects against Random.
    Inputs: Fits keyed by model name, family, language, row count, RNG for the
            Monte Carlo interval.
    Outputs: List of row dicts, one per method contrast.
    """
    m1, m2a, m2b = fits["M1"], fits["M2a"], fits["M2b"]
    b = float(m2b.fe_params["utility_within"])
    b_se = float(m2b.bse_fe["utility_within"])
    b_p = float(m2b.pvalues["utility_within"])
    rows = []
    for method in CONTRASTS:
        t = f"{METHOD}[T.{method}]"
        if t not in m1.fe_params.index:
            continue
        a, a_se = float(m2a.fe_params[t]), float(m2a.bse_fe[t])
        c, c_se, c_p = float(m1.fe_params[t]), float(m1.bse_fe[t]), float(m1.pvalues[t])
        cp, cp_se = float(m2b.fe_params[t]), float(m2b.bse_fe[t])
        draws = rng.normal(a, a_se, MC_DRAWS) * rng.normal(b, b_se, MC_DRAWS)
        ab = a * b
        rows.append({
            "family": family, "tgt_lang": lang, "method": method,
            "a": a, "a_se": a_se, "a_p": float(m2a.pvalues[t]),
            "b": b, "b_se": b_se, "b_p": b_p,
            "indirect_ab": ab,
            "indirect_ci_low": float(np.percentile(draws, 2.5)),
            "indirect_ci_high": float(np.percentile(draws, 97.5)),
            "indirect_p_mc": float(2 * min((draws <= 0).mean(), (draws >= 0).mean())),
            "direct_c_prime": cp, "direct_c_prime_se": cp_se,
            "total_c": c, "total_c_se": c_se, "total_c_p": c_p,
            "identity_gap": c - (cp + ab),
            "prop_mediated": ab / c if c_p < 0.05 else np.nan,
            "n_obs": n_obs,
        })
    return rows


def _with_p(coef: float, p: float) -> str:
    stars = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    return f"{coef:+.3f}{stars}"


def print_mediation(coefs: pd.DataFrame, summ: pd.DataFrame, lang_order: list) -> None:
    """Print the mediation table, intercepts and slopes per family."""
    for family in FAMILIES:
        s = summ[summ["family"] == family]
        c = coefs[coefs["family"] == family]
        if s.empty:
            continue
        print(f"\n=== {family}: method advantage over Random (averaged over k) "
              "and the share running through example utility ===")
        print(pd.DataFrame({
            "lang": s["tgt_lang"], "method": s["method"],
            "a": s["a"].round(2), "b": s["b"].round(4),
            "a·b": s["indirect_ab"].round(3),
            "a·b 95% CI": [f"[{lo:+.3f}, {hi:+.3f}]"
                           for lo, hi in zip(s["indirect_ci_low"], s["indirect_ci_high"])],
            "a·b p (Holm)": s["indirect_p_holm"].map(lambda p: "—" if pd.isna(p) else f"{p:.2g}"),
            "direct c′": s["direct_c_prime"].round(3),
            "total c": s["total_c"].round(3),
            "c p": s["total_c_p"].map(lambda p: f"{p:.2g}"),
            "a·b / c": s["prop_mediated"].map(lambda v: "—" if pd.isna(v) else f"{100 * v:.0f}%"),
        }).to_string(index=False))

        icpt = (c[c["coef_name"] == "Intercept"]
                .pivot(index="tgt_lang", columns="model", values="coef")
                .reindex(index=[l for l in lang_order if l in set(c["tgt_lang"])],
                         columns=list(FORMULAS)))
        print(f"\n--- {family}: intercepts = Random selection, averaged over k and models "
              "(M1/M2b/M3 in chrF++, M2a in utility; M2b/M3 at family-average utility / "
              "reasoning length) ---")
        print(icpt.round(2).to_string())

        slopes = c[c["coef_name"].isin(["utility_within", "tok_within", "tok_mean_c"])].copy()
        slopes["value"] = [_with_p(v, p) for v, p in zip(slopes["coef"], slopes["p_value"])]
        slopes["term"] = slopes["model"] + " " + slopes["coef_name"]
        print(f"\n--- {family}: slopes (* p<.05, ** p<.01) — M2b utility_within = b; "
              "M3 tok_within = longer-than-usual reasoning; tok_mean_c = sentence difficulty ---")
        print(slopes.pivot(index="tgt_lang", columns="term", values="value")
              .reindex(index=[l for l in lang_order if l in set(slopes["tgt_lang"])])
              .to_string())


def run_mediation() -> None:
    """Fit every family × language (+ pooled) and write the mediation CSVs."""
    master = pd.read_csv(os.path.join(ROOT, "regression_dataset_per_sentence.csv"))
    rng = np.random.default_rng(MC_SEED)
    coefs, summ, lang_order = [], [], []
    for family, models in FAMILIES.items():
        d_all = prepare_family(master, models)
        if d_all.empty:
            print(f"[{family}] no usable rows — skipped")
            continue
        langs = sorted(d_all["tgt_lang"].unique()) + ["ALL"]
        lang_order = lang_order or langs
        for lang in langs:
            d = d_all if lang == "ALL" else d_all[d_all["tgt_lang"] == lang]
            fits = {}
            for name, formula in FORMULAS.items():
                res = fit(formula, d, pooled=(lang == "ALL"))
                if res is not None:
                    fits[name] = res
                    coefs += coef_rows(res, family, lang, name, d)
            if all(m in fits for m in ("M1", "M2a", "M2b")):
                summ += mediation_rows(fits, family, lang, len(d), rng)
            print(f"[{family}] {lang}: {len(d):,} rows, {len(fits)}/4 models fitted")
    if not summ:
        print("No mediation results.")
        return
    coefs, summ = pd.DataFrame(coefs), pd.DataFrame(summ)
    per_lang = summ["tgt_lang"] != "ALL"
    summ["indirect_p_holm"] = np.nan
    summ.loc[per_lang, "indirect_p_holm"] = multipletests(
        summ.loc[per_lang, "indirect_p_mc"], method="holm")[1]
    for family in FAMILIES:
        coefs[coefs["family"] == family].to_csv(
            os.path.join(ROOT, f"mediation_lmm_coefs_{family.lower()}.csv"), index=False)
        summ[summ["family"] == family].to_csv(
            os.path.join(ROOT, f"mediation_lmm_summary_{family.lower()}.csv"), index=False)
    print_mediation(coefs, summ, lang_order)


# ─────────────────────────────────────────────────────────────────────────────
# (2) n-gram repetition of final translations
# ─────────────────────────────────────────────────────────────────────────────
#
# rep-n = 1 − unique n-grams / total n-grams over a translation's words:
# lowercased, split on whitespace, punctuation stripped from word edges (a
# regex \w tokenizer would split Indic-script words at their vowel signs). A
# translation with fewer than n words scores 0; an empty one is excluded.
REP_NS = [2, 3, 4, 5]


def words(text) -> list:
    out = []
    for tok in str(text).lower().split():
        i, j = 0, len(tok)
        while i < j and unicodedata.category(tok[i])[0] == "P":
            i += 1
        while j > i and unicodedata.category(tok[j - 1])[0] == "P":
            j -= 1
        if i < j:
            out.append(tok[i:j])
    return out


def repetition(text) -> list:
    """[n_words, rep-2 … rep-5] for one text; all NaN when the text is empty."""
    toks = words(text) if text else []
    if not toks:
        return [np.nan] * (len(REP_NS) + 1)
    out = [len(toks)]
    for n in REP_NS:
        grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        out.append(1.0 - len(set(grams)) / len(grams) if grams else 0.0)
    return out


def read_translations(path: str) -> list:
    # Same parsing as the per-sentence chrF++ stage, so sentence_idx lines up.
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["translation"] for line in f if line.strip()]


def reference_sentences(lang: str) -> list:
    return list(DS.load_sentences(lang)["devtest"])[:100]  # the evaluated test sentences


def run_repetition() -> None:
    """Score every translation and reference; write the repetition CSVs."""
    ps = pd.read_csv(os.path.join(ROOT, "per_sentence_chrfpp.csv"))
    ps = ps[ps["model_key"].isin(_SELECTED)].reset_index(drop=True)
    cols = ["n_words"] + [f"rep{n}" for n in REP_NS]
    stats = np.full((len(ps), len(cols)), np.nan)
    sidx = ps["sentence_idx"].to_numpy().astype(int)
    groups = ps.groupby("translation_path").indices
    print(f"\n[repetition] reading {len(groups):,} translation files …")
    for path, rows in groups.items():
        try:
            preds = read_translations(path)
        except Exception as e:
            print(f"  ! could not read {path}: {e}")
            continue
        for r in rows:
            if sidx[r] < len(preds):
                stats[r] = repetition(preds[sidx[r]])
    for j, col in enumerate(cols):
        ps[col] = stats[:, j]

    ref_rows = []
    for lang in sorted(ps["tgt_lang"].unique()):
        for s, text in enumerate(reference_sentences(lang)):
            ref_rows.append([lang, s] + repetition(text))
    ref_cols = [f"ref_{c}" for c in cols]
    ref = pd.DataFrame(ref_rows, columns=["tgt_lang", "sentence_idx"] + ref_cols)
    ps = ps.merge(ref, on=["tgt_lang", "sentence_idx"], how="left")
    keep = ["model_key", "model_display", "reasoning_state", "method_label", "tgt_lang",
            "k", "sentence_idx", "chrfpp_sentence"] + cols + ref_cols
    ps[keep].to_csv(os.path.join(ROOT, "translation_repetition_per_sentence.csv"), index=False)

    def summarize(df, prefix, keys):
        g = df.groupby(keys)
        out = pd.DataFrame({f"mean_rep{n}": g[f"{prefix}rep{n}"].mean() for n in REP_NS})
        for n in REP_NS:
            out[f"share_rep{n}_gt0"] = g[f"{prefix}rep{n}"].apply(lambda v: float((v > 0).mean()))
        out["n_translations"] = g.size()
        return out.reset_index()

    valid = ps.dropna(subset=["rep2"])
    refv = ref.dropna(subset=["ref_rep2"])
    ref_summary = summarize(refv, "ref_", ["tgt_lang"])
    ref_summary.insert(0, "reasoning_state", "")
    ref_summary.insert(0, "model_display", "Reference")
    summary = pd.concat([summarize(valid, "", ["model_display", "reasoning_state", "tgt_lang"]),
                         ref_summary], ignore_index=True)
    summary.to_csv(os.path.join(ROOT, "translation_repetition_summary.csv"), index=False)

    both = pd.concat([
        valid.assign(rep=100 * valid["rep4"], looped=100.0 * (valid["rep4"] > 0)),
        refv.assign(model_display="Reference", reasoning_state="",
                    rep=100 * refv["ref_rep4"], looped=100.0 * (refv["ref_rep4"] > 0)),
    ], ignore_index=True)
    order = [(m, s) for m in DISPLAY_ORDER for s in ("Off", "On")] + [("Reference", "")]
    for col, title in (("rep", "mean rep-4 = % of a translation's 4-grams that are repeats"),
                       ("looped", "% of translations with any repeated 4-gram")):
        t = both.pivot_table(index=["model_display", "reasoning_state"], columns="tgt_lang",
                             values=col, aggfunc="mean")
        t["All"] = both.groupby(["model_display", "reasoning_state"])[col].mean()
        t = t.reindex([o for o in order if o in t.index])
        print(f"\n=== {title} (rep-2, rep-3 and rep-5 are in the CSVs) ===")
        print(t.round(2).to_string())


def main() -> None:
    print(f"[mediation_and_repetition] dataset={DS.key}  dir={ROOT}")
    failed = []
    for part in (run_mediation, run_repetition):
        try:
            part()
        except Exception:
            traceback.print_exc()
            failed.append(part.__name__)
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
