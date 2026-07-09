"""
plots.py
────────
Plot helpers that appear byte-identically in more than one analysis script.
Only functions whose bodies match exactly across all cells they appear in are
kept here; every other plotting helper (save_metric_plot, save_delta_superplot,
etc.) diverged in whitespace / docstrings / minor tweaks between eval_pipeline
and chrfpp_per_sentence_analysis, so those live in each script's own file.

Inputs:  Matplotlib axes and seaborn-produced legend handles.
Outputs: filtered legend handle/label lists suitable for a shared figure legend.
"""


def _collect_legend(ax, eh=None, el=None, ek=None):
    if ek is None: ek={"run_display","method_label","line_label","reasoning_state"}
    h,l = ax.get_legend_handles_labels()
    if h and eh is None:
        f=[(hh,ll) for hh,ll in zip(h,l) if ll not in ek]
        if f: return [x[0] for x in f],[x[1] for x in f]
    return eh, el
