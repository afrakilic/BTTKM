"""
Builds the results tables from the output of experiments.py.

Usage:
    python results.py

Reads  results/per_run_results.csv
Writes results/tables/{general,uq}.{csv,md,tex}  and prints both tables.
"""

import os
import pandas as pd

RESULTS_DIR = "results"
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")
ORDER = ["airfoil", "energy", "yacht", "concrete"]


def to_md(df):
    """Markdown table without needing `tabulate`."""
    df = df.reset_index(drop=True)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def to_tex(df, caption, label=None):
    """LaTeX table without needing `jinja2`."""
    df = df.reset_index(drop=True)
    esc = lambda t: str(t).replace("±", "$\\pm$").replace("%", "\\%")
    cols = [str(c) for c in df.columns]
    out = ["\\begin{table}[ht]", "\\centering", f"\\caption{{{caption}}}"]
    if label:
        out.append(f"\\label{{{label}}}")
    out += ["\\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", "\\hline",
            " & ".join(esc(c) for c in cols) + " \\\\", "\\hline"]
    for _, r in df.iterrows():
        out.append(" & ".join(esc(v) for v in r.values) + " \\\\")
    out += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def ms(series, scale=1.0, dec=3):
    """mean ± std (population std, ddof=0, matching np.std)."""
    return f"{series.mean() * scale:.{dec}f} ± {series.std(ddof=0) * scale:.{dec}f}"

def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "per_run_results.csv"))
    has_mc = "mc_nll" in df.columns
    has_rank = "avg_eff_rank" in df.columns
    datasets = [d for d in ORDER if d in df.dataset.unique()]

    general_rows, uq_rows = [], []
    for d in datasets:
        g = df[df.dataset == d]
        gen = {"Dataset": d.capitalize(), "N runs": len(g), "RMSE": ms(g.rmse)}
        uq = {"Dataset": d.capitalize(), "N runs": len(g), "Analytic NLL": ms(g.nll)}
        if has_mc:
            gen["MC RMSE"] = ms(g.mc_rmse)
            uq["MC NLL"] = ms(g.mc_nll)
        uq["Analytic Cov. (%)"] = ms(g.coverage, 100, 1)
        if has_mc:
            uq["MC Cov. (%)"] = ms(g.mc_coverage, 100, 1)
        if has_rank:
            gen["Avg. eff. rank"] = ms(g.avg_eff_rank, dec=2)
        gen["Total runtime (s)"] = f"{g.runtime_s.sum():.2f}"
        if has_mc:
            gen["MC runtime (s)"] = f"{g.mc_runtime_s.sum():.2f}"
        general_rows.append(gen)
        uq_rows.append(uq)

    general, uq = pd.DataFrame(general_rows), pd.DataFrame(uq_rows)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    bar = "=" * 100
    print("\n" + bar)
    print("GENERAL PERFORMANCE - SUMMARY ACROSS DATASETS (BTTKM)")
    print(bar)
    print(general.to_string(index=False))
    print("\n" + bar)
    print("UNCERTAINTY QUANTIFICATION - NLL AND 95% COVERAGE (ANALYTIC vs. MC) - SUMMARY ACROSS DATASETS")
    print(bar)
    print(uq.to_string(index=False))
    print(bar)
    tot = df.runtime_s.sum() + (df.mc_runtime_s.sum() if has_mc else 0)
    print(f"\nTotal runtime across all datasets: {tot:.2f} seconds")
    print(f"Tables saved to {TABLE_DIR}/")


if __name__ == "__main__":
    main()

# ====================================================================================================
# GENERAL PERFORMANCE - SUMMARY ACROSS DATASETS (BTTKM)
# ====================================================================================================
#  Dataset  N runs          RMSE       MC RMSE Avg. eff. rank Total runtime (s) MC runtime (s)
#  Airfoil      10 1.555 ± 0.168 1.555 ± 0.168    4.60 ± 0.12             57.98           8.18
#   Energy      10 0.592 ± 0.083 0.592 ± 0.083    4.94 ± 0.10            121.10          11.70
#    Yacht      10 0.609 ± 0.132 0.608 ± 0.133    4.12 ± 0.27             49.33           3.46
# Concrete      10 5.089 ± 0.577 5.090 ± 0.578    3.91 ± 0.17             91.41           9.15

# ====================================================================================================
# UNCERTAINTY QUANTIFICATION - NLL AND 95% COVERAGE (ANALYTIC vs. MC) - SUMMARY ACROSS DATASETS
# ====================================================================================================
#  Dataset  N runs  Analytic NLL        MC NLL Analytic Cov. (%) MC Cov. (%)
#  Airfoil      10 1.889 ± 0.136 1.832 ± 0.108        92.0 ± 1.6  94.6 ± 1.2
#   Energy      10 0.908 ± 0.134 0.913 ± 0.068        94.4 ± 3.0  97.7 ± 1.8
#    Yacht      10 0.909 ± 0.251 0.865 ± 0.282        93.9 ± 1.7  96.8 ± 1.4
# Concrete      10 3.104 ± 0.176 3.054 ± 0.120        91.2 ± 2.9  94.1 ± 2.4
# ====================================================================================================
