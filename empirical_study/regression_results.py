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


def save(df, name, caption, label):
    os.makedirs(TABLE_DIR, exist_ok=True)
    base = os.path.join(TABLE_DIR, name)
    df.to_csv(base + ".csv", index=False)
    open(base + ".md", "w").write(to_md(df))
    open(base + ".tex", "w").write(to_tex(df, caption, label))


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "per_run_results.csv"))
    has_mc = "mc_nll" in df.columns
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
#  Dataset  N runs          RMSE       MC RMSE Total runtime (s) MC runtime (s)
#  Airfoil      10 1.561 ± 0.144 1.562 ± 0.140             68.03           2.85
#   Energy      10 0.575 ± 0.090 0.576 ± 0.090            125.27           4.84
#    Yacht      10 0.427 ± 0.104 0.428 ± 0.105             63.79           2.07
# Concrete      10 5.087 ± 0.590 5.088 ± 0.594            111.20           4.25

# ====================================================================================================
# UNCERTAINTY QUANTIFICATION - NLL AND 95% COVERAGE (ANALYTIC vs. MC) - SUMMARY ACROSS DATASETS
# ====================================================================================================
#  Dataset  N runs  Analytic NLL        MC NLL Analytic Cov. (%) MC Cov. (%)
#  Airfoil      10 1.897 ± 0.122 1.836 ± 0.100        92.1 ± 1.5  94.7 ± 1.2
#   Energy      10 0.901 ± 0.203 0.837 ± 0.107        91.3 ± 3.8  95.8 ± 2.2
#    Yacht      10 0.597 ± 0.236 0.546 ± 0.074        95.8 ± 3.2  99.0 ± 1.5
# Concrete      10 3.103 ± 0.185 3.047 ± 0.123        90.6 ± 2.6  94.0 ± 3.0
# ====================================================================================================