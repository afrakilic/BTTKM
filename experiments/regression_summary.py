"""Print the regression tables from results/per_run_results.csv (made by regression.py).

All values are mean ± std over the 10 splits (population std); run times are totals over the
splits, and Time/run is the mean ± std of the run time over the splits. Run times cover training
and the analytic prediction only (not the Monte Carlo check).
"""

import pandas as pd

from utils import ROOT

ORDER = ["airfoil", "energy", "yacht", "concrete"]
df = pd.read_csv(ROOT / "results/per_run_results.csv")
has_mc = "mc_nll" in df.columns


def ms(series, scale=1.0, dec=3):
    return f"{series.mean() * scale:.{dec}f} ± {series.std(ddof=0) * scale:.{dec}f}"


general, uq = [], []
for name in [d for d in ORDER if d in df.dataset.unique()]:
    g = df[df.dataset == name]
    gen = {"Dataset": name.capitalize(), "N runs": len(g), "RMSE": ms(g.rmse)}
    row = {"Dataset": name.capitalize(), "N runs": len(g), "Analytic NLL": ms(g.nll)}
    if has_mc:
        gen["MC RMSE"] = ms(g.mc_rmse)
        row["MC NLL"] = ms(g.mc_nll)
    row["Analytic Cov. (%)"] = ms(g.coverage, 100, 1)
    if has_mc:
        row["MC Cov. (%)"] = ms(g.mc_coverage, 100, 1)
    gen["Avg. eff. rank"] = ms(g.avg_eff_rank, dec=2)
    gen["Total runtime (s)"] = f"{g.runtime_s.sum():.2f}"
    gen["Time/run (s)"] = ms(g.runtime_s, dec=2)
    general.append(gen)
    uq.append(row)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
bar = "=" * 100
print(f"\n{bar}\nGENERAL PERFORMANCE (BTT-KM)\n{bar}")
print(pd.DataFrame(general).to_string(index=False))
print(f"\n{bar}\nUNCERTAINTY QUANTIFICATION: NLL AND 95% COVERAGE (ANALYTIC vs. MC)\n{bar}")
print(pd.DataFrame(uq).to_string(index=False))
print(f"{bar}\n\nTotal runtime across all datasets: {df.runtime_s.sum():.2f} seconds")

# ====================================================================================================
# GENERAL PERFORMANCE (BTT-KM)
# ====================================================================================================
#  Dataset  N runs          RMSE       MC RMSE Avg. eff. rank Total runtime (s) Time/run (s)
#  Airfoil      10 1.526 ± 0.161 1.527 ± 0.161    4.88 ± 0.12             68.87  6.89 ± 0.20
#   Energy      10 0.480 ± 0.054 0.480 ± 0.054    4.76 ± 0.11            106.00 10.60 ± 0.22
#    Yacht      10 0.587 ± 0.144 0.587 ± 0.144    4.30 ± 0.18             52.83  5.28 ± 0.41
# Concrete      10 5.083 ± 0.611 5.081 ± 0.611    3.90 ± 0.14             94.13  9.41 ± 0.39

# ====================================================================================================
# UNCERTAINTY QUANTIFICATION: NLL AND 95% COVERAGE (ANALYTIC vs. MC)
# ====================================================================================================
#  Dataset  N runs  Analytic NLL        MC NLL Analytic Cov. (%) MC Cov. (%)
#  Airfoil      10 1.810 ± 0.095 1.812 ± 0.093        95.2 ± 1.5  95.4 ± 1.6
#   Energy      10 0.708 ± 0.091 0.722 ± 0.081        96.1 ± 2.7  97.3 ± 1.5
#    Yacht      10 0.814 ± 0.254 0.818 ± 0.256        95.8 ± 2.9  96.1 ± 3.2
# Concrete      10 3.052 ± 0.132 3.050 ± 0.126        93.7 ± 2.5  93.8 ± 2.2
# ====================================================================================================