"""
Diff-in-diff: did TX renewables grow faster after China's 2021 mining ban?
TX did grow faster, but it was already growing faster before the ban.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUTPUT = BASE / "output"
OUTPUT.mkdir(exist_ok=True)

YEARS = [2019, 2020, 2021, 2022, 2023]
TREATMENT_STATE = "TX"
CONTROL_STATES = ["IL", "IA", "OK", "KS", "MN"]
ALL_STATES = [TREATMENT_STATE] + CONTROL_STATES
RENEWABLE_CODES = ["SUN", "WND"]

# --- 1. Load EIA 860 ---
print("=" * 70)
print("LOADING EIA FORM 860 DATA (2019-2023)")
print("=" * 70)

frames = []
for year in YEARS:
    fpath = DATA / f"eia860_{year}" / f"3_1_Generator_Y{year}.xlsx"
    print(f"  Reading {fpath.name} ...")
    df = pd.read_excel(fpath, sheet_name="Operable", header=1)
    df["file_year"] = year
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)
print(f"  Total rows loaded: {len(raw):,}")

# Filter to our states + renewables only
mask = (raw["State"].isin(ALL_STATES)) & (raw["Energy Source 1"].isin(RENEWABLE_CODES))
ren = raw.loc[mask].copy()
print(f"  Renewable (SUN/WND) rows in study states: {len(ren):,}")

# --- 2. Capacity by state and year ---
capacity = (
    ren.groupby(["file_year", "State"])["Nameplate Capacity (MW)"]
    .sum()
    .reset_index()
    .rename(columns={"file_year": "year", "Nameplate Capacity (MW)": "capacity_mw"})
)

# Also by fuel type
capacity_by_fuel = (
    ren.groupby(["file_year", "State", "Energy Source 1"])["Nameplate Capacity (MW)"]
    .sum()
    .reset_index()
    .rename(columns={
        "file_year": "year",
        "Nameplate Capacity (MW)": "capacity_mw",
        "Energy Source 1": "fuel"
    })
)

print("\n── State-Year Solar+Wind Capacity (MW) ──")
pivot = capacity.pivot(index="State", columns="year", values="capacity_mw").fillna(0)
print(pivot.to_string(float_format="{:,.0f}".format))

# --- 3. Set up DiD panel ---
panel = capacity.copy()
panel["treated"] = (panel["State"] == TREATMENT_STATE).astype(int)
panel["post"] = (panel["year"] >= 2022).astype(int)
panel["did"] = panel["treated"] * panel["post"]
panel["log_capacity"] = np.log(panel["capacity_mw"].clip(lower=1))

# --- 4. Parallel trends plot ---
print("\n" + "=" * 70)
print("GENERATING PARALLEL TRENDS FIGURE")
print("=" * 70)

# Control group average
ctrl = capacity[capacity["State"].isin(CONTROL_STATES)]
ctrl_avg = ctrl.groupby("year")["capacity_mw"].mean().reset_index()
ctrl_avg["group"] = "Control Avg (IL, IA, OK, KS, MN)"

tx = capacity[capacity["State"] == "TX"].copy()
tx["group"] = "Texas (TX)"

# Index to 2019=100
tx_base = tx.loc[tx["year"] == 2019, "capacity_mw"].values[0]
ctrl_base = ctrl_avg.loc[ctrl_avg["year"] == 2019, "capacity_mw"].values[0]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (A) Raw capacity
ax = axes[0]
ax.plot(tx["year"], tx["capacity_mw"], "o-", color="#c0392b", linewidth=2.5,
        markersize=8, label="Texas (TX)", zorder=5)
ax.plot(ctrl_avg["year"], ctrl_avg["capacity_mw"], "s-", color="#2980b9",
        linewidth=2.5, markersize=8, label="Control Avg", zorder=5)
# Individual control states
for st in CONTROL_STATES:
    sdf = capacity[capacity["State"] == st]
    ax.plot(sdf["year"], sdf["capacity_mw"], "--", alpha=0.3, color="#2980b9", linewidth=1)
ax.axvline(x=2021, color="gray", linestyle=":", linewidth=1.5, label="China ban (Sep 2021)")
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Solar + Wind Capacity (MW)", fontsize=12)
ax.set_title("(A) Capacity Levels", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.set_xticks(YEARS)
ax.grid(axis="y", alpha=0.3)

# (B) Indexed growth
ax = axes[1]
tx_idx = tx["capacity_mw"] / tx_base * 100
ctrl_idx = ctrl_avg["capacity_mw"] / ctrl_base * 100
ax.plot(tx["year"].values, tx_idx.values, "o-", color="#c0392b", linewidth=2.5,
        markersize=8, label="Texas (TX)", zorder=5)
ax.plot(ctrl_avg["year"].values, ctrl_idx.values, "s-", color="#2980b9",
        linewidth=2.5, markersize=8, label="Control Avg", zorder=5)
ax.axvline(x=2021, color="gray", linestyle=":", linewidth=1.5, label="China ban (Sep 2021)")
ax.axhline(y=100, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Indexed Capacity (2019 = 100)", fontsize=12)
ax.set_title("(B) Growth Indexed to 2019", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.set_xticks(YEARS)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Parallel Trends: TX vs Control States — Solar+Wind Capacity",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT / "did_parallel_trends.png", dpi=200, bbox_inches="tight")
print(f"  Saved: {OUTPUT / 'did_parallel_trends.png'}")

# --- 5. Regressions ---
print("\n" + "=" * 70)
print("DIFFERENCE-IN-DIFFERENCES REGRESSION RESULTS")
print("=" * 70)

# Manual 2x2 DiD
print("\n── Manual 2x2 DiD Calculation ──")
pre_years = [2019, 2020]
post_years = [2022, 2023]

tx_pre = panel.loc[(panel["treated"] == 1) & (panel["year"].isin(pre_years)), "capacity_mw"].mean()
tx_post = panel.loc[(panel["treated"] == 1) & (panel["year"].isin(post_years)), "capacity_mw"].mean()
ctrl_pre = panel.loc[(panel["treated"] == 0) & (panel["year"].isin(pre_years)), "capacity_mw"].mean()
ctrl_post = panel.loc[(panel["treated"] == 0) & (panel["year"].isin(post_years)), "capacity_mw"].mean()

did_manual = (tx_post - tx_pre) - (ctrl_post - ctrl_pre)

print(f"  TX  pre-ban avg:     {tx_pre:>10,.1f} MW")
print(f"  TX  post-ban avg:    {tx_post:>10,.1f} MW")
print(f"  TX  change:          {tx_post - tx_pre:>10,.1f} MW")
print(f"  Ctrl pre-ban avg:    {ctrl_pre:>10,.1f} MW")
print(f"  Ctrl post-ban avg:   {ctrl_post:>10,.1f} MW")
print(f"  Ctrl change:         {ctrl_post - ctrl_pre:>10,.1f} MW")
print(f"  ─────────────────────────────────")
print(f"  DiD estimate:        {did_manual:>10,.1f} MW")

# OLS with state and year fixed effects
print("\n── OLS Regression: capacity ~ treated * post + state FE + year FE ──")

# Dummies
panel_reg = panel.copy()
panel_reg["C_state"] = pd.Categorical(panel_reg["State"])
panel_reg["C_year"] = pd.Categorical(panel_reg["year"])

# Model 1: no FE
mod1 = smf.ols("capacity_mw ~ treated * post", data=panel_reg).fit()

# Model 2: with state + year FE
mod2 = smf.ols("capacity_mw ~ did + C(State) + C(year)", data=panel_reg).fit()

# Model 3: log capacity with FE
panel_reg["log_cap"] = np.log(panel_reg["capacity_mw"])
mod3 = smf.ols("log_cap ~ did + C(State) + C(year)", data=panel_reg).fit()

print("\n  Model 1: Simple DiD (no fixed effects)")
print(f"    DiD coeff (treated:post): {mod1.params.get('treated:post', 'N/A'):>12,.1f} MW")
print(f"    Std Error:                {mod1.bse.get('treated:post', 'N/A'):>12,.1f}")
print(f"    p-value:                  {mod1.pvalues.get('treated:post', 'N/A'):>12.4f}")
print(f"    R-squared:                {mod1.rsquared:>12.4f}")
print(f"    N:                        {int(mod1.nobs):>12d}")

print("\n  Model 2: DiD with State + Year FE (levels)")
print(f"    DiD coeff:                {mod2.params['did']:>12,.1f} MW")
print(f"    Std Error:                {mod2.bse['did']:>12,.1f}")
print(f"    t-stat:                   {mod2.tvalues['did']:>12.3f}")
print(f"    p-value:                  {mod2.pvalues['did']:>12.4f}")
print(f"    R-squared:                {mod2.rsquared:>12.4f}")
print(f"    N:                        {int(mod2.nobs):>12d}")

print("\n  Model 3: DiD with State + Year FE (log capacity)")
print(f"    DiD coeff:                {mod3.params['did']:>12.4f}")
print(f"    Std Error:                {mod3.bse['did']:>12.4f}")
print(f"    t-stat:                   {mod3.tvalues['did']:>12.3f}")
print(f"    p-value:                  {mod3.pvalues['did']:>12.4f}")
print(f"    R-squared:                {mod3.rsquared:>12.4f}")
pct_effect = (np.exp(mod3.params['did']) - 1) * 100
print(f"    Implied % effect:         {pct_effect:>12.1f}%")

# --- 6. Parallel trends check (it doesn't hold cleanly) ---
print("\n" + "=" * 70)
print("PARALLEL TRENDS ASSUMPTION CHECK")
print("=" * 70)

# Event study: interact treatment with year dummies (2019 = reference)
panel_es = panel.copy()
panel_es["year_str"] = panel_es["year"].astype(str)
panel_es["rel_year"] = panel_es["year"] - 2021
for y in YEARS:
    if y == 2019:  # reference year
        continue
    panel_es[f"treat_x_{y}"] = ((panel_es["treated"] == 1) & (panel_es["year"] == y)).astype(int)

formula_es = "capacity_mw ~ " + " + ".join([f"treat_x_{y}" for y in YEARS if y != 2019]) + " + C(State) + C(year)"
mod_es = smf.ols(formula_es, data=panel_es).fit()

print("\n  Event Study Coefficients (reference year: 2019):")
print(f"  {'Year':<8} {'Coeff (MW)':>12} {'Std Err':>12} {'t-stat':>10} {'p-value':>10}")
print(f"  {'----':<8} {'----------':>12} {'-------':>12} {'------':>10} {'-------':>10}")
print(f"  {'2019':<8} {'0 (ref)':>12} {'':>12} {'':>10} {'':>10}")
es_coeffs = {}
for y in [2020, 2021, 2022, 2023]:
    var = f"treat_x_{y}"
    coeff = mod_es.params[var]
    se = mod_es.bse[var]
    t = mod_es.tvalues[var]
    p = mod_es.pvalues[var]
    es_coeffs[y] = (coeff, se)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    print(f"  {y:<8} {coeff:>12,.1f} {se:>12,.1f} {t:>10.3f} {p:>10.4f} {sig}")

# Pre-trend test
pre_coeff = mod_es.params["treat_x_2020"]
pre_p = mod_es.pvalues["treat_x_2020"]
print(f"\n  Pre-trend test (2020 vs 2019):")
print(f"    Coefficient: {pre_coeff:,.1f} MW, p = {pre_p:.4f}")
if pre_p > 0.10:
    print("    --> Cannot reject parallel trends in pre-period (good).")
else:
    print("    --> Pre-trends may differ (potential violation of parallel trends).")

# --- 7. Event study plot ---
fig2, ax2 = plt.subplots(figsize=(8, 5))
es_years = [2019, 2020, 2021, 2022, 2023]
es_vals = [0] + [es_coeffs[y][0] for y in [2020, 2021, 2022, 2023]]
es_ses = [0] + [es_coeffs[y][1] for y in [2020, 2021, 2022, 2023]]
ci95 = [1.96 * s for s in es_ses]

ax2.errorbar(es_years, es_vals, yerr=ci95, fmt="o-", color="#c0392b",
             linewidth=2, markersize=8, capsize=5, capthick=2, zorder=5)
ax2.axhline(y=0, color="black", linewidth=0.8, alpha=0.5)
ax2.axvline(x=2021, color="gray", linestyle=":", linewidth=1.5, label="China ban (Sep 2021)")
ax2.fill_between([2018.5, 2021], ax2.get_ylim()[0] - 5000, ax2.get_ylim()[1] + 5000,
                 alpha=0.07, color="green", label="Pre-treatment")
ax2.fill_between([2021, 2023.5], ax2.get_ylim()[0] - 5000, ax2.get_ylim()[1] + 5000,
                 alpha=0.07, color="red", label="Post-treatment")
ax2.set_xlabel("Year", fontsize=12)
ax2.set_ylabel("DiD Coefficient (MW, relative to 2019)", fontsize=12)
ax2.set_title("Event Study: TX Treatment Effect on Solar+Wind Capacity\n(reference year = 2019)",
              fontsize=13, fontweight="bold")
ax2.set_xticks(YEARS)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
fig2.tight_layout()
fig2.savefig(OUTPUT / "did_event_study.png", dpi=200, bbox_inches="tight")
print(f"\n  Saved: {OUTPUT / 'did_event_study.png'}")

# --- 8. Solar vs wind breakdown ---
print("\n" + "=" * 70)
print("CAPACITY BREAKDOWN: SOLAR vs WIND")
print("=" * 70)

for fuel, label in [("SUN", "Solar"), ("WND", "Wind")]:
    fuel_df = capacity_by_fuel[capacity_by_fuel["fuel"] == fuel]
    tx_f = fuel_df[fuel_df["State"] == "TX"].set_index("year")["capacity_mw"]
    ctrl_f = fuel_df[fuel_df["State"].isin(CONTROL_STATES)].groupby("year")["capacity_mw"].mean()
    print(f"\n  {label}:")
    print(f"    {'Year':<8} {'TX (MW)':>12} {'Ctrl Avg (MW)':>14}")
    for y in YEARS:
        t_val = tx_f.get(y, 0)
        c_val = ctrl_f.get(y, 0)
        print(f"    {y:<8} {t_val:>12,.0f} {c_val:>14,.0f}")
    if 2019 in tx_f.index and 2023 in tx_f.index:
        tx_growth = tx_f.get(2023, 0) - tx_f.get(2019, 0)
        ctrl_growth = ctrl_f.get(2023, 0) - ctrl_f.get(2019, 0)
        print(f"    Growth (2019-2023): TX = {tx_growth:,.0f} MW, Ctrl Avg = {ctrl_growth:,.0f} MW")

# --- 9. Summary ---
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
  Research question: Did China's 2021 Bitcoin mining ban accelerate
  Texas renewable energy capacity growth?

  DiD estimate (Model 2, levels): {mod2.params['did']:,.1f} MW (p = {mod2.pvalues['did']:.4f})
  DiD estimate (Model 3, log):    {mod3.params['did']:.4f} ({pct_effect:.1f}%) (p = {mod3.pvalues['did']:.4f})

  Interpretation: Relative to the control group of midwestern wind states,
  Texas saw an {"additional" if mod2.params['did'] > 0 else "lower"} {abs(mod2.params['did']):,.0f} MW of solar+wind capacity
  in the post-ban period (2022-2023) compared to the pre-ban period (2019-2020),
  beyond what would be expected from common trends.

  Parallel trends: Pre-treatment coefficient (2020 vs 2019) = {pre_coeff:,.0f} MW, p = {pre_p:.4f}
  {"Parallel trends assumption appears satisfied." if pre_p > 0.1 else "Caution: possible pre-trend divergence."}

  Caveats:
  - Small N (6 states x 5 years = 30 obs) limits statistical power
  - Many confounders: state-level energy policy (TX ERCOT deregulation,
    PUCT incentives), federal ITC/PTC, supply chain factors
  - Attribution to mining specifically vs general TX growth is uncertain
  - 2021 is ambiguous (ban announced mid-year); treated as transition year
""")

print("Analysis complete. Figures saved to output/")
