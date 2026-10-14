# Bitcoin Mining and Carbon Emissions

**A data check on a published economic argument: that Bitcoin mining, as a flexible load, makes
renewables pay and lowers grid emissions. It tests the model's assumptions against EIA capacity
and generation data for Texas, 2019 to 2023.**

Li, Ren and Bellos (2026) model miners as a shock absorber. Miners soak up surplus renewable
output and switch off when demand peaks, which raises the value of building wind and solar and
cuts fossil generation. That conclusion rests on empirical premises: renewable supply swings more
than demand, the two are only loosely correlated, capacity factors are spread out, and the
model's cost parameters fall in the region where the emission reduction holds. This repo checks
each premise with public EIA data for Texas, one of the states the code treats as a mining
destination after China's September 2021 ban. It was built as an ASSIP research screening task
under Dr. Jiasun Li (Costello College of Business, George Mason University).

```
EIA Form 860 (capacity) + Form 923 (monthly generation, fuel receipts), 2019 to 2023
  + EIA-930 hourly ERCOT balance data, 2023
    ->  fuel codes grouped into solar and wind, other renewable, nuclear, fossil
    ->  Texas capacity growth, generation mix, demand-supply correlation, variability, capacity factors
    ->  CO2 from fossil generation at 0.386 t/MWh (eGRID 2022 ERCOT fossil average)
    ->  difference-in-differences: Texas vs five control states around the 2021 ban
    ->  the paper's model parameters mapped to Texas data, emission condition checked
    ->  hourly ERCOT variability that monthly averages hide
```

Check the premises, not just the conclusion.

## The core idea

1. **A model's conclusion is only as good as its parameter values.** `calibration.py` maps each
   parameter of the paper's model (peak demand, peak renewable capacity factor, fuel cost,
   renewable capacity cost, retail price, number of mining pools, mining revenue, demand-supply
   correlation) to a Texas value, then checks which of the paper's investment cases applies and
   whether its emission-reduction condition, c/2 < C/S_bar < c, holds.
2. **Time resolution can flip the answer.** Monthly averages smooth wind and solar away. The same
   check is rerun with hourly-scale parameters, and `ercot_analysis.py` measures the hourly swings
   directly from EIA-930.
3. **Use the ban as a natural experiment, and test its weak point.** A difference-in-differences
   compares Texas with five control states before and after the 2021 ban, and an event study
   checks whether Texas was already on a steeper path.

## What it checks (EIA data, Texas, 2019 to 2023)

| Premise or claim | Test | Data | Script |
|---|---|---|---|
| Mining encourages renewable buildout | Texas solar and wind capacity 2019 to 2023; 6 mining vs 10 non-mining states, t-test on share change | EIA 860 | `analysis.py` (Figs 1, 6) |
| Demand and renewable supply are imperfectly correlated | Pearson and Spearman on monthly totals | EIA 923 | `analysis.py` (Fig 3) |
| Renewable supply is more volatile than demand | ratio of coefficients of variation | EIA 923 | `analysis.py` (Fig 4) |
| Capacity factor is uniform on [0, S_bar] | plant-level capacity factors, KS test against uniform | EIA 860 + 923 | `analysis.py` (Fig 5) |
| The grid is getting cleaner | annual CO2 and CO2 intensity, residual demand trend | EIA 923, eGRID 2022 | `analysis.py` (Figs 7, 8) |
| The ban accelerated Texas renewables | 2x2 DiD, state and year fixed effects in levels and logs, event study | EIA 860 | `did_analysis.py` |
| The emission-reduction condition holds for Texas | parameter mapping, case classification, 50 x 50 sensitivity grid over fuel and capacity cost | EIA 860, 923 (incl. fuel receipts), NREL ATB 2023 costs | `calibration.py` (Fig 9) |
| Monthly data is enough | hourly wind, solar and residual demand, hour-to-hour ramps | EIA-930, 2023 | `ercot_analysis.py` (Figs 10, 11) |

The code records two conclusions in its own docstrings. Texas renewables did grow faster than the
controls after the ban, but they were already growing faster before it, so the ban is not a clean
explanation (`did_analysis.py`). And monthly and hourly parameters give opposite answers on
whether mining can cut emissions (`calibration.py`). The scripts rebuild the statistics and
figures behind both; none are committed.

## How it works

**Data.** EIA Form 860 generator capacity (operable units) and Form 923 monthly net generation by
plant and fuel for 2019 to 2023, plus Form 923 fuel receipts for Texas natural gas costs. EIA-930
hourly balance files for ERCOT in 2023 download automatically. Fuel codes SUN and WND count as
solar and wind; hydro, geothermal, biomass and waste fuels as other renewable; NUC as nuclear;
everything else as fossil.

**Texas trends.** Capacity by fuel class, the monthly generation stack, the renewable-share
trend, and residual demand (total minus renewable generation). Total in-state generation stands in
for demand. CO2 uses one rate for all fossil generation, 0.386 t/MWh.

**Difference-in-differences.** Texas against IL, IA, OK, KS and MN. Pre-period 2019 to 2020,
post-period 2022 to 2023, with 2021 treated as a transition year. A manual 2x2 estimate, OLS with
and without state and year fixed effects, a log-capacity version, and an event study with 2019 as
the reference year, on 30 state-year observations.

**Calibration.** Peak and mean demand come from monthly average power, the peak capacity factor
from monthly renewable output over installed capacity, and fuel cost from the weighted Texas gas
price times an 8.0 MMBtu/MWh heat rate. Capacity cost uses NREL ATB 2023 figures ($1,300/kW wind,
$1,000/kW solar, weighted 60/40 and annualized at 8%). The script classifies which of the paper's
three investment cases applies, compares variable cost with and without mining, repeats the check
with hourly-scale parameters, and sweeps fuel cost from $10 to $80/MWh against capacity cost from
$1 to $25/MWh.

**Hourly ERCOT.** Demand, wind, solar and residual demand for every hour of 2023; hourly against
monthly variability of wind; the share of hours above 40%, 50% and 60% renewable; and the size
of hour-to-hour wind ramps.

## Run it

```bash
pip install pandas numpy matplotlib scipy statsmodels openpyxl
# download EIA Form 860 and Form 923 for 2019 to 2023 into data/eia860_YYYY/ and data/eia923_YYYY/
python3 code/analysis.py        # Figures 1 to 8
python3 code/calibration.py     # parameter mapping, emission case, Figure 9
python3 code/did_analysis.py    # difference-in-differences and event study
python3 code/ercot_analysis.py  # downloads EIA-930 for 2023, Figures 10 and 11
```

Run from the repo root; figures are written to `output/`. `code/README.md` lists the download
links.

## Layout

```
code/
  analysis.py        Texas capacity, generation mix, correlation, variability, capacity factors, emissions (Figs 1 to 8)
  calibration.py     paper parameters from EIA data, emission case, hourly-adjusted check, sensitivity (Fig 9)
  did_analysis.py    difference-in-differences and event study around the 2021 ban
  ercot_analysis.py  hourly ERCOT analysis from EIA-930 (Figs 10, 11 and a supplement)
  README.md          packages, downloads, run order
data/                EIA downloads (gitignored, not in the repo)
output/              figures (gitignored, not in the repo)
```

## Honesty

* The repo holds code only. Data, figures and the written report are gitignored, so no result is
  quoted here; every statistic comes from running the scripts.
* The hourly-adjusted calibration is not estimated from the hourly data. Peak capacity factor
  0.85, peak demand 85 GW, minimum demand 30 GW and a 50% subsidy discount on capacity cost are set
  by hand in `calibration.py`. The subsidy changes at the same time as the time resolution, so the
  "monthly and hourly disagree" result mixes two changes.
* Several calibration inputs are assumptions rather than data: the heat rate, NREL capital costs
  and the 8% annualization, a $130/MWh retail price, 30 mining pools, and a monthly mining revenue
  built from constants.
* One CO2 rate covers all fossil generation, so a shift from coal to gas does not change modeled
  emissions per MWh. Demand is proxied by in-state generation, and "Texas" in EIA 860 and 923 is
  the plant's state, which includes plants outside ERCOT.
* The mining and non-mining state lists in `analysis.py` are hand-picked. The DiD has 30
  observations and many confounders (state policy, federal tax credits, supply chains), as the
  script itself notes.
* The closing summary in `analysis.py` prints fixed verdict labels (for example "SUPPORTS") for
  several checks whatever the statistics show. Read the printed numbers, not the labels.
* Scope: Bitcoin mining and the Texas grid. The repo does not model AI or data-center load and
  does not compare demand models.

MIT licensed.
