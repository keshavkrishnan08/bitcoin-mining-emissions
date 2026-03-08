# Bitcoin Mining & Carbon Emissions

ASSIP Screening Task 1 for Professor Jiasun Li, George Mason University.

Tests claims from Li, Ren, and Bellos (2026) using public EIA data for Texas, 2019-2023.

## How to run

1. Install Python packages:
   ```
   pip install pandas numpy matplotlib scipy statsmodels openpyxl
   ```

2. Download EIA data and unzip into `../data/`:
   - [EIA Form 860](https://www.eia.gov/electricity/data/eia860/) (2019-2023) — put each year in `data/eia860_YYYY/`
   - [EIA Form 923](https://www.eia.gov/electricity/data/eia923/) (2019-2023) — put each year in `data/eia923_YYYY/`
   - EIA-930 hourly data downloads automatically when you run `ercot_analysis.py`

3. Run from the project root:
   ```
   python3 code/analysis.py
   python3 code/calibration.py
   python3 code/did_analysis.py
   python3 code/ercot_analysis.py
   ```

Figures go to `output/`. The report is `code/report.tex`.

## What each script does

| Script | Purpose |
|---|---|
| `analysis.py` | Main analysis. Loads EIA 860/923 for Texas. Tests demand-supply correlation, renewable volatility, capacity factor distributions, emission trends, cross-state comparisons. Makes Figures 1-8. |
| `calibration.py` | Maps the paper's model parameters to real EIA numbers. Checks which case from Section 3.6.3 applies at monthly vs hourly resolution. |
| `did_analysis.py` | Diff-in-diff comparing TX renewable growth to five control states around China's 2021 mining ban. Includes parallel trends test. |
| `ercot_analysis.py` | Downloads EIA-930 hourly ERCOT data. Compares hourly vs monthly variability for wind and solar. Finds hours with >50% renewable share. |

## Data sources

- EIA Form 860: plant-level capacity data
- EIA Form 923: monthly generation and fuel data
- EIA-930: hourly grid data
- EPA eGRID (2022): emission factors
