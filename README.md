# Bitcoin mining and carbon emissions

ASSIP screening task for Professor Jiasun Li, George Mason University. It stress-tests the empirical claims in Li, Ren, and Bellos (2026) against public EIA data for Texas, 2019 to 2023, to see which ones hold up.

## What it checks

- **Generation and emissions trends** from EIA Form 860 (capacity) and Form 923 (monthly generation and fuel), with EPA eGRID 2022 emission factors. (`code/analysis.py`, Figures 1 to 8)
- **Did Texas renewables grow faster after China's 2021 mining ban?** A difference-in-differences test. Texas did grow faster, but it was already growing faster before the ban, so the ban is not a clean explanation. (`code/did_analysis.py`)
- **What monthly data hides.** An hourly ERCOT analysis on EIA-930 shows the real volatility of wind and solar that monthly averages smooth away. (`code/ercot_analysis.py`)
- **Calibration** of the generation and emission figures against the published numbers. (`code/calibration.py`)

## Data

Public sources only: EIA Form 860, Form 923, and EIA-930, plus EPA eGRID 2022 emission factors. All figures are written to `output/`.

## Run it

See [`code/README.md`](code/README.md) for the exact packages, the EIA downloads, and the run order.

Part of the ASSIP program under Dr. Jiasun Li, Costello College of Business, George Mason University.
