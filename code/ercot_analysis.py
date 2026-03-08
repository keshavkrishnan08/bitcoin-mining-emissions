#!/usr/bin/env python3
"""
Hourly ERCOT analysis using EIA-930 data (2023).
Shows that monthly data hides the real volatility of wind and solar.
"""

import os
import sys
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

warnings.filterwarnings('ignore')

# Paths (relative to this script)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Step 1: Download EIA hourly data ---

def download_eia_data():
    """Download EIA-930 balance data for 2023 (two 6-month CSV files)."""
    import urllib.request

    urls = [
        "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2023_Jan_Jun.csv",
        "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2023_Jul_Dec.csv",
    ]

    dfs = []
    for url in urls:
        fname = os.path.basename(url)
        fpath = os.path.join(DATA_DIR, fname)

        if os.path.exists(fpath):
            print(f"  Already downloaded: {fname}")
        else:
            print(f"  Downloading: {fname} ...")
            try:
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (research project)'
                })
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = resp.read()
                with open(fpath, 'wb') as f:
                    f.write(data)
                print(f"    Saved ({len(data)/1e6:.1f} MB)")
            except Exception as e:
                print(f"    Download failed: {e}")
                continue

        try:
            df = pd.read_csv(fpath, low_memory=False)
            dfs.append(df)
            print(f"    Loaded {len(df):,} rows")
        except Exception as e:
            print(f"    Failed to read CSV: {e}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Combined dataset: {len(combined):,} rows, {len(combined.columns)} columns")
    return combined


# --- Step 2: Extract ERCOT rows and clean up ---

def build_hourly_dataset(balance_df):
    """Filter to ERCOT and build clean hourly dataset."""

    print("\n--- Extracting ERCOT data ---")

    # ERCOT balancing authority = ERCO
    ba_col = 'Balancing Authority'
    erco = balance_df[balance_df[ba_col] == 'ERCO'].copy()
    print(f"  ERCOT rows: {len(erco):,}")

    if len(erco) == 0:
        print(f"  Available BAs: {sorted(balance_df[ba_col].unique())}")
        for ba in balance_df[ba_col].unique():
            if 'ERC' in str(ba).upper():
                erco = balance_df[balance_df[ba_col] == ba].copy()
                print(f"  Found match: {ba} with {len(erco)} rows")
                break

    # Parse times
    time_col = 'UTC Time at End of Hour'
    erco[time_col] = pd.to_datetime(erco[time_col])

    # Pull out the columns we need
    hourly = pd.DataFrame()
    hourly['datetime'] = erco[time_col].values

    # Demand and generation by source
    hourly['demand_mw'] = pd.to_numeric(erco['Demand (MW)'].values, errors='coerce')

    hourly['total_gen_mw'] = pd.to_numeric(erco['Net Generation (MW)'].values, errors='coerce')
    hourly['wind_mw'] = pd.to_numeric(erco['Net Generation (MW) from Wind'].values, errors='coerce')
    hourly['solar_mw'] = pd.to_numeric(erco['Net Generation (MW) from Solar'].values, errors='coerce')
    hourly['gas_mw'] = pd.to_numeric(erco['Net Generation (MW) from Natural Gas'].values, errors='coerce')
    hourly['coal_mw'] = pd.to_numeric(erco['Net Generation (MW) from Coal'].values, errors='coerce')
    hourly['nuclear_mw'] = pd.to_numeric(erco['Net Generation (MW) from Nuclear'].values, errors='coerce')

    # NaN wind/solar = 0
    hourly['wind_mw'] = hourly['wind_mw'].fillna(0)
    hourly['solar_mw'] = hourly['solar_mw'].fillna(0)

    # Derived columns
    hourly['renewable_mw'] = hourly['wind_mw'] + hourly['solar_mw']
    hourly['renewable_share'] = (hourly['renewable_mw'] / hourly['total_gen_mw'] * 100)
    hourly['residual_demand_mw'] = hourly['demand_mw'] - hourly['renewable_mw']
    hourly['wind_share'] = (hourly['wind_mw'] / hourly['total_gen_mw'] * 100)

    # Clean up
    hourly = hourly.sort_values('datetime').drop_duplicates(subset='datetime').reset_index(drop=True)

    # Drop bad rows
    hourly = hourly.dropna(subset=['demand_mw', 'total_gen_mw']).reset_index(drop=True)
    hourly = hourly[(hourly['total_gen_mw'] > 5000) & (hourly['total_gen_mw'] < 100000)]
    hourly = hourly.reset_index(drop=True)

    print(f"  Final hourly dataset: {len(hourly):,} rows")
    print(f"  Date range: {hourly['datetime'].min()} to {hourly['datetime'].max()}")
    print(f"  Demand range: {hourly['demand_mw'].min():,.0f} - {hourly['demand_mw'].max():,.0f} MW")
    print(f"  Wind range: {hourly['wind_mw'].min():,.0f} - {hourly['wind_mw'].max():,.0f} MW")
    print(f"  Solar range: {hourly['solar_mw'].min():,.0f} - {hourly['solar_mw'].max():,.0f} MW")
    print(f"  Renewable share: {hourly['renewable_share'].min():.1f}% - {hourly['renewable_share'].max():.1f}%")
    print(f"  Residual demand: {hourly['residual_demand_mw'].min():,.0f} - {hourly['residual_demand_mw'].max():,.0f} MW")

    return hourly


# --- Step 3: Key statistics ---

def compute_statistics(hourly):
    """Stats on renewable variability at hourly resolution."""

    print("\n" + "="*70)
    print("KEY STATISTICS: ERCOT Hourly Renewable Generation (2023)")
    print("="*70)

    print(f"\nDataset: {len(hourly):,} hourly observations")
    print(f"Period: {hourly['datetime'].min().strftime('%Y-%m-%d')} to "
          f"{hourly['datetime'].max().strftime('%Y-%m-%d')}")

    print(f"\n--- Wind Generation ---")
    print(f"  Mean:   {hourly['wind_mw'].mean():,.0f} MW")
    print(f"  Median: {hourly['wind_mw'].median():,.0f} MW")
    print(f"  Std:    {hourly['wind_mw'].std():,.0f} MW")
    print(f"  Min:    {hourly['wind_mw'].min():,.0f} MW")
    print(f"  Max:    {hourly['wind_mw'].max():,.0f} MW")
    print(f"  CV:     {hourly['wind_mw'].std()/hourly['wind_mw'].mean()*100:.1f}%")

    print(f"\n--- Solar Generation ---")
    print(f"  Mean:   {hourly['solar_mw'].mean():,.0f} MW")
    print(f"  Median: {hourly['solar_mw'].median():,.0f} MW")
    print(f"  Max:    {hourly['solar_mw'].max():,.0f} MW")

    print(f"\n--- Combined Renewable Generation ---")
    print(f"  Mean:   {hourly['renewable_mw'].mean():,.0f} MW")
    print(f"  Std:    {hourly['renewable_mw'].std():,.0f} MW")
    print(f"  Max:    {hourly['renewable_mw'].max():,.0f} MW")
    print(f"  Mean share: {hourly['renewable_share'].mean():.1f}%")
    print(f"  Max share:  {hourly['renewable_share'].max():.1f}%")

    print(f"\n--- Residual Demand (Demand - Renewables) ---")
    print(f"  Mean:   {hourly['residual_demand_mw'].mean():,.0f} MW")
    print(f"  Std:    {hourly['residual_demand_mw'].std():,.0f} MW")
    print(f"  Min:    {hourly['residual_demand_mw'].min():,.0f} MW")
    print(f"  Max:    {hourly['residual_demand_mw'].max():,.0f} MW")
    print(f"  Range:  {hourly['residual_demand_mw'].max() - hourly['residual_demand_mw'].min():,.0f} MW")

    # Key comparison: hourly vs monthly variability
    hourly_copy = hourly.copy()
    hourly_copy['month'] = hourly_copy['datetime'].dt.month
    monthly_wind = hourly_copy.groupby('month')['wind_mw'].mean()
    monthly_renewable_share = hourly_copy.groupby('month')['renewable_share'].mean()

    print(f"\n--- Hourly vs Monthly Variability (Key Finding) ---")
    print(f"  Hourly wind std:         {hourly['wind_mw'].std():,.0f} MW")
    print(f"  Monthly avg wind std:    {monthly_wind.std():,.0f} MW")
    ratio = hourly['wind_mw'].std() / monthly_wind.std()
    print(f"  Ratio (hourly/monthly):  {ratio:.1f}x")
    print(f"  ==> Hourly data reveals {ratio:.0f}x more variability than monthly averages!")

    print(f"\n  Hourly renewable share std:  {hourly['renewable_share'].std():.1f}%")
    print(f"  Monthly renewable share std: {monthly_renewable_share.std():.1f}%")
    share_ratio = hourly['renewable_share'].std() / monthly_renewable_share.std()
    print(f"  Ratio: {share_ratio:.1f}x")

    # Correlation
    corr = hourly['renewable_mw'].corr(hourly['residual_demand_mw'])
    print(f"\n--- Correlation Analysis ---")
    print(f"  Corr(renewable MW, residual demand): {corr:.3f}")
    print(f"  ==> Strong negative correlation: more renewables = lower residual demand")

    # High vs low renewable periods
    q25 = hourly['renewable_share'].quantile(0.25)
    q75 = hourly['renewable_share'].quantile(0.75)
    high_ren = hourly[hourly['renewable_share'] >= q75]
    low_ren = hourly[hourly['renewable_share'] <= q25]

    print(f"\n--- Impact of High vs Low Renewable Periods ---")
    print(f"  Median renewable share: {hourly['renewable_share'].median():.1f}%")
    print(f"  Top quartile (>={q75:.1f}% renewable):")
    print(f"    N hours:             {len(high_ren):,}")
    print(f"    Avg residual demand: {high_ren['residual_demand_mw'].mean():,.0f} MW")
    print(f"    Avg demand:          {high_ren['demand_mw'].mean():,.0f} MW")
    print(f"  Bottom quartile (<={q25:.1f}% renewable):")
    print(f"    N hours:             {len(low_ren):,}")
    print(f"    Avg residual demand: {low_ren['residual_demand_mw'].mean():,.0f} MW")
    print(f"    Avg demand:          {low_ren['demand_mw'].mean():,.0f} MW")
    drop = low_ren['residual_demand_mw'].mean() - high_ren['residual_demand_mw'].mean()
    print(f"  Residual demand difference: {drop:,.0f} MW")

    # Hours above renewable thresholds
    for thresh in [40, 50, 60]:
        n = len(hourly[hourly['renewable_share'] > thresh])
        print(f"\n  Hours with renewable share > {thresh}%: {n:,} ({n/len(hourly)*100:.1f}%)")

    # Wind ramp rates
    hourly_sorted = hourly.sort_values('datetime')
    wind_ramp = hourly_sorted['wind_mw'].diff()
    print(f"\n--- Wind Ramp Rates (Hour-to-Hour Change) ---")
    print(f"  Mean absolute ramp:   {wind_ramp.abs().mean():,.0f} MW/hour")
    print(f"  95th pctile ramp:     {wind_ramp.abs().quantile(0.95):,.0f} MW/hour")
    print(f"  Max upward ramp:      +{wind_ramp.max():,.0f} MW/hour")
    print(f"  Max downward ramp:    {wind_ramp.min():,.0f} MW/hour")

    # What this means for mining
    print(f"\n--- Implications for Flexible Load (Bitcoin Mining) ---")
    resid_range = hourly['residual_demand_mw'].max() - hourly['residual_demand_mw'].min()
    print(f"  Residual demand total swing: {resid_range:,.0f} MW")
    print(f"  95th pctile hourly wind ramp: {wind_ramp.abs().quantile(0.95):,.0f} MW")
    print(f"  ==> Flexible loads must respond to swings of thousands of MW per hour")
    print(f"  ==> Monthly data completely misses these dynamics")

    return monthly_wind


# --- Step 4: Figures ---

def create_figure_10(hourly):
    """Fig 10: Renewable share vs residual demand scatter."""
    print("\n--- Creating Figure 10 ---")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel
    ax = axes[0]
    scatter = ax.scatter(
        hourly['renewable_share'],
        hourly['residual_demand_mw'] / 1000,
        c=hourly['datetime'].dt.month,
        cmap='twilight',
        alpha=0.15, s=3, rasterized=True
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Month', fontsize=10)

    mask = hourly['renewable_share'].notna() & hourly['residual_demand_mw'].notna()
    slope, intercept, r, p, se = stats.linregress(
        hourly.loc[mask, 'renewable_share'],
        hourly.loc[mask, 'residual_demand_mw'] / 1000
    )
    x_line = np.linspace(hourly['renewable_share'].min(), hourly['renewable_share'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2,
            label=f'Linear fit (R={r:.2f})')

    ax.set_xlabel('Renewable Share of Generation (%)', fontsize=12)
    ax.set_ylabel('Residual Demand (GW)', fontsize=12)
    ax.set_title('Higher Renewables Reduce Residual Demand', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right panel
    ax = axes[1]
    scatter = ax.scatter(
        hourly['wind_mw'] / 1000,
        hourly['residual_demand_mw'] / 1000,
        c=hourly['datetime'].dt.hour,
        cmap='viridis',
        alpha=0.15, s=3, rasterized=True
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Hour of Day', fontsize=10)

    slope2, intercept2, r2, p2, se2 = stats.linregress(
        hourly.loc[mask, 'wind_mw'] / 1000,
        hourly.loc[mask, 'residual_demand_mw'] / 1000
    )
    x_line = np.linspace(hourly['wind_mw'].min()/1000, hourly['wind_mw'].max()/1000, 100)
    ax.plot(x_line, slope2 * x_line + intercept2, 'r-', linewidth=2,
            label=f'Linear fit (R={r2:.2f})')

    ax.set_xlabel('Wind Generation (GW)', fontsize=12)
    ax.set_ylabel('Residual Demand (GW)', fontsize=12)
    ax.set_title('Wind Output vs Residual Demand', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Figure 10: ERCOT Hourly Renewable Impact on Residual Demand (2023)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'fig10_renewable_vs_residual_demand.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def create_figure_11(hourly):
    """Fig 11: Time series of demand and wind during high-variability periods."""
    print("\n--- Creating Figure 11 ---")

    hourly_sorted = hourly.sort_values('datetime').reset_index(drop=True)
    hourly_sorted['date'] = hourly_sorted['datetime'].dt.date

    # Spring period with big wind swings
    spring = hourly_sorted[hourly_sorted['datetime'].dt.month.isin([3, 4])]
    daily_var = spring.groupby('date').agg(
        wind_range=('wind_mw', lambda x: x.max() - x.min()),
        max_ren_share=('renewable_share', 'max'),
        min_residual=('residual_demand_mw', 'min')
    ).sort_values('wind_range', ascending=False)

    if len(daily_var) > 0:
        target_date = pd.Timestamp(daily_var.index[0])
    else:
        daily_all = hourly_sorted.groupby('date').agg(
            wind_range=('wind_mw', lambda x: x.max() - x.min())
        ).sort_values('wind_range', ascending=False)
        target_date = pd.Timestamp(daily_all.index[0])

    start = target_date - pd.Timedelta(days=2)
    end = target_date + pd.Timedelta(days=3)
    window = hourly_sorted[
        (hourly_sorted['datetime'] >= start) & (hourly_sorted['datetime'] <= end)
    ]

    # Summer period
    summer = hourly_sorted[hourly_sorted['datetime'].dt.month.isin([7, 8])]
    summer_daily = summer.groupby('date').agg(
        wind_range=('wind_mw', lambda x: x.max() - x.min()),
        demand_max=('demand_mw', 'max')
    ).sort_values('wind_range', ascending=False)

    summer_window = None
    if len(summer_daily) > 0:
        summer_target = pd.Timestamp(summer_daily.index[0])
        summer_start = summer_target - pd.Timedelta(days=2)
        summer_end = summer_target + pd.Timedelta(days=3)
        summer_window = hourly_sorted[
            (hourly_sorted['datetime'] >= summer_start) &
            (hourly_sorted['datetime'] <= summer_end)
        ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Spring panel
    ax = axes[0]
    ax.fill_between(window['datetime'], 0, window['wind_mw']/1000,
                    alpha=0.4, color='#2196F3', label='Wind')
    ax.fill_between(window['datetime'], window['wind_mw']/1000,
                    (window['wind_mw'] + window['solar_mw'])/1000,
                    alpha=0.4, color='#FFC107', label='Solar')
    ax.plot(window['datetime'], window['demand_mw']/1000,
            'k-', linewidth=2, label='Total Demand')
    ax.plot(window['datetime'], window['residual_demand_mw']/1000,
            'r-', linewidth=2, label='Residual Demand\n(Demand - Renewables)')

    ax.set_ylabel('Power (GW)', fontsize=12)
    ax.set_title(f'Spring Period: {start.strftime("%b %d")} - {end.strftime("%b %d, %Y")} '
                 f'(High Wind Variability)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))

    # Annotate min residual demand
    if len(window) > 0:
        min_idx = window['residual_demand_mw'].idxmin()
        min_row = window.loc[min_idx]
        ax.annotate(
            f'Min residual: {min_row["residual_demand_mw"]/1000:.1f} GW\n'
            f'Wind: {min_row["wind_mw"]/1000:.1f} GW\n'
            f'Ren share: {min_row["renewable_share"]:.0f}%',
            xy=(min_row['datetime'], min_row['residual_demand_mw']/1000),
            xytext=(30, 30), textcoords='offset points',
            fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'),
            arrowprops=dict(arrowstyle='->', color='red')
        )

    # Summer panel
    ax = axes[1]
    if summer_window is not None and len(summer_window) > 50:
        ax.fill_between(summer_window['datetime'], 0, summer_window['wind_mw']/1000,
                        alpha=0.4, color='#2196F3', label='Wind')
        ax.fill_between(summer_window['datetime'], summer_window['wind_mw']/1000,
                        (summer_window['wind_mw'] + summer_window['solar_mw'])/1000,
                        alpha=0.4, color='#FFC107', label='Solar')
        ax.plot(summer_window['datetime'], summer_window['demand_mw']/1000,
                'k-', linewidth=2, label='Total Demand')
        ax.plot(summer_window['datetime'], summer_window['residual_demand_mw']/1000,
                'r-', linewidth=2, label='Residual Demand')

        ax.set_title(f'Summer Period: {summer_start.strftime("%b %d")} - '
                     f'{summer_end.strftime("%b %d, %Y")} (Peak Demand + Wind Swings)',
                     fontsize=13, fontweight='bold')

        if len(summer_window) > 0:
            min_idx = summer_window['residual_demand_mw'].idxmin()
            min_row = summer_window.loc[min_idx]
            ax.annotate(
                f'Min residual: {min_row["residual_demand_mw"]/1000:.1f} GW\n'
                f'Ren share: {min_row["renewable_share"]:.0f}%',
                xy=(min_row['datetime'], min_row['residual_demand_mw']/1000),
                xytext=(30, 30), textcoords='offset points',
                fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'),
                arrowprops=dict(arrowstyle='->', color='red')
            )
    else:
        # Fallback
        monthly_share = hourly_sorted.groupby(
            hourly_sorted['datetime'].dt.month)['renewable_share'].mean()
        ax.hist(hourly_sorted['renewable_share'], bins=80, alpha=0.6,
                color='#2196F3', density=True, label='Hourly distribution')
        for m, v in monthly_share.items():
            ax.axvline(v, color='red', alpha=0.5, linestyle='--', linewidth=1)
        ax.axvline(monthly_share.mean(), color='red', linewidth=2, linestyle='--',
                   label='Monthly averages')
        ax.set_xlabel('Renewable Share (%)')
        ax.set_title('Hourly vs Monthly Renewable Share Distribution',
                     fontsize=13, fontweight='bold')

    ax.set_ylabel('Power (GW)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))

    fig.suptitle('Figure 11: ERCOT Hourly Demand-Supply Dynamics (2023)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'fig11_hourly_wind_demand_timeseries.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def create_supplementary_figures(hourly):
    """Extra plots: renewable share distribution, daily profile, ramp rates."""

    print("\n--- Creating supplementary figures ---")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Renewable share distribution
    ax = axes[0, 0]
    ax.hist(hourly['renewable_share'], bins=80, color='#2196F3', alpha=0.7, edgecolor='white')
    ax.axvline(hourly['renewable_share'].mean(), color='red', linewidth=2, linestyle='--',
               label=f'Mean: {hourly["renewable_share"].mean():.1f}%')
    ax.axvline(hourly['renewable_share'].median(), color='orange', linewidth=2, linestyle='--',
               label=f'Median: {hourly["renewable_share"].median():.1f}%')
    ax.set_xlabel('Renewable Share (%)', fontsize=11)
    ax.set_ylabel('Count (hours)', fontsize=11)
    ax.set_title('Distribution of Hourly Renewable Share', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Daily profile
    ax = axes[0, 1]
    hourly_copy = hourly.copy()
    hourly_copy['hour'] = hourly_copy['datetime'].dt.hour
    profile = hourly_copy.groupby('hour').agg(
        wind=('wind_mw', 'mean'),
        solar=('solar_mw', 'mean'),
        demand=('demand_mw', 'mean'),
        residual=('residual_demand_mw', 'mean')
    )
    ax.bar(profile.index, profile['wind']/1000, color='#2196F3', alpha=0.7, label='Wind')
    ax.bar(profile.index, profile['solar']/1000, bottom=profile['wind']/1000,
           color='#FFC107', alpha=0.7, label='Solar')
    ax.plot(profile.index, profile['demand']/1000, 'k-o', linewidth=2, markersize=4,
            label='Demand')
    ax.plot(profile.index, profile['residual']/1000, 'r-s', linewidth=2, markersize=4,
            label='Residual Demand')
    ax.set_xlabel('Hour of Day (UTC)', fontsize=11)
    ax.set_ylabel('Power (GW)', fontsize=11)
    ax.set_title('Average Daily Profile', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) Wind ramp rates
    ax = axes[1, 0]
    hourly_s = hourly.sort_values('datetime')
    ramp = hourly_s['wind_mw'].diff()
    ax.hist(ramp.dropna()/1000, bins=100, color='#4CAF50', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Wind Ramp Rate (GW/hour)', fontsize=11)
    ax.set_ylabel('Count (hours)', fontsize=11)
    ax.set_title('Distribution of Hourly Wind Ramp Rates', fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3)

    # (d) Monthly box plot
    ax = axes[1, 1]
    month_data = [hourly[hourly['datetime'].dt.month == m]['renewable_share'].values
                  for m in range(1, 13)]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    bp = ax.boxplot(month_data, labels=month_labels, patch_artist=True,
                    showfliers=False, whis=[5, 95])
    for patch in bp['boxes']:
        patch.set_facecolor('#2196F3')
        patch.set_alpha(0.6)
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Renewable Share (%)', fontsize=11)
    ax.set_title('Monthly Distribution of Hourly Renewable Share\n(whiskers: 5th-95th pctile)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.suptitle('ERCOT Hourly Renewable Variability Analysis (2023)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'fig_ercot_hourly_variability_supplement.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


# --- Run everything ---

def main():
    print("="*70)
    print("ERCOT HOURLY RENEWABLE GENERATION ANALYSIS (2023)")
    print("Proving demand-supply mismatch at hourly resolution")
    print("="*70)

    # Step 1: Download data
    print("\n[1/4] Downloading EIA Hourly Grid Monitor data...")
    balance_df = download_eia_data()
    if balance_df is None:
        print("ERROR: Could not download data")
        sys.exit(1)

    # Step 2: Process
    print("\n[2/4] Processing ERCOT data...")
    hourly = build_hourly_dataset(balance_df)

    # Save processed data
    csv_path = os.path.join(DATA_DIR, 'ercot_hourly_2023.csv')
    hourly.to_csv(csv_path, index=False)
    print(f"  Saved processed data: {csv_path}")

    # Step 3: Statistics
    print("\n[3/4] Computing statistics...")
    compute_statistics(hourly)

    # Step 4: Figures
    print("\n[4/4] Creating figures...")
    create_figure_10(hourly)
    create_figure_11(hourly)
    create_supplementary_figures(hourly)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nFigures saved to: {OUTPUT_DIR}/")
    print("  - fig10_renewable_vs_residual_demand.png")
    print("  - fig11_hourly_wind_demand_timeseries.png")
    print("  - fig_ercot_hourly_variability_supplement.png")
    print(f"\nData saved to: {csv_path}")


if __name__ == '__main__':
    main()
