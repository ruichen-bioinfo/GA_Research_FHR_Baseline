# =============================================================================
# plot_visualization.py
# FHR Monitoring — Visualization & Publication-Quality Plots
#
# Purpose:
#   Generate all figures used in the study poster and report.
#   Five independent plot modules, each callable standalone or together.
#
#   Plot 1 — FHR Density Distribution by GA (KDE curves per GA group)
#   Plot 2 — FHR Linear Trend with 50 bpm Max-Density Band (scatter + band)
#   Plot 3 — Alarm Rate Comparison: 3-method bar chart (N=2253)
#   Plot 4 — Alarm Rate Comparison: 4-method bar chart (with New Threshold)
#   Plot 5 — ROC Curve: 5-fold CV Random Forest vs. Linear Baseline
#
# All plots are saved as high-resolution PNG (300 dpi) to RESULT_DIR.
# Call plot_all() from __main__ to generate every figure in sequence.
#
# Authors: Rui Chen,  Jiayi Wang
# =============================================================================

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

# ── Paths ──────────────────────────────────────────────────────────────────────
# DATA_DIR   : per-GA CSV files (columns: time, fhr) for density & trend plots
# RESULT_DIR : .hea + WFDB clinical records; also used as output folder
DATA_DIR   = r"/Users/jiayi/PycharmProjects/pythonProject9/venv/data"
RESULT_DIR = r"/Users/jiayi/PycharmProjects/pythonProject9/venv/result"

# Excel produced by test_control_variables.py — needed for ROC plot (Plot 5)
EXCEL_PATH = os.path.join(RESULT_DIR, "FHR_ML_Enhanced_Accuracy.xlsx")

# ── Shared font sizes (publication-quality, consistent across all figures) ─────
FS_TITLE  = 22
FS_AXIS   = 22
FS_TICK   = 22
FS_LEGEND = 16

# ── GA-adjusted dynamic baseline constants ─────────────────────────────────────
SLOPE     = -0.33
INTERCEPT = 150.21


# =============================================================================
# SECTION 1 — Shared Data Loading Helpers
# =============================================================================

def extract_ga_from_filename(filename):
    """Parse GA value from CSV filename pattern GA_<number>.csv."""
    m = re.search(r'GA_(\d+)', filename)
    return int(m.group(1)) if m else None


def load_csv_data(data_dir=DATA_DIR):
    """
    Read all per-GA CSV files from data_dir.
    Filters FHR values to physiological range [50, 200] bpm.
    Returns a single concatenated DataFrame with columns: time, fhr, ga.
    """
    all_dfs = []
    files   = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    for file in files:
        ga = extract_ga_from_filename(file)
        if ga is None:
            continue
        try:
            df = pd.read_csv(os.path.join(data_dir, file), usecols=['time', 'fhr'])
            df = df[(df['fhr'] >= 50) & (df['fhr'] <= 200)].copy()
            df['ga'] = ga
            all_dfs.append(df)
        except Exception:
            continue

    if not all_dfs:
        print("[WARN] No CSV data found in DATA_DIR.")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def load_fhr_only(data_dir=DATA_DIR):
    """
    Like load_csv_data() but only requires the 'fhr' column.
    Used for linear trend analysis where 'time' column may be absent.
    """
    all_dfs = []
    files   = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    for file in files:
        ga = extract_ga_from_filename(file)
        if ga is None:
            continue
        try:
            df = pd.read_csv(os.path.join(data_dir, file), usecols=['fhr'])
            df = df[(df['fhr'] >= 50) & (df['fhr'] <= 200)].copy()
            df['ga'] = ga
            all_dfs.append(df)
        except Exception:
            continue

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


# =============================================================================
# SECTION 2 — Plot 1: FHR Density Distribution by GA
# =============================================================================

def find_max_density_range(data, window=50):
    """
    Sliding-window search: find the 'window'-wide interval containing
    the most data points (highest density).
    Returns (start, end) in bpm.
    """
    if len(data) == 0:
        return (None, None)
    sorted_data = np.sort(data)
    max_count   = 0
    best_range  = (sorted_data[0], sorted_data[0] + window)
    right       = 0

    for left in range(len(sorted_data)):
        while right < len(sorted_data) and sorted_data[right] <= sorted_data[left] + window:
            right += 1
        if (right - left) > max_count:
            max_count  = right - left
            best_range = (sorted_data[left], sorted_data[left] + window)

    return best_range


def plot_fhr_density_by_ga(save=True):
    """
    Plot 1: Kernel Density Estimate (KDE) of FHR for each gestational age group.

    Each GA group gets a distinct colour curve. The 50 bpm window with the
    highest point density is shaded to illustrate the normal range shift.

    Also runs an overall linear regression of FHR ~ Time across the full
    dataset (from original distribution.py) and prints the equation.

    Output: FHR_Density_by_GA.png
    """
    full_data = load_csv_data()   # requires 'time' and 'fhr' columns
    if full_data.empty:
        print("[SKIP] Plot 1: no data available.")
        return

    # ── Overall linear regression: FHR ~ Time (from distribution.py) ──────────
    # This captures the long-term temporal drift of FHR across the full recording
    print(f"\n[INFO] Reading {full_data['ga'].nunique()} GA groups "
          f"({len(full_data):,} samples total)...")
    X_time = full_data['time'].values.reshape(-1, 1)
    y_fhr  = full_data['fhr'].values
    reg    = LinearRegression().fit(X_time, y_fhr)
    print(f"\n--- Overall Linear Regression (FHR ~ Time) ---")
    print(f"  Regression equation: FHR = {reg.intercept_:.2f} + "
          f"({reg.coef_[0]:.6f}) × Time")

    # ── Per-GA max-density intervals ───────────────────────────────────────────
    ga_groups         = full_data.groupby('ga')
    density_intervals = {}

    print(f"\n--- Max-Density Intervals per GA group (window = 50 bpm) ---")
    for ga, group in ga_groups:
        start, end = find_max_density_range(group['fhr'].values)
        density_intervals[ga] = (start, end)
        print(f"  GA {ga}: [{start:.1f} – {end:.1f}] bpm")

    # ── Draw KDE curves ────────────────────────────────────────────────────────
    plt.figure(figsize=(9, 6))
    colors = plt.cm.get_cmap('tab10', len(ga_groups))

    for i, (ga, group) in enumerate(ga_groups):
        kde    = gaussian_kde(group['fhr'])
        x_axis = np.linspace(50, 200, 500)
        plt.plot(x_axis, kde(x_axis), label=f'GA {ga}', color=colors(i))

        # Shade the highest-density interval for this GA group
        start, end = density_intervals[ga]
        if start is not None:
            plt.axvspan(start, end, alpha=0.10, color=colors(i))

    plt.title("FHR Density Distribution by GA",
              fontweight='bold', fontsize=FS_TITLE)
    plt.xlabel("FHR (bpm)",  fontsize=FS_AXIS)
    plt.ylabel("Density",    fontsize=FS_AXIS)
    plt.xlim(50, 200)
    plt.xticks(fontsize=FS_TICK)
    plt.yticks(fontsize=FS_TICK)
    plt.legend(fontsize=FS_LEGEND)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULT_DIR, 'FHR_Density_by_GA.png')
        plt.savefig(path, dpi=300)
        print(f"[INFO] Plot 1 saved → {path}")
    plt.show()


# =============================================================================
# SECTION 3 — Plot 2: FHR Linear Trend with 50 bpm Max-Density Band
# =============================================================================

def find_best_offset_for_window(residuals, window=50):
    """
    Given regression residuals, find the centre of the 50 bpm window
    that captures the most residuals (densest region).
    Returns the optimal centre offset (bpm).
    """
    sorted_res = np.sort(residuals)
    max_count  = 0
    best_center = 0
    right = 0

    for left in range(len(sorted_res)):
        while right < len(sorted_res) and sorted_res[right] <= sorted_res[left] + window:
            right += 1
        if (right - left) > max_count:
            max_count   = right - left
            best_center = sorted_res[left] + window / 2

    return best_center


def plot_fhr_linear_trend(save=True):
    """
    Plot 2: Scatter of all FHR readings against GA, overlaid with:
      - Red line   : density-centred linear regression trend
      - Blue band  : ±25 bpm window around the trend (50 bpm total width)

    The band captures the physiologically normal zone at each GA.
    Output: FHR_Linear_Trend.png
    """
    full_data = load_fhr_only()
    if full_data.empty:
        print("[SKIP] Plot 2: no data available.")
        return

    # Fit linear regression: FHR ~ GA
    X = full_data['ga'].values.reshape(-1, 1)
    y = full_data['fhr'].values
    model     = LinearRegression().fit(X, y)
    residuals = y - model.predict(X)

    print(f"\n[INFO] Linear regression: FHR = {model.intercept_:.2f} + "
          f"({model.coef_[0]:.4f}) × GA")

    # Shift the trend line to the centre of the densest residual window
    center_offset = find_best_offset_for_window(residuals, window=50)

    ga_min, ga_max = full_data['ga'].min(), full_data['ga'].max()
    x_trend = np.array([[ga_min], [ga_max]])
    y_mid   = model.predict(x_trend) + center_offset
    y_upper = y_mid + 25
    y_lower = y_mid - 25

    plt.figure(figsize=(12, 7))

    # Scatter (very transparent — many overlapping points)
    plt.scatter(full_data['ga'], full_data['fhr'],
                alpha=0.03, s=1, color='gray')

    # Density-centred trend line
    plt.plot(x_trend, y_mid, color='red', linewidth=2,
             label='Density Centre Trend')

    # 50 bpm normal band around the trend
    plt.fill_between(x_trend.flatten(), y_lower, y_upper,
                     color='blue', alpha=0.2,
                     label='Linear 50 bpm Density Band')

    plt.title("FHR Linear Trend with 50 bpm Max-Density Band",
              fontsize=FS_TITLE, fontweight='bold')
    plt.xlabel("Gestational Age (GA weeks)", fontsize=FS_AXIS)
    plt.ylabel("FHR (bpm)",                  fontsize=FS_AXIS)
    plt.xticks(sorted(full_data['ga'].unique()), fontsize=FS_TICK)
    plt.yticks(fontsize=FS_TICK)
    plt.ylim(50, 200)
    plt.legend(fontsize=FS_LEGEND)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULT_DIR, 'FHR_Linear_Trend.png')
        plt.savefig(path, dpi=300)
        print(f"[INFO] Plot 2 saved → {path}")
    plt.show()


# =============================================================================
# SECTION 4 — Plot 3: Alarm Rate Comparison (3-method, N=2253)
# =============================================================================

def plot_alarm_rate_3method(save=True):
    """
    Plot 3: Bar chart comparing alarm rates across three monitoring methods.

    Values are from the final validated results (poster figures):
      Fixed 110–160 bpm : 11.80%
      Linear Band        : 10.21%
      Random Forest 0.5  :  5.20%

    An annotation arrow highlights the precision improvement of the RF model.
    Output: Alarm_Rate_3method.png
    """
    rate_fixed  = 11.8
    rate_linear = 10.21
    rate_rf     = 5.20

    models = ['Fixed (110–160)', 'Linear Band', 'Random Forest (0.5)']
    rates  = [rate_fixed, rate_linear, rate_rf]
    colors = ['#bdc3c7', '#3498db', '#2ecc71']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, rates, color=colors, width=0.6)

    # Label each bar with its exact percentage
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                 f'{h:.2f}%', ha='center', va='bottom',
                 fontsize=15, fontweight='bold')

    plt.title('Final Comparison of Alarm Rates (N=2253)',
              fontweight='bold', pad=20, fontsize=FS_TITLE)
    plt.ylabel('Alarm Rate (%)', fontsize=18)
    plt.xticks(fontsize=FS_TICK)
    plt.yticks(fontsize=FS_TICK)
    plt.ylim(0, 16)
    plt.grid(axis='y', linestyle=':', alpha=0.7)

    # Arrow annotation pointing to the RF bar to highlight improvement
    plt.annotate(
        'Precision Improvement',
        xy=(2, rate_rf), xytext=(0.1, 15),
        arrowprops=dict(facecolor='black', arrowstyle='->',
                        connectionstyle='arc3,rad=.2'),
        fontsize=14
    )
    plt.tight_layout()

    if save:
        path = os.path.join(RESULT_DIR, 'Alarm_Rate_3method.png')
        plt.savefig(path, dpi=300)
        print(f"[INFO] Plot 3 saved → {path}")
    plt.show()


# =============================================================================
# SECTION 5 — Plot 4: Alarm Rate Comparison (4-method, with New Threshold)
# =============================================================================

def plot_alarm_rate_4method(save=True):
    """
    Plot 4: Extended alarm rate bar chart adding the literature-driven
    GA-stratified threshold (New Threshold) as a fourth method.

    Values:
      Fixed 110–160 bpm : 14.70%
      New Threshold      : 13.89%
      Linear Band        : 10.21%
      Random Forest 0.5  :  5.20%

    Output: Alarm_Rate_4method.png
    """
    rate_fixed    = 14.7
    rate_new_lit  = 13.89
    rate_linear   = 10.21
    rate_rf       = 5.20

    models = ['Fixed (110–160)', 'New Threshold', 'Linear Band', 'Random Forest (0.5)']
    rates  = [rate_fixed, rate_new_lit, rate_linear, rate_rf]
    colors = ['#bdc3c7', '#9b59b6', '#3498db', '#2ecc71']

    plt.figure(figsize=(12, 8))
    bars = plt.bar(models, rates, color=colors, width=0.6)

    # Label each bar
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                 f'{h:.2f}%', ha='center', va='bottom',
                 fontsize=20, fontweight='bold')

    plt.title('Final Comparison of Alarm Rates (N=2253)',
              fontweight='bold', pad=10, fontsize=FS_TITLE)
    plt.ylabel('Alarm Rate (%)', fontsize=18)
    plt.xticks(fontsize=FS_TICK)
    plt.yticks(fontsize=FS_TICK)
    plt.ylim(0, 18)
    plt.grid(axis='y', linestyle=':', alpha=0.7)

    # Annotation arrow pointing to RF bar
    plt.annotate(
        'Precision Improvement',
        xy=(2.8, rate_rf), xytext=(0, 16),
        arrowprops=dict(facecolor='black', arrowstyle='->',
                        connectionstyle='arc3,rad=.2'),
        fontsize=20
    )
    plt.tight_layout()

    if save:
        path = os.path.join(RESULT_DIR, 'Alarm_Rate_4method.png')
        plt.savefig(path, dpi=300)
        print(f"[INFO] Plot 4 saved → {path}")
    plt.show()


# =============================================================================
# SECTION 6 — Plot 5: ROC Curve — 5-Fold CV Random Forest vs. Baseline
# =============================================================================

def plot_roc_curve(save=True):
    """
    Plot 5: 5-fold stratified cross-validation ROC curve for the RF model,
    with individual fold curves (thin, transparent) and a bold mean curve.
    A red dashed baseline ROC uses the linear In-Band Ratio as a risk score.

    Requires EXCEL_PATH (FHR_ML_Enhanced_Accuracy.xlsx) generated by
    test_control_variables.py. Each fold's AUC is shown in the legend.

    Output: ROC_Publication_Quality.png
    """
    if not os.path.exists(EXCEL_PATH):
        print(f"[SKIP] Plot 5: Excel file not found at {EXCEL_PATH}\n"
              f"       Run test_control_variables.py (validation.py) first.")
        return

    df = pd.read_excel(EXCEL_PATH)
    df.columns = df.columns.str.strip().str.lower()

    # Features and target matching test_control_variables.py
    features = ['ga', 'be', 'meconium', 'ltv_std', 'stv', 'in_band_ratio', 'dec_area']
    target   = 'ph'

    df = df.dropna(subset=features + [target])
    X  = df[features].values
    y  = (df[target] < 7.15).astype(int).values

    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classifier = RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    )

    tprs     = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(10, 8))
    plt.rcParams['axes.unicode_minus'] = False

    # ── Draw one curve per fold ────────────────────────────────────────────────
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        classifier.fit(X[train_idx], y[train_idx])
        proba = classifier.predict_proba(X[test_idx])
        fpr, tpr, _ = roc_curve(y[test_idx], proba[:, 1])

        interp_tpr    = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        fold_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.5, alpha=0.3,
                 label=f'Fold {i+1} (AUC = {fold_auc:.2f})')

    # ── Mean ROC across all folds ──────────────────────────────────────────────
    mean_tpr     = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc     = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue', lw=4, alpha=0.9,
             label=f'Mean ML ROC (AUC = {mean_auc:.2f})')

    # ── Linear baseline ROC: in-band ratio as a continuous risk score ──────────
    # Higher in-band ratio → lower pathological risk, so invert for AUC direction
    baseline_probs = 1 - df['in_band_ratio'].values
    fpr_bs, tpr_bs, _ = roc_curve(y, baseline_probs)
    baseline_auc   = auc(fpr_bs, tpr_bs)
    plt.plot(fpr_bs, tpr_bs, color='red', linestyle='--', lw=2.5,
             label=f'Baseline (AUC = {baseline_auc:.2f})')

    plt.title('ROC Curve: ML vs. Baseline',
              fontsize=FS_TITLE, fontweight='bold', pad=20)
    plt.xlabel('False Positive Rate (1 − Specificity)',
               fontsize=FS_AXIS, labelpad=12)
    plt.ylabel('True Positive Rate (Sensitivity)',
               fontsize=FS_AXIS, labelpad=12)
    plt.xticks(fontsize=FS_TICK)
    plt.yticks(fontsize=FS_TICK)
    plt.legend(loc='lower right', fontsize=FS_LEGEND,
               frameon=True, edgecolor='black')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULT_DIR, 'ROC_Publication_Quality.png')
        plt.savefig(path, dpi=300)
        print(f"[INFO] Plot 5 saved → {path}")
    plt.show()


# =============================================================================
# SECTION 7 — Bonus Plot: Fetal Heart Rate Trend with 95% CI by GA
#              (poster figure: "Fetal Heart Rate Trend 95% CI")
# =============================================================================

def plot_fhr_trend_ci(save=True):
    """
    Bonus Plot: Mean FHR ± 95% confidence interval per GA group.
    Shows the physiological decline of baseline FHR across gestation.
    Overlays the dynamic regression line (Baseline = -0.33×GA + 150.21).

    Output: FHR_Trend_95CI.png
    """
    full_data = load_fhr_only()
    if full_data.empty:
        print("[SKIP] Bonus Plot: no data available.")
        return

    # Compute mean and 95% CI per GA group
    ga_stats = (
        full_data.groupby('ga')['fhr']
        .agg(
            mean='mean',
            std='std',
            count='count'
        )
        .reset_index()
    )
    # 95% CI = 1.96 * (std / sqrt(n))
    ga_stats['ci95'] = 1.96 * ga_stats['std'] / np.sqrt(ga_stats['count'])

    ga_range  = np.linspace(ga_stats['ga'].min(), ga_stats['ga'].max(), 200)
    reg_line  = SLOPE * ga_range + INTERCEPT

    plt.figure(figsize=(12, 7))

    # Scatter: mean FHR per GA
    plt.errorbar(ga_stats['ga'], ga_stats['mean'],
                 yerr=ga_stats['ci95'],
                 fmt='o', color='steelblue', ecolor='lightblue',
                 capsize=4, label='Mean FHR ± 95% CI', zorder=3)

    # Regression line from the dynamic baseline equation
    plt.plot(ga_range, reg_line, color='red', linewidth=2.5, linestyle='--',
             label=f'Regression: FHR = {SLOPE}×GA + {INTERCEPT}')

    # Shade the ±25 bpm normal band around the regression line
    plt.fill_between(ga_range, reg_line - 25, reg_line + 25,
                     color='orange', alpha=0.15, label='±25 bpm Normal Band')

    plt.title("Fetal Heart Rate Trend with 95% Confidence Interval by GA",
              fontsize=FS_TITLE, fontweight='bold')
    plt.xlabel("Gestational Age (GA weeks)", fontsize=FS_AXIS)
    plt.ylabel("FHR (bpm)",                  fontsize=FS_AXIS)
    plt.xticks(ga_stats['ga'].values,         fontsize=FS_TICK)
    plt.yticks(fontsize=FS_TICK)
    plt.ylim(80, 180)
    plt.legend(fontsize=FS_LEGEND)
    plt.grid(axis='both', linestyle=':', alpha=0.5)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULT_DIR, 'FHR_Trend_95CI.png')
        plt.savefig(path, dpi=300)
        print(f"[INFO] Bonus plot saved → {path}")
    plt.show()


# =============================================================================
# SECTION 8 — Master Runner
# =============================================================================

def plot_all():
    """
    Generate all plots in sequence.
    Each plot saves a PNG to RESULT_DIR automatically.

    Order:
      1. FHR Density Distribution by GA
      2. FHR Linear Trend + 50 bpm Band
      3. Alarm Rate — 3-Method Bar Chart
      4. Alarm Rate — 4-Method Bar Chart (with New Threshold)
      5. ROC Curve — RF vs Baseline (requires Excel from validation.py)
      6. FHR Trend with 95% CI (bonus)
    """
    print("=" * 55)
    print("  Generating all visualisation plots...")
    print("=" * 55)

    print("\n[Plot 1] FHR Density Distribution by GA")
    plot_fhr_density_by_ga()

    print("\n[Plot 2] FHR Linear Trend with 50 bpm Band")
    plot_fhr_linear_trend()

    print("\n[Plot 3] Alarm Rate — 3-Method Comparison")
    plot_alarm_rate_3method()

    print("\n[Plot 4] Alarm Rate — 4-Method Comparison (with New Threshold)")
    plot_alarm_rate_4method()

    print("\n[Plot 5] ROC Curve — Random Forest vs. Linear Baseline")
    plot_roc_curve()

    print("\n[Plot 6] FHR Trend with 95% Confidence Interval")
    plot_fhr_trend_ci()

    print("\n[DONE] All plots generated and saved to:")
    print(f"       {RESULT_DIR}")


# =============================================================================
# SECTION 9 — Entry Point
# =============================================================================

if __name__ == "__main__":
    plot_all()
