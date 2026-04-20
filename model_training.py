# =============================================================================
# model_training.py
# FHR Fetal Distress Prediction — Full Training Pipeline
#
# Pipeline overview (must run in this order):
#
#   STEP 1 — Load all per-GA CSV recordings and build a combined DataFrame
#   STEP 2 — Compute FHR density distribution per GA group
#             Find the 50 bpm window with the highest point density per GA
#   STEP 3 — Fit linear regression: FHR ~ GA across the full dataset
#             Align the regression line to the centre of the densest residual
#             window → this yields the dynamic baseline equation:
#                 Baseline FHR = SLOPE × GA + INTERCEPT
#             SLOPE and INTERCEPT are derived from the data here, not assumed.
#   STEP 4 — Parse clinical variables from WFDB .hea files
#   STEP 5 — Extract FHR signal features using the derived baseline
#   STEP 6 — Train Random Forest classifier on combined features + clinical vars
#   STEP 7 — Save trained model as fetal_rf_model.mod (joblib binary)
#   STEP 8 — New patient prediction interface
#
# Authors: Rui Chen, Jiayi Wang
# Dataset: CTU-UHB (552 CTG, GA>36) + Trium private (1691 CTG, GA>=24)
# =============================================================================

import pandas as pd
import numpy as np
import os
import re
import wfdb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ── Directory paths ────────────────────────────────────────────────────────────
# DATA_DIR   : per-GA CSV files (columns: time, fhr) — used in Steps 1–3
# RESULT_DIR : WFDB .hea + signal files (clinical records) — used in Steps 4–7
DATA_DIR   = r"/Users/jiayi/PycharmProjects/pythonProject9/venv/data"
RESULT_DIR = r"/Users/jiayi/PycharmProjects/pythonProject9/venv/result"

# Model is saved with .mod extension (joblib binary format)
MODEL_PATH = os.path.join(RESULT_DIR, "fetal_rf_model.mod")

# Band half-width around the dynamic baseline (±25 bpm = 50 bpm total window)
BAND_HALF = 25


# =============================================================================
# STEP 1 — Load Per-GA CSV Data
# =============================================================================

def extract_ga_from_filename(filename):
    """Parse GA week value from filename pattern, e.g. 'GA_36.csv' → 36."""
    m = re.search(r'GA_(\d+)', filename)
    return int(m.group(1)) if m else None


def load_csv_data():
    """
    Read all per-GA CSV files from DATA_DIR.
    Each file contains continuous FHR recordings for one gestational age group.
    Filters out physiologically implausible values (outside 50–200 bpm).

    Returns a combined DataFrame with columns: time, fhr, ga
    """
    all_dfs = []
    files   = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

    for file in files:
        ga = extract_ga_from_filename(file)
        if ga is None:
            continue
        try:
            # Only load the two columns we need to save memory
            df = pd.read_csv(os.path.join(DATA_DIR, file), usecols=['time', 'fhr'])
            df = df[(df['fhr'] >= 50) & (df['fhr'] <= 200)].copy()
            df['ga'] = ga
            all_dfs.append(df)
        except Exception:
            continue

    if not all_dfs:
        raise FileNotFoundError(f"[ERROR] No valid CSV files found in {DATA_DIR}")

    full_data = pd.concat(all_dfs, ignore_index=True)
    print(f"[STEP 1] Loaded {len(full_data):,} FHR samples across "
          f"{full_data['ga'].nunique()} GA groups.")
    return full_data


# =============================================================================
# STEP 2 — FHR Density Distribution per GA Group
# =============================================================================

def find_max_density_range(data, window=50):
    """
    Sliding-window search over sorted FHR values.
    Find the interval of width 'window' bpm that contains the most data points.

    This identifies the physiologically normal FHR zone for a given GA group
    without assuming a fixed range (unlike the static 110–160 bpm threshold).

    Returns (start, end) of the densest interval in bpm.
    """
    if len(data) == 0:
        return (None, None)

    sorted_data = np.sort(data)
    max_count   = 0
    best_range  = (sorted_data[0], sorted_data[0] + window)
    right       = 0

    for left in range(len(sorted_data)):
        # Advance right pointer while within the window width
        while right < len(sorted_data) and sorted_data[right] <= sorted_data[left] + window:
            right += 1
        if (right - left) > max_count:
            max_count  = right - left
            best_range = (sorted_data[left], sorted_data[left] + window)

    return best_range


def compute_density_intervals(full_data):
    """
    For each GA group, find the 50 bpm window with the highest FHR density.
    Prints the intervals to console and returns a dict: {ga: (start, end)}.

    These intervals reveal how the 'normal' FHR zone shifts with gestational age,
    motivating the dynamic baseline approach over fixed 110–160 bpm.
    """
    ga_groups         = full_data.groupby('ga')
    density_intervals = {}

    print(f"\n[STEP 2] Max-density intervals per GA group (window = 50 bpm)")
    print("-" * 40)
    for ga, group in ga_groups:
        start, end = find_max_density_range(group['fhr'].values, window=50)
        density_intervals[ga] = (start, end)
        print(f"  GA {ga:>2}: [{start:.1f} – {end:.1f}] bpm")

    return density_intervals


# =============================================================================
# STEP 3 — Derive Dynamic Baseline: Linear Regression + Density-Centre Offset
# =============================================================================

def find_best_offset_for_window(residuals, window=50):
    """
    Given the residuals from a linear regression (FHR ~ GA), find the centre
    of the 50 bpm window that captures the most residuals.

    This shifts the regression line so it passes through the densest region
    of the data rather than the mean, making the baseline more physiologically
    representative of the normal FHR zone.

    Returns the optimal centre offset in bpm.
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


def derive_dynamic_baseline(full_data):
    """
    Fit a linear regression of FHR on GA across all recordings.
    Shift the fitted line to the centre of the densest 50 bpm residual window.

    This produces the dynamic baseline equation:
        Baseline FHR(GA) = SLOPE × GA + EFFECTIVE_INTERCEPT

    where EFFECTIVE_INTERCEPT = regression intercept + density centre offset.

    Returns (slope, intercept) as floats.
    These values are then used throughout feature extraction and classification.
    """
    X = full_data['ga'].values.reshape(-1, 1)
    y = full_data['fhr'].values

    model     = LinearRegression().fit(X, y)
    residuals = y - model.predict(X)

    # Shift intercept so the line runs through the densest FHR zone
    center_offset      = find_best_offset_for_window(residuals, window=50)
    effective_intercept = model.intercept_ + center_offset

    slope = float(model.coef_[0])
    intercept = float(effective_intercept)

    print(f"\n[STEP 3] Linear regression fit:  FHR = {model.intercept_:.4f} + "
          f"({slope:.4f}) × GA")
    print(f"         Density centre offset:   {center_offset:+.4f} bpm")
    print(f"         Dynamic baseline eq:     FHR = {intercept:.2f} + "
          f"({slope:.2f}) × GA")
    print(f"         (Normal band = baseline ± {BAND_HALF} bpm)")

    return slope, intercept


# =============================================================================
# STEP 4 — Clinical Data Parsing from WFDB .hea Files
# =============================================================================

def parse_hea_comprehensive(file_path):
    """
    Extract all clinical variables from a WFDB .hea header file.
    Uses regex to find labelled fields in the header comment section.
    Returns a dict; missing fields default to safe physiological values.

    Fields extracted: ph, ga, be, age, meconium, hypertension,
                      diabetes, preeclampsia
    """
    fields = {
        'ph': np.nan, 'ga': 40, 'be': 0, 'age': 30,
        'meconium': 0, 'hypertension': 0, 'diabetes': 0, 'preeclampsia': 0
    }
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            c = f.read()
            # Continuous clinical variables
            fields['ph']  = float(m.group(1)) if (m := re.search(r'#\s*pH\s+([\d.]+)', c))          else np.nan
            fields['ga']  = int(m.group(1))   if (m := re.search(r'#\s*Gest\.\s*weeks\s+(\d+)', c)) else 40
            fields['be']  = float(m.group(1)) if (m := re.search(r'#\s*BE\s+([-\d.]+)', c))         else 0
            fields['age'] = int(m.group(1))   if (m := re.search(r'#\s*Age\s+(\d+)', c))            else 30
            # Binary obstetric risk flags
            fields['meconium']     = 1 if re.search(r'#\s*Meconium\s+1',     c) else 0
            fields['hypertension'] = 1 if re.search(r'#\s*Hypertension\s+1', c) else 0
            fields['diabetes']     = 1 if re.search(r'#\s*Diabetes\s+1',     c) else 0
            fields['preeclampsia'] = 1 if re.search(r'#\s*Preeclampsia\s+1', c) else 0
    except Exception:
        pass
    return fields


# =============================================================================
# STEP 5 — Signal Feature Extraction (uses derived baseline)
# =============================================================================

def extract_signal_features(rid, ga, slope, intercept):
    """
    Load a WFDB signal file and compute five FHR-derived features.
    The dynamic baseline is evaluated at the record's GA using the
    slope and intercept derived in Step 3.

    Features
    --------
    LTV_STD         : Long-term variability (std of FHR over recording)
    STV             : Short-term variability (mean |diff| of consecutive samples)
    In_Band_Ratio   : Fraction of time FHR stays within baseline ± BAND_HALF bpm
    Dec_Area        : Cumulative area of decelerations below (baseline − 20 bpm)
    Tachycardia_Flag: 1 if mean FHR > 160 bpm (possible tachycardia)

    Returns None if signal is unreadable or has fewer than 100 valid samples.
    """
    try:
        signals, _ = wfdb.rdsamp(os.path.join(RESULT_DIR, rid))
        fhr = signals[:, 0]
        # Remove noise and artefacts
        fhr = fhr[(fhr >= 50) & (fhr <= 200)]
        if len(fhr) < 100:
            return None

        # GA-specific expected baseline from the derived equation
        base  = slope * ga + intercept
        diffs = np.abs(np.diff(fhr))

        return {
            'LTV_STD'         : np.std(fhr),
            'STV'             : np.mean(diffs),
            'In_Band_Ratio'   : ((fhr >= base - BAND_HALF) & (fhr <= base + BAND_HALF)).sum() / len(fhr),
            'Dec_Area'        : np.sum(base - fhr[fhr < (base - 20)]) / len(fhr),
            'Tachycardia_Flag': 1 if np.mean(fhr) > 160 else 0
        }
    except Exception:
        return None


# =============================================================================
# STEP 6 — Build Training Dataset
# =============================================================================

# Feature columns (must remain identical between training and inference)
FEATURES = [
    'ga', 'be', 'meconium', 'age', 'hypertension', 'diabetes', 'preeclampsia',
    'LTV_STD', 'STV', 'In_Band_Ratio', 'Dec_Area', 'Tachycardia_Flag'
]


def build_dataset(slope, intercept):
    """
    Scan RESULT_DIR for all .hea records.
    For each record: parse clinical fields, extract signal features,
    and assign a binary label (1 = Pathological if pH < 7.15).

    The derived slope and intercept are passed in so feature extraction
    uses the same baseline that was computed from the distribution data.

    Returns a combined DataFrame ready for model training.
    """
    hea_files = [f for f in os.listdir(RESULT_DIR) if f.lower().endswith('.hea')]
    print(f"\n[STEP 6] Building dataset from {len(hea_files)} records...")

    all_data = []
    for h in hea_files:
        rid  = h.replace('.hea', '')
        clin = parse_hea_comprehensive(os.path.join(RESULT_DIR, h))
        sig  = extract_signal_features(rid, clin['ga'], slope, intercept)

        # Skip records with missing pH or unusable signals
        if sig is None or np.isnan(clin['ph']):
            continue

        row = {**clin, **sig, 'ID': rid}
        all_data.append(row)

    df = pd.DataFrame(all_data)
    df['Label'] = (df['ph'] < 7.15).astype(int)  # ground truth: pH < 7.15 → pathological

    print(f"         Valid records : {len(df)}")
    print(f"         Pathological  : {df['Label'].sum()} ({df['Label'].mean():.1%})")
    print(f"         Normal        : {(df['Label']==0).sum()} ({(df['Label']==0).mean():.1%})")
    return df


# =============================================================================
# STEP 7 — Train Random Forest & Save Model
# =============================================================================

def train_and_save_model(df):
    """
    Train a Random Forest classifier on the combined feature set.
    Uses class_weight='balanced' to handle the natural class imbalance
    (pathological cases are a minority in CTG datasets).

    Saves the trained model to MODEL_PATH (.mod extension, joblib format).
    Prints test set metrics and saves a feature importance plot.

    Returns (model, importance_df)
    """
    X = df[FEATURES]
    y = df['Label']

    # Stratified 75/25 split preserves pathological case ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',  # compensate for low prevalence of acidosis
        random_state=42
    )
    rf.fit(X_train, y_train)

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    y_pred = rf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n[STEP 7] Training complete.")
    print(f"         Test Accuracy  : {test_acc:.2%}")
    print(f"         Sensitivity    : {sensitivity:.2%}  (TP / (TP+FN))")
    print(f"         Specificity    : {specificity:.2%}  (TN / (TN+FP))")

    # ── Save model as .mod (joblib serialised) ────────────────────────────────
    joblib.dump(rf, MODEL_PATH)
    print(f"         Model saved    → {MODEL_PATH}")

    # ── Feature importance ────────────────────────────────────────────────────
    importance = (
        pd.DataFrame({'Feature': FEATURES, 'Importance': rf.feature_importances_})
        .sort_values('Importance', ascending=False)
    )
    plot_feature_importance(importance)

    return rf, importance


def plot_feature_importance(importance_df):
    """
    Horizontal bar chart of Random Forest feature importance scores.
    Higher score = stronger contribution to pathological outcome prediction.
    Saved as Feature_Importance_Plot.png.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Feature Contribution to Fetal Outcome Prediction',
              fontsize=15, fontweight='bold')
    plt.xlabel('Importance Score (Mean Decrease in Impurity)', fontsize=12)
    plt.ylabel('Clinical & Signal Indicators', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, 'Feature_Importance_Plot.png')
    plt.savefig(save_path, dpi=300)
    print(f"         Importance plot → {save_path}")
    plt.show()


# =============================================================================
# STEP 8 — New Patient Prediction Interface
# =============================================================================

def predict_new_patient(fhr_array, clinical_info, slope, intercept):
    """
    Run inference for a new patient using the saved .mod model.
    The same slope and intercept derived from the distribution must be passed in
    to ensure feature extraction is consistent with training.

    Parameters
    ----------
    fhr_array     : np.ndarray  — raw FHR signal (bpm values)
    clinical_info : dict        — required key: 'ga'
                                  optional: 'id', 'age', 'be', 'meconium',
                                  'hypertension', 'diabetes', 'preeclampsia'
    slope         : float       — derived in Step 3
    intercept     : float       — derived in Step 3

    Prints a clinical risk report. Decision threshold = 0.40 (tuned for
    higher sensitivity to avoid missing true pathological cases).
    """
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model file not found. Run the training pipeline first.")
        return

    model = joblib.load(MODEL_PATH)

    # Clean signal the same way as during training
    fhr = np.asarray(fhr_array, dtype=float)
    fhr = fhr[(fhr >= 50) & (fhr <= 200)]
    if len(fhr) < 100:
        print("[ERROR] Fewer than 100 valid FHR samples — cannot predict.")
        return

    ga   = clinical_info.get('ga', 40)
    base = slope * ga + intercept

    # Feature vector must exactly match the FEATURES list used in training
    features = pd.DataFrame([{
        'ga'             : ga,
        'be'             : clinical_info.get('be', 0),
        'meconium'       : clinical_info.get('meconium', 0),
        'age'            : clinical_info.get('age', 30),
        'hypertension'   : clinical_info.get('hypertension', 0),
        'diabetes'       : clinical_info.get('diabetes', 0),
        'preeclampsia'   : clinical_info.get('preeclampsia', 0),
        'LTV_STD'        : np.std(fhr),
        'STV'            : np.mean(np.abs(np.diff(fhr))),
        'In_Band_Ratio'  : np.mean((fhr >= base - BAND_HALF) & (fhr <= base + BAND_HALF)),
        'Dec_Area'       : np.sum(base - fhr[fhr < (base - 20)]) / len(fhr),
        'Tachycardia_Flag': 1 if np.mean(fhr) > 160 else 0
    }])

    prob   = model.predict_proba(features)[0, 1]
    result = "PATHOLOGICAL (High Risk)" if prob > 0.40 else "NORMAL (Low Risk)"

    print("\n" + "=" * 50)
    print(f"  FETAL RISK REPORT  |  {clinical_info.get('ID', 'Unknown')}")
    print("=" * 50)
    print(f"  Assessment              : {result}")
    print(f"  Probability of Acidosis : {prob:.2%}")
    print(f"  GA (weeks)              : {ga}")
    print(f"  Dynamic Baseline        : {base:.1f} bpm  "
          f"(band {base - BAND_HALF:.1f} – {base + BAND_HALF:.1f})")
    print(f"  STV                     : {features['STV'][0]:.4f} bpm")
    print(f"  In-Band Ratio           : {features['In_Band_Ratio'][0]:.2%}")
    print(f"  Tachycardia Flag        : {int(features['Tachycardia_Flag'][0])}")
    print("=" * 50)


# =============================================================================
# Entry Point — Run Full Pipeline
# =============================================================================

if __name__ == "__main__":

    # STEP 1: Load all per-GA CSV data
    full_data = load_csv_data()

    # STEP 2: Compute density intervals per GA (shows how normal zone shifts)
    density_intervals = compute_density_intervals(full_data)

    # STEP 3: Derive dynamic baseline equation from the data
    #         slope and intercept are outputs here, not assumptions
    slope, intercept = derive_dynamic_baseline(full_data)

    # STEP 4–5 happen inside build_dataset (parsing + feature extraction)
    # STEP 6: Build labelled training dataset
    dataset = build_dataset(slope, intercept)

    # STEP 7: Train model and save as .mod
    rf_model, feat_importance = train_and_save_model(dataset)

    # STEP 8: Example inference — replace fhr_array with real patient data
    example_fhr = np.random.normal(loc=140, scale=10, size=2400).clip(50, 200)
    example_info = {
        'ID'           : 'P_DEMO_001',
        'ga'           : 28,
        'age'          : 35,
        'be'           : -2.0,
        'meconium'     : 0,
        'hypertension' : 0,
        'diabetes'     : 1,
        'preeclampsia' : 0
    }
    predict_new_patient(example_fhr, example_info, slope, intercept)
