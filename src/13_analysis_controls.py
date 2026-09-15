
"""
STEP 13 — REQUIRED ANALYSIS CONTROLS

13A. Shuffled-label control
13B. Detection-vs-prediction control
13C. Future-horizon sweep

These are analysis/control experiments only.
They do not modify the frozen final test results.

Outputs:
    outputs/analysis_controls/
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path("/content/drive/MyDrive/ecg_temporal_ai")
FEATURE_DIR = ROOT / "outputs" / "features"
WINDOW_DIR = ROOT / "outputs" / "windowing"
OUT_DIR = ROOT / "outputs" / "analysis_controls"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

TASK_A_FILE = FEATURE_DIR / "task_a_hrv_features.csv"
TASK_B_FILE = FEATURE_DIR / "task_b_hrv_features.csv"

TASK_A_WINDOWS = WINDOW_DIR / "task_a_windows.csv"

TRAIN = "train"
VAL = "validation"
TEST = "test"


# =============================================================================
# HELPERS
# =============================================================================

def build_task_b_X(df):
    """
    Use exactly the five historical HRV minutes already constructed by
    the existing feature pipeline.

    Eight core HR/HRV features per minute:
        mean_rr_sec
        median_rr_sec
        mean_hr_bpm
        median_hr_bpm
        sdnn_sec
        rmssd_sec
        pnn50
        cv_rr
    """

    base_features = [
        "mean_rr_sec",
        "median_rr_sec",
        "mean_hr_bpm",
        "median_hr_bpm",
        "sdnn_sec",
        "rmssd_sec",
        "pnn50",
        "cv_rr",
    ]

    columns = []

    for lag in ["tminus5", "tminus4", "tminus3", "tminus2", "tminus1"]:
        columns.extend([f"{f}_{lag}" for f in base_features])

    X = df[columns].copy()
    return X, columns


def fit_logistic(X_train, y_train, seed=42):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )),
    ])

    model.fit(X_train, y_train)
    return model


def get_threshold(y_val, p_val):
    """
    Select threshold on validation only.
    F1 is used here only for the analysis controls.
    The original final-model thresholds remain unchanged.
    """

    thresholds = np.linspace(0.01, 0.99, 199)

    best_threshold = 0.5
    best_f1 = -1

    for t in thresholds:
        pred = (p_val >= t).astype(int)
        score = f1_score(y_val, pred, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = t

    return best_threshold, best_f1


def metric_row(y_true, p, threshold):

    pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1]
    ).ravel()

    prevalence = float(np.mean(y_true))

    return {
        "n": len(y_true),
        "positives": int(np.sum(y_true)),
        "prevalence": prevalence,
        "auroc": roc_auc_score(y_true, p)
        if len(np.unique(y_true)) > 1 else np.nan,
        "auprc": average_precision_score(y_true, p),
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("STEP 13 — REQUIRED ANALYSIS CONTROLS")
print("=" * 80)

task_a = pd.read_csv(TASK_A_FILE)
task_b = pd.read_csv(TASK_B_FILE)
task_a_windows = pd.read_csv(TASK_A_WINDOWS)

print("\nLoaded:")
print("Task A:", task_a.shape)
print("Task B:", task_b.shape)


# =============================================================================
# COMMON TASK B DATA
# =============================================================================

task_b = task_b[
    task_b["primary_hrv_analysis"].astype(int) == 1
].copy()

X_b, b_features = build_task_b_X(task_b)
y_b = task_b["target"].astype(int)

train_mask = task_b["split"] == TRAIN
val_mask = task_b["split"] == VAL
test_mask = task_b["split"] == TEST

X_train = X_b.loc[train_mask]
y_train = y_b.loc[train_mask]

X_val = X_b.loc[val_mask]
y_val = y_b.loc[val_mask]

X_test = X_b.loc[test_mask]
y_test = y_b.loc[test_mask]

print("\nTask B complete-valid data:")
print("Train:", X_train.shape, "positive:", int(y_train.sum()))
print("Val:  ", X_val.shape, "positive:", int(y_val.sum()))
print("Test: ", X_test.shape, "positive:", int(y_test.sum()))


# =============================================================================
# 13A — SHUFFLED LABEL CONTROL
# =============================================================================

print("\n")
print("=" * 80)
print("13A — SHUFFLED-LABEL CONTROL")
print("=" * 80)

rng = np.random.default_rng(SEED)

y_train_shuffled = rng.permutation(y_train.to_numpy())

shuffled_model = fit_logistic(
    X_train,
    y_train_shuffled,
    seed=SEED
)

p_val_shuffled = shuffled_model.predict_proba(X_val)[:, 1]
p_test_shuffled = shuffled_model.predict_proba(X_test)[:, 1]

shuffled_threshold, shuffled_val_f1 = get_threshold(
    y_val,
    p_val_shuffled
)

val_metrics = metric_row(
    y_val,
    p_val_shuffled,
    shuffled_threshold
)

test_metrics = metric_row(
    y_test,
    p_test_shuffled,
    shuffled_threshold
)

shuffled_results = pd.DataFrame([
    {
        "experiment": "shuffled_labels",
        "split": "validation",
        **val_metrics,
    },
    {
        "experiment": "shuffled_labels",
        "split": "test",
        **test_metrics,
    }
])

shuffled_results.to_csv(
    OUT_DIR / "shuffled_label_control.csv",
    index=False
)

print(shuffled_results.to_string(index=False))

print(
    "\nExpected behavior: AUROC should be near 0.5 and "
    "AUPRC should be near the positive prevalence."
)


# =============================================================================
# 13B — DETECTION VS PREDICTION CONTROL
# =============================================================================

print("\n")
print("=" * 80)
print("13B — DETECTION VS PREDICTION CONTROL")
print("=" * 80)

# -------------------------------------------------------------------------
# Train the ORIGINAL Task A Logistic detector on TRAIN only.
# -------------------------------------------------------------------------

task_a_valid = task_a[
    task_a["primary_hrv_analysis"].astype(int) == 1
].copy()

# Current-minute HRV features used in the Task A baseline
a_features = [
    "mean_rr_sec",
    "median_rr_sec",
    "mean_hr_bpm",
    "median_hr_bpm",
    "sdnn_sec",
    "rmssd_sec",
    "pnn50",
    "cv_rr",
]

X_a = task_a_valid[a_features]
y_a = task_a_valid["target"].astype(int)

a_train = task_a_valid["split"] == TRAIN
a_val = task_a_valid["split"] == VAL
a_test = task_a_valid["split"] == TEST

detector = fit_logistic(
    X_a.loc[a_train],
    y_a.loc[a_train],
    seed=SEED
)

p_a_val = detector.predict_proba(X_a.loc[a_val])[:, 1]
det_threshold, det_val_f1 = get_threshold(
    y_a.loc[a_val],
    p_a_val
)

# -------------------------------------------------------------------------
# Apply the CURRENT-EVENT detector to the FUTURE TARGET MINUTE of Task B.
#
# This is intentionally NOT a valid prospective prediction.
# It is an oracle/current-event detection control.
# -------------------------------------------------------------------------

task_b_control = task_b.copy()

task_b_control["target_minute"] = task_b_control[
    "target_minute"
].astype(int)

# Map Task-A current-minute HRV features by:
# subject_id + minute_index
a_lookup = task_a_valid[
    ["subject_id", "minute_index"] + a_features
].drop_duplicates(
    ["subject_id", "minute_index"]
)

a_lookup = a_lookup.rename(
    columns={"minute_index": "target_minute"}
)

control = task_b_control[
    [
        "subject_id",
        "split",
        "target_minute",
        "previous_label",
        "current_label",
        "target",
    ]
].merge(
    a_lookup,
    on=["subject_id", "target_minute"],
    how="left",
    validate="one_to_one",
)

control_valid = control[a_features].notna().all(axis=1)

control = control.loc[control_valid].copy()

# Only evaluate records belonging to the corresponding split.
for split_name in [TRAIN, VAL, TEST]:

    sub = control[control["split"] == split_name].copy()

    p = detector.predict_proba(sub[a_features])[:, 1]

    result = metric_row(
        sub["current_label"].astype(int),
        p,
        det_threshold
    )

    result["experiment"] = "current_event_detection_control"
    result["split"] = split_name

    # Save per-window predictions
    pred_df = sub[
        [
            "subject_id",
            "target_minute",
            "previous_label",
            "current_label",
            "target",
        ]
    ].copy()

    pred_df["probability_current_apnea"] = p
    pred_df["prediction_current_apnea"] = (
        p >= det_threshold
    ).astype(int)

    pred_df.to_csv(
        OUT_DIR / f"detection_control_{split_name}.csv",
        index=False
    )

    if split_name == TRAIN:
        detection_train_result = result
    elif split_name == VAL:
        detection_val_result = result
    else:
        detection_test_result = result

detection_summary = pd.DataFrame([
    detection_train_result,
    detection_val_result,
    detection_test_result,
])

detection_summary.to_csv(
    OUT_DIR / "detection_vs_prediction_control.csv",
    index=False
)

print(detection_summary.to_string(index=False))

print(
    "\nInterpretation: this control asks whether current apnea is "
    "detectable from ECG/HRV when the target minute itself is visible. "
    "It is deliberately NOT a prospective forecasting model."
)


# =============================================================================
# 13C — HORIZON SWEEP
# =============================================================================

print("\n")
print("=" * 80)
print("13C — FUTURE HORIZON SWEEP")
print("=" * 80)

# We construct future-onset labels from the original minute annotations.
#
# For prediction minute m:
#
#   previous_label at m-1 MUST be 0.
#
# Positive for horizon H iff:
#
#   an N -> A transition occurs at minute k
#   where m <= k < m+H.
#
# This prevents an already-running apnea episode from becoming a
# "future prediction" solely because it continues into the horizon.

annotation = task_a_windows[
    ["subject_id", "minute_index", "symbol", "target"]
].copy()

annotation["minute_index"] = annotation["minute_index"].astype(int)
annotation["target"] = annotation["target"].astype(int)

# Organize minute labels by record.
record_annotations = {}

for subject_id, g in annotation.groupby("subject_id"):

    g = g.sort_values("minute_index").copy()

    label_map = dict(
        zip(
            g["minute_index"].astype(int),
            g["target"].astype(int)
        )
    )

    record_annotations[subject_id] = label_map


def future_onset_label(subject_id, prediction_minute, horizon, label_map):

    m = int(prediction_minute)

    previous = label_map.get(m - 1, None)

    # Prediction is only defined if there is a known non-apnea
    # state immediately before prediction time.
    if previous is None or previous != 0:
        return np.nan

    for k in range(m, m + horizon):

        current = label_map.get(k, None)

        if current is None:
            # Future annotation unavailable.
            return np.nan

        # New onset
        if label_map.get(k - 1, None) == 0 and current == 1:
            return 1

    return 0


horizon_rows = []

for horizon in [1, 2, 3, 5]:

    rows = task_b.copy()

    future_labels = []

    for _, row in rows.iterrows():

        subject_id = row["subject_id"]
        m = int(row["target_minute"])

        label_map = record_annotations[subject_id]

        y_future = future_onset_label(
            subject_id,
            m,
            horizon,
            label_map
        )

        future_labels.append(y_future)

    rows[f"target_h{horizon}"] = future_labels

    rows = rows[
        rows[f"target_h{horizon}"].notna()
        & rows["history_complete_valid"].astype(bool)
    ].copy()

    y = rows[f"target_h{horizon}"].astype(int)

    X, _ = build_task_b_X(rows)

    tr = rows["split"] == TRAIN
    va = rows["split"] == VAL
    te = rows["split"] == TEST

    # Need both classes for fitting.
    model = fit_logistic(
        X.loc[tr],
        y.loc[tr],
        seed=SEED
    )

    p_val = model.predict_proba(X.loc[va])[:, 1]

    threshold, val_f1 = get_threshold(
        y.loc[va],
        p_val
    )

    p_test = model.predict_proba(X.loc[te])[:, 1]

    val_result = metric_row(
        y.loc[va],
        p_val,
        threshold
    )

    test_result = metric_row(
        y.loc[te],
        p_test,
        threshold
    )

    horizon_rows.append({
        "horizon_min": horizon,
        "split": "validation",
        **val_result,
    })

    horizon_rows.append({
        "horizon_min": horizon,
        "split": "test",
        **test_result,
    })

    # Save predictions for reproducibility.
    pred = rows.loc[te, [
        "subject_id",
        "target_minute",
        "prediction_time_sec",
        "future_end_sec",
    ]].copy()

    pred["horizon_min"] = horizon
    pred["target"] = y.loc[te].to_numpy()
    pred["probability"] = p_test
    pred["threshold"] = threshold
    pred["prediction"] = (
        p_test >= threshold
    ).astype(int)

    pred.to_csv(
        OUT_DIR / f"horizon_{horizon}min_test_predictions.csv",
        index=False
    )

horizon_results = pd.DataFrame(horizon_rows)

horizon_results.to_csv(
    OUT_DIR / "horizon_sweep_results.csv",
    index=False
)

print(
    horizon_results[
        [
            "horizon_min",
            "split",
            "n",
            "positives",
            "prevalence",
            "auroc",
            "auprc",
            "threshold",
            "precision",
            "recall",
            "f1",
        ]
    ].to_string(index=False)
)


# =============================================================================
# FINAL CONTROL SUMMARY
# =============================================================================

summary_lines = []

summary_lines.append(
    "STEP 13 ANALYSIS CONTROLS SUMMARY"
)

summary_lines.append("=" * 70)

summary_lines.append(
    "\n13A SHUFFLED LABELS:\n"
    "Training labels were randomly permuted while features and the "
    "record-level split were retained."
)

summary_lines.append(
    "\n13B DETECTION VS PREDICTION:\n"
    "The Task A current-event detector was evaluated on Task B target "
    "minutes using the target-minute HRV features. This is an oracle/current-"
    "event detection control and is not a prospective forecasting model."
)

summary_lines.append(
    "\n13C HORIZON SWEEP:\n"
    "Future onset was defined using N->A transitions occurring after the "
    "prediction time, while requiring the state immediately before prediction "
    "time to be non-apnea."
)

summary_lines.append(
    "\nOutputs:\n"
    + str(OUT_DIR)
)

with open(OUT_DIR / "ANALYSIS_CONTROLS_SUMMARY.txt", "w") as f:
    f.write("\n".join(summary_lines))

print("\n")
print("=" * 80)
print("STEP 13A–13C COMPLETE")
print("=" * 80)
print("Outputs:", OUT_DIR)
