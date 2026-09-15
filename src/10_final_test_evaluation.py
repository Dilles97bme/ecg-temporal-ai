

import os
import random
import warnings
import copy

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from IPython.display import display

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    brier_score_loss,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE = "/content/drive/MyDrive/ecg_temporal_ai"

FEATURE_DIR = f"{BASE}/outputs/features"
BASELINE_DIR = f"{BASE}/outputs/baseline_results"
TCN_DIR = f"{BASE}/outputs/tcn_results"
FINAL_DIR = f"{BASE}/outputs/final_test_results"

os.makedirs(FINAL_DIR, exist_ok=True)

TASK_A_PATH = (
    f"{FEATURE_DIR}/task_a_hrv_features.csv"
)

TASK_B_PATH = (
    f"{FEATURE_DIR}/task_b_hrv_features.csv"
)

BASELINE_RESULTS_PATH = (
    f"{BASELINE_DIR}/baseline_validation_results.csv"
)

TCN_RESULTS_PATH = (
    f"{TCN_DIR}/tcn_validation_results.csv"
)

SEED = 42

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# LOAD DATA
# ============================================================

task_a = pd.read_csv(TASK_A_PATH)
task_b = pd.read_csv(TASK_B_PATH)

baseline_results = pd.read_csv(
    BASELINE_RESULTS_PATH
)

tcn_results = pd.read_csv(
    TCN_RESULTS_PATH
)

print("=" * 80)
print("STEP 10 — FINAL TEST EVALUATION")
print("=" * 80)

print("\nTEST SET IS NOW BEING USED FOR FINAL EVALUATION ONLY.")

print("\nTask A:", task_a.shape)
print("Task B:", task_b.shape)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

HRV_FEATURES = [
    "mean_rr_sec",
    "median_rr_sec",
    "mean_hr_bpm",
    "median_hr_bpm",
    "sdnn_sec",
    "rmssd_sec",
    "pnn50",
    "cv_rr",
]


# ============================================================
# EXTRACT FROZEN VALIDATION THRESHOLDS
# ============================================================

def get_baseline_threshold(
    task_name,
    model_name
):
    row = baseline_results[
        (baseline_results["task"] == task_name)
        & (baseline_results["model"] == model_name)
    ]

    if len(row) != 1:
        raise ValueError(
            f"Expected exactly one baseline result for "
            f"{task_name} / {model_name}, found {len(row)}"
        )

    return float(
        row.iloc[0]["threshold"]
    )


def get_tcn_threshold(
    task_name
):
    row = tcn_results[
        tcn_results["task"] == task_name
    ]

    if len(row) != 1:
        raise ValueError(
            f"Expected exactly one TCN result for {task_name}"
        )

    return float(
        row.iloc[0]["threshold"]
    )


# Frozen thresholds selected on validation only
TASK_A_LR_THRESHOLD = get_baseline_threshold(
    "Task_A",
    "LogisticRegression"
)

TASK_B_LR_THRESHOLD = get_baseline_threshold(
    "Task_B",
    "LogisticRegression"
)

TASK_A_TCN_THRESHOLD = get_tcn_threshold(
    "Task_A"
)

TASK_B_TCN_THRESHOLD = get_tcn_threshold(
    "Task_B"
)

print("\nFrozen validation thresholds:")
print(
    "Task A Logistic Regression:",
    TASK_A_LR_THRESHOLD
)
print(
    "Task A TCN:",
    TASK_A_TCN_THRESHOLD
)
print(
    "Task B Logistic Regression:",
    TASK_B_LR_THRESHOLD
)
print(
    "Task B TCN:",
    TASK_B_TCN_THRESHOLD
)


# ============================================================
# METRICS
# ============================================================

def evaluate_test(
    y_true,
    y_prob,
    threshold,
    task_name,
    model_name
):

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    auroc = roc_auc_score(
        y_true,
        y_prob
    )

    auprc = average_precision_score(
        y_true,
        y_prob
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    brier = brier_score_loss(
        y_true,
        y_prob
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    prevalence = float(
        np.mean(y_true)
    )

    return {
        "task": task_name,
        "model": model_name,

        # Primary metric
        "auprc": auprc,

        # No-skill AUPRC = positive prevalence
        "positive_prevalence": prevalence,
        "auprc_no_skill": prevalence,

        "auroc": auroc,
        "precision": precision,
        "recall_sensitivity": recall,
        "f1": f1,
        "accuracy": accuracy,

        "brier": brier,

        "threshold": threshold,

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        "n_test": len(y_true),
        "n_positive": int(np.sum(y_true)),
        "n_negative": int(len(y_true) - np.sum(y_true)),
    }


# ============================================================
# LOGISTIC REGRESSION FACTORY
# ============================================================

def make_logistic_model():

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=SEED
            )
        )
    ])


# ============================================================
# TCN DEFINITION
# ============================================================

class TemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        dropout=0.15,
    ):
        super().__init__()

        padding = (
            (kernel_size - 1) * dilation
        ) // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.bn1 = nn.BatchNorm1d(
            out_channels
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.bn2 = nn.BatchNorm1d(
            out_channels
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

        if in_channels != out_channels:
            self.residual = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):

        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out + residual


class LightweightTCN(nn.Module):

    def __init__(
        self,
        input_features,
        hidden_channels=32,
        dropout=0.15,
    ):
        super().__init__()

        self.block1 = TemporalBlock(
            input_features,
            hidden_channels,
            dilation=1,
            dropout=dropout,
        )

        self.block2 = TemporalBlock(
            hidden_channels,
            hidden_channels,
            dilation=2,
            dropout=dropout,
        )

        self.block3 = TemporalBlock(
            hidden_channels,
            hidden_channels,
            dilation=4,
            dropout=dropout,
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                hidden_channels,
                16
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                16,
                1
            ),
        )

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.pool(x)

        return self.classifier(
            x
        ).squeeze(-1)


# ============================================================
# TASK A TCN SEQUENCE BUILDER
# ============================================================

def build_task_a_sequences(df):

    df = df[
        df["primary_hrv_analysis"] == 1
    ].copy()

    df = df.sort_values(
        ["subject_id", "minute_index"]
    )

    lookup = df.set_index(
        ["subject_id", "minute_index"]
    )

    sequences = []
    labels = []
    splits = []
    subjects = []
    target_minutes = []

    for _, row in df.iterrows():

        subject = row["subject_id"]
        target_minute = int(
            row["minute_index"]
        )

        history = []
        valid = True

        for minute in range(
            target_minute - 4,
            target_minute + 1
        ):

            key = (
                subject,
                minute
            )

            if key not in lookup.index:
                valid = False
                break

            h = lookup.loc[key]

            if int(
                h["primary_hrv_analysis"]
            ) != 1:
                valid = False
                break

            history.append(
                h[HRV_FEATURES].values.astype(
                    np.float32
                )
            )

        if not valid:
            continue

        sequences.append(
            np.stack(history)
        )

        labels.append(
            int(row["target"])
        )

        splits.append(
            row["split"]
        )

        subjects.append(
            subject
        )

        target_minutes.append(
            target_minute
        )

    return (
        np.asarray(
            sequences,
            dtype=np.float32
        ),
        np.asarray(
            labels,
            dtype=np.int64
        ),
        np.asarray(splits),
        np.asarray(subjects),
        np.asarray(target_minutes),
    )


# ============================================================
# TASK B TCN SEQUENCE BUILDER
# ============================================================

def build_task_b_sequences(df):

    df = df[
        df["primary_hrv_analysis"] == 1
    ].copy()

    df = df[
        df["history_complete_valid"] == 1
    ].copy()

    sequences = []
    labels = []
    splits = []
    subjects = []
    target_minutes = []

    for _, row in df.iterrows():

        seq = []

        for lag in [5, 4, 3, 2, 1]:

            seq.append(
                np.asarray(
                    [
                        row[
                            f"{feature}_tminus{lag}"
                        ]
                        for feature in HRV_FEATURES
                    ],
                    dtype=np.float32
                )
            )

        sequences.append(
            np.stack(seq)
        )

        labels.append(
            int(row["target"])
        )

        splits.append(
            row["split"]
        )

        subjects.append(
            row["subject_id"]
        )

        target_minutes.append(
            int(row["target_minute"])
        )

    return (
        np.asarray(
            sequences,
            dtype=np.float32
        ),
        np.asarray(
            labels,
            dtype=np.int64
        ),
        np.asarray(splits),
        np.asarray(subjects),
        np.asarray(target_minutes),
    )


# ============================================================
# TCN TEST PREDICTION
# ============================================================

def tcn_predict(
    model,
    scaler,
    X
):

    X_scaled = scaler.transform(
        X.reshape(
            -1,
            X.shape[-1]
        )
    ).reshape(
        X.shape
    )

    X_tensor = torch.tensor(
        X_scaled.transpose(0, 2, 1),
        dtype=torch.float32
    ).to(DEVICE)

    model.eval()

    with torch.no_grad():

        logits = model(
            X_tensor
        )

        prob = torch.sigmoid(
            logits
        ).cpu().numpy()

    return prob


# ============================================================
# TASK A — LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 80)
print("TASK A — FINAL TEST")
print("=" * 80)

a = task_a[
    task_a["primary_hrv_analysis"] == 1
].copy()

train_mask = (
    a["split"] == "train"
)

test_mask = (
    a["split"] == "test"
)

X_train_a = a.loc[
    train_mask,
    HRV_FEATURES
]

y_train_a = a.loc[
    train_mask,
    "target"
].astype(int)

X_test_a = a.loc[
    test_mask,
    HRV_FEATURES
]

y_test_a = a.loc[
    test_mask,
    "target"
].astype(int)

model_a_lr = make_logistic_model()

# Refit on TRAIN ONLY using frozen model configuration.
model_a_lr.fit(
    X_train_a,
    y_train_a
)

prob_a_lr = model_a_lr.predict_proba(
    X_test_a
)[:, 1]

result_a_lr = evaluate_test(
    y_true=y_test_a.values,
    y_prob=prob_a_lr,
    threshold=TASK_A_LR_THRESHOLD,
    task_name="Task_A",
    model_name="LogisticRegression",
)

print("\nTask A Logistic Regression")
print(result_a_lr)


# ============================================================
# TASK A — TCN
# ============================================================

Xa, ya, sa, subj_a, min_a = (
    build_task_a_sequences(task_a)
)

test_idx_a = (
    sa == "test"
)

X_test_a_tcn = Xa[
    test_idx_a
]

y_test_a_tcn = ya[
    test_idx_a
]

subjects_a_tcn = subj_a[
    test_idx_a
]

minutes_a_tcn = min_a[
    test_idx_a
]

task_a_model = LightweightTCN(
    input_features=len(HRV_FEATURES),
    hidden_channels=32,
    dropout=0.15,
).to(DEVICE)

task_a_model.load_state_dict(
    torch.load(
        f"{TCN_DIR}/task_a_tcn_best.pt",
        map_location=DEVICE
    )
)

task_a_scaler = joblib.load(
    f"{TCN_DIR}/task_a_tcn_scaler.joblib"
)

prob_a_tcn = tcn_predict(
    task_a_model,
    task_a_scaler,
    X_test_a_tcn
)

result_a_tcn = evaluate_test(
    y_true=y_test_a_tcn,
    y_prob=prob_a_tcn,
    threshold=TASK_A_TCN_THRESHOLD,
    task_name="Task_A",
    model_name="TCN",
)

print("\nTask A TCN")
print(result_a_tcn)


# ============================================================
# TASK B — LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 80)
print("TASK B — FINAL TEST")
print("=" * 80)

b = task_b[
    task_b["primary_hrv_analysis"] == 1
].copy()

train_mask = (
    b["split"] == "train"
)

test_mask = (
    b["split"] == "test"
)

TASK_B_FEATURES = []

for lag in [5, 4, 3, 2, 1]:

    for feature in HRV_FEATURES:

        TASK_B_FEATURES.append(
            f"{feature}_tminus{lag}"
        )

X_train_b = b.loc[
    train_mask,
    TASK_B_FEATURES
]

y_train_b = b.loc[
    train_mask,
    "target"
].astype(int)

X_test_b = b.loc[
    test_mask,
    TASK_B_FEATURES
]

y_test_b = b.loc[
    test_mask,
    "target"
].astype(int)

model_b_lr = make_logistic_model()

# Refit on TRAIN ONLY using frozen configuration.
model_b_lr.fit(
    X_train_b,
    y_train_b
)

prob_b_lr = model_b_lr.predict_proba(
    X_test_b
)[:, 1]

result_b_lr = evaluate_test(
    y_true=y_test_b.values,
    y_prob=prob_b_lr,
    threshold=TASK_B_LR_THRESHOLD,
    task_name="Task_B",
    model_name="LogisticRegression",
)

print("\nTask B Logistic Regression")
print(result_b_lr)


# ============================================================
# TASK B — TCN
# ============================================================

Xb, yb, sb, subj_b, min_b = (
    build_task_b_sequences(task_b)
)

test_idx_b = (
    sb == "test"
)

X_test_b_tcn = Xb[
    test_idx_b
]

y_test_b_tcn = yb[
    test_idx_b
]

subjects_b_tcn = subj_b[
    test_idx_b
]

minutes_b_tcn = min_b[
    test_idx_b
]

task_b_model = LightweightTCN(
    input_features=len(HRV_FEATURES),
    hidden_channels=32,
    dropout=0.15,
).to(DEVICE)

task_b_model.load_state_dict(
    torch.load(
        f"{TCN_DIR}/task_b_tcn_best.pt",
        map_location=DEVICE
    )
)

task_b_scaler = joblib.load(
    f"{TCN_DIR}/task_b_tcn_scaler.joblib"
)

prob_b_tcn = tcn_predict(
    task_b_model,
    task_b_scaler,
    X_test_b_tcn
)

result_b_tcn = evaluate_test(
    y_true=y_test_b_tcn,
    y_prob=prob_b_tcn,
    threshold=TASK_B_TCN_THRESHOLD,
    task_name="Task_B",
    model_name="TCN",
)

print("\nTask B TCN")
print(result_b_tcn)


# ============================================================
# FINAL RESULTS TABLE
# ============================================================

final_results = pd.DataFrame([
    result_a_lr,
    result_a_tcn,
    result_b_lr,
    result_b_tcn,
])

final_results.to_csv(
    f"{FINAL_DIR}/final_test_results.csv",
    index=False
)

print("\n" + "=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)

print(
    final_results[
        [
            "task",
            "model",
            "n_test",
            "n_positive",
            "positive_prevalence",
            "auprc",
            "auprc_no_skill",
            "auroc",
            "precision",
            "recall_sensitivity",
            "f1",
            "accuracy",
            "brier",
            "threshold",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE INDIVIDUAL TEST PREDICTIONS
# ============================================================

# ------------------------------------------------------------
# Task A Logistic Regression
# ------------------------------------------------------------

task_a_test_mask = (
    a["split"] == "test"
)

pred_a_lr = a.loc[
    task_a_test_mask,
    [
        "subject_id",
        "minute_index",
        "target",
    ]
].copy()

pred_a_lr["model"] = "LogisticRegression"
pred_a_lr["predicted_probability"] = prob_a_lr
pred_a_lr["threshold"] = TASK_A_LR_THRESHOLD
pred_a_lr["predicted_label"] = (
    prob_a_lr >= TASK_A_LR_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# Task A TCN
# ------------------------------------------------------------

pred_a_tcn = pd.DataFrame({
    "subject_id": subjects_a_tcn,
    "minute_index": minutes_a_tcn,
    "target": y_test_a_tcn,
    "model": "TCN",
    "predicted_probability": prob_a_tcn,
    "threshold": TASK_A_TCN_THRESHOLD,
})

pred_a_tcn["predicted_label"] = (
    prob_a_tcn >= TASK_A_TCN_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# Task B Logistic Regression
# ------------------------------------------------------------

task_b_test_mask = (
    b["split"] == "test"
)

pred_b_lr = b.loc[
    task_b_test_mask,
    [
        "subject_id",
        "target_minute",
        "target",
    ]
].copy()

pred_b_lr["model"] = "LogisticRegression"
pred_b_lr["predicted_probability"] = prob_b_lr
pred_b_lr["threshold"] = TASK_B_LR_THRESHOLD
pred_b_lr["predicted_label"] = (
    prob_b_lr >= TASK_B_LR_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# Task B TCN
# ------------------------------------------------------------

pred_b_tcn = pd.DataFrame({
    "subject_id": subjects_b_tcn,
    "target_minute": minutes_b_tcn,
    "target": y_test_b_tcn,
    "model": "TCN",
    "predicted_probability": prob_b_tcn,
    "threshold": TASK_B_TCN_THRESHOLD,
})

pred_b_tcn["predicted_label"] = (
    prob_b_tcn >= TASK_B_TCN_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

pred_a_lr.to_csv(
    f"{FINAL_DIR}/task_a_logistic_predictions.csv",
    index=False
)

pred_a_tcn.to_csv(
    f"{FINAL_DIR}/task_a_tcn_predictions.csv",
    index=False
)

pred_b_lr.to_csv(
    f"{FINAL_DIR}/task_b_logistic_predictions.csv",
    index=False
)

pred_b_tcn.to_csv(
    f"{FINAL_DIR}/task_b_tcn_predictions.csv",
    index=False
)

print("\nPrediction files saved successfully.")

# ============================================================
# CONFUSION MATRICES
# ============================================================

confusion_rows = []

for result in [
    result_a_lr,
    result_a_tcn,
    result_b_lr,
    result_b_tcn
]:

    confusion_rows.append({
        "task": result["task"],
        "model": result["model"],
        "TN": result["tn"],
        "FP": result["fp"],
        "FN": result["fn"],
        "TP": result["tp"],
    })

confusion_df = pd.DataFrame(
    confusion_rows
)

confusion_df.to_csv(
    f"{FINAL_DIR}/confusion_matrices.csv",
    index=False
)


# ============================================================
# FINAL SANITY CHECKS
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST SANITY CHECKS")
print("=" * 80)

print(
    "\nTask A Logistic test rows:",
    len(pred_a_lr)
)

print(
    "Task A TCN test rows:",
    len(pred_a_tcn)
)

print(
    "Task B Logistic test rows:",
    len(pred_b_lr)
)

print(
    "Task B TCN test rows:",
    len(pred_b_tcn)
)

assert (
    pred_a_lr["target"].sum()
    == 1011
)

assert (
    pred_b_lr["target"].sum()
    == 29
)

print(
    "\nPASS: Task A test positives = 1011"
)

print(
    "PASS: Task B test positives = 29"
)

print(
    "\nTest evaluation complete."
)

print(
    "\nSaved final results to:",
    FINAL_DIR
)
