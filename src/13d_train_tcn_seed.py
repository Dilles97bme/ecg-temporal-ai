
import os
import random
import copy
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
)

from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE = "/content/drive/MyDrive/ecg_temporal_ai"

FEATURE_DIR = f"{BASE}/outputs/features"

# ------------------------------------------------------------
# Seed is supplied externally:
#   TCN_SEED=42
#   TCN_SEED=123
#   TCN_SEED=2026
#
# Default = 42
# ------------------------------------------------------------
SEED = int(
    os.environ.get("TCN_SEED", "42")
)

# Separate output directory for each seed
OUT_DIR = (
    f"{BASE}/outputs/analysis_controls/"
    f"tcn_seeds/seed_{SEED}"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

TASK_A_PATH = (
    f"{FEATURE_DIR}/task_a_hrv_features.csv"
)

TASK_B_PATH = (
    f"{FEATURE_DIR}/task_b_hrv_features.csv"
)

# ------------------------------------------------------------
# Original Step 9 hyperparameters — unchanged
# ------------------------------------------------------------

BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE = 12

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DROPOUT = 0.15
HIDDEN_CHANNELS = 32

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

# Python
random.seed(SEED)

# NumPy
np.random.seed(SEED)

# PyTorch
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Deterministic CuDNN behavior where applicable
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Request deterministic algorithms where supported
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass


print("=" * 80)
print("STEP 13D — TCN THREE-SEED REPEATABILITY RUN")
print("=" * 80)

print("Seed:", SEED)
print("Device:", DEVICE)

print("\nHyperparameters")
print("-" * 80)
print("BATCH_SIZE:", BATCH_SIZE)
print("MAX_EPOCHS:", MAX_EPOCHS)
print("PATIENCE:", PATIENCE)
print("LEARNING_RATE:", LEARNING_RATE)
print("WEIGHT_DECAY:", WEIGHT_DECAY)
print("DROPOUT:", DROPOUT)
print("HIDDEN_CHANNELS:", HIDDEN_CHANNELS)


# ============================================================
# LOAD DATA
# ============================================================

task_a = pd.read_csv(
    TASK_A_PATH
)

task_b = pd.read_csv(
    TASK_B_PATH
)

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
# TCN BLOCK
# ============================================================

class TemporalBlock(nn.Module):
    """
    Residual temporal convolution block.

    Input:
        [batch, channels, time]

    Dilated convolutions provide a larger temporal
    receptive field while keeping the architecture small.
    """

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
                kernel_size=1,
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


# ============================================================
# LIGHTWEIGHT TCN
# ============================================================

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

        self.pool = nn.AdaptiveAvgPool1d(
            1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                hidden_channels,
                16,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                16,
                1,
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
# THRESHOLD SELECTION
# ============================================================

def select_f1_threshold(
    y_true,
    y_prob,
):
    """
    Select classification threshold from validation data only.
    """

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            y_prob
        )
    )

    if len(thresholds) == 0:
        return 0.5

    f1_values = (
        2.0
        * precision[:-1]
        * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_idx = int(
        np.argmax(f1_values)
    )

    return float(
        thresholds[best_idx]
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(
    y_true,
    y_prob,
    threshold,
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

    brier = brier_score_loss(
        y_true,
        y_prob
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "auroc": auroc,
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": brier,
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ============================================================
# SCALING
# ============================================================

def fit_scaler_and_transform(
    X_train,
    X_val,
):
    """
    Fit StandardScaler on training data only.

    Input shape:
        [N, T, F]
    """

    n_train, t_train, f_train = (
        X_train.shape
    )

    n_val, t_val, f_val = (
        X_val.shape
    )

    scaler = StandardScaler()

    train_flat = X_train.reshape(
        -1,
        f_train
    )

    val_flat = X_val.reshape(
        -1,
        f_val
    )

    # IMPORTANT:
    # scaler is fitted ONLY on training data.
    scaler.fit(train_flat)

    train_scaled = scaler.transform(
        train_flat
    ).reshape(
        n_train,
        t_train,
        f_train,
    )

    val_scaled = scaler.transform(
        val_flat
    ).reshape(
        n_val,
        t_val,
        f_val,
    )

    return (
        train_scaled.astype(
            np.float32
        ),

        val_scaled.astype(
            np.float32
        ),

        scaler,
    )


# ============================================================
# BUILD TASK A SEQUENCES
# ============================================================

def build_task_a_sequences(df):
    """
    Task A:

        input:
            [t-4, t-3, t-2, t-1, t]

        target:
            apnea label at t

    A sequence is retained only when all five
    minutes have valid HRV features.
    """

    df = df[
        df["primary_hrv_analysis"] == 1
    ].copy()

    df = df.sort_values(
        [
            "subject_id",
            "minute_index",
        ]
    )

    lookup = df.set_index(
        [
            "subject_id",
            "minute_index",
        ]
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
            target_minute + 1,
        ):

            key = (
                subject,
                minute,
            )

            if key not in lookup.index:

                valid = False

                break

            h = lookup.loc[key]

            if int(
                h[
                    "primary_hrv_analysis"
                ]
            ) != 1:

                valid = False

                break

            history.append(
                h[
                    HRV_FEATURES
                ].values.astype(
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

        np.asarray(
            splits
        ),

        np.asarray(
            subjects
        ),

        np.asarray(
            target_minutes
        ),
    )


# ============================================================
# BUILD TASK B SEQUENCES
# ============================================================

def build_task_b_sequences(df):
    """
    Task B:

        input:
            [t-5, t-4, t-3, t-2, t-1]

        target:
            onset at t

    Only complete valid histories are retained.
    """

    df = df[
        df[
            "primary_hrv_analysis"
        ] == 1
    ].copy()

    df = df[
        df[
            "history_complete_valid"
        ] == 1
    ].copy()

    sequences = []
    labels = []
    splits = []
    subjects = []
    target_minutes = []

    for _, row in df.iterrows():

        seq = []

        for lag in [
            5,
            4,
            3,
            2,
            1,
        ]:

            features = [
                row[
                    f"{feature}_tminus{lag}"
                ]

                for feature
                in HRV_FEATURES
            ]

            seq.append(
                np.asarray(
                    features,
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
            int(
                row["target_minute"]
            )
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

        np.asarray(
            splits
        ),

        np.asarray(
            subjects
        ),

        np.asarray(
            target_minutes
        ),
    )


# ============================================================
# TRAIN ONE TCN
# ============================================================

def train_tcn(
    X_train,
    y_train,
    X_val,
    y_val,
    task_name,
):
    """
    Training protocol:

        - training split for fitting
        - validation AUPRC for model selection/
          early stopping
        - validation-only F1 threshold selection
        - class-weighted BCE loss
        - no test data
    """

    (
        X_train,
        X_val,
        scaler,
    ) = fit_scaler_and_transform(
        X_train,
        X_val,
    )

    # PyTorch shape:
    # [N, features, time]

    X_train_t = torch.tensor(
        X_train.transpose(
            0,
            2,
            1
        )
    )

    X_val_t = torch.tensor(
        X_val.transpose(
            0,
            2,
            1
        )
    )

    y_train_t = torch.tensor(
        y_train.astype(
            np.float32
        )
    )

    train_dataset = TensorDataset(
        X_train_t,
        y_train_t,
    )

    # --------------------------------------------------------
    # Deterministic DataLoader shuffling
    # --------------------------------------------------------

    loader_generator = torch.Generator()

    loader_generator.manual_seed(
        SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = LightweightTCN(
        input_features=X_train.shape[-1],
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
    ).to(DEVICE)

    # --------------------------------------------------------
    # Class weighting
    # --------------------------------------------------------

    n_positive = max(
        int(
            y_train.sum()
        ),
        1,
    )

    n_negative = max(
        int(
            len(y_train)
            - y_train.sum()
        ),
        1,
    )

    pos_weight_value = (
        n_negative
        / n_positive
    )

    pos_weight = torch.tensor(
        pos_weight_value,
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Early stopping state
    # --------------------------------------------------------

    best_state = None

    best_val_auprc = -np.inf

    best_epoch = 0

    patience_counter = 0

    print(
        f"\n{task_name}"
    )

    print(
        "Training samples:",
        len(y_train)
    )

    print(
        "Positive:",
        int(
            y_train.sum()
        )
    )

    print(
        "Negative:",
        n_negative
    )

    print(
        "pos_weight:",
        round(
            pos_weight_value,
            4
        )
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        epoch_losses = []

        for xb, yb in train_loader:

            xb = xb.to(
                DEVICE
            )

            yb = yb.to(
                DEVICE
            )

            optimizer.zero_grad()

            logits = model(
                xb
            )

            loss = criterion(
                logits,
                yb
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            epoch_losses.append(
                loss.item()
            )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_logits = model(
                X_val_t.to(
                    DEVICE
                )
            )

            val_prob = (
                torch.sigmoid(
                    val_logits
                )
                .cpu()
                .numpy()
            )

        val_auprc = (
            average_precision_score(
                y_val,
                val_prob
            )
        )

        val_auroc = (
            roc_auc_score(
                y_val,
                val_prob
            )
        )

        mean_loss = float(
            np.mean(
                epoch_losses
            )
        )

        # ----------------------------------------------------
        # Select best model using VALIDATION AUPRC
        # ----------------------------------------------------

        if val_auprc > (
            best_val_auprc + 1e-5
        ):

            best_val_auprc = (
                val_auprc
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch

            patience_counter = 0

        else:

            patience_counter += 1

        if (
            epoch == 1
            or epoch % 5 == 0
            or patience_counter == 0
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"loss={mean_loss:.4f} | "
                f"val_AUROC={val_auroc:.4f} | "
                f"val_AUPRC={val_auprc:.4f}"
            )

        if patience_counter >= PATIENCE:

            print(
                f"Early stopping at epoch "
                f"{epoch}"
            )

            break

    # ========================================================
    # RESTORE BEST VALIDATION MODEL
    # ========================================================

    if best_state is None:

        raise RuntimeError(
            f"No best model state was "
            f"recorded for {task_name}."
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    # Recompute validation probabilities
    # using the best model.

    with torch.no_grad():

        val_logits = model(
            X_val_t.to(
                DEVICE
            )
        )

        val_prob = (
            torch.sigmoid(
                val_logits
            )
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Threshold selected on validation only
    # --------------------------------------------------------

    threshold = (
        select_f1_threshold(
            y_val,
            val_prob
        )
    )

    metrics = evaluate_predictions(
        y_true=y_val,
        y_prob=val_prob,
        threshold=threshold,
    )

    print(
        "\nBest epoch:",
        best_epoch
    )

    print(
        "Best validation AUROC:",
        round(
            metrics["auroc"],
            4
        )
    )

    print(
        "Best validation AUPRC:",
        round(
            metrics["auprc"],
            4
        )
    )

    print(
        "Validation threshold:",
        round(
            metrics["threshold"],
            6
        )
    )

    print(
        "Validation F1:",
        round(
            metrics["f1"],
            4
        )
    )

    return (
        model,
        scaler,
        metrics,
        best_epoch,
    )


# ============================================================
# TASK A
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "TASK A TCN"
)

print(
    "=" * 80
)

(
    Xa,
    ya,
    sa,
    subj_a,
    min_a,
) = build_task_a_sequences(
    task_a
)

print(
    "All valid Task A TCN sequences:",
    Xa.shape
)

train_idx_a = (
    sa == "train"
)

val_idx_a = (
    sa == "validation"
)

test_idx_a = (
    sa == "test"
)

print(
    "Task A train sequences:",
    int(
        train_idx_a.sum()
    )
)

print(
    "Task A validation sequences:",
    int(
        val_idx_a.sum()
    )
)

print(
    "Task A test sequences:",
    int(
        test_idx_a.sum()
    ),
    "(NOT USED)"
)

(
    task_a_model,
    task_a_scaler,
    task_a_metrics,
    task_a_best_epoch,
) = train_tcn(
    Xa[train_idx_a],
    ya[train_idx_a],
    Xa[val_idx_a],
    ya[val_idx_a],
    "Task A TCN",
)


# ============================================================
# SAVE TASK A MODEL
# ============================================================

torch.save(
    task_a_model.state_dict(),
    f"{OUT_DIR}/task_a_tcn_best.pt"
)

try:
    import joblib

    joblib.dump(
        task_a_scaler,
        f"{OUT_DIR}/task_a_tcn_scaler.joblib"
    )

except ImportError:

    print(
        "WARNING: joblib not available; "
        "Task A scaler was not saved."
    )


# ============================================================
# TASK B
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "TASK B TCN"
)

print(
    "=" * 80
)

(
    Xb,
    yb,
    sb,
    subj_b,
    min_b,
) = build_task_b_sequences(
    task_b
)

print(
    "Complete valid Task B sequences:",
    Xb.shape
)

train_idx_b = (
    sb == "train"
)

val_idx_b = (
    sb == "validation"
)

test_idx_b = (
    sb == "test"
)

print(
    "Task B train sequences:",
    int(
        train_idx_b.sum()
    )
)

print(
    "Task B validation sequences:",
    int(
        val_idx_b.sum()
    )
)

print(
    "Task B test sequences:",
    int(
        test_idx_b.sum()
    ),
    "(NOT USED)"
)

(
    task_b_model,
    task_b_scaler,
    task_b_metrics,
    task_b_best_epoch,
) = train_tcn(
    Xb[train_idx_b],
    yb[train_idx_b],
    Xb[val_idx_b],
    yb[val_idx_b],
    "Task B TCN",
)


# ============================================================
# SAVE TASK B MODEL
# ============================================================

torch.save(
    task_b_model.state_dict(),
    f"{OUT_DIR}/task_b_tcn_best.pt"
)

try:
    import joblib

    joblib.dump(
        task_b_scaler,
        f"{OUT_DIR}/task_b_tcn_scaler.joblib"
    )

except ImportError:

    print(
        "WARNING: joblib not available; "
        "Task B scaler was not saved."
    )


# ============================================================
# SAVE VALIDATION RESULTS
# ============================================================

tcn_results = pd.DataFrame([
    {
        "seed": SEED,
        "task": "Task_A",
        "model": "TCN",
        "sequence_length": 5,
        "best_epoch": task_a_best_epoch,
        **task_a_metrics,
    },

    {
        "seed": SEED,
        "task": "Task_B",
        "model": "TCN",
        "sequence_length": 5,
        "best_epoch": task_b_best_epoch,
        **task_b_metrics,
    },
])


tcn_results.to_csv(
    f"{OUT_DIR}/tcn_validation_results.csv",
    index=False
)


# ============================================================
# SAVE RUN CONFIGURATION
# ============================================================

config = pd.DataFrame([
    {
        "seed": SEED,
        "device": str(DEVICE),
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "hidden_channels": HIDDEN_CHANNELS,
    }
])

config.to_csv(
    f"{OUT_DIR}/run_configuration.csv",
    index=False
)


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "STEP 13D SEED RUN COMPLETE"
)

print(
    "=" * 80
)

print(
    "\nValidation TCN results:"
)

print(
    tcn_results.to_string(
        index=False
    )
)

print(
    "\nSaved to:"
)

print(
    OUT_DIR
)

print(
    "\nTEST SET WAS NOT USED FOR "
    "TRAINING, EARLY STOPPING, "
    "THRESHOLD SELECTION, OR MODEL SELECTION."
)
