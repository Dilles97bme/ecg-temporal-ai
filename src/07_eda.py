
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE = "/content/drive/MyDrive/ecg_temporal_ai"

FEATURE_DIR = os.path.join(
    BASE,
    "outputs",
    "features"
)

EDA_DIR = os.path.join(
    BASE,
    "outputs",
    "eda"
)

FIG_DIR = os.path.join(
    EDA_DIR,
    "figures"
)

os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TASK_A_PATH = os.path.join(
    FEATURE_DIR,
    "task_a_hrv_features.csv"
)

TASK_B_PATH = os.path.join(
    FEATURE_DIR,
    "task_b_hrv_features.csv"
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

HRV_BASE_FEATURES = [
    "mean_rr_sec",
    "median_rr_sec",
    "mean_hr_bpm",
    "median_hr_bpm",
    "sdnn_sec",
    "rmssd_sec",
    "pnn50",
    "cv_rr",
]

PLOT_FEATURES = {
    "mean_hr_bpm": "Mean HR (bpm)",
    "median_hr_bpm": "Median HR (bpm)",
    "sdnn_sec": "SDNN (s)",
    "rmssd_sec": "RMSSD (s)",
    "pnn50": "pNN50",
    "cv_rr": "RR CV",
}

TRAJECTORY_FEATURES = [
    "mean_hr_bpm",
    "sdnn_sec",
    "rmssd_sec",
    "pnn50",
    "cv_rr",
]

LAGS = [5, 4, 3, 2, 1]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("STEP 7 — EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    task_a = pd.read_csv(TASK_A_PATH)
    task_b = pd.read_csv(TASK_B_PATH)

    print("\nTask A:", task_a.shape)
    print("Task B:", task_b.shape)

    print("\nTask A columns:")
    print(task_a.columns.tolist())

    print("\nTask B columns:")
    print(task_b.columns.tolist())


    # ========================================================
    # TASK A — BASIC CLASS DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 80)
    print("TASK A — CLASS DISTRIBUTION")
    print("=" * 80)

    task_a_counts = (
        task_a["target"]
        .value_counts()
        .sort_index()
    )

    task_a_prevalence = (
        task_a["target"]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    class_distribution_a = pd.DataFrame({
        "count": task_a_counts,
        "prevalence": task_a_prevalence,
    })

    print("\nTask A class distribution:")
    print(class_distribution_a)

    class_distribution_a.to_csv(
        os.path.join(
            EDA_DIR,
            "task_a_class_distribution.csv"
        )
    )


    # ========================================================
    # TASK A — HRV SUMMARY BY TARGET
    # ========================================================

    print("\n" + "=" * 80)
    print("TASK A — HRV SUMMARY BY TARGET")
    print("=" * 80)

    valid_a = task_a[
        task_a["hrv_complete_valid"] == True
    ].copy()

    summary_a = (
        valid_a
        .groupby("target")[
            HRV_BASE_FEATURES
        ]
        .agg(
            ["mean", "median", "std"]
        )
    )

    print(summary_a)

    summary_a.to_csv(
        os.path.join(
            EDA_DIR,
            "hrv_summary_by_target.csv"
        )
    )


    # ========================================================
    # TASK A — FEATURE BOXPLOTS
    # ========================================================

    print("\n" + "=" * 80)
    print("TASK A — HRV FEATURE BOXPLOTS")
    print("=" * 80)

    for feature, label in PLOT_FEATURES.items():

        data_normal = valid_a.loc[
            valid_a["target"] == 0,
            feature
        ].dropna()

        data_apnea = valid_a.loc[
            valid_a["target"] == 1,
            feature
        ].dropna()

        plt.figure(
            figsize=(7, 5)
        )

        # Use matplotlib defaults; no explicit colors.
        plt.boxplot(
            [
                data_normal,
                data_apnea
            ],
            labels=[
                "Normal",
                "Apnea"
            ],
            showfliers=False
        )

        plt.ylabel(label)

        plt.title(
            f"{label}: Normal vs Apnea"
        )

        plt.tight_layout()

        output_path = os.path.join(
            FIG_DIR,
            f"task_a_{feature}_boxplot.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )


    # ========================================================
    # TASK B — POSITIVE EVENT TRAJECTORY
    # ========================================================

    print("\n" + "=" * 80)
    print("TASK B — PRE-ONSET POSITIVE TRAJECTORY")
    print("=" * 80)

    positive_b = task_b[
        (task_b["target"] == 1)
        &
        (
            task_b["history_complete_valid"]
            == 1
        )
    ].copy()

    print(
        "Positive Task B windows available "
        "for trajectory analysis:",
        len(positive_b)
    )

    trajectory_rows = []

    for feature in TRAJECTORY_FEATURES:

        for lag in LAGS:

            column = (
                f"{feature}_tminus{lag}"
            )

            trajectory_rows.append({
                "feature": feature,
                "lag": lag,
                "mean": positive_b[
                    column
                ].mean(),
                "median": positive_b[
                    column
                ].median(),
                "std": positive_b[
                    column
                ].std(),
            })

    trajectory = pd.DataFrame(
        trajectory_rows
    )

    print("\nPre-onset trajectory summary:")
    print(trajectory)

    trajectory.to_csv(
        os.path.join(
            EDA_DIR,
            "pre_onset_trajectory_positive.csv"
        ),
        index=False
    )


    # ========================================================
    # TASK B — TRAJECTORY PLOTS
    # ========================================================

    print("\n" + "=" * 80)
    print("TASK B — PRE-ONSET TRAJECTORY PLOTS")
    print("=" * 80)

    for feature in TRAJECTORY_FEATURES:

        temp = trajectory[
            trajectory["feature"] == feature
        ].sort_values(
            "lag",
            ascending=False
        )

        x_labels = [
            "t-5",
            "t-4",
            "t-3",
            "t-2",
            "t-1",
        ]

        plt.figure(
            figsize=(7, 5)
        )

        plt.plot(
            x_labels,
            temp["mean"],
            marker="o"
        )

        plt.xlabel(
            "Minutes before apnea onset"
        )

        plt.ylabel(feature)

        plt.title(
            f"Pre-onset trajectory: {feature}"
        )

        plt.tight_layout()

        output_path = os.path.join(
            FIG_DIR,
            f"task_b_pre_onset_{feature}.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )


    # ========================================================
    # SUBJECT-LEVEL SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("SUBJECT-LEVEL SUMMARY")
    print("=" * 80)

    subject_summary = (
        valid_a
        .groupby(
            [
                "subject_id",
                "target"
            ]
        )[HRV_BASE_FEATURES]
        .mean()
        .reset_index()
    )

    print(
        "Subject-level summary shape:",
        subject_summary.shape
    )

    print(
        subject_summary.head(20)
    )

    subject_summary.to_csv(
        os.path.join(
            EDA_DIR,
            "subject_level_summary.csv"
        ),
        index=False
    )


    # ========================================================
    # SUBJECTS REPRESENTED BY TARGET
    # ========================================================

    subjects_by_target = (
        valid_a
        .groupby("target")[
            "subject_id"
        ]
        .nunique()
        .reset_index(
            name="n_subjects"
        )
    )

    print(
        "\nSubjects represented by target:"
    )

    print(
        subjects_by_target
    )

    subjects_by_target.to_csv(
        os.path.join(
            EDA_DIR,
            "subjects_by_target.csv"
        ),
        index=False
    )


    # ========================================================
    # SUMMARY FILE
    # ========================================================

    summary = {
        "task_a_rows": int(
            len(task_a)
        ),
        "task_b_rows": int(
            len(task_b)
        ),
        "task_a_valid_hrv_rows": int(
            len(valid_a)
        ),
        "task_b_positive_trajectory_rows": int(
            len(positive_b)
        ),
        "task_a_positive_count": int(
            task_a["target"].sum()
        ),
        "task_a_negative_count": int(
            (task_a["target"] == 0).sum()
        ),
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        os.path.join(
            EDA_DIR,
            "eda_summary.csv"
        ),
        index=False
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 7 COMPLETE")
    print("=" * 80)

    print("\nEDA tables saved under:")
    print(EDA_DIR)

    print("\nEDA figures saved under:")
    print(FIG_DIR)


if __name__ == "__main__":
    main()
