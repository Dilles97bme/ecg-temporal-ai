
"""
===============================================================================
03_WINDOWING.PY
===============================================================================


Creates metadata for:

Task A:
    Current one-minute apnea detection.

Task B:
    Five-minute ECG history -> new apnea onset in the next one-minute period.

Important:
    Task B predicts NEW onset, not continuation of an already ongoing event.

Minute-level annotation definition:
    Annotation index m corresponds to ECG interval:

        [m * 6000, (m + 1) * 6000)

Task B:
    prediction time = start of minute m

    history:
        minutes [m-5, m)

    target:
        onset at minute m

    onset[m] = 1 if:
        label[m-1] == 0 AND label[m] == 1

The first five annotated minutes of every record cannot be used for Task B
because five minutes of history are unavailable.

Unlabeled ECG tails are excluded because no ground-truth label is available.
===============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(
    "/content/drive/MyDrive/ecg_temporal_ai"
)

ANNOTATION_FILE = (
    PROJECT_DIR
    / "outputs"
    / "qc"
    / "annotation_qc.csv"
)

SPLIT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "splits"
    / "record_split.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "windowing"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FS = 100
SECONDS_PER_MINUTE = 60
SAMPLES_PER_MINUTE = FS * SECONDS_PER_MINUTE

HISTORY_MINUTES = 5
HORIZON_MINUTES = 1


# =============================================================================
# LOAD ANNOTATIONS
# =============================================================================

def load_annotations():

    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Annotation file not found:\n{ANNOTATION_FILE}"
        )

    df = pd.read_csv(ANNOTATION_FILE)

    required_columns = {
        "subject_id",
        "minute_index",
        "annotation_sample",
        "symbol",
        "label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required annotation columns: {missing}"
        )

    return df


# =============================================================================
# LOAD SPLITS
# =============================================================================

def load_splits():

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split file not found:\n{SPLIT_FILE}"
        )

    split_df = pd.read_csv(SPLIT_FILE)

    required_columns = {
        "subject_id",
        "split",
    }

    missing = required_columns - set(split_df.columns)

    if missing:
        raise ValueError(
            f"Missing split columns: {missing}"
        )

    return split_df[
        ["subject_id", "split"]
    ].drop_duplicates()


# =============================================================================
# BUILD TASK A WINDOWS
# =============================================================================

def build_task_a(annotation_df, split_df):

    df = annotation_df.copy()

    df = df.merge(
        split_df,
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    if df["split"].isna().any():
        raise RuntimeError(
            "Some annotation records do not have a split assignment."
        )

    # Each annotation corresponds to one complete minute.
    df["window_start_sample"] = (
        df["minute_index"] *
        SAMPLES_PER_MINUTE
    )

    df["window_end_sample"] = (
        (df["minute_index"] + 1) *
        SAMPLES_PER_MINUTE
    )

    df["window_start_sec"] = (
        df["minute_index"] *
        SECONDS_PER_MINUTE
    )

    df["window_end_sec"] = (
        (df["minute_index"] + 1) *
        SECONDS_PER_MINUTE
    )

    df["task"] = "A_detection"

    df["target"] = df["label"].astype(int)

    columns = [
        "subject_id",
        "split",
        "task",
        "minute_index",
        "window_start_sample",
        "window_end_sample",
        "window_start_sec",
        "window_end_sec",
        "symbol",
        "target",
    ]

    return df[columns]


# =============================================================================
# BUILD TASK B WINDOWS
# =============================================================================

def build_task_b(annotation_df, split_df):

    rows = []

    for subject_id, group in annotation_df.groupby(
        "subject_id",
        sort=True
    ):

        group = group.sort_values(
            "minute_index"
        ).reset_index(drop=True)

        labels = group["label"].astype(int).to_numpy()
        minutes = group["minute_index"].to_numpy()

        # ---------------------------------------------------------------------
        # Verify consecutive minute indexing.
        # ---------------------------------------------------------------------

        if len(minutes) > 1:

            differences = np.diff(minutes)

            if not np.all(differences == 1):

                raise RuntimeError(
                    f"Non-consecutive annotation minutes found for "
                    f"{subject_id}."
                )

        # ---------------------------------------------------------------------
        # Determine split.
        # ---------------------------------------------------------------------

        subject_split = split_df.loc[
            split_df["subject_id"] == subject_id,
            "split"
        ]

        if len(subject_split) != 1:

            raise RuntimeError(
                f"Expected exactly one split for {subject_id}."
            )

        split = subject_split.iloc[0]

        # ---------------------------------------------------------------------
        # Need 5 complete minutes of history.
        #
        # Target minute = m
        # History      = [m-5, m)
        #
        # Therefore m starts at index 5.
        # ---------------------------------------------------------------------

        for i in range(
            HISTORY_MINUTES,
            len(group)
        ):

            target_minute = int(
                minutes[i]
            )

            previous_label = int(
                labels[i - 1]
            )

            current_label = int(
                labels[i]
            )

            # New apnea onset.
            onset = int(
                previous_label == 0
                and current_label == 1
            )

            history_start_minute = (
                target_minute -
                HISTORY_MINUTES
            )

            history_end_minute = (
                target_minute
            )

            prediction_time_sec = (
                target_minute *
                SECONDS_PER_MINUTE
            )

            future_end_sec = (
                (target_minute + HORIZON_MINUTES)
                * SECONDS_PER_MINUTE
            )

            rows.append({

                "subject_id":
                    subject_id,

                "split":
                    split,

                "task":
                    "B_prediction",

                "target_minute":
                    target_minute,

                "history_start_minute":
                    history_start_minute,

                "history_end_minute":
                    history_end_minute,

                "history_start_sec":
                    history_start_minute *
                    SECONDS_PER_MINUTE,

                "prediction_time_sec":
                    prediction_time_sec,

                "future_end_sec":
                    future_end_sec,

                "history_start_sample":
                    history_start_minute *
                    SAMPLES_PER_MINUTE,

                "prediction_time_sample":
                    target_minute *
                    SAMPLES_PER_MINUTE,

                "future_end_sample":
                    (target_minute + 1) *
                    SAMPLES_PER_MINUTE,

                "previous_label":
                    previous_label,

                "current_label":
                    current_label,

                "target":
                    onset,

            })

    return pd.DataFrame(rows)


# =============================================================================
# CLASS DISTRIBUTION
# =============================================================================

def calculate_distribution(df):

    rows = []

    for task in sorted(df["task"].unique()):

        task_df = df[
            df["task"] == task
        ]

        for split in [
            "train",
            "validation",
            "test",
        ]:

            sub = task_df[
                task_df["split"] == split
            ]

            total = len(sub)

            positive = int(
                sub["target"].sum()
            )

            negative = (
                total - positive
            )

            prevalence = (
                positive / total
                if total > 0
                else np.nan
            )

            rows.append({

                "task":
                    task,

                "split":
                    split,

                "total_windows":
                    total,

                "positive_windows":
                    positive,

                "negative_windows":
                    negative,

                "positive_prevalence":
                    prevalence,

            })

    return pd.DataFrame(rows)


# =============================================================================
# VALIDATE TASK B LABELS
# =============================================================================

def validate_task_b(task_b):

    # Every positive target must correspond to
    # previous = 0 and current = 1.
    positives = task_b[
        task_b["target"] == 1
    ]

    assert (
        (
            positives["previous_label"] == 0
        )
        &
        (
            positives["current_label"] == 1
        )
    ).all()

    # Every negative target must NOT be a new onset.
    negatives = task_b[
        task_b["target"] == 0
    ]

    assert not (
        (
            negatives["previous_label"] == 0
        )
        &
        (
            negatives["current_label"] == 1
        )
    ).any()

    # History should always end exactly at prediction time.
    assert (
        task_b["history_end_minute"]
        ==
        task_b["target_minute"]
    ).all()

    # Five-minute history.
    assert (
        task_b["history_end_minute"]
        -
        task_b["history_start_minute"]
        ==
        HISTORY_MINUTES
    ).all()


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("SECTION 3 — CLASS IMBALANCE + TEMPORAL WINDOW LABELS")
    print("=" * 80)

    annotation_df = load_annotations()
    split_df = load_splits()

    # -------------------------------------------------------------------------
    # Task A
    # -------------------------------------------------------------------------

    task_a = build_task_a(
        annotation_df,
        split_df
    )

    # -------------------------------------------------------------------------
    # Task B
    # -------------------------------------------------------------------------

    task_b = build_task_b(
        annotation_df,
        split_df
    )

    validate_task_b(task_b)

    # -------------------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------------------

    windows_df = pd.concat(
        [task_a, task_b],
        ignore_index=True,
        sort=False
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    task_a_file = (
        OUTPUT_DIR
        / "task_a_windows.csv"
    )

    task_b_file = (
        OUTPUT_DIR
        / "task_b_windows.csv"
    )

    all_windows_file = (
        OUTPUT_DIR
        / "all_window_metadata.csv"
    )

    task_a.to_csv(
        task_a_file,
        index=False
    )

    task_b.to_csv(
        task_b_file,
        index=False
    )

    windows_df.to_csv(
        all_windows_file,
        index=False
    )

    # -------------------------------------------------------------------------
    # Class distributions
    # -------------------------------------------------------------------------

    distribution_df = calculate_distribution(
        windows_df
    )

    distribution_file = (
        OUTPUT_DIR
        / "class_distribution.csv"
    )

    distribution_df.to_csv(
        distribution_file,
        index=False
    )

    # -------------------------------------------------------------------------
    # Print Task A
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("TASK A — CURRENT-MINUTE DETECTION")
    print("=" * 80)

    task_a_stats = distribution_df[
        distribution_df["task"] ==
        "A_detection"
    ]

    print(
        task_a_stats.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Print Task B
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("TASK B — FUTURE NEW-ONSET PREDICTION")
    print("=" * 80)

    task_b_stats = distribution_df[
        distribution_df["task"] ==
        "B_prediction"
    ]

    print(
        task_b_stats.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Important label examples
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("TASK B POSITIVE EXAMPLES")
    print("=" * 80)

    print(
        task_b[
            task_b["target"] == 1
        ].head(20).to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Final counts
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL COUNTS")
    print("=" * 80)

    print(
        "Task A windows:",
        len(task_a)
    )

    print(
        "Task B windows:",
        len(task_b)
    )

    print(
        "Task B positive:",
        int(task_b["target"].sum())
    )

    print(
        "Task B negative:",
        int(
            len(task_b)
            -
            task_b["target"].sum()
        )
    )

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(task_a_file)
    print(task_b_file)
    print(all_windows_file)
    print(distribution_file)

    print()
    print("TEMPORAL WINDOW CONSTRUCTION COMPLETE.")


if __name__ == "__main__":
    main()
