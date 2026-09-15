
"""
===============================================================================
02_SPLIT.PY
===============================================================================

Section 2: Data Split and Leakage Prevention

Purpose:
    Create and verify a fixed record-level train/validation/test split.

Important:
    - Splitting is performed at the recording/subject level.
    - No individual windows are split independently.
    - The test set is frozen and must not be used for model selection.
    - The split is deterministic and explicitly stored.


===============================================================================
"""

from pathlib import Path

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(
    "/content/drive/MyDrive/ecg_temporal_ai"
)

QC_FILE = (
    PROJECT_DIR
    / "outputs"
    / "qc"
    / "record_qc.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "splits"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_RECORDS = [
    "a02",
    "a03",
    "a05",
    "a06",
    "a07",
    "a08",
    "a09",
    "a11",
    "a13",
    "a15",
    "a16",
    "a19",
    "a20",
    "b04",
    "b05",
    "c02",
    "c03",
    "c04",
    "c09",
    "c10",
]

VAL_RECORDS = [
    "a18",
    "a01",
    "b01",
    "c07",
    "c01",
]

TEST_RECORDS = [
    "a12",
    "a14",
    "b02",
    "c08",
    "c06",
]


# =============================================================================
# VALIDATE SPLIT DEFINITION
# =============================================================================

def validate_split_definition():

    train = set(TRAIN_RECORDS)
    val = set(VAL_RECORDS)
    test = set(TEST_RECORDS)

    # No overlap.
    assert train.isdisjoint(val), (
        "Train and validation sets overlap."
    )

    assert train.isdisjoint(test), (
        "Train and test sets overlap."
    )

    assert val.isdisjoint(test), (
        "Validation and test sets overlap."
    )

    # Expected sizes.
    assert len(train) == 20
    assert len(val) == 5
    assert len(test) == 5

    # All 30 records should be represented exactly once.
    combined = train | val | test

    assert len(combined) == 30, (
        f"Expected 30 unique records, found {len(combined)}."
    )

    return True


# =============================================================================
# ADD SPLIT LABEL
# =============================================================================

def assign_split(record_id):

    if record_id in TRAIN_RECORDS:
        return "train"

    if record_id in VAL_RECORDS:
        return "validation"

    if record_id in TEST_RECORDS:
        return "test"

    raise ValueError(
        f"Record {record_id} is not present in the frozen split."
    )


# =============================================================================
# CALCULATE SPLIT STATISTICS
# =============================================================================

def calculate_split_statistics(df):

    rows = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        sub = df[
            df["split"] == split_name
        ].copy()

        total_minutes = int(
            sub["n_annotations"].sum()
        )

        apnea_minutes = int(
            sub["n_apnea"].sum()
        )

        normal_minutes = int(
            sub["n_normal"].sum()
        )

        if total_minutes > 0:

            prevalence = (
                apnea_minutes /
                total_minutes
            )

        else:

            prevalence = 0.0

        rows.append({

            "split":
                split_name,

            "n_records":
                len(sub),

            "annotated_minutes":
                total_minutes,

            "apnea_minutes":
                apnea_minutes,

            "normal_minutes":
                normal_minutes,

            "apnea_prevalence":
                prevalence,

        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("SECTION 2 — RECORD-LEVEL DATA SPLIT")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Validate fixed split
    # -------------------------------------------------------------------------

    validate_split_definition()

    print()
    print("Frozen split definition validated.")

    # -------------------------------------------------------------------------
    # Load QC data
    # -------------------------------------------------------------------------

    if not QC_FILE.exists():

        raise FileNotFoundError(
            f"QC file not found:\n{QC_FILE}"
        )

    df = pd.read_csv(QC_FILE)

    print()
    print(
        "QC records loaded:",
        len(df)
    )

    # -------------------------------------------------------------------------
    # Verify expected records
    # -------------------------------------------------------------------------

    expected_records = set(
        TRAIN_RECORDS
        + VAL_RECORDS
        + TEST_RECORDS
    )

    actual_records = set(
        df["subject_id"]
    )

    missing = expected_records - actual_records
    unexpected = actual_records - expected_records

    if missing:

        raise RuntimeError(
            f"Records missing from QC file: {sorted(missing)}"
        )

    if unexpected:

        raise RuntimeError(
            f"Unexpected records in QC file: {sorted(unexpected)}"
        )

    # -------------------------------------------------------------------------
    # Assign split
    # -------------------------------------------------------------------------

    df["split"] = df["subject_id"].apply(
        assign_split
    )

    assert df["split"].notna().all()

    assert (
        df["split"].value_counts().to_dict()
        ==
        {
            "train": 20,
            "validation": 5,
            "test": 5,
        }
    )

    # -------------------------------------------------------------------------
    # Save record-level split
    # -------------------------------------------------------------------------

    split_columns = [
        "subject_id",
        "split",
        "n_annotations",
        "n_apnea",
        "n_normal",
        "apnea_prevalence",
        "duration_hr",
    ]

    split_df = df[
        split_columns
    ].sort_values(
        ["split", "subject_id"]
    )

    split_file = (
        OUTPUT_DIR
        / "record_split.csv"
    )

    split_df.to_csv(
        split_file,
        index=False
    )

    # -------------------------------------------------------------------------
    # Save individual split files
    # -------------------------------------------------------------------------

    train_df = split_df[
        split_df["split"] == "train"
    ]

    val_df = split_df[
        split_df["split"] == "validation"
    ]

    test_df = split_df[
        split_df["split"] == "test"
    ]

    train_df.to_csv(
        OUTPUT_DIR / "train_records.csv",
        index=False
    )

    val_df.to_csv(
        OUTPUT_DIR / "validation_records.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test_records.csv",
        index=False
    )

    # -------------------------------------------------------------------------
    # Calculate statistics
    # -------------------------------------------------------------------------

    stats_df = calculate_split_statistics(
        df
    )

    stats_file = (
        OUTPUT_DIR
        / "split_statistics.csv"
    )

    stats_df.to_csv(
        stats_file,
        index=False
    )

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FROZEN SPLIT")
    print("=" * 80)

    print()
    print("TRAIN (20):")
    print(", ".join(TRAIN_RECORDS))

    print()
    print("VALIDATION (5):")
    print(", ".join(VAL_RECORDS))

    print()
    print("TEST (5):")
    print(", ".join(TEST_RECORDS))

    print()
    print("=" * 80)
    print("SPLIT STATISTICS")
    print("=" * 80)

    print(
        stats_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Leakage checks
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LEAKAGE CHECK")
    print("=" * 80)

    train_set = set(
        train_df["subject_id"]
    )

    val_set = set(
        val_df["subject_id"]
    )

    test_set = set(
        test_df["subject_id"]
    )

    print(
        "Train ∩ Validation:",
        train_set & val_set
    )

    print(
        "Train ∩ Test:",
        train_set & test_set
    )

    print(
        "Validation ∩ Test:",
        val_set & test_set
    )

    assert not train_set & val_set
    assert not train_set & test_set
    assert not val_set & test_set

    print()
    print("All leakage checks passed.")

    # -------------------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(split_file)
    print(OUTPUT_DIR / "train_records.csv")
    print(OUTPUT_DIR / "validation_records.csv")
    print(OUTPUT_DIR / "test_records.csv")
    print(stats_file)

    print()
    print("SECTION 2 COMPLETE.")
    print("The test split is now frozen.")


if __name__ == "__main__":
    main()
