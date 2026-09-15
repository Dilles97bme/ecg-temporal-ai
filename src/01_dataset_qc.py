
"""
===============================================================================
01_DATASET_QC.PY
===============================================================================

Temporal AI for ECG-Based Event Detection and Short-Horizon Prediction

Section 1:
    - Load WFDB ECG recordings
    - Load apnea annotations
    - Verify sampling rate
    - Verify one-minute annotation alignment
    - Quantify recording duration
    - Quantify annotation coverage
    - Quantify class distribution
    - Check basic ECG signal quality

Important:
    This script does NOT train any model.
    It does NOT remove noisy records.
    It only performs dataset quality control.

===============================================================================
"""

from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd
import wfdb

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(
    "/content/drive/MyDrive/ecg_temporal_ai"
)

DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "qc"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

EXPECTED_FS = 100.0
SAMPLES_PER_MINUTE = int(EXPECTED_FS * 60)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# FIND PRIMARY RECORDS
# =============================================================================

def find_primary_records(data_dir):
    """
    Find primary ECG records.

    Primary records have names such as:
        a01
        a02
        b01
        c01

    We exclude:
        a01r
        a01er

    because these correspond to additional/combined records.
    """

    records = []

    for hea_file in data_dir.glob("*.hea"):

        record_id = hea_file.stem

        if record_id.endswith("r"):
            continue

        if record_id.endswith("er"):
            continue

        records.append(record_id)

    return sorted(set(records))


# =============================================================================
# ECG SIGNAL QUALITY
# =============================================================================

def calculate_signal_quality(ecg):
    """
    Calculate basic signal-quality statistics.

    IMPORTANT:
        These statistics are used for QC only.
        No records/windows are silently removed here.
    """

    ecg = np.asarray(ecg, dtype=float)

    total = len(ecg)

    if total == 0:
        return {
            "nan_fraction": 1.0,
            "inf_fraction": 0.0,
            "finite_fraction": 0.0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "peak_to_peak": np.nan,
            "flat_fraction": 1.0,
        }

    finite_mask = np.isfinite(ecg)

    finite_fraction = finite_mask.mean()
    nan_fraction = np.isnan(ecg).mean()
    inf_fraction = np.isinf(ecg).mean()

    if finite_mask.sum() == 0:

        return {
            "nan_fraction": float(nan_fraction),
            "inf_fraction": float(inf_fraction),
            "finite_fraction": float(finite_fraction),
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "peak_to_peak": np.nan,
            "flat_fraction": 1.0,
        }

    x = ecg[finite_mask]

    mean_value = np.mean(x)
    std_value = np.std(x)

    minimum = np.min(x)
    maximum = np.max(x)

    peak_to_peak = maximum - minimum

    # Simple flatness indicator.
    # This is intentionally conservative and is only a QC statistic.
    tolerance = max(1e-10, 1e-6 * max(1.0, abs(np.median(x))))

    flat_fraction = np.mean(
        np.abs(x - np.median(x)) <= tolerance
    )

    return {
        "nan_fraction": float(nan_fraction),
        "inf_fraction": float(inf_fraction),
        "finite_fraction": float(finite_fraction),
        "mean": float(mean_value),
        "std": float(std_value),
        "min": float(minimum),
        "max": float(maximum),
        "peak_to_peak": float(peak_to_peak),
        "flat_fraction": float(flat_fraction),
    }


# =============================================================================
# FIND ECG CHANNEL
# =============================================================================

def find_ecg_channel(record):
    """
    Identify the ECG channel.

    Prefer a channel whose name contains 'ECG'.

    If no ECG name is found, use channel 0 as a fallback and record that
    choice in the QC output.
    """

    signal_names = record.sig_name

    for idx, name in enumerate(signal_names):

        if "ecg" in name.lower():
            return idx, "name_match"

    if len(signal_names) > 0:
        return 0, "fallback_channel_0"

    return None, "no_channel"


# =============================================================================
# PROCESS ONE RECORD
# =============================================================================

def process_record(record_id):

    record_path = str(DATA_DIR / record_id)

    result = {
        "subject_id": record_id,
        "status": "OK",
    }

    try:

        # ---------------------------------------------------------------------
        # Read ECG
        # ---------------------------------------------------------------------

        record = wfdb.rdrecord(record_path)

        # ---------------------------------------------------------------------
        # Read apnea annotations
        # ---------------------------------------------------------------------

        annotation = wfdb.rdann(
            record_path,
            extension="apn"
        )

        # ---------------------------------------------------------------------
        # Basic recording information
        # ---------------------------------------------------------------------

        fs = float(record.fs)
        n_samples = int(record.sig_len)
        n_channels = int(record.n_sig)

        duration_sec = n_samples / fs
        duration_min = duration_sec / 60.0
        duration_hr = duration_sec / 3600.0

        # ---------------------------------------------------------------------
        # ECG channel
        # ---------------------------------------------------------------------

        ecg_channel, ecg_channel_method = find_ecg_channel(record)

        if ecg_channel is None:

            raise RuntimeError(
                "No signal channel available."
            )

        ecg = record.p_signal[:, ecg_channel]

        # ---------------------------------------------------------------------
        # Annotation information
        # ---------------------------------------------------------------------

        annotation_samples = np.asarray(
            annotation.sample,
            dtype=np.int64
        )

        annotation_symbols = list(annotation.symbol)

        n_annotations = len(annotation_samples)

        unique_symbols = sorted(
            set(annotation_symbols)
        )

        # ---------------------------------------------------------------------
        # Annotation spacing
        # ---------------------------------------------------------------------

        if n_annotations >= 2:

            annotation_diffs = np.diff(
                annotation_samples
            )

            spacing_ok = bool(
                np.all(
                    annotation_diffs == SAMPLES_PER_MINUTE
                )
            )

            min_spacing = int(
                np.min(annotation_diffs)
            )

            max_spacing = int(
                np.max(annotation_diffs)
            )

            n_bad_spacing = int(
                np.sum(
                    annotation_diffs != SAMPLES_PER_MINUTE
                )
            )

        else:

            spacing_ok = False
            min_spacing = np.nan
            max_spacing = np.nan
            n_bad_spacing = np.nan

        # ---------------------------------------------------------------------
        # Annotation starting point
        # ---------------------------------------------------------------------

        if n_annotations > 0:

            first_annotation_sample = int(
                annotation_samples[0]
            )

            last_annotation_sample = int(
                annotation_samples[-1]
            )

            first_annotation_time_sec = (
                first_annotation_sample / fs
            )

            last_annotation_time_sec = (
                last_annotation_sample / fs
            )

        else:

            first_annotation_sample = np.nan
            last_annotation_sample = np.nan
            first_annotation_time_sec = np.nan
            last_annotation_time_sec = np.nan

        # ---------------------------------------------------------------------
        # Annotation coverage
        #
        # If annotation starts at sample 0 and occurs every 6000 samples,
        # N annotations cover N complete minute windows.
        # ---------------------------------------------------------------------

        annotation_coverage_min = float(
            n_annotations
        )

        unlabeled_tail_min = max(
            0.0,
            duration_min - annotation_coverage_min
        )

        # ---------------------------------------------------------------------
        # Class distribution
        # ---------------------------------------------------------------------

        n_apnea = int(
            np.sum(
                np.asarray(annotation_symbols) == "A"
            )
        )

        n_normal = int(
            np.sum(
                np.asarray(annotation_symbols) == "N"
            )
        )

        if n_annotations > 0:

            apnea_prevalence = (
                n_apnea / n_annotations
            )

        else:

            apnea_prevalence = np.nan

        # ---------------------------------------------------------------------
        # Signal quality
        # ---------------------------------------------------------------------

        quality = calculate_signal_quality(ecg)

        # ---------------------------------------------------------------------
        # Store result
        # ---------------------------------------------------------------------

        result.update({

            "fs": fs,

            "n_samples": n_samples,

            "duration_sec": duration_sec,

            "duration_min": duration_min,

            "duration_hr": duration_hr,

            "n_channels": n_channels,

            "signal_names": "|".join(
                record.sig_name
            ),

            "ecg_channel_index": ecg_channel,

            "ecg_channel_method": ecg_channel_method,

            "n_annotations": n_annotations,

            "annotation_symbols": "|".join(
                unique_symbols
            ),

            "first_annotation_sample":
                first_annotation_sample,

            "last_annotation_sample":
                last_annotation_sample,

            "first_annotation_time_sec":
                first_annotation_time_sec,

            "last_annotation_time_sec":
                last_annotation_time_sec,

            "annotation_spacing_ok":
                spacing_ok,

            "min_annotation_spacing_samples":
                min_spacing,

            "max_annotation_spacing_samples":
                max_spacing,

            "n_bad_annotation_spacings":
                n_bad_spacing,

            "annotation_coverage_min":
                annotation_coverage_min,

            "unlabeled_tail_min":
                unlabeled_tail_min,

            "n_apnea":
                n_apnea,

            "n_normal":
                n_normal,

            "apnea_prevalence":
                apnea_prevalence,

            **quality,

        })

    except Exception as exc:

        result["status"] = "ERROR"
        result["error"] = str(exc)

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("ECG TEMPORAL AI — DATASET QUALITY CONTROL")
    print("=" * 80)

    print()
    print("Project directory:")
    print(PROJECT_DIR)

    print()
    print("Data directory:")
    print(DATA_DIR)

    # -------------------------------------------------------------------------
    # Check data directory
    # -------------------------------------------------------------------------

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            f"Data directory does not exist:\n{DATA_DIR}"
        )

    # -------------------------------------------------------------------------
    # Find primary records
    # -------------------------------------------------------------------------

    records = find_primary_records(DATA_DIR)

    print()
    print("Primary records found:", len(records))

    print()
    print("Records:")
    print(records)

    # -------------------------------------------------------------------------
    # Process all records
    # -------------------------------------------------------------------------

    results = []

    for i, record_id in enumerate(records, start=1):

        print()
        print(
            f"[{i:02d}/{len(records):02d}] Processing {record_id}"
        )

        result = process_record(record_id)

        results.append(result)

        if result["status"] == "OK":

            print(
                f"  duration = "
                f"{result['duration_hr']:.2f} h"
            )

            print(
                f"  annotations = "
                f"{result['n_annotations']}"
            )

            print(
                f"  apnea prevalence = "
                f"{result['apnea_prevalence']:.3f}"
            )

            print(
                f"  annotation spacing OK = "
                f"{result['annotation_spacing_ok']}"
            )

            print(
                f"  unlabeled tail = "
                f"{result['unlabeled_tail_min']:.2f} min"
            )

        else:

            print(
                f"  ERROR: {result.get('error')}"
            )

    # -------------------------------------------------------------------------
    # Convert to DataFrame
    # -------------------------------------------------------------------------

    qc_df = pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # Save complete record-level QC
    # -------------------------------------------------------------------------

    record_qc_path = (
        OUTPUT_DIR / "record_qc.csv"
    )

    qc_df.to_csv(
        record_qc_path,
        index=False
    )

    # -------------------------------------------------------------------------
    # Create annotation-level QC
    # -------------------------------------------------------------------------

    annotation_rows = []

    for record_id in records:

        try:

            record = wfdb.rdrecord(
                str(DATA_DIR / record_id)
            )

            annotation = wfdb.rdann(
                str(DATA_DIR / record_id),
                extension="apn"
            )

            fs = float(record.fs)

            for idx, (sample, symbol) in enumerate(
                zip(
                    annotation.sample,
                    annotation.symbol
                )
            ):

                annotation_rows.append({

                    "subject_id":
                        record_id,

                    "minute_index":
                        idx,

                    "annotation_sample":
                        int(sample),

                    "annotation_time_sec":
                        float(sample / fs),

                    "annotation_time_min":
                        float(sample / fs / 60.0),

                    "symbol":
                        symbol,

                    "label":
                        1 if symbol == "A" else 0,

                    "expected_window_start_sample":
                        idx * SAMPLES_PER_MINUTE,

                    "expected_window_end_sample":
                        (idx + 1) * SAMPLES_PER_MINUTE,

                })

        except Exception:
            continue

    annotation_df = pd.DataFrame(
        annotation_rows
    )

    annotation_qc_path = (
        OUTPUT_DIR / "annotation_qc.csv"
    )

    annotation_df.to_csv(
        annotation_qc_path,
        index=False
    )

    # -------------------------------------------------------------------------
    # Dataset-level summary
    # -------------------------------------------------------------------------

    successful = qc_df[
        qc_df["status"] == "OK"
    ].copy()

    print()
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    print(
        "Primary records:",
        len(records)
    )

    print(
        "Successfully processed:",
        len(successful)
    )

    print(
        "Failed:",
        len(records) - len(successful)
    )

    if len(successful) > 0:

        print()

        print(
            "Total ECG duration (hours):",
            round(
                successful["duration_hr"].sum(),
                3
            )
        )

        print(
            "Mean recording duration (hours):",
            round(
                successful["duration_hr"].mean(),
                3
            )
        )

        print(
            "Sampling rates:",
            sorted(
                successful["fs"].unique().tolist()
            )
        )

        print()

        print(
            "Total apnea minutes:",
            int(
                successful["n_apnea"].sum()
            )
        )

        print(
            "Total normal minutes:",
            int(
                successful["n_normal"].sum()
            )
        )

        total_annotated = (
            successful["n_annotations"].sum()
        )

        total_apnea = (
            successful["n_apnea"].sum()
        )

        if total_annotated > 0:

            print(
                "Overall apnea prevalence:",
                round(
                    total_apnea /
                    total_annotated,
                    4
                )
            )

        print()

        print(
            "Records with correct annotation spacing:",
            int(
                successful[
                    "annotation_spacing_ok"
                ].sum()
            ),
            "/",
            len(successful)
        )

        print(
            "Total unlabeled ECG tail (minutes):",
            round(
                successful[
                    "unlabeled_tail_min"
                ].sum(),
                2
            )
        )

    # -------------------------------------------------------------------------
    # Print record summary table
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("RECORD SUMMARY")
    print("=" * 80)

    display_columns = [
        "subject_id",
        "duration_hr",
        "fs",
        "n_annotations",
        "n_apnea",
        "n_normal",
        "apnea_prevalence",
        "annotation_spacing_ok",
        "unlabeled_tail_min",
        "nan_fraction",
        "flat_fraction",
    ]

    print(
        qc_df[display_columns].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Output locations
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(record_qc_path)
    print(annotation_qc_path)

    print()
    print("QC COMPLETE.")


if __name__ == "__main__":
    main()
