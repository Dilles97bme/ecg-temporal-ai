
"""
===============================================================================
04_PREPROCESSING.PY
===============================================================================

Temporal AI for ECG-Based Event Detection and Short-Horizon Prediction


Pipeline:

    Raw WFDB ECG
        |
        v
    finite-value handling
        |
        v
    Butterworth band-pass filter
        |
        v
    robust normalization
        |
        v
    R-peak detection
        |
        v
    RR interval calculation
        |
        v
    visual / numerical QC

Important:
    - ECG is sampled at 100 Hz.
    - The supplied .qrs annotations are NOT used.
    - This script is a preprocessing/QC script only.
    - It does not train a model.
    - It does not use apnea labels.
    - It does not use future information.

===============================================================================
"""

from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import wfdb
import neurokit2 as nk

from scipy.signal import butter, sosfiltfilt


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(
    "/content/drive/MyDrive/ecg_temporal_ai"
)

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "preprocessing"
)

FIGURE_DIR = (
    PROJECT_DIR
    / "outputs"
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 42

FS = 100

LOWCUT_HZ = 0.5
HIGHCUT_HZ = 40.0

FILTER_ORDER = 4


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# LOAD ECG
# =============================================================================

def load_ecg(record_id):
    """
    Load the primary ECG channel from a WFDB record.
    """

    record_path = str(
        DATA_DIR / record_id
    )

    record = wfdb.rdrecord(
        record_path
    )

    # -------------------------------------------------------------------------
    # Find ECG channel.
    # -------------------------------------------------------------------------

    ecg_channel = None

    for i, name in enumerate(
        record.sig_name
    ):

        if "ecg" in name.lower():

            ecg_channel = i
            break

    # Fallback to channel zero.
    if ecg_channel is None:

        ecg_channel = 0

    ecg = np.asarray(
        record.p_signal[:, ecg_channel],
        dtype=np.float64
    )

    return ecg, float(record.fs)


# =============================================================================
# HANDLE NON-FINITE VALUES
# =============================================================================

def handle_nonfinite(ecg):
    """
    Replace non-finite values safely.

    For isolated missing values, linear interpolation is used.

    If the entire signal is invalid, raise an error.

    The number of non-finite samples is returned for QC.
    """

    x = np.asarray(
        ecg,
        dtype=np.float64
    ).copy()

    finite = np.isfinite(x)

    n_invalid = int(
        np.sum(~finite)
    )

    if n_invalid == 0:

        return x, 0

    if np.sum(finite) == 0:

        raise ValueError(
            "ECG contains no finite samples."
        )

    indices = np.arange(
        len(x)
    )

    x[~finite] = np.interp(
        indices[~finite],
        indices[finite],
        x[finite]
    )

    return x, n_invalid


# =============================================================================
# BAND-PASS FILTER
# =============================================================================

def bandpass_filter(
    ecg,
    fs=FS,
    lowcut=LOWCUT_HZ,
    highcut=HIGHCUT_HZ,
    order=FILTER_ORDER,
):
    """
    Zero-phase Butterworth band-pass filter.

    sosfiltfilt is used to avoid phase distortion.
    """

    nyquist = fs / 2.0

    if not (
        0 < lowcut < highcut < nyquist
    ):

        raise ValueError(
            f"Invalid filter range: "
            f"{lowcut}-{highcut} Hz "
            f"for sampling rate {fs} Hz."
        )

    sos = butter(
        order,
        [
            lowcut / nyquist,
            highcut / nyquist,
        ],
        btype="bandpass",
        output="sos",
    )

    filtered = sosfiltfilt(
        sos,
        ecg
    )

    return filtered


# =============================================================================
# ROBUST NORMALIZATION
# =============================================================================

def robust_normalize(ecg):
    """
    Median/MAD normalization.

    Robust to large amplitude outliers.
    """

    x = np.asarray(
        ecg,
        dtype=np.float64
    )

    median = np.median(x)

    mad = np.median(
        np.abs(x - median)
    )

    # Convert MAD to a standard-deviation-like scale.
    robust_scale = 1.4826 * mad

    if (
        not np.isfinite(robust_scale)
        or robust_scale < 1e-10
    ):

        # Fall back to standard deviation.
        std = np.std(x)

        if (
            not np.isfinite(std)
            or std < 1e-10
        ):

            raise ValueError(
                "ECG segment has essentially zero variation."
            )

        robust_scale = std

    normalized = (
        x - median
    ) / robust_scale

    return normalized


# =============================================================================
# COMPLETE PREPROCESSING
# =============================================================================

def preprocess_ecg(ecg, fs):
    """
    Complete ECG preprocessing pipeline.
    """

    original = np.asarray(
        ecg,
        dtype=np.float64
    )

    cleaned, n_invalid = (
        handle_nonfinite(original)
    )

    filtered = bandpass_filter(
        cleaned,
        fs=fs
    )

    normalized = robust_normalize(
        filtered
    )

    return {
        "raw": original,
        "finite": cleaned,
        "filtered": filtered,
        "normalized": normalized,
        "n_invalid": n_invalid,
    }


# =============================================================================
# R-PEAK DETECTION
# =============================================================================

def detect_r_peaks(
    ecg,
    fs=FS,
    chunk_minutes=5,
):
    """
    Detect R-peaks using NeuroKit2 in manageable chunks.

    Chunking is used to avoid running the detector on an entire
    multi-hour ECG recording at once.

    A small overlap is used between chunks so that beats near
    chunk boundaries are not missed.
    """

    ecg = np.asarray(
        ecg,
        dtype=np.float64
    )

    chunk_samples = int(
        chunk_minutes * 60 * fs
    )

    # 10-second overlap on both sides
    overlap_samples = int(
        10 * fs
    )

    all_rpeaks = []

    n_samples = len(ecg)

    start = 0
    chunk_number = 0

    while start < n_samples:

        chunk_number += 1

        end = min(
            start + chunk_samples,
            n_samples
        )

        # Add overlap around the chunk
        detect_start = max(
            0,
            start - overlap_samples
        )

        detect_end = min(
            n_samples,
            end + overlap_samples
        )

        chunk = ecg[
            detect_start:detect_end
        ]

        try:

            _, info = nk.ecg_process(
                chunk,
                sampling_rate=fs,
            )

            local_peaks = np.asarray(
                info["ECG_R_Peaks"],
                dtype=int
            )

            # Convert chunk-local indices
            # back to full-record indices
            global_peaks = (
                local_peaks
                + detect_start
            )

            # Keep only peaks belonging to the
            # central non-overlapping region
            keep = (
                (global_peaks >= start)
                &
                (global_peaks < end)
            )

            global_peaks = global_peaks[keep]

            all_rpeaks.extend(
                global_peaks.tolist()
            )

        except Exception as exc:

            warnings.warn(
                f"R-peak detection failed "
                f"for chunk {chunk_number}: {exc}"
            )

        start = end

    if len(all_rpeaks) == 0:

        return (
            np.array([], dtype=int),
            "neurokit2_chunked"
        )

    # Remove duplicate peaks and sort
    rpeaks = np.unique(
        np.asarray(
            all_rpeaks,
            dtype=int
        )
    )

    return (
        rpeaks,
        "neurokit2_chunked"
    )

# =============================================================================
# RR INTERVALS
# =============================================================================


def calculate_rr_intervals(
    rpeaks,
    fs=FS,
):
    """
    Calculate successive RR intervals in seconds.
    """

    rpeaks = np.asarray(
        rpeaks,
        dtype=int
    )

    if len(rpeaks) < 2:

        return np.array(
            [],
            dtype=float
        )

    rr = np.diff(
        rpeaks
    ) / fs

    return rr


# =============================================================================
# BASIC PEAK QC
# =============================================================================

def peak_quality_statistics(
    rpeaks,
    rr_intervals,
    duration_sec,
):
    """
    Basic numerical R-peak QC.

    Physiologically implausible RR intervals are NOT silently removed here.
    They are counted so that later feature extraction can apply explicit
    artifact handling.
    """

    n_peaks = len(rpeaks)

    if duration_sec > 0:

        peaks_per_minute = (
            n_peaks /
            duration_sec *
            60.0
        )

    else:

        peaks_per_minute = np.nan

    if len(rr_intervals) > 0:

        rr_min = float(
            np.min(rr_intervals)
        )

        rr_max = float(
            np.max(rr_intervals)
        )

        rr_median = float(
            np.median(rr_intervals)
        )

        rr_invalid_fraction = float(
            np.mean(
                (rr_intervals < 0.3)
                |
                (rr_intervals > 2.0)
            )
        )

    else:

        rr_min = np.nan
        rr_max = np.nan
        rr_median = np.nan
        rr_invalid_fraction = 1.0

    return {

        "n_rpeaks":
            n_peaks,

        "rpeaks_per_minute":
            peaks_per_minute,

        "rr_min_sec":
            rr_min,

        "rr_median_sec":
            rr_median,

        "rr_max_sec":
            rr_max,

        "rr_invalid_fraction":
            rr_invalid_fraction,

    }


# =============================================================================
# PLOT ECG
# =============================================================================

def plot_ecg_qc(
    record_id,
    raw,
    filtered,
    normalized,
    rpeaks,
    fs,
    seconds=10,
):
    """
    Plot the first `seconds` of the ECG.

    Three separate figures are produced:
        1. raw ECG
        2. filtered ECG
        3. normalized ECG with R-peaks
    """

    n = min(
        len(raw),
        int(seconds * fs)
    )

    time = (
        np.arange(n) / fs
    )

    # -------------------------------------------------------------------------
    # Raw ECG
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(16, 4)
    )

    plt.plot(
        time,
        raw[:n]
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        f"{record_id} — Raw ECG ({seconds} s)"
    )

    plt.tight_layout()

    raw_path = (
        FIGURE_DIR
        / f"{record_id}_raw_ecg.png"
    )

    plt.savefig(
        raw_path,
        dpi=150
    )

    plt.show()

    plt.close()

    # -------------------------------------------------------------------------
    # Filtered ECG
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(16, 4)
    )

    plt.plot(
        time,
        filtered[:n]
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        f"{record_id} — Filtered ECG "
        f"({LOWCUT_HZ}-{HIGHCUT_HZ} Hz)"
    )

    plt.tight_layout()

    filtered_path = (
        FIGURE_DIR
        / f"{record_id}_filtered_ecg.png"
    )

    plt.savefig(
        filtered_path,
        dpi=150
    )

    plt.show()

    plt.close()

    # -------------------------------------------------------------------------
    # Normalized ECG + R-peaks
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(16, 4)
    )

    plt.plot(
        time,
        normalized[:n]
    )

    visible_peaks = rpeaks[
        rpeaks < n
    ]

    plt.scatter(
        visible_peaks / fs,
        normalized[visible_peaks],
        s=25,
        marker="o",
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Normalized amplitude"
    )

    plt.title(
        f"{record_id} — Normalized ECG + detected R-peaks"
    )

    plt.tight_layout()

    peak_path = (
        FIGURE_DIR
        / f"{record_id}_rpeaks.png"
    )

    plt.savefig(
        peak_path,
        dpi=150
    )

    plt.show()

    plt.close()

    return (
        raw_path,
        filtered_path,
        peak_path
    )


# =============================================================================
# PROCESS ONE RECORD
# =============================================================================

def process_record(
    record_id,
    make_plots=True,
):

    print()
    print("=" * 80)
    print(
        f"PREPROCESSING RECORD: {record_id}"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    ecg, fs = load_ecg(
        record_id
    )

    print(
        "Sampling rate:",
        fs,
        "Hz"
    )

    print(
        "Samples:",
        len(ecg)
    )

    print(
        "Duration:",
        round(
            len(ecg) / fs / 60,
            2
        ),
        "minutes"
    )

    # -------------------------------------------------------------------------
    # Preprocess
    # -------------------------------------------------------------------------

    processed = preprocess_ecg(
        ecg,
        fs
    )

    print(
        "Invalid samples handled:",
        processed["n_invalid"]
    )

    # -------------------------------------------------------------------------
    # R-peaks
    # -------------------------------------------------------------------------

    rpeaks, detector = detect_r_peaks(
        processed["normalized"],
        fs
    )

    rr = calculate_rr_intervals(
        rpeaks,
        fs
    )

    stats = peak_quality_statistics(
        rpeaks,
        rr,
        len(ecg) / fs
    )

    stats.update({

        "subject_id":
            record_id,

        "fs":
            fs,

        "n_samples":
            len(ecg),

        "duration_min":
            len(ecg) / fs / 60,

        "n_invalid_samples":
            processed["n_invalid"],

        "rpeak_detector":
            detector,

    })

    print()
    print("R-peak detector:")
    print(detector)

    print(
        "Detected R-peaks:",
        stats["n_rpeaks"]
    )

    print(
        "Approximate heart beats/minute:",
        round(
            stats["rpeaks_per_minute"],
            2
        )
    )

    print(
        "Median RR:",
        round(
            stats["rr_median_sec"],
            4
        ),
        "sec"
    )

    print(
        "RR range:",
        round(
            stats["rr_min_sec"],
            4
        ),
        "-",
        round(
            stats["rr_max_sec"],
            4
        ),
        "sec"
    )

    print(
        "RR outside 0.3-2.0 sec:",
        round(
            stats["rr_invalid_fraction"] * 100,
            2
        ),
        "%"
    )

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    if make_plots:

        plot_ecg_qc(
            record_id,
            processed["raw"],
            processed["filtered"],
            processed["normalized"],
            rpeaks,
            fs,
            seconds=10,
        )

    return processed, rpeaks, rr, stats


def main():

    print("=" * 80)
    print("ECG TEMPORAL AI — ECG PREPROCESSING QC")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # FROZEN RECORD LIST
    # -------------------------------------------------------------------------

    records = [
        "a01", "a02", "a03", "a05", "a06", "a07", "a08", "a09",
        "a11", "a12", "a13", "a14", "a15", "a16", "a18", "a19",
        "a20", "b01", "b02", "b04", "b05", "c01", "c02", "c03",
        "c04", "c06", "c07", "c08", "c09", "c10"
    ]

    all_stats = []

    # -------------------------------------------------------------------------
    # PROCESS ALL RECORDS
    # -------------------------------------------------------------------------

    for i, record_id in enumerate(records, start=1):

        print()
        print("#" * 80)
        print(f"PROCESSING RECORD {i}/{len(records)}: {record_id}")
        print("#" * 80)

        try:

            processed, rpeaks, rr, stats = process_record(
                record_id,
                make_plots=False
            )

            # -----------------------------------------------------------------
            # Save R-peaks
            # -----------------------------------------------------------------

            rpeak_df = pd.DataFrame({
                "sample": rpeaks,
                "time_sec": rpeaks / FS,
            })

            rpeak_file = (
                OUTPUT_DIR
                / f"{record_id}_rpeaks.csv"
            )

            rpeak_df.to_csv(
                rpeak_file,
                index=False
            )

            # -----------------------------------------------------------------
            # Save RR intervals
            # -----------------------------------------------------------------

            rr_df = pd.DataFrame({
                "rr_interval_sec": rr,
            })

            rr_file = (
                OUTPUT_DIR
                / f"{record_id}_rr_intervals.csv"
            )

            rr_df.to_csv(
                rr_file,
                index=False
            )

            # -----------------------------------------------------------------
            # Save individual QC
            # -----------------------------------------------------------------

            stats_df = pd.DataFrame(
                [stats]
            )

            stats_file = (
                OUTPUT_DIR
                / f"{record_id}_preprocessing_qc.csv"
            )

            stats_df.to_csv(
                stats_file,
                index=False
            )

            all_stats.append(stats)

            print(f"[SUCCESS] {record_id}")

        except Exception as exc:

            print(
                f"[FAILED] {record_id}: {exc}"
            )

            all_stats.append({
                "subject_id": record_id,
                "processing_status": "FAILED",
                "error": str(exc),
            })

    # -------------------------------------------------------------------------
    # COMBINED QC TABLE
    # -------------------------------------------------------------------------

    qc_df = pd.DataFrame(all_stats)

    qc_file = (
        OUTPUT_DIR
        / "all_records_preprocessing_qc.csv"
    )

    qc_df.to_csv(
        qc_file,
        index=False
    )

    # -------------------------------------------------------------------------
    # OUTPUT SUMMARY
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("PREPROCESSING SUMMARY")
    print("=" * 80)

    print(
        "Records attempted:",
        len(records)
    )

    print(
        "Records completed:",
        len(qc_df)
    )

    if "processing_status" in qc_df.columns:

        print(
            "Failed records:",
            int(
                (qc_df["processing_status"] == "FAILED").sum()
            )
        )

    print()
    print("Combined QC file:")
    print(qc_file)

    print()
    print("PREPROCESSING QC COMPLETE.")


if __name__ == "__main__":
    main()
