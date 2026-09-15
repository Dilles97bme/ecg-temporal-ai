"""
ROBUSTNESS AND REPRODUCIBILITY

Checks:
1. Signal robustness: NaN/Inf, flatness, extreme values
2. Channel availability: ECG / respiratory / SpO2
3. Deterministic seed configuration
4. Environment/version capture
5. Reproducibility summary

Outputs:
outputs/robustness/
"""

import os
import sys
import json
import random
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# Optional torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Optional wfdb
try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    wfdb = None
    WFDB_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/ecg_temporal_ai")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "robustness"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

PRIMARY_RECORDS = [
    'a01', 'a02', 'a03', 'a05', 'a06', 'a07', 'a08', 'a09',
    'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a18', 'a19',
    'a20', 'b01', 'b02', 'b04', 'b05', 'c01', 'c02', 'c03',
    'c04', 'c06', 'c07', 'c08', 'c09', 'c10'
]


# =============================================================================
# RANDOM SEED CONFIGURATION
# =============================================================================

def set_global_seed(seed=42):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Deterministic behavior where supported
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


set_global_seed(SEED)


# =============================================================================
# SIGNAL QUALITY CHECKS
# =============================================================================

def signal_quality_metrics(signal):

    signal = np.asarray(signal, dtype=np.float64).reshape(-1)

    n = len(signal)

    finite_mask = np.isfinite(signal)

    n_nonfinite = int((~finite_mask).sum())

    if finite_mask.sum() == 0:
        return {
            "n_samples": n,
            "nan_inf_count": n_nonfinite,
            "nan_inf_fraction": 1.0,
            "std": np.nan,
            "range": np.nan,
            "flat_fraction": 1.0,
            "extreme_fraction": np.nan,
        }

    x = signal[finite_mask]

    std = float(np.std(x))
    value_range = float(np.max(x) - np.min(x))

    # Flatness criterion:
    # essentially no variation
    flat = np.isclose(x, np.median(x), atol=1e-10)

    flat_fraction = float(np.mean(flat))

    # Robust extreme-value indicator using median/MAD
    median = np.median(x)
    mad = np.median(np.abs(x - median))

    if mad > 0:
        robust_z = np.abs(x - median) / (1.4826 * mad)
        extreme_fraction = float(np.mean(robust_z > 10))
    else:
        extreme_fraction = 0.0

    return {
        "n_samples": n,
        "nan_inf_count": n_nonfinite,
        "nan_inf_fraction": n_nonfinite / max(n, 1),
        "std": std,
        "range": value_range,
        "flat_fraction": flat_fraction,
        "extreme_fraction": extreme_fraction,
    }


# =============================================================================
# CHANNEL AVAILABILITY
# =============================================================================

def classify_channel(name):

    name_lower = name.lower()

    if any(k in name_lower for k in ["ecg", "ekg"]):
        return "ECG"

    if any(k in name_lower for k in [
        "resp", "respiration", "thorax", "abdomen", "airflow"
    ]):
        return "RESP"

    if any(k in name_lower for k in [
        "spo2", "sao2", "oxygen", "o2"
    ]):
        return "SPO2"

    return "OTHER"


def inspect_record(record_id):

    base = RAW_DIR / record_id

    result = {
        "subject_id": record_id,
        "record_available": False,
        "channels": [],
        "n_channels": np.nan,
        "ecg_channels": [],
        "resp_channels": [],
        "spo2_channels": [],
        "other_channels": [],
        "sampling_rate": np.nan,
        "duration_sec": np.nan,
        "error": "",
    }

    if not WFDB_AVAILABLE:
        result["error"] = "wfdb_not_installed"
        return result

    try:
        record = wfdb.rdrecord(str(base))

        result["record_available"] = True
        result["sampling_rate"] = float(record.fs)
        result["duration_sec"] = float(record.sig_len / record.fs)

        channel_names = list(record.sig_name)

        result["channels"] = channel_names
        result["n_channels"] = len(channel_names)

        for ch in channel_names:

            category = classify_channel(ch)

            if category == "ECG":
                result["ecg_channels"].append(ch)
            elif category == "RESP":
                result["resp_channels"].append(ch)
            elif category == "SPO2":
                result["spo2_channels"].append(ch)
            else:
                result["other_channels"].append(ch)

        # Signal robustness check for each channel
        signal_rows = []

        for idx, ch in enumerate(channel_names):

            signal = record.p_signal[:, idx]

            metrics = signal_quality_metrics(signal)

            row = {
                "subject_id": record_id,
                "channel": ch,
                **metrics,
            }

            signal_rows.append(row)

        return result, signal_rows

    except Exception as e:

        result["error"] = repr(e)

        return result, []


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("STEP 12 — ROBUSTNESS AND REPRODUCIBILITY")
    print("=" * 80)

    print("\nSeed configuration")
    print("-" * 80)
    print(f"PYTHONHASHSEED = {os.environ.get('PYTHONHASHSEED')}")
    print(f"Python random seed = {SEED}")
    print(f"NumPy random seed = {SEED}")

    if TORCH_AVAILABLE:
        print(f"PyTorch seed = {SEED}")
        print(f"CUDA available = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU count = {torch.cuda.device_count()}")
            print(f"GPU = {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch not available")

    print("\nInspecting records...")
    print("-" * 80)

    record_rows = []
    signal_rows = []

    for record_id in PRIMARY_RECORDS:

        print(f"Checking {record_id} ...")

        inspected = inspect_record(record_id)

        if len(inspected) == 2:
            record_info, signal_info = inspected
        else:
            record_info = inspected
            signal_info = []

        record_rows.append(record_info)
        signal_rows.extend(signal_info)

    record_df = pd.DataFrame(record_rows)
    signal_df = pd.DataFrame(signal_rows)

    # Save channel lists as readable strings
    for col in [
        "channels",
        "ecg_channels",
        "resp_channels",
        "spo2_channels",
        "other_channels",
    ]:
        if col in record_df.columns:
            record_df[col] = record_df[col].apply(
                lambda x: ",".join(x) if isinstance(x, list) else x
            )

    record_df.to_csv(
        OUTPUT_DIR / "record_channel_robustness.csv",
        index=False
    )

    signal_df.to_csv(
        OUTPUT_DIR / "signal_quality_robustness.csv",
        index=False
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    total_records = len(record_df)
    successful_records = int(record_df["record_available"].sum())

    resp_records = int(
        record_df["resp_channels"].fillna("").apply(
            lambda x: len(x.strip()) > 0
        ).sum()
    )

    spo2_records = int(
        record_df["spo2_channels"].fillna("").apply(
            lambda x: len(x.strip()) > 0
        ).sum()
    )

    ecg_records = int(
        record_df["ecg_channels"].fillna("").apply(
            lambda x: len(x.strip()) > 0
        ).sum()
    )

    noisy_rows = int(
        (
            (signal_df["nan_inf_count"] > 0)
            | (signal_df["extreme_fraction"] > 0.01)
        ).sum()
    )

    flat_rows = int(
        (signal_df["flat_fraction"] > 0.95).sum()
    )

    summary = {
        "seed": SEED,
        "n_records_requested": total_records,
        "n_records_successfully_read": successful_records,
        "records_with_ecg": ecg_records,
        "records_with_resp": resp_records,
        "records_with_spo2": spo2_records,
        "channels_with_nonfinite_samples": int(
            (signal_df["nan_inf_count"] > 0).sum()
        ),
        "channels_flagged_extreme": int(
            (signal_df["extreme_fraction"] > 0.01).sum()
        ),
        "channels_flagged_flat": flat_rows,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", None)
        if TORCH_AVAILABLE else None,
        "numpy_version": np.__version__,
        "wfdb_version": getattr(wfdb, "__version__", None)
        if WFDB_AVAILABLE else None,
    }

    with open(
        OUTPUT_DIR / "robustness_summary.json",
        "w"
    ) as f:
        json.dump(summary, f, indent=2)

    # Also write a human-readable text report
    with open(
        OUTPUT_DIR / "ROBUSTNESS_REPORT.txt",
        "w"
    ) as f:

        f.write("STEP 12 — ROBUSTNESS AND REPRODUCIBILITY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Random seed: {SEED}\n")
        f.write(f"Requested records: {total_records}\n")
        f.write(f"Successfully read: {successful_records}\n")
        f.write(f"Records with ECG: {ecg_records}\n")
        f.write(f"Records with respiratory channel: {resp_records}\n")
        f.write(f"Records with SpO2 channel: {spo2_records}\n\n")

        f.write("Signal robustness\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"Channels with NaN/Inf samples: "
            f"{int((signal_df['nan_inf_count'] > 0).sum())}\n"
        )
        f.write(
            f"Channels flagged for extreme-value fraction: "
            f"{int((signal_df['extreme_fraction'] > 0.01).sum())}\n"
        )
        f.write(
            f"Channels flagged as predominantly flat: "
            f"{flat_rows}\n\n"
        )

        f.write("Handling policy\n")
        f.write("-" * 60 + "\n")
        f.write(
            "Non-finite signal values are detected and handled by the "
            "existing preprocessing interpolation policy.\n"
        )
        f.write(
            "Flat or severely corrupted segments are flagged rather than "
            "causing the pipeline to crash.\n"
        )
        f.write(
            "The current supervised pipeline uses ECG only. Respiratory "
            "and SpO2 channels are inspected when present but are not "
            "required for execution.\n"
        )
        f.write(
            "The dataset is therefore robust to absent respiratory or "
            "SpO2 channels at the channel-loading stage.\n"
        )
        f.write(
            "No silent dropping of records is permitted: failures are "
            "recorded in the robustness output files.\n"
        )

    print("\n" + "=" * 80)
    print("ROBUSTNESS SUMMARY")
    print("=" * 80)

    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nOutputs saved to:")
    print(OUTPUT_DIR)

    print("\nSTEP 12 COMPLETE")


if __name__ == "__main__":
    main()
