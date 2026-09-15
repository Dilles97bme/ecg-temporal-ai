
"""
Run the ECG Temporal AI pipeline end-to-end.

    Usage:
        python run_pipeline.py
    
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PIPELINE = [
    ("01", "Dataset QC", "01_dataset_qc.py"),
    ("02", "Record-level split", "02_split.py"),
    ("03", "Minute windowing", "03_windowing.py"),
    ("04", "ECG preprocessing", "04_preprocessing.py"),
    ("05", "HRV feature extraction", "05_hrv_features.py"),
    ("06", "Feature merge", "06_merge_features.py"),
    ("07", "EDA", "07_eda.py"),
    ("08", "Baseline models", "08_baseline_models.py"),
    ("09", "TCN training", "09_train_tcn.py"),
    ("10", "Final frozen-test evaluation", "10_final_test_evaluation.py"),
    ("11", "Error analysis and calibration", "11_error_analysis_calibration.py"),
    ("12", "Robustness and reproducibility", "12_robustness_reproducibility.py"),
    ("13", "Analysis controls", "13_analysis_controls.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ECG Temporal AI pipeline in dependency order."
    )
    parser.add_argument(
        "--from",
        dest="start",
        default=PIPELINE[0][0],
        help="First stage to run, e.g. 07",
    )
    parser.add_argument(
        "--to",
        dest="end",
        default=PIPELINE[-1][0],
        help="Last stage to run, e.g. 11",
    )
    parser.add_argument(
        "--with-reproducibility",
        action="store_true",
        help="Also run the optional 3-seed TCN repeatability experiment.",
    )
    return parser.parse_args()


def resolve_root() -> Path:
    # run_pipeline.py is expected at the repository root.
    return Path(__file__).resolve().parent


def run_stage(stage_id: str, name: str, script: str, root: Path) -> None:
    script_path = root / "src" / script
    if not script_path.exists():
        raise FileNotFoundError(f"Missing pipeline script: {script_path}")

    print("\n" + "=" * 80)
    print(f"STAGE {stage_id} — {name}")
    print(f"Script: {script_path.relative_to(root)}")
    print("=" * 80)

    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(root),
        env=os.environ.copy(),
        check=False,
    )
    elapsed = time.perf_counter() - started

    if result.returncode != 0:
        raise RuntimeError(
            f"Stage {stage_id} failed with exit code {result.returncode}: {script}"
        )

    print(f"✓ Stage {stage_id} completed in {elapsed / 60:.2f} min")


def main() -> int:
    args = parse_args()
    root = resolve_root()

    if not (root / "src").is_dir():
        print(f"ERROR: src/ directory not found under {root}", file=sys.stderr)
        return 2

    ids = [stage_id for stage_id, _, _ in PIPELINE]
    if args.start not in ids or args.end not in ids:
        print(
            f"ERROR: --from/--to must be one of: {', '.join(ids)}",
            file=sys.stderr,
        )
        return 2

    start_idx = ids.index(args.start)
    end_idx = ids.index(args.end)
    if start_idx > end_idx:
        print("ERROR: --from cannot come after --to.", file=sys.stderr)
        return 2

    stages = PIPELINE[start_idx : end_idx + 1]

    print("=" * 80)
    print("ECG TEMPORAL AI — END-TO-END PIPELINE")
    print("=" * 80)
    print(f"Repository root : {root}")
    print(f"Python          : {sys.executable}")
    print(f"Stages          : {args.start} → {args.end}")
    print("=" * 80)

    total_started = time.perf_counter()

    try:
        for stage_id, name, script in stages:
            run_stage(stage_id, name, script, root)

        if args.with_reproducibility:
            print("\n" + "=" * 80)
            print("OPTIONAL REPRODUCIBILITY EXPERIMENT")
            print("=" * 80)
            for script in (
                "13d_train_tcn_seed.py",
                "13d_three_seed_repeatability.py",
            ):
                run_stage("13D", f"Repeatability: {script}", script, root)

    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\n❌ PIPELINE STOPPED: {exc}", file=sys.stderr)
        return 1

    elapsed_total = time.perf_counter() - total_started

    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total runtime: {elapsed_total / 60:.2f} min")
    print("\nExpected key outputs include:")
    print("  - predictions.csv")
    print("  - outputs/final_test_results/")
    print("  - outputs/eda/")
    print("  - outputs/error_analysis/")
    print("  - outputs/robustness/")
    print("  - outputs/analysis_controls/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
