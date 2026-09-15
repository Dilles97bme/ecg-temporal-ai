import os
import subprocess
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(
    "/content/drive/MyDrive/ecg_temporal_ai"
)

SCRIPT = (
    BASE
    / "src"
    / "13d_train_tcn_seed.py"
)

OUT_DIR = (
    BASE
    / "outputs"
    / "analysis_controls"
    / "tcn_seeds"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEEDS = [
    42,
    123,
    2026,
]


# ============================================================
# RUN THREE SEEDS
# ============================================================

print(
    "=" * 80
)

print(
    "STEP 13D — THREE-SEED TCN REPEATABILITY"
)

print(
    "=" * 80
)

print(
    "Seeds:",
    SEEDS
)


for seed in SEEDS:

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"RUNNING TCN WITH SEED = {seed}"
    )

    print(
        "=" * 80
    )

    env = os.environ.copy()

    env["TCN_SEED"] = str(
        seed
    )

    result = subprocess.run(
        [
            "python",
            str(SCRIPT),
        ],
        env=env,
        check=False,
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"TCN failed for seed "
            f"{seed} with exit code "
            f"{result.returncode}"
        )


# ============================================================
# COLLECT RESULTS
# ============================================================

rows = []


for seed in SEEDS:

    result_file = (
        OUT_DIR
        / f"seed_{seed}"
        / "tcn_validation_results.csv"
    )

    if not result_file.exists():

        raise FileNotFoundError(
            f"Missing results for "
            f"seed {seed}: "
            f"{result_file}"
        )

    df = pd.read_csv(
        result_file
    )

    rows.append(
        df
    )


all_results = pd.concat(
    rows,
    ignore_index=True
)


# Save raw seed-level results
all_results.to_csv(
    OUT_DIR
    / "three_seed_results.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

metrics = [
    "auprc",
    "auroc",
    "precision",
    "recall",
    "f1",
    "brier",
]


summary_rows = []


for task_name, task_df in (
    all_results.groupby("task")
):

    for metric in metrics:

        values = (
            task_df[metric]
            .astype(float)
        )

        mean_value = (
            values.mean()
        )

        std_value = (
            values.std(
                ddof=1
            )
        )

        summary_rows.append(
            {
                "task": task_name,
                "metric": metric,
                "mean": mean_value,
                "std": std_value,
                "mean_plus_minus_sd":
                    f"{mean_value:.4f} "
                    f"± {std_value:.4f}",
            }
        )


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    OUT_DIR
    / "three_seed_summary.csv",
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "THREE-SEED RESULTS"
)

print(
    "=" * 80
)

print(
    all_results[
        [
            "seed",
            "task",
            "best_epoch",
            "auroc",
            "auprc",
            "precision",
            "recall",
            "f1",
            "brier",
            "threshold",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 80
)

print(
    "MEAN ± SD"
)

print(
    "=" * 80
)


for task_name in (
    summary["task"].unique()
):

    print(
        f"\n{task_name}"
    )

    sub = summary[
        summary["task"] == task_name
    ]

    print(
        sub[
            [
                "metric",
                "mean_plus_minus_sd",
            ]
        ].to_string(
            index=False
        )
    )


print(
    "\n"
    + "=" * 80
)

print(
    "13D COMPLETE"
)

print(
    "=" * 80
)

print(
    "\nDetailed results:"
)

print(
    OUT_DIR
    / "three_seed_results.csv"
)

print(
    "\nSummary:"
)

print(
    OUT_DIR
    / "three_seed_summary.csv"
)
