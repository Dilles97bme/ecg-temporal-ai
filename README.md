# Temporal AI for ECG-Based Event Detection and Short-Horizon Prediction

## Project Overview

This repository contains a reproducible pipeline for two temporal binary
classification tasks using the private ECG dataset supplied with the
take-home assignment.

### Task A — Event Detection

Given a one-minute ECG window, predict whether apnea is present in the
current minute.

    Current ECG window → P(apnea in current window)

### Task B — Short-Horizon Prediction

Using only information available during the five minutes before prediction
time `t`, predict whether a new apnea onset will occur during the following
one-minute horizon.

    Five-minute history → P(new apnea onset in next minute)

An apnea episode already in progress at prediction time is not counted as a
future event.

The dataset is treated purely as a binary temporal classification problem.
No external or public version of the dataset is used.

---

## Dataset

The supplied dataset is private and is not included in the repository.

Place the supplied WFDB files under:

    data/raw/

Example primary-record files include:

    a01.dat
    a01.hea
    a01.apn

The supplied dataset may contain associated record files. The analysis
identifies the 30 designated primary recordings and performs the supervised
ECG analysis on those primary records only.

No external or public dataset was used.

---

## Environment

The experiments were developed and run in Google Colab using Google Drive
to persist the private dataset and project outputs.

Primary reproducibility-audit environment:

    Python 3.13.15
    NumPy 2.1.3
    pandas 2.2.3
    SciPy 1.16.3
    scikit-learn 1.6.1
    matplotlib 3.10.0
    WFDB 4.3.1
    NeuroKit2 0.2.13
    PyTorch 2.11.0+cpu
    joblib 1.6.0

The final robustness/reproducibility audit was performed using CPU
execution.

Pinned dependencies are provided in:

    requirements.txt

The submitted scripts currently use the Google Colab/Google Drive project
root:

    /content/drive/MyDrive/ecg_temporal_ai

For execution outside Colab, the project-root paths in the scripts must be
adjusted to the local repository location.

---

## Dataset Quality Control

The initial dataset QC established:

- 30 primary ECG recordings
- 100 Hz sampling rate for all records
- approximately 244.61 total recording hours
- approximately 8.15 hours mean recording duration
- one ECG channel per primary record
- one-minute apnea annotations
- annotation spacing of exactly 6,000 samples
- 5,727 apnea-labelled minutes
- 8,917 non-apnea-labelled minutes
- overall apnea prevalence of 39.11%
- no NaN/Inf ECG samples
- approximately 44.42 minutes of ECG beyond supplied annotation coverage

All 30 primary recordings passed the annotation-spacing check.

The unlabeled ECG tails after the supplied annotation coverage were not
assigned artificial labels and were excluded from supervised training and
evaluation.

Downstream QC and preprocessing outputs are stored under:

    outputs/splits/
    outputs/windowing/
    outputs/preprocessing/
    outputs/features/

---

## Pipeline Structure

The source code follows the assignment section numbering:

    src/
        01_dataset_qc.py
        02_split.py
        03_windowing.py
        04_preprocessing.py
        05_hrv_features.py
        06_merge_features.py
        07_eda.py
        08_baseline_models.py
        09_train_tcn.py
        10_final_test_evaluation.py
        11_error_analysis_calibration.py
        12_robustness_reproducibility.py
        13_analysis_controls.py
        13d_train_tcn_seed.py
        13d_three_seed_repeatability.py

---

## Data Splitting and Leakage Prevention

The dataset is split at the recording/subject level rather than at the
individual window level.

The fixed split is:

### Train — 20 records

    a02, a03, a05, a06, a07, a08, a09, a11, a13, a15,
    a16, a19, a20, b04, b05, c02, c03, c04, c09, c10

### Validation — 5 records

    a18, a01, b01, c07, c01

### Test — 5 records

    a12, a14, b02, c08, c06

No recording occurs in more than one split.

Record-level splitting is necessary because neighboring physiological windows
from the same overnight recording are highly correlated. A random
window-level split could place highly similar observations from one subject
in both training and evaluation sets and produce optimistic estimates of
generalization.

For Task B, five-minute history windows overlap naturally between neighboring
prediction times. This overlap is allowed within a split, but cannot cross
record-level split boundaries.

The test set was frozen before model development and was not used for:

- feature selection
- hyperparameter tuning
- model selection
- early stopping
- threshold selection
- repeated development decisions

The split-generation code also checks that the train, validation, and test
record sets do not intersect.

Run the split-generation step with:

    python src/02_split.py

---

## Preprocessing

The final modeling pipeline uses ECG only.

The supplied primary records did not contain respiratory or SpO2 channels in
the records used for modeling.

The ECG recordings are sampled at 100 Hz, so no resampling was required.

The implemented ECG preprocessing consists of:

- non-finite value handling by interpolation if required
- fourth-order Butterworth band-pass filtering from 0.5 to 40 Hz
- zero-phase filtering
- robust per-record normalization using median and MAD
- R-peak detection using NeuroKit2
- RR-interval calculation
- HR/HRV feature extraction

The supplied `.qrs` files were not used for R-peak extraction.

For long overnight records, R-peak detection is performed in chunks with
overlap to keep processing practical.

RR intervals outside the valid range used by the pipeline are excluded from
HRV feature calculations without changing the original ECG signal.

Preprocessing QC is stored under:

    outputs/preprocessing/

The main preprocessing QC table is:

    outputs/preprocessing/all_records_preprocessing_qc.csv

---

## Window and Label Construction

### Task A — Current Event Detection

Each annotated minute represents a 60-second ECG window:

    [m, m + 60 seconds)

The corresponding annotation is the target:

    0 = non-apnea
    1 = apnea

Annotation alignment was explicitly verified using the 100 Hz sampling rate
and 6,000-sample annotation spacing.

### Task B — Prospective Onset Prediction

For prediction at minute `m`, the model receives the preceding five annotated
minutes:

    [m-5, m-4, m-3, m-2, m-1]

The future target is the minute beginning at `m`.

A positive target is defined by:

    previous label = 0
    current label  = 1

Therefore, an apnea episode that has already started before prediction time
is not counted as a future onset merely because it continues into the
prediction horizon.

Because the ground truth is available only at one-minute resolution, the
exact within-minute physiological onset time cannot be determined. The
N-to-A minute boundary is therefore used as the operational onset time.

The first five annotated minutes of each recording cannot be used for Task B
because the model requires five complete minutes of history.

Task B contains:

    14,494 constructed prediction windows
    264 positive onset events
    1.82% overall positive prevalence

After HR/HRV validity filtering, the primary modeling-valid Task B sets are:

    Train       9,517 windows
    Validation  2,344 windows
    Test        2,559 windows

---

## Class Imbalance

The two tasks have very different positive prevalence.

### Task A

    Train       3,785 / 9,682 positive = 39.09%
    Validation    931 / 2,378 positive = 39.15%
    Test        1,011 / 2,584 positive = 39.13%

### Task B

    Train         216 / 9,582 positive = 2.25%
    Validation     19 / 2,353 positive = 0.81%
    Test            29 / 2,559 positive = 1.13%

Task B is therefore a severe rare-event prediction problem.

Accuracy was not used as the primary metric because a model could obtain a
high accuracy by predominantly predicting the negative class.

Logistic Regression used balanced class weighting.

The TCN used class-weighted binary cross-entropy with the positive weight
calculated from the training set only.

Classification thresholds were selected using validation data only and then
frozen before final test evaluation.

---

## Feature Representation

The feature-based baseline uses eight one-minute HR/HRV features:

    mean_rr_sec
    median_rr_sec
    mean_hr_bpm
    median_hr_bpm
    sdnn_sec
    rmssd_sec
    pnn50
    cv_rr

For Task A, the baseline uses the eight current-minute features.

For Task B, the same eight feature types are available for each of the five
historical minutes, producing:

    5 time steps × 8 features = 40 feature values

The deep temporal model also receives:

    5 time steps × 8 HR/HRV features

The TCN therefore models temporal relationships among engineered HR/HRV
features rather than directly processing the raw 100-Hz ECG waveform.

This design was chosen to remain lightweight and practical on free-tier
Google Colab.

---

## Baseline Model

Two non-deep models were investigated:

    Logistic Regression
    Random Forest

Logistic Regression produced the stronger validation AUPRC and was therefore
used as the primary feature-based reference model.

Validation results:

    Task A:
        Logistic Regression AUPRC = 0.4794
        Random Forest AUPRC       = 0.3971

    Task B:
        Logistic Regression AUPRC = 0.0669
        Random Forest AUPRC       = 0.0367

The test set was not used to select the baseline model.

Baseline outputs are stored under:

    outputs/baseline_results/

---

## Temporal Deep Model

A single lightweight Temporal Convolutional Network (TCN) was used.

The TCN contains three residual temporal convolution blocks with dilation
rates:

    1
    2
    4

Model configuration:

    hidden channels = 32
    dropout = 0.15
    batch size = 128
    optimizer = AdamW
    learning rate = 1e-3
    weight decay = 1e-4

Training uses class-weighted binary cross-entropy.

Task A sequence:

    [t-4, t-3, t-2, t-1, t]

Task B sequence:

    [t-5, t-4, t-3, t-2, t-1]

The scaler is fitted on training data only.

Validation AUPRC is the primary model-selection and early-stopping criterion.

The TCN implementation is:

    src/09_train_tcn.py

The seed-repeatability implementation is:

    src/13d_train_tcn_seed.py

---

## Model Selection

Validation AUPRC was the primary model-selection criterion.

Using this criterion, the TCN was selected over Logistic Regression for both
tasks:

    Task A:
        Logistic Regression validation AUPRC = 0.4794
        TCN validation AUPRC              = 0.5370

    Task B:
        Logistic Regression validation AUPRC = 0.0669
        TCN validation AUPRC              = 0.1024

The frozen test set was not used for this selection.

This is important because the final test results are allowed to show that the
validation-selected TCN did not generalize better than the baseline.

---

## Important Model Comparison Note

The Task A Logistic Regression baseline uses current-minute HR/HRV features,
whereas the TCN uses a five-minute temporal sequence.

Therefore, the Task A comparison is a comparison between current-minute
feature-based modeling and short temporal-sequence modeling, rather than a
pure architecture-only comparison.

The record-level train/validation/test split is identical between the models.

---

## Evaluation Protocol

The primary metric is:

    AUPRC

AUPRC is always interpreted together with positive prevalence as the
no-skill baseline.

Additional metrics are:

    AUROC
    precision
    recall / sensitivity
    F1
    confusion matrix
    Brier score
    expected calibration error (ECE)

Thresholds are selected on the validation set only.

The final test set is evaluated only after model development, model
selection, and threshold selection are complete.

---

## Final Held-Out Test Results

### Task A — Current Apnea Detection

| Model | AUPRC | AUROC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4294 | 0.4995 | 0.3665 | 0.6706 | 0.4740 |
| TCN | 0.4939 | 0.4216 | 0.2942 | 0.4382 | 0.3520 |

### Task B — Future Apnea-Onset Prediction

| Model | AUPRC | AUROC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0600 | 0.7583 | 0.1111 | 0.0345 | 0.0526 |
| TCN | 0.0483 | 0.7136 | 0.0000 | 0.0000 | 0.0000 |

Test-set positive prevalences were approximately:

    Task A Logistic comparison: 39.13%
    Task A TCN comparison:     39.43%
    Task B:                     1.13%

The TCN used 2,564 Task A test windows rather than all 2,584 annotated test
minutes because complete five-minute temporal history was required.
All 1,011 positive Task A test windows remained represented.

The TCN improved Task A AUPRC but had worse AUROC, precision, recall, F1,
and Brier score than Logistic Regression.

For Task B, Logistic Regression outperformed the TCN in both AUPRC and AUROC.

At the frozen validation-selected threshold, the TCN detected none of the
29 positive Task B test onset events.

---

## Error Analysis and Calibration

Error analysis is implemented in:

    src/11_error_analysis_calibration.py

The analysis includes:

- TP/TN/FP/FN categorization
- per-record error summaries
- HR/HRV feature characteristics by error type
- Task B historical-feature analysis
- temporal transition analysis
- reliability diagrams
- Brier score
- expected calibration error

Outputs are stored under:

    outputs/eda/

Final test confusion matrices:

| Model | TP | TN | FP | FN |
|---|---:|---:|---:|---:|
| Task A Logistic | 678 | 401 | 1,172 | 333 |
| Task A TCN | 443 | 490 | 1,063 | 568 |
| Task B Logistic | 1 | 2,522 | 8 | 28 |
| Task B TCN | 0 | 2,529 | 1 | 29 |

For Task B, all Logistic Regression false negatives and all TCN false
negatives corresponded to genuine 0-to-1 onset transitions.

The dominant Task B failure mode was therefore missed onset detection rather
than confusion with already-ongoing apnea.

### Calibration Results

| Model | Brier | ECE |
|---|---:|---:|
| Task A Logistic | 0.3067 | 0.2461 |
| Task A TCN | 0.3827 | 0.3927 |
| Task B Logistic | 0.1137 | 0.2565 |
| Task B TCN | 0.1392 | 0.2801 |

Lower values are better.

Logistic Regression was better calibrated than the TCN for both tasks.

Task B calibration should be interpreted cautiously because only 29 positive
test events were available.

---

## Robustness and Reproducibility

The robustness audit is implemented in:

    src/12_robustness_reproducibility.py

The audit verified:

    30 / 30 records successfully read
    30 / 30 records with ECG
    0 channels with NaN/Inf samples
    0 predominantly flat channels
    0 records with respiratory channels
    0 records with SpO2 channels

Some channels were flagged by the robust extreme-value screening criterion.
These flags were treated as QC indicators rather than automatic exclusion
criteria.

The current pipeline uses ECG only and does not require respiratory or SpO2
inputs.

Random seeds are explicitly controlled for:

    Python
    NumPy
    PyTorch

The final robustness audit was performed using CPU execution.

Robustness outputs are stored under:

    outputs/robustness/

---

## Required Analysis Experiments

The additional analysis required by the assignment is implemented through
the following experiments:

    shuffled-label control
    detection-versus-prediction control
    future-horizon sweep
    three-seed TCN repeatability

The detailed interpretation is documented in:

    ANALYSIS.md

### Shuffled-Label Control

Training labels were randomly shuffled while preserving the same feature
matrix and record-level split.

Test result:

    AUROC = 0.3901
    AUPRC = 0.0093
    prevalence = 0.0113

The shuffled-label AUPRC was close to the prevalence baseline and did not
reproduce the ranking performance obtained using the true labels.

### Detection-versus-Prediction Control

A current-event Logistic Regression detector was applied to the Task B target
minute. This is an oracle/current-event detection control and is not a valid
prospective forecasting model.

Validation:

    AUROC = 0.6094
    AUPRC = 0.4832

Test:

    AUROC = 0.4979
    AUPRC = 0.4322

The control helps distinguish current-event recognition from future-onset
prediction, while the held-out detection result also shows substantial
cross-record generalization difficulty.

### Future-Horizon Sweep

A five-minute history was retained while the future onset horizon was varied:

    1 minute
    2 minutes
    3 minutes
    5 minutes

Test results:

| Horizon | Prevalence | AUROC | AUPRC |
|---:|---:|---:|---:|
| 1 min | 0.0187 | 0.7339 | 0.0554 |
| 2 min | 0.0369 | 0.6253 | 0.0732 |
| 3 min | 0.0519 | 0.6238 | 0.1127 |
| 5 min | 0.0783 | 0.6335 | 0.1390 |

Positive prevalence increases as the horizon becomes longer. Therefore,
AUPRC is interpreted relative to the corresponding horizon-specific
prevalence.

### Three-Seed Repeatability

The TCN was repeated with:

    seed = 42
    seed = 123
    seed = 2026

The same split, architecture, features, hyperparameters, and validation
protocol were retained.

No test data were used during these repeatability runs.

Validation summary:

| Task | AUPRC mean ± SD | AUROC mean ± SD | F1 mean ± SD |
|---|---:|---:|---:|
| Task A | 0.5319 ± 0.0322 | 0.6730 ± 0.0352 | 0.6214 ± 0.0196 |
| Task B | 0.0894 ± 0.0041 | 0.6001 ± 0.0172 | 0.1648 ± 0.0102 |

Task B performance was reasonably stable across seeds, indicating that the
weak forecasting result was not explained solely by one random initialization.

Detailed control outputs are stored under:

    outputs/analysis_controls/

---

## Final Submission Files

The required submission artifacts are:

    predictions.csv
    requirements.txt
    README.md
    REPORT.md
    ANALYSIS.md

### predictions.csv

The required columns are:

    subject_id
    window_start
    window_end
    task
    probability
    prediction

The final prediction file uses the validation-selected TCN for both tasks.

It contains:

    2,564 Task A rows
    2,559 Task B rows
    5,123 rows total

For Task A:

    window_start = beginning of current one-minute detection window
    window_end   = end of current one-minute detection window

For Task B:

    window_start = beginning of five-minute observation history
    window_end   = end of one-minute future prediction horizon

The file contains no missing probabilities or predictions, and every
probability is within [0, 1].

---

## Reproducibility and Execution

The core pipeline can be executed in the following order:

    python src/01_dataset_qc.py
    python src/02_split.py
    python src/03_windowing.py
    python src/04_preprocessing.py
    python src/05_hrv_features.py
    python src/06_merge_features.py
    python src/07_eda.py
    python src/08_baseline_models.py
    python src/09_train_tcn.py
    python src/10_final_test_evaluation.py
    python src/11_error_analysis_calibration.py
    python src/12_robustness_reproducibility.py

The required analysis controls are run separately:

    python src/13_analysis_controls.py

Three-seed TCN repeatability is run with:

    python src/13d_three_seed_repeatability.py

The final dependency versions are pinned in:

    requirements.txt

---

## Output Structure

Important output directories include:

    outputs/
        splits/
        windowing/
        preprocessing/
        features/
        baseline_results/
        tcn_results/
        final_test_results/
        eda/
        robustness/
        analysis_controls/

Final held-out predictions are stored under:

    outputs/final_test_results/

The required repository-level prediction file is:

    predictions.csv

Error-analysis and calibration outputs are stored under:

    outputs/eda/

Robustness outputs are stored under:

    outputs/robustness/

Required analysis-control outputs are stored under:

    outputs/analysis_controls/

---

## Computational Considerations

The pipeline is designed to be practical on free-tier Google Colab.

The temporal model is intentionally lightweight:

    three residual TCN blocks
    32 hidden channels
    five-minute sequences
    8 HR/HRV features per time step

No multi-GPU training is required.

The final robustness audit was performed on CPU.

Runtime can vary depending on the available Colab hardware, CPU allocation,
Google Drive filesystem performance, and environment startup time.

---

## Assumptions and Limitations

1. Apnea annotations are available only at one-minute resolution, so exact
   within-minute physiological onset time is unavailable.

2. Task B uses the N-to-A minute boundary as the operational future-onset
   definition.

3. The supplied records used for modeling contained ECG but no respiratory
   or SpO2 channels.

4. The temporal TCN operates on engineered HR/HRV feature sequences rather
   than directly on raw ECG waveforms.

5. Only 30 recordings are available, with five recordings reserved for the
   final test set.

6. Task B contains only 29 positive test events, making threshold-based
   precision, recall, F1, and calibration estimates sensitive to individual
   events.

7. The three-seed repeatability experiment uses only three seeds.

8. The TCN did not consistently outperform the non-deep baseline on the
   frozen held-out test set.

---

## Final Conclusion

The TCN was selected using validation AUPRC for both tasks, but its validation
advantage did not consistently transfer to the frozen held-out test set.

For Task A, the TCN achieved higher AUPRC but worse AUROC, precision, recall,
F1, and calibration than Logistic Regression.

For Task B, Logistic Regression outperformed the TCN on both AUPRC and AUROC,
and the TCN detected none of the 29 positive onset events at the frozen
validation-selected threshold.

The complete analysis therefore supports a cautious conclusion: the available
ECG/HRV representation contains some information useful for ranking future
onset risk, but the evidence is insufficient to support reliable
short-horizon apnea-onset detection.

The detailed reasoning, controls, and interpretation are provided in
`ANALYSIS.md`.
