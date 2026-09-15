# Temporal AI for ECG-Based Event Detection and Short-Horizon Prediction

## 1. Data, preprocessing, and task definition

The supplied private dataset contains 30 overnight WFDB recordings, all with a single ECG channel sampled at 100 Hz and one-minute apnea annotations. All 30 records were read successfully: 244.61 h of ECG, 14,644 annotated minutes, 5,727 apnea minutes (39.11%) and 8,917 non-apnea minutes. Annotation positions were exactly 6,000 samples apart in every record, matching 60 s at 100 Hz. No NaN/Inf ECG samples were found. A total of 44.42 min of ECG extended beyond annotation coverage and was excluded rather than assigned artificial labels. The supplied primary records had no respiratory or SpO2 inputs, so the final pipeline used ECG only; no external dataset was used.

ECG preprocessing used a fourth-order Butterworth 0.5-40 Hz band-pass filter with zero-phase filtering, followed by per-record median/MAD normalization. NeuroKit2 R-peak detection was performed in 5-min chunks with overlap; invalid RR intervals outside 0.3-2.0 s were excluded from HR/HRV calculation. The eight retained minute-level features were mean RR, median RR, mean HR, median HR, SDNN, RMSSD, pNN50 and RR coefficient of variation.

**Task A (detection):** classify the current 1-min annotated ECG window as apnea/non-apnea. **Task B (prediction):** for minute $m$, use only the preceding five minutes $[m-5,m)$ and predict a new apnea onset in minute $m$. A positive Task B target therefore requires an $N\rightarrow A$ transition; an event already in progress at prediction time is not counted. Because labels are available only at 1-min resolution, the minute boundary is the operational onset time.

## 2. Split, leakage prevention, and imbalance

Splitting was performed at the record level to prevent correlated windows from the same overnight recording crossing partitions. The frozen split was 20 train records, 5 validation records (`a18, a01, b01, c07, c01`) and 5 test records (`a12, a14, b02, c08, c06`). Task A prevalence was closely matched across train/validation/test (39.09%, 39.15%, 39.13%). The test set was isolated before model development and was not used for feature selection, hyperparameter tuning, threshold selection, early stopping or model selection.

Task B is highly imbalanced: 216/9,582 positives in train (2.25%), 19/2,353 in validation (0.81%) and 29/2,559 in test (1.13%). The primary metric was AUPRC, interpreted against the positive-prevalence baseline. Logistic Regression used class balancing; the TCN used class-weighted BCE with weights calculated from training data only. Operating thresholds were selected on validation data and then frozen for test evaluation.

## 3. Baseline and temporal model

The non-deep baseline was feature-based Logistic Regression. A Random Forest was also screened during validation but had lower AUPRC on both tasks (Task A 0.3971 vs 0.4794 for Logistic Regression; Task B 0.0367 vs 0.0669), so Logistic Regression was retained as the baseline for the frozen test comparison. Task A used the current eight HR/HRV features; Task B represented the five-minute history as 40 features.

The temporal model was a lightweight residual TCN with three dilated blocks (dilations 1, 2, 4), 32 hidden channels, dropout 0.15, global temporal pooling and a small classifier. It used five sequential HR/HRV minutes rather than raw ECG. Training used AdamW (learning rate $10^{-3}$, weight decay $10^{-4}$), batch size 128 and class-weighted BCE. Training and scaling used train data only; validation AUPRC controlled early stopping/model selection and threshold selection. The TCN was selected for both tasks because its validation AUPRC exceeded Logistic Regression (Task A 0.5370 vs 0.4794; Task B 0.1024 vs 0.0669).

## 4. Frozen held-out results

| Task | Model | Test n | AUPRC | Prev. | AUROC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | Logistic Regression | 2,584 | 0.4294 | 0.3913 | 0.4995 | 0.3665 | 0.6706 | 0.4740 |
| A | TCN | 2,564 | **0.4939** | 0.3943 | 0.4216 | 0.2942 | 0.4382 | 0.3520 |
| B | Logistic Regression | 2,559 | **0.0600** | 0.0113 | **0.7583** | **0.1111** | 0.0345 | 0.0526 |
| B | TCN | 2,559 | 0.0483 | 0.0113 | 0.7136 | 0.0000 | 0.0000 | 0.0000 |

For Task A, the TCN evaluated 2,564 windows because a complete five-minute history is required; all 1,011 positive test windows remained represented. Confusion matrices were: Logistic A TP/TN/FP/FN = 678/401/1172/333; TCN = 443/490/1063/568. For Task B, Logistic = 1/2522/8/28 and TCN = 0/2529/1/29. Thus the TCN improved Task A AUPRC but was worse on AUROC, precision, recall, F1 and calibration, while Logistic Regression was better on both primary discrimination metrics for Task B. The temporal model therefore did **not** produce a consistent held-out advantage.

## 5. Error analysis and calibration

Errors were heterogeneous across held-out records rather than concentrated in a single universal operating pattern. For Task B, the dominant failure mode was missed onset prediction: all 28 Logistic Regression false negatives and all 29 TCN false negatives were genuine 0-to-1 onset transitions. Calibration also favored Logistic Regression. Brier/ECE were 0.3067/0.2461 (Task A Logistic), 0.3827/0.3927 (Task A TCN), 0.1137/0.2565 (Task B Logistic) and 0.1392/0.2801 (Task B TCN); lower is better. Task B calibration estimates are necessarily uncertain because only 29 positive test events were available.

## 6. Controls, robustness, and conclusion

The required controls were completed in a separate `ANALYSIS.md`. Shuffled Task B labels produced test AUPRC 0.0093 at a prevalence of 0.0113, providing a negative control close to chance. A current-event detector evaluated on the Task B target minute had test AUROC 0.4979/AUPRC 0.4322, separating current-state detection from prospective onset prediction. A five-minute-history horizon sweep gave test AUROC 0.7339, 0.6253, 0.6238 and 0.6335 for 1-, 2-, 3- and 5-min future horizons, respectively, while prevalence increased with horizon; thresholded detection remained weak. Three TCN seeds gave Task B validation AUPRC $0.0894\pm0.0041$ and AUROC $0.6001\pm0.0172$, indicating that the weak forecasting result was not due to one initialization.

The robustness audit processed all 30 records, explicitly set Python/NumPy/PyTorch seeds, handled absent respiratory/SpO2 channels, and found no NaN/Inf ECG samples. The final evidence supports **limited ranking information about future apnea onset but not reliable short-horizon event prediction** at the validation-selected decision threshold. The TCN did not earn a clear advantage over the feature baseline under the frozen held-out protocol.

## Limitations and next steps

The main limitations are the small number of recordings, only 29 positive Task B test events, one-minute annotation resolution, and reliance on engineered ECG-derived HR/HRV features rather than richer raw physiological representations. With more time, I would evaluate richer ECG temporal/morphology representations, uncertainty estimation, post-hoc calibration, larger subject-level cohorts, and cost-sensitive operating points.
