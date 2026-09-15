
# ANALYSIS.md

# Analysis of Short-Horizon Apnea Prediction

## 1. Question

The central question was whether the large performance gap between current-event detection (Task A) and prospective short-horizon onset prediction (Task B) was caused by a flaw in the detection setup, leakage, or an overly difficult forecasting problem.

The analysis therefore used four controls:

1. shuffled-label control,
2. current-event detection applied to prediction-task target minutes,
3. a future-horizon sweep,
4. three-seed repeatability.

The frozen record-level split was retained throughout. The test set was not used for model fitting, early stopping, threshold tuning, or model selection.

---

## 2. Why detection and prediction were treated as different problems

Task A detects the state at the current minute:

> ECG/HRV information at the current window -> probability of apnea in that window.

Task B is prospective:

> Five minutes of history available before prediction time -> probability of a new apnea onset in the future.

For Task B, an apnea episode already in progress at prediction time was not counted as a future onset. A positive event required an N->A transition after the prediction time.

This distinction is important because detecting an event that is already present can be substantially easier than predicting a new onset before it begins.

---

## 3. Control 1 — Shuffled-label experiment

### Purpose

A shuffled-label negative control was used to test whether the feature pipeline could produce apparent predictive performance when the correspondence between inputs and labels was destroyed.

The Task B HR/HRV features were kept unchanged, while the training labels were randomly permuted. The same train/validation/test record split was retained.

### Results

| Model | Split | AUROC | AUPRC | Prevalence |
|---|---|---:|---:|---:|
| Shuffled-label Logistic | Validation | 0.3465 | 0.0060 | 0.0081 |
| Shuffled-label Logistic | Test | 0.3901 | 0.0093 | 0.0113 |

For comparison, the original Task B Logistic Regression test result was:

- AUROC = 0.7583
- AUPRC = 0.0600
- positive prevalence = 0.0113

### Interpretation

The shuffled-label model did not reproduce the predictive ranking obtained with the true labels. Its test AUPRC (0.0093) was close to the positive-prevalence baseline (0.0113).

This provides evidence against the explanation that the observed Task B signal is simply an artifact of the pipeline that would also appear under randomized labels.

### What this control rules out

It reduces concern that:

- the preprocessing pipeline alone generates meaningful predictive performance;
- the model obtains useful performance without a real relationship between the features and the target.

### What this control does not rule out

It does **not** prove that the original Task B model is well calibrated or practically useful.

It also does not rule out:

- subject-specific distribution shifts,
- limitations of the chosen HR/HRV representation,
- an insufficient model capacity,
- limitations caused by only having minute-level labels,
- instability caused by the small number of positive prediction events.

---

## 4. Control 2 — Detection versus prediction

### Purpose

To separate the difficulty of recognizing apnea from the difficulty of predicting a new onset, the Task A Logistic Regression detector was applied to the target minute of the Task B windows.

This is deliberately an **oracle/current-event detection control**, not a valid prospective forecasting model, because it uses the HR/HRV features from the target minute itself.

The purpose was diagnostic:

> Can the same feature representation distinguish a current apnea state when the target minute is visible, even if it struggles to predict a new onset before that minute?

### Results

| Split | AUROC | AUPRC | Recall | F1 |
|---|---:|---:|---:|---:|
| Train | 0.8396 | 0.7128 | 0.9783 | 0.6487 |
| Validation | 0.6094 | 0.4832 | 0.9517 | 0.6091 |
| Test | 0.4979 | 0.4322 | 0.6726 | 0.4769 |

The original prospective Task B results were substantially weaker.

The final held-out Task B results were:

| Model | AUROC | AUPRC | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7583 | 0.0600 | 0.0345 | 0.0526 |
| TCN | 0.7136 | 0.0483 | 0.0000 | 0.0000 |

### Interpretation

The control demonstrates that **current-event detection and future-onset prediction are operationally different tasks**.

On validation data, current-event detection produced much stronger AUPRC than prospective prediction. This indicates that ECG/HRV information can contain information associated with the current apnea state that does not automatically translate into reliable advance warning of a new onset.

However, the held-out test detection AUROC was only 0.4979. Therefore, the control does **not** establish strong cross-record generalization of current-event detection.

The appropriate conclusion is narrower:

> The poor Task B result cannot be explained simply by the fact that ECG/HRV cannot represent apnea at all. The evidence instead supports a substantial additional difficulty in prospective onset prediction, although generalization of the current-event detector itself was limited on the held-out records.

### What this control rules out

It reduces the likelihood that Task B is failing simply because:

- the ECG/HRV features contain no information about apnea;
- the preprocessing makes apnea completely undetectable.

### What it does not rule out

It does not establish that the detection problem is robust across unseen subjects.

It also cannot prove that the remaining Task B difficulty is purely physiological rather than partly caused by:

- the coarse one-minute annotations,
- feature limitations,
- model limitations,
- subject heterogeneity.

---

## 5. Control 3 — Future-horizon sweep

### Purpose

A horizon sweep was used to determine whether the amount of future time included in the prediction target changes the difficulty of prospective onset prediction.

The history remained five minutes:

> [t-5, t)

The future onset target was defined using N->A transitions after the prediction time. A window was only considered when the state immediately before prediction was non-apnea, preventing an already ongoing apnea episode from becoming a false "future" event.

Horizon values tested:

- 1 minute
- 2 minutes
- 3 minutes
- 5 minutes

### Test results

| Horizon | Positive prevalence | AUROC | AUPRC | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 min | 0.0187 | 0.7339 | 0.0554 | 0.0000 | 0.0000 | 0.0000 |
| 2 min | 0.0369 | 0.6253 | 0.0732 | 0.1000 | 0.0351 | 0.0519 |
| 3 min | 0.0519 | 0.6238 | 0.1127 | 0.1311 | 0.2000 | 0.1584 |
| 5 min | 0.0783 | 0.6335 | 0.1390 | 0.1494 | 0.1083 | 0.1256 |

### Interpretation

The model produced AUROC values above 0.5 at every tested horizon, indicating some ability to rank future windows differently from non-event windows.

However, the task becomes less sparse as the horizon increases. The positive prevalence increased from 1.87% at one minute to 7.83% at five minutes. Therefore the AUPRC values must always be interpreted together with their corresponding prevalence baselines.

The thresholded one-minute prediction was particularly weak: despite AUROC = 0.7339, the validation-selected threshold produced zero true positives on the test set.

This demonstrates an important distinction:

> **Ranking ability does not necessarily translate into reliable event decisions at a practical threshold.**

The horizon sweep therefore provides evidence of some prospective predictive information, but not evidence that reliable short-horizon onset detection has been achieved.

### What this control rules out

It reduces the likelihood that the Task B definition itself is producing a single anomalous result at exactly one horizon.

It also shows that predictive ranking behavior persists across multiple future horizons.

### What this control does not rule out

The sweep does not establish that longer-horizon forecasting is practically better, because prevalence changes with horizon.

It also does not determine the optimal horizon. Only four horizons were tested, using a single feature-based baseline.

---

## 6. Control 4 — Three-seed repeatability

### Purpose

The TCN was retrained three times with different random seeds while keeping the following fixed:

- record-level train/validation split,
- architecture,
- features,
- preprocessing,
- class-weighting strategy,
- learning rate,
- weight decay,
- dropout,
- hidden width,
- early-stopping protocol,
- validation-based threshold selection.

Seeds:

- 42
- 123
- 2026

The held-out test set was not used during these repeatability runs.

### Task A

| Seed | AUPRC | AUROC | F1 |
|---:|---:|---:|---:|
| 42 | 0.5655 | 0.7096 | 0.6363 |
| 123 | 0.5289 | 0.6700 | 0.6287 |
| 2026 | 0.5013 | 0.6393 | 0.5993 |
| **Mean ± SD** | **0.5319 ± 0.0322** | **0.6730 ± 0.0352** | **0.6214 ± 0.0196** |

### Task B

| Seed | AUPRC | AUROC | F1 |
|---:|---:|---:|---:|
| 42 | 0.0937 | 0.5813 | 0.1739 |
| 123 | 0.0855 | 0.6040 | 0.1538 |
| 2026 | 0.0891 | 0.6150 | 0.1667 |
| **Mean ± SD** | **0.0894 ± 0.0041** | **0.6001 ± 0.0172** | **0.1648 ± 0.0102** |

### Interpretation

The TCN results vary somewhat across seeds, particularly for Task A. However, Task B performance is relatively stable:

- AUPRC = 0.0894 ± 0.0041
- AUROC = 0.6001 ± 0.0172

The weak Task B result therefore is not explained by a single unlucky random initialization.

### What this control rules out

It reduces the likelihood that the observed Task B behavior is caused entirely by stochastic training variability.

### What this control does not rule out

Three seeds are not enough to characterize the complete distribution of model variability.

They also do not eliminate:

- uncertainty caused by the small number of positive validation events,
- uncertainty caused by the limited number of test records,
- sensitivity to architecture or hyperparameters.

---

## 7. Error analysis

The final held-out Task B results contained very few correctly detected onset events.

For Logistic Regression:

- TP = 1
- FN = 28
- FP = 8
- TN = 2,522

For the TCN:

- TP = 0
- FN = 29
- FP = 1
- TN = 2,529

The Task B temporal error analysis showed:

### Logistic Regression

- all 28 false negatives corresponded to an actual 0->1 transition;
- the single true positive also corresponded to 0->1;
- therefore, the dominant failure mode was missed onset events.

### TCN

- all 29 positive test events were false negatives;
- the model detected none of the positive onset events at the frozen validation-selected threshold.

This suggests that the main failure is not confusion with already-ongoing apnea. The models are specifically failing to produce reliable advance warnings before the onset boundary.

Task A errors were also heterogeneous by subject. False-positive and false-negative rates differed substantially between held-out records, indicating substantial record-level variation rather than a uniform error process.

---

## 8. What result would have convinced me that short-horizon prediction is learnable?

I would consider short-horizon prediction convincingly demonstrated if all of the following were observed:

1. AUPRC consistently exceeded the appropriate positive-prevalence baseline on held-out records.
2. AUROC was materially above 0.5.
3. The improvement remained present across multiple random seeds.
4. The model detected a non-trivial fraction of actual onset events at a threshold selected only on validation data.
5. The result was reproducible across a reasonable range of forecasting horizons.
6. The improvement was not reproduced by shuffled labels.

The current experiments satisfy only some of these criteria.

The real-label models show evidence of ranking information:

- Task B Logistic test AUROC = 0.7583
- Task B TCN test AUROC = 0.7136
- Task B AUPRC values exceed the prevalence baseline

The horizon sweep also produced AUROC values above 0.5.

However, the thresholded event detection was weak, especially for the one-minute horizon, and the final TCN detected 0/29 positive test events.

Therefore, the evidence is **not sufficient to claim reliable short-horizon apnea-onset prediction**.

---

## 9. Final conclusion

The experiments support a nuanced conclusion rather than either "prediction works" or "prediction is impossible."

First, the shuffled-label control did not reproduce the real-label performance, reducing concern that the predictive signal is purely an artifact of the pipeline.

Second, the detection-vs-prediction control showed that current-event recognition and prospective onset prediction behave differently. ECG/HRV features can contain information associated with the current apnea state, but that information does not automatically provide reliable advance warning.

Third, the horizon sweep showed above-chance ranking behavior at multiple future horizons, suggesting that some predictive information may be present. However, increasing the horizon also increases the positive prevalence, so higher AUPRC at longer horizons cannot be interpreted directly as stronger forecasting ability.

Fourth, the three-seed experiment showed that the Task B TCN results were reasonably stable across random initializations (AUPRC 0.0894 ± 0.0041; AUROC 0.6001 ± 0.0172 on validation), indicating that the weak result was not simply a seed-42 failure.

Overall:

> **The experiments provide evidence for limited prospective predictive signal, but they do not demonstrate reliable short-horizon apnea-onset prediction. The principal difficulty appears to be forecasting a genuinely new event before it begins, rather than merely detecting an apnea state once it is present.**

This is also consistent with the final held-out comparison: the TCN did not provide a consistent improvement over the non-deep Logistic Regression baseline, and for Task B the TCN achieved lower AUPRC (0.0483 vs. 0.0600) and lower AUROC (0.7136 vs. 0.7583).

The most defensible conclusion is therefore that **the temporal deep model did not earn a clear advantage over the feature-based baseline under the frozen held-out evaluation protocol**.

---

## 10. Limitations

Several limitations should be kept in mind.

- The dataset contains only 30 recordings, with five held out for final testing.
- Task B contains only 29 positive test onset events.
- Labels are available at one-minute resolution, so exact physiological onset time within a minute cannot be determined.
- The current implementation uses ECG-derived HR/HRV features only; respiratory and SpO₂ channels were not present in the supplied recordings.
- The temporal TCN operates on five minute-level HR/HRV feature vectors rather than raw ECG waveforms.
- Only three random seeds were used for repeatability.
- The detection-vs-prediction control establishes a diagnostic comparison, not a valid prospective forecasting model.
- The horizon sweep evaluates a small, predefined set of horizons rather than exhaustively optimizing the forecast horizon.

These limitations make the results appropriate for a take-home experimental assessment, but insufficient to establish clinical or operational readiness.
