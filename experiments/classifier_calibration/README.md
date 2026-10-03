# Isolated classifier-calibration experiment

This analysis evaluates original classifier probabilities. It does not fit a new calibrator, change the deployed pipeline, edit source data/notebooks, or modify the feature-ranking experiment. Every invocation writes a new timestamped directory under this folder; all existing project files outside this experiment are hashed before and after execution.

## Execute

From any working directory, use an absolute script path, or run from the repository root:

```sh
MPLCONFIGDIR=/tmp/classifier-calibration-mpl PYTHONDONTWRITEBYTECODE=1 \
  /tmp/feature-ranking-env/bin/python experiments/classifier_calibration/calibration.py
```

The existing isolated `/tmp/feature-ranking-env` contains the dependencies. To create a separate environment:

```sh
python3 -m venv /tmp/classifier-calibration-env
/tmp/classifier-calibration-env/bin/python -m pip install -r experiments/classifier_calibration/requirements.txt
MPLCONFIGDIR=/tmp/classifier-calibration-mpl PYTHONDONTWRITEBYTECODE=1 \
  /tmp/classifier-calibration-env/bin/python experiments/classifier_calibration/calibration.py
```

Python 3.11 or later is required. Exact runtime software versions and original model random seeds are recorded in each run's provenance. The saved pipeline's hyperparameters, random states, soft-voting weights, preprocessing steps, and ordered features are preserved by cloning; no new tuning protocol is introduced. Folds are executed sequentially in source-workbook row order; the saved Random Forest retains `n_jobs=-1`. Row ordering and software can affect numerical reproduction, so neither is assumed equivalent to historical sessions.

## Required inputs and configuration evidence

The available configuration requires:

- `Miyokardit_08.12.xlsx`, first worksheet, `GRUP` labels 1=myocarditis and 2=ACS.
- Trusted local `best_model_finetuned.pkl` and `model_metadata.pkl`.
- Original `uncertainty_transformer.py` / `uncertainty_utils.py`, imported read-only.
- `app.py`, `finetuning.ipynb`, and `uncertainty.ipynb`, read for configuration/evaluation provenance.

The app loads the saved 44-feature no-ECG pipeline: UncertaintyTransformer → StandardScaler → equal-weight soft voting over Logistic Regression, SVC, and Random Forest. Metadata and fitted pipeline feature lists/order are checked. The workbook determines the actual eligible complete-case denominator; counts are not forced to manuscript values. Missing tokens and whitespace are cleaned, selected features are coerced to numbers, and rows with missing required values/labels are excluded without imputation, as in the original notebook.

The notebook's LOOCV code computes transient probabilities but does not export patient-level probabilities with identities and verifiable fold exclusion. `prediction_search.json` records candidate artifacts and the decision to generate fresh predictions.

The 54-feature configuration is now available from [EKG-uncertainty at commit 25a4992135c9b193e4c360874ddf8e362d5f0a87](https://github.com/eylulpelinkilic/EKG-uncertainty/tree/25a4992135c9b193e4c360874ddf8e362d5f0a87). Its exact ordered list is the original 44 features followed by three echocardiographic and seven ECG variables. The echo variables are `EF`, `Segmentary Wall Motion Abnormality`, and `Pericardial Effusion`. The ECG variables are `ECG_ST depression`, `ECG_Location of ST depression ` (trailing space retained), `Level of ST-Dep_mm`, `ECG_T neg`, `ECG_Location of T negativity`, `Level of T invertion_mm`, and `ECG_Q waves`.

The pinned pipeline uses equal soft voting over LR + SVC + KNN. The source verifier checks the exact checkout commit and clean worktree; metadata, notebook, fitted transformer and constructor feature order; transformer/utility byte compatibility; diagnostic class mapping; original transformer settings; and saved training statistics/scaler mean against the original workbook using the pinned split. The actual complete-case count is 158 (61 myocarditis, 97 ACS), with 126 cases in the saved training subset. These counts are computed, not imposed. The 44-feature cohort has 160 cases (63 myocarditis, 97 ACS).

To reproduce both configurations with fresh fold fits in a new output directory:

```sh
git clone https://github.com/eylulpelinkilic/EKG-uncertainty.git /tmp/calibration-EKG-source-25a4992
git -C /tmp/calibration-EKG-source-25a4992 checkout --detach 25a4992135c9b193e4c360874ddf8e362d5f0a87
MPLCONFIGDIR=/tmp/classifier-calibration-mpl PYTHONDONTWRITEBYTECODE=1 \
  /tmp/classifier-calibration-env/bin/python experiments/classifier_calibration/calibration.py \
  --with-ecg-source-repo /tmp/calibration-EKG-source-25a4992
```

Use a new clone destination if it already exists. The execution environment can be installed from this experiment's requirements; the recorded six package versions can be replayed with `experiments/feature_ranking_recurrence/requirements-replay.txt`. The original 44-feature run remains unchanged. The new run contains both prediction sheets, source commit/fold evidence, independent checks, and unmodified copies of the pinned model, metadata, split, transformer/utilities and selection notebook in `source_artifacts/`. All tracked EKG source files and all prior calibration runs are hash-checked at completion. The former missing-pipeline status remains accurate for its original historical run and is not edited retroactively.

The standalone `--with-ecg-model` / `--with-ecg-data` options remain available, but a standalone pipeline without repository metadata is a weaker provenance path; the pinned-repository option is used for the new analysis.

`--data` can specify an alternate no-ECG workbook. The two configurations are separate modeling pipelines; they are not a controlled ECG-only ablation. Current interface loading is verifiable from present files. Historical clinician-session artifact/configuration identity remains unverified.

## Evaluation scope and patient identity

Each eligible patient is excluded once. A fresh clone refits the complete pipeline on the remaining patients, including class-reference distributions/statistics and StandardScaler. The held-out patient never enters fold training; probabilities are mapped using the fitted `classes_`. VotingClassifier internally encodes member labels; member classes are inverse-mapped using its fitted label encoder and saved in the audit.

The original project identifies patients by zero-based workbook row index, which is retained as `patient_id` and `source_row`; `source_excel_row` adds two for the header and Excel's one-based numbering. Each prediction also records the fitted probability-column class ordering. No undocumented clinical identifier is invented.

Original family selection and 150-trial retuning used the original training subset, which overlaps this complete cohort. These predictions are **LOOCV of a fixed, previously selected model specification**, not fully nested or completely unbiased evaluation of model selection. The previously reported aggregate notebook metrics are not substituted for this fresh evaluation.

## Workbook and outputs

`patient_predictions.xlsx` contains:

- `predictions_noECG` and, where available, `predictions_withECG`: original row ID, configuration, fold, training size, true diagnosis and binary myocarditis outcome, both class probabilities, predicted diagnosis, confidence, correctness, and Brier contribution.
- `calibration_bins`: bin boundaries, counts, mean probability, and observed frequency for both diagrams and both resolutions.
- `summary`: patient denominators, binary Brier scores, accuracy, and descriptive calibration statistics; missing configurations explicitly marked unavailable.
- `provenance`: software, feature lists, model specification/hyperparameters/weights, seeds, probability mappings, input hashes, preprocessing, and limitations.

Patient predictions are saved to Excel **before** aggregate calculations. Summary and reliability bins are calculated by reloading that saved workbook, then appended to its aggregate sheets. Matching CSVs allow independent recalculation. Only this newly created experiment workbook is edited during reporting.

The binary Brier contribution is `(p_myocarditis − y_myocarditis)²`, and Brier is its patient mean. The two class errors are not summed. Brier measures overall probabilistic prediction quality, not calibration alone.

Primary reliability plots use pooled OOF myocarditis probabilities with five equal-width bins over [0,1]; ten bins provide sensitivity. Intervals are `[lower,upper)`, except the last includes 1. Empty bins retain count zero and missing means/rates. Counts and probability histograms expose sparse bins. Supplementary confidence-versus-correctness plots use the predicted class's confidence and binary prediction correctness; these are distinct from class-probability calibration and Brier scoring. Weighted absolute gaps (descriptive ECE) are explicitly bin-dependent descriptive summaries, not a test or certification of clinical calibration.

Figures are saved as `reliability_<configuration>_<diagram>_<5|10>bins.png` and `.pdf`. Additional outputs include `fold_audit.jsonl`, eligible/excluded patient files, prediction search and provenance, predefined bin definitions, executable code snapshots and hashes, independent verification, protected-file hashes, logs, and a concise `methods_results.txt` manuscript draft. Small-sample estimates are descriptive; no new calibrated model is produced.

## Checks

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/feature-ranking-env/bin/python -m unittest discover \
  -s experiments/classifier_calibration -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 /tmp/feature-ranking-env/bin/python \
  experiments/classifier_calibration/verify_results.py experiments/classifier_calibration/run_TIMESTAMP
```

Tests cover reversed probability-column mappings, the single binary squared error, exact bin boundaries and probability 1, empty-bin missing rates, invalid probability simplexes, and held-out exclusion/duplicate identities. The separate verification script reads exported Excel/CSV data and independently recomputes Brier scores and bin statistics with direct interval masks, checks source labels, each patient's unique OOF prediction, complete training identity sets, and per-fold class mappings. The program also checks freshly cloned preprocessing and the scaler's fitted sample count in every fold.

For the pinned extension, independently verify archived metadata/order, complete-case reconstruction, every fold's source commit and exclusion, the preserved 44-feature replay, and all protected hashes:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/classifier-calibration-env/bin/python \
  experiments/classifier_calibration/verify_extension.py \
  experiments/classifier_calibration/run_20261003T162010_515531Z
```

The resulting `extension_verification.json` also summarizes captured fit warnings. All 158 with-ECG folds recorded `Unknown solver options: iprint`, a legacy verbosity option passed by sklearn to the installed SciPy optimizer. The saved solver and hyperparameters were retained; no convergence warning was recorded. Historical probabilities may differ with software versions, so the actual versions remain part of the experiment provenance.
