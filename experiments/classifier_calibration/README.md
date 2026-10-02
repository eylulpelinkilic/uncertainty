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

No verified 54-feature with-ECG LR+SVC+KNN pipeline, its metadata, or original selection code was found. That analysis is explicitly unavailable; its 158-patient cohort cannot be verified from a manuscript description alone. The workbook has a `predictions_withECG` status sheet, without invented patient rows or metrics. If a trusted complete fitted pipeline and its appropriate workbook become available, run:

```sh
python experiments/classifier_calibration/calibration.py \
  --with-ecg-model /absolute/path/with_ecg_pipeline.pkl \
  --with-ecg-data /absolute/path/source_workbook.xlsx
```

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
