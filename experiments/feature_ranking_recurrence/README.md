# Corrected feature-ranking recurrence experiment

This experiment reads the existing 44-feature no-ECG pipeline without changing the app, notebooks, transformer, source data, fitted models, scalers, or embeddings. Every execution creates a new `run_<UTC timestamp>` directory. Previous runs remain intact; their saved quicksort outputs are comparisons, not overwritten corrections.

## Execute

The surviving original experiment environment is `/tmp/feature-ranking-env`. Its NumPy, pandas, SciPy, scikit-learn, matplotlib, and openpyxl versions match all six versions recorded in `run_20261002T122800_471911Z/provenance.json`. That run did not record its Python version; the corrected run records both Python and the executable. None of these checks establishes historical clinician-session software.

From the project root:

```sh
MPLCONFIGDIR=/tmp/feature-ranking-mpl PYTHONDONTWRITEBYTECODE=1 \
  /tmp/feature-ranking-env/bin/python feature_ranking_recurrence.py
```

For a new environment, use Python 3.11 or later and install `requirements.txt` from this experiment directory. Exact previously recorded package versions are in `requirements-replay.txt`; compatibility/wheel availability depends on Python version. A quicksort replay in a different environment is explicitly a current-environment comparison and may differ. Stable ordering is the deterministic analysis.

```sh
python3 -m venv /tmp/feature-ranking-new-env
/tmp/feature-ranking-new-env/bin/python -m pip install -r experiments/feature_ranking_recurrence/requirements.txt
MPLCONFIGDIR=/tmp/feature-ranking-mpl PYTHONDONTWRITEBYTECODE=1 \
  /tmp/feature-ranking-new-env/bin/python feature_ranking_recurrence.py
```

Options: `--data /absolute/path/workbook.xlsx`, `--previous-run /absolute/path/previous/run`, and `--tau VALUE --tau-source 'document and section'` for an existing documented cutoff. Output paths must remain inside this experiment directory. Alternate inputs must reproduce the saved reference statistics and embedding inputs; mismatches fail explicitly.

## Required original inputs

- `Miyokardit_08.12.xlsx`, first worksheet, with original zero-based row indices and `GRUP` labels.
- `best_model_finetuned.pkl`, `model_metadata.pkl`, `split_indices.pkl`: trusted local artifacts containing the feature configuration and fitted reference statistics.
- `demo_patients.txt`: ten fully specified demonstration cases.
- `app_artifacts/embedding_data.npz`, `app_artifacts/tsne_scaler.pkl`: unchanged app landscape, standardized reference inputs, labels, and feature ordering.
- `uncertainty_transformer.py` and `uncertainty_utils.py`: imported, never modified.
- The selected previous run's `provenance.json`, `thresholds.json`, `patient_rankings.csv`, `feature_results.csv`, and `spearman.csv` for replay verification.

The available manuscript/revision report were inspected previously. The exact requested `Uncertainty_TOCHI_v4(2).pdf`, **Revizeler** annotations, and the older `NEW_Miyokardit_08.12.2025.xlsx` / `NEW_uncertainty.ipynb` references are unavailable. Current-data/artifact reproduction is numerically checked, but historical clinician-session figure identity cannot be verified.

## Ranking and population definitions

The ten demonstration cases, 128 reference patients underlying the saved landscape, and all 160 complete-case patients are separate descriptive populations. The complete cohort includes all reference patients and all demo cases. `population_overlap.json` gives exact overlap counts and mapped demo row IDs. They are not independent samples; rounded demonstration measurements need not be bit-identical to source workbook values. The complete cohort does not refit statistics or scalers.

The uncertainty panel sorts signed scores descending. The low-score panel excludes scores ≤0 and sorts ascending. Up to twenty features are displayed; recurrence uses the first five. Absence produces missing conditional ranks, never rank zero. Displayed and full eligible rank means both report appearance/eligibility denominators. In the low-score panel, full eligibility still requires a positive score.

`stable` uses stable sorting with exact score ties resolved by the panel's original ordered feature list (transformer order for uncertainty, saved scaler order for low-score). It uses the same rule before and after capping. `app_quicksort` reproduces the app's default unstable pandas quicksort. **The previous claim that quicksort preserves feature order for ties was incorrect.** Its ordering may depend on NumPy/pandas versions. Neither rule changes the app. Spearman uses average ranks for tied feature-level DJS/rank statistics, pairwise omission of missing means, and explicit feature counts; reported p-values are descriptive, without multiple-testing adjustment.

## Predefined exploratory thresholds

The original p01/p05/p10 caps remain an `original_exploratory` family, computed from strictly positive **stored** DJS including the existing epsilon floor. Their values are verified against the prior run.

The supplementary grid is fixed at **1, 5, 10, 20, 25 percentiles**, using positive **unfloored** DJS from nonconstant reference features, with linear interpolation. Unfloored means before `max(DJS, epsilon)`; the original probability-logarithm epsilon is still retained. Exclusions and reasons are recorded. Features excluded from threshold estimation remain eligible for capping and ranking.

Both families cap only the original stored denominator:

```text
x_capped = H*z / (max(DJS, tau) + epsilon)
factor = (DJS + epsilon) / (max(DJS, tau) + epsilon)
```

The numerator, signs, epsilon, statistics, and feature order are preserved. `threshold_definitions.json` and `threshold_feature_factors.csv` are saved before capped patient calculations. All percentile thresholds are exploratory, not clinical cutoffs. Thresholds are not selected by patient movement.

“No denominator perturbation,” “no raw change within tolerance,” “no top-five membership change,” and “no ordering change” are distinct outcomes. `threshold_summary.csv`, `raw_score_perturbations.csv`, and `threshold_ranking_comparison.csv` report them separately. In particular, unchanged membership does not mean unchanged ordering or unchanged scores.

## Landscape interpretation

Refitting StandardScaler on corresponding positively rescaled reference columns cancels their multipliers. Every threshold is checked independently with explicit population mean/standard-deviation calculations for reference and demo inputs, in addition to sklearn transforms (`rtol=1e-9`, `atol=1e-10`). If invariant, the verified saved embedding and patient coordinates are reused; no stochastic embedding differences are interpreted as cap effects. Unexpected non-invariance fails explicitly instead of making a landscape claim.

The separate fixed-original-scaler diagnostic retains the old reference embedding and its reference inputs while transforming capped demo values with the original scaler. It changes the scaling procedure. Five Euclidean neighbors determine mean patient coordinates; neighbor distances are independently checked by brute force. Displacements are normalized by the reference embedding RMS radius. Region labels use the app's Scott KDE, a 600×600 grid, padding 2, and 60th-percentile density contours. Regions are landscape density classifications, not classifier diagnoses.

## Review outputs

- `table9.csv`, `table9.pdf`, `table9_*.png`: all features/populations/panels. CSV contains both sorting procedures; PDF/PNG use deterministic stable ties and include both rank denominators.
- `figure2.png` / `figure2.pdf`: deterministic reference rank plot and original primary-cap landscape. `figure2_<scenario>.*` includes every supplementary cap.
- `tie_handling_comparison.*`, `tie_feature_comparison.csv`, `tie_patient_comparison.csv`, `tie_correlation_comparison.csv`: changes to recurrence, denominators, rank statistics, memberships/order, and both correlations.
- `threshold_definitions.json`, `threshold_feature_factors.csv`, `threshold_summary.csv`: numerical thresholds, excluded/affected features, and every feature's factor.
- `patient_rankings.csv`, `feature_results.csv`, `spearman.csv`: all original/capped scores and results for both sorts and all populations.
- `ranking_changes.csv`, `threshold_ranking_comparison.csv`, `raw_score_perturbations.csv`: patient-level and aggregate raw, membership and ordering effects.
- `scaling_checks.csv`, `patients_6_8_landscape.csv`, `intermediates_*.npz`, `scaler_*.pkl`: independent input checks, coordinates/displacements/regions, and experiment-specific arrays and scalers.
- `previous_run_replay.json`, `previous_run_feature_replay.csv`, `previous_run_correlation_replay.csv`, `software_comparison.json`: prior-run replay and environment evidence.
- `original_scores.npz`, `reference_statistics.json`, `demo_ids.csv`, `demo_row_verification.csv`, `population_overlap.json`: reproduction and patient identities.
- `methods_results.txt`: concise Section 4.3 draft based on computed results.
- `provenance.json`, `verification.json`, `independent_verification.json`, `experiment.log`, `completion.json` or `failure.json`: inputs, hashes, software, settings, checks, and exact limitations. Source and previous-run hashes are rechecked at completion.

## Tests and independent audit

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/feature-ranking-env/bin/python -m unittest discover \
  -s experiments/feature_ranking_recurrence -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 /tmp/feature-ranking-env/bin/python \
  experiments/feature_ranking_recurrence/verify_results.py experiments/feature_ranking_recurrence/run_TIMESTAMP
```

The independent audit does not import the ranking/summarization helpers: it uses NumPy lexsort, direct appearance statistics, Pearson correlation of average rank vectors, independent percentile interpolation, and explicit denominator factors. It also verifies saved membership and ordering changes. Tests cover exact ties crossing ranks five and twenty, absent ranks, signed/positive panel eligibility, membership versus ordering, and standardization cancellation.
