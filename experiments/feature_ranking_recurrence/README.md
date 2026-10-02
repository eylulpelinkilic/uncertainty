# Feature-ranking recurrence experiment

Run from any working directory using Python 3.11 or later. This experiment reads the original project artifacts and imports `UncertaintyTransformer` without changing its source. Each invocation creates a unique `run_<UTC timestamp>` directory here; no results are overwritten. Models are deep-copied after loading, capping acts on new arrays, and each capped scaler is newly fitted. Original input SHA-256 hashes are checked again at completion.

## Setup and execution

Use a separate virtual environment, preferably with Python 3.11–3.13 to match available scikit-learn wheels:

```sh
python3 -m venv /tmp/feature-ranking-env
/tmp/feature-ranking-env/bin/python -m pip install -r experiments/feature_ranking_recurrence/requirements.txt
MPLCONFIGDIR=/tmp/feature-ranking-mpl PYTHONDONTWRITEBYTECODE=1 /tmp/feature-ranking-env/bin/python feature_ranking_recurrence.py
```

An alternate workbook can be supplied with `--data /absolute/path/to/workbook.xlsx`. It must reproduce the saved reference statistics and landscape inputs; the program fails explicitly if it cannot. An existing documented cutoff can be supplied with `--tau VALUE --tau-source 'document and section'`. Without it, the exploratory linear 1st/5th/10th percentile rule is used and saved before patient sensitivity calculations. This is not a clinical cutoff.

## Required existing inputs

- `Miyokardit_08.12.xlsx`, first worksheet; original zero-based row indices are retained.
- `best_model_finetuned.pkl`: trusted local fitted pipeline, including reference statistics.
- `model_metadata.pkl`, `split_indices.pkl`: 44-feature no-ECG configuration and saved training row IDs.
- `demo_patients.txt`: exactly ten fully specified patients; display patient numbers 1–10 are the IDs used in ranking outputs.
- `app_artifacts/embedding_data.npz` and `app_artifacts/tsne_scaler.pkl`: app reference embedding, standardized inputs, labels, and feature order.
- `uncertainty_transformer.py`, `uncertainty_utils.py`: imported original implementations.

The saved landscape training cohort is called `reference`. The separate `complete_cohort` population includes all complete cases, including held-out patients, but uses the unchanged saved reference statistics. It is not used to fit reference statistics or scalers.

Available `ACMCHI/Uncertainty_TOCHI_v4.pdf` §3.4.5/§4.3 and `ACMCHI/Revision_Report_v4.pdf` were inspected. The exact requested `Uncertainty_TOCHI_v4(2).pdf` and **Revizeler** annotations were not available in this workspace. The older workbook `NEW_Miyokardit_08.12.2025.xlsx` and notebook `NEW_uncertainty.ipynb` referenced by existing files are also missing. Numerical verification against current saved artifacts does not establish that they are identical to artifacts used during historical clinician sessions. Do not remove that limitation from publication text without establishing session provenance.

## Outputs and interpretation

- `thresholds.json`: committed cutoff definitions, units, epsilon floor, affected counts.
- `reference_statistics.json`, `demo_ids.csv`, `demo_row_verification.csv`: original class means, standard deviations, entropies, divergence, and patient identity evidence.
- `verification.json`: independent signed formula reproduction, reconstructed fitted statistics, saved standardized inputs and labels.
- `table9.csv`, paginated vector `table9.pdf`, complete per-page `table9_*.png`: all features, populations and panels. `feature_results.csv` adds all capped scenarios.
- `patient_rankings.csv`: signed scores for every eligible feature; flags identify displayed/top-five membership. Absent/ineligible features have no patient ranking row, and feature summaries report zero appearances and missing conditional ranks.
- `spearman.csv`: both conditional displayed and full eligible rank correlations, including feature inclusion counts. Spearman uses average ranks for ties; missing ranks are excluded pairwise. Full eligible ranks are conditional on strictly positive eligibility in the low-score panel and have their own denominators.
- `top_five_changes.csv`: membership retention, additions/removals and Jaccard similarity for every patient/panel/cutoff.
- `scaling_checks.csv`, `patients_6_8_landscape.csv`: actual projection-input changes, normalized displacement, and KDE region assignments.
- `figure2.png` / `figure2.pdf`: original reference-cohort DJS versus conditional displayed rank, with the three most recurrent features per panel labelled; primary cap landscape comparison. `figure2_p01.*` / `figure2_p10.*` provide supplemental comparisons. Missing displayed ranks cannot be plotted and remain explicit in Table 9.
- `intermediates_*.npz`, `scaler_*.pkl`: experiment-specific copies/results. Pickles are local trusted artifacts.
- `methods_results.txt`: computed Methods and Results draft; descriptive results require interpretation before publication.
- `provenance.json`, `experiment.log`, `completion.json` (or `failure.json`): inputs, hashes, row IDs, seeds, software, configuration, checks and limitations.

The transformer stores DJS with an epsilon floor; `djs_unfloored` exposes recomputed values before that original floor. Threshold percentiles use the stored values so the original denominator is preserved. Zero-DJS values, if present, are affected by every positive cutoff.

Uncertainty scores are ordered descending in transformer feature order. The low-score panel filters scores ≤0 and sorts ascending in the saved projection-scaler feature order. Both use pandas' original default quicksort, including its handling of score ties; no absolute values or classifier feature importance enter the ranking.

For the app landscape, demonstration patients use the mean coordinates of five Euclidean nearest neighbors. Refitting a per-column scaler can cancel the positive capping multiplier. If both reference and demo inputs are invariant within `rtol=1e-9, atol=1e-10`, the experiment reuses original positions and reports zero displacement. It does not interpret stochastic refits as capping effects. The fixed-original-scaler diagnostic intentionally changes the scaling procedure and leaves the original reference embedding fixed.

If refitted inputs change, original and capped t-SNE are fitted with matching settings (`perplexity=50`, PCA initialization, automatic learning rate, seed 42), and aligned by translation, rotation/reflection, and least-squares scale over common reference rows. Displacement is normalized by baseline reference RMS radius. Region assignments for separately fitted embeddings are explicitly unavailable; original app regions use Scott KDE, a 600×600 grid, padding 2, and the 60th percentile density contour. Landscape regions are not classifier predictions.

## Verification tests

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/feature-ranking-env/bin/python -m unittest discover -s experiments/feature_ranking_recurrence -p 'test_*.py'
```
