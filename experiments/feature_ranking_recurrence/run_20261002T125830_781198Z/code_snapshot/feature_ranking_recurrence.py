"""Read-only source analysis. See experiments/feature_ranking_recurrence/README.md."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
from datetime import datetime, timezone

# Avoid writing bytecode into the original project.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent

def write_json(path, value):
    def native(obj):
        if isinstance(obj, dict):
            return {str(k): native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [native(v) for v in obj]
        if hasattr(obj, 'item'):
            return native(obj.item())
        return obj
    path.write_text(json.dumps(native(value), indent=2, default=str, allow_nan=False) + '\n')


def panel_order(values, features, panel, sorting="stable"):
    """Signed panel ordering; stable ties keep feature order, quicksort is app comparison."""
    import pandas as pd
    s = pd.Series(values, index=features)
    if panel == 'low_score':
        s = s[s > 0]
    if sorting not in ['stable', 'app_quicksort']:
        raise ValueError(f'Unknown sorting procedure: {sorting}')
    return s.sort_values(ascending=panel == 'low_score', kind='stable' if sorting == 'stable' else 'quicksort')


def summarize(values, ids, features, low_features, djs, population, scenario, sorting="stable"):
    import numpy as np
    import pandas as pd
    records, summaries = [], []
    for panel in ['uncertainty', 'low_score']:
        full, shown, top = {f: [] for f in features}, {f: [] for f in features}, dict.fromkeys(features, 0)
        order_features = features if panel == 'uncertainty' else low_features
        idx = [features.index(f) for f in order_features]
        for patient, row in zip(ids, values):
            ordered = panel_order(row[idx], order_features, panel, sorting)
            for rank, (feature, score) in enumerate(ordered.items(), 1):
                full[feature].append(rank)
                if rank <= 20:
                    shown[feature].append(rank)
                if rank <= 5:
                    top[feature] += 1
                records.append(dict(population=population, scenario=scenario, sorting=sorting, patient_id=str(patient),
                                    panel=panel, feature=feature, score=float(score), full_rank=rank,
                                    displayed=rank <= 20, top_five=rank <= 5))
        jsr = pd.Series(djs, index=features).rank(method='average', ascending=True)
        for f, js in zip(features, djs):
            summaries.append(dict(population=population, scenario=scenario, sorting=sorting, panel=panel, feature=f,
                djs=float(js), djs_rank_ascending=float(jsr[f]), patients=len(ids), top_five_count=top[f],
                top_five_percent=100 * top[f] / len(ids), displayed_count=len(shown[f]),
                displayed_rank_denominator=len(shown[f]), mean_displayed_rank=np.mean(shown[f]) if shown[f] else np.nan,
                median_displayed_rank=np.median(shown[f]) if shown[f] else np.nan,
                full_eligible_rank_denominator=len(full[f]), mean_full_eligible_rank=np.mean(full[f]) if full[f] else np.nan))
    return pd.DataFrame(summaries), pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'Miyokardit_08.12.xlsx')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'experiments/feature_ranking_recurrence')
    parser.add_argument('--previous-run', type=Path, default=ROOT / 'experiments/feature_ranking_recurrence/run_20261002T122800_471911Z')
    parser.add_argument('--tau', type=float, help='Only use if a documented project threshold is supplied')
    parser.add_argument('--tau-source', help='Required citation for an explicit tau')
    args = parser.parse_args()
    if args.tau is not None and (args.tau <= 0 or not args.tau_source):
        parser.error('--tau must be positive and accompanied by --tau-source')
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(ROOT / 'experiments/feature_ranking_recurrence'):
        parser.error('Outputs must stay under experiments/feature_ranking_recurrence')
    out = output_root / datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'experiment.log').write_text('Started isolated experiment. Source artifacts are read-only.\n')
    try:
        run(args, out)
    except Exception as exc:
        write_json(out / 'failure.json', {'status': 'incomplete', 'error': str(exc), 'type': type(exc).__name__})
        with (out / 'experiment.log').open('a') as fh:
            fh.write(f'FAILED: {type(exc).__name__}: {exc}\n')
        raise
    print(f'Experiment complete: {out}')


def run(args, out):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, gaussian_kde
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    from uncertainty_transformer import UncertaintyTransformer
    from uncertainty_utils import discretise, get_distribution, js_divergence

    source_paths = [args.data.resolve()] + [ROOT / p for p in [
        'best_model_finetuned.pkl', 'model_metadata.pkl', 'split_indices.pkl', 'demo_patients.txt',
        'app_artifacts/tsne_scaler.pkl', 'app_artifacts/embedding_data.npz', 'app.py',
        'uncertainty_transformer.py', 'uncertainty_utils.py', 'generate_embedding.py',
        'uncertainty.ipynb', 'case_study.ipynb', 'finetuning.ipynb',
        'ACMCHI/Uncertainty_TOCHI_v4.pdf', 'ACMCHI/Revision_Report_v4.pdf']]
    source_paths += [p for p in (ROOT / 'experiments/feature_ranking_recurrence').glob('run_*/*') if p.is_file() and p.parent != out]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths if p.exists()}
    provenance = dict(input_sha256=hashes, seed=42, software={p: importlib.metadata.version(p) for p in
        ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'openpyxl']},
        limitations=['Exact Uncertainty_TOCHI_v4(2).pdf and Revizeler were not supplied/found; available v4 and Revision_Report_v4 inspected.',
                    'Historical clinician-session artifact identity cannot be independently established from current files.',
                    'NEW_Miyokardit_08.12.2025.xlsx and NEW_uncertainty.ipynb referenced by older code are missing.'],
        rank_rules='Signed descending uncertainty; strictly positive ascending low-score; first 20 displayed, first 5 recurrence. Stable analysis preserves original panel feature order for exact ties. App comparison uses unstable quicksort: it does NOT guarantee original order for ties.',
        python=sys.version, executable=sys.executable,
        correlations='Pairwise complete feature ranks; Spearman average ranks for ties; undefined constant inputs reported missing.',
        reference='Complete-case saved training split underlying the app landscape; full complete cohort also analyzed separately.',
        landscape='Saved app reference embedding; new patients positioned by mean of 5 Euclidean nearest neighbors. KDE Scott bandwidth, pad=2, 600x600 grid, class threshold quantile=0.6.',
        tolerance={'rtol': 1e-9, 'atol': 1e-10})
    write_json(out / 'provenance.json', provenance)
    def load(p):
        with (ROOT / p).open('rb') as fh:
            return copy.deepcopy(pickle.load(fh))
    model, metadata, split, scaler = [load(p) for p in ['best_model_finetuned.pkl', 'model_metadata.pkl', 'split_indices.pkl', 'app_artifacts/tsne_scaler.pkl']]
    transformer = model.named_steps['uncertainty']
    features = list(transformer.feature_names_in_)
    low_features = list(scaler.feature_names_in_)
    if len(features) != 44 or any('ECG' in f.upper() for f in features):
        raise ValueError('Saved configuration is not the documented 44-feature no-ECG configuration.')
    if list(metadata['features']) != features or set(low_features) != set(features):
        raise ValueError('Model, metadata, and landscape scaler feature configurations disagree.')
    with np.load(ROOT / 'app_artifacts/embedding_data.npz') as artifact:
        saved_std, emb, labels = [artifact[k].copy() for k in ['X_std', 'X_emb', 'y']]
    raw = pd.read_excel(args.data, sheet_name=0)
    clean = raw[features + ['GRUP']].copy().replace(
        [' ', '', '-', '--', 'nan', 'NaN', 'None', '#VALUE!', '#N/A', '#REF!', '#DIV/0!', '#NUM!', '#NAME?', '#NULL!'], np.nan)
    clean = clean.replace(r'^\s*$', np.nan, regex=True)
    for f in features:
        clean[f] = pd.to_numeric(clean[f], errors='coerce')
    cohort = clean.dropna().copy()
    ref = clean.loc[split['train_idx']].dropna().copy()
    if set(ref.GRUP.unique()) != set(transformer.classes_):
        raise ValueError('Reference labels disagree with transformer.')
    refitted = UncertaintyTransformer(feature_names=features, n_bins=transformer.n_bins, eps=transformer.eps,
                                      class_labels=transformer.classes_).fit(ref[features], ref.GRUP)
    for f in features:
        a, b = transformer._feat_stats_[f], refitted._feat_stats_[f]
        np.testing.assert_allclose([a.js] + list(a.mu.values()) + list(a.std.values()) + list(a.entropy.values()),
                                   [b.js] + list(b.mu.values()) + list(b.std.values()) + list(b.entropy.values()), rtol=1e-9, atol=1e-10,
                                   err_msg=f'Reference data does not reproduce saved statistics: {f}')
    demos_source = json.loads((ROOT / 'demo_patients.txt').read_text())
    demos = demos_source['patients'] if isinstance(demos_source, dict) else demos_source
    if len(demos) != 10:
        raise ValueError('Expected exactly ten demonstration patients.')
    demo_df = pd.DataFrame([p['data'] for p in demos])[features].copy()
    if demo_df.isna().any().any():
        raise ValueError('Demonstration feature data incomplete.')
    ids = list(range(1, 11))
    pd.DataFrame({'patient_id': ids, 'name': [p['name'] for p in demos]}).to_csv(out / 'demo_ids.csv', index=False)
    # Workbook row mappings are claims in the supplied demo note, not verified patient identities.
    provenance['demo_source_note'] = demos_source.get('note') if isinstance(demos_source, dict) else None
    claimed_rows = [2, 5, 7, 12, 9, 87, 86, 90, 83, 92]
    identity_rows = []
    for i, row_id in enumerate(claimed_rows):
        match = row_id in clean.index and np.allclose(
            clean.loc[row_id, features].to_numpy(float), demo_df.iloc[i].to_numpy(float),
            rtol=1e-5, atol=1e-6, equal_nan=False)
        identity_rows.append({'patient_id': i+1, 'claimed_workbook_row': row_id,
                              'all_selected_values_match_with_rounding_tolerance': bool(match)})
    pd.DataFrame(identity_rows).to_csv(out / 'demo_row_verification.csv', index=False)
    provenance['reference_row_ids'] = list(ref.index)
    provenance['cohort_row_ids'] = list(cohort.index)
    provenance['features'] = features
    provenance['projection_features'] = low_features
    provenance['eps'] = transformer.eps
    stats_rows = []
    for f in features:
        fs = transformer._feat_stats_[f]
        disc = discretise(ref[f], n_bins=transformer.n_bins)
        raw_js = js_divergence(get_distribution(disc[ref.GRUP == transformer.classes_[0]]),
                               get_distribution(disc[ref.GRUP == transformer.classes_[1]]), eps=transformer.eps)
        stats_rows.append({'feature': f, 'djs': fs.js, 'djs_unfloored': raw_js, 'mu': fs.mu, 'std': fs.std, 'entropy': fs.entropy})
    write_json(out / 'reference_statistics.json', stats_rows)
    djs = np.array([transformer._feat_stats_[f].js for f in features])
    positive = djs[djs > 0]
    thresholds = {f'p{p:02}': float(np.percentile(positive, p, method='linear')) for p in [1, 5, 10]}
    unfloored = np.array([row['djs_unfloored'] for row in stats_rows])
    exclusions = []
    eligible = []
    for j, f in enumerate(features):
        reasons = []
        if ref[f].nunique() <= 1:
            reasons.append('constant across reference patients')
        if not np.isfinite(unfloored[j]):
            reasons.append('nonfinite unfloored DJS')
        elif unfloored[j] <= 0:
            reasons.append('unfloored DJS is not strictly positive')
        if reasons:
            exclusions.append({'feature': f, 'reasons': reasons, 'unfloored_djs': float(unfloored[j])})
        else:
            eligible.append(j)
    if not eligible:
        raise ValueError('No nonconstant features with positive unfloored DJS for supplementary grid.')
    supplementary_grid = (1, 5, 10, 20, 25)  # predefined; never selected using patient outcomes
    supplementary = {f'supp_p{p:02}': float(np.percentile(unfloored[eligible], p, method='linear'))
                     for p in supplementary_grid}
    if args.tau is not None:
        thresholds['documented'] = args.tau
    primary = 'documented' if args.tau is not None else 'p05'
    # Commit threshold definitions before calculating ANY capped patient results.
    write_json(out / 'thresholds.json', dict(primary=primary, thresholds=thresholds, units='bits (log2)',
        method='numpy.percentile(method=linear) over strictly positive stored reference DJS, including original EPS floor',
        positive_features=len(positive), original_floor=transformer.eps, zero_features=int((djs == 0).sum()),
        exploratory=args.tau is None, source=args.tau_source,
        affected={k: int((djs < t).sum()) for k, t in thresholds.items()}))
    thresholds.update(supplementary)
    definitions = []
    factor_rows = []
    for name, tau in thresholds.items():
        factor = (djs + transformer.eps) / (np.maximum(djs, tau) + transformer.eps)
        affected = [f for f, js in zip(features, djs) if js < tau]
        definitions.append({'scenario': name, 'family': 'supplementary_unfloored_nonconstant' if name.startswith('supp_') else 'original_exploratory',
            'tau_bits': tau, 'affected_features': affected, 'affected_count': len(affected),
            'effective_denominator_perturbation': bool(np.any(factor != 1)), 'exploratory': name != 'documented'})
        for f, js, raw_js, multiplier in zip(features, djs, unfloored, factor):
            factor_rows.append({'scenario': name, 'feature': f, 'djs': js, 'unfloored_djs': raw_js,
                'tau_bits': tau, 'affected': js < tau, 'multiplicative_score_change_factor': multiplier})
    write_json(out / 'threshold_definitions.json', {'definitions': definitions, 'supplementary_percentile_grid': supplementary_grid,
        'supplementary_included_features': [features[j] for j in eligible], 'supplementary_exclusions': exclusions,
        'supplementary_method': 'linear percentiles of positive unfloored DJS from nonconstant reference features; cap still uses original stored DJS and epsilon',
        'units': 'bits, log2', 'original_epsilon': transformer.eps,
        'timing': 'Saved before calculating capped patient results. All percentile thresholds exploratory.'})
    pd.DataFrame(factor_rows).to_csv(out / 'threshold_feature_factors.csv', index=False)
    arrays = { 'demonstration': (demo_df, ids), 'reference': (ref[features], list(ref.index)),
               'complete_cohort': (cohort[features], list(cohort.index))}
    originals = {p: transformer.transform(frame.copy()) for p, (frame, _) in arrays.items()}
    # Independent numerator reconstruction checks signed outputs against the imported transformer.
    for population, (frame, _) in arrays.items():
        rebuilt = np.zeros_like(originals[population])
        for j, f in enumerate(features):
            fs = transformer._feat_stats_[f]; c1, c2 = transformer.classes_
            z1 = (frame[f].to_numpy(float) - fs.mu[c1]) / fs.std[c1]
            z2 = (frame[f].to_numpy(float) - fs.mu[c2]) / fs.std[c2]
            z1[~np.isfinite(z1)] = 0; z2[~np.isfinite(z2)] = 0
            choose = abs(z1) <= abs(z2)
            rebuilt[:, j] = np.where(choose, z1, z2) * (np.where(choose, fs.entropy[c1], fs.entropy[c2]) / (fs.js + transformer.eps))
        np.testing.assert_allclose(rebuilt, originals[population], rtol=1e-12, atol=1e-12)
    projection_idx = [features.index(f) for f in low_features]
    ref_original = pd.DataFrame(originals['reference'][:, projection_idx], columns=low_features)
    np.testing.assert_allclose(scaler.transform(ref_original), saved_std, rtol=1e-9, atol=1e-10,
                               err_msg='Recovered reference/scaler do not reproduce saved projection inputs.')
    np.testing.assert_array_equal(ref.GRUP.to_numpy(), labels)
    write_json(out / 'verification.json', {'manual_transform': 'passed for all populations',
        'reference_statistics': 'passed', 'saved_projection_inputs_and_labels': 'passed'})
    from experiments.feature_ranking_recurrence.analysis import analyze
    analyze(out, args.previous_run, arrays, originals, features, low_features, djs, transformer.eps,
            thresholds, primary, scaler, saved_std, emb, labels, provenance, summarize)
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Source changed during experiment: {path}')
    write_json(out / 'completion.json', {'status': 'complete', 'source_and_previous_run_hashes_unchanged': True})
    with (out / 'experiment.log').open('a') as fh:
        fh.write('Reproduction, independent calculation checks, and source/previous-run hash checks passed. Complete.\n')

if __name__ == '__main__':
    main()
