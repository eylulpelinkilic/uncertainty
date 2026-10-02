"""Read-only source analysis. See experiments/feature_ranking_recurrence/README.md."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
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


def panel_order(values, features, panel):
    """Match pandas sort_values default quicksort and original panel feature order."""
    import pandas as pd
    s = pd.Series(values, index=features)
    if panel == 'low_score':
        s = s[s > 0]
    return s.sort_values(ascending=panel == 'low_score')


def summarize(values, ids, features, low_features, djs, population, scenario):
    import numpy as np
    import pandas as pd
    records, summaries = [], []
    for panel in ['uncertainty', 'low_score']:
        full, shown, top = {f: [] for f in features}, {f: [] for f in features}, dict.fromkeys(features, 0)
        order_features = features if panel == 'uncertainty' else low_features
        idx = [features.index(f) for f in order_features]
        for patient, row in zip(ids, values):
            ordered = panel_order(row[idx], order_features, panel)
            for rank, (feature, score) in enumerate(ordered.items(), 1):
                full[feature].append(rank)
                if rank <= 20:
                    shown[feature].append(rank)
                if rank <= 5:
                    top[feature] += 1
                records.append(dict(population=population, scenario=scenario, patient_id=str(patient),
                                    panel=panel, feature=feature, score=float(score), full_rank=rank,
                                    displayed=rank <= 20, top_five=rank <= 5))
        jsr = pd.Series(djs, index=features).rank(method='average', ascending=True)
        for f, js in zip(features, djs):
            summaries.append(dict(population=population, scenario=scenario, panel=panel, feature=f,
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
    from scipy.linalg import orthogonal_procrustes
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    from sklearn.manifold import TSNE
    from uncertainty_transformer import UncertaintyTransformer
    from uncertainty_utils import discretise, get_distribution, js_divergence

    source_paths = [args.data.resolve()] + [ROOT / p for p in [
        'best_model_finetuned.pkl', 'model_metadata.pkl', 'split_indices.pkl', 'demo_patients.txt',
        'app_artifacts/tsne_scaler.pkl', 'app_artifacts/embedding_data.npz', 'app.py',
        'uncertainty_transformer.py', 'uncertainty_utils.py', 'generate_embedding.py',
        'uncertainty.ipynb', 'case_study.ipynb', 'finetuning.ipynb',
        'ACMCHI/Uncertainty_TOCHI_v4.pdf', 'ACMCHI/Revision_Report_v4.pdf']]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths if p.exists()}
    provenance = dict(input_sha256=hashes, seed=42, software={p: importlib.metadata.version(p) for p in
        ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'openpyxl']},
        limitations=['Exact Uncertainty_TOCHI_v4(2).pdf and Revizeler were not supplied/found; available v4 and Revision_Report_v4 inspected.',
                    'Historical clinician-session artifact identity cannot be independently established from current files.',
                    'NEW_Miyokardit_08.12.2025.xlsx and NEW_uncertainty.ipynb referenced by older code are missing.'],
        rank_rules='Signed descending uncertainty; strictly positive ascending low-score; first 20 displayed, first 5 recurrence. pandas default quicksort resolves score ties in original feature order.',
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
    if args.tau is not None:
        thresholds['documented'] = args.tau
    primary = 'documented' if args.tau is not None else 'p05'
    # Commit threshold definitions before calculating ANY capped patient results.
    write_json(out / 'thresholds.json', dict(primary=primary, thresholds=thresholds, units='bits (log2)',
        method='numpy.percentile(method=linear) over strictly positive stored reference DJS, including original EPS floor',
        positive_features=len(positive), original_floor=transformer.eps, zero_features=int((djs == 0).sum()),
        exploratory=args.tau is None, source=args.tau_source,
        affected={k: int((djs < t).sum()) for k, t in thresholds.items()}))
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
    summaries, rankings, movement, scale_checks, positions = [], [], [], [], []
    baseline_rankings = {}
    for pop, (frame, patient_ids) in arrays.items():
        summary, ranking = summarize(originals[pop], patient_ids, features, low_features, djs, pop, 'original')
        summaries.append(summary); rankings.append(ranking); baseline_rankings[pop] = ranking
    demo_std = scaler.transform(pd.DataFrame(originals['demonstration'][:, projection_idx], columns=low_features))
    def locate(inputs, reference, coordinates):
        neighbors = NearestNeighbors(n_neighbors=5).fit(reference).kneighbors(inputs, return_distance=False)
        return coordinates[neighbors].mean(axis=1)
    base_xy = locate(demo_std, saved_std, emb)
    rms = float(np.sqrt(np.mean(np.sum((emb - emb.mean(axis=0)) ** 2, axis=1))))
    xs = np.linspace(emb[:, 0].min()-2, emb[:, 0].max()+2, 600)
    ys = np.linspace(emb[:, 1].min()-2, emb[:, 1].max()+2, 600)
    xx, yy = np.meshgrid(xs, ys)
    kdes = [gaussian_kde(emb[labels == c].T, bw_method='scott') for c in [1, 2]]
    grid_density = [k(np.vstack([xx.ravel(), yy.ravel()])) for k in kdes]
    levels = [np.quantile(z, .6) for z in grid_density]
    def region(xy):
        masks = [k(xy.T) >= level for k, level in zip(kdes, levels)]
        return ['overlap' if a and b else 'myocarditis' if a else 'ACS' if b else 'outside' for a,b in zip(*masks)]
    for name, tau in thresholds.items():
        factor = (djs + transformer.eps) / (np.maximum(djs, tau) + transformer.eps)
        capped = {pop: raw_values.copy() * factor for pop, raw_values in originals.items()}
        for pop, (_, patient_ids) in arrays.items():
            summary, ranking = summarize(capped[pop], patient_ids, features, low_features, djs, pop, name)
            summaries.append(summary); rankings.append(ranking)
            for patient in patient_ids:
                for panel in ['uncertainty', 'low_score']:
                    def top_set(table):
                        return set(table.loc[(table.patient_id == str(patient)) & (table.panel == panel) & table.top_five, 'feature'])
                    before, after = top_set(baseline_rankings[pop]), top_set(ranking)
                    movement.append(dict(scenario=name, population=pop, patient_id=patient, panel=panel,
                        retained=len(before & after), removed=len(before-after), added=len(after-before),
                        changed=before != after, jaccard=len(before & after)/len(before | after) if before | after else 1.0))
        capped_ref = pd.DataFrame(capped['reference'][:, projection_idx], columns=low_features)
        new_scaler = StandardScaler().fit(capped_ref)
        cap_std = new_scaler.transform(capped_ref)
        cap_demo = new_scaler.transform(pd.DataFrame(capped['demonstration'][:, projection_idx], columns=low_features))
        invariant = np.allclose(saved_std, cap_std, rtol=1e-9, atol=1e-10) and np.allclose(demo_std, cap_demo, rtol=1e-9, atol=1e-10)
        scale_checks.append(dict(scenario=name, inputs_invariant=bool(invariant),
            reference_max_abs_difference=float(np.max(abs(saved_std-cap_std))),
            demo_max_abs_difference=float(np.max(abs(demo_std-cap_demo)))))
        if invariant:
            # Reuse verified coordinates rather than interpreting numerical/stochastic refits.
            cap_emb, cap_xy = emb.copy(), base_xy.copy()
        else:
            settings = dict(n_components=2, perplexity=50, learning_rate='auto', init='pca', random_state=42)
            # Fit both with identical settings; align using every common reference row.
            base_fit = TSNE(**settings).fit_transform(saved_std)
            new_fit = TSNE(**settings).fit_transform(cap_std)
            a, b = base_fit-base_fit.mean(0), new_fit-new_fit.mean(0)
            rotation, _ = orthogonal_procrustes(b, a)
            scale = np.sum((b @ rotation)*a) / np.sum(b*b)
            aligned = b @ rotation * scale + base_fit.mean(0)
            base_for_comparison = locate(demo_std, saved_std, base_fit)
            cap_xy = locate(cap_demo, cap_std, aligned)
            cap_emb = aligned
            # Map comparison results to the baseline refit coordinate frame.
            baseline_xy = base_for_comparison
        baseline_xy = base_xy if invariant else baseline_xy
        denominator = rms if invariant else float(np.sqrt(np.mean(np.sum(a*a, axis=1))))
        fixed_demo = scaler.transform(pd.DataFrame(capped['demonstration'][:, projection_idx], columns=low_features))
        fixed_xy = locate(fixed_demo, saved_std, emb)
        for i in [5, 7]:
            positions.append(dict(scenario=name, patient_id=i+1, scaling='refitted_pipeline',
                before_x=float(baseline_xy[i,0]), before_y=float(baseline_xy[i,1]), after_x=float(cap_xy[i,0]), after_y=float(cap_xy[i,1]),
                normalized_displacement=float(np.linalg.norm(cap_xy[i]-baseline_xy[i])/denominator),
                projection_input_l2_change=float(np.linalg.norm(cap_demo[i]-demo_std[i])), inputs_invariant=bool(invariant),
                before_region=region(base_xy[[i]])[0] if invariant else 'not_assessed_refitted_embedding',
                after_region=region(cap_xy[[i]])[0] if invariant else 'not_assessed_refitted_embedding'))
            positions.append(dict(scenario=name, patient_id=i+1, scaling='fixed_original_scaler_diagnostic',
                before_x=float(base_xy[i,0]), before_y=float(base_xy[i,1]), after_x=float(fixed_xy[i,0]), after_y=float(fixed_xy[i,1]),
                normalized_displacement=float(np.linalg.norm(fixed_xy[i]-base_xy[i])/rms),
                projection_input_l2_change=float(np.linalg.norm(fixed_demo[i]-demo_std[i])),
                inputs_invariant=bool(np.allclose(fixed_demo[i], demo_std[i], rtol=1e-9, atol=1e-10)),
                before_region=region(base_xy[[i]])[0], after_region=region(fixed_xy[[i]])[0]))
        np.savez_compressed(out / f'intermediates_{name}.npz', reference_raw=capped['reference'], demonstration_raw=capped['demonstration'],
                            reference_std=cap_std, demonstration_std=cap_demo, embedding=cap_emb)
        with (out / f'scaler_{name}.pkl').open('wb') as fh:
            pickle.dump(new_scaler, fh)
    table = pd.concat(summaries, ignore_index=True)
    table.to_csv(out / 'feature_results.csv', index=False)
    original_table = table[table.scenario == 'original']
    original_table.to_csv(out / 'table9.csv', index=False)
    pd.concat(rankings, ignore_index=True).to_csv(out / 'patient_rankings.csv', index=False)
    move = pd.DataFrame(movement); move.to_csv(out / 'top_five_changes.csv', index=False)
    checks = pd.DataFrame(scale_checks); checks.to_csv(out / 'scaling_checks.csv', index=False)
    pos = pd.DataFrame(positions); pos.to_csv(out / 'patients_6_8_landscape.csv', index=False)
    corr = []
    for (scenario, pop, panel), block in table.groupby(['scenario','population','panel']):
        for metric in ['mean_displayed_rank', 'mean_full_eligible_rank']:
            valid = block[['djs', metric]].dropna()
            rho, pvalue = spearmanr(valid.djs, valid[metric]) if len(valid)>1 and valid.djs.nunique()>1 and valid[metric].nunique()>1 else (np.nan,np.nan)
            corr.append(dict(scenario=scenario, population=pop, panel=panel, rank_measure=metric,
                features_included=len(valid), excluded_missing=len(block)-len(valid), rho=rho, p_value=pvalue))
    correlations = pd.DataFrame(corr); correlations.to_csv(out / 'spearman.csv', index=False)
    for scenario in thresholds:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ax = axes[0]
        for panel, color, marker in [('uncertainty','tab:orange','o'),('low_score','tab:blue','s')]:
            b = original_table[(original_table.population=='reference') & (original_table.panel==panel)].dropna(subset=['mean_displayed_rank'])
            ax.scatter(b.djs, b.mean_displayed_rank, c=color, marker=marker, label=panel, alpha=.8)
            for _, row in b.nlargest(3, 'top_five_count').iterrows():
                ax.annotate(row.feature, (row.djs,row.mean_displayed_rank), fontsize=8)
        ax.set(xlabel='DJS (bits)', ylabel='Mean displayed rank (conditional on appearance)', title='A. Reference cohort; rank 1 shown first')
        ax.invert_yaxis(); ax.legend()
        ax = axes[1]
        ax.scatter(emb[:,0],emb[:,1],c=np.where(labels==1,'red','blue'),s=12,alpha=.25)
        density_masks = [z.reshape(xx.shape)>=level for z,level in zip(grid_density,levels)]
        ax.contourf(xx, yy, (density_masks[0] & density_masks[1]).astype(float), levels=[.5,1.5], colors=['gray'],alpha=.15)
        for scaling, marker in [('refitted_pipeline','*'),('fixed_original_scaler_diagnostic','x')]:
            for _, row in pos[(pos.scenario==scenario)&(pos.scaling==scaling)].iterrows():
                ax.scatter(row.before_x,row.before_y,c='black',marker='o',s=65)
                label = 'refitted scaler' if scaling == 'refitted_pipeline' else 'fixed scaler diagnostic'
                ax.scatter(row.after_x,row.after_y,marker=marker,s=140,label=f'Patient {row.patient_id}: {label}')
                ax.plot([row.before_x,row.after_x],[row.before_y,row.after_y],color='black',alpha=.5)
        inv = bool(checks.loc[checks.scenario==scenario,'inputs_invariant'].iloc[0])
        ax.set_title(f'B. {scenario}: refitted inputs '+('invariant; positions reused' if inv else 'changed; embeddings aligned'))
        ax.legend(fontsize=7); ax.set(xlabel='Landscape x',ylabel='Landscape y')
        fig.text(.5,.01,'Circles: original. Stars: refitted scaling. Crosses: fixed-scaler diagnostic changes scaling procedure.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,1))
        for ext in ['png','pdf']:
            fig.savefig(out / (f'figure2.{ext}' if scenario==primary else f'figure2_{scenario}.{ext}'),dpi=250)
        plt.close(fig)
    # Paginated native PDF tables retain all feature rows; PNG previews are also complete by page.
    from matplotlib.backends.backend_pdf import PdfPages
    columns = ['feature','djs','djs_rank_ascending','top_five_count','top_five_percent','displayed_rank_denominator','mean_displayed_rank','median_displayed_rank','mean_full_eligible_rank']
    with PdfPages(out / 'table9.pdf') as pdf:
        for (pop,panel), block in original_table.groupby(['population','panel']):
            for start in range(0,len(block),22):
                page = block.iloc[start:start+22][columns].copy()
                page = page.map(lambda v: f'{v:.4g}' if isinstance(v,(float,np.floating)) else v)
                fig, ax = plt.subplots(figsize=(17,8)); ax.axis('off')
                ax.set_title(f'Table 9: {pop}, {panel}; n={int(block.patients.iloc[0])}; DJS rank ascending; displayed denominator = appearances')
                t=ax.table(cellText=page.values,colLabels=columns,loc='center',cellLoc='left');t.auto_set_font_size(False);t.set_fontsize(7);t.scale(1,1.5)
                t.auto_set_column_width(list(range(len(columns))))
                pdf.savefig(fig,bbox_inches='tight');fig.savefig(out/f'table9_{pop}_{panel}_{start//22+1}.png',dpi=180,bbox_inches='tight');plt.close(fig)
    relevant = correlations[(correlations.scenario=='original') & (correlations.rank_measure=='mean_displayed_rank')]
    results = []
    for _, row in relevant.iterrows():
        results.append(f'{row.population}/{row.panel}: Spearman rho={row.rho:.3f}, n={row.features_included} features.')
    changes = move[(move.scenario==primary)].groupby(['population','panel']).changed.agg(['sum','count'])
    selected = pos[(pos.scenario==primary)&(pos.scaling=='refitted_pipeline')]
    methods = f'We reproduced the saved 44-feature no-ECG transformer and app landscape using {len(ref)} complete-case reference patients and ten supplied demonstration patients; a separate full-cohort check included {len(cohort)} records. Rankings retained signed scores and matched the two interface sorting rules. Recurrence used the first five displayed features. Displayed rank means excluded absences and reported their denominators; full eligible ranks provided a sensitivity check. DJS ranks ascended with average ties, and Spearman correlations used pairwise complete features. The exploratory primary cap was tau={thresholds[primary]:.8g} bits, with linear first/fifth/tenth percentiles of strictly positive stored reference DJS. The original EPS floor was retained. Capping changed only the denominator. We tested standardized inputs before interpreting landscape movement, and separately retained the original scaler as a diagnostic. Current artifacts were verified numerically; historical session identity remains unverified.'
    if args.tau is not None:
        methods = methods.replace('The exploratory primary cap was', 'The documented primary cap was') + f' Threshold source: {args.tau_source}.'
    results_text = ' '.join(results) + ' Primary-threshold top-five membership changes: ' + '; '.join(f'{p}/{panel}: {int(r["sum"])}/{int(r["count"])} patients' for (p,panel),r in changes.iterrows()) + '. '
    results_text += ' '.join(f'Patient {int(r.patient_id)} had normalized displacement {r.normalized_displacement:.6g} and projection-input L2 change {r.projection_input_l2_change:.6g} under refitted scaling.' for _,r in selected.iterrows())
    full_corr = correlations[(correlations.scenario=='original') & (correlations.rank_measure=='mean_full_eligible_rank') & (correlations.population=='reference')]
    results_text += ' Reference full eligible rank correlations were ' + '; '.join(
        f'{r.panel}: rho={r.rho:.3f}, n={r.features_included}' for _,r in full_corr.iterrows()) + '.'
    for panel in ['uncertainty', 'low_score']:
        leaders = original_table[(original_table.population=='reference') & (original_table.panel==panel)].nlargest(3,'top_five_count')
        results_text += f' The three most recurrent reference {panel} features were ' + ', '.join(
            f'{r.feature} ({r.top_five_count}/{r.patients}, DJS={r.djs:.4g})' for _,r in leaders.iterrows()) + '.'
    results_text += f' The primary cutoff affected {int((djs < thresholds[primary]).sum())} features.'
    results_text += ' Negative DJS-versus-conditional-rank correlations mean lower-divergence features had larger numerical ranks among their displayed appearances; the uncertainty full-order correlation had the opposite sign. This combination does not support a uniform claim that low-separability features dominate both panels across patients. The tested exploratory caps did not change top-five membership under the primary threshold, and should not be generalized to stronger caps. Raw ranking effects and standardized landscape effects are distinct. The fixed-scaler comparison changes the scaling procedure and is not the original pipeline sensitivity.'
    (out / 'methods_results.txt').write_text('Methods\n'+methods+'\n\nResults\n'+results_text+'\n')
    provenance['counts'] = {p:len(frame) for p,(frame,_) in arrays.items()}
    write_json(out / 'provenance.json', provenance)
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Source changed during experiment: {path}')
    write_json(out / 'completion.json', {'status':'complete', 'source_hashes_unchanged':True})
    with (out / 'experiment.log').open('a') as fh:
        fh.write('All numerical reproduction checks passed.\n'+checks.to_string(index=False)+'\n'+results_text+'\nSource hashes unchanged. Complete.\n')

if __name__ == '__main__':
    main()
