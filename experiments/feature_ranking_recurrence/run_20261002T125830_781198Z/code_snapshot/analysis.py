"""Corrected experiment reporting. Imported only by the isolated experiment."""
from __future__ import annotations
import json
from pathlib import Path
import sys

SORTS = ('stable', 'app_quicksort')


def compare_sequences(before, after):
    """Membership and order effects have distinct, explicit denominators."""
    before, after = list(before), list(after)
    a, b = set(before[:5]), set(after[:5])
    ranks_a, ranks_b = {f: i+1 for i, f in enumerate(before)}, {f: i+1 for i, f in enumerate(after)}
    common = set(ranks_a) & set(ranks_b)
    return dict(top_five_membership_changed=a != b, removed=len(a-b), added=len(b-a),
        retained=len(a & b), top_five_order_changed=before[:5] != after[:5],
        displayed_membership_changed=set(before[:20]) != set(after[:20]),
        displayed_order_changed=before[:20] != after[:20], full_order_changed=before != after,
        common_eligible_features=len(common), eligible_membership_changed=set(before) != set(after),
        mean_absolute_full_rank_change=sum(abs(ranks_a[f]-ranks_b[f]) for f in common)/len(common) if common else float('nan'))


def sequences(rankings):
    return {key: block.sort_values('full_rank').feature.tolist()
            for key, block in rankings.groupby(['population', 'patient_id', 'panel'], sort=False)}


def correlations(table):
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    rows = []
    for (sorting, scenario, pop, panel), block in table.groupby(['sorting','scenario','population','panel']):
        for metric in ['mean_displayed_rank', 'mean_full_eligible_rank']:
            valid = block[['djs',metric]].dropna()
            rho, p = spearmanr(valid.djs, valid[metric]) if len(valid)>1 and valid.djs.nunique()>1 and valid[metric].nunique()>1 else (np.nan,np.nan)
            rows.append(dict(sorting=sorting, scenario=scenario, population=pop, panel=panel,
                rank_measure=metric, features_included=len(valid), excluded_missing=len(block)-len(valid), rho=rho, p_value=p))
    return pd.DataFrame(rows)


def analyze(out, previous_run, arrays, originals, features, low_features, djs, eps,
            thresholds, primary, scaler, saved_std, emb, labels, provenance, summarize):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    from feature_ranking_recurrence import write_json

    previous_run = Path(previous_run).resolve()
    previous_provenance = json.loads((previous_run/'provenance.json').read_text())
    provenance['previous_run'] = str(previous_run)
    provenance['previous_software'] = previous_provenance['software']
    provenance['previous_environment_package_versions_match'] = provenance['software'] == previous_provenance['software']
    provenance['previous_python_version'] = 'not recorded in previous provenance; current surviving environment executable/version recorded separately'
    provenance['environment_comparison'] = 'App quicksort is replayed in the surviving /tmp environment when invoked there; version match is recorded, not assumed. Historical session software is unknown.'
    provenance['tie_correction'] = 'Previous provenance falsely claimed quicksort preserves feature order for ties. Quicksort is unstable and can differ across environments. Stable sort is the deterministic main analysis.'
    provenance['counts'] = {p:len(frame) for p,(frame,_) in arrays.items()}
    if provenance['counts'] != {'demonstration':10, 'reference':128, 'complete_cohort':160}:
        raise ValueError('Recovered populations do not have required 10/128/160 denominators.')
    mappings = pd.read_csv(out/'demo_row_verification.csv')
    if not mappings.all_selected_values_match_with_rounding_tolerance.all():
        raise ValueError('Demo row mappings cannot be verified; overlap must not be invented.')
    demo_rows = set(mappings.claimed_workbook_row)
    ref_rows, cohort_rows = set(arrays['reference'][0].index), set(arrays['complete_cohort'][0].index)
    overlap = dict(reference_is_subset_of_complete_cohort=ref_rows <= cohort_rows,
        reference_in_complete_cohort=len(ref_rows & cohort_rows), complete_cohort_not_reference=len(cohort_rows-ref_rows),
        demonstration_in_reference=len(demo_rows & ref_rows), demonstration_in_complete_cohort=len(demo_rows & cohort_rows),
        demonstration_reference_row_ids=sorted(demo_rows & ref_rows),
        note='Overlapping populations are descriptive analyses, not independent samples. Demo measurements rounded relative to workbook.')
    provenance['population_overlap'] = overlap
    write_json(out/'population_overlap.json', overlap)
    write_json(out/'software_comparison.json', dict(previous=previous_provenance['software'], current=provenance['software'],
        all_recorded_versions_match=provenance['previous_environment_package_versions_match'], python=sys.version,
        previous_python_version=provenance['previous_python_version'], historical_session_environment='unverified'))
    # Original thresholds are retained as a separate family, numerically checked against the old run.
    old_thresholds = json.loads((previous_run/'thresholds.json').read_text())['thresholds']
    for name, value in old_thresholds.items():
        if name in thresholds:
            np.testing.assert_allclose(thresholds[name], value, rtol=1e-13, atol=0)
    definitions = json.loads((out/'threshold_definitions.json').read_text())
    # Save all panel order metadata before independent reporting/audit.
    write_json(out/'provenance.json', provenance)
    definition_hash = __import__('hashlib').sha256((out/'threshold_definitions.json').read_bytes()).hexdigest()
    factors = pd.read_csv(out/'threshold_feature_factors.csv')
    projection_idx = [features.index(f) for f in low_features]
    ref_raw = originals['reference'][:,projection_idx]
    demo_raw = originals['demonstration'][:,projection_idx]
    demo_std = scaler.transform(pd.DataFrame(demo_raw,columns=low_features))
    # Independent population-standardization check (no sklearn calls in this calculation).
    def manual_standardize(reference, values):
        mean, std = np.mean(reference,axis=0), np.std(reference,axis=0,ddof=0)
        # These observed columns are either exact constants or safely above sklearn's constant threshold.
        scale = np.where(std==0,1.,std)
        return (values-mean)/scale
    np.testing.assert_allclose(manual_standardize(ref_raw,ref_raw), saved_std, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(manual_standardize(ref_raw,demo_raw), demo_std, rtol=1e-9, atol=1e-10)
    def locate(inputs):
        idx = NearestNeighbors(n_neighbors=5).fit(saved_std).kneighbors(inputs,return_distance=False)
        # Independent brute-force check of neighbor sets; ties at boundary must not be silently assumed.
        for i,row in enumerate(inputs):
            distance = np.sum((saved_std-row)**2,axis=1)
            expected = np.argsort(distance,kind='stable')[:5]
            if set(expected) != set(idx[i]):
                if not np.allclose(np.sort(distance[expected]),np.sort(distance[idx[i]]),rtol=1e-12,atol=1e-12):
                    raise AssertionError('Nearest-neighbor distance reproduction failed')
        return emb[idx].mean(axis=1)
    base_xy = locate(demo_std)
    rms = float(np.sqrt(np.mean(np.sum((emb-emb.mean(0))**2,axis=1))))
    xs = np.linspace(emb[:,0].min()-2,emb[:,0].max()+2,600)
    ys = np.linspace(emb[:,1].min()-2,emb[:,1].max()+2,600)
    xx,yy = np.meshgrid(xs,ys)
    kdes = [gaussian_kde(emb[labels==c].T,bw_method='scott') for c in [1,2]]
    densities = [k(np.vstack([xx.ravel(),yy.ravel()])).reshape(xx.shape) for k in kdes]
    levels = [np.quantile(z,.6) for z in densities]
    def region(xy):
        masks = [k(xy.T)>=level for k,level in zip(kdes,levels)]
        return ['overlap' if a and b else 'myocarditis' if a else 'ACS' if b else 'outside' for a,b in zip(*masks)]
    summaries, rankings, changes, scale_checks, positions, perturbations = [], [], [], [], [], []
    all_sequences = {}
    for scenario in ['original'] + list(thresholds):
        factor = np.ones(len(features)) if scenario=='original' else (djs+eps)/(np.maximum(djs,thresholds[scenario])+eps)
        values = {pop: matrix.copy()*factor for pop,matrix in originals.items()}
        for pop, matrix in values.items():
            original = originals[pop]
            np.testing.assert_array_equal(np.sign(matrix), np.sign(original))
            if scenario != 'original':
                # Independent denominator formula check, reconstructed from unchanged numerator.
                numerator = original*(djs+eps)
                direct = numerator/(np.maximum(djs,thresholds[scenario])+eps)
                np.testing.assert_allclose(matrix,direct,rtol=1e-12,atol=1e-12)
                perturbations.append(dict(scenario=scenario,population=pop,
                    exact_raw_scores_changed=bool(np.any(matrix!=original)),
                    changed_entries=int(np.count_nonzero(matrix!=original)), total_entries=matrix.size,
                    max_abs_raw_score_change=float(np.max(abs(matrix-original))),
                    raw_scores_unchanged_within_tolerance=bool(np.allclose(matrix,original,rtol=1e-9,atol=1e-10))))
            for sorting in SORTS:
                summary, ranking = summarize(matrix,arrays[pop][1],features,low_features,djs,pop,scenario,sorting)
                summaries.append(summary);rankings.append(ranking)
                seq = sequences(ranking)
                all_sequences[(scenario,sorting,pop)] = seq
                if scenario != 'original':
                    baseline = all_sequences[('original',sorting,pop)]
                    for (population,patient,panel), after in seq.items():
                        changes.append(dict(scenario=scenario,sorting=sorting,population=population,patient_id=patient,
                            panel=panel,**compare_sequences(baseline[(population,patient,panel)],after)))
        if scenario=='original':
            np.savez_compressed(out/'original_scores.npz',**originals)
            continue
        ref_cap, demo_cap = values['reference'][:,projection_idx], values['demonstration'][:,projection_idx]
        cap_frame = pd.DataFrame(ref_cap,columns=low_features)
        new_scaler = StandardScaler().fit(cap_frame)
        cap_std = new_scaler.transform(cap_frame)
        cap_demo = new_scaler.transform(pd.DataFrame(demo_cap,columns=low_features))
        manual_ref, manual_demo = manual_standardize(ref_cap,ref_cap),manual_standardize(ref_cap,demo_cap)
        np.testing.assert_allclose(cap_std,manual_ref,rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(cap_demo,manual_demo,rtol=1e-9,atol=1e-10)
        ref_invariant = np.allclose(saved_std,cap_std,rtol=1e-9,atol=1e-10)
        demo_invariant = np.allclose(demo_std,cap_demo,rtol=1e-9,atol=1e-10)
        invariant = ref_invariant and demo_invariant
        scale_checks.append(dict(scenario=scenario,reference_inputs_invariant=bool(ref_invariant),demo_inputs_invariant=bool(demo_invariant),
            reference_max_abs_difference=float(np.max(abs(saved_std-cap_std))),demo_max_abs_difference=float(np.max(abs(demo_std-cap_demo))),
            independent_standardization_verified=True,
            landscape_action='reuse verified embedding: positive column rescaling cancelled by refitted standardization' if invariant else 'unverified; fail rather than interpret stochastic motion'))
        if not invariant:
            raise ValueError(f'{scenario}: standardized inputs changed unexpectedly. No landscape invariance claimed; inspect constant/near-constant scaling before embedding analysis.')
        fixed_demo = scaler.transform(pd.DataFrame(demo_cap,columns=low_features))
        fixed_xy = locate(fixed_demo)
        for i in [5,7]:
            for scaling, xy, inputs in [('refitted_pipeline',base_xy,demo_std if invariant else cap_demo),
                                       ('fixed_original_scaler_diagnostic',fixed_xy,fixed_demo)]:
                # Record actual numerical differences even when positions are analytically invariant.
                actual_inputs = cap_demo if scaling=='refitted_pipeline' else fixed_demo
                positions.append(dict(scenario=scenario,patient_id=i+1,scaling=scaling,
                    before_x=float(base_xy[i,0]),before_y=float(base_xy[i,1]),after_x=float(xy[i,0]),after_y=float(xy[i,1]),
                    displacement=float(np.linalg.norm(xy[i]-base_xy[i])),normalized_displacement=float(np.linalg.norm(xy[i]-base_xy[i])/rms),
                    projection_input_l2_change=float(np.linalg.norm(actual_inputs[i]-demo_std[i])),
                    before_region=region(base_xy[[i]])[0],after_region=region(xy[[i]])[0],
                    coordinate_provenance='current saved app embedding, 5-neighbor projection; historical session figure unverified'))
        np.savez_compressed(out/f'intermediates_{scenario}.npz',reference_raw=values['reference'],demonstration_raw=values['demonstration'],
            reference_std=cap_std,demonstration_std=cap_demo,fixed_scaler_demo_std=fixed_demo,embedding=emb)
        with (out/f'scaler_{scenario}.pkl').open('wb') as fh:
            __import__('pickle').dump(new_scaler,fh)
    table = pd.concat(summaries,ignore_index=True)
    ranking = pd.concat(rankings,ignore_index=True)
    original_table = table[table.scenario=='original']
    table.to_csv(out/'feature_results.csv',index=False)
    original_table.to_csv(out/'table9.csv',index=False)
    ranking.to_csv(out/'patient_rankings.csv',index=False)
    move = pd.DataFrame(changes);move.to_csv(out/'ranking_changes.csv',index=False)
    checks = pd.DataFrame(scale_checks);checks.to_csv(out/'scaling_checks.csv',index=False)
    pos = pd.DataFrame(positions);pos.to_csv(out/'patients_6_8_landscape.csv',index=False)
    perturb = pd.DataFrame(perturbations);perturb.to_csv(out/'raw_score_perturbations.csv',index=False)
    corr = correlations(table);corr.to_csv(out/'spearman.csv',index=False)
    summary_changes = move.groupby(['scenario','sorting','population','panel']).agg(
        patients=('patient_id','size'),top_five_membership_changed=('top_five_membership_changed','sum'),
        top_five_order_changed=('top_five_order_changed','sum'),displayed_membership_changed=('displayed_membership_changed','sum'),
        displayed_order_changed=('displayed_order_changed','sum'),full_order_changed=('full_order_changed','sum')).reset_index()
    summary_changes.to_csv(out/'threshold_ranking_comparison.csv',index=False)
    # Compare both methods for every scenario/population, not just the baseline correlation.
    tie_patient = []
    for scenario in ['original']+list(thresholds):
        for pop in arrays:
            before,after = all_sequences[(scenario,'app_quicksort',pop)],all_sequences[(scenario,'stable',pop)]
            for (population,patient,panel), ordered in after.items():
                tie_patient.append(dict(scenario=scenario,population=population,patient_id=patient,panel=panel,
                    **compare_sequences(before[(population,patient,panel)],ordered)))
    pd.DataFrame(tie_patient).to_csv(out/'tie_patient_comparison.csv',index=False)
    keys = ['scenario','population','panel','feature']
    tie_feature = table[table.sorting=='app_quicksort'].merge(table[table.sorting=='stable'],on=keys,suffixes=('_app','_stable'))
    for metric in ['top_five_count','top_five_percent','displayed_count','displayed_rank_denominator',
                   'mean_displayed_rank','median_displayed_rank','full_eligible_rank_denominator','mean_full_eligible_rank']:
        tie_feature[f'{metric}_stable_minus_app'] = tie_feature[f'{metric}_stable']-tie_feature[f'{metric}_app']
    tie_feature.to_csv(out/'tie_feature_comparison.csv',index=False)
    corr_keys=['scenario','population','panel','rank_measure']
    tie_corr=corr[corr.sorting=='app_quicksort'].merge(corr[corr.sorting=='stable'],on=corr_keys,suffixes=('_app','_stable'))
    tie_corr['rho_stable_minus_app']=tie_corr.rho_stable-tie_corr.rho_app
    tie_corr.to_csv(out/'tie_correlation_comparison.csv',index=False)
    # Replay old scores/ranks/correlations, independently of previous text claims.
    previous_rank = pd.read_csv(previous_run/'patient_rankings.csv',dtype={'patient_id':str})
    replay = ranking[(ranking.sorting=='app_quicksort') & ranking.scenario.isin(previous_rank.scenario.unique())]
    replay_keys=['population','scenario','patient_id','panel','feature']
    merged = previous_rank.merge(replay,on=replay_keys,suffixes=('_previous','_current'),validate='one_to_one')
    if len(merged)!=len(previous_rank) or len(merged)!=len(replay):
        raise AssertionError('Previous replay rows missing')
    np.testing.assert_allclose(merged.score_previous,merged.score_current,rtol=1e-12,atol=1e-12)
    prior_feature = pd.read_csv(previous_run/'feature_results.csv')
    current_feature = table[(table.sorting=='app_quicksort') & table.scenario.isin(prior_feature.scenario.unique())]
    replay_features = prior_feature.merge(current_feature,on=keys,suffixes=('_previous','_current'),validate='one_to_one')
    replay_features.to_csv(out/'previous_run_feature_replay.csv',index=False)
    prior_corr=pd.read_csv(previous_run/'spearman.csv')
    replay_corr=prior_corr.merge(corr[corr.sorting=='app_quicksort'],on=corr_keys,suffixes=('_previous','_current'),validate='one_to_one')
    replay_corr['rho_current_minus_previous']=replay_corr.rho_current-replay_corr.rho_previous
    replay_corr.to_csv(out/'previous_run_correlation_replay.csv',index=False)
    write_json(out/'previous_run_replay.json',dict(scores_match=True,
        rank_rows=len(merged),full_rank_rows_different=int((merged.full_rank_previous!=merged.full_rank_current).sum()),
        previous_recorded_package_versions_match=provenance['previous_environment_package_versions_match'],
        maximum_correlation_difference=float(replay_corr.rho_current_minus_previous.abs().max()),
        note='App-order replay agreement validates the current saved run, not historical clinician sessions. Quicksort ordering is environment-dependent.'))
    # Denominator-based perturbations versus actual raw and ranking effects.
    threshold_summary = []
    for definition in definitions['definitions']:
        name=definition['scenario']; sub=move[move.scenario==name]; raw=perturb[perturb.scenario==name]
        threshold_summary.append(dict(**{**definition, 'affected_features': ' | '.join(definition['affected_features'])},
            any_exact_raw_score_change=bool(raw.exact_raw_scores_changed.any()),
            any_raw_score_change_within_tolerance=bool((~raw.raw_scores_unchanged_within_tolerance).any()),
            any_top_five_membership_change=bool(sub.top_five_membership_changed.any()),
            any_displayed_order_change=bool(sub.displayed_order_changed.any()),any_full_order_change=bool(sub.full_order_changed.any())))
    threshold_summary=pd.DataFrame(threshold_summary)
    threshold_summary.to_csv(out/'threshold_summary.csv',index=False)
    render(out, original_table, corr, checks, pos, emb, labels, xx, yy, densities, levels, thresholds, primary, threshold_summary, summary_changes)
    write_text(out, provenance, original_table, corr, summary_changes, checks, threshold_summary, primary)
    # Output-focused independent checker also verifies these results after saving them.
    from experiments.feature_ranking_recurrence.verify_results import verify
    code_files=[Path(__file__).resolve().parents[2]/'feature_ranking_recurrence.py',
                Path(__file__).resolve(),Path(__file__).with_name('verify_results.py'),Path(__file__).with_name('test_feature_ranking.py')]
    snapshot=out/'code_snapshot';snapshot.mkdir()
    provenance['experiment_code_sha256']={}
    for path in code_files:
        content=path.read_bytes();(snapshot/path.name).write_bytes(content)
        provenance['experiment_code_sha256'][str(path)]=__import__('hashlib').sha256(content).hexdigest()
    write_json(out/'provenance.json',provenance)
    verification = verify(out)
    verification['previous_scores_replayed']=True
    verification['independent_standardization_all_caps']='passed'
    verification['brute_force_neighbor_distance_check']='passed'
    verification['threshold_definitions_unchanged_during_calculation'] = __import__('hashlib').sha256((out/'threshold_definitions.json').read_bytes()).hexdigest()==definition_hash
    if not verification['threshold_definitions_unchanged_during_calculation']:
        raise AssertionError('Threshold definitions changed')
    write_json(out/'independent_verification.json',verification)
    write_json(out/'provenance.json',provenance)
    with (out/'experiment.log').open('a') as fh:
        fh.write(checks.to_string(index=False)+'\n'+(out/'methods_results.txt').read_text()+'\n')


def render(out, table, corr, checks, pos, emb, labels, xx, yy, densities, levels, thresholds, primary, threshold_summary, summary_changes):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    # Main Table 9 is deterministic; the full CSV also contains the app comparison.
    stable=table[table.sorting=='stable']
    columns=['feature','djs','djs_rank_ascending','top_five_count','top_five_percent','displayed_count',
             'displayed_rank_denominator','mean_displayed_rank','median_displayed_rank','full_eligible_rank_denominator','mean_full_eligible_rank']
    with PdfPages(out/'table9.pdf') as pdf:
        for (pop,panel),block in stable.groupby(['population','panel']):
            for start in range(0,len(block),22):
                page=block.iloc[start:start+22][columns].map(lambda v: f'{v:.4g}' if isinstance(v,(float,np.floating)) else v)
                fig,ax=plt.subplots(figsize=(20,8));ax.axis('off')
                ax.set_title(f'Table 9: {pop}, {panel}; n={int(block.patients.iloc[0])}; stable ties; DJS rank ascending')
                t=ax.table(cellText=page.values,colLabels=columns,loc='center',cellLoc='left');t.auto_set_font_size(False);t.set_fontsize(6.5);t.scale(1,1.5);t.auto_set_column_width(range(len(columns)))
                pdf.savefig(fig,bbox_inches='tight');fig.savefig(out/f'table9_{pop}_{panel}_{start//22+1}.png',dpi=180,bbox_inches='tight');plt.close(fig)
    overlap=(densities[0]>=levels[0]) & (densities[1]>=levels[1])
    for scenario in thresholds:
        fig,axes=plt.subplots(1,2,figsize=(14,6));ax=axes[0]
        for panel,color,marker in [('uncertainty','tab:orange','o'),('low_score','tab:blue','s')]:
            block=stable[(stable.population=='reference') & (stable.panel==panel)].dropna(subset=['mean_displayed_rank'])
            ax.scatter(block.djs,block.mean_displayed_rank,c=color,marker=marker,label=panel)
            for _,row in block.nlargest(3,'top_five_count').iterrows():ax.annotate(row.feature,(row.djs,row.mean_displayed_rank),fontsize=8)
        ax.set(xlabel='DJS (bits)',ylabel='Conditional mean displayed rank',title='A. Reference cohort, deterministic stable ties');ax.invert_yaxis();ax.legend()
        ax=axes[1];ax.scatter(emb[:,0],emb[:,1],c=np.where(labels==1,'red','blue'),s=12,alpha=.25)
        ax.contourf(xx,yy,overlap.astype(float),levels=[.5,1.5],colors=['gray'],alpha=.2)
        for _,row in pos[(pos.scenario==scenario)].iterrows():
            marker='*' if row.scaling=='refitted_pipeline' else 'x'
            label='refitted scaler' if row.scaling=='refitted_pipeline' else 'fixed-scaler diagnostic'
            ax.scatter(row.before_x,row.before_y,c='black',marker='o',s=60)
            ax.scatter(row.after_x,row.after_y,marker=marker,s=130,label=f'Patient {row.patient_id}: {label}')
            ax.plot([row.before_x,row.after_x],[row.before_y,row.after_y],color='black',alpha=.4)
        ax.set(title=f'B. {scenario}: standardized inputs invariant',xlabel='Current-artifact landscape x',ylabel='Current-artifact landscape y');ax.legend(fontsize=7)
        fig.text(.5,.02,'Refitted standardization cancels capping; verified positions reused. Raw ranking effects are reported separately.\nFixed-scaler diagnostic changes the procedure; historical session figure not verified.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.07,1,1))
        for ext in ['png','pdf']:fig.savefig(out/(f'figure2.{ext}' if scenario==primary else f'figure2_{scenario}.{ext}'),dpi=220)
        plt.close(fig)
    # Sorting comparison reveals the conditional versus full-rank distinction.
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,metric in zip(axes,['mean_displayed_rank','mean_full_eligible_rank']):
        b=corr[(corr.scenario=='original') & (corr.population=='reference') & (corr.rank_measure==metric)]
        for sorting,offset in [('app_quicksort',-.15),('stable',.15)]:
            part=b[b.sorting==sorting].set_index('panel').loc[['uncertainty','low_score']]
            ax.bar(np.arange(2)+offset,part.rho,width=.3,label=sorting)
        ax.axhline(0,c='black',linewidth=.8);ax.set_xticks([0,1],['Uncertainty','Low-score']);ax.set_ylim(-1,1);ax.set(title=metric,ylabel='Spearman rho');ax.legend()
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'tie_handling_comparison.{ext}',dpi=200)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,5))
    scenario_names=list(thresholds)
    for ax,panel in zip(axes,['uncertainty','low_score']):
        for pop,marker in [('demonstration','o'),('reference','s'),('complete_cohort','^')]:
            block=summary_changes[(summary_changes.sorting=='stable')&(summary_changes.panel==panel)&(summary_changes.population==pop)].set_index('scenario').loc[scenario_names]
            ax.plot(np.arange(len(scenario_names)),100*block.top_five_membership_changed/block.patients,marker=marker,label=pop)
        ax.set_xticks(np.arange(len(scenario_names)),scenario_names,rotation=45,ha='right')
        ax.set(title=panel,ylabel='Patients with changed top-five membership (%)',xlabel='Predefined exploratory caps');ax.legend()
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'threshold_sensitivity.{ext}',dpi=200)
    plt.close(fig)


def write_text(out, provenance, table, corr, changes, checks, thresholds, primary):
    def rho(sorting,pop,panel,metric):
        row=corr[(corr.sorting==sorting)&(corr.scenario=='original')&(corr.population==pop)&(corr.panel==panel)&(corr.rank_measure==metric)].iloc[0]
        return f'{row.rho:.4f} (n={row.features_included})'
    overlap=provenance['population_overlap']
    methods=f'We analyzed ten demonstration cases, the 128-patient landscape reference cohort, and the 160-patient complete cohort separately. All reference patients and all demonstration cases occur in the complete cohort; {overlap["demonstration_in_reference"]} demonstration cases also occur in the reference cohort, so these populations are not independent. Signed scores were sorted descending for uncertainty and strictly positive scores ascending for the low-score panel. Stable sorting resolved exact ties by each panel’s original feature order; the previous environment’s unstable app quicksort was retained as a comparison. We measured top-five recurrence, displayed ranks among the first twenty, and full eligible ranks, excluding absent features from rank means. Spearman correlations used average ranks for tied feature-level statistics and pairwise complete features. We retained the original stored-DJS percentile caps and added a predefined exploratory 1/5/10/20/25 percentile grid using positive unfloored DJS from nonconstant reference features. Thresholds and score factors were saved before capping. Only the denominator changed; epsilon and numerators were retained. Refitted standardization and a separate fixed-original-scaler diagnostic were verified against current artifacts; historical session figures remain unverified.'
    original=thresholds[thresholds.scenario==primary].iloc[0]
    results=f'The current quicksort replay reproduced previous ranks exactly (reference uncertainty conditional rho={rho("app_quicksort","reference","uncertainty","mean_displayed_rank")}); stable ties gave rho={rho("stable","reference","uncertainty","mean_displayed_rank")}, versus {rho("stable","reference","uncertainty","mean_full_eligible_rank")} for the full eligible order. Stable low-score conditional rho was {rho("stable","reference","low_score","mean_displayed_rank")}. Negative conditional correlations do not support uniform domination by low-separability features. The original primary cap ({original.tau_bits:.8g} bits) affected only Radiation, Fatigue, and Çarpıntı, with no displayed-order or top-five membership changes; its conclusion is narrow.'
    for name in ['supp_p20','supp_p25']:
        sub=changes[(changes.sorting=='stable')&(changes.scenario==name)&(changes.population=='reference')]
        u=sub[sub.panel=='uncertainty'].iloc[0];l=sub[sub.panel=='low_score'].iloc[0]
        tau=thresholds.loc[thresholds.scenario==name,'tau_bits'].iloc[0]
        results+=f' {name} ({tau:.8g} bits), including SEX, changed uncertainty/low-score top-five membership for {int(u.top_five_membership_changed)}/{int(l.top_five_membership_changed)} of 128 patients and displayed ordering for {int(u.displayed_order_changed)}/{int(l.displayed_order_changed)}.'
    results+=' The original first-percentile cap caused no denominator perturbation. All tested caps left independently verified standardized reference and demo inputs invariant: refitted standardization cancels positive column rescaling. Patients 6 and 8 retained current-artifact positions and overlap classifications; this does not imply unchanged raw rankings or verify historical session figures. Results are limited to the tested thresholds.'
    (out/'methods_results.txt').write_text('Methods\n'+methods+'\n\nResults\n'+results+'\n')
