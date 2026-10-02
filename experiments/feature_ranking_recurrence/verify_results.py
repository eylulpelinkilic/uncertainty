"""Independent NumPy-based audit of saved results; no ranking helper imports.
Run: python experiments/feature_ranking_recurrence/verify_results.py RUN_DIRECTORY
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def verify(directory):
    out=Path(directory)
    provenance=json.loads((out/'provenance.json').read_text()) if (out/'provenance.json').exists() else None
    # During in-process checks final provenance is written afterward; original stats give feature order.
    stats=json.loads((out/'reference_statistics.json').read_text())
    features=[row['feature'] for row in stats]
    djs=np.array([row['djs'] for row in stats])
    factors=pd.read_csv(out/'threshold_feature_factors.csv')
    definitions=json.loads((out/'threshold_definitions.json').read_text())
    eps=definitions['original_epsilon']
    eligible=definitions['supplementary_included_features']
    unfloored={row['feature']:row['djs_unfloored'] for row in stats}
    for definition in definitions['definitions']:
        name,tau=definition['scenario'],definition['tau_bits']
        f=factors[factors.scenario==name].set_index('feature').loc[features]
        np.testing.assert_allclose(f.multiplicative_score_change_factor,(djs+eps)/(np.maximum(djs,tau)+eps),rtol=1e-13,atol=0)
        assert set(f.index[f.affected])==set(definition['affected_features'])
        if name.startswith('supp_'):
            percentile=int(name.rsplit('p',1)[1])
            # Independent linear percentile via sorted order and fractional interpolation.
            vals=sorted(unfloored[feature] for feature in eligible)
            x=(len(vals)-1)*percentile/100;lo=int(np.floor(x));hi=int(np.ceil(x))
            independent_tau=vals[lo]+(x-lo)*(vals[hi]-vals[lo])
            np.testing.assert_allclose(tau,independent_tau,rtol=1e-13,atol=0)
    ranks=pd.read_csv(out/'patient_rankings.csv',dtype={'patient_id':str})
    summaries=pd.read_csv(out/'feature_results.csv')
    # Recompute statistics directly from saved appearances, without summarize().
    stat_checks=0
    for key,block in summaries.groupby(['scenario','sorting','population','panel']):
        scenario,sorting,pop,panel=key
        patient_ranks=ranks[(ranks.scenario==scenario)&(ranks.sorting==sorting)&(ranks.population==pop)&(ranks.panel==panel)]
        for _,row in block.iterrows():
            feature=patient_ranks[patient_ranks.feature==row.feature]
            full=feature.full_rank.to_numpy();shown=full[full<=20]
            assert row.top_five_count==np.count_nonzero(full<=5)
            assert row.displayed_count==len(shown)==row.displayed_rank_denominator
            assert row.full_eligible_rank_denominator==len(full)
            np.testing.assert_allclose(row.top_five_percent,100*np.count_nonzero(full<=5)/row.patients,rtol=1e-13)
            for actual,expected in [(row.mean_displayed_rank,np.mean(shown) if len(shown) else np.nan),
                                    (row.median_displayed_rank,np.median(shown) if len(shown) else np.nan),
                                    (row.mean_full_eligible_rank,np.mean(full) if len(full) else np.nan)]:
                np.testing.assert_allclose(actual,expected,rtol=1e-13,equal_nan=True)
            stat_checks+=1
    assert ranks.loc[ranks.panel=='low_score','score'].gt(0).all()
    assert (ranks.displayed==(ranks.full_rank<=20)).all()
    assert (ranks.top_five==(ranks.full_rank<=5)).all()
    # Stable order independently uses NumPy lexsort(score, original feature position).
    # Primary key is signed score, secondary key is original panel feature position.
    feature_lists={'uncertainty':features,'low_score':provenance['projection_features'] if provenance and 'projection_features' in provenance else features}
    original_npz=np.load(out/'original_scores.npz')
    demo_ids=pd.read_csv(out/'demo_ids.csv').patient_id.astype(str).tolist()
    # Recovered row identities in saved original rankings avoid positional guessing.
    ids_by_pop={p:ranks[(ranks.scenario=='original')&(ranks.sorting=='stable')&(ranks.population==p)&(ranks.panel=='uncertainty')].patient_id.drop_duplicates().tolist() for p in original_npz.files}
    stable=ranks[ranks.sorting=='stable']
    ordered_lookup={(scenario,pop,patient,panel):b.sort_values('full_rank').feature.tolist()
        for (scenario,pop,patient,panel),b in stable.groupby(['scenario','population','patient_id','panel'])}
    order_checks=0
    for scenario in stable.scenario.unique():
        factor=np.ones(len(features)) if scenario=='original' else factors[factors.scenario==scenario].set_index('feature').loc[features].multiplicative_score_change_factor.to_numpy()
        for pop in original_npz.files:
            for patient,raw in zip(ids_by_pop[pop],original_npz[pop]):
                values=raw*factor
                for panel,order in feature_lists.items():
                    idx=np.array([features.index(f) for f in order]);v=values[idx]
                    selected=np.arange(len(order)) if panel=='uncertainty' else np.flatnonzero(v>0)
                    score_key=-v[selected] if panel=='uncertainty' else v[selected]
                    ordered=selected[np.lexsort((selected,score_key))]
                    expected=[order[j] for j in ordered]
                    assert ordered_lookup.get((scenario,pop,patient,panel),[])==expected,(scenario,pop,patient,panel)
                    order_checks+=1
    original_npz.close()
    # Spearman is Pearson correlation of average rank vectors, independently of scipy.spearmanr.
    corr=pd.read_csv(out/'spearman.csv');corr_checks=0
    for _,row in corr.iterrows():
        block=summaries[(summaries.scenario==row.scenario)&(summaries.sorting==row.sorting)&(summaries.population==row.population)&(summaries.panel==row.panel)]
        valid=block[['djs',row.rank_measure]].dropna()
        assert len(valid)==row.features_included
        x,y=rankdata(valid.djs,method='average'),rankdata(valid[row.rank_measure],method='average')
        expected=np.corrcoef(x,y)[0,1] if len(x)>1 and np.std(x)>0 and np.std(y)>0 else np.nan
        np.testing.assert_allclose(row.rho,expected,rtol=1e-12,atol=1e-12,equal_nan=True);corr_checks+=1
    # Verify saved membership/order change counts from ordered sequences, not from analysis helpers.
    lookup={(sorting,scenario,pop,patient,panel):b.sort_values('full_rank').feature.tolist()
        for (sorting,scenario,pop,patient,panel),b in ranks.groupby(['sorting','scenario','population','patient_id','panel'])}
    change_table=pd.read_csv(out/'ranking_changes.csv',dtype={'patient_id':str})
    for _,row in change_table.iterrows():
        a=lookup.get((row.sorting,'original',row.population,row.patient_id,row.panel),[])
        b=lookup.get((row.sorting,row.scenario,row.population,row.patient_id,row.panel),[])
        assert row.top_five_membership_changed==(set(a[:5])!=set(b[:5]))
        assert row.top_five_order_changed==(a[:5]!=b[:5])
        assert row.displayed_order_changed==(a[:20]!=b[:20])
        assert row.full_order_changed==(a!=b)
    return dict(status='passed',feature_statistics_checked=stat_checks,stable_patient_orders_checked=order_checks,
        correlations_checked=corr_checks,patient_change_rows_checked=len(change_table),
        independent_methods='NumPy lexsort for stable order; direct appearance statistics; rankdata plus Pearson for Spearman; linear percentile interpolation; explicit denominator factors')

if __name__=='__main__':
    print(json.dumps(verify(Path(sys.argv[1])),indent=2))
