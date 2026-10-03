"""Independent audit from saved Excel/CSV predictions and fold evidence.
Usage: python verify_results.py RUN_DIRECTORY
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def verify(directory):
    out=Path(directory);provenance=json.loads((out/'provenance.json').read_text())
    workbook=out/'patient_predictions.xlsx'
    summary=pd.read_excel(workbook,sheet_name='summary')
    bins=pd.read_excel(workbook,sheet_name='calibration_bins')
    folds=[json.loads(line) for line in (out/'fold_audit.jsonl').read_text().splitlines()]
    checked=[]
    for config in provenance['available_configurations']:
        p=pd.read_excel(workbook,sheet_name=f'predictions_{config}')
        csv=pd.read_csv(out/f'predictions_{config}.csv')
        pd.testing.assert_frame_equal(p,csv,check_dtype=False,check_exact=False,rtol=1e-13,atol=1e-14)
        expected=json.loads((out/f'eligible_{config}.json').read_text())['patient_ids']
        if len(p)!=len(expected) or p.patient_id.duplicated().any() or set(p.patient_id)!=set(expected):raise AssertionError('Incomplete or repeated OOF patients')
        assert (p.source_row==p.patient_id).all()
        assert (p.training_fold_size==len(expected)-1).all()
        probability=p[['p_myocarditis','p_acs']].to_numpy(float)
        assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
        np.testing.assert_allclose(probability.sum(1),1,rtol=0,atol=1e-12)
        source=pd.read_excel(provenance['source_workbooks'][config],sheet_name=0)
        np.testing.assert_array_equal(p.true_label,source.loc[p.source_row,'GRUP'])
        np.testing.assert_array_equal(p.y_myocarditis,(p.true_label==1).astype(int))
        np.testing.assert_array_equal(p.prediction_correct,p.predicted_label==p.true_label)
        np.testing.assert_allclose(p.predicted_class_confidence,np.where(p.predicted_label==1,p.p_myocarditis,p.p_acs),rtol=1e-13)
        contributions=np.square(p.p_myocarditis.to_numpy()-p.y_myocarditis.to_numpy())
        np.testing.assert_allclose(p.brier_contribution,contributions,rtol=1e-13,atol=1e-14)
        aggregate=float(sum(float(v) for v in contributions)/len(p))
        saved=summary[summary.configuration==config].iloc[0]
        np.testing.assert_allclose(saved.brier_score,aggregate,rtol=1e-13,atol=1e-14)
        assert saved.patients==len(p)
        selected_folds=[f for f in folds if f['configuration']==config]
        assert len(selected_folds)==len(p)
        for f in selected_folds:
            assert f['held_out_id'] not in f['training_ids']
            assert len(f['training_ids'])==len(expected)-1
            assert set(f['training_ids'])==set(expected)-{f['held_out_id']}
            assert f['scaler_n_samples_seen']==len(expected)-1
            row=p[p.loocv_fold==f['fold']].iloc[0];assert row.patient_id==f['held_out_id']
            assert set(f['classes'])=={1,2}
            assert json.loads(row.fitted_classes)==f['classes']
            for member in f['member_mappings'].values():assert set(member['diagnostic_classes'])=={1,2}
        bin_checks=0
        for kind in ['myocarditis_probability','confidence_correctness']:
            probability=p.p_myocarditis.to_numpy() if kind=='myocarditis_probability' else p.predicted_class_confidence.to_numpy()
            outcome=p.y_myocarditis.to_numpy() if kind=='myocarditis_probability' else p.prediction_correct.to_numpy(int)
            for n in [5,10]:
                saved_bins=bins[(bins.configuration==config)&(bins.diagram==kind)&(bins.n_bins==n)].sort_values('bin')
                assert len(saved_bins)==n and saved_bins.patient_count.sum()==len(p)
                independent_edges=np.arange(n+1,dtype=float)/n
                for i,(_,row) in enumerate(saved_bins.iterrows()):
                    # Direct interval masks, independently of the analysis searchsorted implementation.
                    lower,upper=independent_edges[i],independent_edges[i+1]
                    np.testing.assert_allclose([row.lower,row.upper],[lower,upper],rtol=0,atol=1e-15)
                    # Use serialized exact bounds to avoid last-bit differences at decimal endpoints.
                    mask=(probability>=row.lower)&((probability<row.upper) if i<n-1 else (probability<=row.upper))
                    assert row.patient_count==int(mask.sum())
                    if not mask.any():assert pd.isna(row.observed_proportion) and pd.isna(row.mean_predicted_probability)
                    else:
                        np.testing.assert_allclose([row.mean_predicted_probability,row.observed_proportion],[probability[mask].mean(),outcome[mask].mean()],rtol=1e-13,atol=1e-14)
                    bin_checks+=1
        checked.append({'configuration':config,'patients':len(p),'brier_independently_recomputed':aggregate,'bins_checked':bin_checks,'fold_exclusion_checks':len(selected_folds)})
    return {'status':'passed','source':'saved Excel workbook and matching CSV predictions','configurations':checked,
            'checks':'unique complete eligible patients; source labels; per-fold class mappings and training exclusion; probability simplex; single binary squared error; direct-interval bin recalculation'}

if __name__=='__main__':print(json.dumps(verify(sys.argv[1]),indent=2))
