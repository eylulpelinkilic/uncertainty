"""Independent checks of the pinned-source extension and preserved 44-feature replay."""
import json
import hashlib
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True


def verify(directory):
    out=Path(directory);provenance=json.loads((out/'provenance.json').read_text())
    source=json.loads((out/'source_verification_withECG.json').read_text())
    commit='25a4992135c9b193e4c360874ddf8e362d5f0a87'
    assert source['commit']==commit==provenance['source_withECG']['commit']
    with (out/'source_artifacts/model_metadata.pkl').open('rb') as fh:metadata=pickle.load(fh)
    with (out/'source_artifacts/best_model_finetuned.pkl').open('rb') as fh:model=pickle.load(fh)
    features=metadata['features']
    assert features==list(model.named_steps['uncertainty'].feature_names_in_)==provenance['models']['withECG']['features']==source['exact_feature_order']
    assert len(features)==54 and len(source['ecg_features'])==7 and len(source['echocardiographic_features'])==3
    assert features[:44]==provenance['models']['noECG']['features']
    assert features[44:47]==['EF','Segmentary Wall Motion Abnormality','Pericardial Effusion']
    assert features[47:]==source['ecg_features']
    assert list(model.classes_)==list(metadata['class_labels'])==[1,2]
    for name in ['uncertainty_transformer.py','uncertainty_utils.py']:
        assert (out/'source_artifacts'/name).read_bytes()==(ROOT/name).read_bytes()
    workbook=pd.read_excel(provenance['source_workbooks']['withECG'])
    numeric=workbook[features].apply(pd.to_numeric,errors='coerce')
    independent_eligible=workbook.index[numeric.notna().all(axis=1)&workbook.GRUP.notna()].tolist()
    predictions=pd.read_csv(out/'predictions_withECG.csv')
    assert set(independent_eligible)==set(predictions.patient_id) and len(independent_eligible)==158
    assert predictions.training_fold_size.eq(157).all()
    folds=[json.loads(line) for line in (out/'fold_audit.jsonl').read_text().splitlines()]
    ecg=[row for row in folds if row['configuration']=='withECG']
    assert len(ecg)==158
    for row in ecg:
        assert row['source_commit']==commit
        assert set(row['training_ids'])==set(independent_eligible)-{row['held_out_id']}
        assert row['classes']==[1,2] and row['scaler_n_samples_seen']==157
    previous=Path(__file__).resolve().parent/'run_20261002T165335_102689Z'
    old=pd.read_csv(previous/'predictions_noECG.csv').set_index('patient_id').sort_index()
    current=pd.read_csv(out/'predictions_noECG.csv').set_index('patient_id').sort_index()
    np.testing.assert_array_equal(old.index,current.index)
    np.testing.assert_allclose(old[['p_myocarditis','p_acs','brier_contribution']],current[['p_myocarditis','p_acs','brier_contribution']],rtol=1e-12,atol=1e-12)
    audit=json.loads((out/'protected_file_audit.json').read_text())
    assert not audit['changed_files']
    for path,digest in audit['sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    warnings=[w for row in ecg for w in row['warnings']]
    warning_counts={w:warnings.count(w) for w in set(warnings)}
    return dict(status='passed',source_commit=commit,exact_feature_order_verified=True,transformer_byte_compatibility_verified=True,
        independently_reconstructed_complete_cases=len(independent_eligible),fold_source_commit_checks=len(ecg),
        previous_noECG_probabilities_and_contributions_reproduced=True,previous_noECG_run=str(previous),
        protected_files_unchanged=len(audit['sha256']),warnings=warning_counts,
        warning_note='The recorded Unknown solver options: iprint warning concerns a legacy verbosity option in sklearn/SciPy; saved solver and hyperparameters were retained. No convergence warning was recorded.' if set(warnings)<={'Unknown solver options: iprint'} else 'Inspect fold audit warnings.')

if __name__=='__main__':
    result=verify(sys.argv[1]);print(json.dumps(result,indent=2))
    (Path(sys.argv[1])/'extension_verification.json').write_text(json.dumps(result,indent=2)+'\n')
