"""Read-only verification of the user-pinned 54-feature repository artifacts."""
import ast
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import numpy as np

PINNED_COMMIT='25a4992135c9b193e4c360874ddf8e362d5f0a87'
SOURCE_URL='https://github.com/eylulpelinkilic/EKG-uncertainty'
ECG_FEATURES=['ECG_ST depression','ECG_Location of ST depression ','Level of ST-Dep_mm',
              'ECG_T neg','ECG_Location of T negativity','Level of T invertion_mm','ECG_Q waves']
ECHO_FEATURES=['EF','Segmentary Wall Motion Abnormality','Pericardial Effusion']


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def verify_source(repo,commit,original_root,data,no_ecg_features,prepare,model_spec,out,save_json):
    from sklearn.base import clone
    repo=Path(repo).resolve()
    if commit!=PINNED_COMMIT or git(repo,'rev-parse','HEAD')!=commit:raise ValueError('Source checkout does not match the user-pinned commit')
    if git(repo,'status','--porcelain'):raise ValueError('Pinned source checkout contains modifications')
    # Byte equality establishes compatibility without modifying/importing foreign source modules.
    compatible={}
    for name in ['uncertainty_transformer.py','uncertainty_utils.py']:
        local=(original_root/name).read_bytes();foreign=(repo/name).read_bytes()
        compatible[name]={'byte_identical':local==foreign,'sha256':hashlib.sha256(foreign).hexdigest()}
        if local!=foreign:raise ValueError(f'{name} differs; do not deserialize using an unverified transformer implementation')
    with (repo/'best_model_finetuned.pkl').open('rb') as fh:model=pickle.load(fh)
    with (repo/'model_metadata.pkl').open('rb') as fh:metadata=pickle.load(fh)
    spec=model_spec(model);features=spec['features'];transformer=model.named_steps['uncertainty']
    expected=list(no_ecg_features)+ECHO_FEATURES+ECG_FEATURES
    if features!=expected or list(metadata['features'])!=expected or list(transformer.feature_names)!=expected:
        raise ValueError('Metadata, fitted transformer, constructor feature order, or 44+3+7 specification disagree')
    notebook=json.loads((repo/'finetuning.ipynb').read_text());declared=None
    for cell in notebook['cells']:
        code=''.join(cell.get('source',[]))
        if cell['cell_type']=='code' and code.startswith('FEATURES ='):
            assignment=ast.parse(code).body[0];declared=ast.literal_eval(assignment.value);break
    if declared!=features:raise ValueError('Original notebook feature order differs from saved artifacts')
    if list(metadata['class_labels'])!=list(model.classes_) or list(model.classes_)!=[1,2]:raise ValueError('Source diagnostic class mappings disagree')
    if metadata['n_bins']!=transformer.n_bins or metadata['eps']!=transformer.eps:raise ValueError('Metadata transformer settings differ')
    if [m['class'] for m in spec['ensemble_members']]!=['LogisticRegression','SVC','KNeighborsClassifier']:
        raise ValueError('54-feature ensemble members/order differ from saved LR+SVC+KNN specification')
    frame,excluded=prepare(data,features)
    with (repo/'split_indices.pkl').open('rb') as fh:split=pickle.load(fh)
    train_ids=[i for i in split['train_idx'] if i in frame.index]
    training=frame.loc[train_ids,features]
    rebuilt=clone(transformer).fit(training.copy(),frame.loc[train_ids,'GRUP'].to_numpy())
    for f in features:
        a,b=transformer._feat_stats_[f],rebuilt._feat_stats_[f]
        for attribute in ['mu','std','entropy']:
            np.testing.assert_allclose(list(getattr(a,attribute).values()),list(getattr(b,attribute).values()),rtol=1e-9,atol=1e-10,err_msg=f'Saved reference {attribute} differs for {f}')
        np.testing.assert_allclose(a.js,b.js,rtol=1e-9,atol=1e-10)
    unc=transformer.transform(training.copy())
    np.testing.assert_allclose(model.named_steps['scaler'].mean_,unc.mean(0),rtol=1e-9,atol=1e-10)
    if int(model.named_steps['scaler'].n_samples_seen_)!=len(training):raise ValueError('Saved scaler training size disagrees')
    # Hash all tracked source files, not only model artifacts, for an end-of-run preservation check.
    tracked=git(repo,'ls-files','-z').split('\0')
    hashes={str(repo/name):hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in tracked if name}
    archive=out/'source_artifacts';archive.mkdir()
    for name in ['best_model_finetuned.pkl','model_metadata.pkl','split_indices.pkl','uncertainty_transformer.py',
                 'uncertainty_utils.py','finetuning.ipynb']:
        (archive/name).write_bytes((repo/name).read_bytes())
    evidence=dict(repository=SOURCE_URL,commit=commit,checkout=str(repo),configuration='withECG',
        configuration_description='54 features including seven ECG and three echocardiographic variables; separate LR+SVC+KNN pipeline',
        exact_feature_order=features,ecg_features=ECG_FEATURES,echocardiographic_features=ECHO_FEATURES,
        transformer_compatibility=compatible,metadata_and_notebook_order_verified=True,class_mapping={'1':'Myocarditis','2':'ACS'},
        complete_case_patients=len(frame),class_counts=frame.GRUP.value_counts().to_dict(),
        saved_training_reference_patients=len(training),saved_reference_statistics_and_scaler_mean_verified=True,
        source_tracked_file_sha256=hashes,
        model_selection='Pinned finetuning.ipynb: 5-fold nested family screening, top-three families by best inner recall, 150-trial retuning on original full training subset; fixed saved specification retained for LOOCV.',
        interpretation='Adds three echocardiographic and seven ECG variables and changes ensemble/hyperparameters and complete-case cohort; not a controlled ECG-only ablation.')
    save_json(out/'source_verification_withECG.json',evidence)
    return model,spec,evidence,hashes
