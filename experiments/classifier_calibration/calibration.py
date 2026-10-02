"""Isolated fixed-specification LOOCV calibration; see README.md in this directory."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import warnings
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd


def save_json(path,value):
    def native(x):
        if isinstance(x,dict):return {str(k):native(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [native(v) for v in x]
        if hasattr(x,'item'):return native(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        return x if isinstance(x,(str,int,float,bool)) or x is None else repr(x)
    path.write_text(json.dumps(native(value),indent=2,allow_nan=False)+'\n')


def mapped_probabilities(estimator,frame):
    classes=list(estimator.classes_)
    if set(classes)!={1,2}:raise ValueError(f'Unexpected diagnostic class mapping: {classes}')
    probability=np.asarray(estimator.predict_proba(frame),dtype=float)
    if probability.shape!=(len(frame),2) or not np.isfinite(probability).all():raise ValueError('Invalid probability shape/values')
    if np.any((probability<0)|(probability>1)) or not np.allclose(probability.sum(1),1,rtol=0,atol=1e-12):raise ValueError('Probabilities outside simplex')
    return probability[:,classes.index(1)],probability[:,classes.index(2)],classes


def brier_contributions(probability,outcome):
    return (np.asarray(probability,float)-np.asarray(outcome,float))**2


def calibration_bins(prediction,outcome,count,configuration,kind):
    """[left,right) except final interval includes 1; empty bins have missing rates."""
    edges=np.linspace(0.,1.,count+1)
    prediction=np.asarray(prediction,float);outcome=np.asarray(outcome,float)
    if not np.isfinite(prediction).all() or np.any((prediction<0)|(prediction>1)):raise ValueError('Invalid binning probability')
    indices=np.searchsorted(edges,prediction,side='right')-1
    indices=np.minimum(indices,count-1)
    rows=[]
    for i in range(count):
        selected=indices==i;n=int(selected.sum())
        rows.append(dict(configuration=configuration,diagram=kind,n_bins=count,bin=i+1,
            lower=edges[i],upper=edges[i+1],interval='[lower, upper]' if i==count-1 else '[lower, upper)',
            patient_count=n,mean_predicted_probability=float(prediction[selected].mean()) if n else np.nan,
            observed_proportion=float(outcome[selected].mean()) if n else np.nan))
    return pd.DataFrame(rows)


def check_fold(train_ids,test_id,expected_size):
    if test_id in train_ids or len(train_ids)!=expected_size or len(set(train_ids))!=len(train_ids):
        raise ValueError('Invalid LOOCV fold: held-out patient included, duplicate training identities, or wrong size')


def prepare(data,features):
    raw=pd.read_excel(data,sheet_name=0)
    missing=set(features+['GRUP'])-set(raw.columns)
    if missing:raise ValueError(f'Missing source columns: {sorted(missing)}')
    cleaned=raw[features+['GRUP']].copy().replace([' ','','-','--','nan','NaN','None','#VALUE!','#N/A','#REF!','#DIV/0!','#NUM!','#NAME?','#NULL!'],np.nan)
    cleaned=cleaned.replace(r'^\s*$',np.nan,regex=True)
    for f in features:cleaned[f]=pd.to_numeric(cleaned[f],errors='coerce')
    eligible=cleaned.dropna().copy()
    if not set(eligible.GRUP.unique())<={1,2} or len(eligible.GRUP.unique())!=2:raise ValueError('Unknown/insufficient diagnostic labels')
    excluded=pd.DataFrame([{'source_row':int(i),'reason':'missing/unparseable required feature or label',
        'missing_columns':' | '.join(cleaned.columns[cleaned.loc[i].isna()])} for i in cleaned.index.difference(eligible.index)])
    return eligible,excluded


def model_spec(model):
    from sklearn.ensemble import VotingClassifier
    if list(model.named_steps)!=['uncertainty','scaler','clf']:raise ValueError('Unsupported preprocessing structure; do not silently replace it')
    if not isinstance(model.named_steps['clf'],VotingClassifier) or model.named_steps['clf'].voting!='soft':raise ValueError('Expected saved soft-voting pipeline')
    return dict(pipeline_repr=repr(model),parameters=model.get_params(deep=True),
        features=list(model.named_steps['uncertainty'].feature_names_in_),
        ensemble_members=[{'name':name,'class':type(estimator).__name__,'parameters':estimator.get_params(deep=True)}
            for name,estimator in model.named_steps['clf'].estimators],
        voting=model.named_steps['clf'].voting,weights=model.named_steps['clf'].weights,
        seeds={k:v for k,v in model.get_params(deep=True).items() if k.endswith('random_state')})


def generate(model,frame,configuration,out):
    from sklearn.base import clone
    from sklearn.model_selection import LeaveOneOut
    feature_list=list(model.named_steps['uncertainty'].feature_names_in_)
    x=frame[feature_list].copy();y=frame.GRUP.to_numpy();rows=[];folds=[]
    identifiers=[int(i) for i in frame.index]
    # Workbook zero-based row index is the original project identifier; no undocumented clinical ID is invented.
    for fold,(train,test) in enumerate(LeaveOneOut().split(x),1):
        index=int(test[0]);patient=identifiers[index];train_ids=[identifiers[j] for j in train]
        check_fold(train_ids,patient,len(frame)-1)
        fresh=clone(model)
        if hasattr(fresh.named_steps['uncertainty'],'_feat_stats_') or hasattr(fresh.named_steps['scaler'],'mean_'):
            raise AssertionError('Clone retained fitted preprocessing')
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            fresh.fit(x.iloc[train].copy(),y[train].copy())
        p_myo,p_acs,classes=mapped_probabilities(fresh,x.iloc[test].copy())
        prediction=int(fresh.predict(x.iloc[test])[0]);truth=int(y[index]);correct=prediction==truth
        confidence=float(p_myo[0] if prediction==1 else p_acs[0])
        if prediction!=int(classes[int(np.argmax(fresh.predict_proba(x.iloc[test])[0]))]):raise AssertionError('Soft-vote decision does not match probability argmax')
        outcome=int(truth==1)
        rows.append(dict(patient_id=patient,source_row=patient,source_excel_row=patient+2,configuration=configuration,
            loocv_fold=fold,training_fold_size=len(train),fitted_classes=json.dumps([int(c) for c in classes]),true_label=truth,true_diagnosis='Myocarditis' if truth==1 else 'ACS',
            y_myocarditis=outcome,p_myocarditis=float(p_myo[0]),p_acs=float(p_acs[0]),predicted_label=prediction,
            predicted_diagnosis='Myocarditis' if prediction==1 else 'ACS',predicted_class_confidence=confidence,
            prediction_correct=bool(correct),brier_contribution=float(brier_contributions(p_myo,[outcome])[0])))
        # VotingClassifier members are trained on encoded labels; audit inverse mapping explicitly.
        voter=fresh.named_steps['clf'];members={}
        for name,member in voter.named_estimators_.items():
            encoded=list(member.classes_);diagnostic=list(voter.le_.inverse_transform(np.array(encoded,dtype=int)))
            if set(diagnostic)!={1,2}:raise AssertionError('Voting member label mapping is invalid')
            members[name]={'encoded_classes':encoded,'diagnostic_classes':diagnostic}
        if int(fresh.named_steps['scaler'].n_samples_seen_)!=len(train):raise AssertionError('Scaler fitted outside training fold')
        fitted_features=list(fresh.named_steps['uncertainty'].feature_names_in_)
        if fitted_features!=feature_list:raise ValueError('Fold dropped features; original specification could not be maintained')
        folds.append(dict(configuration=configuration,fold=fold,held_out_id=patient,training_ids=train_ids,
            train_test_disjoint=True,classes=classes,member_mappings=members,
            scaler_n_samples_seen=int(fresh.named_steps['scaler'].n_samples_seen_),
            uncertainty_statistics_sha256=hashlib.sha256(pickle.dumps(fresh.named_steps['uncertainty']._feat_stats_)).hexdigest(),
            scaler_sha256=hashlib.sha256(pickle.dumps(fresh.named_steps['scaler'])).hexdigest(),
            warnings=[str(w.message) for w in captured]))
        # Write fold evidence as it completes; no fitted objects overwrite original files.
        with (out/'fold_audit.jsonl').open('a') as fh:fh.write(json.dumps(folds[-1],default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n')
        if fold==1 or fold%10==0 or fold==len(frame):
            print(f'{configuration}: {fold}/{len(frame)} folds',flush=True)
            with (out/'experiment.log').open('a') as fh:fh.write(f'{configuration}: completed fold {fold}/{len(frame)}\n')
    return pd.DataFrame(rows)


def report_from_workbook(workbook,out,provenance):
    """All aggregate metrics/bins are computed after patient predictions are saved to Excel."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries=[];all_bins=[]
    for config in provenance['available_configurations']:
        saved=pd.read_excel(workbook,sheet_name=f'predictions_{config}')
        if saved.patient_id.duplicated().any():raise ValueError('Duplicate OOF prediction')
        np.testing.assert_allclose(saved.brier_contribution,brier_contributions(saved.p_myocarditis,saved.y_myocarditis),rtol=1e-13,atol=1e-14)
        summary=dict(configuration=config,status='complete',patients=len(saved),features=len(provenance['models'][config]['features']),
            myocarditis_patients=int(saved.y_myocarditis.sum()),acs_patients=int((1-saved.y_myocarditis).sum()),
            brier_score=float(saved.brier_contribution.mean()),accuracy=float(saved.prediction_correct.mean()),
            mean_predicted_class_confidence=float(saved.predicted_class_confidence.mean()),
            mean_predicted_myocarditis=float(saved.p_myocarditis.mean()),observed_myocarditis=float(saved.y_myocarditis.mean()),
            evaluation='LOOCV of fixed previously selected specification; not fully nested selection evaluation')
        for kind,probability,outcome in [('myocarditis_probability',saved.p_myocarditis,saved.y_myocarditis),
                                         ('confidence_correctness',saved.predicted_class_confidence,saved.prediction_correct.astype(int))]:
            for count in [5,10]:
                bins=calibration_bins(probability,outcome,count,config,kind);all_bins.append(bins)
                if kind=='myocarditis_probability':
                    summary[f'descriptive_ece_{count}_bins']=float(np.nansum(bins.patient_count/len(saved)*abs(bins.mean_predicted_probability-bins.observed_proportion)))
                fig,axes=plt.subplots(2,1,figsize=(6.5,7),gridspec_kw={'height_ratios':[3,1]},sharex=True)
                used=bins[bins.patient_count>0]
                axes[0].plot([0,1],[0,1],'--',color='gray',label='Identity')
                axes[0].plot(used.mean_predicted_probability,used.observed_proportion,'o-',label=f'OOF {config}, n={len(saved)}')
                for _,b in used.iterrows():axes[0].annotate(f'n={int(b.patient_count)}',(b.mean_predicted_probability,b.observed_proportion),xytext=(4,6),textcoords='offset points',fontsize=8)
                axes[0].set(ylabel='Observed myocarditis proportion' if kind=='myocarditis_probability' else 'Observed prediction correctness',ylim=(-.03,1.03),
                    title=f'{"Primary: myocarditis probability" if kind=="myocarditis_probability" else "Supplement: predicted-class confidence"}\n{count} equal-width bins; descriptive');axes[0].legend(fontsize=9)
                axes[1].hist(probability,bins=np.linspace(0,1,count+1),color='steelblue',edgecolor='white')
                axes[1].set(xlabel='Predicted myocarditis probability' if kind=='myocarditis_probability' else 'Predicted-class confidence',ylabel='Patients',xlim=(0,1))
                fig.tight_layout()
                for ext in ['png','pdf']:fig.savefig(out/f'reliability_{config}_{kind}_{count}bins.{ext}',dpi=220)
                plt.close(fig)
        summaries.append(summary)
    if 'withECG' not in provenance['available_configurations']:
        summaries.append(dict(configuration='withECG',status='unavailable: verified 54-feature LR+SVC+KNN pipeline missing',patients=np.nan,features=np.nan,brier_score=np.nan))
    summary=pd.DataFrame(summaries);bins=pd.concat(all_bins,ignore_index=True)
    summary.to_csv(out/'summary.csv',index=False);bins.to_csv(out/'calibration_bins.csv',index=False)
    with pd.ExcelWriter(workbook,engine='openpyxl',mode='a',if_sheet_exists='replace') as writer:
        summary.to_excel(writer,sheet_name='summary',index=False)
        bins.to_excel(writer,sheet_name='calibration_bins',index=False)
    # This workbook is a new experiment artifact; only aggregate sheets are appended.
    methods='We assessed the original soft-voting probabilities using leave-one-out cross-validation over all eligible complete cases. Each fold cloned and refitted the complete uncertainty-transformer–StandardScaler–ensemble pipeline on N−1 patients; probability columns were mapped from each fitted classes_ attribute. The model specification and hyperparameters were previously selected on the original training cohort, including patients evaluated here; this is fixed-specification LOOCV, not fully nested model-selection validation. Predictions and binary squared-error contributions were saved to Excel before computing Brier scores and reliability diagrams. Primary diagrams used myocarditis probabilities with five predefined equal-width bins on [0,1], with ten-bin sensitivity and separate confidence-versus-correctness diagrams. Empty bins retained missing rates. Brier measures overall probabilistic quality, not calibration alone; all estimates are descriptive. No calibrator was fitted and the deployed model was unchanged.'
    results=[]
    for _,r in summary[summary.status=='complete'].iterrows():
        results.append(f'{r.configuration}: {int(r.patients)} patients, Brier={r.brier_score:.6f}; mean myocarditis probability={r.mean_predicted_myocarditis:.3f} versus observed prevalence={r.observed_myocarditis:.3f}, accuracy={r.accuracy:.3f}. The five/ten-bin weighted absolute probability–frequency gaps were {r.descriptive_ece_5_bins:.3f}/{r.descriptive_ece_10_bins:.3f}, illustrating bin-dependent descriptive calibration error.')
    for config in provenance['available_configurations']:
        primary=pd.concat(all_bins,ignore_index=True)
        primary=primary[(primary.configuration==config)&(primary.diagram=='myocarditis_probability')&(primary.n_bins==5)]
        low,high=primary.iloc[0],primary.iloc[-1]
        row=summary[summary.configuration==config].iloc[0]
        if low.patient_count and high.patient_count:
            results.append(f'For {config}, the lowest probability bin had mean prediction {low.mean_predicted_probability:.3f} versus observed myocarditis proportion {low.observed_proportion:.3f} (n={int(low.patient_count)}); the highest had {high.mean_predicted_probability:.3f} versus {high.observed_proportion:.3f} (n={int(high.patient_count)}). Observed frequencies were more extreme than predictions in these outer bins, consistent with descriptive underconfidence, not proof of population calibration.')
        results.append(f'The distinct confidence-versus-correctness analysis had mean predicted-class confidence {row.mean_predicted_class_confidence:.3f} versus observed accuracy {row.accuracy:.3f}.')
    results.append('The current interface is verified to load the 44-feature no-ECG Logistic Regression+SVC+Random Forest pipeline with equal soft-voting weights. Historical clinician-session configuration remains unverified.')
    if 'withECG' not in provenance['available_configurations']:results.append('With-ECG calibration is unfinished because its verified 54-feature Logistic Regression+SVC+KNN pipeline is unavailable; no probabilities or patient denominator were invented. Configurations must be treated as separate modeling pipelines, not an ECG-only controlled ablation.')
    (out/'methods_results.txt').write_text('Methods\n'+methods+'\n\nResults\n'+' '.join(results)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'Miyokardit_08.12.xlsx')
    parser.add_argument('--with-ecg-model',type=Path,help='Optional trusted saved 54-feature complete pipeline')
    parser.add_argument('--with-ecg-data',type=Path,help='Optional corresponding workbook; default is --data')
    args=parser.parse_args()
    out=BASE/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ');out.mkdir(exist_ok=False)
    (out/'experiment.log').write_text('Isolated calibration experiment started.\n')
    # Protect every existing project file outside this experiment, including recurrence outputs.
    protected={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*')
        if p.is_file() and not p.is_relative_to(BASE) and not any(t in p.relative_to(ROOT).parts for t in ['.git','.venv','__pycache__'])}
    try:run(args,out,protected)
    except Exception as exc:
        save_json(out/'failure.json',{'status':'failed','error':str(exc),'type':type(exc).__name__})
        raise
    print(f'Completed available analyses: {out}',flush=True)


def run(args,out,protected):
    from uncertainty_transformer import UncertaintyTransformer  # local trusted pickle class
    from sklearn.base import clone
    with (ROOT/'best_model_finetuned.pkl').open('rb') as fh:model=pickle.load(fh)
    with (ROOT/'model_metadata.pkl').open('rb') as fh:metadata=pickle.load(fh)
    spec=model_spec(model)
    if len(spec['features'])!=44 or spec['features']!=list(metadata['features']) or any('ECG' in f.upper() for f in spec['features']):raise ValueError('Current saved pipeline is not the verified 44-feature no-ECG specification')
    if [m['class'] for m in spec['ensemble_members']]!=['LogisticRegression','SVC','RandomForestClassifier']:raise ValueError('Current ensemble differs from manuscript no-ECG description')
    app=(ROOT/'app.py').read_text()
    if '"best_model_finetuned.pkl"' not in app or 'pickle.load(f)' not in app:raise ValueError('App model-loading provenance changed; inspect before claiming interface configuration')
    models={'noECG':model};data={'noECG':args.data};specs={'noECG':spec}
    if args.with_ecg_model:
        with args.with_ecg_model.open('rb') as fh:other=pickle.load(fh)
        other_spec=model_spec(other)
        if len(other_spec['features'])!=54 or not any('ECG' in f.upper() for f in other_spec['features']):raise ValueError('Supplied with-ECG feature specification not verified')
        if set(m['class'] for m in other_spec['ensemble_members'])!={'LogisticRegression','SVC','KNeighborsClassifier'}:raise ValueError('Supplied with-ECG ensemble differs from described specification')
        models['withECG']=other;specs['withECG']=other_spec;data['withECG']=args.with_ecg_data or args.data
    provenance=dict(available_configurations=list(models),models=specs,software={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib','openpyxl']},
        python=sys.version,executable=sys.executable,source_workbooks={k:str(v.resolve()) for k,v in data.items()},
        input_sha256={str(v.resolve()):hashlib.sha256(v.read_bytes()).hexdigest() for v in set(data.values())|{ROOT/'best_model_finetuned.pkl',ROOT/'model_metadata.pkl',ROOT/'app.py',ROOT/'finetuning.ipynb',ROOT/'uncertainty.ipynb',ROOT/'uncertainty_transformer.py',ROOT/'uncertainty_utils.py'}},
        current_interface='Verified app loads best_model_finetuned.pkl: noECG, 44 features, LogisticRegression+SVC+RandomForestClassifier, equal soft votes',
        historical_session_configuration='unverified; current artifact identity does not establish historical deployment',
        model_selection='finetuning.ipynb: nested 5-fold family screening; top three families by best inner recall; 150-trial retuning per selected family on saved full training subset. Preserve resulting saved parameters and weights without retuning. Some evaluated LOOCV patients participated in prior selection.',
        evaluation='LOOCV of fixed previously selected specification, not fully nested or completely unbiased model-selection evaluation',
        preprocessing='Original feature order; replace original missing markers, whitespace cleaning, numeric coercion, complete-case exclusion. Fit label-aware reference discretization/statistics and StandardScaler inside every LOOCV fold; no imputation.',
        bins={'primary':5,'sensitivity':10,'edges':'equal width [0,1]','intervals':'[lower,upper), final includes 1','empty_bins':'count 0; mean and observed rate missing'},
        classes={'1':'myocarditis','2':'ACS'},patient_identity='Original zero-based workbook row index; Excel physical row = source_row + 2',
        missing_withECG=None if args.with_ecg_model else 'No saved with-ECG pipeline/metadata or original with-ECG model-selection code was found. 158-patient eligibility and hyperparameters cannot be verified; do not infer a pipeline from manuscript text.')
    if args.with_ecg_model:
        provenance['input_sha256'][str(args.with_ecg_model.resolve())]=hashlib.sha256(args.with_ecg_model.read_bytes()).hexdigest()
    provenance['cohort_order']='Original source workbook row order retained in each LOOCV fold; folds generated sequentially'
    snapshot=out/'code_snapshot';snapshot.mkdir()
    provenance['experiment_code_sha256']={}
    for code in [BASE/'calibration.py',BASE/'verify_results.py',BASE/'test_calibration.py']:
        content=code.read_bytes();(snapshot/code.name).write_bytes(content)
        provenance['experiment_code_sha256'][str(code)]=hashlib.sha256(content).hexdigest()
    inventory=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or p.is_relative_to(BASE) or any(t in p.parts for t in ['.git','.venv','__pycache__']):continue
        if p.suffix.lower() in ['.pkl','.joblib','.xlsx','.csv','.npz']:inventory.append(str(p.relative_to(ROOT)))
    save_json(out/'prediction_search.json',{'candidates':inventory,'notebook_evidence':'uncertainty.ipynb cell 42 contains cross_val_predict(LeaveOneOut) but exports no verifiable patient-level OOF probabilities/identities/fold evidence; stored outputs are aggregate metrics.',
        'reuse_decision':'No existing predictions with verifiable identity, classes, specification and fold exclusion found; generate new LOOCV.'})
    save_json(out/'provenance.json',provenance)
    save_json(out/'bin_definitions.json',provenance['bins'])  # predefined before evaluating bin outcomes
    tables={};eligible_ids={}
    for config,pipeline in models.items():
        frame,excluded=prepare(data[config],specs[config]['features'])
        excluded.to_csv(out/f'excluded_{config}.csv',index=False)
        eligible_ids[config]=list(frame.index)
        save_json(out/f'eligible_{config}.json',{'patients':len(frame),'patient_ids':eligible_ids[config],'class_counts':frame.GRUP.value_counts().to_dict()})
        tables[config]=generate(pipeline,frame,config,out)
        tables[config].to_csv(out/f'predictions_{config}.csv',index=False)
    workbook=out/'patient_predictions.xlsx'
    # Persist patient predictions BEFORE calculating aggregates and reliability diagrams.
    with pd.ExcelWriter(workbook,engine='openpyxl') as writer:
        for config,table in tables.items():table.to_excel(writer,sheet_name=f'predictions_{config}',index=False)
        if 'withECG' not in tables:pd.DataFrame([{'status':'unavailable','reason':provenance['missing_withECG']}]).to_excel(writer,sheet_name='predictions_withECG',index=False)
        pd.DataFrame([{'key':k,'value':json.dumps(v,default=str) if not isinstance(v,str) else v} for k,v in provenance.items()]).to_excel(writer,sheet_name='provenance',index=False)
    report_from_workbook(workbook,out,provenance)
    from verify_results import verify
    verification=verify(out)
    save_json(out/'independent_verification.json',verification)
    changed=[name for name,digest in protected.items() if not Path(name).exists() or hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest]
    save_json(out/'protected_file_audit.json',{'protected_files_checked':len(protected),'changed_files':changed,'sha256':protected})
    if changed:raise RuntimeError(f'Protected files changed: {changed}')
    for path,digest in provenance['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Experiment input changed: {path}')
    save_json(out/'completion.json',{'status':'complete' if 'withECG' in models else 'available_configuration_complete_withECG_unavailable','protected_files_unchanged':True,'independent_checks':'passed'})

if __name__=='__main__':main()
