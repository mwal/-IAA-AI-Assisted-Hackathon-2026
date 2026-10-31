"""Recompute selected submissions from raw competition data and saved models."""
import argparse,sys,json,hashlib,gc,os,time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor
from prepare import prepare_developed,task1_features,assemble_emerging,developed_dir,LINES
from feature_design import design
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_cb(path):
    m=CatBoostRegressor();m.load_model(str(path));return m
def rate(model,X):return np.maximum(model.predict(X[model.feature_names_],prediction_type='Exponent',thread_count=4),1e-12)
def check_inputs(raw,artifacts,verify_artifacts=True):
    expected=json.loads((ROOT/'specifications/raw_input_hashes.json').read_text())
    for name,h in expected.items():
        market,file=name.split('/');p=(developed_dir(raw) if market=='developed' else raw/'emerging')/file
        assert sha(p)==h,'Raw input mismatch: '+name
    if verify_artifacts:
        expected=json.loads((ROOT/'specifications/artifact_hashes.json').read_text())
        for name,h in expected.items():assert sha(artifacts/name)==h,'Artifact mismatch: '+name
def task1(raw,artifacts,work,output,spec):
    prepared=prepare_developed(raw,work);frames=[]
    for line in LINES:
        d,X,original=task1_features(line,prepared,raw);test=d.year.eq(2025);e=d.loc[test,'exposure'].to_numpy()
        models=artifacts/'task1';f=load_cb(models/f'{line}_frequency.cbm');s=load_cb(models/f'{line}_severity.cbm')
        p=e*rate(f,X.loc[test])*1000*rate(s,X.loc[test])
        if line=='buildings':
            newer=e*np.maximum(rate(load_cb(models/'buildings_round2.cbm'),X.loc[test]),1e-10)
            baseline=e*np.maximum(rate(load_cb(models/'buildings_baseline.cbm'),original.loc[test]),1e-10)
            fm,sm,cols=joblib.load(models/'buildings_legacy_glm.joblib')
            glm=np.maximum(e*fm.predict(original.loc[test,cols])*sm.predict(original.loc[test,cols]),1e-10)
            a=spec['legacy']['frequency_severity_weight'];legacy=(1-a)*baseline+a*glm
            b=spec['round2']['new_model_weight'];r2=(1-b)*legacy+b*newer
            c=spec['round4'][line]['frequency_severity_weight'];p=(1-c)*r2+c*p
        frames.append(pd.DataFrame({'row_id':d.loc[test,'row_id'].astype(str),'prediction':p}));print('Predicted Task1',line,len(p),flush=True)
        del d,X,original,f,s;gc.collect()
    return export(pd.concat(frames),pd.read_parquet(developed_dir(raw)/'sample_submission.parquet'),output/'task1.csv')
def task2(raw,artifacts,work,output,spec):
    prepared=assemble_emerging(raw,work,artifacts);frames=[];models=artifacts/'task2'
    for line in LINES:
        d=pd.read_parquet(prepared/f'{line}.parquet');d=d[d.year.eq(2025)]
        direct=next(x for x in spec['direct']['models'] if x['line']==line);report=next(x for x in spec['reporting']['models'] if x['line']==line);gender=next(x for x in spec['gender'] if x['line']==line)
        prep=joblib.load(models/f'direct_{line}_preprocess.joblib');source=load_cb(models/f'direct_{line}_developed_direct_cost.cbm')
        D=source.predict(prep.transform(design(d,spec['source_features'][line])),prediction_type='Exponent',thread_count=4)*20000
        em=joblib.load(models/f'frequency_{line}_emerging_frequency.joblib');severity=joblib.load(models/f'frequency_{line}_emerging_severity.joblib')
        X=design(d,spec['emerging_features'][line]);ef=em.predict(X);s=severity.predict(X)
        prep=joblib.load(models/f'frequency_{line}_preprocess.joblib');source=load_cb(models/f'frequency_{line}_developed_frequency.cbm')
        fd=source.predict(prep.transform(design(d,spec['frequency_source_features'][line])),prediction_type='Exponent',thread_count=4)
        z=report['Z_emerging'];frequency=z*ef+(1-z)*report['reporting_calibration_k']*fd
        e=d.exposure.to_numpy();reference=e*frequency*s;annual=reference/e
        g=d.gender.astype(str).str.upper().map(gender['factors']).to_numpy();assert np.isfinite(g).all()
        w=direct['developed_cost_weight'];k=direct['final_source_calibration_k']
        p=e*((1-w)*annual+w*k*D*g)
        frames.append(pd.DataFrame({'row_id':d.row_id,'prediction':p}));print('Predicted Task2',line,len(p),flush=True)
        del d,X;gc.collect()
    return export(pd.concat(frames),pd.read_parquet(raw/'emerging/sample_submission.parquet'),output/'task2.csv')
def export(frame,sample,path):
    assert frame.row_id.is_unique and set(frame.row_id)==set(sample.row_id)
    out=frame.set_index('row_id').loc[sample.row_id].reset_index();assert np.isfinite(out.prediction).all() and out.prediction.ge(0).all()
    out.to_csv(path,index=False);return out
def compare(out,reference):
    assert out.row_id.tolist()==reference.row_id.tolist();delta=np.abs(out.prediction-reference.prediction)
    assert np.allclose(out.prediction,reference.prediction,rtol=1e-10,atol=1e-7),'Predictions differ: max '+str(delta.max())
    return {'rows':len(out),'all_rows_match':True,'max_absolute_difference':float(delta.max()),'relative_tolerance':1e-10,'absolute_tolerance':1e-7}
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True,type=Path);p.add_argument('--artifacts-dir',type=Path,default=ROOT/'artifacts');p.add_argument('--work-dir',type=Path,default=ROOT/'work');p.add_argument('--output-dir',type=Path,default=ROOT/'results');p.add_argument('--task',choices=['all','1','2'],default='all');p.add_argument('--forbid-path',action='append',default=[]);p.add_argument('--skip-input-hashes',action='store_true',help='Only for models newly trained by train_selected.py; raw data must remain unchanged.')
    args=p.parse_args();args.data_dir=args.data_dir.resolve();args.artifacts_dir=args.artifacts_dir.resolve();args.work_dir=args.work_dir.resolve();args.output_dir=args.output_dir.resolve()
    # A clean-room run can explicitly forbid reads from original workspaces.
    blocked=[str(Path(x).resolve()) for x in args.forbid_path]
    def audit(event,values):
        if event=='open' and values and isinstance(values[0],(str,bytes,os.PathLike)):
            path=str(Path(os.fsdecode(values[0])).absolute())
            if any(path==b or path.startswith(b+os.sep) for b in blocked) and '/work/venv/' not in path:raise PermissionError('Original workspace read forbidden: '+path)
    sys.addaudithook(audit);start=time.time();args.work_dir.mkdir(parents=True,exist_ok=True);args.output_dir.mkdir(parents=True,exist_ok=True)
    check_inputs(args.data_dir,args.artifacts_dir,verify_artifacts=not args.skip_input_hashes)
    parameters=args.artifacts_dir/'selected_models.json'
    spec=json.loads((parameters if parameters.exists() else ROOT/'specifications/selected_models.json').read_text());results={}
    for task in ([1,2] if args.task=='all' else [int(args.task)]):
        out=(task1 if task==1 else task2)(args.data_dir,args.artifacts_dir,args.work_dir,args.output_dir,spec[f'task{task}'])
        path=args.artifacts_dir/'reference'/('task1.parquet' if task==1 else 'task2.csv');ref=pd.read_parquet(path) if task==1 else pd.read_csv(path)
        results[f'task{task}']=compare(out,ref);results[f'task{task}']['csv_sha256']=sha(args.output_dir/f'task{task}.csv')
    results.update({'elapsed_seconds':time.time()-start,'raw_data_and_models_recomputed_predictions':True,'references_used_only_after_prediction_for_comparison':True,'forbidden_original_workspaces':blocked})
    (args.output_dir/'verification.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__':main()
