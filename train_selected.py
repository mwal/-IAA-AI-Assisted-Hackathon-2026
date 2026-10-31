"""Refit fixed selected models; hyperparameter searches are deliberately not rerun."""
from pathlib import Path
import argparse,sys,json,shutil,gc,os,time
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from catboost import CatBoostRegressor,Pool
from prepare import prepare_developed,task1_features,assemble_emerging,source_data,LINES
from feature_design import design
from delay_support import observable_exposure,fit_delay
from reproduce import load_cb,check_inputs

def fit_cb(template,X,label,weight,mask,path):
    pars=template.get_params().copy();pars.pop('eval_metric',None);pars['iterations']=template.tree_count_
    m=CatBoostRegressor(**pars);features=template.feature_names_;cats=[features[i] for i in template.get_cat_feature_indices()]
    m.fit(Pool(X.loc[mask,features],label=np.asarray(label)[mask],weight=np.asarray(weight)[mask],cat_features=cats));m.save_model(str(path));return m
def fit_pipe(template,X,label,weight,mask,path):
    m=clone(template);m.fit(X.loc[mask],np.asarray(label)[mask],model__sample_weight=np.asarray(weight)[mask]);joblib.dump(m,path);return m
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True,type=Path);p.add_argument('--artifacts-dir',type=Path,default=ROOT/'artifacts');p.add_argument('--work-dir',type=Path,default=ROOT/'work');p.add_argument('--trained-dir',type=Path,default=ROOT/'work/retrained_artifacts');p.add_argument('--task',choices=['all','1','2'],default='all');args=p.parse_args();start=time.time()
    raw=args.data_dir.resolve();original=args.artifacts_dir.resolve();dest=args.trained_dir.resolve();work=args.work_dir.resolve();assert dest!=original
    check_inputs(raw,original);shutil.copytree(original,dest,dirs_exist_ok=True)
    spec=json.loads((ROOT/'specifications/selected_models.json').read_text());training=[]
    if args.task in ['all','1']:
        prepared=prepare_developed(raw,work)
        for line in LINES:
            d,X,X0=task1_features(line,prepared,raw);mask=d.year.le(2023).to_numpy();e=d.exposure.to_numpy();n=d.positive_reported_claim_count.to_numpy();y=d.target_net_incurred.to_numpy()
            for comp in ['frequency','severity']:
                file=f'{line}_{comp}.cbm';model=load_cb(original/'task1'/file)
                with np.errstate(divide='ignore',invalid='ignore'):target=n/e if comp=='frequency' else y/n/1000
                fit_cb(model,X,target,e if comp=='frequency' else n,mask if comp=='frequency' else mask&(n>0),dest/'task1'/file)
                print('Refit Task1',line,comp,flush=True)
            if line=='buildings':
                for file,matrix in [('buildings_baseline.cbm',X0),('buildings_round2.cbm',X)]:fit_cb(load_cb(original/'task1'/file),matrix,y/e,np.sqrt(e),mask,dest/'task1'/file)
                fm,sm,cols=joblib.load(original/'task1/buildings_legacy_glm.joblib')
                with np.errstate(divide='ignore',invalid='ignore'):target=y/n
                fm=fit_pipe(fm,X0[cols],n/e,e,mask,dest/'task1/temporary_frequency.joblib');sm=fit_pipe(sm,X0[cols],target,n,mask&(n>0)&(y>0),dest/'task1/temporary_severity.joblib')
                joblib.dump((fm,sm,cols),dest/'task1/buildings_legacy_glm.joblib')
                for pth in (dest/'task1').glob('temporary_*'):pth.unlink()
            training.append({'task':1,'line':line,'train_rows':int(mask.sum())});del d,X,X0;gc.collect()
    if args.task in ['all','2']:
        prepared=prepare_developed(raw,work);emerging=assemble_emerging(raw,work,original);s=spec['task2'];new_direct=[];new_reporting=[];new_gender=[]
        for line in LINES:
            d=source_data(line,prepared,raw,original);mask=np.ones(len(d),dtype=bool);e=d.exposure.to_numpy();y=d.target_net_incurred.to_numpy();n=d.positive_reported_claim_count.to_numpy()
            for kind,features,objective in [('direct',s['source_features'][line],'cost'),('frequency',s['frequency_source_features'][line],'count')]:
                prepfile=f'{kind}_{line}_preprocess.joblib';prep=clone(joblib.load(original/'task2'/prepfile));X=prep.fit_transform(design(d,features));joblib.dump(prep,dest/'task2'/prepfile)
                file=f'{kind}_{line}_developed_'+('direct_cost' if objective=='cost' else 'frequency')+'.cbm';template=load_cb(original/'task2'/file)
                model=CatBoostRegressor(**template.get_params());model.fit(Pool(X,label=y/e/20000 if objective=='cost' else n/e,weight=np.sqrt(e) if objective=='cost' else e));model.save_model(str(dest/'task2'/file));print('Refit Task2 source',line,kind,flush=True)
            del d,X;gc.collect()
            d=pd.read_parquet(emerging/f'{line}.parquet');d=d[d.year.lt(2025)];e=d.exposure.to_numpy();y=d.target_net_incurred.to_numpy();n=d.positive_claim_count.to_numpy();f=s['emerging_features'][line];X=design(d,f)
            report=next(x.copy() for x in s['reporting']['models'] if x['line']==line)
            claims=pd.read_parquet(emerging/'claims.parquet');claims=claims[claims.row_id.notna()&claims.line.eq(line)&claims.status.ne('repudiated')&claims.net.gt(0)&claims.loss_date.dt.year.eq(2023)]
            curve=fit_delay(claims,pd.Timestamp('2024-12-31'));report['delay_curve']=curve;eo=observable_exposure(d,pd.Timestamp('2024-12-31'),curve)
            mask=np.ones(len(d),dtype=bool);freqfile=f'frequency_{line}_emerging_frequency.joblib';sevfile=f'frequency_{line}_emerging_severity.joblib'
            fit_pipe(joblib.load(original/'task2'/freqfile),X,n/eo,eo,mask,dest/'task2'/freqfile)
            with np.errstate(divide='ignore',invalid='ignore'):target=y/n
            fit_pipe(joblib.load(original/'task2'/sevfile),X,target,n,mask&(n>0),dest/'task2'/sevfile)
            prep=joblib.load(dest/'task2'/f'frequency_{line}_preprocess.joblib');m=load_cb(dest/'task2'/f'frequency_{line}_developed_frequency.cbm');fd=m.predict(prep.transform(design(d,s['frequency_source_features'][line])),prediction_type='Exponent',thread_count=4);report['reporting_calibration_k']=float(n.sum()/np.dot(eo,fd));new_reporting.append(report)
            prep=joblib.load(dest/'task2'/f'direct_{line}_preprocess.joblib');m=load_cb(dest/'task2'/f'direct_{line}_developed_direct_cost.cbm');D=m.predict(prep.transform(design(d,s['source_features'][line])),prediction_type='Exponent',thread_count=4)*20000
            direct=next(x.copy() for x in s['direct']['models'] if x['line']==line);k=float(y.sum()/np.dot(eo,D));direct['final_source_calibration_k']=k;new_direct.append(direct)
            gender=next(x.copy() for x in s['gender'] if x['line']==line);h=gender['selected_h'];E=eo*k*D;g=d.gender.astype(str).str.upper().to_numpy();Y=np.array([y[g==sex].sum() for sex in ['F','M']]);expected=np.array([E[g==sex].sum() for sex in ['F','M']]);mu=y.sum()/n.sum();factor=np.ones(2) if h=='global' else (Y+float(h)*mu)/(expected+float(h)*mu);factor*=y.sum()/np.dot(expected,factor);gender['factors']=dict(zip(['F','M'],map(float,factor)));new_gender.append(gender)
            training.append({'task':2,'line':line,'emerging_rows':len(d),'reporting_curve':'refitted older2023 right-truncated curve','source_direct_calibration_k':k});print('Refit Task2 emerging',line,flush=True);del d,X;gc.collect()
        spec['task2']['direct']['models']=new_direct;spec['task2']['reporting']['models']=new_reporting;spec['task2']['gender']=new_gender
    # Prediction accepts explicit trained parameters; frozen original specifications are untouched.
    (dest/'selected_models.json').write_text(json.dumps(spec,indent=2)+'\n');(dest/'training_verification.json').write_text(json.dumps({'status':'selected configurations refitted; compare predictions separately','elapsed_seconds':time.time()-start,'records':training,'searches_rerun':False,'reviewed_repairs':'frozen decision artifacts; reporting curves refitted'},indent=2)+'\n')
    print('Refitted models at',dest,flush=True)
if __name__=='__main__':main()
