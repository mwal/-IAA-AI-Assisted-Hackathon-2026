"""Raw-data assembly for the exact selected models; no original workspace reads."""
from pathlib import Path
import contextlib,json,gc
import numpy as np
import pandas as pd
import assemble_data,repair_property,train_baselines
from history import enrich_history
from shared_features import enrich

LINES=('motor','buildings','contents')
def developed_dir(root):return root/'developed' if (root/'developed').is_dir() else root
def prepare_developed(raw,work):
    out=work/'developed'
    if (out/'complete.json').exists():return out
    out.mkdir(parents=True,exist_ok=True)
    with (out/'assembly.log').open('w') as log,contextlib.redirect_stdout(log):
        assemble_data.assemble(developed_dir(raw),out)
        repair_property.ROOT=out;repair_property.main()
    (out/'complete.json').write_text(json.dumps({'stage':'original conservative assembly and repair','source':str(raw)}))
    return out
def task1_features(line,prepared,raw):
    d=pd.read_parquet(prepared/f'assembled_{line}.parquet')
    events,first=train_baselines.history_events(prepared)
    X,cats=train_baselines.make_features(d,events,first)
    original=X.copy()
    for c in cats:X[c]=X[c].str.strip().str.lower()
    links=pd.read_parquet(prepared/'claim_assignments.parquet',columns=['row_id','line','loss_date','report_date'])
    enrich_history(d,X,links)
    fields=['prior_claims_5y','licence_points','years_at_address']
    customers=pd.read_parquet(developed_dir(raw)/'customers.parquet',columns=['customer_id']+fields).set_index('customer_id')
    for c in fields:X[c]=pd.to_numeric(d.customer_id.astype(str).map(customers[c]),errors='coerce').to_numpy()
    return d,X,original
def apply_repairs(d,line,market,artifacts,check_raw=True):
    fixes=json.loads((artifacts/'frozen_repairs.json').read_text())[market+'_'+line]
    lookup=pd.DataFrame.from_dict(fixes,orient='index')
    mask=d.row_id.isin(lookup.index)
    if mask.any():
        patch=lookup.loc[d.loc[mask,'row_id']]
        if check_raw:assert np.allclose(d.loc[mask,'sum_insured'],patch.raw)
        d.loc[mask,'sum_insured']=patch.revised.to_numpy()
    return d
def assemble_emerging(raw,work,artifacts):
    out=work/'emerging';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return out
    root=raw/'emerging';customers=pd.read_parquet(root/'customers.parquet');addresses=pd.read_parquet(root/'addresses.parquet');claims=pd.read_parquet(root/'claims.parquet')
    for c in ['loss_date','report_date']:claims[c]=pd.to_datetime(claims[c])
    claims['net']=claims.incurred_to_date-claims.recoveries
    assert customers.customer_id.is_unique and addresses.address_id.is_unique and claims.net.ge(0).all()
    linked=[]
    for line in LINES:
        d=pd.concat([pd.read_parquet(root/f'{pre}_{line}.parquet').assign(is_test=pre=='test') for pre in ['policies','test']],ignore_index=True)
        d['line']=line
        for c in ['period_start','period_end']:d[c]=pd.to_datetime(d[c])
        d['source_exposure']=d.exposure;d['exposure']=(d.period_end-d.period_start).dt.days.add(1)/365.;d['year']=d.period_start.dt.year
        ordered=d.sort_values(['policy_id','period_start']);previous=ordered.groupby('policy_id').period_end.shift()
        assert not ordered.period_start.le(previous).any() and d.row_id.is_unique
        h=d[~d.is_test][['row_id','policy_id','period_start','period_end']];c=claims[claims.line.eq(line)].copy()
        matched=pd.merge_asof(c.sort_values('loss_date'),h.sort_values('period_start'),left_on='loss_date',right_on='period_start',by='policy_id',direction='backward')
        ok=matched.row_id.notna()&matched.loss_date.le(matched.period_end)
        first=h.groupby('policy_id').period_start.min();assert matched.loc[~ok,'loss_date'].lt(matched.loc[~ok,'policy_id'].map(first)).all()
        matched.loc[~ok,'row_id']=None;linked.append(matched)
        eligible=matched[ok&matched.status.ne('repudiated')]
        d['target_net_incurred']=d.row_id.map(eligible.groupby('row_id').net.sum()).fillna(0).where(~d.is_test)
        d['positive_claim_count']=d.row_id.map(eligible[eligible.net.gt(0)].groupby('row_id').size()).fillna(0).where(~d.is_test)
        if line!='motor':d=apply_repairs(d,line,'emerging',artifacts)
        d=d.merge(customers,on='customer_id',how='left',validate='many_to_one',indicator='_customer');assert d._customer.eq('both').all()
        d=d.merge(addresses,on='address_id',how='left',validate='many_to_one',indicator='_address');assert d._address.eq('both').all()
        d['building_age']=d.year-d.year_built
        d=enrich(d,line);d.to_parquet(out/f'{line}.parquet',index=False)
        print('Prepared emerging',line,len(d),flush=True)
    all_claims=pd.concat(linked,ignore_index=True)
    # Preserve original within-product claim order for deterministic likelihood sums.
    order=pd.Series(np.arange(len(claims)),index=claims.claim_id)
    all_claims['_order']=all_claims.claim_id.map(order);all_claims['_line']=pd.Categorical(all_claims.line,categories=LINES,ordered=True)
    all_claims=all_claims.sort_values(['_line','_order']).drop(columns=['_line','_order'])
    all_claims.to_parquet(out/'claims.parquet',index=False)
    (out/'complete.json').write_text(json.dumps({'source':str(raw),'rows':209063}))
    return out
def source_data(line,prepared,raw,artifacts):
    d=pd.read_parquet(prepared/f'assembled_{line}.parquet');d=d[d.year.le(2023)].copy()
    if line!='motor':d=apply_repairs(d,line,'developed',artifacts,check_raw=False)
    exact='driver_age' if line=='motor' else 'policyholder_age';band=exact+'_band'
    d[band]=pd.cut(d[exact],bins=[17,25,35,50,65,np.inf],right=False,labels=['17-24','25-34','35-49','50-64','65+']).astype('string')
    customers=pd.read_parquet(developed_dir(raw)/'customers.parquet').set_index('customer_id')
    for field in ['prior_claims_5y','years_at_address','licence_points']:
        if field not in d:d[field]=d.customer_id.astype(str).map(customers[field])
    return enrich(d,line)
