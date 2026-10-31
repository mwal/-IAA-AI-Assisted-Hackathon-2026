"""Commercial premium build-ups; never replace Kaggle risk-cost predictions."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
def calculate(data_dir,prediction_csv,output_csv):
    raw=Path(data_dir)/'emerging';r=pd.read_parquet(raw/'reference_policies.parquet');expenses=pd.read_parquet(raw/'expenses.parquet').set_index('item').emerging
    predicted=pd.read_csv(prediction_csv);assert predicted.row_id.is_unique
    rows=pd.concat([pd.read_parquet(raw/f'test_{line}.parquet').assign(line=line) for line in ['motor','buildings','contents']],ignore_index=True)
    rows=rows[rows.policy_id.isin(r.policy_id)].merge(predicted,on='row_id',validate='one_to_one');assert rows.prediction.notna().all()
    rows['days']=(pd.to_datetime(rows.period_end)-pd.to_datetime(rows.period_start)).dt.days+1
    rows['fixed_expense']=expenses['fixed_expense_per_policy']*rows.days/rows.groupby(['line','policy_id']).days.transform('sum')
    rows['risk_premium']=rows.prediction
    rows['cat_reinsurance']=np.where(rows.line.eq('motor'),0,expenses['cat_reinsurance']*rows.risk_premium)
    commission=rows.channel.map({x:expenses['commission_'+x] for x in ['direct','broker','comparison_site']});assert commission.notna().all()
    denominator=1-expenses['variable_expense']-commission-expenses['premium_tax']-expenses['target_profit_margin'];assert denominator.gt(0).all()
    rows['office_premium']=(rows.risk_premium+rows.fixed_expense+rows.cat_reinsurance)/denominator
    for name,rate in [('variable_expense',expenses['variable_expense']),('commission',commission),('premium_tax',expenses['premium_tax']),('profit',expenses['target_profit_margin'])]:rows[name]=rows.office_premium*rate
    cols=['risk_premium','fixed_expense','cat_reinsurance','variable_expense','commission','premium_tax','profit']
    assert np.allclose(rows[cols].sum(axis=1),rows.office_premium) and np.allclose(rows.groupby(['line','policy_id']).fixed_expense.sum(),85)
    rows['whole_policy_office_premium']=rows.groupby(['line','policy_id']).office_premium.transform('sum')
    out=r[['row_id']].merge(rows[['row_id','policy_id','line','channel']+cols+['office_premium','whole_policy_office_premium']],on='row_id',validate='one_to_one');assert len(out)==20
    out.to_csv(output_csv,index=False);return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();calculate(a.data_dir,a.predictions,a.output);print('20 reference premiums written; fixed85 per policy allocated by covered days.')
