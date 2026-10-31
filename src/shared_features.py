import pandas as pd
import numpy as np
MAPPING={'brick':'brick','brk':'brick','stone':'stone','stn':'stone','t':'timber','w':'timber','timber':'timber','wood':'timber','wod':'timber','non_standard':'other','non-standard':'other','mixed':'other','mix':'other'}
def enrich(d,line):
    d=d.copy();d['period_start']=pd.to_datetime(d.period_start);d['period_end']=pd.to_datetime(d.period_end)
    days=(d.period_end-d.period_start).dt.days+1
    assert d.period_start.dt.year.eq(d.period_end.dt.year).all() and days.gt(0).all()
    assert np.allclose(d.exposure,days/365)
    d['coverage_start_month']=d.period_start.dt.month;d['coverage_end_month']=d.period_end.dt.month
    for season,months in {'winter':[1,2,12],'spring':[3,4,5],'summer':[6,7,8],'autumn':[9,10,11]}.items():
        covered=np.zeros(len(d))
        for month in months:
            lo=pd.to_datetime({'year':d.year.to_numpy(),'month':month,'day':1}).set_axis(d.index)
            hi=lo+pd.offsets.MonthEnd(0)
            a=d.period_start.where(d.period_start>lo,lo);b=d.period_end.where(d.period_end<hi,hi)
            covered+=((b-a).dt.days+1).clip(lower=0).to_numpy()
        d['coverage_'+season+'_fraction']=covered/days.to_numpy()
    assert np.allclose(d[['coverage_'+s+'_fraction' for s in ['winter','spring','summer','autumn']]].sum(axis=1),1)
    d['observed_customer_tenure']=(d.year-d.customer_since_year).clip(lower=0)
    d['total_excess']=d.voluntary_excess+d.compulsory_excess
    if 'building_age' not in d:d['building_age']=d.year-d.year_built
    if 'construction_type' in d:
        raw=d.construction_type.astype('string').str.strip().str.lower();assert not (raw.notna()&~raw.isin(MAPPING)).any()
        d['construction_group']=raw.map(MAPPING).astype('string')
    return d
