"""Discrete lognormal reporting delays fitted with individual right truncation."""
import numpy as np
import pandas as pd
from scipy.special import log_ndtr,ndtr
from scipy.optimize import minimize

def log_cdf(days,mu,sigma):
    d=np.asarray(days,float);out=np.full(d.shape,-np.inf);ok=d>=0
    out[ok]=log_ndtr((np.log(d[ok]+1)-mu)/sigma)
    return out
def cdf(days,fit):return np.exp(log_cdf(days,fit['mu'],fit['sigma']))
def log_mass(days,mu,sigma):
    d=np.asarray(days,float);hi=(np.log(d+1)-mu)/sigma
    lo=np.full(len(d),-np.inf);ok=d>0;lo[ok]=(np.log(d[ok])-mu)/sigma
    def difference(a,b):
        with np.errstate(divide='ignore',invalid='ignore'):
            return a+np.log(-np.expm1(np.minimum(b-a,0)))
    lower=difference(log_ndtr(hi),log_ndtr(lo))
    upper=difference(log_ndtr(-lo),log_ndtr(-hi))
    return np.where(hi<=0,lower,upper)
def objective(theta,delay,age):
    mu,logsigma=theta;sigma=np.exp(logsigma)
    values=log_mass(delay,mu,sigma)-log_cdf(age,mu,sigma)
    return float(-values.sum()) if np.isfinite(values).all() else 1e100
def fit_delay(claims,cutoff):
    cutoff=pd.Timestamp(cutoff);q=claims[pd.to_datetime(claims.report_date).le(cutoff)].copy()
    delay=(pd.to_datetime(q.report_date)-pd.to_datetime(q.loss_date)).dt.days.to_numpy(float)
    age=(cutoff-pd.to_datetime(q.loss_date)).dt.days.to_numpy(float)
    assert len(q)>=50 and (delay>=0).all() and (age>=delay).all()
    logd=np.log(delay+.5);mu=float(np.clip(logd.mean(),-1,7));sig=float(np.clip(logd.std(),.3,2))
    starts=[(mu,np.log(sig)),(min(mu+.3,7.5),np.log(min(sig*1.2,2.5)))]
    bounds=[(-2,8),(np.log(.2),np.log(3))];results=[]
    for start in starts:
        result=minimize(objective,start,args=(delay,age),method='L-BFGS-B',bounds=bounds,options={'maxiter':1000,'ftol':1e-12,'gtol':1e-6})
        results.append(result)
    good=[r for r in results if r.success and np.isfinite(r.fun) and r.fun<1e99]
    assert good,'No converged reporting-delay fit'
    best=min(good,key=lambda r:r.fun);mu,logs=best.x;sigma=float(np.exp(logs))
    interior=all(lo+1e-4<x<hi-1e-4 for x,(lo,hi) in zip(best.x,bounds));assert interior,'Delay fit on parameter bound; no tail correction exported'
    fit={'mu':float(mu),'sigma':sigma,'negative_log_likelihood':float(best.fun),'claims':len(q),'delay_min':int(delay.min()),'delay_max':int(delay.max()),'observed_delay_above365':int((delay>365).sum()),'cutoff':str(cutoff.date()),'interior':interior,'converged':True,'optimization_starts':len(starts)}
    fit['cdf']={str(t):float(cdf(np.array([t]),fit)[0]) for t in [0,7,30,90,180,365,730]}
    return fit
def observable_exposure(rows,cutoff,fit):
    cutoff=pd.Timestamp(cutoff)
    end=pd.to_datetime(rows.period_end);start=pd.to_datetime(rows.period_start)
    assert end.le(cutoff).all()
    lo=(cutoff-end).dt.days.to_numpy();hi=(cutoff-start).dt.days.to_numpy()
    curve=cdf(np.arange(int(hi.max())+1),fit);prefix=np.r_[0,np.cumsum(curve)]
    expo=(prefix[hi+1]-prefix[lo])/365
    assert np.isfinite(expo).all() and (expo>0).all() and (expo<=rows.exposure.to_numpy()+1e-12).all()
    return expo
