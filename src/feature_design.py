import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor,GammaRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings
from model_support import CategoryPrep,SafeOHE
def design(d,f):
 X=d[f['numeric']+f['categorical']].copy()
 for c in f['numeric']:
  X[c]=pd.to_numeric(X[c]).astype(float)
  if c in ['written_premium','sum_insured','distance_to_river_km']:X[c]=np.log1p(X[c])
 for c in f['categorical']:X[c]=X[c].astype('string').str.strip().str.lower()
 return X

def fit(X,target,weight,f,alpha,component):
 num=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())])
 cat=Pipeline([('labels',CategoryPrep(min_count=50)),('onehot',SafeOHE())])
 prep=ColumnTransformer([('numeric',num,f['numeric']),('categorical',cat,f['categorical'])],sparse_threshold=1.)
 model=(PoissonRegressor if component=='frequency' else GammaRegressor)(alpha=alpha,max_iter=2500,tol=1e-7)
 m=Pipeline([('preprocess',prep),('model',model)])
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always',ConvergenceWarning)
  m.fit(X,target,model__sample_weight=weight)
 assert not any(issubclass(w.category,ConvergenceWarning) for w in ws),'fit did not converge'
 return m
