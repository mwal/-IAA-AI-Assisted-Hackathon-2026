"""Serializable training-only category encoding for emerging baseline."""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class CategoryPrep(BaseEstimator,TransformerMixin):
 def __init__(self,min_count=50):self.min_count=min_count
 def fit(self,X,y=None):
  X=pd.DataFrame(X).astype('string').fillna('__MISSING__')
  self.seen_=[];self.keep_=[]
  for c in X:
   counts=X[c].value_counts();self.seen_.append(set(counts.index));self.keep_.append(set(counts[counts>=self.min_count].index)|{'__MISSING__'})
  return self
 def transform(self,X):
  X=pd.DataFrame(X).astype('string').fillna('__MISSING__');arr=[]
  for i,c in enumerate(X):
   s=X[c];s=s.where(s.isin(self.keep_[i]),'__RARE__');s=s.where(X[c].isin(self.seen_[i])|X[c].eq('__MISSING__'),'__UNSEEN__');arr.append(s.to_numpy(dtype=object))
  return np.column_stack(arr)
class SafeOHE(BaseEstimator,TransformerMixin):
 def fit(self,X,y=None):
  cats=[np.array(sorted(set(X[:,i])|{'__MISSING__','__RARE__','__UNSEEN__'}),dtype=object) for i in range(X.shape[1])]
  self.encoder_=OneHotEncoder(categories=cats,handle_unknown='error',sparse_output=True)
  self.encoder_.fit(X);return self
 def transform(self,X):return self.encoder_.transform(X)
