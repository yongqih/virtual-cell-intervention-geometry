from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];SCRIPT_ROOT=ROOT/"scripts"/"renge_markov_autopsy_v2";RESULT_ROOT=ROOT/"results"/"renge_markov_autopsy_v2";CACHE_ROOT=RESULT_ROOT/"cache";FROZEN_CACHE=ROOT/"data"/"propagation_reproduction"/"cache"/"renge_processed.npz"
sys.path.insert(0,str(ROOT/"scripts"/"renge_endpoint_benchmark"))
from benchmark_common import (atomic_json,atomic_npz,complete_metrics,direct_indices,geometry_rho,grouped_twofold_splits,rank_metrics,representation,response_cosines,safe_pearson,safe_spearman,sha256,source_rows,strict_trans_cosine_distance)  # noqa:E402

def fit_ridge(x,y,alpha):
  x=np.asarray(x,float);y=np.asarray(y,float);xm=x.mean(0);xs=np.maximum(x.std(0),1e-6);ym=y.mean(0);xc=(x-xm)/xs;yc=y-ym;coef=xc.T@np.linalg.solve(xc@xc.T+alpha*np.eye(len(xc)),yc)
  return {"mean":xm.astype(np.float32),"scale":xs.astype(np.float32),"ymean":ym.astype(np.float32),"coef":coef.astype(np.float32)}
def apply(model,q):return (((np.asarray(q)-model["mean"])/model["scale"])@model["coef"]+model["ymean"]).astype(np.float32)
def pooled_xy(states,rows):return np.concatenate([states[rows,0],states[rows,1],states[rows,2]]),np.concatenate([states[rows,1],states[rows,2],states[rows,3]])
def repeated_a(a,rows):return np.concatenate([a[rows],a[rows],a[rows]])

def fit_impulse(states,train,alpha):
  x,y=pooled_xy(states,train);return {"kind":"impulse","f":fit_ridge(x,y,alpha)}
def fit_additive(states,a,train,alpha):
  x,y=pooled_xy(states,train);f=fit_ridge(x,y,alpha);res=y-apply(f,x);g=fit_ridge(repeated_a(a,train),res,alpha);return {"kind":"additive","f":f,"g":g}
def interaction_basis(states,a,train,rank):
  x,_=pooled_xy(states,train);aa=repeated_a(a,train);xm=x.mean(0);am=aa.mean(0);_,_,xvt=np.linalg.svd(x-xm,full_matrices=False);_,_,avt=np.linalg.svd(aa-am,full_matrices=False)
  return xm.astype(np.float32),am.astype(np.float32),xvt[:rank].T.astype(np.float32),avt[:rank].T.astype(np.float32)
def conditional_features(state,a,basis):
  xm,am,xc,ac=basis;zx=(state-xm)@xc;za=(a-am)@ac;return np.concatenate([state,a,zx*za],axis=1)
def fit_conditional(states,a,train,alpha,rank):
  x,y=pooled_xy(states,train);aa=repeated_a(a,train);basis=interaction_basis(states,a,train,rank);return {"kind":"conditional","basis":basis,"joint":fit_ridge(conditional_features(x,aa,basis),y,alpha)}
def fit_additive_xy(x,y,a,alpha):
  f=fit_ridge(x,y,alpha);g=fit_ridge(a,y-apply(f,x),alpha);return {"kind":"additive","f":f,"g":g}
def fit_conditional_xy(x,y,a,alpha,rank):
  xm=x.mean(0);am=a.mean(0);_,_,xvt=np.linalg.svd(x-xm,full_matrices=False);_,_,avt=np.linalg.svd(a-am,full_matrices=False);basis=(xm.astype(np.float32),am.astype(np.float32),xvt[:rank].T.astype(np.float32),avt[:rank].T.astype(np.float32));return {"kind":"conditional","basis":basis,"joint":fit_ridge(conditional_features(x,a,basis),y,alpha)}
def step(model,state,a):
  if model["kind"]=="impulse":return apply(model["f"],state)
  if model["kind"]=="additive":return apply(model["f"],state)+apply(model["g"],a)
  return apply(model["joint"],conditional_features(state,a,model["basis"]))
def rollout(model,entry,a,steps):
  out=[];state=entry
  for _ in range(steps):state=step(model,state,a);out.append(state)
  return out
def group_metrics(split,model,scope,stage,pred,truth,sources,genes,k,extra=None):
  test=split["test"];m=complete_metrics(pred,truth,sources[test],genes,k);return {"repeat":split["repeat"],"group":split["group"],"split_seed":split["seed"],"model":model,"scope":scope,"stage":stage,"n_train_sources":len(split["train"]),"n_test_sources":len(test),"one_model_for_entire_heldout_group":True,"heldout_source_absent_all_days":True,"outer_test_used_for_selection":False,**m,**(extra or {})}
def residual_metrics(pred,truth,names,genes):
  base=complete_metrics(pred,truth,names,genes,3)
  zero_mse=complete_metrics(np.zeros_like(truth),truth,names,genes,3)["strict_trans_mse"]
  base["mse_reduction_vs_zero"]=1-base["strict_trans_mse"]/max(zero_mse,1e-12)
  return base
def paired_delta(left,right,metric,seed,resamples):
  l=left.groupby("repeat")[metric].mean();r=right.groupby("repeat")[metric].mean();v=(l-r).dropna().to_numpy(float);rng=np.random.default_rng(seed);draw=np.mean(rng.choice(v,(resamples,len(v)),replace=True),axis=1);return {"point":float(v.mean()),"ci_low":float(np.quantile(draw,.025)),"ci_high":float(np.quantile(draw,.975)),"n_repeats":len(v)}
