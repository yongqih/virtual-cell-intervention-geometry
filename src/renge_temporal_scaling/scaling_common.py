from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];SCRIPT_ROOT=ROOT/"scripts"/"renge_temporal_scaling";RESULT_ROOT=ROOT/"results"/"renge_temporal_scaling";CACHE_ROOT=RESULT_ROOT/"cache"
FROZEN_CACHE=ROOT/"data"/"propagation_reproduction"/"cache"/"renge_processed.npz";PSEUDO_CACHE=ROOT/"results"/"renge_state_vs_wave"/"cache"/"state_wave_pseudoreplicates.npz"
sys.path.insert(0,str(ROOT/"scripts"/"renge_state_vs_wave"))
from state_wave_common import (apply_affine,atomic_json,atomic_npz,complete_metrics,direct_indices,fit_dense_transition,geometry_rho,grouped_twofold_splits,
 rank_metrics,repeat_bootstrap,representation,response_cosines,safe_pearson,safe_spearman,sha256,standardized_ridge,strict_trans_cosine_distance)  # noqa:E402

def fixed_ridge(train_x,train_y,query,alpha):
  x=np.asarray(train_x,float);y=np.asarray(train_y,float);q=np.asarray(query,float);xm=x.mean(0);xs=np.maximum(x.std(0),1e-6);ym=y.mean(0);xc=(x-xm)/xs;qc=(q-xm)/xs;yc=y-ym
  coef=xc.T@np.linalg.solve(xc@xc.T+alpha*np.eye(len(xc)),yc);return (qc@coef+ym).astype(np.float32)

def temporal_descriptor(states,train,transitions,shuffle,seed):
  blocks=[];rng=np.random.default_rng(seed)
  for index,(a,b) in enumerate(transitions):
    x=states[train,a].astype(float);y=states[train,b].astype(float)
    if shuffle:y=y[rng.permutation(len(y))]
    xc=x-x.mean(0);yc=y-y.mean(0);blocks.append((xc.T@yc/np.maximum(np.sum(xc*xc,axis=0)[:,None],1e-4)).astype(np.float32))
  return np.concatenate(blocks,axis=1)

def outer_splits(source_count,repeats,seed,test_count):
  out=[]
  for repeat in range(repeats):
    order=np.random.default_rng(seed+repeat*1009).permutation(source_count);test=np.sort(order[:test_count]);pool=np.sort(order[test_count:])
    out.append({"repeat":repeat,"seed":seed+repeat*1009,"test":test,"pool":pool})
  return out

def training_subsets(pool,n,count,seed):
  if n==len(pool):return [np.sort(pool)]
  seen=set();out=[];rng=np.random.default_rng(seed)
  while len(out)<count:
    take=tuple(sorted(rng.choice(pool,n,replace=False).tolist()))
    if take not in seen:seen.add(take);out.append(np.asarray(take,int))
  return out

def paired_repeat_bootstrap(frame,value,seed,resamples):
  vals=frame.groupby("repeat")[value].mean().to_numpy(float);rng=np.random.default_rng(seed);draw=np.mean(rng.choice(vals,(resamples,len(vals)),replace=True),axis=1)
  return {"point":float(vals.mean()),"ci_low":float(np.quantile(draw,.025)),"ci_high":float(np.quantile(draw,.975)),"n_repeats":len(vals)}
