"""Replay the two 32-pair K562 to RPE1 fits using measured calibration responses."""
import argparse
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from replay import row_indices

def predict(x, y, query, source_copy, query_copy, penalty, beta, normalize):
    scaler=StandardScaler().fit(x)
    a, b = scaler.transform(x),scaler.transform(query)
    center=a.mean(0)
    a,b=a-center,b-center
    width=a.shape[1] if normalize else 1
    kernel=a@a.T/width
    cross=b@a.T/width
    vals,vecs=eigh(kernel,check_finite=False)
    target=y-beta*source_copy
    mu=target.mean(0)
    return beta*query_copy+mu+((cross@vecs)/(np.maximum(vals,0)+penalty))@(vecs.T@(target-mu))

def calibrate(x,y,query,copy_train,copy_query,seed):
    candidates=[(lam,beta) for lam in [.0003,.003,.03,.3,3.] for beta in [0.,.5,1.]]
    assignment=np.random.default_rng(seed).permutation(len(x))%3
    losses=np.zeros(len(candidates))
    for f in range(3):
        test=assignment==f
        for j,(lam,beta) in enumerate(candidates):
            fit=predict(x[~test],y[~test],x[test],copy_train[~test],copy_train[test],lam,beta,True)
            losses[j]+=np.square(fit-y[test]).sum()/y.shape[1]
    lam,beta=candidates[np.argmin(losses)]
    return predict(x,y,query,copy_train,copy_query,lam,beta,True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path)
    p.add_argument('--fold',type=int,choices=range(5),default=0)
    p.add_argument('--panel',type=int,choices=range(5),default=0)
    args=p.parse_args()
    f,seed=args.fold,args.panel
    with np.load(args.data/'transfer/K562_3536_ALL_TARGETS.npz') as source,np.load(args.data/f'subspaces/biology_fold{f}.npz') as bio,np.load(args.data/f'transfer/direct/fold{f}_panel{seed}.npz') as expected:
        ids,queries=expected['training_names'],expected['names']
        assert len(ids)==32 and not set(ids).intersection(queries)
        cols=row_indices(expected['genes'],bio['genes'])
        y=bio['train_truth'][row_indices(ids,bio['train_names'])][:,cols]+bio['reference'][cols]
        matrix=source['delta']
        x=matrix[row_indices(ids,source['names'])].astype(float)
        q=matrix[row_indices(queries,source['names'])].astype(float)
        prior=np.zeros((len(matrix),len(cols)))
        lookup={g:i for i,g in enumerate(source['genes'])}
        for j,g in enumerate(expected['genes']):
            if g in lookup:prior[:,j]=matrix[:,lookup[g]]
        cx=prior[row_indices(ids,source['names'])];cq=prior[row_indices(queries,source['names'])]
        with threadpool_limits(1):
            outputs={'Anchor_Ridge1':predict(x,y,q,cx,cq,1.,0.,False),
                     'ResidualLinearCV':calibrate(x,y,q,cx,cq,261102+100*f+10*seed+32)}
        for name,value in outputs.items():
            error=np.abs(value-expected[name]).max()
            if not np.allclose(value,expected[name],atol=2e-6,rtol=2e-6):
                raise ValueError(f'{name}: predictions differ by {error:.6g}')
            print(f'{name}: {len(queries)} held queries, maximum absolute difference {error:.3g}')

if __name__=='__main__':main()
