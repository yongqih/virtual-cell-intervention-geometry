"""Recover published simultaneous intervals from the archived paired bootstrap draws."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def check_family(points, draws, low, high):
    draws=np.asarray(draws)
    center=np.asarray(points)[:,None]
    sd=draws.std(axis=1,ddof=1,keepdims=True)
    critical=np.quantile(np.max(np.abs((draws-center)/sd),axis=0),.95)
    for computed,saved in [(center-critical*sd,low),(center+critical*sd,high)]:
        np.testing.assert_allclose(computed.ravel(),saved,atol=1e-9,rtol=0)
    return len(center)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    root=parser.parse_args().data
    b=pd.read_csv(root/'plot_inputs/fig12/PAIRED_METRIC_INTERVALS.csv').query('simultaneous_family')
    with np.load(root/'benchmark/paired_bootstrap.npz') as z:
        contrasts=[(z[row.endpoint+'__'+row.model_a]-z[row.endpoint+'__'+row.model_b])*(1 if row.higher_is_better else -1) for row in b.itertuples()]
        n=check_family(b.advantage_a_over_b,contrasts,b.simultaneous_ci_low,b.simultaneous_ci_high)
    t=pd.read_csv(root/'spectrum/tables/PRIMARY_SEVEN_MODEL_SIMULTANEOUS.csv')
    with np.load(root/'spectrum/PAIRED_BOOTSTRAP.npz') as z:
        n+=check_family(t.estimate,[z['average|'+m+'|primary_tail_deficit_fraction'] for m in t.model],t.simultaneous_low,t.simultaneous_high)
    folder=root/'transfer/uncertainty'
    ca=pd.read_csv(folder/'ca_fold_panel.csv')
    joints=pd.read_csv(folder/'joint_intervals.csv')
    with np.load(folder/'CA_BOOTSTRAP.npz') as z:
        pooled={m:np.mean([z[str(i)] for i in group.result_id],axis=0) for m,group in ca.groupby('method')}
    with np.load(folder/'AGGREGATED_CONTRASTS.npz') as z:
        for (a,b),g in joints.groupby(['method','comparator']):
            contrasts=[pooled[a]-pooled[b] if r.metric=='CA' else z['__'.join([a,b,r.reference,r.metric])] for r in g.itertuples()]
            n+=check_family(g.delta,contrasts,g.simultaneous_low,g.simultaneous_high)
    print(f'{n} simultaneous intervals reproduced (Fig. 1, repeatable spectra and 32-pair direct transfer).')

if __name__=='__main__':main()
