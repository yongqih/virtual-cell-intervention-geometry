"""Recalculate reported point estimates from archived predictions and measurements."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

MODELS = ['GEARS', 'scGPT', 'Scouter', 'CellFlow', 'PerturbDiff', 'State-ST-GenePT', 'TxPert-public']

def global_rho(pred, truth, decimals=None):
    a = pdist(pred, 'cosine')
    b = pdist(truth, 'cosine')
    if decimals is not None:
        a, b = np.round(a, decimals), np.round(b, decimals)
    return spearmanr(a, b).statistic

def ca(pred, truth):
    d = cdist(pred, truth)
    return (d > d.diagonal()[:, None]).sum() / (len(d) * (len(d)-1))

def top10(pred, truth):
    def neighbors(a):
        d = cdist(a, a, 'cosine')
        np.fill_diagonal(d, np.inf)
        return np.argsort(d, axis=1, kind='stable')[:, :10]
    return np.mean([len(set(a) & set(b))/10 for a,b in zip(neighbors(pred),neighbors(truth))])

def residual_profile(z):
    return z['pred'] - z['control'] - z['reference'], z['truth'] - z['control'] - z['reference']

def row_indices(wanted, available):
    lookup = {name: i for i, name in enumerate(available)}
    if len(lookup) != len(available):
        raise ValueError('Duplicate intervention names in reference array')
    return [lookup[n] for n in wanted]

class Replay:
    def __init__(self, data):
        self.root = data
        self.tables = data/'plot_inputs'
        self.rows = []

    def compare(self, label, actual, expected, tolerance=2e-7):
        self.rows.append((label, float(actual), float(expected)))
        if not np.isclose(actual, expected, atol=tolerance, rtol=0):
            raise ValueError(f'{label}: recomputed {actual}, expected {expected}')

    def benchmark(self):
        from analysis_metrics import measure, representations
        endpoints = pd.read_csv(self.tables/'fig12/main_E_benchmark_by_fold.csv').set_index(['model','fold','endpoint']).estimate
        dim = pd.read_csv(self.tables/'fig12/display_metrics_by_fold.csv').set_index(['model','fold'])
        mapping = dict(Systema_CA=('expression','centroid_accuracy'),PB_expression_original_rank=('expression','original_rank'),
                       PB_logFC_cosine_original_rank=('pb_logfc','cosine_original_rank'),PB_logFC_Frobenius_RMS=('pb_logfc','frob_rmse'),
                       IGC_global_rho=('residual','global_spearman'),IGC_top10=('residual','top10_overlap'))
        for model in MODELS:
            for f in range(5):
                with np.load(self.root/f'benchmark/{model}_fold{f}.npz') as z:
                    reps=representations(z)
                    scores={k:measure(*reps[k])[0] for k in ['expression','pb_logfc','residual']}
                for key,(rep,metric) in mapping.items():
                    self.compare(f'F1/{model}/{f}/{key}',scores[rep][metric],endpoints[model,f,key])
                self.compare(f'F2/{model}/{f}/effective rank',scores['residual']['effective_rank'],dim.loc[(model,f),'effective_rank'])
        energy=pd.read_csv(self.root/'spectrum/tables/ENERGY_METRICS_BY_FOLD.csv').set_index(['fold','object','metric']).estimate
        for f in range(5):
            with np.load(self.root/f'spectrum/half_measurements_fold{f}.npz') as z:
                a, b = z['half_A'], z['half_B']
            a-=a.mean(axis=1,keepdims=True); b-=b.mean(axis=1,keepdims=True)
            total=(a*b).sum(axis=2).mean()
            self.compare(f'F2/{f}/repeatable variance',total,energy[f,'Observed','reproducible_total_variance'])
            with np.load(self.root/f'subspaces/unseen_bases_fold{f}.npz') as bases:
                for model in MODELS:
                    q=int(dim.loc[(model,f),'pc80'])
                    u=bases[model][:,:q]
                    with np.load(self.root/f'benchmark/{model}_fold{f}.npz') as z:
                        y=z['pred'].astype(float)
                    y-=y.mean(0)
                    measured_tail=total-((a@u)*(b@u)).sum(axis=2).mean()
                    predicted_tail=(y*y).sum(axis=1).mean()-((y@u)**2).sum(axis=1).mean()
                    self.compare(f'F2/{f}/{model}/tail deficit',(measured_tail-predicted_tail)/total,energy[f,model,'primary_tail_deficit_fraction'])

    def coordinates(self):
        pairs=pd.read_csv(self.tables/'fig34/TRAIN_PAIRWISE_FOLD.csv')
        for row in pairs.query('training_panel=="full" and model_i=="Scouter" and model_j=="TxPert-public"').itertuples():
            with np.load(self.root/f'subspaces/train_bases_fold{row.fold}_full.npz') as z:
                a,b=[z[k][:,:row.q] for k in ['Scouter','TxPert-public']]
            self.compare(f'F3/{row.fold}/{row.q}/Scouter vs TxPert',np.square(a.T@b).sum()/row.q,row.overlap)
        results=[]
        for f in range(5):
            with np.load(self.root/f'coordinate_inputs/assignment/fold{f}_inputs.npz') as parent:
                with np.load(self.root/f'coordinate_inputs/P1/fold{f}_q8.npz') as z:
                    i=row_indices(z['names'],parent['names']);cols=z['columns']; truth=parent['truth'][i][:,cols]
                    for method in ['Frozen_model','SharedRidge_Paired','K562_raw_corr','K562_resid_corr','K562_response']:
                        pred=z[method][:,cols]
                        results.append((method,global_rho(pred,truth),top10(pred,truth),ca(pred,truth)))
                with np.load(self.root/f'coordinate_inputs/assignment/anchor_fold{f}_q8.npz') as z:
                    i=row_indices(z['names'],parent['names']);cols=parent['common_columns']; u=parent['U'][:,:8]
                    pred=parent['pred'][i];truth=parent['truth'][i]; co=z['anchor_coordinates']
                    for mode,entry in [('anchor_coordinates',co),('wrong_target_anchor',co[z['wrong_permutation']])]:
                        rebuilt=pred+(entry-pred@u)@u.T
                        results.append((mode,global_rho(rebuilt[:,cols],truth[:,cols]),top10(rebuilt[:,cols],truth[:,cols]),ca(rebuilt[:,cols],truth[:,cols])))
        matched=pd.read_csv(self.tables/'fig34/P1_SUMMARY.csv').set_index('method')
        identity=pd.read_csv(self.tables/'fig34/tx_SUMMARY_PERFORMANCE.csv').query('experiment=="anchor" and budget==0 and q==8 and readout=="common"').set_index('method')
        for method,row in pd.DataFrame(results,columns=['method','rho','top10','CA']).groupby('method').mean().iterrows():
            for metric,v in row.items():
                expected=matched.loc[method,metric] if method in matched.index else identity.loc[method,metric.lower()]
                self.compare(f'F4/{method}/{metric}',v,expected)

    def temporal(self):
        ref=pd.read_csv(self.tables/'protein/grouped_geometry_summary.csv').query('metric=="cosine_geometry" and early==24 and late==48')
        with np.load(self.root/'temporal/proteintalks/grouped_predictions.npz') as z:
            for c,cell in enumerate(z['cells']):
                for model in ['ode_measured','ridge_measured','ode_predicted','ridge_predicted','full_ridge','persistence']:
                    vals=[]
                    for f in range(5):
                        key=f'f{f}_c{c}_e24_l48'
                        vals.append(global_rho(z[key+'__'+model],z[key+'_truth'],decimals=12))
                    target=ref[(ref.cell==cell)&(ref.model==model)].estimate.item()
                    self.compare(f'F5/{cell}/{model}',np.mean(vals),target)

    def transfer(self):
        points=[]
        for f in range(5):
            with np.load(self.root/f'transfer/TxPert_fold{f}_response.npz') as tx,np.load(self.root/f'subspaces/biology_fold{f}.npz') as bio:
                index=row_indices(tx['names'],bio['held_names']);cols=tx['readout_columns']
                truth=bio['held_truth'][index][:,cols]+bio['reference'][cols]
                for panel in range(5):
                    with np.load(self.root/f'transfer/direct/fold{f}_panel{panel}.npz') as z:
                        assert np.array_equal(z['names'],tx['names']) and not set(z['training_names']).intersection(z['names'])
                        ref=z['reference11']
                        for name in ['TxPert_frozen','Anchor_Ridge1','ResidualLinearCV']:
                            y=tx['response'] if name=='TxPert_frozen' else z[name]
                            points.append((name,global_rho(y-ref,truth-ref),top10(y-ref,truth-ref),ca(y,truth)))
        table=pd.read_csv(self.tables/'fig6/benchmark_n32.csv').query('reference=="paired11_primary"').set_index('method')
        for method,row in pd.DataFrame(points,columns=['method','rho','top10','CA']).groupby('method').mean().iterrows():
            for metric,v in row.items():self.compare(f'F6/direct/{method}/{metric}',v,table.loc[method,metric])
        table=pd.read_csv(self.tables/'fig6/two_hop/FOLD_METRICS.csv').query('context=="target"').set_index(['route','fold','method'])
        for route in ['K562_RPE1_Jurkat','Jurkat_RPE1_K562']:
            for f in range(5):
                with np.load(self.root/f'bridges/{route}/fold{f}.npz') as z:
                    assert not set(z['first_training']).intersection(z['second_training'])
                    truth,ref=z['target_truth'],z['target_reference']
                    for method in ['Measured_two_hop','TxPert_one_hop','Source_Copy','Wrong_source_two_hop']:
                        actual=global_rho(z[method]-ref,truth-ref)
                        self.compare(f'F6/{route}/{f}/{method}/rho',actual,table.loc[(route,f,method),'rho'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path,help='Extracted Zenodo data/ directory')
    parser.add_argument('--table',type=Path,help='Optional per-estimate comparison CSV')
    args=parser.parse_args()
    replay=Replay(args.data)
    with threadpool_limits(1):
        for section in [replay.benchmark,replay.coordinates,replay.temporal,replay.transfer]:
            section()
            print(f'Completed {section.__name__}',flush=True)
    result=pd.DataFrame(replay.rows,columns=['comparison','recomputed','archived'])
    if args.table:result.to_csv(args.table,index=False)
    print(f'{len(result)} manuscript estimates reproduced from numerical arrays.')

if __name__=='__main__':main()
