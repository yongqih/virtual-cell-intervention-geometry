"""Exact paper metrics plus explicitly separated harmonized diagnostics."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

def corr(a,b,w=None):
 a=np.asarray(a,float);b=np.asarray(b,float)
 if w is None:w=np.ones_like(a)
 w=np.asarray(w,float); sw=w.sum()
 if sw<=0:return np.nan
 a=a-np.sum(a*w)/sw;b=b-np.sum(b*w)/sw
 den=np.sqrt(np.sum(w*a*a)*np.sum(w*b*b))
 return float(np.sum(w*a*b)/den) if den>1e-25 else np.nan
def spearman(a,b):return corr(rankdata(a),rankdata(b))
def cosine_similarity(x):
 x=np.asarray(x,float);norm=np.linalg.norm(x,axis=1)
 if np.any(norm<=1e-12):raise ValueError('Cosine geometry undefined for zero-length row')
 z=x/norm[:,None];s=z@z.T;np.fill_diagonal(s,1.)
 return s
def discrimination(pred,truth,metric='euclidean'):
 d=cdist(pred,truth,metric=metric); n=len(d); own=np.diag(d)
 ca=(d>own[:,None]).sum(1)/(n-1)
 rt=((d<=own[:,None]).sum(1)-1)/(n-1)
 ro=((d<=own[None,:]).sum(0)-1)/(n-1)
 eq=d==own[:,None];np.fill_diagonal(eq,False)
 return ca,ro,rt,int(eq.sum())
def spectrum(x):
 x=x-x.mean(0);eig=np.linalg.eigvalsh(x@x.T);eig=np.maximum(eig,0);eig=eig[::-1]
 p=eig/eig.sum() if eig.sum()>1e-24 else np.zeros_like(eig)
 q=p[p>1e-15]
 return dict(variance=float(eig.sum()/len(x)/x.shape[1]),stable_rank=float(1/p[0]) if p[0]>0 else 0.,entropy_rank=float(np.exp(-np.sum(q*np.log(q)))) if len(q) else 0.,participation_rank=float(1/np.sum(p*p)) if len(q) else 0.,pc1=float(p[0]))
def measure(pred,truth,k=10):
 ca,ro,rt,ties=discrimination(pred,truth); cc,cr,ct,cties=discrimination(pred,truth,'cosine')
 sp=cosine_similarity(pred);st=cosine_similarity(truth);n=len(pred);off=~np.eye(n,dtype=bool);ij=np.triu_indices(n,1)
 p=sp[off].reshape(n,n-1);t=st[off].reshape(n,n-1)
 rp=rankdata(p,axis=1);rtg=rankdata(t,axis=1);rp-=rp.mean(1)[:,None];rtg-=rtg.mean(1)[:,None]
 local=np.sum(rp*rtg,axis=1)/np.sqrt(np.sum(rp*rp,axis=1)*np.sum(rtg*rtg,axis=1))
 dp=-sp.copy();dt=-st.copy();np.fill_diagonal(dp,np.inf);np.fill_diagonal(dt,np.inf)
 pn=np.argsort(dp,axis=1,kind='stable')[:,:k];tn=np.argsort(dt,axis=1,kind='stable')[:,:k]
 overlap=np.array([len(set(a)&set(b))/k for a,b in zip(pn,tn)])
 row_frob=np.sqrt(np.mean((p-t)**2,axis=1))
 a=spectrum(pred);b=spectrum(truth)
 global_metrics=dict(centroid_accuracy=float(ca.mean()),original_rank=float(ro.mean()),transposed_rank=float(rt.mean()),cosine_centroid_accuracy=float(cc.mean()),cosine_original_rank=float(cr.mean()),cosine_transposed_rank=float(ct.mean()),frob=float(np.linalg.norm(sp-st)),frob_rmse=float(np.sqrt(np.mean((p-t)**2))),global_spearman=spearman(sp[ij],st[ij]),local_spearman=float(np.nanmean(local)),top10_overlap=float(overlap.mean()),variance_retention=a['variance']/b['variance'],effective_rank=a['entropy_rank'],truth_effective_rank=b['entropy_rank'],effective_rank_ratio=a['entropy_rank']/b['entropy_rank'],stable_rank=a['stable_rank'],participation_rank=a['participation_rank'],identity_max_error=float(np.max(abs(ca+rt-1))),cosine_identity_max_error=float(np.max(abs(cc+ct-1))),cross_row_ties=ties,cosine_cross_row_ties=cties)
 per=dict(centroid_accuracy=ca,original_rank=ro,transposed_rank=rt,cosine_centroid_accuracy=cc,cosine_original_rank=cr,cosine_transposed_rank=ct,local_spearman=local,top10_overlap=overlap,frob_row_rmse=row_frob)
 return global_metrics,per,sp,st
def representations(z):
 p=z['pred'];t=z['truth'];c=z['control'];r=z['reference'];ir=z['inner_reference']
 with np.errstate(invalid='ignore',divide='ignore'):
  lp=np.log2(p+.1)-np.log2(c+.1);lt=np.log2(t+.1)-np.log2(c+.1)
 # Follow the upstream log-fold-change convention for missing entries.
 lp=np.where(np.isnan(lp),0,lp);lt=np.where(np.isnan(lt),0,lt)
 assert np.isfinite(lp).all() and np.isfinite(lt).all()
 return {'expression':(p,t),'response':(p-c,t-c),'residual':(p-c-r,t-c-r),'pb_logfc':(lp,lt),'inner_train_residual':(p-c-ir,t-c-ir)}
