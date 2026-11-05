"""Draw response subspaces and intervention-coordinate comparisons."""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter

from settings import INPUT, OUTPUT as OUT, CHECKS as LAYOUT
DATA=INPUT/'fig34'
from layout import record_panel_positions
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':6.7,'axes.labelsize':6.7,'axes.titlesize':7,'xtick.labelsize':6.1,
 'ytick.labelsize':6.1,'legend.fontsize':6.1,'axes.linewidth':.65,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','legend.frameon':False,'lines.linewidth':1.15,
 'patch.linewidth':.65,'figure.facecolor':'white','savefig.facecolor':'white'})
INK='#222222';GREY='#8A8F94';GRID='#E4E7E8';BLUE='#7B9CAF';TEAL='#86A79B'
OCHRE='#C99A70';PALE='#E9D9C8';LIGHTBLUE='#B7CBD6';DARKTEAL='#688F82'
COLORS={'Frozen_model':GREY,'SharedRidge_Paired':LIGHTBLUE,'SharedRidge_Original':LIGHTBLUE,
 'K562_raw_corr':OCHRE,'K562_resid_corr':PALE,'K562_response':TEAL,
 'K562_raw_cov':OCHRE,'K562_resid_cov':PALE,'RPE1_raw_corr':BLUE,
 'RPE1_resid_corr':LIGHTBLUE,'EstablishedOBS71':OCHRE,'Mean_coordinates':'#D0D4D4',
 'Oracle_coordinates_diagnostic':DARKTEAL}
LABELS={'Frozen_model':'Original model','SharedRidge_Paired':'Shared calibration',
 'SharedRidge_Original':'Shared (all training)','K562_raw_corr':'Natural correlation',
 'K562_resid_corr':'Natural, residualized','K562_response':'Measured response',
 'K562_raw_cov':'Natural covariance','K562_resid_cov':'Covariance, residualized',
 'RPE1_raw_corr':'RPE1 correlation','RPE1_resid_corr':'RPE1, residualized',
 'EstablishedOBS71':'OBS71 descriptors','Mean_coordinates':'Mean coordinates',
 'Oracle_coordinates_diagnostic':'True coordinates'}
MODEL_LABEL={'TxPert-public':'TxPert','State-ST-GenePT':'State','CellFlow':'CellFlow',
 'Biology_train_matched':'Biological basis','Scouter':'Scouter','GEARS':'GEARS',
 'scGPT':'scGPT','PerturbDiff':'PerturbDiff'}
METRIC={'CA':'CA','ca':'CA','rho':'Global ρ','top10':'Top-10',
 'coordinate_variance_retention':'Coordinate variance ratio',
 'response_variance_retention':'Response variance ratio','coordinate_mse':'Coordinate MSE'}
def read(n):return pd.read_csv(DATA/n)
CAP=read('sub_HELDOUT_REPRESENTATIONAL_CAPACITY.csv')
ALIGN=read('sub_MODEL_BIOLOGY_ALIGNMENT.csv')
SHIFT=read('sub_TRAIN_TO_UNSEEN_SUBSPACE_SHIFT.csv')
P1=read('P1_SUMMARY.csv');P2=read('P2_SUMMARY.csv');P3=read('P3_SUMMARY.csv')
TX=read('tx_SUMMARY_PERFORMANCE.csv');SC=read('sc_SUMMARY_PERFORMANCE.csv')
PLOT_ROWS=[]

class Page:
 def __init__(self,name,height=220):
  self.name=name;self.w=183.;self.h=float(height)
  self.fig=plt.figure(figsize=(self.w/25.4,self.h/25.4))
  self.cv=self.fig.add_axes([0,0,1,1],label='design_canvas')
  self.cv.set(xlim=(0,self.w),ylim=(0,self.h));self.cv.axis('off')
  self.axes=[];self.ids=[];self.rows=[];self.cols=[]
 def text(self,x,y,s,size=6.7,ha='left',color=INK,weight='normal',**kw):
  return self.cv.text(x,y,s,fontsize=size,ha=ha,va='center',color=color,fontweight=weight,**kw)
 def panel(self,letter,title,x,y):
  self.text(x,y,letter,9,weight='bold');self.text((x+180)/2 if x>90 else (x+90)/2 if self.name=='Figure_3' and letter in ['a','c'] else 94,y,title,7.1,ha='center',weight='bold')
 def ax(self,x,y,w,h,id):
  a=self.fig.add_axes([x/self.w,y/self.h,w/self.w,h/self.h],label=id)
  a.tick_params(length=2.2,width=.6,pad=2);self.axes.append(a);self.ids.append(id);return a
 def arrow(self,a,b,color=INK,ls='-',lw=.85,scale=6):
  obj=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=scale,color=color,
   linewidth=lw,linestyle=ls,shrinkA=0,shrinkB=0)
  self.cv.add_patch(obj);return obj
 def key(self,x,y,label,color,ls='-',marker=None):
  self.cv.plot([x,x+4],[y,y],color=color,lw=1.2,ls=ls)
  if marker:self.cv.plot([x+2],[y],marker=marker,ms=3,mfc=color,mec=INK,mew=.3)
  self.text(x+5.3,y,label,6.1)
 def save(self):
  self.fig.canvas.draw()
  report=record_panel_positions(self.fig,axes=self.axes,panel_ids=self.ids,
   row_groups=self.rows,column_groups=self.cols,json_out=LAYOUT/f'{self.name}.alignment.json',
   overlay_svg=LAYOUT/f'{self.name}.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
  self.fig.savefig(OUT/f'{self.name}.pdf',dpi=600)
  self.fig.savefig(OUT/f'{self.name}.svg',dpi=600)
  self.fig.savefig(OUT/f'{self.name}.png',dpi=300)
  print(json.dumps({'figure':self.name,'axes':len(self.axes),'alignment':report['status']}),flush=True)
  plt.close(self.fig)

def clean(a,axis='y'):
 a.set_axisbelow(True);a.grid(axis=axis,lw=.4,ls=(0,(2,3)),color=GRID)
def log_rows(p,a,frame,role):
 for r in frame.to_dict('records'):PLOT_ROWS.append({'figure':p.name,'panel':a.get_label(),'role':role,**r})
def fold_bar(a,x,values,color,width=.58):
 v=np.array(values,dtype=float);assert len(v)>0 and np.isfinite(v).all()
 a.bar(x,v.mean(),width=width,color=color,edgecolor=INK,lw=.65,zorder=2)
 jit=np.linspace(-width*.22,width*.22,len(v)) if len(v)>1 else [0]
 a.scatter(x+np.array(jit),v,s=11,facecolors='white',edgecolors=INK,lw=.55,zorder=4)
def points(a,frame,methods,metric,xlim,show_labels=True,colors=None,labels=None):
 for i,m in enumerate(methods):
  d=frame[frame.method.eq(m)];assert len(d)==1,(m,len(d),metric)
  r=d.iloc[0];v,lo,hi=[float(r[k]) for k in [metric,metric+'_ci_low',metric+'_ci_high']]
  assert np.isfinite([v,lo,hi]).all() and lo<=v<=hi,(m,metric,v,lo,hi)
  c=(colors or COLORS).get(m,BLUE)
  a.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',ms=4,mfc=c,mec=INK,mew=.45,
   ecolor=c,elinewidth=1.2,capsize=2,zorder=3)
 a.set(xlim=xlim,ylim=(len(methods)-.55,-.55),yticks=range(len(methods)),
  yticklabels=(labels or [LABELS.get(m,m) for m in methods]) if show_labels else [])
 a.tick_params(axis='y',length=0,pad=4);clean(a,'x');a.xaxis.set_major_locator(MaxNLocator(3))
def grouped_folds(a,frame,models,cols,colors,ylim):
 for i,m in enumerate(models):
  d=frame[frame.model.eq(m)].sort_values('fold')
  for j,col in enumerate(cols):fold_bar(a,i+(j-(len(cols)-1)/2)*.32,d[col],colors[j],.29)
 a.set(xticks=range(len(models)),xticklabels=[MODEL_LABEL[m] for m in models],ylim=ylim)
 a.yaxis.set_major_locator(MaxNLocator(4));clean(a)
def curve_ci(a,d,metric,color=TEAL,levels=None,key='cells'):
 levels=levels or ['5','15','30','all']; d=d.copy();d[key]=d[key].astype(str)
 d=d.set_index(key).loc[list(map(str,levels))]
 x=np.arange(len(d));v=d[metric].to_numpy();lo=d[metric+'_ci_low'].to_numpy();hi=d[metric+'_ci_high'].to_numpy()
 assert np.isfinite(np.r_[v,lo,hi]).all()
 a.fill_between(x,lo,hi,color=color,alpha=.18,lw=0)
 a.plot(x,v,'o-',color=color,ms=3.5,mec=INK,mew=.4)
 a.set(xticks=x,xticklabels=['All' if str(k)=='all' else str(k) for k in levels],xlim=(-.2,len(x)-.8))
 clean(a);a.yaxis.set_major_locator(MaxNLocator(4))
def fold_curve(a,d,col,color,ls='-',label=None):
 assert (d.q>0).all()
 for _,g in d.groupby('fold'):
  g=g.sort_values('q');a.plot(np.log2(g.q),g[col],color=color,lw=.6,alpha=.28,ls=ls)
 g=d.groupby('q')[col].mean().sort_index()
 a.plot(np.log2(g.index),g.values,color=color,marker='o',ms=3,lw=1.25,ls=ls,label=label,mec='white',mew=.3)
 a.set(xlim=(.85,4.15),xticks=[1,2,3,4],xticklabels=[2,4,8,16]);clean(a)
def small_title(p,x,y,w,s):p.text(x+w/2,y,s,6.8,ha='center')

from matplotlib.patches import Ellipse, Arc, PathPatch, Circle
from matplotlib.path import Path as ArtPath

def heading(p,letter,title,x,y,w):
 p.text(x,y,letter,9,weight='bold');p.text(x+w/2,y,title,7.1,ha='center',weight='bold')

def vector_plane(p,x,y,w,h):
 skew=w*7/43;ext=w*3/43
 p.cv.add_patch(Polygon([(x,y),(x+w,y),(x+w+skew,y+h),(x+skew,y+h)],closed=True,facecolor='#F1F5F4',edgecolor='#C9D9D3',lw=.6))
 p.arrow((x,y),(x+w+ext,y),GREY,lw=.6,scale=6);p.arrow((x,y),(x+skew*8/7,y+h*32/29),GREY,lw=.6,scale=6)

def cellular_input(p,x,y,color,fill,perturbed=False):
 """Original vector cell illustration; organelle positions are schematic."""
 def curve(v,lw=.45,alpha=1):
  vv=np.asarray(v)+[x,y]
  p.cv.add_patch(PathPatch(ArtPath(vv,[ArtPath.MOVETO]+[ArtPath.CURVE4]*(len(vv)-1)),fc='none',ec=color,lw=lw,alpha=alpha,capstyle='round'))
 # An asymmetric closed membrane gives the cell a soft, organic silhouette.
 membrane=[(-10,-.8),(-10.7,2.4),(-7.5,5.7),(-3.9,5.5),(-.3,5.1),(1.8,6.0),(5.3,4.3),(8.5,2.9),(10.5,.4),(9.3,-2.0),(7.3,-5.2),(3.6,-5.6),(.6,-5.0),(-2.7,-4.4),(-5.0,-5.7),(-7.7,-3.8),(-9.2,-2.9),(-10.0,-2.0),(-10,-.8)]
 p.cv.add_patch(PathPatch(ArtPath(np.asarray(membrane)+[x,y],[ArtPath.MOVETO]+[ArtPath.CURVE4]*18),fc=fill,ec=color,lw=.75,joinstyle='round'))
 # Nuclear envelope and nucleolus.
 p.cv.add_patch(Ellipse((x-2.6,y+.2),6.6,5.8,angle=-15,fc='white',ec=color,lw=.55))
 p.cv.add_patch(Ellipse((x-2.6,y+.2),5.9,5.1,angle=-15,fc='none',ec=color,lw=.27,alpha=.6))
 p.cv.add_patch(Circle((x-1.7,y+.8),.62,fc=color,ec='none',alpha=.45))
 curve([(-4.3,-.2),(-3.6,1.8),(-1.9,-1.5),(-.8,-.2)],.4,.6)
 # Mitochondrion, with recognizable inner-membrane folds.
 bean=[(3.1,2.1),(3.6,3.9),(7.5,3.7),(7.3,2.1),(7.0,.9),(4.0,.6),(3.1,2.1)]
 p.cv.add_patch(PathPatch(ArtPath(np.asarray(bean)+[x,y],[ArtPath.MOVETO]+[ArtPath.CURVE4]*6),fc='white',ec=color,lw=.5))
 for off in [0,.9,1.8]:curve([(4.0+off,2.8),(3.8+off,1.7),(5.0+off,1.6),(4.7+off,1.2)],.36)
 # Short endoplasmic-reticulum sheets and cytoplasmic RNA strands.
 for off in [0,.7,1.4]:curve([(-.2,-2.2-off),(2.2,-3.2-off),(3.5,-1.1-off),(5.4,-1.9-off)],.45,.75)
 for xx,yy in [(-7.2,1.6),(6.8,-1.1)]:
  curve([(xx,yy),(xx+.35,yy+.5),(xx+.8,yy-.5),(xx+1.25,yy)],.43,.8)
 if perturbed:
  # Targeted input cue; this is not a molecular mechanism or measured trajectory.
  p.arrow((x-11.5,y+6),(x-4.8,y+1.8),color,lw=.85,scale=5)
  p.cv.add_patch(Circle((x-4.4,y+1.5),.75,fc='white',ec=color,lw=.5))

def main3_new():
 p=Page('Figure_3',178)
 heading(p,'a','Directions and positions',3,172,54)
 vector_plane(p,12,138,28,19)
 origin=(14.6,140)
 truth=[(22.4,153.7),(35.5,149.8),(31.5,142.6)]
 pred=[(24.4,147.8),(28.3,149.1),(27.7,145.2)]
 for t,h,c in zip(truth,pred,[BLUE,TEAL,OCHRE]):
  p.arrow(origin,t,c,lw=1,scale=5.7)
  p.arrow(origin,h,c,ls=(0,(2.4,1.8)),lw=.9,scale=5.4)
  p.cv.plot(*t,'o',ms=2.7,mfc=c,mec=INK,mew=.35)
 p.text(28,134.4,'Fixed model basis',6,ha='center',color=GREY)
 p.key(4,127,'True',GREY);p.key(29,127,'Predicted',GREY,'--')

 models=['Scouter','TxPert-public']
 model_colors=[BLUE,TEAL]
 heading(p,'b','Training subspace overlap',61,172,55)
 a=p.ax(74,138,39,23,'b')
 alignment=ALIGN.query('model_basis=="train" and training_panel=="full" and biology_basis=="matched_train" and fold<2 and model in ["Scouter","TxPert-public"]')
 for m,c in zip(models,model_colors):
  d=alignment[alignment.model.eq(m)];fold_curve(a,d,'overlap',c)
  if m=='TxPert-public':a.lines[-1].set_marker('s')
 pair=pd.read_csv(DATA/'TRAIN_PAIRWISE_FOLD.csv')
 pair=pair.query('training_panel=="full" and model_i=="Scouter" and model_j=="TxPert-public" and fold<2').copy()
 assert len(pair)==8 and np.isclose(pair[pair.q.eq(8)].overlap.mean(),0.9484550421262856)
 pair_color='#665B72'
 fold_curve(a,pair,'overlap',pair_color,'--')
 a.lines[-1].set(marker='D',mfc='white',mec=pair_color,mew=.7)
 log_rows(p,a,pair,'direct_cross_model_basis_overlap_two_matched_folds')
 a.set(ylim=(.7,1.025),yticks=[.7,.8,.9,1],ylabel='Subspace overlap',xlabel='Dimensions, q')
 log_rows(p,a,alignment,'training_basis_alignment_q_scan_common_folds01')
 heading(p,'c','Held-out capacity',123,172,57)
 a=p.ax(139,138,39,23,'c')
 capacity=CAP.query('training_panel=="full" and fold<2 and basis_source in ["train_model_predictions","train_biology"] and model in ["Scouter","TxPert-public","Biology_train_matched"]')
 for m,c,ls,marker in [('Biology_train_matched',OCHRE,'--','D'),('Scouter',BLUE,'-','o'),('TxPert-public',TEAL,'-','s')]:
  d=capacity[capacity.model.eq(m)]
  for _,g in d.groupby('fold'):
   g=g.sort_values('q');a.plot(np.log2(g.q),g.held_true_energy,color=c,lw=.6,alpha=.28,ls=ls)
  g=d.groupby('q').held_true_energy.mean().sort_index()
  assert (g.index > 0).all()
  a.plot(np.log2(g.index),g.values,color=c,ls=ls,marker=marker,ms=4.8 if marker=='D' else 3.0,
    mfc='white' if marker=='D' else c,mec=c if marker=='D' else 'white',mew=.5,lw=1.1)
 a.set(xlim=(.85,4.15),xticks=[1,2,3,4],xticklabels=[2,4,8,16],ylim=(0,.82),yticks=[0,.4,.8],
   ylabel='Response energy captured',xlabel='Dimensions, q');clean(a)
 log_rows(p,a,capacity,'same_rank_capacity_q_scan_common_folds01')
 p.rows.append(['b','c'])
 for y,l,r in [(125,'Scouter vs measured','Scouter'),(120.9,'TxPert vs measured','TxPert'),(116.8,'Scouter vs TxPert','Measured basis')]:
  index={125:0,120.9:1,116.8:2}[y]
  mark=['o','s','D'][index];line=['-','-','--'][index]
  p.key(71,y,l,[BLUE,TEAL,pair_color][index],line,marker=mark)
  p.key(136,y,r,[BLUE,TEAL,OCHRE][index],line,marker=mark)

 heading(p,'d','Coordinate assignment across response dimensions',3,111,177)
 for j,m in enumerate(models):
  p.text(52.5+92*j,104.7,MODEL_LABEL[m],6.7,ha='center',weight='bold')
  d=capacity[capacity.model.eq(m)]
  for i,(metric,pred_col,true_col,ylim,ticks) in enumerate([
    ('Global ρ','predicted_coordinate_rho','oracle_global_rho',(0,1.05),[0,.5,1]),
    ('Top-10','predicted_coordinate_top10','oracle_top10',(0,.7),[0,.3,.6])]):
   x=18+92*j+43*i;pid=f'd{j*2+i}'
   a=p.ax(x,72,26,24,pid)
   fold_curve(a,d,pred_col,BLUE,'--');fold_curve(a,d,true_col,TEAL,'-')
   a.set(ylim=ylim,yticks=ticks)
   small_title(p,x,99,26,metric);log_rows(p,a,d,'predicted_vs_true_projection_q_scan_common_folds01')
  p.rows.append([f'd{j*2}',f'd{j*2+1}'])
 p.text(98,62,'Retained dimensions, q',6.3,ha='center')
 p.key(51,56,'Predicted',BLUE,'--');p.key(105,56,'True projection',TEAL)

 heading(p,'e','Generalization within the learned space',3,48,117)
 a=p.ax(18,15,38,24,'e0')
 d=SHIFT.query('training_panel=="full" and q==8 and model in ["Scouter","TxPert-public"]')
 for i,m in enumerate(models):
  fold_bar(a,i,d[d.model.eq(m)].fixed_basis_diversity_shift_normalized_to_biology,model_colors[i],.47)
 a.axhline(1,ls=(0,(3,2)),lw=.7,color=GREY)
 a.set(ylim=(0,1.6),yticks=[0,.5,1,1.5],xticks=[0,1],xticklabels=['Scouter','TxPert'],ylabel='Normalized variance');clean(a)
 small_title(p,18,42,38,'Unseen / training')
 log_rows(p,a,d,'composition_normalized_variance_all_complete_folds')
 a=p.ax(80,15,38,24,'e1')
 d=CAP.query('training_panel=="full" and q==8 and basis_source=="train_model_predictions" and model in ["Scouter","TxPert-public"]')
 for m,c,marker in zip(models,model_colors,['o','s']):
  g=d[d.model.eq(m)].sort_values('fold')
  vals=g[['train_predicted_coordinate_rho','predicted_coordinate_rho']].to_numpy()
  for v in vals:a.plot([0,1],v,color=c,alpha=.35,lw=.6,marker=marker,ms=2.7,mfc='white',mec=c,mew=.45)
  a.plot([0,1],vals.mean(axis=0),color=c,lw=1.2,marker=marker,ms=3.8,mec='white',mew=.45)
 a.set(xlim=(-.18,1.18),ylim=(0,1.05),yticks=[0,.5,1],xticks=[0,1],xticklabels=['Training','Unseen'],ylabel='Global ρ');clean(a)
 small_title(p,80,42,38,'Coordinate relations')
 log_rows(p,a,d,'seen_unseen_assignment_all_complete_folds')
 p.rows.append(['e0','e1'])

 heading(p,'f','Basis stability',126,48,54)
 a=p.ax(139,15,39,24,'f')
 stability=read('sub_SAMPLING_STABILITY_SUMMARY.csv')
 for m,c,marker in zip(models,model_colors,['o','s']):
  g=stability[(stability.model==m)&(stability.q==8)&(stability.kind=='random_subset_to_full')].sort_values('sample_size')
  assert (g.sample_size > 0).all()
  x=np.log2(g.sample_size)
  a.fill_between(x,g.overlap_p025,g.overlap_p975,color=c,alpha=.12,lw=0)
  a.plot(x,g.overlap_mean,color=c,lw=1.1,marker=marker,ms=3,mec='white',mew=.35)
  log_rows(p,a,g,'q8_training_subset_stability_saved_percentiles_not_CI')
 a.set(ylim=(0,1.05),yticks=[0,.5,1],xticks=np.log2([16,64,128,500]),xticklabels=[16,64,128,500],
   ylabel='Overlap with full basis',xlabel='Training interventions')
 a.tick_params(axis='x',labelsize=5.8);clean(a)
 p.save()

def main4_new():
 p=Page('Figure_4',195)
 heading(p,'a','Intervention information',3,189,177)
 cellular_input(p,27.7,175,OCHRE,'#F8F3ED')
 cellular_input(p,27.7,161,TEAL,'#F0F5F2',perturbed=True)
 p.text(42.7,175,'Natural expression',6.3);p.text(42.7,161,'Measured response',6.3)
 p.arrow((71.7,175),(84.7,169),OCHRE);p.arrow((71.7,161),(84.7,167),TEAL)
 p.text(99.7,171,'Matched map',6.5,ha='center',weight='bold');p.text(99.7,165,'K562 → RPE1',6,ha='center',color=GREY)
 p.arrow((114.7,168),(129.7,168),GREY)
 vector_plane(p,137.7,158,25,17)
 for tip,c,ls in [((150.7,173),TEAL,'-'),((155.7,164),OCHRE,'--'),((147.7,168),BLUE,'-')]:p.arrow((140.7,160),tip,c,ls=ls,scale=5)
 p.text(152.2,154,'Fixed response space',6,ha='center',color=GREY)

 heading(p,'b','Matched inputs (1,412 interventions)',3,148,177)
 methods=['Frozen_model','SharedRidge_Paired','K562_raw_corr','K562_resid_corr','K562_response']
 for i,(metric,xlim) in enumerate([('CA',(0,.85)),('rho',(0,.54)),('top10',(0,.25))]):
  x=46+48*i;a=p.ax(x,114,36,23,'b'+str(i));points(a,P1,methods,metric,xlim,i==0)
  small_title(p,x,140,36,METRIC[metric]);log_rows(p,a,P1[P1.method.isin(methods)],metric)
 p.rows.append(['b0','b1','b2'])
 heading(p,'c','Anchor assignment (1,432 interventions)',3,103.7,177)
 anchor=TX.query('experiment=="anchor" and budget==0 and q==8 and readout=="common"').copy()
 conditions=['anchor_coordinates','wrong_target_anchor']
 anchor=anchor.set_index('method').loc[conditions].reset_index()
 assert anchor.n_queries_total.eq(1432).all() and anchor.n_folds.eq(5).all()
 assert np.isclose(*anchor.coordinate_variance_retention) and np.allclose(anchor.rho,[.442926,-.002429],atol=1e-6)
 for i,(metric,title,limits,ticks) in enumerate([
    ('coordinate_variance_retention','Coordinate variance ratio',(0,.70),[0,.2,.4,.6]),
    ('rho','Global ρ',(-.045,.55),[0,.2,.4])]):
  x=45+74*i
  a=p.ax(x,84,48,10.5,'c'+str(i))
  points(a,anchor,conditions,metric,limits,i==0,
         colors={conditions[0]:TEAL,conditions[1]:GREY},labels=['Matched target','Reassigned target'])
  a.set_xticks(ticks)
  if metric=='rho':a.axvline(0,ls='--',color=GREY,lw=.55,zorder=1)
  small_title(p,x,97.5,48,title);log_rows(p,a,anchor,metric)
 p.rows.append(['c0','c1'])
 heading(p,'d','Measurement precision',3,73,177)
 for i,(metric,ylim,col) in enumerate([('rho',(0,.55),TEAL),('top10',(0,.25),TEAL),('coordinate_variance_retention',(0,1.9),OCHRE)]):
  x=19+59*i;a=p.ax(x,43,42,20,'d'+str(i));curve_ci(a,P2,metric,col);a.set_ylim(ylim)
  small_title(p,x,66.5,42,METRIC[metric]);log_rows(p,a,P2,metric)
 p.rows.append(['d0','d1','d2']);p.text(98,35,'Source cells per query',6.3,ha='center')
 heading(p,'e','Recovery within GEARS',3,28.5,177)
 methods=['Frozen_model','SharedRidge_Paired','K562_response'];d=P3[P3.q.eq(2)]
 for i,(metric,xlim) in enumerate([('CA',(0,.8)),('rho',(0,.54)),('top10',(0,.14))]):
  x=46+48*i;a=p.ax(x,6,36,14,'e'+str(i));points(a,d,methods,metric,xlim,i==0)
  small_title(p,x,22.8,36,METRIC[metric]);log_rows(p,a,d[d.method.isin(methods)],metric)
 p.rows.append(['e0','e1','e2']);p.save()
