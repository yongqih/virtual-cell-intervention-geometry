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
LABELS={'Frozen_model':'Frozen model','SharedRidge_Paired':'Shared calibration',
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





def supp33():
 p=Page('Supplementary_Figure_7',214)
 p.text(91.5,208,'Unseen-output diagnostics',7.1,ha='center',weight='bold')
 p.text(91.5,200,'q = 8, five folds',6.1,ha='center',color=GREY)
 models=['GEARS','scGPT','Scouter','CellFlow','PerturbDiff','State-ST-GenePT','TxPert-public']
 d=CAP.query('basis_source=="unseen_model_predictions" and q==8')
 al=ALIGN.query('model_basis=="unseen" and biology_basis=="Biology_train" and q==8')
 dim=read('sub_PHASE_A_DIMENSIONALITY.csv')
 specs=[('Biological alignment',al,'overlap',(0,1.05)),('Response energy',d,'held_true_energy',(0,.9)),
  ('Global ρ',d,'predicted_coordinate_rho',(0,1.08)),('Top-10',d,'predicted_coordinate_top10',(0,.7)),
  ('Effective rank',dim,'effective_rank',(0,14)),('Variance / truth',dim,'variance_retention_vs_held_truth',(0,1.5))]
 for k,(title,ds,col,ylim) in enumerate(specs):
  i,j=divmod(k,2);x=28+89*j;y=144-64*i;a=p.ax(x,y,61,40,chr(97+k))
  p.text(x-25,y+47,chr(97+k),9,weight='bold');small_title(p,x,y+47,61,title)
  for z,m in enumerate(models):
   vals=ds[ds.model.eq(m)][col];fold_bar(a,z,vals,BLUE,.62)
   if k in [2,3]:
    oc='oracle_global_rho' if k==2 else 'oracle_top10';od=ds[ds.model.eq(m)][oc]
    a.scatter(np.full(len(od),z)+np.linspace(-.15,.15,len(od)),od,s=9,facecolors=TEAL,edgecolors=INK,lw=.35,zorder=4)
  a.set(xticks=range(7),xticklabels=[MODEL_LABEL[m] for m in models],ylim=ylim)
  a.tick_params(axis='x',labelrotation=45,labelsize=5.7);plt.setp(a.get_xticklabels(),ha='right',rotation_mode='anchor');a.yaxis.set_major_locator(MaxNLocator(4));clean(a)
  log_rows(p,a,ds[ds.model.isin(models)],title)
 for i in range(3):p.rows.append([chr(97+i*2),chr(98+i*2)])
 p.key(27,5,'Predicted coordinates',BLUE);p.key(87,5,'True-projection coordinates',TEAL,marker='o');p.save()

def supp41():
 p=Page('Supplementary_Figure_8',224)
 p.panel('a','Matched input representations',3,217)
 methods=['Frozen_model','SharedRidge_Paired','SharedRidge_Original','Mean_coordinates','K562_raw_corr',
  'K562_resid_corr','K562_raw_cov','K562_resid_cov','RPE1_raw_corr','RPE1_resid_corr','EstablishedOBS71','K562_response','Oracle_coordinates_diagnostic']
 for i,(metric,xlim) in enumerate([('CA',(0,1)),('rho',(-.03,1.04)),('top10',(0,.6))]):
  x=49+46*i;a=p.ax(x,116,34,87,'a'+str(i));points(a,P1,methods,metric,xlim,i==0)
  a.tick_params(axis='y',labelsize=5.9);small_title(p,x,207,34,METRIC[metric]);log_rows(p,a,P1,metric)
 p.rows.append(['a0','a1','a2'])
 p.panel('b','Measured − natural input',3,104)
 d=read('P1_PRIMARY_SIMULTANEOUS_INTERVALS.csv')
 for i,metric in enumerate(['rho','top10']):
  x=45+76*i;a=p.ax(x,72,53,21,'b'+str(i));z=d[d.metric.eq(metric)]
  for j,c in enumerate(['K562_raw_corr','K562_resid_corr']):
   r=z[z.comparator.eq(c)].iloc[0];a.errorbar(r.delta,j,xerr=[[r.delta-r.simultaneous_low],[r.simultaneous_high-r.delta]],fmt='o',ms=4,color=TEAL,mec=INK,mew=.4,capsize=2)
  a.axvline(0,color=GREY,ls='--',lw=.6);a.set(ylim=(1.55,-.55),xlim=(-.02,.51) if metric=='rho' else (-.005,.12),yticks=[0,1],yticklabels=['Raw correlation','Residualized'] if i==0 else [])
  a.tick_params(axis='y',length=0);small_title(p,x,96,53,'Δρ' if metric=='rho' else 'ΔTop-10');clean(a,'x');log_rows(p,a,z,'primary_simultaneous_95')
 p.rows.append(['b0','b1'])
 p.panel('c','Coordinate prediction signal',3,59)
 d=read('P1_SEEN_CV_UNSEEN_SUMMARY.csv');d=d[d.phase.eq('P1')]
 methods=['K562_raw_corr','K562_resid_corr','RPE1_raw_corr','EstablishedOBS71','K562_response']
 a=p.ax(46,14,132,34,'c')
 for j,m in enumerate(methods):
  r=d[d.method.eq(m)].iloc[0]
  for k,col in enumerate(['inner_oof_MSE_reduction_vs_mean','unseen_MSE_reduction_vs_mean']):
   a.barh(j+(k-.5)*.3,r[col],height=.27,color=[LIGHTBLUE,TEAL][k],edgecolor=INK,lw=.45)
 a.set(ylim=(4.6,-.6),xlim=(0,.56),yticks=range(5),yticklabels=[LABELS[m] for m in methods],xlabel='Coordinate MSE reduction versus mean')
 a.xaxis.set_major_formatter(PercentFormatter(1));a.tick_params(axis='y',length=0);clean(a,'x')
 p.key(104,53,'Inner OOF',LIGHTBLUE);p.key(141,53,'Unseen',TEAL);log_rows(p,a,d[d.method.isin(methods)],'descriptive_MSE_reduction');p.save()

CAL_LABEL={'identity':'Frozen','mean_shift':'Mean shift','diagonal':'Diagonal','ridge_mse':'Ridge MSE',
 'ridge_geometry':'Ridge geometry','selected_geometry':'Selected geometry','constant_coordinates':'Mean coordinates',
 'train_ridge_mse':'Training Ridge','anchor_coordinates':'Response transfer','anchor_residual_update':'Residual update',
 'source_only_projection':'Source projection','wrong_target_anchor':'Wrong identity','calibrated_copy':'Copy + mean',
 'train_selected_geometry':'Training geometry'}
CAL_COL={'identity':GREY,'mean_shift':OCHRE,'diagonal':PALE,'ridge_mse':BLUE,'ridge_geometry':LIGHTBLUE,
 'selected_geometry':DARKTEAL,'constant_coordinates':'#D0D4D4','train_ridge_mse':BLUE,
 'anchor_coordinates':TEAL,'anchor_residual_update':LIGHTBLUE,'source_only_projection':BLUE,
 'wrong_target_anchor':'#D0D4D4','calibrated_copy':OCHRE,'train_selected_geometry':LIGHTBLUE}
def supp42():
 p=Page('Supplementary_Figure_9',225)
 p.panel('a','Shared calibration',3,218)
 methods=['identity','mean_shift','diagonal','ridge_mse','ridge_geometry','selected_geometry','constant_coordinates']
 d=TX.query('experiment=="train" and q==8 and budget==0 and readout=="full757"')
 for i,(metric,xlim) in enumerate([('ca',(0,.8)),('rho',(0,.34)),('top10',(0,.18))]):
  x=42+48*i;a=p.ax(x,168,38,40,'a'+str(i));points(a,d,methods,metric,xlim,i==0,CAL_COL,[CAL_LABEL[m] for m in methods]);small_title(p,x,211,38,METRIC[metric]);log_rows(p,a,d[d.method.isin(methods)],metric)
 p.rows.append(['a0','a1','a2'])
 p.panel('b','Target-context references',3,154)
 for i,(metric,ylim) in enumerate([('ca',(.5,.72)),('rho',(.07,.31)),('top10',(.1,.25))]):
  x=19+59*i;a=p.ax(x,111,42,31,'b'+str(i))
  for method,col,ls in [('identity',GREY,'--'),('held_ridge_mse',OCHRE,'-'),('updated_ridge_mse',TEAL,'-'),('updated_selected_geometry',BLUE,'-')]:
   d=TX[(TX.experiment=='crossfit')&(TX.q==8)&(TX.readout=='full757')&(TX.method==method)].sort_values('budget')
   assert (d.budget>0).all()
   xx=np.log2(d.budget);a.fill_between(xx,d[metric+'_ci_low'],d[metric+'_ci_high'],color=col,alpha=.09,lw=0)
   a.plot(xx,d[metric],marker='o',ms=2.7,color=col,ls=ls,lw=.9);log_rows(p,a,d,metric)
  a.set(xticks=[5,6,7],xticklabels=[32,64,128],ylim=ylim);small_title(p,x,145,42,METRIC[metric]);clean(a)
 p.rows.append(['b0','b1','b2'])
 p.text(97,101,'Calibration identities (cross-fit query blocks)',6.2,ha='center')
 p.key(6,94,'Frozen',GREY,'--');p.key(39,94,'Held-only Ridge',OCHRE);p.key(94,94,'Updated Ridge',TEAL);p.key(140,94,'Geometry',BLUE)
 p.panel('c','Cross-context controls',3,82)
 methods=['identity','train_ridge_mse','anchor_coordinates','anchor_residual_update','source_only_projection','wrong_target_anchor','calibrated_copy']
 d=TX.query('experiment=="anchor" and q==8 and budget==0 and readout=="common"')
 for i,(metric,xlim) in enumerate([('ca',(0,.85)),('rho',(-.06,.6)),('top10',(0,.34))]):
  x=42+48*i;a=p.ax(x,40,38,32,'c'+str(i));points(a,d,methods,metric,xlim,i==0,CAL_COL,[CAL_LABEL[m] for m in methods]);small_title(p,x,75,38,METRIC[metric]);log_rows(p,a,d[d.method.isin(methods)],metric)
 p.rows.append(['c0','c1','c2'])
 p.panel('d','Scouter replication',3,28)
 methods=['identity','train_ridge_mse','anchor_coordinates'];d=SC.query('experiment=="anchor" and q==8 and budget==0 and readout=="common" and mode=="train"')
 for i,(metric,xlim) in enumerate([('ca',(0,.85)),('rho',(-.06,.6)),('top10',(0,.34))]):
  x=42+48*i;a=p.ax(x,6,38,16,'d'+str(i));points(a,d,methods,metric,xlim,i==0,CAL_COL,[CAL_LABEL[m] for m in methods]);log_rows(p,a,d[d.method.isin(methods)],metric)
 p.rows.append(['d0','d1','d2']);p.save()

def supp43():
 p=Page('Supplementary_Figure_10',224)
 p.panel('a','Query precision',3,217)
 tech=read('P2_TECHNICAL_REPEAT_METRICS.csv')
 specs=[('rho',(0,.55)),('top10',(0,.25)),('CA',(.6,.8)),('coordinate_mse',(0,3.7)),('coordinate_variance_retention',(0,1.9)),('response_variance_retention',(0,1.4))]
 for k,(metric,ylim) in enumerate(specs):
  i,j=divmod(k,3);x=19+59*j;y=165-48*i;a=p.ax(x,y,42,31,'a'+str(k))
  for n,c in enumerate([5,15,30]):
   z=tech[tech.cells.eq(c)];a.scatter(n+np.linspace(-.1,.1,len(z)),z[metric],s=6,color=GREY,alpha=.5,zorder=2)
  curve_ci(a,P2,metric,TEAL if k<3 else OCHRE);a.set_ylim(ylim);small_title(p,x,y+34,42,METRIC[metric]);log_rows(p,a,P2,metric)
 for i in range(2):p.rows.append(['a'+str(i*3+j) for j in range(3)])
 p.text(98,107,'Source cells per query intervention',6.1,ha='center')
 p.panel('b','GEARS response space',3,96)
 d=read('P3_REPRESENTATIONAL_CAPACITY.csv');a=p.ax(20,55,42,30,'b0')
 for i,q in enumerate([2,8]):
  z=d[d.q.eq(q)];fold_bar(a,i-.16,z.basis_energy,BLUE,.29);fold_bar(a,i+.16,z.biological_same_rank_energy,OCHRE,.29)
 a.set(ylim=(0,.85),xticks=[0,1],xticklabels=['q = 2','q = 8'],ylabel='Response energy');clean(a);log_rows(p,a,d,'capacity')
 p.key(17,88,'GEARS',BLUE);p.key(39,88,'Biological',OCHRE)
 methods=['Frozen_model','SharedRidge_Paired','K562_raw_corr','K562_resid_corr','K562_response']
 for j,(metric,ylim) in enumerate([('rho',(-.02,.55)),('top10',(0,.26))]):
  x=79+59*j;a=p.ax(x,55,42,30,'b'+str(j+1))
  for m in methods:
    z=P3[P3.method.eq(m)].sort_values('q');xx=np.arange(len(z))
    a.errorbar(xx,z[metric],yerr=[z[metric]-z[metric+'_ci_low'],z[metric+'_ci_high']-z[metric]],fmt='o',linestyle='--' if m=='Frozen_model' else '-',ms=2.5,color=COLORS[m],capsize=1,lw=.85)
  a.set(xticks=[0,1],xticklabels=['q = 2','q = 8'],xlim=(-.17,1.17),ylim=ylim);small_title(p,x,88,42,METRIC[metric]);clean(a);log_rows(p,a,P3[P3.method.isin(methods)],metric)
 p.rows.append(['b0','b1','b2'])
 for x,m,label in [(6,'Frozen_model','Frozen'),(35,'SharedRidge_Paired','Shared'),(65,'K562_raw_corr','Natural'),(103,'K562_resid_corr','Residualized'),(147,'K562_response','Measured')]:
  p.key(x,46,label,COLORS[m],'--' if m=='Frozen_model' else '-')
 p.panel('c','GEARS paired improvements',3,41)
 d=read('P3_PRIMARY_SIMULTANEOUS_INTERVALS.csv')
 comps=[('SharedRidge_Paired','Frozen_model'),('K562_response','Frozen_model'),('K562_response','SharedRidge_Paired')]
 for i,metric in enumerate(['CA','rho','top10']):
  x=45+47*i;a=p.ax(x,9,36,23,'c'+str(i));z=d[d.metric.eq(metric)]
  for j,(m,c) in enumerate(comps):
   r=z[(z.method==m)&(z.comparator==c)].iloc[0];a.errorbar(r.delta,j,xerr=[[r.delta-r.simultaneous_low],[r.simultaneous_high-r.delta]],fmt='o',color=LIGHTBLUE if j==0 else TEAL,ms=3.5,mec=INK,mew=.35,capsize=2)
  a.axvline(0,color=GREY,ls='--',lw=.6);a.set(ylim=(2.55,-.55),yticks=range(3),yticklabels=['Shared − frozen','Measured − frozen','Measured − shared'] if i==0 else [])
  a.tick_params(axis='y',length=0);a.xaxis.set_major_locator(MaxNLocator(3));small_title(p,x,35,36,'Δ'+METRIC[metric].replace('Global ',''));clean(a,'x');log_rows(p,a,z,'primary_simultaneous_95')
 p.rows.append(['c0','c1','c2']);p.save()


"""Compact figures from frozen source tables. Schematics are illustrative only."""

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



def supp6():
 p=Page('Supplementary_Figure_6',225)
 p.key(10,219,'Model / predicted',BLUE);p.key(67,219,'True projection',TEAL);p.key(120,219,'Biological basis',OCHRE)
 models=['GEARS','Scouter','TxPert-public']
 for j,m in enumerate(models):
  d=CAP[(CAP.basis_source=='train_model_predictions')&(CAP.training_panel=='full')&(CAP.model==m)]
  bio=CAP[(CAP.basis_source=='train_biology')&(CAP.training_panel=='full')&CAP.fold.isin(d.fold.unique())]
  for i,(title,cols,ylim) in enumerate([('Response energy',['held_true_energy'],(0,1)),('Global ρ',['predicted_coordinate_rho','oracle_global_rho'],(0,1.08)),('Top-10',['predicted_coordinate_top10','oracle_top10'],(0,.85))]):
   x=19+59*j;y=178-38*i;a=p.ax(x,y,42,24,f'{i}{j}')
   if i==0:fold_curve(a,d,cols[0],BLUE);fold_curve(a,bio,cols[0],OCHRE)
   else:fold_curve(a,d,cols[0],BLUE);fold_curve(a,d,cols[1],TEAL)
   a.set_ylim(ylim);a.yaxis.set_major_locator(MaxNLocator(3))
   if j==0:a.set_ylabel(title)
   if i==2:a.set_xlabel('Retained dimensions, q')
   if i==0:small_title(p,x,207,42,MODEL_LABEL[m])
   p.text(x-15,y+27,chr(97+i*3+j),8.5,weight='bold');log_rows(p,a,d,title)
  p.cols.append([f'{i}{j}' for i in range(3)])
 for i in range(3):p.rows.append([f'{i}{j}' for j in range(3)])
 d=read('sub_SAMPLING_STABILITY_SUMMARY.csv')
 for j,m in enumerate(models):
  x=19+59*j;a=p.ax(x,60,42,24,'s'+str(j));g=d[(d.model==m)&(d.q==8)&(d.kind=='random_subset_to_full')].sort_values('sample_size')
  a.fill_between(np.log2(g.sample_size),g.overlap_p025,g.overlap_p975,color=BLUE,alpha=.18,lw=0)
  a.plot(np.log2(g.sample_size),g.overlap_mean,'o-',color=BLUE,ms=3,mec=INK,mew=.35)
  a.set(ylim=(0,1.05),yticks=[0,.5,1],xticks=np.log2([16,64,128,500]),xticklabels=[16,64,128,500],xlabel='Training identities')
  if j==0:a.set_ylabel('Basis stability')
  p.text(x-15,87,chr(106+j),8.5,weight='bold');clean(a);log_rows(p,a,g,'sampling_percentile_range_not_CI')
 p.rows.append(['s0','s1','s2'])
 heading(p,'m','Training-output coverage',3,43,177)
 av=read('sub_TRAIN_PREDICTION_AVAILABILITY.csv')
 ms=['GEARS','scGPT','Scouter','CellFlow','PerturbDiff','State-ST-GenePT','TxPert-public']
 a=p.ax(34,6,74,31,'coverage');mat=av.pivot(index='model',columns='fold',values='usable_training_vectors').loc[ms]
 cm=LinearSegmentedColormap.from_list('coverage',['#FFFFFF',LIGHTBLUE,TEAL])
 a.imshow(mat.values,aspect='auto',cmap=cm,vmin=0,vmax=1250,interpolation='nearest')
 a.set(xticks=range(5),xticklabels=range(5),yticks=range(7),yticklabels=[MODEL_LABEL[m] for m in ms]);a.tick_params(length=0,labelsize=5.7);a.spines[['left','bottom']].set_visible(False)
 for yy in range(7):
  for xx in range(5):a.text(xx,yy,str(int(mat.iloc[yy,xx])),ha='center',va='center',fontsize=5.7,color=INK)
 p.text(113,31,'Complete output folds',6.2,weight='bold');p.text(113,24,'GEARS 3, Scouter 2, TxPert 5',6)
 p.text(71,2,'Fold',6,ha='center')
 log_rows(p,a,av,'coverage');p.save()
