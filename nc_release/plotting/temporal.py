"""Render frozen Fig5 evidence; no fitting, resampling or metric estimation."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Ellipse, PathPatch, Polygon
from matplotlib.path import Path as MP

from settings import INPUT, OUTPUT as OUT, CHECKS as LAYOUT
from layout import record_panel_positions
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':6.5,'axes.labelsize':6.5,'axes.titlesize':7,'xtick.labelsize':6,'ytick.labelsize':6.3,
 'axes.linewidth':.55,'axes.spines.top':False,'axes.spines.right':False,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','legend.frameon':False,
 'figure.facecolor':'white','savefig.facecolor':'white','lines.linewidth':1.0})
BLUE='#7B9CAF'; OCHRE='#C99A70'; TEAL='#86A79B'; GRAY='#B9BDC2'; INK='#222222'; PALE='#EAF1F4'
files={}; rows=[]
def read(folder,name):
 p=INPUT/folder/f'{name}.csv'
 files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return pd.read_csv(p)
FOLDER='protein'
G=read(FOLDER,'grouped_geometry_summary'); P=read(FOLDER,'paired_comparisons')
E=read(FOLDER,'lopo_representation_summary'); O=read(FOLDER,'temporal_overlap_distribution')
N=read(FOLDER,'neighborhood_summary'); GAP=read(FOLDER,'entry_gap_closure')
I=read(FOLDER,'single_drug_influence')
R='renge'
RP=read(R,'paired_contrasts'); RH=read(R,'historical_corrected_bootstrap')
RS=read(R,'repeated_fourfold_contrasts'); RN=read(R,'temporal_noise_coupling_summary')
def one(df,**kw):
 d=df
 for k,v in kw.items(): d=d[d[k]==v]
 assert len(d)==1,(kw,len(d))
 row=d.iloc[0]; rows.append(row.to_dict()); return row
def geo(model,cell='cell_average',early=24,late=48,metric='cosine_geometry'):
 return one(G,model=model,cell=cell,early=early,late=late,metric=metric)
def pair(a,b,cell='cell_average',early=24,late=48,metric='cosine_geometry'):
 return one(P,model_a=a,model_b=b,cell=cell,early=early,late=late,metric=metric)
def point(ax,r,y,c=BLUE,delta=False,scale=1,marker='o'):
 k='delta' if delta else 'estimate'; x=float(r[k])*scale
 ax.errorbar(x,y,xerr=[[x-float(r.ci_low)*scale],[float(r.ci_high)*scale-x]],fmt=marker,
  color=c,mec=c,mfc='white',ms=4,lw=1.1,capsize=2,capthick=.8,zorder=3)
def clean(ax,xlabel,yticks,labels,xlim=None,zero=True):
 ax.set_yticks(yticks,labels); ax.set_xlabel(xlabel,labelpad=4)
 ax.tick_params(length=2.5,width=.55,pad=2)
 if xlim: ax.set_xlim(*xlim)
 if zero: ax.axvline(0,c=GRAY,lw=.65,ls='--',zorder=0)
 ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
class Sheet:
 def __init__(self,h):
  self.w=183; self.h=h; self.fig=plt.figure(figsize=(183/25.4,h/25.4)); self.axes=[];self.ids=[]
  self.cv=self.fig.add_axes([0,0,1,1],label='canvas');self.cv.set(xlim=(0,183),ylim=(0,h));self.cv.axis('off')
 def t(self,x,y,s,size=6.5,ha='center',**kw):
  assert '·' not in s
  return self.cv.text(x,y,s,fontsize=size,ha=ha,va='center',color=INK,**kw)
 def title(self,letter,title,x,y,w):
  self.t(x,y,letter,9,'left',fontweight='bold');self.t(x+w/2,y,title,7,fontweight='bold')
 def ax(self,x,y,w,h,id):
  a=self.fig.add_axes([x/183,y/self.h,w/183,h/self.h],label=id)
  self.axes.append(a);self.ids.append(id);return a
 def save(self,name,groups):
  self.fig.canvas.draw()
  record_panel_positions(self.fig,axes=self.axes,panel_ids=self.ids,row_groups=groups,
   json_out=LAYOUT/f'{name}.alignment.json',overlay_svg=LAYOUT/f'{name}.alignment.svg',strict=True)
  self.fig.savefig(OUT/f'{name}.pdf')
  self.fig.savefig(OUT/f'{name}.svg')
  self.fig.savefig(OUT/f'{name}.png',dpi=300)
  plt.close(self.fig)
def arrow(s,start,end,c=INK,dashed=False,curve=0):
 s.cv.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=6,lw=.85,color=c,
  linestyle='--' if dashed else '-',connectionstyle=f'arc3,rad={curve}'))
def vector_fan(s,x,y,r,angles,c,dashed=False):
 for deg in angles:
  a=np.deg2rad(deg);arrow(s,(x,y),(x+r*np.cos(a),y+r*np.sin(a)),c,dashed)
def supplement11():
 s=Sheet(127)
 s.title('a','Temporal subspace overlap',3,122,86)
 a=s.ax(23,65,64,46,'a')
 for cell,c,mk in zip(['HCC1143','HCC1806','MDA-MB-453-1'],[BLUE,OCHRE,TEAL],['o','s','^']):
  d=O[(O.cell==cell)&(O.early==24)&(O.late==48)&(O.metric=='overlap')].sort_values('q')
  a.plot(d.q,d['median'],marker=mk,color=c,ms=3,label=cell)
  a.fill_between(d.q,d.p05,d.p95,color=c,alpha=.10,lw=0)
 a.set(xlabel='Dimensions',ylabel='Subspace overlap',xticks=[2,4,8],ylim=(0,.55));a.legend(fontsize=5.5,loc='upper left')
 s.title('b','Held-response capacity',95,122,85)
 a=s.ax(112,65,65,46,'b')
 for model,c in [('early_basis',BLUE),('late_basis',OCHRE)]:
  d=E[(E.cell=='cell_average')&(E.early==24)&(E.late==48)&(E.metric=='full_panel_energy_represented')&(E.model==model)].sort_values('q')
  a.errorbar(d.q,d.estimate*100,yerr=[(d.estimate-d.ci_low)*100,(d.ci_high-d.estimate)*100],fmt='o-',color=c,ms=3,capsize=2,label='24 h basis' if model=='early_basis' else '48 h basis')
 a.set(xlabel='Dimensions',ylabel='Held energy (%)',xticks=[2,4,8],ylim=(0,35));a.legend(loc='upper left',fontsize=6)
 s.title('c','Projected late relationships',3,48,86)
 a=s.ax(32,14,55,23,'c')
 for y,m,c in [(1,'oracle_early_q8',BLUE),(0,'oracle_late_q8',OCHRE)]:point(a,geo(m),y,c)
 clean(a,'Global ρ',[1,0],['24 h basis','48 h basis'],(0,1));a.set_ylim(-.7,1.7)
 s.title('d','Representation and assignment',95,48,85)
 a=s.ax(128,14,49,23,'d')
 for y,m,c in [(1,'oracle_shared',TEAL),(0,'direct_metadata_late_projected',GRAY)]:point(a,geo(m),y,c)
 clean(a,'Global ρ',[1,0],['True projection','Metadata'],(0,1));a.set_ylim(-.7,1.7)
 s.save('Supplementary_Figure_11',[['a','b'],['c','d']])
def supplement12():
 s=Sheet(212)
 titles=['Measurement time','Readout and reference','Local neighbours','Recoverable error gap','Leave-one-drug-out influence','Temporal baseline contrasts']
 for k,title in enumerate(titles):s.title(chr(97+k),title,3 if k%2==0 else 95,207-68*(k//2),86)
 a=s.ax(32,158,55,38,'a')
 for idx,(early,late) in enumerate([(6,24),(6,48),(24,48)]):
  for m,dy,c in [('ode',.12,BLUE),('ridge',-.12,OCHRE)]:point(a,pair(m+'_measured',m+'_predicted',early=early,late=late),2-idx+dy,c,True)
 clean(a,'Δρ',[2,1,0],['6→24 h','6→48 h','24→48 h'],(-.5,.55));a.set_ylim(-.5,2.5)
 a=s.ax(129,158,48,38,'b')
 for idx,metric in enumerate(['cosine_geometry','observed_cosine_geometry','common_early_reference_cosine_geometry']):
  for m,dy,c in [('ode',.12,BLUE),('ridge',-.12,OCHRE)]:point(a,pair(m+'_measured',m+'_predicted',metric=metric),2-idx+dy,c,True)
 clean(a,'Δρ',[2,1,0],['Primary','Observed proteins','Common origin'],(-.05,.55));a.set_ylim(-.5,2.5)
 a=s.ax(32,90,55,38,'c')
 methods=['ode_predicted','ode_measured','ridge_predicted','ridge_measured','persistence','full_ridge']
 for y,m,c in zip(range(5,-1,-1),methods,[GRAY,BLUE,GRAY,OCHRE,TEAL,INK]):
  point(a,one(N,cell='cell_average',early=24,late=48,model=m,metric='neighbor_recall3'),y,c)
 clean(a,'Top-3 overlap',range(5,-1,-1),['ODE predicted','ODE measured','Ridge predicted','Ridge measured','Persistence','Full Ridge'],(0,.8));a.set_ylim(-.5,5.5)
 a=s.ax(129,90,48,38,'d')
 for y,m,c in [(1,'ode',BLUE),(0,'ridge',OCHRE)]:
  point(a,one(GAP,cell='cell_average',early=24,late=48,method=m,readout='full_panel',metric='recoverable_gap_closed'),y,c,scale=100)
 clean(a,'MSE gap closed (%)',[1,0],['ODE','Ridge'],(0,65));a.set_ylim(-.6,1.6);a.axvline(50,color=GRAY,ls=':',lw=.6)
 a=s.ax(23,20,64,40,'e')
 for m,c in [('ode',BLUE),('ridge',OCHRE)]:
  d=I[(I.cell=='cell_average')&(I.early==24)&(I.late==48)&(I.method==m)].sort_values('drug_index')
  assert len(d)==48
  a.plot(np.arange(1,49),d.delta,marker='o',ms=2,lw=.6,color=c,label=m.upper() if m=='ode' else 'Ridge')
 a.set(xlabel='Omitted drug',ylabel='Δρ',ylim=(0,.5),xticks=[1,24,48]);a.legend(loc='upper right',fontsize=6)
 a=s.ax(129,20,48,40,'f')
 for i,(base,label) in enumerate([('persistence','Persistence'),('direct_metadata_late','Metadata'),('full_ridge','Full Ridge'),('wrong','Wrong identity')]):
  for m,dy,c in [('ode',.12,BLUE),('ridge',-.12,OCHRE)]:point(a,pair(m+'_measured',m+'_wrong' if base=='wrong' else base),3-i+dy,c,True)
 clean(a,'Δρ vs baseline',[3,2,1,0],['Persistence','Metadata','Full Ridge','Wrong identity'],(-.25,.85));a.set_ylim(-.5,3.5);a.set_xticks([0,.4,.8])
 s.save('Supplementary_Figure_12',[['a','b'],['c','d'],['e','f']])
def supplement13():
 s=Sheet(198)
 s.title('a','Historical entry comparisons',3,193,86)
 a=s.ax(34,144,53,38,'a')
 for y,b,c in [(3,'PredictedEntry',BLUE),(2,'Persistence',TEAL),(1,'Ridge',OCHRE),(0,'SelectedBaseline',GRAY)]:point(a,one(RH,left='TrueEntry',right=b),y,c,True)
 clean(a,'Measured-entry Δρ',[3,2,1,0],['Predicted entry','Persistence','Ridge','Selected baseline'],(-.25,.55));a.set_ylim(-.6,3.6)
 s.title('b','Fixed-fold comparisons',95,193,85)
 a=s.ax(131,144,46,38,'b')
 contrasts=['CorrectLag_vs_Static','True_vs_PredictedEntry','Teacher_vs_Free','PRIMARY_T2_Original_vs_Selected']
 for y,c,co in zip(range(3,-1,-1),contrasts,[GRAY,BLUE,OCHRE,TEAL]):point(a,one(RP,contrast=c,readout='strict_trans',metric='geometry_delta'),y,co,True)
 clean(a,'Δρ',range(3,-1,-1),['Lag vs static','Measured entry','Teacher forcing','Selected baseline'],(-.85,.85));a.set_ylim(-.6,3.6);a.set_xticks([-.5,0,.5])
 s.title('c','Split sensitivity',3,125,86)
 a=s.ax(22,76,65,38,'c')
 a.axhline(0,c=GRAY,lw=.65,ls='--');a.plot(np.arange(1,len(RS)+1),RS.True_vs_PredictedEntry,'o',color=BLUE,ms=3)
 a.set(xlabel='Four-fold assignment',ylabel='Measured-entry Δρ',xticks=[1,10,20],ylim=(-.2,.65))
 s.title('d','Shared-measurement coupling',95,125,85)
 a=s.ax(119,76,58,38,'d')
 d=RN.set_index('pair').loc[['W23_W34','W34_W45']]
 for j,(col,c,label) in enumerate([('same_half_cosine',GRAY,'Same halves'),('cross_half_cosine',BLUE,'Independent halves')]):
  a.bar(np.arange(2)+(j-.5)*.27,d[col],width=.25,color=c,label=label)
 a.axhline(0,c=INK,lw=.6);a.set(xticks=[0,1],xticklabels=['W23 / W34','W34 / W45'],ylabel='Wave cosine',ylim=(-.5,.2));a.legend(loc='upper left',fontsize=5.8)
 for col,(metric,title,xlab) in enumerate([('mse_delta','Endpoint response recovery','MSE reduction (×0.001)'),('geometry_delta','Endpoint relationship recovery','Δρ vs persistence')]):
  s.title(chr(101+col),title,3 if col==0 else 95,57,86)
  a=s.ax(33 if col==0 else 127,20,54 if col==0 else 50,25,chr(101+col))
  for y,contrast,c in [(1,'T2_Endpoint_Original_vs_LastState',BLUE),(0,'T2_Endpoint_ResidualRidge_vs_LastState',OCHRE)]:
   r=one(RP,contrast=contrast,readout='strict_trans',metric=metric).copy()
   if metric=='mse_delta':r['delta'],r['ci_low'],r['ci_high']=-r['delta'],-r['ci_high'],-r['ci_low']
   point(a,r,y,c,True,scale=1000 if metric=='mse_delta' else 1)
  clean(a,xlab,[1,0],['Original','Residual Ridge'],(0,2) if metric=='mse_delta' else (-.65,.65));a.set_ylim(-.6,1.6)
 s.save('Supplementary_Figure_13',[['a','b'],['c','d'],['e','f']])
