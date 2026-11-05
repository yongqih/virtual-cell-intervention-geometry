"""Publication figures from frozen response summaries and cached-axis projections."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, ScalarFormatter

from settings import INPUT, OUTPUT as OUT, CHECKS as LAYOUT
DATA=INPUT/'fig12'
from layout import record_panel_positions
mpl.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':6.6,'axes.labelsize':6.5,'axes.titlesize':7.0,
    'xtick.labelsize':6.0,'ytick.labelsize':6.0,'legend.fontsize':6.2,
    'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'legend.frameon':False,'lines.linewidth':1.1,
    'figure.facecolor':'white','savefig.facecolor':'white',
})
MODELS=['GEARS','scGPT','Scouter','CellFlow','PerturbDiff','State','TxPert']
ORDER=['Observed']+MODELS
COLORS={'Observed':'#343434','GEARS':'#7B9CAF','scGPT':'#BCD0D9',
        'Scouter':'#C99A70','CellFlow':'#B9BDC2','PerturbDiff':'#E1CFB9',
        'State':'#C8BFB2','TxPert':'#86A79B'}
MARKERS=dict(zip(MODELS,['o','s','D','^','v','P','X']))
INK='#242424'; SOFT='#73777B'; GRID='#E2E4E6'
DD=pd.read_csv(DATA/'display_metrics_by_fold.csv')
SPEC=pd.read_csv(DATA/'spectra.csv')
SPEC['model_display']=SPEC.model.replace({'Biology_unseen':'Observed','State-ST-GenePT':'State','TxPert-public':'TxPert'})
XY=pd.read_csv(DATA/'training_PC_display_coordinates.csv')
MEANS=DD.groupby('model_display',sort=False).mean(numeric_only=True)
FOLD_JITTER=np.array([-.18,-.09,0,.09,.18])

class Page:
    def __init__(self,name,height):
        self.name=name;self.w=183.;self.h=float(height)
        self.fig=plt.figure(figsize=(self.w/25.4,self.h/25.4))
        self.cv=self.fig.add_axes([0,0,1,1],label='page_canvas')
        self.cv.set(xlim=(0,self.w),ylim=(0,self.h));self.cv.axis('off')
        self.axes=[];self.ids=[];self.rows=[];self.cols=[]
    def text(self,x,y,s,size=6.6,color=INK,weight='normal',ha='left',**kw):
        return self.cv.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va='center',**kw)
    def panel(self,letter,title,x,y):
        self.text(x,y,letter,9,weight='bold')
        width=86 if x<90 else 85
        if self.name=='Supplementary_Figure_4' and letter in 'abcdefgh': width=43
        if self.name=='Figure_2' and letter=='a': width=100
        self.text(x+width/2,y,title,7.1,weight='bold',ha='center')
    def ax(self,x,y,w,h,id):
        ax=self.fig.add_axes([x/self.w,y/self.h,w/self.w,h/self.h],label=id)
        ax.tick_params(length=2.1,width=.55,pad=2)
        self.axes.append(ax);self.ids.append(id);return ax
    def line_key(self,x,y,label,color,ls='-'):
        self.cv.plot([x,x+4],[y,y],color=color,lw=1,ls=ls)
        self.text(x+5.5,y,label,6.2)
    def save(self,crops=None):
        self.fig.canvas.draw()
        layout=record_panel_positions(self.fig,axes=self.axes,panel_ids=self.ids,
            row_groups=self.rows,column_groups=self.cols,
            json_out=LAYOUT/f'{self.name}.alignment.json',overlay_svg=LAYOUT/f'{self.name}.alignment.svg',
            tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
        base=OUT/self.name
        self.fig.savefig(base.with_suffix('.pdf'),dpi=600);self.fig.savefig(base.with_suffix('.svg'),dpi=600)
        self.fig.savefig(base.with_suffix('.png'),dpi=300)
        for label,(x0,y0,x1,y1) in (crops or {}).items():
            bbox=mpl.transforms.Bbox.from_extents(x0/25.4,y0/25.4,x1/25.4,y1/25.4)
            self.fig.savefig(LAYOUT/f'{self.name}_{label}.png',dpi=300,bbox_inches=bbox)
        print(json.dumps({'figure':self.name,'size_mm':[self.w,self.h],
                          'axes':len(self.axes),'alignment':layout['status']}),flush=True)
        plt.close(self.fig)

def bars(ax,models,metric,ylim,ylabel,ref=None):
    for i,model in enumerate(models):
        vals=DD[DD.model_display.eq(model)].sort_values('fold')[metric].to_numpy()
        assert len(vals)==5 and np.isfinite(vals).all()
        ax.bar(i,vals.mean(),width=.65,color=COLORS[model],edgecolor=INK,linewidth=.65,zorder=2)
        ax.scatter(i+FOLD_JITTER,vals,s=10,color='#454545',alpha=.65,edgecolor='none',zorder=4)
    if ref is not None:
        ax.axhline(ref,color=SOFT,lw=.75,ls=(0,(3,2)),zorder=1)
    ax.set(xlim=(-.65,len(models)-.35),ylim=ylim,xticks=np.arange(len(models)),xticklabels=models,ylabel=ylabel)
    for label in ax.get_xticklabels():
        label.set_rotation(60);label.set_ha('right');label.set_rotation_mode('anchor')
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_axisbelow(True);ax.grid(axis='y',color=GRID,lw=.4,ls=(0,(2,3)))

def curves(ax,models,max_pc=40,log=False,fold_lines=False):
    for model in models:
        mat=SPEC[SPEC.model_display.eq(model)].pivot(index='pc',columns='fold',values='cumulative_fraction').sort_index()
        # All plotted ranks are supported in every fold; complete raw spectra remain in source data.
        if log:
            x=np.arange(1,int(SPEC[SPEC.model_display.eq(model)].pc.max())+1)
            mat=mat.reindex(x).fillna(1.0)
        else:
            mat=mat.loc[mat.index<=max_pc]
        assert np.all(np.diff(mat.to_numpy(),axis=0)>=-1e-12)
        x=mat.index.to_numpy();mu=mat.mean(axis=1).to_numpy()
        ls='-' if model=='Observed' else (0,(3.1,1.7))
        if fold_lines:
            for fold in range(5):
                ax.plot(x,mat[fold],color=COLORS[model],lw=.45,alpha=.25,ls=ls,zorder=1)
        else:
            ax.fill_between(x,mat.min(axis=1),mat.max(axis=1),color=COLORS[model],alpha=.09,lw=0,zorder=1)
        ax.plot(x,mu,color=COLORS[model],ls=ls,lw=1.25 if model=='Observed' else 1.2,zorder=3)
    ax.axhline(.8,color=SOFT,lw=.6,ls=(0,(1,2)),zorder=0)
    if log:
        assert np.all(x > 0), 'Logarithmic component indices must be positive'
        ax.set_xscale('log');ax.set_xlim(1,400);ax.set_xticks([1,10,100]);ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(which='minor',bottom=False)
    else:
        ax.set_xlim(0,max_pc);ax.set_xticks([0,10,20,30,40])
    ax.set_ylim(0,1.035);ax.set_yticks([0,.5,1.0])

def limits(frame):
    out=[]
    for col in ['PC1','PC2']:
        lo,hi=frame[col].min(),frame[col].max();pad=.07*(hi-lo)
        out.append((float(lo-pad),float(hi+pad)))
    return out

def cloud(ax,frame,model,lims,size=3.5):
    sub=frame[frame.model_display.eq(model)]
    ax.scatter(sub.PC1,sub.PC2,s=size,c=COLORS[model],alpha=.65 if model=='Observed' else .82,
               edgecolor='none',rasterized=True,zorder=3)
    ax.axhline(0,color=GRID,lw=.5,zorder=0);ax.axvline(0,color=GRID,lw=.5,zorder=0)
    ax.set(xlim=lims[0],ylim=lims[1])
    ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3))
    assert ((sub.PC1>=lims[0][0])&(sub.PC1<=lims[0][1])&(sub.PC2>=lims[1][0])&(sub.PC2<=lims[1][1])).all()

def main_figure():
    p=Page('Figure_2',193)
    p.panel('a','Response landscapes',3,187)
    p.text(179,187,'Fold 1, 339 unseen interventions',6.2,color=SOFT,ha='right')
    examples=['Observed','CellFlow','GEARS','Scouter']
    f=XY[XY.fold.eq(0)&XY.model_display.isin(examples)];lim=limits(f)
    for i,m in enumerate(examples):
        x=15+43*i
        ax=p.ax(x,143,34,31,f'a{i}')
        cloud(ax,f,m,lim,size=5)
        ax.set_xticks([-20,-10,0]);ax.set_yticks([-5,0,5])
        p.text(x+17,179,m,6.8,color=INK,weight='bold',ha='center')
        if i==0:ax.set_ylabel('Training-response PC2',labelpad=2)
        else:ax.tick_params(labelleft=False)
    p.rows.append(['a0','a1','a2','a3'])
    p.text(96,133.5,'Training-response PC1',6.4,ha='center')

    p.panel('b','Reduced response variation',3,124)
    p.panel('c','Concentrated variation',94,124)
    b=p.ax(17,87,67,28,'b')
    bars(b,MODELS,'variance_retention_vs_held_truth',(0,1.16),'Variance retained',ref=1)
    b.set_yticks([0,.5,1])
    c=p.ax(108,87,67,28,'c')
    curves(c,['Observed','CellFlow','GEARS','Scouter'])
    c.set_ylabel('Cumulative variance',labelpad=3);c.set_xlabel('Principal components',labelpad=2)
    c.set_yticks([0,.5,.8,1])
    p.line_key(107,72,'Observed',COLORS['Observed'])
    p.line_key(144,72,'CellFlow',COLORS['CellFlow'],(0,(3,2)))
    p.line_key(107,67,'GEARS',COLORS['GEARS'],(0,(3,2)))
    p.line_key(144,67,'Scouter',COLORS['Scouter'],(0,(3,2)))
    p.rows.append(['b','c'])

    p.panel('d','Effective dimensionality',3,58)
    p.panel('e','Dimensionality and fidelity',94,58)
    d=p.ax(17,18,67,31,'d')
    bars(d,ORDER,'effective_rank',(0,30),'Entropy effective rank');d.set_yticks([0,10,20,30])
    e=p.ax(108,18,67,31,'e')
    positions={'CellFlow':(.1,.042),'GEARS':(.2,.285),'State':(3.1,.275),
               'PerturbDiff':(3.4,.025),'scGPT':(7.0,.048),'Scouter':(9.2,.16),'TxPert':(7.0,.275)}
    for m in MODELS:
        q=DD[DD.model_display.eq(m)]
        e.scatter(q.effective_rank,q.global_spearman,s=10,c=COLORS[m],alpha=.35,edgecolor='none',zorder=2)
        x=MEANS.loc[m,'effective_rank'];y=MEANS.loc[m,'global_spearman']
        e.scatter([x],[y],s=30,c=COLORS[m],marker=MARKERS[m],edgecolor=INK,lw=.45,zorder=4)
        tx,ty=positions[m]
        e.annotate(m,xy=(x,y),xytext=(tx,ty),fontsize=6.1,ha='left',va='center',color=INK,
                   arrowprops=dict(arrowstyle='-',color=COLORS[m],lw=.6,shrinkA=2,shrinkB=4))
    e.set(xlim=(0,14),ylim=(0,.32),xticks=[0,4,8,12],yticks=[0,.1,.2,.3],
          xlabel='Entropy effective rank',ylabel='Global ρ')
    p.rows.append(['d','e']);p.cols=[['b','d'],['c','e']]
    p.save({'A':(0,130,183,193),'BC':(0,64,183,130),'DE':(0,0,183,64)})

def supplementary_spectra():
    p=Page('Supplementary_Figure_4',211)
    p.text(3,205,'Response spectra across five held-out folds',8.2,weight='bold')
    p.line_key(113,205,'Observed',COLORS['Observed'])
    p.line_key(149,205,'Predicted',SOFT,(0,(3,2)))
    for i,m in enumerate(ORDER):
        row,col=divmod(i,4);x=15+43*col;y=151-51*row
        p.panel(chr(97+i),m,x-10,y+39)
        ax=p.ax(x,y,34,31,f's{i}')
        if m!='Observed':
            # Matched measured spectrum provides the same direct reference in each panel.
            curves(ax,['Observed',m],log=True,fold_lines=True)
        else:curves(ax,['Observed'],log=True,fold_lines=True)
        if col==0:ax.set_ylabel('Cumulative variance',labelpad=3)
        else:ax.tick_params(labelleft=False)
        if row==1:ax.set_xlabel('Principal components',labelpad=2)
    p.rows=[['s0','s1','s2','s3'],['s4','s5','s6','s7']]
    p.cols=[[f's{i}',f's{i+4}'] for i in range(4)]
    p.panel('i','Dominance of the first direction',3,75)
    p.panel('j','Directions explaining 80%',94,75)
    i=p.ax(17,24,67,40,'i');j=p.ax(108,24,67,40,'j')
    bars(i,ORDER,'pc1_fraction',(0,1.13),'PC1 variance fraction')
    bars(j,ORDER,'pc80',(0,28),'PC80')
    i.set_yticks([0,.5,1]);j.set_yticks([0,10,20])
    p.rows.append(['i','j'])
    p.save({'spectra':(0,83,183,211),'summaries':(0,0,183,82)})

def supplementary_maps():
    p=Page('Supplementary_Figure_5',212)
    p.text(3,206,'Matched response maps across all models and folds',8.2,weight='bold')
    for col,m in enumerate(ORDER):
        p.text(16+20.5*col+8.75,195,m,6.1,color=INK,weight='bold',ha='center')
    all_rows=[]
    for fold in range(5):
        f=XY[XY.fold.eq(fold)];lim=limits(f);y=163-35*fold
        p.text(16,y+27,f'Fold {fold+1}, n = {f[f.model_display.eq("Observed")].shape[0]}',6.1,color=SOFT)
        row=[]
        for col,m in enumerate(ORDER):
            id=f'f{fold}m{col}';row.append(id)
            ax=p.ax(16+20.5*col,y,17.5,22,id)
            cloud(ax,f,m,lim,size=1.9)
            ax.xaxis.set_major_locator(MaxNLocator(2));ax.yaxis.set_major_locator(MaxNLocator(2))
            if col>0:ax.tick_params(labelleft=False)
        all_rows.append(row)
    p.rows=all_rows;p.cols=[[f'f{fold}m{col}' for fold in range(5)] for col in range(8)]
    p.text(95,9,'Training-response PC1',6.5,ha='center')
    p.text(3,107,'Training-response PC2',6.5,ha='center',rotation=90,rotation_mode='anchor')
    p.save({'folds12':(0,126,183,212),'folds345':(0,0,183,126)})
