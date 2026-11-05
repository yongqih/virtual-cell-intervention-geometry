"""Draw Figure6 and supplements from frozen display tables, exclusively in matplotlib."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Arc
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter

from settings import INPUT, OUTPUT as OUT, CHECKS as LAYOUT
DATA=INPUT/'fig6'
from layout import record_panel_positions

mpl.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':6.7,'axes.labelsize':6.7,'axes.titlesize':7.0,
    'xtick.labelsize':6.1,'ytick.labelsize':6.1,'legend.fontsize':6.2,
    'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'legend.frameon':False,'lines.linewidth':1.15,'patch.linewidth':.65,
    'figure.facecolor':'white','savefig.facecolor':'white',
})
INK='#292C2E'; GREY='#666B6E'; GRID='#E4E7E8'
BLUE='#86AABD'; TEAL='#87AEA7'; OCHRE='#D4BD91'; PALE='#EEE2CE'
COLORS={'TxPert_frozen':'#74797C','Anchor_Ridge1':BLUE,'Anchor_RidgeCV':'#B7CBD6',
        'ResidualLinearCV_UnionInput32':'#5D8F89',
        'ResidualLinearCV':TEAL,'CopyMean':OCHRE,'CopyZero':PALE,
        'FullReadout_CopyMean32':OCHRE,'FullReadout_CopyZero':PALE,
        'Scouter_full':'#74797C','Anchor11':BLUE,'WrongTarget_Anchor11':'#D6D9D9',
        'WrongTarget_Anchor_Ridge1':'#D6D9D9','GenePT_Ridge1':OCHRE}
LABELS={'TxPert_frozen':'TxPert','Anchor_Ridge1':'Ridge','Anchor_RidgeCV':'Ridge-CV',
        'ResidualLinearCV_UnionInput32':'Full-input\nresidual',
        'ResidualLinearCV':'Residual','FullReadout_CopyMean32':'Copy + mean',
        'FullReadout_CopyZero':'Copy','Scouter_full':'Scouter','Anchor11':'Anchor Ridge',
        'WrongTarget_Anchor11':'Wrong anchor','GenePT_Ridge1':'GenePT Ridge',
        'WrongTarget_Anchor_Ridge1':'Wrong anchor'}
PRIMARY='paired11_primary'; SECONDARY='TxPert_train_secondary'
METRIC_NAMES={'CA':'CA','rho':'Global ρ','top10':'Top-10'}
JITTER=np.linspace(-.16,.16,5)
def read(name): return pd.read_csv(DATA/name)
BENCH=read('benchmark_n32.csv'); COPY=read('copy_complete_readout.csv')
CURVE=read('display_calibration_paired.csv'); CA=read('calibration_CA.csv')
MULTI=read('multicontext_curves.csv'); FIXED=read('fixed_26_panels.csv')
TRIP=read('triplet_summary.csv'); CASES=read('fixed_26_cases.csv.gz')

class Page:
    def __init__(self,name,height=220):
        self.name=name; self.w=183.; self.h=float(height)
        self.fig=plt.figure(figsize=(self.w/25.4,self.h/25.4))
        self.cv=self.fig.add_axes([0,0,1,1],label='design_canvas')
        self.cv.set(xlim=(0,self.w),ylim=(0,self.h));self.cv.axis('off')
        self.axes=[];self.ids=[];self.rows=[];self.cols=[]
    def text(self,x,y,s,size=6.7,ha='left',color=INK,weight='normal',**kwargs):
        return self.cv.text(x,y,s,fontsize=size,ha=ha,va='center',color=color,
                            fontweight=weight,**kwargs)
    def panel(self,letter,title,x,y):
        self.text(x,y,letter,9,weight='bold');self.text(x+5,y,title,7.3,weight='bold')
    def ax(self,x,y,w,h,id):
        a=self.fig.add_axes([x/self.w,y/self.h,w/self.w,h/self.h],label=id)
        a.tick_params(length=2.2,width=.6,pad=2)
        self.axes.append(a);self.ids.append(id);return a
    def key(self,x,y,label,color,ls='-',marker=None):
        self.cv.plot([x,x+4],[y,y],color=color,lw=1.2,ls=ls)
        if marker:self.cv.plot([x+2],[y],marker=marker,ms=3.1,mfc=color,mec=INK,mew=.3)
        self.text(x+5.3,y,label,6.2)
    def arrow(self,a,b,color=INK,ls='-',lw=.85,scale=6):
        p=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=scale,color=color,
                          linewidth=lw,linestyle=ls,shrinkA=0,shrinkB=0)
        self.cv.add_patch(p);return p
    def save(self,crops=None):
        self.fig.canvas.draw()
        report=record_panel_positions(self.fig,axes=self.axes,panel_ids=self.ids,
                    row_groups=self.rows,column_groups=self.cols,
                    json_out=LAYOUT/f'{self.name}.alignment.json',
                    overlay_svg=LAYOUT/f'{self.name}.alignment.svg',
                    tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
        self.fig.savefig(OUT/f'{self.name}.pdf',dpi=600)
        self.fig.savefig(OUT/f'{self.name}.svg',dpi=600)
        self.fig.savefig(OUT/f'{self.name}.png',dpi=300)
        self.fig.savefig(OUT/f'{self.name}_600dpi.png',dpi=600)
        for key,box in (crops or {}).items():
            bbox=mpl.transforms.Bbox.from_extents(*[v/25.4 for v in box])
            self.fig.savefig(LAYOUT/f'{self.name}_{key}.png',dpi=300,bbox_inches=bbox)
        print(json.dumps({'figure':self.name,'mm':[self.w,self.h],'axes':len(self.axes),
                          'alignment':report['status']}),flush=True)
        plt.close(self.fig)

def clean(a,axis='y'):
    a.set_axisbelow(True);a.grid(axis=axis,lw=.4,ls=(0,(2,3)),color=GRID)

def bars(a,frame,methods,metric,ylim,labels=None):
    for k,m in enumerate(methods):
        q=frame[frame.method.eq(m)]
        assert len(q)==1,(m,len(q))
        r=q.iloc[0];v=float(r[metric]);lo=float(r[metric+'_ci_low']);hi=float(r[metric+'_ci_high'])
        assert np.isfinite([v,lo,hi]).all() and lo<=v<=hi
        a.bar(k,v,width=.6,color=COLORS[m],edgecolor=INK,linewidth=.65,zorder=2)
        a.errorbar(k,v,yerr=[[v-lo],[hi-v]],fmt='none',ecolor=INK,elinewidth=.75,capsize=2.2,zorder=4)
    a.set(xlim=(-.6,len(methods)-.4),ylim=ylim,xticks=range(len(methods)),
          xticklabels=labels or [LABELS[m] for m in methods])
    a.yaxis.set_major_locator(MaxNLocator(4));clean(a)

def horizontal_points(a,records,xlim,labels,colors):
    for k,(v,lo,hi) in enumerate(records):
        assert np.isfinite([v,lo,hi]).all()
        a.errorbar(v,k,xerr=[[v-lo],[hi-v]],fmt='o',ms=4,mfc=colors[k],mec=INK,
                   mew=.45,ecolor=colors[k],elinewidth=1.1,capsize=2,zorder=3)
    a.set(ylim=(len(records)-.5,-.5),xlim=xlim,yticks=range(len(records)),yticklabels=labels)
    clean(a,'x');a.xaxis.set_major_locator(MaxNLocator(4))

def gain_curve(a,metric,reference=PRIMARY,mark_thresholds=False):
    for m,marker in [('Anchor_Ridge1','s'),('ResidualLinearCV','o')]:
        d=CURVE[CURVE.method.eq(m)&CURVE.metric.eq(metric)&CURVE.reference.eq(reference)].sort_values('n_train')
        assert len(d)>1 and np.all(np.diff(d.n_train)>0)
        x=d.n_train.to_numpy();lo=d.simultaneous_low.to_numpy();hi=d.simultaneous_high.to_numpy()
        a.fill_between(x,lo,hi,color=COLORS[m],alpha=.14,lw=0,zorder=1)
        a.plot(x,d.delta,color=COLORS[m],marker=marker,ms=2.1,mew=.3,markeredgecolor='white',zorder=3)
        if mark_thresholds:
            n=23 if m=='Anchor_Ridge1' else 10
            q=d[d.n_train.eq(n)].iloc[0]
            a.scatter([n],[q.delta],s=21,color=COLORS[m],edgecolor=INK,lw=.5,zorder=5,marker=marker)
    a.axhline(0,color=GREY,ls=(0,(3,2)),lw=.65,zorder=0)
    a.set(xlim=(1,33),xticks=[2,10,23,32])
    a.set_ylim((-.27,.23) if metric=='rho' else (-.13,.105))
    a.set_yticks([-.2,0,.2] if metric=='rho' else [-.1,0,.1]);clean(a)

def ca_curve(a,show_baseline=True):
    for m,marker in [('Anchor_Ridge1','s'),('ResidualLinearCV','o')]:
        d=CA[CA.method.eq(m)].sort_values('n_train')
        a.fill_between(d.n_train,d.CA_ci_low,d.CA_ci_high,color=COLORS[m],alpha=.13,lw=0)
        a.plot(d.n_train,d.CA,color=COLORS[m],marker=marker,ms=2,mew=.25,mec='white')
    if show_baseline:a.axhline(.6501650247650068,color=GREY,ls=(0,(3,2)),lw=.8)
    a.set(xlim=(1,33),ylim=(.49,.69),xticks=[2,10,23,32],yticks=[.5,.6,.65],xlabel='Paired interventions')
    clean(a)

def context_curves(a,metric,include_wrong=False):
    for context,color,marker in [('Co-culture',BLUE,'o'),('IFNγ',TEAL,'s')]:
        d=MULTI[MULTI.context.eq(context)&MULTI.method.eq('ResidualLinearCV')].sort_values('n_calibration')
        a.fill_between(d.n_calibration,d[metric+'_ci_low'],d[metric+'_ci_high'],color=color,alpha=.13,lw=0)
        a.plot(d.n_calibration,d[metric],color=color,marker=marker,ms=3,mec=INK,mew=.35)
        if include_wrong:
            w=MULTI[MULTI.context.eq(context)&MULTI.method.eq('WrongResidualLinearCV')].sort_values('n_calibration')
            a.plot(w.n_calibration,w[metric],color=color,ls=(0,(2,2)),lw=.7)
    a.set(xlim=(3,42),xticks=[5,10,20,40]);clean(a)
    if metric=='rho':a.set(ylim=(-.07,.57) if include_wrong else (0,.57),yticks=[0,.2,.4])
    elif metric=='top10':a.set(ylim=(.06,.21),yticks=[.1,.15,.2])
    else:a.set(ylim=(.43,.58),yticks=[.45,.5,.55])

def schematic(p):
    p.panel('a','Intervention references for context transfer',3,214)
    # Conceptual matrix: displayed rows do not represent actual identity counts.
    x1,x2=35,53; top=202; dy=2.25
    p.text(x1+4.5,207,'K562',6.8,ha='center');p.text(x2+4.5,207,'RPE1',6.8,ha='center')
    for i in range(9):
        y=top-i*dy
        p.cv.add_patch(Rectangle((x1,y),9,1.75,facecolor=BLUE,edgecolor='none'))
        p.cv.add_patch(Rectangle((x2,y),9,1.75,facecolor=TEAL if i<2 else 'white',
                                 edgecolor=TEAL,linewidth=.6,linestyle='-' if i<2 else (0,(2,1))))
    p.text(29,201,'32 paired',6.2,ha='right')
    p.text(29,190,'1,432 queries',6.2,ha='right')
    p.text(48,180.5,'Intervention × context',6.2,color=GREY,ha='center')
    p.arrow((66,199),(80,199),color=GREY)
    p.text(87,203,'Paired responses',6.6)
    p.arrow((111,199),(125,199),color=INK)
    p.text(146,199,'Context map',7,ha='center',weight='bold')
    p.cv.plot([146,146],[196.7,190],color=GREY,lw=.7)
    p.text(87,186,'Measured query',6.6)
    p.arrow((115,186),(130,186),color=TEAL)
    p.arrow((150,186),(161,186),color=TEAL)
    p.cv.add_patch(Rectangle((132,182.9),16,6.2,facecolor='#E7EFEC',edgecolor=TEAL,lw=.65))
    p.text(140,186,'Transfer',6.5,ha='center')
    p.text(166,188,'RPE1',6.6,ha='center')
    p.text(166,184,'prediction',6.3,ha='center')

def vector_case(p,xc,y0,method,label):
    L=15.0
    a=read('case_vector_angles.csv')
    p.text(xc,y0+20,label,6.8,ha='center',weight='bold')
    p.arrow((xc,y0),(xc,y0+L),INK,'-',.9,6)
    p.arrow((xc,y0),(xc,y0+L),INK,(0,(2,2)),.65,5)
    p.text(xc,y0+17,'COX5A',6.1,ha='center')
    for role,color,sgn in [('Related',BLUE,1),('Contrasting',OCHRE,-1)]:
        for observed in [True,False]:
            row=a[a.method.eq('Observed' if observed else method)&a.role.eq(role)].iloc[0]
            theta=float(row.angle_degrees);phi=np.deg2rad(90+sgn*theta)
            xy=(xc+L*np.cos(phi),y0+L*np.sin(phi))
            ls='-' if observed else (0,(2.4,1.8))
            p.arrow((xc,y0),xy,color,ls,.9 if observed else .85,6)
            radius=5.8 if observed else 8.5
            t1,t2=sorted([90,90+sgn*theta])
            p.cv.add_patch(Arc((xc,y0),2*radius,2*radius,theta1=t1,theta2=t2,
                              color=color,lw=.65,linestyle=ls))
        p.text(xc-16 if sgn>0 else xc+16,y0+11,'COX7B' if sgn>0 else 'PSMG3',
               6.1,color=INK,ha='center')


def supplement1():
    p=Page('Supplementary_Figure_6_1',218)
    p.text(3,212,'Sparse context calibration and matched map controls',8.1,weight='bold')
    methods=['TxPert_frozen','Anchor_Ridge1','Anchor_RidgeCV','ResidualLinearCV']
    for row,(budget,y) in enumerate([(10,160),(32,96)]):
        frame=read('benchmark_n10.csv') if budget==10 else BENCH
        frame=frame[frame.reference.eq(PRIMARY)]
        p.text(3,y+46,f'{budget} paired interventions',7.2,weight='bold')
        ids=[]
        for i,(metric,ylim) in enumerate([('CA',(0,.8)),('rho',(0,.48)),('top10',(0,.25))]):
            x=18+59*i;id=f's{row}{i}';a=p.ax(x,y,43,34,id);ids.append(id)
            bars(a,frame,methods,metric,ylim)
            for t in a.get_xticklabels():t.set_rotation(35);t.set_ha('right');t.set_rotation_mode('anchor')
            p.panel(chr(97+row*3+i),METRIC_NAMES[metric],x-14,y+40)
        p.rows.append(ids)
    p.text(3,76,'Reference sensitivity and CA calibration',7.2,weight='bold')
    for i,metric in enumerate(['rho','top10']):
        x=18+59*i;a=p.ax(x,26,43,40,f'q{i}')
        gain_curve(a,metric,SECONDARY,True)
        p.panel(chr(103+i),'Δρ' if metric=='rho' else 'ΔTop-10',x-14,70)
        a.set_xlabel('Paired interventions')
    a=p.ax(136,26,43,40,'q2');ca_curve(a)
    p.panel('i','CA',122,70)
    p.rows.append(['q0','q1','q2'])
    p.cols=[['s00','s10','q0'],['s01','s11','q1'],['s02','s12','q2']]
    p.key(33,6,'Ridge',BLUE,marker='s');p.key(74,6,'Residual',TEAL,marker='o')
    p.key(119,6,'TxPert',GREY,(0,(3,2)))
    p.save({'top':(0,140,183,218),'middle':(0,76,183,140),'bottom':(0,0,183,76)})

def supplement2():
    p=Page('Supplementary_Figure_6_2',201)
    p.text(3,195,'Measurement reliability and cross-context concordance',8.1,weight='bold')
    measure=read('measurement_references.csv');mca=read('measurement_CA.csv').set_index('method')
    labels=['RPE1 cell halves','Measured K562–RPE1','Residual, 32 pairs','TxPert']
    colors=[TEAL,OCHRE,TEAL,COLORS['TxPert_frozen']]
    for i,metric in enumerate(['CA','rho','top10']):
        x=42+47*i;a=p.ax(x,149,39,33,f'a{i}')
        records=[]
        for method in ['RPE1_cell_split','Measured_K562_full_readout','ResidualLinearCV','TxPert_frozen']:
            if method in ['RPE1_cell_split','Measured_K562_full_readout']:
                r=mca.loc[method] if metric=='CA' else measure[measure.method.eq(method)&measure.reference.eq(PRIMARY)].iloc[0]
            else:r=BENCH[BENCH.method.eq(method)&BENCH.reference.eq(PRIMARY)].iloc[0]
            records.append((r[metric],r[metric+'_ci_low'],r[metric+'_ci_high']))
        horizontal_points(a,records,(0,1.02) if metric!='top10' else (0,.63),labels if i==0 else ['']*4,colors)
        p.panel(chr(97+i),METRIC_NAMES[metric],x-4,187)
    p.rows.append(['a0','a1','a2'])
    p.text(42,140,'1,432 matched queries · same scored genes',6.3,color=GREY)

    p.panel('d','Measured source–recipient agreement',3,129)
    for i,metric in enumerate(['rho','top10']):
        x=42+77*i;a=p.ax(x,88,56,29,f'd{i}')
        refs=[PRIMARY,SECONDARY,'native_context_paired11'];rec=[]
        for ref in refs:
            r=measure[measure.method.eq('Measured_K562_full_readout')&measure.reference.eq(ref)].iloc[0]
            rec.append((r[metric],r[metric+'_ci_low'],r[metric+'_ci_high']))
        horizontal_points(a,rec,(.1,.42) if metric=='rho' else (.17,.28),
                          ['Paired mean','Training mean','Native means'] if i==0 else ['']*3,[OCHRE]*3)
        a.set_xlabel(METRIC_NAMES[metric])
    p.rows.append(['d0','d1'])

    p.panel('e','Three recipient cell partitions',3,72)
    part=read('measurement_partitions.csv')
    for i,metric in enumerate(['rho','top10']):
        x=18+41*i;a=p.ax(x,23,30,36,f'e{i}')
        for j,(ref,c,m) in enumerate([(PRIMARY,TEAL,'o'),(SECONDARY,BLUE,'s')]):
            d=part[part.reference.eq(ref)].sort_values('split')
            a.plot(np.arange(1,4),d[metric],color=c,marker=m,ms=3,mec=INK,mew=.35)
        a.set(xlim=(.7,3.3),xticks=[1,2,3],xlabel='Partition')
        a.set_ylim((.87,.92) if metric=='rho' else (.53,.57))
        a.yaxis.set_major_locator(MaxNLocator(3));clean(a)
        p.text(x+15,62,METRIC_NAMES[metric],6.6,ha='center')
    p.rows.append(['e0','e1'])
    p.panel('f','Frangieh cell-split reliability',98,72)
    frel=read('multicontext_reliability.csv')
    for i,metric in enumerate(['rho','top10']):
        x=110+40*i;a=p.ax(x,23,28,36,f'f{i}')
        for j,ctx in enumerate(['Control','Co-culture','IFNγ']):
            r=frel[frel.context.eq(ctx)].iloc[0]
            a.errorbar(j,r[metric],yerr=[[r[metric]-r[metric+'_ci_low']],[r[metric+'_ci_high']-r[metric]]],
                       fmt='o',ms=4,mfc=[OCHRE,BLUE,TEAL][j],mec=INK,mew=.4,
                       ecolor=[OCHRE,BLUE,TEAL][j],capsize=2)
        a.set(xlim=(-.5,2.5),xticks=[0,1,2],xticklabels=['Control','Co-culture','IFNγ'])
        a.set_ylim((.4,.7) if metric=='rho' else (.15,.29));a.yaxis.set_major_locator(MaxNLocator(3))
        for t in a.get_xticklabels():t.set_rotation(40);t.set_ha('right');t.set_rotation_mode('anchor')
        p.text(x+14,62,METRIC_NAMES[metric],6.6,ha='center');clean(a)
    p.rows.append(['f0','f1'])
    p.key(13,6,'Paired mean',TEAL,marker='o');p.key(53,6,'Training mean',BLUE,marker='s')
    p.save({'top':(0,134,183,201),'middle':(0,77,183,134),'bottom':(0,0,183,77)})

def supplement3():
    p=Page('Supplementary_Figure_6_3',223)
    p.text(3,217,'Frozen-case repair and newly introduced inversions',8.1,weight='bold')
    p.panel('a','All 26 frozen cases',3,207)
    ordered=read('fixed_26_original.csv')
    keys=[tuple(r) for r in ordered[['fold','A','B','C']].itertuples(index=False,name=None)]
    heat=[];ticklabels=[]
    for fold,A,B,C in keys:
        q=CASES[CASES.fold.eq(fold)&CASES.A.eq(A)&CASES.B.eq(B)&CASES.C.eq(C)&
                CASES.reference.eq(SECONDARY)&CASES.method.eq('ResidualLinearCV')].sort_values('panel')
        assert len(q)==5
        heat.append(q.correct.astype(int).to_numpy());ticklabels.append(f'{A} / {C} / {B}')
    a=p.ax(54,94,35,103,'a')
    a.imshow(np.asarray(heat),aspect='auto',interpolation='nearest',cmap=ListedColormap(['#E6E9E9',TEAL]),vmin=0,vmax=1)
    a.set(yticks=np.arange(26),yticklabels=ticklabels,xticks=np.arange(5),xticklabels=['1','2','3','4','5'])
    a.tick_params(axis='y',labelsize=5.65,length=0,pad=3)
    a.set_xticks(np.arange(-.5,5,1),minor=True);a.set_yticks(np.arange(-.5,26,1),minor=True)
    a.grid(which='minor',color='white',lw=.8);a.tick_params(which='minor',length=0)
    a.set_xlabel('Calibration panel')
    p.text(3,201,'Reference / related / contrasting',6,color=GREY)
    p.cv.add_patch(Rectangle((7,85),3,2.5,facecolor=TEAL,edgecolor='none'));p.text(11,86.25,'Corrected',6.1)
    p.cv.add_patch(Rectangle((45,85),3,2.5,facecolor='#E6E9E9',edgecolor='none'));p.text(49,86.25,'Still inverted',6.1)

    p.panel('b','Recovery across references',101,207)
    b=p.ax(117,153,59,43,'b')
    for i,ref in enumerate([SECONDARY,PRIMARY]):
        q=FIXED[FIXED.reference.eq(ref)&FIXED.method.eq('ResidualLinearCV')].sort_values('panel')
        vals=q.correct_order_count.to_numpy()
        b.bar(i,vals.mean(),width=.55,color=TEAL if i==0 else BLUE,edgecolor=INK,lw=.6)
        b.scatter(i+JITTER,vals,s=14,color=INK,alpha=.6,zorder=3,edgecolor='none')
    b.set(xlim=(-.6,1.6),ylim=(0,27),yticks=[0,13,26],xticks=[0,1],
          xticklabels=['Training mean','Paired mean'],ylabel='Correct / 26');clean(b)
    p.text(118,144,'Five fixed calibration panels',6.2,color=GREY)

    p.panel('c','Corrections and new errors',101,134)
    c=p.ax(117,94,59,30,'c')
    counts=read('triplet_counts.csv')
    q=counts[counts.quality_mode.eq('all_three_four_criteria')&counts.eligibility.eq('fully_rematched_rule')&
             counts.reference.eq(PRIMARY)&counts.method.eq('ResidualLinearCV')]
    bypanel=q.groupby('panel')[['baseline_errors_repaired','newly_damaged']].sum()
    for i,(col,color) in enumerate([('baseline_errors_repaired',TEAL),('newly_damaged',OCHRE)]):
        vals=bypanel[col].to_numpy();assert len(vals)==5
        c.bar(i,vals.mean(),width=.55,color=color,edgecolor=INK,lw=.6)
        c.scatter(i+JITTER,vals,s=12,color=INK,alpha=.6,zorder=3,edgecolor='none')
    c.set(xlim=(-.6,1.6),ylim=(0,max(bypanel.max())+5),xticks=[0,1],
          xticklabels=['Corrected','New errors'],ylabel='Triplets');clean(c)

    p.panel('d','Overall inversion rate',3,70)
    d=p.ax(37,22,52,38,'d')
    sub=TRIP[TRIP.quality_mode.eq('all_three_four_criteria')&TRIP.eligibility.eq('fully_rematched_rule')]
    for i,ref in enumerate([PRIMARY,SECONDARY]):
        for shift,method,color in [(-.12,'TxPert_frozen',GREY),(.12,'ResidualLinearCV',TEAL)]:
            r=sub[sub.reference.eq(ref)&sub.method.eq(method)].iloc[0]
            v=r.inversion_rate*100;lo=r.inversion_rate_ci_low*100;hi=r.inversion_rate_ci_high*100
            d.errorbar(v,i+shift,xerr=[[v-lo],[hi-v]],fmt='o',ms=4,mfc=color,mec=INK,mew=.35,
                       ecolor=color,capsize=2)
    d.set(xlim=(0,20),ylim=(1.5,-.5),yticks=[0,1],yticklabels=['Paired mean','Training mean'],
          xlabel='Inverted triplets (%)');clean(d,'x')
    p.key(15,7,'TxPert',GREY);p.key(49,7,'Residual',TEAL)
    p.panel('e','Net reduction in errors',101,70)
    e=p.ax(128,22,48,38,'e')
    configs=[('fully_rematched_rule',PRIMARY),('fully_rematched_rule',SECONDARY),
             ('frozen_quality_current_biology',PRIMARY),('frozen_quality_current_biology',SECONDARY)]
    rec=[]
    for el,ref in configs:
        r=TRIP[TRIP.quality_mode.eq('all_three_four_criteria')&TRIP.method.eq('ResidualLinearCV')&
               TRIP.eligibility.eq(el)&TRIP.reference.eq(ref)].iloc[0]
        rec.append(tuple(r[c]*100 for c in ['net_error_reduction_rate','net_error_reduction_ci_low','net_error_reduction_ci_high']))
    horizontal_points(e,rec,(-12,25),['Strict · paired','Strict · training','Frozen · paired','Frozen · training'],
                      [TEAL,TEAL,BLUE,BLUE]);e.tick_params(axis='y',labelsize=5.8)
    e.axvline(0,color=GREY,ls=(0,(3,2)),lw=.65);e.set_xlabel('Percentage points')
    p.rows.append(['d','e'])
    p.save({'top':(0,79,183,223),'bottom':(0,0,183,79)})

def supplement4():
    p=Page('Supplementary_Figure_6_4',212)
    p.text(3,206,'Reference-library reuse and independent-model support',8.1,weight='bold')
    p.text(3,198,'Residual transfer · the same 120 queries in two recipient contexts',6.5,color=GREY)
    for i,metric in enumerate(['CA','rho','top10']):
        x=18+59*i;a=p.ax(x,148,43,36,f'a{i}');context_curves(a,metric,include_wrong=True)
        a.set_xlabel('Paired interventions');p.panel(chr(97+i),METRIC_NAMES[metric],x-14,190)
    p.rows.append(['a0','a1','a2'])
    p.key(24,132,'Co-culture',BLUE,marker='o');p.key(67,132,'IFNγ',TEAL,marker='s')
    p.key(108,132,'Wrong identity',GREY,(0,(2,2)))

    p.text(3,120,'Independent Scouter comparison · 120 queries',7.1,weight='bold')
    independent=read('independent_Scouter.csv')
    for i,metric in enumerate(['rho','top10']):
        x=26+88*i;a=p.ax(x,77,63,34,f'l{i}')
        bars(a,independent,['Scouter_full','Anchor11','WrongTarget_Anchor11'],metric,
             (-.08,.45) if metric=='rho' else (0,.24))
        if metric=='top10':a.axhline(10/119,color=GREY,ls=(0,(3,2)),lw=.7)
        p.panel(chr(100+i),METRIC_NAMES[metric],x-22,113)
    p.rows.append(['l0','l1'])
    p.text(3,62,'Measured versus descriptor input · 11 paired interventions',7.1,weight='bold')
    ctrl=read('identity_controls.csv')
    ctrl=ctrl[ctrl.reference.eq(PRIMARY)&ctrl.n_train.eq(11)]
    methods=['TxPert_frozen','Anchor_Ridge1','GenePT_Ridge1','WrongTarget_Anchor_Ridge1']
    for i,metric in enumerate(['rho','top10']):
        x=26+88*i;a=p.ax(x,20,63,34,f'q{i}')
        bars(a,ctrl,methods,metric,(-.03,.4) if metric=='rho' else (0,.19),
             ['TxPert','Anchor Ridge','GenePT','Wrong anchor'])
        if metric=='top10':a.axhline(float(ctrl.random_top10.iloc[0]),color=GREY,ls=(0,(3,2)),lw=.7)
        p.panel(chr(102+i),METRIC_NAMES[metric],x-22,55)
        for t in a.get_xticklabels():t.set_rotation(25);t.set_ha('right');t.set_rotation_mode('anchor')
    p.rows.append(['q0','q1']);p.cols=[['l0','q0'],['l1','q1']]
    p.save({'reuse':(0,125,183,212),'scouter':(0,65,183,125),'identity':(0,0,183,65)})

def supplement5():
    p=Page('Supplementary_Figure_6_5',146)
    p.text(3,140,'Context transfer with matched source information',8.1,weight='bold')
    p.text(3,133,'32 paired interventions · 1,432 shared queries',6.3,color=GREY)
    data=read('matched_input_n32.csv')
    paired=read('matched_input_paired.csv')
    full='ResidualLinearCV_UnionInput32'
    methods=['FullReadout_CopyMean32','ResidualLinearCV',full]
    labels=['Copy +\nmean','Original\nresidual','Full-input\nresidual']
    for i,(metric,ylim) in enumerate([('CA',(0,.8)),('rho',(0,.48)),('top10',(0,.28))]):
        x=24+55*i
        a=p.ax(x,87,43,34,f't{i}')
        bars(a,data[data.reference.eq(PRIMARY)],methods,metric,ylim,labels=labels)
        p.panel(chr(97+i),METRIC_NAMES[metric],x-20,125)
    p.rows.append(['t0','t1','t2'])

    p.text(3,65,'Full-input residual − Copy + mean',7.2,weight='bold')
    for i,metric in enumerate(['CA','rho','top10']):
        x=24+55*i
        a=p.ax(x,20,43,33,f'd{i}')
        refs=['CA_translation_invariant'] if metric=='CA' else [PRIMARY,SECONDARY]
        for j,ref in enumerate(refs):
            q=paired[paired.method.eq(full)&paired.comparator.eq('FullReadout_CopyMean32')&
                     paired.reference.eq(ref)&paired.metric.eq(metric)]
            assert len(q)==1
            z=q.iloc[0]
            y=.5 if metric=='CA' else j
            color=COLORS[full] if metric=='CA' or ref==PRIMARY else BLUE
            marker='o' if metric=='CA' or ref==PRIMARY else 's'
            a.errorbar(z.delta,y,xerr=[[z.delta-z.simultaneous_low],[z.simultaneous_high-z.delta]],
                       fmt=marker,ms=4.1,mfc=color,mec=INK,mew=.4,ecolor=color,elinewidth=1.2,capsize=2.3)
        a.axvline(0,color=GREY,ls=(0,(3,2)),lw=.65)
        a.set(ylim=(1.5,-.5))
        if metric=='CA':
            a.set(xlim=(-.01,.08),xticks=[0,.04,.08],yticks=[.5],yticklabels=['Both\nreferences'])
        elif metric=='rho':
            a.set(xlim=(-.01,.1),xticks=[0,.05,.1],yticks=[0,1],yticklabels=['Paired\nmean','Training\nmean'])
        else:
            a.set(xlim=(-.04,.01),xticks=[-.04,-.02,0],yticks=[0,1],yticklabels=['Paired\nmean','Training\nmean'])
        a.tick_params(axis='y',labelsize=5.8,length=0,pad=3)
        a.set_xlabel('Difference')
        clean(a,'x')
        p.panel(chr(100+i),'ΔCA' if metric=='CA' else ('Δρ' if metric=='rho' else 'ΔTop-10'),x-20,57)
    p.rows.append(['d0','d1','d2'])
    p.cols=[['t0','d0'],['t1','d1'],['t2','d2']]
    p.save({'top':(0,72,183,146),'bottom':(0,0,183,72)})
