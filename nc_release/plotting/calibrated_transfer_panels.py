"""Render the Fig. 6 data panels from the existing paired comparisons."""
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Arc

from settings import INPUT
from biology_art import BiologyArt

SOURCE = INPUT / 'fig6' / 'calibrated_transfer'
METRICS = ['CA', 'rho', 'top10']
NAMES = ['CA', 'Global ρ', 'Top-10']
GREEN = '#417C70'
BLUE = '#7194A8'
GREY = '#939AA0'
OCHRE = '#C08B60'
INK = '#272B2C'

def key(page, center, y, labels, colors, widths):
    x = center - sum(widths) / 2
    for label, color, width in zip(labels, colors, widths):
        page.cv.plot([x, x+3], [y,y], color=color, lw=2.6, solid_capstyle='butt')
        page.text(x+4.1,y,label,5.9)
        x += width

def bars(page, panel, xs, y, width, height, records, colors, reference=None):
    for i,(met, name, x, ymax) in enumerate(zip(METRICS,NAMES,xs,[.85,.5,.28])):
        ax=page.ax(x,y,width,height,panel+str(i))
        for j,(record,color) in enumerate(zip(records,colors)):
            val,lo,hi=record[met]
            ax.bar(j,val,width=.66,color=color,edgecolor=INK,linewidth=.5)
            ax.errorbar(j,val,[[val-lo],[hi-val]],fmt='none',ecolor=INK,capsize=1.9,lw=.65)
        if reference is not None:
            ax.axhline(reference[met][0],color='#555F65',ls=(0,(2.3,2)),lw=.8,zorder=3.5)
        ticks=[[0,.4,.8],[0,.2,.4],[0,.12,.24]][i]
        ax.set(xlim=(-.6,len(records)-.4),ylim=(0,ymax),xticks=[],yticks=ticks)
        ax.grid(axis='y',ls=(0,(2,3)),lw=.4,color='#E0E2E3');ax.set_axisbelow(True)
        page.text(x+width/2,y+height+3.2,name,6.5,ha='center')
    page.rows.append([panel+str(i) for i in range(3)])

def scores(filename, methods):
    data = pd.read_csv(SOURCE / filename).set_index(['method', 'metric'])
    return [{metric: tuple(data.loc[(method, metric), ['estimate', 'low', 'high']])
             for metric in METRICS} for method in methods]

def main(base):
    p=base.Page('Figure_6',248)
    p.panel('a','Measure interventions, connect contexts',3,242,91.5)
    p.text(47,232.7,'Direct transfer',6.5,ha='center',weight='bold')
    p.text(139,232.7,'Sparse bridges',6.5,ha='center',weight='bold')
    for xx,label,predicted in [(66,'Measured',False),(94,'Predicted',True)]:
        p.cv.add_patch(base.Rectangle((xx,236.4),3.7,2.4,fc='white' if predicted else '#D5DDE0',
            ec='#BCC7CC',lw=.5,ls=(0,(2,1.5)) if predicted else '-'))
        p.text(xx+5.3,237.6,label,6,color='#5E676C')
    base.direct_reference(p);base.resource_bridges(p)
    p.panel('b','Sparse transfer versus zero-shot',3,186,46)
    p.panel('c','Transfer with matched source inputs',94,186,137)
    bars(p,'b',[10,39,68],153,19,23,scores('direct_transfer.csv',['TxPert','Ridge','Calibrated map']),[GREY,BLUE,GREEN])
    bars(p,'c',[102,131,160],153,19,23,scores('matched_source.csv',['Copy','Copy + mean','Calibrated map']),[GREY,OCHRE,GREEN],reference=scores('direct_transfer.csv',['TxPert'])[0])
    key(p,47,148,['TxPert','Ridge','Calibrated map'],[GREY,BLUE,GREEN],[17,16,30])
    p.cv.plot([101,105],[143.6,143.6],color='#555F65',ls=(0,(2.3,2)),lw=.8)
    p.text(106,143.6,'TxPert',5.9)
    key(p,138,148,['Copy','Copy + mean','Calibrated map'],[GREY,OCHRE,GREEN],[14,26,30])

    p.panel('d','Paired interventions needed to outperform TxPert',3,139,91.5)
    all_d=pd.read_csv(SOURCE/'paired_intervention_curves.csv')
    orig=all_d.query('method=="Residual"')
    joint=all_d.query('method=="JointPolicy_scaled"');scaled=all_d.query('method=="MSE_scaled"')
    for i,(m,x) in enumerate(zip(METRICS,[13,73,133])):
        ax=p.ax(x,102,43,25,'d'+str(i))
        ax.axhline(0,color='#7C8285',lw=.65,ls=(0,(3,2)))
        ax.axvline(12,color='#BAC7C3',lw=.6,ls=':')
        values=joint.query('metric==@m').sort_values('k')
        ax.fill_between(values.k.to_numpy(),values.simultaneous_low.to_numpy(),values.simultaneous_high.to_numpy(),color=GREEN,alpha=.13,lw=0)
        for rows,color,style in [(orig,GREY,'-.'),(scaled,BLUE,'--'),(joint,GREEN,'-')]:
            subset=rows.query('metric==@m').sort_values('k')
            if rows is joint:subset=subset.query('k>=12')
            ax.plot(subset.k,subset.delta,color=color,linestyle=style,lw=1.15)
        at=values.loc[values.k==12,'delta'].iat[0]
        ax.scatter([12],[at],s=14,facecolor=GREEN,edgecolor='white',linewidth=.4,zorder=4)
        ax.set(xlim=(1.5,32.5),xticks=[2,12,22,32])
        limits=[(-.155,.12),(-.235,.235),(-.065,.105)][i];ticks=[[-.1,0,.1],[-.2,0,.2],[-.05,0,.05,.1]][i]
        ax.set(ylim=limits,yticks=ticks);base.old.clean(ax)
        p.text(x+21.5,131,['ΔCA','Δ global ρ','ΔTop-10'][i],6.7,ha='center')
    p.rows.append(['d0','d1','d2'])
    p.text(92,94,'Paired interventions',6.2,ha='center')
    for x,lab,col,sty in [(33,'Unscaled map',GREY,'-.'),(69,'Amplitude calibration',BLUE,'--'),(118,'Joint selection + calibration',GREEN,'-')]:
        p.cv.plot([x,x+5],[88,88],color=col,ls=sty,lw=1.2)
        p.text(x+6.5,88,lab,6.0)

    source=SOURCE
    arcs=pd.read_csv(source/'triplet_arcs.csv')
    counts=pd.read_csv(source/'triplet_counts.csv')
    bridges=pd.read_csv(source/'transfer_50_pair_routes.csv')
    p.panel('e','Recovery of the complex IV relationship',3,77,46)
    p.panel('f','Sparse bridges versus the TxPert route',96,77,139)
    BiologyArt(p.cv,INK,BLUE,OCHRE).mitochondrion(25,64)
    for yy in [61,63,65,67]:
        p.cv.add_patch(Ellipse((69,yy),10,3,fc='#EDF0F1',ec='#9BA6AC',lw=.55))
        for dx in [-3,0,3]:p.cv.plot([69+dx]*2,[yy-.8,yy+.8],color='#ABB4B8',lw=.4)
    p.cv.add_patch(Ellipse((75,66),3,2.7,angle=30,fc=OCHRE,ec=OCHRE,lw=.4))
    p.text(25,56.5,'COX5A / COX7B',6,ha='center');p.text(69,56.5,'PSMG3',6,ha='center',color=OCHRE)
    for xc,method,label in [(25,'TxPert','TxPert'),(69,'Joint','Calibrated map')]:
        y0=29
        p.text(xc,y0+21,label,6.7,ha='center',weight='bold')
        p.arrow((xc,y0),(xc,y0+15),INK,'-',.8,6)
        p.text(xc,y0+17,'COX5A',6,ha='center')
        for gene,color,sign in [('COX7B',BLUE,1),('PSMG3',OCHRE,-1)]:
            for dataset,ls,radius in [('Observed','-',5.8),(method,(0,(2.3,1.7)),8.5)]:
                v=arcs.loc[arcs.method.eq(dataset)&arcs.partner.eq(gene),'angle_degrees'].item()
                end=90+v*sign;t=np.deg2rad(end)
                p.arrow((xc,y0),(xc+15*np.cos(t),y0+15*np.sin(t)),color,ls,.85,6)
                low,high=sorted([90,end])
                p.cv.add_patch(Arc((xc,y0),2*radius,2*radius,theta1=low,theta2=high,color=color,lw=.6,linestyle=ls))
            p.text(xc-sign*16,y0+11,gene,6,ha='center')
    mean_corrected=counts.corrected.mean()
    p.text(47,20,f'{mean_corrected:.0f} of 26 corrected on average',6.4,ha='center')
    p.key(16,12,'Observed',INK);p.key(49,12,'Predicted',INK,(0,(2,2)))
    colors={'Two-hop':GREEN,'Copy':'#676D74','Copy + mean':OCHRE}
    for i,(metric,x) in enumerate([('rho',122),('CA',155)]):
        ax=p.ax(x,19,25,43,'f'+str(i))
        ax.axvline(0,color=GREY,lw=.65,ls='--')
        frame=bridges[bridges.metric==metric]
        for j,route in enumerate(['K562_RPE1_Jurkat','Jurkat_RPE1_K562']):
            for k,(model,marker) in enumerate(zip(colors,['o','s','D'])):
                row=frame.query('route==@route and method==@model').iloc[0]
                base.pt(ax,row,metric,j*4+k,colors[model],True,marker=marker,contrast=True)
        ax.set(xlim=(-.012,.289),xticks=[0,.12,.24],
               ylim=(7,-1),yticks=[0,1,2,4,5,6],yticklabels=list(colors)*2 if i==0 else ['']*6)
        ax.tick_params(axis='y',length=0,pad=2,labelsize=5.9)
        ax.spines['left'].set_visible(False);base.old.clean(ax,'x')
        p.text(x+12.5,68,'Δ global ρ' if i==0 else 'ΔCA',6.5,ha='center')
    p.rows.append(['f0','f1'])
    p.text(98,62,'K562 → Jurkat',6.2,weight='bold')
    p.text(98,40.5,'Jurkat → K562',6.2,weight='bold')
    p.text(139,10,'50 paired interventions per link',6.1,ha='center')
    p.save(crops={'concept':(2,190,181,246),'comparison':(92,141.5,182,190)})
