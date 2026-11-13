"""Paper figure assembly from frozen results; no statistical estimation or fitting."""
from pathlib import Path
import sys, json, importlib.util, hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Arc, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath

from settings import INPUT, OUTPUT as OUT, CHECKS as LAYOUT
import transfer_panels as old
BLUE='#7B9CAF'; TEAL='#86A79B'; OCHRE='#C99A70'; GREY='#B9BDC2'; INK='#222222'; GRID='#E7E9EA'
old.BLUE=BLUE;old.TEAL=TEAL;old.OCHRE=OCHRE;old.INK=INK;old.GRID=GRID
old.COLORS.update({'TxPert_frozen':GREY,'Anchor_Ridge1':BLUE,'ResidualLinearCV':TEAL,
 'ResidualLinearCV_UnionInput32':TEAL,'FullReadout_CopyMean32':OCHRE,'Scouter_full':GREY,
 'Anchor11':BLUE,'GenePT_Ridge1':OCHRE})
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':6.7,'axes.titlesize':7,'axes.labelsize':6.5,
 'xtick.labelsize':6,'ytick.labelsize':6,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.linewidth':.6,'lines.linewidth':1.05})
NAME_MAP={'Supplementary_Figure_6_1':'Supplementary_Figure_14',
 'Supplementary_Figure_6_5':'Supplementary_Figure_15','Supplementary_Figure_6_2':'Supplementary_Figure_16',
 'Supplementary_Figure_6_3':'Supplementary_Figure_17','Supplementary_Figure_6_4':'Supplementary_Figure_18'}
def table(study,name):
    return pd.read_csv(INPUT/'fig6'/study/name)

ParentPage=old.Page
class Page(ParentPage):
    def __init__(self,name,height=220):
        super().__init__(NAME_MAP.get(name,name),height)
        self.pending=[]
    def panel(self,letter,title,x,y,center=None):
        if self.name=='Supplementary_Figure_18' and letter in ['f','g']:y+=1.7
        if self.name=='Supplementary_Figure_16' and letter=='d':center=91.5
        self.text(x,y,letter,9,weight='bold')
        self.pending.append((title,x,y,center))
    def save(self,crops=None):
        for title,x,y,center in self.pending:
            if center is None:
                if title in ['CA','Global ρ','Top-10','ΔCA','Δρ','ΔTop-10']:
                    below=[a for a in self.axes if a.get_position().y1*self.h < y+1]
                    a=min(below,key=lambda a:abs(a.get_position().x0*self.w-x-10)+abs(a.get_position().y1*self.h-y))
                    center=(a.get_position().x0+a.get_position().x1)*self.w/2
                else:center=47 if x<95 else 141
            self.text(center,y,title,7.2,ha='center',weight='bold')
        for artist in self.fig.findobj(mpl.text.Text):
            artist.set_text(artist.get_text().replace(' · ', ', ').replace('·', ','))
            if artist.get_text()=='COX7B':artist.set_color(BLUE)
            if artist.get_text()=='PSMG3':artist.set_color(OCHRE)
        for ax in self.axes:
            for which in ['x','y']:
                labels=getattr(ax,f'get_{which}ticklabels')()
                if any('·' in label.get_text() for label in labels):
                    getattr(ax,f'set_{which}ticks')(getattr(ax,f'get_{which}ticks')(),
                        labels=[label.get_text().replace(' · ', ', ').replace('·', ',') for label in labels])
        self.fig.canvas.draw()
        report=old.record_panel_positions(self.fig,axes=self.axes,panel_ids=self.ids,
          row_groups=self.rows,column_groups=self.cols,json_out=LAYOUT/f'{self.name}.alignment.json',
          overlay_svg=LAYOUT/f'{self.name}.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
        self.fig.savefig(OUT/f'{self.name}.pdf',dpi=600)
        self.fig.savefig(OUT/f'{self.name}.svg',dpi=600)
        self.fig.savefig(OUT/f'{self.name}.png',dpi=300)
        for key,box in (crops or {}).items():
            bbox=mpl.transforms.Bbox.from_extents(*[v/25.4 for v in box])
            self.fig.savefig(LAYOUT/f'{self.name}_{key}.png',dpi=300,bbox_inches=bbox)
        print(self.name,report['status'],flush=True);plt.close(self.fig)
old.Page=Page

def pt(ax,r,metric,y,color=TEAL,delta=False,lo='simultaneous_low',hi='simultaneous_high',marker='o',contrast=False):
    v=r['delta'] if delta else r[metric]
    l=r[lo] if delta else r[metric+'_ci_low'];h=r[hi] if delta else r[metric+'_ci_high']
    ax.errorbar(v,y,xerr=[[v-l],[h-v]],fmt=marker,ms=3.7 if contrast else 3.5,
      mfc=color,mec='white' if contrast else INK,mew=.45 if contrast else .3,
      ecolor=color,elinewidth=1,capsize=2)

def matrix(p,x,y,fill,rows=7,cols=2):
    for r in range(rows):
      for c in range(cols):
        measured=fill(r,c)
        p.cv.add_patch(Rectangle((x+c*8,y-r*2.5),6.5,1.8,fc=measured or 'white',
          ec=measured or TEAL,lw=.45,ls='-' if measured else (0,(2,1))))

def context_node(p,x,y,label,color):
    p.cv.add_patch(Ellipse((x,y),12,8,fc=mpl.colors.to_rgba(color,.15),ec=color,lw=.7))
    p.cv.add_patch(Ellipse((x+1,y+.4),3,2.5,fc=mpl.colors.to_rgba(color,.35),ec='none'))
    p.text(x,y-6.5,label,6.3,ha='center')

def direct_reference(p):
    """Direct query transfer and its zero-shot and matched-input comparators."""
    x0,pitch,w,h=22.,4.25,3.65,3.1
    source_y,target_y=224.,210.
    qx=x0+12*pitch+w/2
    query_color='#47776D'
    # Paired identities calibrate both the learned map and the mean-copy comparator.
    left=x0;right=x0+3*pitch+w
    p.cv.add_patch(Rectangle((left,target_y+h/2),right-left,source_y-target_y-h,
        fc=mpl.colors.to_rgba(BLUE,.16),ec='none',zorder=1))
    for c in range(4):
        xx=x0+c*pitch+w/2
        p.cv.plot([xx,xx],[target_y+h/2,source_y-h/2],lw=.45,color=BLUE,alpha=.65,zorder=2)
    for row,(label,yy) in enumerate([('K562',source_y),('RPE1',target_y)]):
        p.text(18.5,yy,label,6.3,ha='right')
        for c in range(13):
            color=BLUE if c<4 else '#D5DDE0'
            fill=color if label=='RPE1' or c<4 else 'white'
            edge=BLUE if c<4 else '#BCC7CC'
            if c==12:fill=query_color if row==0 else 'white';edge=query_color
            p.cv.add_patch(Rectangle((x0+c*pitch,yy-h/2),w,h,fc=fill,ec=edge,lw=.5,
                ls=(0,(2,1.5)) if row==1 and c==12 else '-',zorder=3))
    p.text(40.5,219.2,'2–32 pairs',6.0)
    p.text(qx,228.8,'Query',6.0,ha='center',color=query_color)
    p.arrow((qx,source_y-h/2-.6),(qx,target_y+h/2+.6),query_color,
        ls=(0,(2,1.5)),lw=.9,scale=5.8)
    p.text(qx-2,215.2,'Transfer',6.0,ha='right',color=query_color)
    p.cv.add_patch(FancyArrowPatch((qx+1.5,source_y-h/2-.3),(qx+1.5,target_y+h/2+.3),
        connectionstyle='arc3,rad=-.95',arrowstyle='-|>',mutation_scale=5.5,
        lw=.8,linestyle=(0,(2,1.5)),color=OCHRE,shrinkA=0,shrinkB=0))
    p.text(85.2,217,'Copy\n+ mean',5.8,ha='center',color='#916B47',linespacing=1.15)
    p.text(28.5,197,'Gene identity',6.1,ha='center')
    p.arrow((40.5,197),(47,197),GREY,lw=.7,scale=5.5)
    p.text(55.5,197,'TxPert',6.2,ha='center')
    p.arrow((63,198),(qx-1,207.8),GREY,ls=(0,(2,1.5)),lw=.8,scale=5.5)

def resource_bridges(p):
    """Schematic identity-aligned resource strips, with disjoint calibration bridges."""
    x0,pitch,w,h=106.,3.7,3.15,3.1
    centers=[224.,212.,200.]
    first=set(range(10,14));second=set(range(3,7));query=17
    bridge_colors=['#7B9CAF','#C99A70']
    neutral_fill='#D5DDE0';neutral_edge='#BCC7CC';query_color='#47776D'

    # The bridge columns denote paired training identities, not a change in query identity.
    for cols,upper,lower,color in [(first,centers[0],centers[1],bridge_colors[0]),
                                   (second,centers[1],centers[2],bridge_colors[1])]:
        left=x0+min(cols)*pitch;right=x0+max(cols)*pitch+w
        top=upper-h/2;bottom=lower+h/2;middle=(top+bottom)/2
        vertices=[(left,top),(left+1.1,middle),(left+1.1,middle),(left,bottom),
                  (right,bottom),(right-1.1,middle),(right-1.1,middle),(right,top),(left,top)]
        codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,
               MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
        p.cv.add_patch(PathPatch(MplPath(vertices,codes),fc=mpl.colors.to_rgba(color,.16),ec='none',zorder=1))
        for c in sorted(cols):
            xx=x0+c*pitch+w/2
            p.cv.plot([xx,xx],[bottom,top],lw=.45,color=color,alpha=.65,zorder=2)
        p.text(left-3 if cols==first else right+3,middle,'50 pairs',6.0,color=INK,
            ha='right' if cols==first else 'left')

    rpe1_unpaired = {0, 2, 7, 9, 14, 16, 18}
    for row,(label,yy) in enumerate(zip(['K562','RPE1','Jurkat'],centers)):
        p.text(101,yy,label,6.3,ha='right')
        for c in range(19):
            col=bridge_colors[0] if c in first and row in [0,1] else (
                bridge_colors[1] if c in second and row in [1,2] else None)
            fill=col or (neutral_fill if label == 'RPE1' and c in rpe1_unpaired else 'white')
            edge=col or neutral_edge
            if c==query:
                fill=query_color if row==0 else 'white';edge=query_color
            p.cv.add_patch(Rectangle((x0+c*pitch,yy-h/2),w,h,fc=fill,ec=edge,
                lw=.55 if c==query else .4,ls=(0,(2,1.5)) if c==query and row>0 else '-',zorder=3))
        if row<2:
            qx=x0+query*pitch+w/2
            p.arrow((qx,yy-h/2-.7),(qx,centers[row+1]+h/2+.7),query_color,
                    ls=(0,(2,1.5)),lw=.9,scale=5.8)
    p.text(x0+query*pitch+w/2,228.8,'Query',6.0,ha='center',color=query_color)

def main():
    import calibrated_transfer_panels
    calibrated_transfer_panels.main(sys.modules[__name__])

def summary_panel(p,frame,methods,labels,x,y,w,h,id,metric,colors,lim):
    a=p.ax(x,y,w,h,id)
    for j,m in enumerate(methods):
        r=frame[frame.method.eq(m)];assert len(r)==1,(id,m,len(r));pt(a,r.iloc[0],metric,j,colors[j])
    a.set(ylim=(len(methods)-.5,-.5),yticks=range(len(methods)),yticklabels=labels,xlim=lim)
    a.tick_params(axis='y',length=0);old.clean(a,'x');a.set_xlabel(old.METRIC_NAMES[metric])
    return a

def s14():
    p=Page('Supplementary_Figure_14',218)
    p.text(3,213,'Paired interventions and transfer-map selection',8.1,weight='bold')
    methods=['TxPert_frozen','Anchor_Ridge1','Anchor_RidgeCV','ResidualLinearCV']
    metrics=['CA','rho','top10']
    xs=[16,76,136]
    for row,(k,y) in enumerate([(10,168),(32,118)]):
        frame=old.read('benchmark_n10.csv') if k==10 else old.BENCH
        p.text(3,y+35,f'Unscaled maps, {k} paired interventions',7,weight='bold')
        ids=[]
        for j,(met,x,upper) in enumerate(zip(metrics,xs,[.82,.49,.26])):
            tag=f's{row}{j}';ids.append(tag);ax=p.ax(x,y,43,25,tag)
            old.bars(ax,frame[frame.reference.eq(old.PRIMARY)],methods,met,(0,upper),
                     labels=['TxPert','Ridge','Ridge CV','Residual'])
            ax.tick_params(axis='x',labelsize=5.25,pad=2)
            p.panel(chr(97+row*3+j),old.METRIC_NAMES[met],x-13,y+29,x+21.5)
        p.rows.append(ids)
    p.text(3,103,'Unscaled maps across calibration sizes',7,weight='bold')
    for j,met in enumerate(['rho','top10','CA']):
        x=xs[j];ax=p.ax(x,68,43,25,f'q{j}')
        if met=='CA':old.ca_curve(ax)
        else:old.gain_curve(ax,met,old.SECONDARY,True)
        ax.set_xlabel('')
        p.panel(chr(103+j),['Δ global ρ','ΔTop-10','CA'][j],x-13,97,x+21.5)
    p.rows.append(['q0','q1','q2'])
    p.key(29,59,'Ridge',BLUE,marker='s');p.key(72,59,'Residual',TEAL,marker='o')
    p.key(117,59,'TxPert',GREY,(0,(3,2)))
    p.text(3,53,'Joint versus MSE selection after amplitude calibration',7,weight='bold')
    vals=pd.read_csv(INPUT/'fig6/calibrated_transfer/joint_selection_gain.csv')
    for j,(met,x) in enumerate(zip(metrics,xs)):
        ax=p.ax(x,18,43,25,f'j{j}')
        rr=vals.loc[vals.metric.eq(met)].sort_values('k')
        assert len(rr)==30
        ax.axhline(0,ls='--',lw=.6,color='#858D91')
        ax.fill_between(rr.k.to_numpy(),rr.simultaneous_low.to_numpy(),rr.simultaneous_high.to_numpy(),
                        color=TEAL,lw=0,alpha=.28)
        ax.plot(rr.k,rr.delta,color='#417C70',lw=1.15)
        sample=rr.loc[rr.k==12].iloc[0];ax.scatter([12],[sample.delta],s=12,color='#417C70',zorder=4)
        ax.set(xlim=(2,33),xticks=[3,12,22,32],xlabel='Paired interventions')
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(4));old.clean(ax)
        p.panel(chr(106+j),['ΔCA','Δ global ρ','ΔTop-10'][j],x-13,47,x+21.5)
    p.rows.append(['j0','j1','j2'])
    p.cols=[[f's0{i}',f's1{i}',f'q{i}',f'j{i}'] for i in range(3)]
    p.save()

def s16():
    data=pd.read_csv(INPUT/'fig6/calibrated_transfer/direct_transfer.csv')
    old.supplement2(data.loc[data.method.eq('Calibrated map')].set_index('metric'))

def s17():
    from matplotlib.colors import ListedColormap
    src=INPUT/'fig6/calibrated_transfer'
    cases=pd.read_csv(src/'triplet_repair_by_case.csv')
    fixed=pd.read_csv(src/'triplet_orders_by_panel.csv')
    counts=pd.read_csv(src/'triplet_counts_by_fold.csv')
    rates=pd.read_csv(src/'triplet_population_rates.csv')
    primary, secondary='paired11_primary','TxPert_train_secondary'
    jit=np.linspace(-0.13,0.13,5)
    p=Page('Supplementary_Figure_17',223)
    p.text(3,217,'Repair of intervention relationships by cross-context transfer',8.1,weight='bold')
    p.panel('a','Original TxPert inversions',3,207)
    ordered=pd.read_csv(INPUT/'fig6/fixed_26_original.csv')
    keys=[tuple(r) for r in ordered[['fold','A','B','C']].itertuples(index=False,name=None)]
    heat=[];ticklabels=[]
    for fold,A,B,C in keys:
        q=cases[cases.fold.eq(fold)&cases.A.eq(A)&cases.B.eq(B)&cases.C.eq(C)&
                cases.reference.eq(secondary)].sort_values('panel')
        assert len(q)==5
        heat.append(q.correct.astype(int).to_numpy());ticklabels.append(f'{A} / {C} / {B}')
    a=p.ax(54,94,35,103,'a')
    a.imshow(np.asarray(heat),aspect='auto',interpolation='nearest',cmap=ListedColormap(['#E6E9E9',TEAL]),vmin=0,vmax=1)
    a.set(yticks=np.arange(26),yticklabels=ticklabels,xticks=np.arange(5),xticklabels=['1','2','3','4','5'])
    a.tick_params(axis='y',labelsize=5.65,length=0,pad=3)
    a.set_xticks(np.arange(-.5,5,1),minor=True);a.set_yticks(np.arange(-.5,26,1),minor=True)
    a.grid(which='minor',color='white',lw=.8);a.tick_params(which='minor',length=0)
    a.set_xlabel('Calibration panel')
    p.text(3,201,'Reference / related / contrasting',6,color='#697578')
    p.cv.add_patch(Rectangle((7,85),3,2.5,facecolor=TEAL,edgecolor='none'));p.text(11,86.25,'Corrected',6.1)
    p.cv.add_patch(Rectangle((45,85),3,2.5,facecolor='#E6E9E9',edgecolor='none'));p.text(49,86.25,'Still inverted',6.1)

    p.panel('b','Correct orders across references',101,207)
    b=p.ax(117,153,59,43,'b')
    for i,ref in enumerate([secondary,primary]):
        q=fixed[fixed.reference.eq(ref)].sort_values('panel')
        vals=q.correct.to_numpy()
        b.bar(i,vals.mean(),width=.55,color=TEAL if i==0 else BLUE,edgecolor=INK,lw=.6)
        b.scatter(i+jit,vals,s=14,color=INK,alpha=.6,zorder=3,edgecolor='none')
    b.set(xlim=(-.6,1.6),ylim=(0,27),yticks=[0,13,26],xticks=[0,1],
          xticklabels=['Training mean','Paired mean'],ylabel='Correct / 26');old.clean(b)
    p.text(146,144,'Five calibration panels',6.2,color=INK,ha='center')

    p.panel('c','Corrections and new errors',101,134)
    c=p.ax(117,94,59,30,'c')
    q=counts[counts.quality_mode.eq('all_three_four_criteria')&counts.eligibility.eq('fully_rematched_rule')&
             counts.reference.eq(primary)]
    bypanel=q.groupby('panel')[['baseline_errors_repaired','newly_damaged']].sum()
    for i,(col,color) in enumerate([('baseline_errors_repaired',TEAL),('newly_damaged',OCHRE)]):
        vals=bypanel[col].to_numpy();assert len(vals)==5
        c.bar(i,vals.mean(),width=.55,color=color,edgecolor=INK,lw=.6)
        c.scatter(i+jit,vals,s=12,color=INK,alpha=.6,zorder=3,edgecolor='none')
    c.set(xlim=(-.6,1.6),ylim=(0,max(bypanel.max())+5),xticks=[0,1],
          xticklabels=['Corrected','New errors'],ylabel='Triplets');old.clean(c)

    p.panel('d','Overall inversion rate',3,70)
    d=p.ax(37,22,52,38,'d')
    sub=rates[rates.quality_mode.eq('all_three_four_criteria')&rates.eligibility.eq('fully_rematched_rule')]
    for i,ref in enumerate([primary,secondary]):
        r=sub[sub.reference.eq(ref)].iloc[0]
        for shift,col,color in [(-.12,'baseline_inversion_rate','#959da0'),(.12,'inversion_rate',TEAL)]:
            v,lo,hi=(100*r[col+suffix] for suffix in ['', '_low','_high'])
            d.errorbar(v,i+shift,xerr=[[v-lo],[hi-v]],fmt='o',ms=4,mfc=color,mec=INK,mew=.35,
                       ecolor=color,capsize=2)
    d.set(xlim=(0,20),ylim=(1.5,-.5),yticks=[0,1],yticklabels=['Paired mean','Training mean'],
          xlabel='Inverted triplets (%)');old.clean(d,'x')
    p.key(15,7,'TxPert','#959da0',marker='o');p.key(46,7,'Calibrated map',TEAL,marker='o')
    p.panel('e','Net reduction in inversion rate',101,70)
    e=p.ax(128,22,48,38,'e')
    configs=[('fully_rematched_rule',primary),('fully_rematched_rule',secondary),
             ('frozen_quality_current_biology',primary),('frozen_quality_current_biology',secondary)]
    rec=[]
    for el,ref in configs:
        r=rates[rates.quality_mode.eq('all_three_four_criteria')&
               rates.eligibility.eq(el)&rates.reference.eq(ref)].iloc[0]
        rec.append(tuple(r[c]*100 for c in ['net_error_reduction_rate','net_error_reduction_rate_low','net_error_reduction_rate_high']))
    old.horizontal_points(e,rec,(-12,25),['Strict, paired','Strict, training','Original quality, paired','Original quality, training'],
                      [TEAL,TEAL,BLUE,BLUE]);e.tick_params(axis='y',labelsize=5.8)
    e.axvline(0,color=GREY,ls=(0,(3,2)),lw=.65);e.set_xlabel('Percentage points')
    p.rows.append(['d','e'])
    p.save({'top':(0,79,183,223),'bottom':(0,0,183,79)})

def s19():
    p=Page('Supplementary_Figure_19',213)
    p.panel('a','Unscaled transfer with all available pairs',3,206,91.5)
    full=table('all_pairs','SUMMARY.csv');full=full[full.context.eq('C')]
    methods=['TxPert_A_to_C','Measured_B_to_A_to_C','Direct_B_Copy','Wrong_B_to_A_to_C','Oracle_measured_A_to_C']
    labelmap={'TxPert_A_to_C':'TxPert route','Measured_B_to_A_to_C':'Measured two-hop','Direct_B_Copy':'Copy',
      'Wrong_B_to_A_to_C':'Wrong identity','Oracle_measured_A_to_C':'Measured RPE1'}
    fullfold=table('all_pairs','FOLD_METRICS.csv')
    for i,metric in enumerate(['rho','CA']):
        a=summary_panel(p,full,methods,[labelmap.get(m,m.replace('_',' ')) for m in methods] if i==0 else ['']*len(methods),
          47+73*i,164,58,32,f'a{i}',metric,[TEAL if m.startswith('Measured_B') else OCHRE if 'Copy' in m else GREY for m in methods],(-.1,.85))
        for j,m in enumerate(methods):
            q=fullfold[fullfold.method.eq(m)].sort_values('fold')
            a.scatter(q[metric],j+np.linspace(-.15,.15,len(q)),s=5,color=INK,alpha=.35,lw=0,zorder=4)
    p.rows.append(['a0','a1'])
    frozen=table('two_hop','SUMMARY.csv')
    fold=table('two_hop','FOLD_METRICS.csv')
    for j,(route,title) in enumerate([('K562_RPE1_Jurkat','Unscaled transfer to Jurkat, 50 pairs per link'),('Jurkat_RPE1_K562','Unscaled transfer to K562, 50 pairs per link')]):
        y=114-j*50;p.panel(chr(98+j),title,3,y+41,91.5)
        frame=frozen[frozen.route.eq(route)]
        methods=['TxPert_one_hop','Measured_two_hop','Source_Copy','Wrong_source_two_hop','Measured_A_diagnostic']
        labels=['TxPert route','Measured two-hop','Copy','Wrong identity','Measured RPE1']
        for i,metric in enumerate(['rho','CA']):
            a=summary_panel(p,frame,methods,labels if i==0 else ['']*5,47+73*i,y,58,31,f'r{j}{i}',metric,
              [GREY,TEAL,OCHRE,GREY,BLUE],(-.1,.9))
            for k,m in enumerate(methods):
                q=fold[fold.route.eq(route)&fold.method.eq(m)].sort_values('fold')
                a.scatter(q[metric],k+np.linspace(-.15,.15,len(q)),s=5,color=INK,alpha=.35,lw=0,zorder=4)
        p.rows.append([f'r{j}0',f'r{j}1'])
    p.panel('d','CA before and after common terminal scaling',3,53,91.5)
    sensitivity=pd.read_csv(INPUT/'fig6/calibrated_transfer/bridge_before_after_scaling.csv')
    for i,(rt,label) in enumerate([('K562_RPE1_Jurkat','K562 → Jurkat'),('Jurkat_RPE1_K562','Jurkat → K562')]):
        ax=p.ax(47+73*i,18,58,27,'d'+str(i))
        frame=sensitivity.loc[sensitivity.route.eq(rt)]
        for m,col,mk in [('learned','#417C70','o'),('txroute','#86929B','s')]:
            vals=frame.loc[frame.method.eq(m),['CA_before_terminal_scale','CA']].mean()
            ax.plot([0,1],vals,color=col,marker=mk,ms=3.3,lw=1)
        ax.set(xlim=(-.35,1.35),xticks=[0,1],xticklabels=['Unscaled','Scaled'],
               ylim=(.54,.78),yticks=[.55,.65,.75],ylabel='CA' if i==0 else '')
        old.clean(ax);p.text(76+73*i,48,label,6.1,ha='center')
    p.key(56,5,'Measured two-hop','#417C70',marker='o');p.key(108,5,'TxPert route','#86929B',marker='s')
    p.rows.append(['d0','d1'])
    p.cols=[['a0','r00','r10','d0'],['a1','r01','r11','d1']]
    p.save()

def s20():
    p=Page('Supplementary_Figure_20',257)
    cp=table('bridge_copy','SUMMARY.csv')
    direct=table('direct_pairs','SUMMARY.csv')
    for row,(frame,routes,title,methods,labels) in enumerate([
       (cp,['K562_RPE1_Jurkat','Jurkat_RPE1_K562'],'Unscaled transfer across two bridges',['Source_Copy','Bridge_mean_corrected_Copy','Measured_two_hop'],['Copy','Calibrated Copy','Two-hop']),
       (direct,['K562_to_Jurkat','Jurkat_to_K562'],'50 direct pairs',['Source_Copy','Mean_corrected_Copy','Direct_residual_transfer'],['Copy','Calibrated Copy','Transfer'])]):
      for j,route in enumerate(routes):
        y=205-row*101-j*43
        for i,metric in enumerate(['rho','CA']):
          id=f'{row}{j}{i}';x=43+74*i
          a=summary_panel(p,frame[frame.route.eq(route)],methods,labels if i==0 else ['']*3,x,y,60,28,id,metric,[OCHRE,BLUE,TEAL],(.25,.56) if metric=='rho' else (.4,.9))
          a.set_xlabel('' if j==0 else old.METRIC_NAMES[metric])
        p.rows.append([f'{row}{j}0',f'{row}{j}1'])
        p.text(16,y+14,'K→J' if j==0 else 'J→K',6.4,ha='center',weight='bold')
      p.panel(chr(97+row),title,3,243-row*101,91.5)
    p.text(73,237,'Global ρ',6.7,ha='center');p.text(147,237,'CA',6.7,ha='center')
    p.panel('c','Learning beyond mean correction',3,44,91.5)
    bridge=table('bridge_copy','PRIMARY_SIMULTANEOUS_CI.csv')
    direct_delta=table('direct_pairs','PRIMARY_SIMULTANEOUS_CI.csv')
    for i,metric in enumerate(['rho','CA']):
      a=p.ax(43+74*i,10,60,26,f'paired{i}')
      for row,(frame,routes) in enumerate([(bridge,['K562_RPE1_Jurkat','Jurkat_RPE1_K562']),
                 (direct_delta,['K562_to_Jurkat','Jurkat_to_K562'])]):
        for j,route in enumerate(routes):
          r=frame[frame.route.eq(route)&frame.metric.eq(metric)].iloc[0]
          pt(a,r,metric,row*2+j,BLUE if row==0 else TEAL,True)
      a.set(ylim=(3.5,-.5),yticks=range(4),yticklabels=['Bridge K→J','Bridge J→K','Direct K→J','Direct J→K'] if i==0 else ['']*4,
            xlim=(-.05,.05) if metric=='rho' else (-.3,0),xlabel='Δρ' if metric=='rho' else 'ΔCA')
      a.axvline(0,ls='--',color=GREY,lw=.7);a.tick_params(axis='y',length=0);old.clean(a,'x')
    p.rows.append(['paired0','paired1'])
    p.save()
