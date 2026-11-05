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
            fill=color if row==0 or c<4 else 'white'
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

    measured=[set(range(19))-second-{1,8,15},first|second,second]
    for row,(label,yy) in enumerate(zip(['K562','RPE1','Jurkat'],centers)):
        p.text(101,yy,label,6.3,ha='right')
        for c in range(19):
            col=bridge_colors[0] if c in first and row in [0,1] else (
                bridge_colors[1] if c in second and row in [1,2] else None)
            fill=col or (neutral_fill if c in measured[row] else 'white')
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
    p=Page('Figure_6',248)
    p.panel('a','Measure interventions, connect contexts',3,242,91.5)
    p.text(47,232.7,'Direct transfer',6.5,ha='center',weight='bold')
    p.text(139,232.7,'Sparse bridges',6.5,ha='center',weight='bold')
    for xx,label,predicted in [(66,'Measured',False),(94,'Predicted',True)]:
        p.cv.add_patch(Rectangle((xx,236.4),3.7,2.4,fc='white' if predicted else '#D5DDE0',
            ec='#BCC7CC',lw=.5,ls=(0,(2,1.5)) if predicted else '-'))
        p.text(xx+5.3,237.6,label,6.0,color='#5E676C')
    direct_reference(p)
    resource_bridges(p)
    p.panel('b','Sparse transfer versus zero-shot',3,185,91.5)
    for i,(metric,lim) in enumerate([('CA',(0,.8)),('rho',(0,.48)),('top10',(0,.25))]):
        x=18+59*i;a=p.ax(x,142,43,32,f'b{i}')
        old.bars(a,old.BENCH[old.BENCH.reference.eq(old.PRIMARY)],['TxPert_frozen','Anchor_Ridge1','ResidualLinearCV'],metric,lim)
        p.text(x+21.5,177,old.METRIC_NAMES[metric],6.8,ha='center')
    p.rows.append(['b0','b1','b2'])
    p.panel('c','Adaptation with matched inputs',3,127,48)
    p.panel('d','Fewer paired interventions',101,127,143)
    same=old.read('matched_input_n32.csv');same=same[same.reference.eq(old.PRIMARY)]
    for i,(metric,lim) in enumerate([('CA',(0,.8)),('rho',(0,.48)),('top10',(0,.28))]):
        x=12+29*i;a=p.ax(x,92,20,25,f'c{i}')
        old.bars(a,same,['FullReadout_CopyMean32','ResidualLinearCV_UnionInput32'],metric,lim,['Copy','Residual'])
        for t in a.get_xticklabels():t.set_rotation(40);t.set_ha('right');t.set_rotation_mode('anchor')
        p.text(x+10,120,old.METRIC_NAMES[metric],6.5,ha='center')
    p.rows.append(['c0','c1','c2'])
    for i,metric in enumerate(['rho','top10']):
        x=111+39*i;a=p.ax(x,92,28,25,f'd{i}');old.gain_curve(a,metric,mark_thresholds=True)
        p.text(x+14,120,'Δρ' if i==0 else 'ΔTop-10',6.5,ha='center')
    p.rows.append(['d0','d1']);p.text(145,83,'Paired interventions',6.1,ha='center')
    p.key(110,77,'Ridge',BLUE,marker='s');p.key(143,77,'Residual',TEAL,marker='o')
    p.panel('e','Biological relationship repair',3,70,47)
    p.panel('f','Sparse bridges versus TxPert route',99,70,140)
    from biology_art import BiologyArt
    BiologyArt(p.cv,INK,BLUE,OCHRE).mitochondrion(26,59)
    for yy in [56,58,60,62]:
        p.cv.add_patch(Ellipse((70,yy),10,3,fc='#EDF0F1',ec='#9BA6AC',lw=.55))
        for dx in [-3,0,3]:p.cv.plot([70+dx,70+dx],[yy-.8,yy+.8],color='#ABB4B8',lw=.4)
    p.cv.add_patch(Ellipse((76,61),3,2.7,angle=30,fc=OCHRE,ec=OCHRE,lw=.4))
    p.text(24.7,51.5,'COX5A',5.8,ha='right');p.text(26,51.5,'/',5.8,ha='center')
    p.text(27.3,51.5,'COX7B',5.8,color=BLUE);p.text(70,51.5,'PSMG3',5.8,ha='center',color=OCHRE)
    old.vector_case(p,26,27,'TxPert_frozen','TxPert');old.vector_case(p,70,27,'ResidualLinearCV','Residual')
    p.text(47,19,'23/26 previous inversions corrected',6.6,ha='center')
    p.key(15,11,'Observed',INK);p.key(51,11,'Predicted',INK,(0,(2,2)))
    primary=table('two_hop','PRIMARY_JOINT_FOUR_ENDPOINT_CI.csv')
    cp=table('bridge_copy','RESOURCE_SIMULTANEOUS_CI.csv')
    bridge_blue='#4F788A';copy_grey='#62666D';calibrated_ochre='#BE8D5C'
    for i,metric in enumerate(['rho','CA']):
        a=p.ax(121+34*i,17,25,40,f'f{i}');a.axvline(0,color=GREY,lw=.7,ls='--')
        for j,route in enumerate(['K562_RPE1_Jurkat','Jurkat_RPE1_K562']):
            r=primary[primary.route.eq(route)&primary.metric.eq(metric)].iloc[0]
            pt(a,r,metric,j*4,bridge_blue,True,'joint4_low','joint4_high',marker='o',contrast=True)
            for k,(method,c,marker) in enumerate([('Source_Copy',copy_grey,'s'),
                    ('Bridge_mean_corrected_Copy',calibrated_ochre,'D')]):
                r=cp[cp.route.eq(route)&cp.method.eq(method)&cp.metric.eq(metric)].iloc[0]
                pt(a,r,metric,j*4+k+1,c,True,marker=marker,contrast=True)
        a.set(ylim=(7,-1),xlim=(-.03,.34),xticks=[0,.15,.3],yticks=[0,1,2,4,5,6],
              yticklabels=['Two-hop','Copy','Cal. Copy']*2 if i==0 else ['']*6)
        a.tick_params(axis='y',length=0,pad=3,labelsize=6.0);old.clean(a,'x')
        a.spines['left'].set_visible(False)
        p.text(133.5+34*i,62,'Δρ' if i==0 else 'ΔCA',6.6,ha='center')
    p.rows.append(['f0','f1'])
    p.text(100,57.3,'K562 → Jurkat',6.2,weight='bold')
    p.text(100,37.3,'Jurkat → K562',6.2,weight='bold')
    p.save(crops={'concept':(2,190,181,246),'comparison':(98,10,182,73)})

def summary_panel(p,frame,methods,labels,x,y,w,h,id,metric,colors,lim):
    a=p.ax(x,y,w,h,id)
    for j,m in enumerate(methods):
        r=frame[frame.method.eq(m)];assert len(r)==1,(id,m,len(r));pt(a,r.iloc[0],metric,j,colors[j])
    a.set(ylim=(len(methods)-.5,-.5),yticks=range(len(methods)),yticklabels=labels,xlim=lim)
    a.tick_params(axis='y',length=0);old.clean(a,'x');a.set_xlabel(old.METRIC_NAMES[metric])
    return a

def s19():
    p=Page('Supplementary_Figure_19',213)
    p.panel('a','Full bridge pools',3,205,91.5)
    full=table('all_pairs','SUMMARY.csv');full=full[full.context.eq('C')]
    methods=['TxPert_A_to_C','Measured_B_to_A_to_C','Direct_B_Copy','Wrong_B_to_A_to_C','Oracle_measured_A_to_C']
    labelmap={'TxPert_A_to_C':'TxPert route','Measured_B_to_A_to_C':'Measured two-hop','Direct_B_Copy':'Copy',
      'Wrong_B_to_A_to_C':'Wrong identity','Oracle_measured_A_to_C':'Measured RPE1'}
    fullfold=table('all_pairs','FOLD_METRICS.csv')
    for i,metric in enumerate(['rho','CA']):
        a=summary_panel(p,full,methods,[labelmap.get(m,m.replace('_',' ')) for m in methods] if i==0 else ['']*len(methods),
          47+73*i,151,58,42,f'a{i}',metric,[TEAL if m.startswith('Measured_B') else OCHRE if 'Copy' in m else GREY for m in methods],(-.1,.85))
        for j,m in enumerate(methods):
            q=fullfold[fullfold.method.eq(m)].sort_values('fold')
            a.scatter(q[metric],j+np.linspace(-.15,.15,len(q)),s=5,color=INK,alpha=.35,lw=0,zorder=4)
    p.rows.append(['a0','a1'])
    frozen=table('two_hop','SUMMARY.csv')
    fold=table('two_hop','FOLD_METRICS.csv')
    for j,(route,title) in enumerate([('K562_RPE1_Jurkat','50 + 50 pairs: K562 → Jurkat'),('Jurkat_RPE1_K562','50 + 50 pairs: Jurkat → K562')]):
        y=85-j*66;p.panel(chr(98+j),title,3,y+54,91.5)
        frame=frozen[frozen.route.eq(route)]
        methods=['TxPert_one_hop','Measured_two_hop','Source_Copy','Wrong_source_two_hop','Measured_A_diagnostic']
        labels=['TxPert route','Measured two-hop','Copy','Wrong identity','Measured RPE1']
        for i,metric in enumerate(['rho','CA']):
            a=summary_panel(p,frame,methods,labels if i==0 else ['']*5,47+73*i,y,58,42,f'r{j}{i}',metric,
              [GREY,TEAL,OCHRE,GREY,BLUE],(-.1,.9))
            for k,m in enumerate(methods):
                q=fold[fold.route.eq(route)&fold.method.eq(m)].sort_values('fold')
                a.scatter(q[metric],k+np.linspace(-.15,.15,len(q)),s=5,color=INK,alpha=.35,lw=0,zorder=4)
        p.rows.append([f'r{j}0',f'r{j}1'])
    p.cols=[['a0','r00','r10'],['a1','r01','r11']]
    p.save()

def s20():
    p=Page('Supplementary_Figure_20',257)
    cp=table('bridge_copy','SUMMARY.csv')
    direct=table('direct_pairs','SUMMARY.csv')
    for row,(frame,routes,title,methods,labels) in enumerate([
       (cp,['K562_RPE1_Jurkat','Jurkat_RPE1_K562'],'Indirect bridges',['Source_Copy','Bridge_mean_corrected_Copy','Measured_two_hop'],['Copy','Calibrated Copy','Two-hop']),
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
