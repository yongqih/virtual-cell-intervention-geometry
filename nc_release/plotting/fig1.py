"""Response correspondence and biological relationship recovery.

Panels a/b are explicitly schematic. Empirical vector views align the two
pairwise planes independently; only the reference-relative angles are encoded.
No training, inference, experiments, resampling or new metric estimates.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Wedge, FancyArrowPatch, Circle, Ellipse, Rectangle, PathPatch
from matplotlib.path import Path as MPath

from settings import INPUT, OUTPUT as OUT, CHECKS as LAYOUT
DATA=INPUT/'fig12'
from layout import record_panel_positions
mpl.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':6.5,'axes.labelsize':6.3,'xtick.labelsize':6.0,'ytick.labelsize':6.3,
    'axes.titlesize':6.7,'axes.linewidth':.55,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','legend.frameon':False,
    'lines.linewidth':.9,'figure.facecolor':'white','savefig.facecolor':'white',
})
MODELS=['GEARS','scGPT','Scouter','CellFlow','PerturbDiff','State','TxPert']
INK='#222222'; GRAY='#777777'; LIGHT='#D8DBDE'; MODEL_GRAY='#B9BDC2'
BLUE='#7B9CAF'; ORANGE='#C99A70'; PALE_BLUE='#EAF3F7'; PALE_ORANGE='#FAF0E5'
W,H=183,221
fig=plt.figure(figsize=(183/25.4,221/25.4))
cv=fig.add_axes([0,0,1,1],label='vector_canvas');cv.set(xlim=(0,W),ylim=(0,H));cv.axis('off')
axs=[];ids=[];rows=[]

def txt(x,y,s,size=6.5,color=INK,weight='normal',ha='left',**kw):
    return cv.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va='center',**kw)

def line(x1,y1,x2,y2,c=LIGHT,lw=.6,**kw):
    return cv.plot([x1,x2],[y1,y2],color=c,lw=lw,solid_capstyle='round',**kw)

def panel(letter,title,x,y):
    txt(x,y,letter,size=9,weight='bold')
    centre={'a':47,'b':143,'c':94,'d':94,'e':94,'f':39,'g':123}[letter]
    txt(centre,y,title,size=7.2,weight='bold',ha='center')

def axis(x,y,w,h,id):
    ax=fig.add_axes([x/W,y/H,w/W,h/H],label=id);axs.append(ax);ids.append(id)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['bottom','left']].set_color(INK)
    ax.tick_params(axis='both',length=2,pad=2,width=.5,color=INK)
    return ax

def unit(angle):
    return np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])

def arrow(origin,angle,length,color,dashed=False,lw=1.0,head=6.4):
    tip=np.asarray(origin)+length*unit(angle)
    cv.add_patch(FancyArrowPatch(origin,tip,arrowstyle='-|>' if not dashed else '->',
        mutation_scale=head,linewidth=lw,color=color,shrinkA=0,shrinkB=0,
        linestyle=(0,(2.2,1.8)) if dashed else '-',zorder=4 if not dashed else 2))
    return tip

def sector(origin,theta1,theta2,r,c,fill):
    lo,hi=sorted([theta1,theta2])
    cv.add_patch(Wedge(origin,r,lo,hi,facecolor=fill,edgecolor='none',zorder=0))
    cv.add_patch(Arc(origin,2*r,2*r,theta1=lo,theta2=hi,color=c,lw=.65,zorder=1))

def curve(vertices,c,lw=.65):
    cv.add_patch(PathPatch(MPath(vertices,[MPath.MOVETO]+[MPath.CURVE4]*(len(vertices)-1)),
                           fc='none',ec=c,lw=lw,capstyle='round'))

from biology_art import BiologyArt
bio=BiologyArt(cv,INK,BLUE,ORANGE)

def style_key(x,y,gap=24):
    line(x,y,x+4,y,c=INK,lw=1.0);txt(x+5.5,y,'Observed',size=6)
    line(x+gap,y,x+gap+4,y,c=GRAY,lw=.8,ls=(0,(2.2,1.8)));txt(x+gap+5.5,y,'Predicted',size=6)

def gene_pair_label(x,y,query,partner):
    txt(x-1.2,y,query,size=6,color=INK,ha='right')
    txt(x,y,'/',size=6,color=GRAY,ha='center')
    txt(x+1.2,y,partner,size=6,color=BLUE)

# a: close underlying responses; each triplet is displayed aligned to its query.
panel('a','Accurate responses, reversed order',3,216)
angles={'Observed':{'Reference':90.,'Blue':130.,'Orange':45.},
        'Predicted':{'Reference':87.,'Blue':131.,'Orange':50.}}
display_angles={kind:{name:a+90-v['Reference'] for name,a in v.items()} for kind,v in angles.items()}
u={kind:{k:unit(v) for k,v in a.items()} for kind,a in angles.items()}
matching={k:float(u['Observed'][k]@u['Predicted'][k]) for k in u['Observed']}
assert min(matching.values())>.99 and max(matching.values())<1
dist={kind:{k:float(1-v['Reference']@v[k]) for k in ['Blue','Orange']} for kind,v in u.items()}
assert dist['Observed']['Blue']<dist['Observed']['Orange']
assert dist['Predicted']['Blue']>dist['Predicted']['Orange']
for name,pred in u['Predicted'].items():
    assert all(pred@u['Observed'][name]>pred@other for k,other in u['Observed'].items() if k!=name)
for kind,vs in display_angles.items():
    assert vs['Reference']==90.
    for name in ['Blue','Orange']:
        assert np.isclose(1-unit(vs['Reference'])@unit(vs[name]),dist[kind][name],atol=1e-12)
origin=(47.,188.5)
for kind,radius in [('Observed',6.5),('Predicted',9.0)]:
    for name,c,fill in [('Blue',BLUE,PALE_BLUE),('Orange',ORANGE,PALE_ORANGE)]:
        lo,hi=sorted([display_angles[kind]['Reference'],display_angles[kind][name]])
        if kind=='Observed':
            cv.add_patch(Wedge(origin,radius,lo,hi,fc=fill,ec='none',zorder=0))
        cv.add_patch(Arc(origin,2*radius,2*radius,theta1=lo,theta2=hi,color=c,lw=.8,
            linestyle=(0,(2.2,1.8)) if kind=='Predicted' else '-',zorder=1))
for kind in ['Observed','Predicted']:
    for name,c in [('Blue',BLUE),('Orange',ORANGE)]:
        arrow(origin,display_angles[kind][name],21,c,kind=='Predicted',.85 if kind=='Predicted' else 1.05,7.)
arrow(origin,90.,21,INK,False,1.05,7.)
cv.plot(*origin,marker='o',ms=1.8,color=INK)
txt(47,212.4,'Reference',size=6,ha='center')
style_key(19,182.5,25)
(LAYOUT/'schematic_geometry.json').write_text(json.dumps({'illustrative':True,'unaligned_angles_deg':angles,
    'reference_aligned_display_angles_deg':display_angles,
    'own_vector_cosine':matching,'reference_cosine_distances':dist,
    'all_predictions_change':True,'own_matching_preserved':True,'order_reversed':True,
    'display_reference_is_aligned_not_perfect_prediction':True,
    'not_CA_measurements':True},indent=2),encoding='utf-8')

# b: existing global/local evaluation views. Miniatures are schematic.
panel('b','Relational evaluation',102,216)
txt(145,209.5,'Observed',size=6,ha='center');txt(169,209.5,'Predicted',size=6,ha='center')
txt(104,203.8,'Global ρ',size=6.5,weight='bold');txt(104,200.3,'All pairs',size=6,color=GRAY)
vals=[[.15,.4,.75,.9,.6,.25],[.45,.8,.2,.7,.15,.6]]
for x,v in zip([140,164],vals):
    k=0
    for i in range(4):
        for j in range(4):
            if j>i:
                cv.add_patch(Rectangle((x+j*2.1,200.4-i*2.1),1.9,1.9,
                    fc=mpl.colors.to_rgba(BLUE,.2+.8*v[k]),ec='none'));k+=1
            elif i==j:
                cv.add_patch(Rectangle((x+j*2.1,200.4-i*2.1),1.9,1.9,fc='#ECEEEF',ec='none'))
txt(104,190,'Top-10',size=6.5,weight='bold');txt(104,186.5,'Neighbours',size=6,color=GRAY)
for col,x in enumerate([140,164]):
    for j in range(10):
        c=BLUE if col==0 or j<6 else MODEL_GRAY
        cv.add_patch(Circle((x+1.1+(j%5)*2.1,190-(j//5)*2.8),.75,fc=c,ec='none'))

# c: frozen benchmark, moved before representative biological examples.
panel('c','Seven-model benchmark',3,178)
E=pd.read_csv(DATA/'main_E_benchmark.csv')
endpoints=['Systema_CA','PB_expression_original_rank','PB_logFC_cosine_original_rank','PB_logFC_Frobenius_RMS','IGC_global_rho','IGC_top10']
titles=['CA ↑','PB expression\nrank ↓','PB logFC\nrank ↓','PB matrix\nerror ↓','Global ρ ↑','Top-10 ↑']
limits=[(.7,[0,.35,.7]),(.5,[0,.25,.5]),(.5,[0,.25,.5]),(.8,[0,.4,.8]),(.3,[0,.15,.3]),(.15,[0,.075,.15])]
cids=[];highlighted_models=[]
for j,(ep,title,(upper,ticks)) in enumerate(zip(endpoints,titles,limits)):
    ax=axis(22+j*26.3,140,21.6,26.5,f'c{j+1}');cids.append(f'c{j+1}')
    sub=E[E.endpoint==ep].set_index('model_display').loc[MODELS]
    est=sub.estimate.to_numpy();lo=sub.ci_low.to_numpy();hi=sub.ci_high.to_numpy()
    assert np.all((lo<=est)&(est<=hi)) and sub.higher_is_better.nunique()==1
    higher=bool(sub.higher_is_better.iloc[0]);best=int(np.argmax(est) if higher else np.argmin(est))
    assert sub['rank'].iloc[best]==1 and (sub['rank']==1).sum()==1
    colors=[BLUE if i==best else MODEL_GRAY for i in range(len(MODELS))]
    highlighted_models.append({'endpoint':ep,'higher_is_better':higher,'top1':MODELS[best],'n_colored_bars':1})
    ax.barh(np.arange(7),est,height=.58,color=colors,edgecolor='none')
    ax.errorbar(est,np.arange(7),xerr=[est-lo,hi-est],fmt='none',ecolor='#555B60',elinewidth=.65,capsize=1.5,capthick=.6)
    ax.set(xlim=(0,upper),ylim=(6.65,-.65),yticks=np.arange(7),xticks=ticks)
    ax.set_yticklabels(MODELS if j==0 else ['']*7)
    ax.set_title(title,pad=4,fontsize=6.4,linespacing=1.3)
    ax.tick_params(axis='y',length=0,pad=2.5);ax.spines['left'].set_visible(False)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,pos:f'{v:g}'))
rows.append(cids)
(LAYOUT/'benchmark_highlights.json').write_text(json.dumps(highlighted_models,indent=2),encoding='utf-8')

# d: model judgments reverse; this is not a claim that rho itself is negative.
panel('d','Reversed model judgments',3,130)
raw=pd.read_csv((DATA/'PAIRED_METRIC_INTERVALS.csv'))
display_rows=[]
for a,b,label,primary in [('PerturbDiff','State-ST-GenePT','PerturbDiff - State','Systema_CA'),('Scouter','TxPert-public','Scouter - TxPert','PB_logFC_Frobenius_RMS')]:
    for endpoint in [primary,'IGC_global_rho','IGC_top10']:
        sub=raw[(raw.model_a==a)&(raw.model_b==b)&(raw.endpoint==endpoint)]
        assert len(sub)==1,(a,b,endpoint)
        r=sub.iloc[0]
        display_rows.append(dict(comparison=label,endpoint=endpoint,estimate=r.advantage_a_over_b,ci_low=r.simultaneous_ci_low,ci_high=r.simultaneous_ci_high))
F=pd.DataFrame(display_rows)
F.to_csv(DATA/'main_d_updated_252_family.csv',index=False)
for j,(comp,x) in enumerate([('PerturbDiff - State',26),('Scouter - TxPert',112)]):
    txt(x+32,124.5,comp.replace(' - ',' − '),size=6.8,ha='center')
    sub=F[F.comparison==comp].reset_index(drop=True);ax=axis(x,105,64,14,f'd{j+1}')
    ax.axvline(0,color='#9AA0A6',lw=.65,ls=(0,(3,2)),zorder=0)
    for i,r in sub.iterrows():
        c,fill=(BLUE,PALE_BLUE) if i==0 else (ORANGE,PALE_ORANGE)
        ax.errorbar(r.estimate,2-i,xerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],fmt='o',
            ms=3.9,mfc=fill,mec=c,mew=.85,color=c,ecolor=c,elinewidth=1.05,capsize=2,capthick=.8)
    ax.set(xlim=(-.2,.06) if j==0 else (-.2,.22),ylim=(-.55,2.55),yticks=[2,1,0],xticks=[-.2,-.1,0,.05] if j==0 else [-.2,0,.2])
    ax.set_yticklabels(['CA','Global ρ','Top-10'] if j==0 else ['PB error','Global ρ','Top-10'])
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=3)
    ax.set_xlabel('Performance difference',fontsize=6.2,labelpad=3)
rows.append(['d1','d2'])

# Empirical vector views: independently aligned pairwise planes, not embeddings.
angle_records=[]
def angle_overlay(x,y,query,partners,ds,length=17.5,label_size=6.1):
    origin=(x,y);ref=90.;tips={}
    for kind in ['Observed','Predicted']:
        for pair,partner,c,fill,sign in [('AC',partners[0],BLUE,PALE_BLUE,1),('AB',partners[1],ORANGE,PALE_ORANGE,-1)]:
            d=float(ds[(ds.kind==kind)&(ds.pair==pair)].iloc[0].cosine_distance)
            assert 0<=d<=2
            theta=float(np.rad2deg(np.arccos(1-d)))
            assert np.isclose(1-np.cos(np.deg2rad(theta)),d,atol=1e-12)
            angle=ref+sign*theta;radius=5.4 if kind=='Observed' else 7.3
            lo,hi=sorted([ref,angle])
            if kind=='Observed':
                cv.add_patch(Wedge(origin,radius,lo,hi,fc=fill,ec='none',zorder=0))
            cv.add_patch(Arc(origin,2*radius,2*radius,theta1=lo,theta2=hi,color=c,lw=.7,
                linestyle=(0,(2.2,1.8)) if kind=='Predicted' else '-',zorder=1))
            tip=arrow(origin,angle,length,c,kind=='Predicted',.9 if kind=='Predicted' else 1.05,7.)
            tips[(partner,kind)]=tip
            angle_records.append({'query':query,'partner':partner,'kind':kind,'cosine_distance':d,
                'angle_degrees':theta,'pairwise_planes_independently_aligned':True})
    for partner,c,sign in [(partners[0],BLUE,1),(partners[1],ORANGE,-1)]:
        tip=(tips[(partner,'Observed')]+tips[(partner,'Predicted')])/2
        txt(tip[0]+(-2 if sign==1 else 2),tip[1],partner,size=label_size,color=c,
            ha='right' if sign==1 else 'left')
    # The shared query is an aligned axis; its fixed orientation is not fidelity.
    tip=arrow(origin,ref,length,INK,head=7.)
    txt(tip[0],tip[1]+3.1,query,size=label_size,ha='center')
    cv.plot(*origin,marker='o',ms=1.8,color=INK)

# e: larger, recognizable biological process cartoons and one overlay per case.
panel('e','Biological relationship inversions',3,94)
B=pd.read_csv(DATA/'main_B_gene_vertices.csv');BD=pd.read_csv(DATA/'main_B_gene_distances.csv')
for j,(model,x0) in enumerate([('PerturbDiff',0),('GEARS',92)]):
    dat=B[B.model==model].set_index('role');ds=BD[BD.model==model]
    txt(x0+24,88.6,model,size=6.8,weight='bold',ha='center')
    txt(x0+68,88.6,f'CA {dat.CA.min():.3f}–{dat.CA.max():.3f}',size=6.1,color=GRAY,ha='center')
    if j==0:
        txt(x0+24,82.7,'RNA splicing',size=6.1,color=INK,ha='center');bio.splice(x0+24,75.4)
        txt(x0+70,82.7,'Translation initiation',size=6.1,color=INK,ha='center');bio.initiation(x0+70,75.4)
    else:
        txt(x0+24,82.7,'Complex III',size=6.1,color=INK,ha='center');bio.mitochondrion(x0+24,75.4)
        txt(x0+70,82.7,'PBAF',size=6.1,color=INK,ha='center');bio.chromatin(x0+70,75.4)
    gene_pair_label(x0+24,67.5,dat.loc['A','gene'],dat.loc['C','gene'])
    txt(x0+70,67.5,dat.loc['B','gene'],size=6,color=ORANGE,ha='center')
    angle_overlay(x0+46,48.5,dat.loc['A','gene'],[dat.loc['C','gene'],dat.loc['B','gene']],ds)

# f: zero-length bars without marker symbols; printed denominators retained.
panel('f','Strict inversion counts',3,44)
C=pd.read_csv(DATA/'main_C_strict_inversion_rates.csv').set_index('model_display').loc[MODELS]
axf=axis(22,9,29,27,'f');values=C.inversion_rate.to_numpy()*100
axf.barh(np.arange(7),values,height=.57,color=MODEL_GRAY,edgecolor='none')
axf.set(yticks=np.arange(7),yticklabels=MODELS,xlim=(0,7),ylim=(6.65,-.65),xticks=[0,3,6])
axf.spines['left'].set_visible(False);axf.tick_params(axis='y',length=0,pad=2.5)
axf.set_xlabel('Inversion rate (%)',labelpad=3)
txt(62,39,'n / N',size=6,ha='center')
for i,m in enumerate(MODELS):
    txt(62,9+27*(6.65-i)/7.3,f'{int(C.loc[m,"inverted_triplets"])}/{int(C.loc[m,"qualified_triplets"])}',size=6.3,ha='center')

# g: released State in held context, distinct from benchmark State implementation.
panel('g','Official State–Tahoe',79,44)
txt(179,44,'HOP62, 0.5 μM',size=6,color=GRAY,ha='right')
D=pd.read_csv(DATA/'main_D_state_vertices.csv');DD=pd.read_csv(DATA/'main_D_state_distances.csv')
txt(133,38,f'CA {D.systema_centroid_accuracy.min():.3f}–{D.systema_centroid_accuracy.max():.3f}, HDAC / mTOR',
    size=6.1,color=GRAY,ha='center')
style_key(111,33.7,25)
angle_overlay(133,8,'Tucidinostat',['Panobinostat','Temsirolimus'],DD,length=19,label_size=6.0)

pd.DataFrame(angle_records).to_csv(DATA/'display_pairwise_angles.csv',index=False)
(LAYOUT/'panel_mapping.json').write_text(json.dumps({
    'a':'schematic response agreement and relation reversal','b':'schematic global/local definitions',
    'c':'main_E_benchmark.csv','d':'main_d_updated_252_family.csv; 252-contrast simultaneous family',
    'e':'main_B_gene_vertices.csv and main_B_gene_distances.csv',
    'f':'main_C_strict_inversion_rates.csv',
    'g':'main_D_state_vertices.csv and main_D_state_distances.csv'},indent=2),encoding='utf-8')

fig.canvas.draw()
alignment=record_panel_positions(fig,axes=axs,panel_ids=ids,row_groups=rows,column_groups=[],
    json_out=LAYOUT/'Figure_1.alignment.json',overlay_svg=LAYOUT/'Figure_1.alignment.svg',
    tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
base=OUT/'Figure_1'
fig.savefig(base.with_suffix('.pdf'));fig.savefig(base.with_suffix('.svg'))
fig.savefig(base.with_suffix('.png'),dpi=300)
for name,bounds in {'AB':(0,180,183,221),'CD':(0,97,183,180),'E':(0,46,183,97),'FG':(0,0,183,46)}.items():
    x0,y0,x1,y1=bounds;bbox=mpl.transforms.Bbox.from_extents(x0/25.4,y0/25.4,x1/25.4,y1/25.4)
    fig.savefig(LAYOUT/f'Figure_1_{name}.png',dpi=300,bbox_inches=bbox)
print(json.dumps({'figure':str(base),'size_mm':[W,H],'axes':len(axs),'alignment':alignment['status']},indent=2))
