"""Render frozen Figure 1 supporting evidence; no statistical reanalysis."""
from pathlib import Path
import sys
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

from settings import INPUT, OUTPUT as OUT, CHECKS as LAYOUT
SRC=INPUT/'fig12'
from layout import record_panel_positions

MODELS = ['GEARS', 'scGPT', 'Scouter', 'CellFlow', 'PerturbDiff', 'State', 'TxPert']
DISPLAY = {'State-ST-GenePT': 'State', 'TxPert-public': 'TxPert'}
COLORS = dict.fromkeys(MODELS, '#B9BDC2')
INK = '#222222'
MUTED = '#666666'
GRID = '#E7E7E7'
REF = '#555555'

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 7, 'legend.fontsize': 6.3,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .6,
    'axes.edgecolor': '#222222', 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'legend.frameon': False,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})


def read_table(name):
    return pd.read_csv(SRC / name)

def label(ax, letter, title):
    ax.text(0, 1, letter, transform=ax.transAxes + ScaledTranslation(-11/72, 6/72, ax.figure.dpi_scale_trans),
            fontsize=8, fontweight='bold', ha='left', va='bottom', color=INK)
    ax.set_title(title, loc='center', pad=7, fontweight='bold')


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.tick_params(axis='y', length=0, pad=5)
    ax.tick_params(axis='x', length=2.5, width=.5)


def export(fig, stem, axes):
    fig.canvas.draw()
    record_panel_positions(fig, axes=axes, panel_ids=[chr(97+i) for i in range(len(axes))],
        json_out=LAYOUT / f'{stem}.alignment.json', overlay_svg=LAYOUT / f'{stem}.alignment.svg',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    fig.savefig(OUT / f'{stem}.pdf')
    fig.savefig(OUT / f'{stem}.svg')
    fig.savefig(OUT / f'{stem}.png', dpi=300)
    plt.close(fig)


def supplementary1():
    summary = read_table('Supplementary_Figure_1_summary.csv')
    folds = read_table('Supplementary_Figure_1_by_fold.csv')
    for d in (summary, folds):
        d['model_display'] = d['model'].replace(DISPLAY)
    fig, axs = plt.subplots(2, 2, figsize=(183/25.4, 140/25.4))
    fig.subplots_adjust(left=.13, right=.91, bottom=.145, top=.94, wspace=.56, hspace=.44)
    conditions = [('all_three_CA075', 'CA ≥ 0.75'),
                  ('all_three_four_criteria', 'Four criteria')]
    for col, (mode, title) in enumerate(conditions):
        d = summary[summary.quality_mode == mode].set_index('model_display').loc[MODELS]
        ax = axs[0,col]
        y = np.arange(7)
        vals = d.inversion_rate.to_numpy() * 100
        ax.barh(y, vals, color=[COLORS[m] for m in MODELS], height=.62, edgecolor='none', zorder=2)
        # Zero-length bars have no added symbols; printed n/N retains the zero.
        ax.set(yticks=y, yticklabels=MODELS, ylim=(6.7,-.7), xlim=(0,18), xticks=[0,5,10,15], xlabel='Inverted eligible triplets (%)')
        for j, (_, row) in enumerate(d.iterrows()):
            ax.text(1.015,j, f'{int(row.inverted_triplets)}/{int(row.qualified_triplets):,}',
                    transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=6.5, color=INK)
        label(ax, chr(97+col), title)
        ax.text(1.015,1.015,'n/N',transform=ax.transAxes,fontsize=6,ha='left',va='bottom',color=MUTED)
        style(ax)

        ax = axs[1,col]
        markers = ['o','s','^','D','v']
        offsets = np.linspace(-.23,.23,5)
        for j,m in enumerate(MODELS):
            sub = folds[(folds.quality_mode == mode)&(folds.model_display == m)].sort_values('fold')
            assert len(sub)==5
            missing_folds = []
            for _,r in sub.iterrows():
                f = int(r.fold)
                if r.qualified_triplets:
                    ax.scatter(float(r.inversion_rate)*100,j+offsets[f], color='#777777', s=17,
                               marker=markers[f], edgecolor='white', linewidth=.25, zorder=3, clip_on=False)
                else:
                    missing_folds.append(str(f+1))
            if missing_folds:
                ax.text(1.015,j,', '.join(missing_folds),transform=ax.get_yaxis_transform(),
                        fontsize=5.5,va='center',color=MUTED)
        ax.set(yticks=y, yticklabels=MODELS, ylim=(6.7,-.7), xlim=(0,40), xticks=[0,10,20,30,40], xlabel='Inverted eligible triplets (%)')
        label(ax, chr(99+col), 'CA ≥ 0.75, by fold' if col==0 else 'Four criteria, by fold')
        ax.text(1.015,1.005,'Empty\nfolds',transform=ax.transAxes,fontsize=5.5,ha='left',va='bottom',color=MUTED)
        style(ax)
    handles=[Line2D([0],[0],marker=m,linestyle='none',color=MUTED,markersize=4,label=f'Fold {i+1}') for i,m in enumerate(markers)]
    # Keep every saved simultaneous reversal; no new contrasts or inference.
    fig.set_size_inches(183/25.4,210/25.4)
    for ax in axs.ravel():
        box=ax.get_position()
        ax.set_position([box.x0,box.y0*2/3+1/3,box.width,box.height*2/3])
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.355),ncol=5,columnspacing=1.8)
    rev=read_table('S1_simultaneous_reversals.csv')
    assert len(rev)==6 and rev.simultaneous_reversal.all()
    short={'PB_logFC_cosine_transposed_rank':'PB rank (transposed)','PB_logFC_Frobenius_RMS':'PB matrix error','residual_Frobenius_RMS':'Residual matrix error','Systema_CA':'CA'}
    labs=[f'{r.model_a} / {DISPLAY.get(r.model_b,r.model_b)}\n{short[r.better_metric]}' for r in rev.itertuples()]
    extra=[]
    for j,(prefix,title,color) in enumerate([('other','Other metric','#7B9CAF'),('rho','Global ρ','#C99A70'),('top10','Top-10','#C99A70')]):
        ax=fig.add_axes([.29+.225*j,.065,.19,.24]);extra.append(ax)
        mu=rev[f'{prefix}_point'].to_numpy();low=rev[f'{prefix}_sim_low'].to_numpy();high=rev[f'{prefix}_sim_high'].to_numpy()
        ax.errorbar(mu,np.arange(6),xerr=[mu-low,high-mu],fmt='o',ms=3.5,color=color,mfc='white',capsize=2,elinewidth=.8)
        ax.axvline(0,color='#91989D',ls=(0,(3,2)),lw=.6)
        ax.set(ylim=(5.65,-.65),yticks=np.arange(6),yticklabels=labs if j==0 else ['']*6,xlabel='Performance difference')
        ax.tick_params(axis='y',labelsize=5.7)
        ax.set_xlim((-.01,.21) if j==0 else (-.2,.015) if j==1 else (-.075,.005))
        label(ax,'efg'[j],title);style(ax)
    export(fig,'Supplementary_Figure_1',list(axs.ravel())+extra)


def reliability_bars(ax,d,metric,scale,title,letter,limit):
    entities=['experimental_split_half']+MODELS[:5]
    z=d[(d.scope=='two_fold_mean')&(d.min_cells==20)&(d.metric==metric)&(d.scale==scale)].set_index('entity').loc[entities]
    y=np.arange(6)
    cols=[REF]+[COLORS[m] for m in MODELS[:5]]
    vals=z.estimate.to_numpy()
    err=np.vstack([vals-z.ci_low.to_numpy(),z.ci_high.to_numpy()-vals])
    ax.barh(y,vals,color=cols,height=.62,edgecolor='none',zorder=2)
    ax.errorbar(vals,y,xerr=err,fmt='none',ecolor=INK,elinewidth=.75,capsize=1.8,capthick=.65,zorder=4)
    chance=float(z.chance_raw.iloc[0]) if scale=='raw' else 0.
    if chance>0:
        ax.axvline(chance,color='#89949C',ls=(0,(3,2)),lw=.8,zorder=1)
        ax.text(chance+.014,5.72,'Chance',fontsize=5.8,color=MUTED,va='center')
    ax.set(yticks=y,yticklabels=['Split-half']+MODELS[:5],ylim=(6.05,-.7),xlim=(0,limit),xlabel='Score')
    if limit==1.05: ax.set_xticks([0,.25,.5,.75,1.0])
    else: ax.set_xticks([0,.2,.4,.6])
    label(ax,letter,title)
    style(ax)


def supplementary2():
    d=read_table('Supplementary_Figure_2_reliability.csv')
    geo=read_table('Supplementary_Figure_2_rho_partitions.csv')
    fig,axs=plt.subplots(3,2,figsize=(183/25.4,180/25.4))
    fig.subplots_adjust(left=.14,right=.97,bottom=.085,top=.96,wspace=.55,hspace=.52)
    reliability_bars(axs[0,0],d,'CA_expression','raw','CA','a',1.05)
    reliability_bars(axs[0,1],d,'Top10_residual','raw','Top-10','b',.65)
    reliability_bars(axs[1,0],d,'CA_expression','chance_corrected','Chance-adjusted CA','c',1.05)
    reliability_bars(axs[1,1],d,'Top10_residual','chance_corrected','Chance-adjusted Top-10','d',.65)

    ax=axs[2,0]
    for f in [0,1]:
        sub=geo[(geo.min_cells==20)&(geo.fold==f)].sort_values('split')
        assert len(sub)==20
        # Deterministic packing offsets only; positions encode no new statistic.
        offsets=np.array([-.16,-.08,0,.08,.16]*4)
        ax.scatter(sub.rho.to_numpy(),f+offsets,color=REF if f==0 else '#A0A0A0',s=17,alpha=.85,
                   edgecolor='white',linewidth=.3,zorder=3)
    ax.set(yticks=[0,1],yticklabels=['Fold 1\n339 targets','Fold 2\n355 targets'],ylim=(1.55,-.55),
           xlim=(.84,.96),xticks=[.85,.90,.95],xlabel='Spearman ρ')
    label(ax,'e','Split-half global geometry')
    style(ax)

    ax=axs[2,1]
    combos=[('CA_expression',20),('CA_expression',50),('Top10_residual',20),('Top10_residual',50)]
    labs=['CA, ≥20 cells','CA, ≥50 cells','Top-10, ≥20 cells','Top-10, ≥50 cells']
    rows=[]
    for metric,n in combos:
        z=d[(d.entity=='experimental_split_half')&(d.scope=='two_fold_mean')&(d.metric==metric)&(d.scale=='raw')&(d.min_cells==n)]
        assert len(z)==1
        rows.append(z.iloc[0])
    vals=np.array([r.estimate for r in rows]); lo=np.array([r.ci_low for r in rows]); hi=np.array([r.ci_high for r in rows])
    cols=[REF,'#BDBDBD',REF,'#BDBDBD']
    ax.barh(np.arange(4),vals,color=cols,height=.58,edgecolor='none',zorder=2)
    ax.errorbar(vals,np.arange(4),xerr=np.vstack([vals-lo,hi-vals]),fmt='none',ecolor=INK,elinewidth=.75,capsize=1.8,capthick=.65,zorder=3)
    ax.set(yticks=np.arange(4),yticklabels=labs,ylim=(3.6,-.6),xlim=(0,1.05),xticks=[0,.25,.5,.75,1],xlabel='Split-half score')
    label(ax,'f','Cell-count sensitivity')
    style(ax)
    export(fig,'Supplementary_Figure_2',list(axs.ravel()))
