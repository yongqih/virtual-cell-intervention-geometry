"""Build the retained Figure 4/5 and Supplementary Figure 7-10 components.

All quantitative values are read from frozen, publication-grade CSV artifacts.
This script performs no fitting, tuning, resampling, or experiment replay.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "formal_manuscript_revision_20260817"
MAIN = OUT / "main_figures"
SUPP = OUT / "supplementary_figures"
PANELS = OUT / "panel_exports"
SOURCE = OUT / "source_data_used"
QA = OUT / "qa"
DATA_ROOT = ROOT / "data"
MAIN_SOURCE = DATA_ROOT / "source_data" / "main"
SUPP_SOURCE = DATA_ROOT / "source_data" / "supplementary"
AUDIT_SOURCES = DATA_ROOT / "original_audit_sources" / "results"

C = {
    "truth": "#2B2B2B",
    "prediction": "#5F8FB5",
    "failure": "#B65F42",
    "gears": "#4F8F7B",
    "scgpt": "#8B79A8",
    "transformer": "#86ABC5",
    "mlp": "#A2AEC0",
    "baseline": "#A3A3A3",
    "grid": "#DEDEDE",
    "white": "#FFFFFF",
}

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 7.0,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.15,
    "svg.fonttype": "none",
    "svg.hashsalt": "formal_revision_20260817",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def clean(ax, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.75, length=3)
    if grid:
        ax.grid(axis="y", color=C["grid"], lw=0.55, zorder=0)


def letter(ax, value: str, x: float = -0.14, y: float = 1.10) -> None:
    ax.text(x, y, value, transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="left", va="top", clip_on=False)


def save_pair(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", metadata={"Date": None})
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


def panel_figure(name: str, plotter, figsize=(3.25, 2.35),
                 margins=(.24, .97, .23, .83), **kwargs) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    plotter(ax, panel_letter=None, **kwargs)
    left, right, bottom, top = margins
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    save_pair(fig, PANELS / name)


def copy_source(path: Path, name: str | None = None) -> Path:
    target = SOURCE / (name or path.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    return target


def plot_q_curve(ax, data: pd.DataFrame, title: str, color: str, marker: str,
                 panel_letter: str | None = None, thin_groups: pd.DataFrame | None = None,
                 letter_xy: tuple[float, float] = (-.22, 1.18)) -> None:
    if thin_groups is not None:
        group_cols = [c for c in ["pathway", "cell_line", "outer_fold"] if c in thin_groups.columns]
        for _, group in thin_groups.groupby(group_cols, dropna=False):
            group = group.sort_values("q")
            ax.plot(group.q, group.rho, color=color, alpha=.14, lw=.65, zorder=1)
    data = data.sort_values("q")
    if {"ci_low", "ci_high"}.issubset(data.columns) and data.ci_low.notna().any():
        ax.fill_between(data.q, data.ci_low, data.ci_high, color=color, alpha=.13, lw=0)
    ax.plot(data.q, data.rho, color=color, marker=marker, ms=4.2, lw=1.7, zorder=3)
    ax.axhline(0, color=C["grid"], lw=.7, zorder=0)
    ax.set_xticks(sorted(data.q.unique()))
    ax.set_xlabel("Residual coordinates, q")
    ax.set_ylabel("Grouped geometry, Spearman ρ")
    ax.set_title(title, pad=5)
    ax.set_ylim(min(-.16, float(data.rho.min())-.07), 1.02)
    clean(ax)
    if panel_letter:
        letter(ax, panel_letter, x=letter_xy[0], y=letter_xy[1])


def plot_orientation_concept(ax, panel_letter="a") -> None:
    ax.set_axis_off()
    ax.set_xlim(-.2, 1.1); ax.set_ylim(-.15, 1.05)
    base = np.array([.12, .25])
    p1 = np.array([.55, .14]); p2 = np.array([.14, .48])
    ax.scatter(*base, s=30, color=C["baseline"], zorder=4)
    ax.annotate("", base+p1, base, arrowprops=dict(arrowstyle="->", color=C["gears"], lw=1.7))
    ax.annotate("", base+p2, base, arrowprops=dict(arrowstyle="->", color=C["prediction"], lw=1.7))
    ax.text(*(base+p1+[.02,-.02]), "P1", color=C["gears"], fontsize=8)
    ax.text(*(base+p2+[-.02,.03]), "P2", color=C["prediction"], fontsize=8)
    target = base + .70*p1 - .40*p2
    ax.annotate("", target, base, arrowprops=dict(arrowstyle="->", color=C["truth"], lw=2.1))
    ax.scatter(*target, marker="D", s=25, color=C["truth"], zorder=5)
    ax.text(.46, .13, "held target\n0.7 P1 − 0.4 P2", fontsize=6.7, ha="center")
    ax.text(.43, .96, r"$r_g=\hat r_g^{base}+\sum_k u_{gk}P_k$", ha="center", fontsize=8)
    ax.text(.43, .84, "axes learned from seen interventions", ha="center", color=C["gears"], fontsize=6.5)
    ax.text(.43, -.05, "response axes are learnable;\nheld-target coordinates are missing",
            ha="center", va="top", fontsize=7, fontweight="bold")
    if panel_letter:
        ax.text(-.13, 1.04, panel_letter, fontsize=9, fontweight="bold", va="top")


def plot_random_subspaces(ax, data: pd.DataFrame, panel_letter="d",
                          letter_xy: tuple[float, float] = (-.08, 1.10)) -> None:
    offsets = {"K562": -.08, "Jiang": .08}
    markers = {"K562": "o", "Jiang": "s"}
    colors = {"K562": C["gears"], "Jiang": C["prediction"]}
    for dataset, group in data.groupby("dataset"):
        x = group.q.to_numpy(float) + offsets[dataset]
        ax.vlines(x, group.null_median, group.null_95th, color=C["baseline"], lw=3.8, alpha=.55, zorder=1)
        ax.scatter(x, group.true_rho, color=colors[dataset], marker=markers[dataset], s=26,
                   edgecolor="white", linewidth=.4, label=dataset, zorder=3)
    ax.set_xticks(sorted(data.q.unique()))
    ax.set_xlabel("Residual coordinates, q")
    ax.set_ylabel("Grouped geometry, Spearman ρ")
    ax.set_title("Learned residual directions exceed matched random subspaces", pad=5)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.text(.98, .04, "grey: null median → 95th percentile", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.8, color=C["baseline"])
    clean(ax)
    if panel_letter: letter(ax, panel_letter, x=letter_xy[0], y=letter_xy[1])


def plot_orientation_ladder(ax, data: pd.DataFrame, panel_letter="e", dataset="K562") -> None:
    labels = {
        "baseline": "Baseline", "q1_sign": "P1 sign only", "q2_sign": "P1+P2 sign only",
        "q2_fixed_radius_direction": "Exact q2 direction\n(fixed radius)", "q2_exact": "Exact q2 coordinates",
    }
    order = ["baseline", "q1_sign", "q2_sign", "q2_fixed_radius_direction", "q2_exact"]
    q = data[(data.dataset == dataset) & (data.pathway == "ALL")].set_index("condition").loc[order].reset_index()
    colors = [C["baseline"], "#85B6A7", "#6CA18F", C["prediction"], C["gears"]]
    yy = np.arange(len(q))[::-1]
    ax.hlines(yy, 0, q.rho, color=colors, lw=2.2)
    ax.scatter(q.rho, yy, color=colors, s=30, zorder=3)
    ax.axvline(0, color=C["grid"], lw=.7)
    ax.set_yticks(yy, [labels[v] for v in q.condition])
    ax.set_xlabel("Grouped geometry, Spearman ρ")
    ax.set_title("Polarity contributes, but does not specify the full code", pad=5)
    if dataset == "K562":
        ax.annotate("sign-only q2 retains 39.86%\nof the exact-q2 gain",
                    xy=(float(q[q.condition=="q2_sign"].rho.iloc[0]), yy[2]),
                    xytext=(.34, yy[1]+.15), fontsize=6.2, color=C["gears"],
                    arrowprops=dict(arrowstyle="-", color=C["gears"], lw=.7))
    ax.set_xlim(min(-.16, q.rho.min()-.05), .72)
    clean(ax)
    if panel_letter: letter(ax, panel_letter, x=-.12)


def plot_synthetic(ax, data: pd.DataFrame, panel_letter="f") -> None:
    order = ["NO_GRAPH", "CORRECT_DIRECTED_SIGNED", "CORRECT_DIRECTED_UNSIGNED",
             "REVERSED_DIRECTED_SIGNED", "DEGREE_PRESERVING_SHUFFLE", "SIGN_SHUFFLED"]
    labels = ["No graph", "Correct directed + signed", "Unsigned", "Reversed", "Degree shuffle", "Sign shuffle"]
    q = data.set_index("condition").loc[order]
    colors = [C["baseline"], C["gears"], C["prediction"], C["failure"], C["baseline"], C["baseline"]]
    yy = np.arange(len(q))[::-1]
    for x, y, error, color in zip(q.response_distance_correlation_mean, yy,
                                  q.response_distance_correlation_std, colors):
        ax.errorbar(x, y, xerr=error, fmt="none", ecolor=color, elinewidth=1.2, capsize=2)
        ax.scatter(x, y, s=27, color=color, zorder=3)
    ax.axvline(0, color=C["grid"], lw=.7)
    ax.set_yticks(yy, labels)
    ax.set_xlabel("Response-distance correlation")
    ax.set_title("Correct signed structure identifies geometry\nin a matched world", pad=4)
    ax.text(.98, .03, "synthetic identifiability control", transform=ax.transAxes,
            ha="right", fontsize=5.8, color=C["baseline"])
    ax.set_xlim(-.16, .79)
    clean(ax)
    if panel_letter: letter(ax, panel_letter, x=-.16)


def plot_renge_design(ax, panel_letter="a") -> None:
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    pts = [(0.12,.77,"R2","Day 2"),(0.72,.77,"R3","Day 3"),(0.72,.40,"R4","Day 4"),(0.25,.15,"R5","Day 5")]
    for i,(x,y,r,d) in enumerate(pts):
        ax.scatter(x,y,s=85,color=["#9FC0DF",C["prediction"],"#2E75B6","#174A73"][i],edgecolor="#245A93",lw=.7)
        ax.text(x,y-.12,r,ha="center",color=["#9FC0DF",C["prediction"],"#2E75B6","#174A73"][i],fontsize=7)
        ax.text(x,y+.12,d,ha="center",fontsize=6.3)
    ax.plot([.16,.68],[.77,.77],color=C["truth"],lw=1); ax.text(.42,.82,"W23",ha="center",fontsize=6)
    ax.plot([.72,.72],[.71,.46],color=C["truth"],lw=1); ax.text(.78,.59,"W34",rotation=-90,va="center",fontsize=6)
    ax.plot([.67,.29],[.38,.18],color=C["truth"],lw=1); ax.text(.49,.31,"W45",rotation=-26,fontsize=6)
    ax.text(.5,.96,"RENGE temporal design",ha="center",fontsize=8)
    ax.text(.98,.025,"endpoint axes\n(training only)",ha="right",va="bottom",fontsize=5.7,color=C["gears"])
    if panel_letter: ax.text(-.06,1.03,panel_letter,fontsize=9,fontweight="bold",va="top")


def plot_dumbbell(ax, left, right, left_label, right_label, xlabel, title, panel_letter=None) -> None:
    ax.hlines(0, min(left,right), max(left,right), color=C["grid"], lw=4, zorder=1)
    ax.scatter(left,0,color=C["prediction"],s=32,zorder=3)
    ax.scatter(right,0,color=C["truth"],s=32,zorder=3)
    ax.text(left,.12,f"{left_label}\n{left:.3f}",ha="center",va="bottom",fontsize=6.3,color=C["prediction"])
    ax.text(right,-.12,f"{right_label}\n{right:.3f}",ha="center",va="top",fontsize=6.3,color=C["truth"])
    ax.set_yticks([]); ax.set_ylim(-.35,.35); ax.set_xlim(-.005,max(left,right)*1.12)
    ax.set_xlabel(xlabel); ax.set_title(title,pad=5); clean(ax)
    if panel_letter: letter(ax,panel_letter)


def plot_early_assay(ax, panel_letter="c") -> None:
    ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1)
    steps=[("learn P1/P2 axes",.25,.65),("exclude held target",.75,.65),
           ("measure Day 2/3/4",.25,.38),("project on frozen axes",.75,.38)]
    ax.annotate("",(.66,.65),(.34,.65),arrowprops=dict(arrowstyle="->",lw=.8,color=C["baseline"]))
    ax.annotate("",(.25,.46),(.25,.56),arrowprops=dict(arrowstyle="->",lw=.8,color=C["baseline"]))
    ax.annotate("",(.66,.38),(.34,.38),arrowprops=dict(arrowstyle="->",lw=.8,color=C["baseline"]))
    for text,x,y in steps:
        ax.text(x,y,text,ha="center",va="center",fontsize=5.7,
                bbox=dict(boxstyle="round,pad=.24",fc="white",ec=C["grid"],lw=.7))
    ax.text(.5,.88,"Early target-specific orientation assay",ha="center",fontsize=8)
    ax.text(.5,.20,"not zero-shot — empirical target anchor diagnostic",ha="center",fontsize=6.5,
            color=C["failure"],fontweight="bold")
    if panel_letter: ax.text(-.04,1.03,panel_letter,fontsize=9,fontweight="bold",va="top")


def plot_early_orientation(ax, data: pd.DataFrame, panel_letter="d") -> None:
    ax.plot(data.early_day,data.p1_accuracy,color=C["gears"],marker="o",ms=4,label="P1 sign")
    ax.plot(data.early_day,data.p2_accuracy,color=C["prediction"],marker="s",ms=4,label="P2 sign")
    ax.plot(data.early_day,data.exact_high_reliability_accuracy,color=C["truth"],marker="D",ms=3.8,
            ls="--",label="Exact 2-bit (reliable subset)")
    ax.fill_between(data.early_day,data.p1_accuracy_ci_low,data.p1_accuracy_ci_high,color=C["gears"],alpha=.10,lw=0)
    ax.axhline(.5,color=C["grid"],ls="--",lw=.7)
    ax.set_xticks([2,3,4],["Day 2","Day 3","Day 4"])
    ax.set_ylim(.45,1.04); ax.set_ylabel("Orientation accuracy")
    ax.set_title("Target-specific orientation is visible early after perturbation",pad=5)
    ax.legend(frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(.5,-.39))
    clean(ax)
    if panel_letter: letter(ax,panel_letter,x=-.10)


def plot_synthesis(ax, panel_letter="e") -> None:
    ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1)
    upper=["Genuinely unseen target","orientation unknown","wrong entry","coherent wrong trajectory","low geometry"]
    lower=["Early same-target response","orientation exposed","correct entry","propagation","higher geometry"]
    xs=np.linspace(.07,.93,5)
    for labels,y,color in [(upper,.69,C["failure"]),(lower,.30,C["gears"])]:
        for i,x in enumerate(xs[:-1]):
            ax.annotate("",(xs[i+1]-.055,y),(x+.055,y),arrowprops=dict(arrowstyle="->",color=color,lw=1.1))
        for x,text in zip(xs,labels):
            ax.text(x,y,text,ha="center",va="center",fontsize=5.7,color=color,
                    bbox=dict(fc="white",ec="none",pad=.8))
    ax.text(.5,.95,"Intervention orientation determines trajectory entry",ha="center",fontsize=8)
    ax.text(.5,.02,"Intervention identity is an entry-coordinate problem before it is a propagation problem.",
            ha="center",fontsize=7,fontweight="bold")
    if panel_letter: ax.text(.0,1.03,panel_letter,fontsize=9,fontweight="bold",va="top")


def build_figure4() -> None:
    base=MAIN_SOURCE/"Figure4"
    k=pd.read_csv(copy_source(base/"fig4b_k562_q_curve.csv")); j=pd.read_csv(copy_source(base/"fig4c_jiang_q_curve.csv"))
    null=pd.read_csv(copy_source(base/"fig4d_random_subspace_null.csv")); ladder=pd.read_csv(copy_source(base/"fig4e_sign_only_rescue.csv"))
    synth=pd.read_csv(copy_source(AUDIT_SOURCES/"clean_synthetic_directional_control"/"overall_graph_condition_summary.csv"))
    fig=plt.figure(figsize=(7.1,5.7))
    axa=fig.add_axes([.04,.43,.23,.49]); axb=fig.add_axes([.34,.70,.34,.23]); axc=fig.add_axes([.75,.70,.22,.23])
    axd=fig.add_axes([.34,.40,.63,.20]); axe=fig.add_axes([.14,.08,.42,.22]); axf=fig.add_axes([.71,.08,.26,.22])
    plot_orientation_concept(axa); plot_q_curve(axb,k,"K562: few residual coordinates restore geometry",C["gears"],"o","b")
    plot_q_curve(axc,j[(j.pathway=="ALL")],"Jiang: independent pathway confirmation",C["prediction"],"s","c"); axc.set_ylabel("")
    plot_random_subspaces(axd,null); plot_orientation_ladder(axe,ladder); plot_synthetic(axf,synth)
    save_pair(fig,MAIN/"Figure4_orientation_code")
    fig,ax=plt.subplots(figsize=(2.7,3.2)); plot_orientation_concept(ax,None); save_pair(fig,PANELS/"Fig4a")
    panel_figure("Fig4b",plot_q_curve,data=k,title="K562: few residual coordinates restore geometry",color=C["gears"],marker="o")
    panel_figure("Fig4c",plot_q_curve,data=j[j.pathway=="ALL"],title="Jiang: independent pathway confirmation",color=C["prediction"],marker="s")
    panel_figure("Fig4d",plot_random_subspaces,data=null,figsize=(4.2,2.4))
    panel_figure("Fig4e",plot_orientation_ladder,data=ladder,figsize=(4.0,2.6))
    panel_figure("Fig4f",plot_synthetic,data=synth,figsize=(3.8,2.7),margins=(.34,.97,.23,.83))


def build_figure5() -> None:
    base=MAIN_SOURCE/"Figure5"
    entry=pd.read_csv(copy_source(base/"fig5b_teacher_true_entry.csv")); early=pd.read_csv(copy_source(base/"fig5d_early_target_sign.csv"))
    vals=entry.set_index("comparison").geometry_rho
    fig=plt.figure(figsize=(7.1,6.3))
    axa=fig.add_axes([.04,.68,.22,.25]); axb1=fig.add_axes([.36,.72,.25,.19]); axb2=fig.add_axes([.70,.72,.27,.19])
    axc=fig.add_axes([.04,.38,.27,.20]); axd=fig.add_axes([.40,.39,.57,.19]); axe=fig.add_axes([.05,.07,.92,.20])
    plot_renge_design(axa); plot_dumbbell(axb1,vals.free_rollout,vals.teacher_forced,"Free rollout","Teacher forced","W45 grouped geometry","Teacher forcing vs free rollout","b")
    plot_dumbbell(axb2,vals.predicted_entry,vals.true_entry,"Predicted entry","True entry","W45 grouped geometry","Predicted vs true entry",None)
    axb1.text(1.18,1.18,"Correct entry enables downstream propagation",transform=axb1.transAxes,ha="center",fontsize=8)
    plot_early_assay(axc); plot_early_orientation(axd,early); plot_synthesis(axe)
    save_pair(fig,MAIN/"Figure5_temporal_orientation")
    fig,ax=plt.subplots(figsize=(2.8,2.6)); plot_renge_design(ax,None); save_pair(fig,PANELS/"Fig5a")
    fig,axes=plt.subplots(1,2,figsize=(5.2,2.3)); plot_dumbbell(axes[0],vals.free_rollout,vals.teacher_forced,"Free rollout","Teacher forced","W45 grouped geometry","Teacher forcing vs rollout"); plot_dumbbell(axes[1],vals.predicted_entry,vals.true_entry,"Predicted entry","True entry","W45 grouped geometry","Entry state"); fig.subplots_adjust(left=.08,right=.98,bottom=.23,top=.82,wspace=.35); save_pair(fig,PANELS/"Fig5b")
    fig,ax=plt.subplots(figsize=(3.2,2.3)); plot_early_assay(ax,None); save_pair(fig,PANELS/"Fig5c")
    panel_figure("Fig5d",plot_early_orientation,data=early,figsize=(4.8,2.7))
    fig,ax=plt.subplots(figsize=(6.8,1.9)); plot_synthesis(ax,None); save_pair(fig,PANELS/"Fig5e")


def build_supp7() -> None:
    base=SUPP_SOURCE/"SupplementaryFigure7"
    kfold=pd.read_csv(copy_source(base/"SuppFig7_k562_fold_metrics.csv"))
    ksum=pd.read_csv(copy_source(base/"fig4b_k562_q_curve.csv"))
    jfold=pd.read_csv(copy_source(base/"SuppFig7_jiang_fold_metrics.csv"))
    jsummary=pd.read_csv(copy_source(base/"fig4c_jiang_q_curve.csv"))
    null=pd.read_csv(copy_source(base/"fig4d_random_subspace_null.csv"))
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.5))
    supp_letter=(-.18,1.12)
    plot_q_curve(axes[0,0],ksum,"K562 fold-level q-curves",C["gears"],"o","a",kfold,supp_letter)
    plot_random_subspaces(axes[0,1],null[null.dataset=="K562"],"b",supp_letter)
    plot_q_curve(axes[1,0],jsummary[jsummary.pathway=="ALL"],"Jiang pathway/cell-block confirmation",C["prediction"],"s","c",jfold,supp_letter)
    plot_random_subspaces(axes[1,1],null[null.dataset=="Jiang"],"d",supp_letter)
    fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.94,wspace=.32,hspace=.42); save_pair(fig,SUPP/"Supplementary_Figure7_orientation_code")
    panel_figure("SuppFig7a",plot_q_curve,data=ksum,title="K562 fold-level q-curves",color=C["gears"],marker="o",thin_groups=kfold)
    panel_figure("SuppFig7b",plot_random_subspaces,data=null[null.dataset=="K562"])
    panel_figure("SuppFig7c",plot_q_curve,data=jsummary[jsummary.pathway=="ALL"],title="Jiang pathway/cell-block confirmation",color=C["prediction"],marker="s",thin_groups=jfold)
    panel_figure("SuppFig7d",plot_random_subspaces,data=null[null.dataset=="Jiang"])


def plot_probability(ax, k, j, panel_letter="c"):
    for q,color,marker,label in [(k,C["gears"],"o","K562"),(j,C["prediction"],"s","Jiang")]:
        ax.plot(q.p,q.mean_rho,color=color,marker=marker,ms=3.3,label=label)
        ax.fill_between(q.p,q.mc_ci_low,q.mc_ci_high,color=color,alpha=.12,lw=0)
    ax.set_xlabel("Per-sign information accuracy, p"); ax.set_ylabel("Grouped geometry, Spearman ρ")
    ax.set_title("Probabilistic sign-information curve"); ax.legend(frameon=False); clean(ax)
    if panel_letter: letter(ax,panel_letter,x=-.09)


def build_supp8() -> None:
    base=SUPP_SOURCE/"SupplementaryFigure8"
    ladder=pd.read_csv(copy_source(base/"fig4e_sign_only_rescue.csv"))
    kp=pd.read_csv(copy_source(base/"SuppFig8_k562_probability_curve.csv"))
    jp=pd.read_csv(copy_source(base/"SuppFig8_jiang_probability_curve.csv"))
    fig=plt.figure(figsize=(7.1,5.3)); a=fig.add_axes([.15,.58,.32,.32]); b=fig.add_axes([.61,.58,.36,.32]); c=fig.add_axes([.10,.12,.87,.27])
    plot_orientation_ladder(a,ladder,"a","K562"); a.set_title("K562 orientation ladder")
    plot_orientation_ladder(b,ladder,"b","Jiang"); b.set_title("Jiang orientation ladder")
    plot_probability(c,kp,jp,"c"); save_pair(fig,SUPP/"Supplementary_Figure8_orientation_decomposition")
    panel_figure("SuppFig8a",plot_orientation_ladder,data=ladder,dataset="K562",figsize=(3.8,2.6),margins=(.31,.97,.23,.83))
    panel_figure("SuppFig8b",plot_orientation_ladder,data=ladder,dataset="Jiang",figsize=(3.8,2.6),margins=(.31,.97,.23,.83))
    panel_figure("SuppFig8c",plot_probability,k=kp,j=jp,figsize=(5.0,2.6))


def plot_temporal_controls(ax,data,panel_letter="a"):
    q=data.copy(); q=q.sort_values("mean_response_distance_rho")
    colors=[C["gears"] if x=="CorrectLag" else C["baseline"] for x in q.model]
    yy=np.arange(len(q))
    for x,y,error,c in zip(q.mean_response_distance_rho,yy,q.std_response_distance_rho,colors):
        ax.errorbar(x,y,xerr=error,fmt="none",ecolor=c,capsize=2)
        ax.scatter(x,y,color=c,s=25)
    ax.axvline(0,color=C["grid"],lw=.7); ax.set_yticks(yy,q.label); ax.set_xlabel("W23 grouped geometry"); ax.set_title("First-wave temporal controls"); clean(ax)
    if panel_letter: letter(ax,panel_letter,x=-.20)


def plot_static_breadth(ax,data,panel_letter="b"):
    ax.plot(data.n_sources,data.geometry,color=C["baseline"],marker="o",ms=3); ax.set_xlabel("Training sources"); ax.set_ylabel("Grouped geometry"); ax.set_title("Static breadth scaling"); clean(ax)
    if panel_letter: letter(ax,panel_letter)


def plot_temporal_order(ax,data,panel_letter="c"):
    labels=list(dict.fromkeys(data.label)); yy=np.arange(len(labels)); width=.16
    for i,(cond,color,marker) in enumerate([("CorrectTemporal",C["gears"],"o"),("TemporalShuffle",C["failure"],"s")]):
        q=data[data.temporal_order==cond].set_index("label").reindex(labels); ax.scatter(q.response_distance_rho,yy+(i-.5)*width,color=color,marker=marker,s=24,label=cond.replace("Temporal"," "))
    ax.axvline(0,color=C["grid"],lw=.7); ax.set_yticks(yy,labels); ax.set_xlabel("Grouped geometry"); ax.set_title("Temporal-order controls"); ax.legend(frameon=False); clean(ax)
    if panel_letter: letter(ax,panel_letter,x=-.18)


def build_supp9() -> None:
    base=SUPP_SOURCE/"SupplementaryFigure9"
    a=pd.read_csv(copy_source(base/"SuppFig9_temporal_controls.csv")); b=pd.read_csv(copy_source(base/"SuppFig9_static_breadth.csv")); c=pd.read_csv(copy_source(base/"SuppFig9_temporal_order.csv"))
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.8),gridspec_kw={"width_ratios":[1.3,1,1.2]}); plot_temporal_controls(axes[0],a); plot_static_breadth(axes[1],b); plot_temporal_order(axes[2],c)
    fig.subplots_adjust(left=.15,right=.98,bottom=.22,top=.83,wspace=.55); save_pair(fig,SUPP/"Supplementary_Figure9_temporal_controls")
    panel_figure("SuppFig9a",plot_temporal_controls,data=a,margins=(.36,.97,.23,.83)); panel_figure("SuppFig9b",plot_static_breadth,data=b); panel_figure("SuppFig9c",plot_temporal_order,data=c)


def plot_forest(ax,data,title,panel_letter=None,x="estimate",labels="condition",ci=True):
    q=data.copy(); yy=np.arange(len(q))[::-1]; values=q[x].to_numpy(float)
    if ci and {"ci_low","ci_high"}.issubset(q.columns):
        err=np.vstack([values-q.ci_low.to_numpy(float),q.ci_high.to_numpy(float)-values]); ax.errorbar(values,yy,xerr=err,fmt="none",ecolor=C["gears"],capsize=2)
    ax.scatter(values,yy,color=C["gears"],s=26); ax.axvline(0,color=C["grid"],lw=.7); ax.set_yticks(yy,q[labels]); ax.set_xlabel("Geometry / contrast"); ax.set_title(title); clean(ax)
    if panel_letter: letter(ax,panel_letter,x=-.22)


def plot_reliability(ax,data,panel_letter="e"):
    q=data.sort_values("mode"); x=np.arange(len(q)); ax.scatter(x,q.half1_vs_half2,color=[C["gears"],C["prediction"]],s=35)
    ax.vlines(x,0,q.half1_vs_half2,color=[C["gears"],C["prediction"]],lw=2); ax.set_xticks(x,["P1","P2"]); ax.set_ylim(0,1.05); ax.set_ylabel("Split-half sign reliability"); ax.set_title("Endpoint-orientation reliability"); clean(ax)
    if panel_letter: letter(ax,panel_letter)


def plot_target_heatmap(ax,data,panel_letter="f"):
    pivot=data.pivot(index="source",columns="early_day",values="exact_correct").sort_index(); values=pivot.to_numpy()
    ax.pcolormesh(np.arange(values.shape[1]+1)-0.5,np.arange(values.shape[0]+1)-0.5,values,vmin=0,vmax=1,cmap=mpl.colors.ListedColormap([C["failure"],C["gears"]]),shading="flat")
    ax.set_ylim(values.shape[0]-0.5,-0.5)
    ax.set_xticks(range(3),["Day 2","Day 3","Day 4"]); ax.set_yticks(range(len(pivot)),pivot.index,fontsize=4.8); ax.set_title("Per-target exact orientation agreement"); ax.set_xlabel("Early empirical response");
    if panel_letter: letter(ax,panel_letter,x=-.18)


def plot_rollout_autopsy(ax, data, panel_letter="d"):
    q=data.copy()
    direct=q[q.label.isin(["Direct", "HistoryConditioned"])]
    other=q[~q.label.isin(["Direct", "HistoryConditioned"])]
    if not direct.empty:
        row=direct.iloc[0]
        ax.scatter(row.between_variance_ratio,row.response_distance_rho,color=C["prediction"],s=28)
        ax.annotate("Direct = history",(row.between_variance_ratio,row.response_distance_rho),
                    xytext=(4,4),textcoords="offset points",fontsize=5.0,ha="left",va="bottom")
    offsets={"Impulse":(4,-2,"left","center"),"Additive":(-4,5,"right","bottom"),
             "Conditional":(4,5,"left","bottom")}
    for row in other.itertuples(index=False):
        ax.scatter(row.between_variance_ratio,row.response_distance_rho,color=C["prediction"],s=28)
        dx,dy,ha,va=offsets.get(row.label,(4,4,"left","bottom"))
        ax.annotate(row.label,(row.between_variance_ratio,row.response_distance_rho),xytext=(dx,dy),
                    textcoords="offset points",fontsize=5.2,ha=ha,va=va)
    ax.set_xlabel("Variance ratio"); ax.set_ylabel("Grouped geometry")
    ax.set_title("Rollout compression autopsy"); ax.margins(x=.10,y=.14); clean(ax)
    if panel_letter: letter(ax,panel_letter)


def build_supp10() -> None:
    base=SUPP_SOURCE/"SupplementaryFigure10"
    a=pd.read_csv(copy_source(base/"SuppFig10_entry.csv")); b=pd.read_csv(copy_source(base/"SuppFig10_oracle_rescue.csv")); c=pd.read_csv(copy_source(base/"SuppFig10_markov.csv")); d=pd.read_csv(copy_source(base/"SuppFig10_rollout_compression.csv")); e=pd.read_csv(copy_source(base/"SuppFig10_sign_reliability.csv")); f=pd.read_csv(copy_source(base/"SuppFig10_early_source_predictions.csv"))
    fig,axes=plt.subplots(2,3,figsize=(7.1,5.2)); vals=a.set_index("condition").estimate
    plot_dumbbell(axes[0,0],vals["Predicted W23 entry"],vals["True W23 entry"],"Predicted","True","W45 geometry","Trajectory entry","a")
    plot_forest(axes[0,1],b,"Oracle rescue decomposition","b")
    plot_forest(axes[0,2],c.rename(columns={"response_distance_rho":"estimate","label":"condition"}),"Deployable Markov formulations","c",ci=False)
    plot_rollout_autopsy(axes[1,0],d,"d")
    plot_reliability(axes[1,1],e,"e"); plot_target_heatmap(axes[1,2],f,"f")
    fig.subplots_adjust(left=.11,right=.98,bottom=.10,top=.94,wspace=.50,hspace=.48); save_pair(fig,SUPP/"Supplementary_Figure10_trajectory_reliability")
    figp,axp=plt.subplots(figsize=(3.3,2.3)); plot_dumbbell(axp,vals["Predicted W23 entry"],vals["True W23 entry"],"Predicted","True","W45 geometry","Trajectory entry"); save_pair(figp,PANELS/"SuppFig10a")
    panel_figure("SuppFig10b",plot_forest,data=b,title="Oracle rescue decomposition")
    panel_figure("SuppFig10c",plot_forest,data=c.rename(columns={"response_distance_rho":"estimate","label":"condition"}),title="Deployable Markov formulations",ci=False)
    panel_figure("SuppFig10d",plot_rollout_autopsy,data=d)
    panel_figure("SuppFig10e",plot_reliability,data=e); panel_figure("SuppFig10f",plot_target_heatmap,data=f,figsize=(3.6,3.0))


def sha256(path: Path) -> str:
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def write_manifest() -> None:
    rows=[]
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {"output_manifest.csv"}:
            rows.append({"path":path.relative_to(OUT).as_posix(),"bytes":path.stat().st_size,"sha256":sha256(path)})
    with (OUT/"output_manifest.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["path","bytes","sha256"]); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    for path in [MAIN,SUPP,PANELS,SOURCE,QA]: path.mkdir(parents=True,exist_ok=True)
    build_figure4(); build_figure5(); build_supp7(); build_supp8(); build_supp9(); build_supp10()
    write_manifest()
    print(json.dumps({"status":"PASS","models_refit":0,"main_figures":2,"supplementary_figures":4,"panel_exports":len(list(PANELS.glob('*.svg')))},indent=2))


if __name__ == "__main__":
    main()
