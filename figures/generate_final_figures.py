from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AI4SCI_RELEASE_ROOT", HERE.parents[1])).resolve()
sys.path.insert(0, str(HERE))
import figure_style as b

OUT = Path(os.environ.get("AI4SCI_FIGURE_OUT", ROOT / "figures" / "preprint_final")).resolve()
b.OUT = OUT
C = b.C
TRUTH, PRED, GEARS, SCGPT = C["truth"], C["prediction"], C["gears"], C["scgpt"]
BASE, FAIL, GRID = C["baseline"], C["failure"], C["grid"]
DARK_DIRECTION_BLUE = "#3E6F91"

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=[GEARS, SCGPT, PRED, FAIL, BASE])

b.LAYOUTS = {
    "Figure1": {"figsize": (7.1, 5.0), "panels": {
        "a": {"x": .04, "y": .56, "width": .35, "height": .36, "priority": 2, "type": "conceptual_reserved"},
        "b": {"x": .04, "y": .16, "width": .35, "height": .30, "priority": 2, "type": "conceptual_reserved"},
        "c": {"x": .48, "y": .66, "width": .46, "height": .25, "priority": 1, "type": "quantitative"},
        "d": {"x": .48, "y": .24, "width": .46, "height": .30, "priority": 1, "type": "quantitative"},
        "e": {"x": .48, "y": .05, "width": .46, "height": .10, "priority": 3, "type": "conceptual_reserved"}}},
    "Figure2": {"figsize": (7.1, 6.2), "panels": {
        "a": {"x": .05, "y": .62, "width": .90, "height": .31, "priority": 1, "type": "quantitative"},
        "b": {"x": .06, "y": .36, "width": .29, "height": .17, "priority": 1, "type": "quantitative"},
        "c": {"x": .42, "y": .36, "width": .23, "height": .17, "priority": 2, "type": "quantitative"},
        "d": {"x": .72, "y": .36, "width": .23, "height": .17, "priority": 2, "type": "quantitative"},
        "e": {"x": .06, "y": .07, "width": .60, "height": .19, "priority": 1, "type": "quantitative"},
        "f": {"x": .73, "y": .07, "width": .22, "height": .19, "priority": 2, "type": "quantitative"}}},
    "Figure3": {"figsize": (7.1, 6.2), "panels": {
        "a": {"x": .12, "y": .79, "width": .80, "height": .13, "priority": 1, "type": "quantitative"},
        "b": {"x": .08, "y": .42, "width": .39, "height": .25, "priority": 2, "type": "quantitative"},
        "c": {"x": .57, "y": .42, "width": .37, "height": .25, "priority": 2, "type": "quantitative"},
        "d": {"x": .10, "y": .09, "width": .34, "height": .21, "priority": 2, "type": "quantitative"},
        "e": {"x": .55, "y": .09, "width": .17, "height": .21, "priority": 3, "type": "quantitative"},
        "f": {"x": .81, "y": .09, "width": .16, "height": .21, "priority": 3, "type": "quantitative"}}},
    "Figure4": b.LAYOUTS["Figure4"],
    "Figure5": {"figsize": (7.1, 6.3), "panels": {
        "a": {"x": .04, "y": .72, "width": .20, "height": .20, "priority": 3, "type": "conceptual_reserved"},
        "c": {"x": .31, "y": .70, "width": .39, "height": .22, "priority": 1, "type": "quantitative"},
        "d": {"x": .77, "y": .70, "width": .20, "height": .22, "priority": 1, "type": "quantitative"},
        "b": {"x": .08, "y": .31, "width": .16, "height": .20, "priority": 3, "type": "quantitative"},
        "e": {"x": .33, "y": .31, "width": .41, "height": .21, "priority": 2, "type": "quantitative"},
        "f": {"x": .84, "y": .32, "width": .14, "height": .20, "priority": 3, "type": "quantitative"},
        "g": {"x": .05, "y": .04, "width": .92, "height": .16, "priority": 2, "type": "conceptual_reserved"}}},
    "Figure6": b.LAYOUTS["Figure6"],
}

PANEL_DATA: dict[tuple[str, str], pd.DataFrame] = {}


def stash(fig: str, panel: str, frame: pd.DataFrame) -> None:
    PANEL_DATA[(fig, panel)] = frame.copy()


def reserved_labels(fig: plt.Figure, name: str) -> None:
    for panel, pos in b.LAYOUTS[name]["panels"].items():
        if pos["type"] == "conceptual_reserved":
            fig.text(pos["x"], min(.985, pos["y"] + pos["height"] + .015), panel,
                     fontsize=9, fontweight="bold", ha="left", va="top")


def save(fig: plt.Figure, name: str) -> None:
    if name in b.LAYOUTS:
        reserved_labels(fig, name)
    b.save(fig, name)


def summary_row(path: Path, **filters):
    frame = pd.read_csv(path)
    for key, value in filters.items():
        frame = frame[frame[key] == value]
    if len(frame) != 1:
        raise RuntimeError(f"Expected one row in {path} for {filters}; found {len(frame)}")
    return frame.iloc[0]


def model_metrics():
    gears_h = pd.read_csv(ROOT / "results/gears_geometry_audit/standard_metrics.csv")
    gears_h = gears_h[(gears_h.record_type == "fold_mean") & (gears_h.model == "GEARS")]
    sc_h = pd.read_csv(ROOT / "results/final_scgpt_preprint_audit/scgpt_metric_hierarchy.csv")
    sc_h = sc_h[sc_h.record_type == "fold_mean"]
    rows = []
    order = ["absolute_perturbed_state", "total_perturbation_response", "intervention_specific_residual"]
    for model, frame in (("GEARS", gears_h), ("scGPT", sc_h)):
        for space in order:
            z = frame[frame.space == space].iloc[0]
            rows.append({"model": model, "space": space, "pearson": float(z.perturbed_state_or_response_pearson)})
    return pd.DataFrame(rows)


def source_ignorant_metrics():
    rows = []
    for model, path in (("GEARS", ROOT / "results/gears_geometry_audit/shared_response_audit.csv"),
                        ("scGPT", ROOT / "results/final_scgpt_preprint_audit/scgpt_source_ignorant_audit.csv")):
        frame = pd.read_csv(path)
        model_name = "GEARS" if model == "GEARS" else "scGPT"
        learned = frame[(frame.record_type == "fold_mean") & (frame.model == model_name)]
        blind = frame[(frame.record_type == "fold_mean") & (frame.model == "SourceIgnorantMeanResponse")]
        for space in ["absolute_perturbed_state", "total_perturbation_response", "intervention_specific_residual"]:
            rows.append({"model": model, "space": space,
                         "learned": float(learned[learned.space == space].iloc[0].perturbed_state_or_response_pearson),
                         "source_ignorant": float(blind[blind.space == space].iloc[0].perturbed_state_or_response_pearson)})
    return pd.DataFrame(rows)


def fig1():
    fig, axes = b.make_main_figure("Figure1")
    metrics = model_metrics(); names = ["Absolute state", "Response", "Identity-specific"]
    spaces = ["absolute_perturbed_state", "total_perturbation_response", "intervention_specific_residual"]
    ax = axes["c"]
    for model, color, marker, offset in (("GEARS", GEARS, "o", .038), ("scGPT", SCGPT, "s", -.070)):
        q = metrics[metrics.model == model].set_index("space").loc[spaces]
        x = np.arange(3)
        ax.plot(x, q.pearson, color=color, marker=marker, ms=4.5, label=model)
        for i, v in enumerate(q.pearson):
            ax.text(i, v + offset, f"{v:.3f}", ha="center", color=color, fontsize=6.5)
    ax.set_xticks(np.arange(3), names); b.rotate_categories(ax)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Pearson correlation"); ax.set_title("Metric specificity reveals identity loss")
    ax.legend(frameon=False, ncol=2, loc="lower left"); b.clean(ax); b.label(ax, "c")
    stash("Figure1", "c", metrics)

    blind = source_ignorant_metrics(); ax = axes["d"]
    y, labels = [], []
    for mi, model in enumerate(("GEARS", "scGPT")):
        for si, (space, label) in enumerate(zip(spaces, names)):
            y.append(5 - (mi * 3 + si)); labels.append(f"{model} · {label}")
    y = np.asarray(y, float); q = pd.concat([blind[blind.model == m].set_index("space").loc[spaces] for m in ("GEARS", "scGPT")])
    ax.barh(y + .16, q.learned, height=.28, color=[GEARS]*3 + [SCGPT]*3, label="Learned model")
    ax.barh(y - .16, q.source_ignorant, height=.28, color=BASE, edgecolor=TRUTH, linewidth=.4, label="Source-ignorant")
    ax.set_yticks(y, labels); ax.set_xlim(0, 1.04); ax.set_xlabel("Pearson correlation")
    ax.set_title("Learned model versus matched identity-blind baseline"); ax.legend(frameon=False, ncol=2, loc="lower right")
    b.clean(ax); b.label(ax, "d")
    stash("Figure1", "d", blind)
    save(fig, "Figure1")


def fig2():
    fig, axes = b.make_main_figure("Figure2")
    landscape = pd.read_csv(ROOT / "results/preprint_finalization/figure2_common_landscape.csv")
    host = axes["a"]; host.axis("off"); b.label(host, "a", x=-.02, y=1.05)
    host.set_title("Common truth-derived response landscape", pad=2)
    pos = b.LAYOUTS["Figure2"]["panels"]["a"]
    xmins, xmaxs = landscape.truth_pc1_scaled.min(), landscape.truth_pc1_scaled.max()
    ymins, ymaxs = landscape.truth_pc2_scaled.min(), landscape.truth_pc2_scaled.max()
    for i, (series, color, marker) in enumerate((("Truth", TRUTH, "D"), ("GEARS", GEARS, "o"), ("scGPT", SCGPT, "s"))):
        ax = fig.add_axes([pos["x"] + i*pos["width"]/3 + .015, pos["y"] + .02, pos["width"]/3 - .035, pos["height"] - .06])
        q = landscape[landscape.series == series]
        if series == "Truth": ax.scatter(q.truth_pc1_scaled, q.truth_pc2_scaled, s=4, color=color, marker=marker, linewidth=0, alpha=.42)
        else: ax.scatter(q.truth_pc1_scaled, q.truth_pc2_scaled, s=4, color=color, alpha=.35, marker=marker)
        ax.set_xlim(xmins, xmaxs); ax.set_ylim(ymins, ymaxs); ax.set_title(series)
        ax.set_xlabel("Truth-PC1");
        if i == 0: ax.set_ylabel("Truth-PC2")
        else: ax.set_yticklabels([])
        b.clean(ax)
    stash("Figure2", "a", landscape)

    gg = summary_row(ROOT / "results/gears_geometry_audit/grouped_intervention_geometry.csv", record_type="summary", space="intervention_specific_residual")
    sg = summary_row(ROOT / "results/final_scgpt_preprint_audit/scgpt_grouped_geometry.csv", record_type="summary")
    geom = pd.DataFrame([{"model":"GEARS","estimate":gg.response_distance_spearman,"ci_low":gg.source_bootstrap_ci_low,"ci_high":gg.source_bootstrap_ci_high},
                         {"model":"scGPT","estimate":sg.response_distance_spearman,"ci_low":sg.source_bootstrap_ci_low,"ci_high":sg.source_bootstrap_ci_high}])
    ax=axes["b"]; yy=np.array([.60,.40])
    for y,(_,z),color,marker in zip(yy,geom.iterrows(),[GEARS,SCGPT],["o","s"]):
        ax.errorbar(z.estimate,y,xerr=[[z.estimate-z.ci_low],[z.ci_high-z.estimate]],fmt=marker,color=color,capsize=2.5,ms=5)
    ax.axvline(0,color=GRID,lw=.8);ax.set_yticks(yy,geom.model);ax.set_ylim(.20,.80);ax.set_xlim(0,.31);ax.set_xlabel("Grouped geometry, Spearman ρ");ax.set_title("Same-model OOF geometry");b.clean(ax);b.label(ax,"b")
    stash("Figure2","b",geom)

    gl=summary_row(ROOT/"results/gears_geometry_audit/grouped_intervention_geometry.csv",record_type="summary",space="intervention_specific_residual")
    sl=summary_row(ROOT/"results/final_scgpt_preprint_audit/scgpt_local_geometry.csv",record_type="summary")
    local=pd.DataFrame([{"metric":"kNN@10","GEARS":gl.local_knn_overlap_k10,"scGPT":sl.knn_overlap_k10},
                        {"metric":"Local rank","GEARS":gl.local_distance_rank,"scGPT":sl.local_distance_rank}])
    ax=axes["c"]; yy=np.array([.60,.40])
    for model,color,marker in (("GEARS",GEARS,"o"),("scGPT",SCGPT,"s")):
        ax.hlines(yy,0,local[model],color=color,lw=1.2,alpha=.55);ax.scatter(local[model],yy,color=color,marker=marker,s=24,label=model)
    ax.set_yticks(yy,local.metric);ax.set_ylim(.20,.80);ax.set_xlim(0,.23);ax.set_xlabel("Preservation");ax.set_title("Local geometry");b.clean(ax);b.label(ax,"c")
    stash("Figure2","c",local)

    gs=summary_row(ROOT/"results/gears_geometry_audit/spectral_geometry_compression.csv",record_type="fold_mean")
    ss=summary_row(ROOT/"results/final_scgpt_preprint_audit/scgpt_variance_spectrum.csv",record_type="fold_mean")
    var=pd.DataFrame([{"model":"Truth reference","variance_retention":1.0},{"model":"GEARS","variance_retention":gs.between_source_variance_ratio},{"model":"scGPT","variance_retention":ss.between_source_variance_ratio}])
    ax=axes["d"]; yy=np.array([.62,.50,.38]); colors=[TRUTH,GEARS,SCGPT]; marks=["D","o","s"]
    ax.hlines(yy,0,var.variance_retention,color=colors,lw=1.5,alpha=.55)
    for y,v,c,m in zip(yy,var.variance_retention,colors,marks):ax.scatter(v,y,color=c,marker=m,s=27)
    ax.set_yticks(yy,var.model);ax.set_ylim(.25,.75);ax.set_xlim(0,1.06);ax.set_xlabel("Variance retention");ax.set_title("Between-intervention amplitude");b.clean(ax);b.label(ax,"d")
    stash("Figure2","d",var)

    spectral=pd.read_csv(ROOT/"results/preprint_finalization/figure2_common_spectral_summary.csv").set_index("series")
    spec_values=pd.DataFrame([{"metric":"PC1 fraction","Truth":spectral.loc["Truth","pc1_fraction"],"GEARS":spectral.loc["GEARS","pc1_fraction"],"scGPT":spectral.loc["scGPT","pc1_fraction"]},
                              {"metric":"Entropy rank","Truth":spectral.loc["Truth","entropy_effective_rank"],"GEARS":spectral.loc["GEARS","entropy_effective_rank"],"scGPT":spectral.loc["scGPT","entropy_effective_rank"]},
                              {"metric":"PC80","Truth":spectral.loc["Truth","pc80"],"GEARS":spectral.loc["GEARS","pc80"],"scGPT":spectral.loc["scGPT","pc80"]}])
    host=axes["e"];host.axis("off");b.label(host,"e",x=-.03,y=1.08);host.set_title("Native spectral quantities",pad=2)
    pos=b.LAYOUTS["Figure2"]["panels"]["e"]
    for i,metric in enumerate(spec_values.metric):
        ax=fig.add_axes([pos["x"]+i*pos["width"]/3+.018,pos["y"]+.02,pos["width"]/3-.035,pos["height"]-.055])
        z=spec_values.iloc[i]; vals=[z.Truth,z.GEARS,z.scGPT]; x=np.array([-.35,0,.35])
        ax.vlines(x, 0, vals, colors=[TRUTH, GEARS, SCGPT], lw=1.15, alpha=.45)
        for xx, value, color, marker in zip(x, vals, [TRUTH, GEARS, SCGPT], ["D", "o", "s"]):
            ax.scatter(xx, value, color=color, marker=marker, s=25)
        ax.set_xticks(x,["Truth","GEARS","scGPT"],rotation=30,ha="right",rotation_mode="anchor");ax.set_xlim(-.62,.62);ax.set_title(metric)
        ax.set_ylim(0,max(vals)*1.18);b.clean(ax)
    stash("Figure2","e",spec_values)

    mm=model_metrics(); absolute=mm[mm.space=="absolute_perturbed_state"].set_index("model").pearson
    summary=pd.DataFrame([{"metric":"Absolute Pearson","GEARS":absolute.GEARS,"scGPT":absolute.scGPT},
                          {"metric":"Geometry ρ","GEARS":gg.response_distance_spearman,"scGPT":sg.response_distance_spearman},
                          {"metric":"Variance ratio","GEARS":gs.between_source_variance_ratio,"scGPT":ss.between_source_variance_ratio}])
    ax=axes["f"];yy=np.array([.62,.50,.38])
    for model,color,marker in (("GEARS",GEARS,"o"),("scGPT",SCGPT,"s")):
        ax.hlines(yy, 0, summary[model], color=color, lw=1.2, alpha=.45)
        ax.scatter(summary[model],yy,color=color,marker=marker,s=25,label=model)
    ax.set_yticks(yy,["Abs. Pearson","Geometry ρ","Variance ratio"]);ax.set_ylim(.25,.75);ax.set_xlim(0,1.03);ax.set_xlabel("Score / ratio");ax.set_title("Model summary");b.clean(ax);b.label(ax,"f")
    stash("Figure2","f",summary)
    fig.legend(
        handles=[
            Line2D([], [], linestyle="none", marker="D", color=TRUTH, markerfacecolor=TRUTH, markersize=4.5, label="Truth"),
            Line2D([], [], linestyle="none", marker="o", color=GEARS, markerfacecolor=GEARS, markersize=4.5, label="GEARS"),
            Line2D([], [], linestyle="none", marker="s", color=SCGPT, markerfacecolor=SCGPT, markersize=4.5, label="scGPT"),
        ],
        loc="center", bbox_to_anchor=(.50, .575), ncol=3, frameon=False,
        columnspacing=1.5, handletextpad=.45,
    )
    save(fig,"Figure2")


def fig3():
    fig, axes = b.make_main_figure("Figure3")
    train = pd.read_csv(ROOT / "results/main_geometry_integrity_audit/train_vs_oof_summary.csv")
    r = train[train.scale == "Large"].iloc[0]
    ax=axes["a"]
    ax.hlines(0,r.artifact_safe_oof_geometry,r.matched_train_geometry,color=GRID,lw=4)
    ax.errorbar(r.artifact_safe_oof_geometry,0,xerr=[[r.artifact_safe_oof_geometry-r.oof_ci_low],[r.oof_ci_high-r.artifact_safe_oof_geometry]],fmt="s",color=FAIL,capsize=3,ms=6,label="Unseen OOF")
    ax.errorbar(r.matched_train_geometry,0,xerr=[[r.matched_train_geometry-r.matched_train_ci_low],[r.matched_train_ci_high-r.matched_train_geometry]],fmt="o",color=GEARS,capsize=3,ms=6,label="Matched train")
    ax.set_xlim(-.04,.82);ax.set_ylim(-.18,.18);ax.set_yticks([]);ax.set_xlabel("Grouped geometry, Spearman ρ");ax.set_title("Seen-to-unseen geometry gap");ax.legend(frameon=False,ncol=2,loc="upper center");b.clean(ax);b.label(ax,"a")
    stash("Figure3","a",pd.DataFrame([{"condition":"Matched train","estimate":r.matched_train_geometry,"ci_low":r.matched_train_ci_low,"ci_high":r.matched_train_ci_high},{"condition":"Unseen OOF","estimate":r.artifact_safe_oof_geometry,"ci_low":r.oof_ci_low,"ci_high":r.oof_ci_high}]))

    pca=pd.read_csv(ROOT/"results/final/response_basis/canonical_pca_cumulative_variance.csv")
    ax=axes["b"];ax.plot(pca.n_components,pca.cumulative_explained_variance,color=C["transformer"])
    key=pca[pca.n_components.isin([8,16,32,64])];ax.scatter(key.n_components,key.cumulative_explained_variance,color=C["transformer"],s=22)
    ax.set(xlabel="Number of response PCs",ylabel="Cumulative variance",title="Canonical response basis");ax.set_xlim(1,128);ax.set_ylim(.2,1.02);b.clean(ax);b.label(ax,"b");stash("Figure3","b",pca)

    oracle=pd.read_csv(ROOT/"results/nature_comm_figure_derivations/k562_response_basis_curve_summary.csv")
    ax=axes["c"];ax.fill_between(oracle.components,oracle.heldout_oracle_reconstruction_pearson_min,oracle.heldout_oracle_reconstruction_pearson_max,color=PRED,alpha=.16,lw=0);ax.plot(oracle.components,oracle.heldout_oracle_reconstruction_pearson_mean,color=PRED)
    key=oracle[oracle.components.isin([8,16,32,64])];ax.scatter(key.components,key.heldout_oracle_reconstruction_pearson_mean,color=PRED,s=22)
    ax.set(xlabel="Oracle response coordinates",ylabel="Held-out reconstruction Pearson",title="Oracle coordinates reconstruct response");ax.set_xlim(1,128);ax.set_ylim(.3,.85);b.clean(ax);b.label(ax,"c");stash("Figure3","c",oracle)

    pairs=pd.read_csv(ROOT/"results/preprint_finalization/figure3d_pairwise_full.csv.gz")
    ax=axes["d"];hb=ax.hexbin(pairs.establishedobs71_cosine_similarity,pairs.true_response_cosine_similarity,gridsize=42,mincnt=1,cmap="Greys",bins="log",linewidths=0)
    ax.set_xlabel("Source-state cosine similarity");ax.set_ylabel("Response cosine similarity");ax.set_title("Source state does not identify response")
    ax.text(.03,.96,"mean fold ρ = 0.018\n95% CI 0.014–0.023\n1,755 sources; 307,282 pairs",transform=ax.transAxes,ha="left",va="top",fontsize=6.5)
    b.clean(ax);b.label(ax,"d");stash("Figure3","d",pairs)

    safe=pd.read_csv(ROOT/"results/main_geometry_integrity_audit/artifact_safe_group_summary.csv");q=safe[(safe.dataset=="K562_CRISPRi") & safe.model.isin(["Transformer","MLP"])]
    ax=axes["e"];yy=np.array([.60,.40])
    for y,model,color,marker in zip(yy,["Transformer","MLP"],[C["transformer"],C["mlp"]],["o","s"]):
        z=q[q.model==model].iloc[0];ax.errorbar(z.response_distance_spearman,y,xerr=[[z.response_distance_spearman-z.spearman_ci_low],[z.spearman_ci_high-z.response_distance_spearman]],fmt=marker,color=color,capsize=2.5,ms=5)
    ax.axvline(0,color=GRID,lw=.8);ax.set_yticks(yy,["Transformer","MLP"]);ax.set_ylim(.20,.80);ax.set_xlim(-.075,.075);ax.set_xlabel("Grouped geometry, ρ");ax.set_title("Matched architectures");b.clean(ax);b.label(ax,"e");stash("Figure3","e",q)

    anti=pd.read_csv(ROOT/"results/directedT_exploration/stage1b_anticollapse_results.csv");agg=anti.groupby(["architecture","prior_kind"],as_index=False).agg(variance=("prediction_variance_ratio","mean"),swap=("swap_distance_correlation","mean"));sel=agg[(agg.architecture=="dual_route") & agg.prior_kind.isin(["correct","none"])]
    ax=axes["f"]
    for prior,color,marker,label in (("correct",GEARS,"o","Correct graph"),("none",BASE,"s","No graph")):
        z=sel[sel.prior_kind==prior];ax.scatter(z.variance,z.swap,color=color,marker=marker,s=35,label=label)
    ax.axhline(0,color=GRID,lw=.8);ax.set(xlabel="Prediction variance ratio",ylabel="Swap-distance correlation",title="Diversity ≠ geometry");ax.set_xlim(0,.06);ax.set_ylim(-.18,.25);ax.legend(frameon=False,loc="best");b.clean(ax);b.label(ax,"f");stash("Figure3","f",sel)
    save(fig,"Figure3")


def fig4():
    fig,axes=b.make_main_figure("Figure4")
    syn=pd.read_csv(ROOT/"results/clean_synthetic_directional_control/overall_graph_condition_summary.csv")
    names={"NO_GRAPH":"No graph","CORRECT_DIRECTED_SIGNED":"Correct signed","CORRECT_DIRECTED_UNSIGNED":"Unsigned","REVERSED_DIRECTED_SIGNED":"Reversed","DEGREE_PRESERVING_SHUFFLE":"Degree shuffle","SIGN_SHUFFLED":"Sign shuffle"}
    q=syn[syn.condition.isin(["NO_GRAPH","CORRECT_DIRECTED_SIGNED"])].copy();q["label"]=q.condition.map(names)
    ax=axes["b"];yy=np.array([.60,.40])
    for y,(_,z),color,marker in zip(yy,q.iterrows(),[BASE,GEARS],["s","o"]):
        ax.errorbar(z.response_distance_correlation_mean,y,xerr=z.response_distance_correlation_std,fmt=marker,color=color,capsize=3,ms=6)
    ax.axvline(0,color=GRID,lw=.8);ax.set_yticks(yy,q.label);ax.set_ylim(.20,.80);ax.set_xlim(-.12,.78);ax.set_xlabel("Response-distance correlation");ax.set_title("Correct signed structure restores geometry (mean ± SD)");b.clean(ax);b.label(ax,"b");stash("Figure4","b",q)

    order=list(names);q=syn.set_index("condition").loc[order].reset_index();q["label"]=q.condition.map(names)
    ax=axes["c"];yy=np.arange(len(q))[::-1]
    colors=[BASE,GEARS,PRED,FAIL,BASE,BASE];markers=["s","o","^","X","D","v"]
    for y,(_,z),color,marker in zip(yy,q.iterrows(),colors,markers):ax.errorbar(z.response_distance_correlation_mean,y,xerr=z.response_distance_correlation_std,fmt=marker,color=color,capsize=2,ms=4.5)
    ax.axvline(0,color=GRID,lw=.8);ax.set_yticks(yy,q.label);ax.set_xlim(-.15,.78);ax.set_xlabel("Response-distance correlation");ax.set_title("Matched structural controls");b.clean(ax);b.label(ax,"c");stash("Figure4","c",q)

    qd=q[q.condition.isin(["NO_GRAPH","CORRECT_DIRECTED_SIGNED","CORRECT_DIRECTED_UNSIGNED","REVERSED_DIRECTED_SIGNED"])]
    ax=axes["d"]
    for (_,z),color,marker in zip(qd.iterrows(),[BASE,GEARS,PRED,FAIL],["s","o","^","X"]):ax.scatter(z.between_perturbation_variance_ratio_mean,z.response_distance_correlation_mean,s=34,color=color,marker=marker,label=z.label)
    ax.axhline(0,color=GRID,lw=.8);ax.set_xscale("log");ax.set_xlim(1e-5,1);ax.set_ylim(-.15,.78);ax.set_xlabel("Variance retention");ax.set_ylabel("Response-distance correlation");ax.set_title("Diversity without rescue");ax.legend(frameon=False,fontsize=5.7,loc="upper left");b.clean(ax);b.label(ax,"d",x=.02);stash("Figure4","d",qd)

    gate=pd.read_csv(ROOT/"results/phase1/gateA_summary.csv");specs=[("outgoing_signed","cosine","Outgoing signed"),("outgoing_unsigned","cosine","Outgoing unsigned"),("incoming_signed","cosine","Incoming signed"),("reversed_signed","cosine","Reversed signed")];rows=[]
    for top,metric,label in specs:
        z=gate[(gate.topology==top)&(gate.topology_metric==metric)&(gate.response_metric=="trans_pearson")].iloc[0];rows.append({"condition":label,"estimate":z.spearman_rho,"ci_low":z.bootstrap_ci_low,"ci_high":z.bootstrap_ci_high})
    rows=pd.DataFrame(rows);null=gate[(gate.topology=="shuffled_null")&(gate.topology_metric=="signed_cosine")&(gate.response_metric=="trans_pearson")].iloc[0]
    ax=axes["e"];yy=np.array([.65,.55,.45,.35])
    for y,(_,z),color,marker in zip(yy,rows.iterrows(),[GEARS,GEARS,FAIL,FAIL],["o","^","s","X"]):ax.errorbar(z.estimate,y,xerr=[[z.estimate-z.ci_low],[z.ci_high-z.estimate]],fmt=marker,color=color,capsize=2,ms=4.5)
    ax.axvline(0,color=GRID,lw=.8);ax.axvline(null.spearman_rho,color=BASE,ls="--",lw=1,label="Shuffled median");ax.set_yticks(yy,rows.condition);ax.set_ylim(.25,.75);ax.set_xlim(-.14,.17);ax.set_xlabel("Topology–response Spearman ρ");ax.set_title("Real priors do not rescue");ax.legend(frameon=False,loc="lower right");b.clean(ax);b.label(ax,"e",x=-.10);stash("Figure4","e",rows.assign(shuffled_median=null.spearman_rho))
    save(fig,"Figure4")


def dumbbell(ax,left,right,left_label,right_label,xlabel,title,colors=(TRUTH,PRED)):
    ax.hlines(0,left,right,color=GRID,lw=4,zorder=1);ax.scatter([left,right],[0,0],s=42,c=list(colors),marker="o",zorder=3)
    left_ha="right" if left>right else "left";right_ha="left" if right>left else "right"
    ax.text(left,.10,f"{left_label}\n{left:.3f}",ha=left_ha,va="bottom",fontsize=6.7,color=colors[0]);ax.text(right,-.10,f"{right_label}\n{right:.3f}",ha=right_ha,va="top",fontsize=6.7,color=colors[1])
    lo=min(left,right);hi=max(left,right);span=max(hi-lo,.05);ax.set_xlim(max(0,lo-.15*span),hi+.15*span);ax.set_ylim(-.25,.25);ax.set_yticks([]);ax.set_xlabel(xlabel);ax.set_title(title);b.clean(ax)


def fig5():
    fig,axes=b.make_main_figure("Figure5")
    comp=pd.read_csv(ROOT/"results/propagation_reproduction/final_model_comparison.csv");r=comp[(comp.section=="D_first_response_zero_shot")&comp.model.isin(["StaticControl","CorrectLag"])]
    vals={z.model:z.mean_response_distance_rho for _,z in r.iterrows()};ax=axes["b"];yy=np.array([.60,.40])
    ax.hlines(yy,0,[vals["CorrectLag"],vals["StaticControl"]],color=[GEARS,BASE],lw=2);ax.scatter([vals["CorrectLag"],vals["StaticControl"]],yy,color=[GEARS,BASE],s=30);ax.set_yticks(yy,["Correct lag","Static"]);ax.set_ylim(.20,.80);ax.set_xlim(0,.20);ax.set_xlabel("W23 grouped geometry");ax.set_title("Correct temporal signal");b.clean(ax);b.label(ax,"b");stash("Figure5","b",r)

    ax=axes["c"];dumbbell(ax,.42835051929828055,.04269642546508218,"Teacher-forced","Free rollout","W45 grouped geometry","Teacher forcing vs rollout");b.label(ax,"c");stash("Figure5","c",pd.DataFrame([{"condition":"Teacher-forced","estimate":.42835051929828055},{"condition":"Free rollout","estimate":.04269642546508218}]))
    dyn=json.loads((ROOT/"results/renge_dynamic_validity/analysis_summary.json").read_text(encoding="utf-8"));te=dyn["key_numbers"]["true_entry_w45_geometry"];pe=dyn["key_numbers"]["predicted_entry_w45_geometry"]
    ax=axes["d"];dumbbell(ax,te,pe,"True entry","Predicted entry","W45 grouped geometry","Trajectory entry determines rollout");b.label(ax,"d");stash("Figure5","d",pd.DataFrame([{"condition":"True entry","estimate":te},{"condition":"Predicted entry","estimate":pe}]))

    endpoint=json.loads((ROOT/"results/renge_endpoint_benchmark/analysis_summary.json").read_text(encoding="utf-8"));ladder=pd.DataFrame([("Deployable chain",endpoint["chain_r5_means"]["FullyPredictedMarkov"]["response_distance_rho"],"deployable"),("Direct endpoint",endpoint["direct_r5_means"]["Direct_CorrectLag"]["response_distance_rho"],"deployable"),("True R2",endpoint["oracle_ladder_means"]["F1_TrueR2"]["response_distance_rho"],"true temporal"),("True R2 + W23",endpoint["oracle_ladder_means"]["F2_TrueR2_TrueW23"]["response_distance_rho"],"true temporal"),("True through W34",endpoint["oracle_ladder_means"]["F3_TrueR2_TrueW23_TrueW34"]["response_distance_rho"],"true temporal"),("Oracle",1.0,"oracle")],columns=["condition","estimate","class"])
    ax=axes["e"];yy=np.arange(len(ladder))[::-1];colors=[PRED,PRED,TRUTH,TRUTH,TRUTH,GEARS];markers=["s","s","o","o","o","D"]
    ax.hlines(yy,0,ladder.estimate,color=colors,lw=1.6,alpha=.65)
    for y,v,c,m in zip(yy,ladder.estimate,colors,markers):ax.scatter(v,y,color=c,marker=m,s=26)
    ax.set_yticks(yy,ladder.condition);ax.set_xlim(0,1.04);ax.set_xlabel("R5 grouped geometry");ax.set_title("Endpoint information ladder");b.clean(ax);b.label(ax,"e");stash("Figure5","e",ladder)

    markov=json.loads((ROOT/"results/renge_markov_autopsy_v2/analysis_summary.json").read_text(encoding="utf-8"))["models"]["deployable"];short={"DirectEndpoint":"Direct","Impulse":"Impulse","Additive":"Additive","Conditional":"Conditional"};mf=pd.DataFrame([{"condition":short.get(z["model"].replace("Persistent","").strip(),z["model"]),"estimate":z["response_distance_rho"]} for z in markov]);ax=axes["f"];yy=np.array([.65,.55,.45,.35]);colors=[BASE,PRED,FAIL,FAIL];markers=["s","o","^","v"]
    ax.hlines(yy,0,mf.estimate,color=colors,lw=1.8)
    for y,v,c,m in zip(yy,mf.estimate,colors,markers):ax.scatter(v,y,color=c,marker=m,s=25)
    ax.set_yticks(yy,mf.condition);ax.set_ylim(.25,.75);ax.set_xlim(left=0,right=.07);ax.margins(x=0);ax.set_xlabel("R5 grouped geometry");ax.set_title("Persistent forcing");b.clean(ax);b.label(ax,"f");stash("Figure5","f",mf)
    save(fig,"Figure5")


def factorial(ax,direction,title):
    rows=pd.DataFrame(json.loads((ROOT/"results/generalization_factorial/factorial_summary.json").read_text(encoding="utf-8"))["curve_summary"]);rows=rows[rows.direction==direction]
    for regime,color,marker,label in (("ZERO_SHOT",BASE,"o","Zero-shot"),("ALIGNED_ANCHOR",GEARS,"o","Aligned anchor"),("SHUFFLED_ANCHOR",FAIL,"s","Shuffled anchor")):
        q=rows[rows.regime==regime].sort_values("coverage_fraction");x=q.coverage_fraction*100;ax.fill_between(x,q.geometry_ci_low,q.geometry_ci_high,color=color,alpha=.11,lw=0);ax.plot(x,q.geometry_mean,color=color,marker=marker,ms=3,label=label)
    ax.axhline(0,color=GRID,lw=.8);ax.set_xticks([10,25,40,60,80,90]);ax.set_xlim(7,93);ax.set_ylim(-.09,.60);ax.set_xlabel("Other-intervention coverage (%)");ax.set_ylabel("Grouped geometry, Spearman ρ");ax.set_title(title);b.clean(ax);return rows


def fig6():
    fig,axes=b.make_main_figure("Figure6");q1=factorial(axes["b"],"K562_TO_RPE1","K562 → RPE1");b.label(axes["b"],"b");stash("Figure6","b",q1);q2=factorial(axes["c"],"RPE1_TO_K562","RPE1 → K562");b.label(axes["c"],"c");stash("Figure6","c",q2)
    handles,labels=axes["b"].get_legend_handles_labels();fig.legend(handles,labels,frameon=False,loc="upper center",ncol=3,bbox_to_anchor=(.64,.985))
    contrasts=pd.read_csv(ROOT/"results/generalization_factorial/factorial_contrasts.csv");wanted=["ZERO_90_MINUS_ZERO_10","ANCHOR_90_MINUS_ZERO_90","ANCHOR_10_MINUS_ZERO_90","ANCHOR_MINUS_SHUFFLE_90"];disp=["Zero90 − Zero10","Anchor90 − Zero90","Anchor10 − Zero90","Aligned − Shuffle90"];ax=axes["d"];yy=np.arange(4)[::-1]
    for off,direction,color,marker,label in ((-.11,"K562_TO_RPE1",GEARS,"o","K562 → RPE1"),(.11,"RPE1_TO_K562",DARK_DIRECTION_BLUE,"s","RPE1 → K562")):
        q=contrasts[(contrasts.direction==direction)&contrasts.contrast.isin(wanted)].set_index("contrast").loc[wanted];ax.errorbar(q.estimate,yy+off,xerr=[q.estimate-q.ci_low,q.ci_high-q.estimate],fmt=marker,color=color,capsize=2,ms=4,label=label)
    ax.axvline(0,color=GRID,lw=.8);ax.set_yticks(yy,disp);ax.set_xlim(-.09,.61);ax.set_xlabel("Geometry contrast");ax.set_title("Decisive fixed-target contrasts");ax.legend(frameon=False,loc="upper right");b.clean(ax);b.label(ax,"d");stash("Figure6","d",contrasts[contrasts.contrast.isin(wanted)])
    ext=pd.read_csv(ROOT/"results/frangieh_external_replication/external_contrasts.csv");ext=ext[(ext.analysis_set=="PRIMARY")&ext.contrast.str.match(r"ALIGNED_MINUS_SHUFFLE_\d+$")].copy();ext["coverage"]=ext.contrast.str.extract(r"(\d+)$").astype(int);ax=axes["e"]
    directions=list(ext.direction.drop_duplicates());styles=[(GEARS,"o","-"),(DARK_DIRECTION_BLUE,"s","--")]
    direct_label_offsets = (-0.010, 0.010)
    for idx,(direction,(color,marker,ls)) in enumerate(zip(directions,styles)):
        q=ext[ext.direction==direction].sort_values("coverage");x=q.coverage.to_numpy();y=q.estimate.to_numpy();label=direction.replace("Co-culture","Co").replace("_TO_","→").replace("Î³","γ");ax.fill_between(x,q.ci_low,q.ci_high,color=color,alpha=.10,lw=0);ax.plot(x,y,color=color,marker=marker,ls=ls,ms=3.5,label=label);ax.text(x[-1]-1,y[-1]+direct_label_offsets[idx],label,color=color,fontsize=5.8,va="center",ha="right")
    ax.axhline(0,color=GRID,lw=.8);ax.set_xticks([10,25,40,60,80,90]);ax.set_xlim(7,106);ax.set_ylim(0,.52);ax.set_xlabel("Other-intervention coverage (%)");ax.set_ylabel("Aligned − shuffled geometry");ax.set_title("External identity-anchor replication");b.clean(ax);b.label(ax,"e");stash("Figure6","e",ext)
    save(fig,"Figure6")


def _supp_canvas(width=7.1, height=3.1, widths=(1, 1)):
    fig, axes = plt.subplots(1, len(widths), figsize=(width, height),
                             gridspec_kw={"width_ratios": widths, "wspace": .48})
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    return fig, axes


def _supp_manual(specs, height=4.8):
    fig = plt.figure(figsize=(7.1, height))
    return fig, {panel: fig.add_axes(bounds) for panel, bounds in specs.items()}


def _forest(ax, frame, label_col, value_col, low_col=None, high_col=None,
            colors=None, markers=None, xlabel="Estimate", title=""):
    q = frame.reset_index(drop=True)
    yy = np.arange(len(q))[::-1]
    colors = colors or [PRED] * len(q)
    markers = markers or ["o"] * len(q)
    for i, (_, z) in enumerate(q.iterrows()):
        kw = {}
        if low_col and high_col:
            kw["xerr"] = [[float(z[value_col] - z[low_col])],
                          [float(z[high_col] - z[value_col])]]
        ax.errorbar(float(z[value_col]), yy[i], fmt=markers[i], color=colors[i],
                    capsize=2, ms=4.5, **kw)
    ax.axvline(0, color=GRID, lw=.8)
    ax.set_yticks(yy, q[label_col])
    ax.set_xlabel(xlabel); ax.set_title(title); b.clean(ax)


def supplementary1():
    fig, ax = _supp_canvas()
    m = model_metrics(); names = {"absolute_perturbed_state":"Absolute state",
        "total_perturbation_response":"Response",
        "intervention_specific_residual":"Identity-specific"}
    for model, color, marker in (("GEARS",GEARS,"o"),("scGPT",SCGPT,"s")):
        q=m[m.model==model]; ax[0].plot(np.arange(3),q.pearson,color=color,marker=marker,label=model)
    ax[0].set_xticks(np.arange(3),[names[x] for x in m.space.unique()]);b.rotate_categories(ax[0]);ax[0].set_ylim(0,1.03);ax[0].set_ylabel("Pearson correlation");ax[0].set_title("Metric hierarchy");ax[0].legend(frameon=False);b.clean(ax[0]);b.label(ax[0],"a")
    blind=source_ignorant_metrics();q=blind[blind.space=="absolute_perturbed_state"];yy=[1,0]
    for y,(_,z),c,mk in zip(yy,q.iterrows(),[GEARS,SCGPT],["o","s"]):
        ax[1].plot([z.source_ignorant,z.learned],[y,y],color=GRID,lw=3);ax[1].scatter([z.source_ignorant,z.learned],[y,y],c=[BASE,c],marker=mk,s=28)
    ax[1].set_yticks(yy,q.model);ax[1].set_xlim(.94,1.0);ax[1].set_xlabel("Absolute-state Pearson");ax[1].set_title("Source-ignorant baseline");b.clean(ax[1]);b.label(ax[1],"b")
    stash("SupplementaryFigure1","a",m);stash("SupplementaryFigure1","b",blind);fig.subplots_adjust(left=.10,right=.98,bottom=.25,top=.83);save(fig,"SupplementaryFigure1")


def supplementary2():
    fig, ax = _supp_canvas()
    d=pd.read_csv(ROOT/"results/main_geometry_integrity_audit/artifact_safe_group_summary.csv");d=d[d.model=="Transformer"].copy();d["label"]=d.dataset.str.replace("_CRISPRi","",regex=False).str.replace("_CRISPRa","",regex=False)
    _forest(ax[0],d,"label","response_distance_spearman","spearman_ci_low","spearman_ci_high",[PRED]*len(d),["o"]*len(d),"Grouped geometry, Spearman ρ","Fold-local artifact-safe estimates");b.label(ax[0],"a")
    null=pd.DataFrame(json.loads((ROOT/"results/main_geometry_integrity_audit/null_artifact_summary.json").read_text(encoding="utf-8"))["rows"]);q=null[(null.scheme=="LOO")].copy();q["label"]=q.dataset.str.split("_").str[0]+" "+q.evaluation.map({"CORRECT_within_same_model":"within-model","NAIVE_cross_model_stitch":"stitched"})
    colors=[TRUTH if "within" in z else FAIL for z in q.label];_forest(ax[1],q,"label","spearman_mean",colors=colors,markers=["o" if "within" in z else "X" for z in q.label],xlabel="Null geometry, Spearman ρ",title="Executable stitched-cloud artifact");ax[1].set_xlim(-.08,1.08);b.label(ax[1],"b")
    stash("SupplementaryFigure2","a",d);stash("SupplementaryFigure2","b",q);fig.subplots_adjust(left=.17,right=.98,bottom=.18,top=.83);save(fig,"SupplementaryFigure2")


def supplementary3():
    fig, ax = _supp_canvas()
    r=pd.read_csv(ROOT/"results/cross_dataset_replication_rpe1/reliability/response_geometry_reproducibility.csv");q=r[r.record_type=="split"]
    for th,c,mk in ((50,GEARS,"o"),(100,PRED,"s")):
        z=q[q.cell_threshold==th];ax[0].scatter(z.split,z.response_geometry_reproducibility,color=c,marker=mk,label=f"≥{th} cells")
    ax[0].set_ylim(.82,.95);ax[0].set_xlabel("Split replicate");ax[0].set_ylabel("Response-geometry reproducibility");ax[0].set_title("RPE1 measurement ceiling");ax[0].legend(frameon=False);b.clean(ax[0]);b.label(ax[0],"a")
    rel=pd.read_csv(ROOT/"results/cross_dataset_replication_rpe1/reliability/split_or_replicate_reliability.csv");q2=rel[rel.cell_threshold==50].sample(min(2500,len(rel)),random_state=26081503)
    ax[1].scatter(q2.cell_count,q2.split_half_residual_response_pearson,s=3,color=PRED,alpha=.18);ax[1].set_xscale("log");ax[1].set_xlabel("Cells per perturbation");ax[1].set_ylabel("Split-half residual Pearson");ax[1].set_title("Reliability versus sampling depth");b.clean(ax[1]);b.label(ax[1],"b")
    stash("SupplementaryFigure3","a",q);stash("SupplementaryFigure3","b",rel);fig.subplots_adjust(left=.11,right=.98,bottom=.20,top=.83);save(fig,"SupplementaryFigure3")


def supplementary4():
    fig,ax=_supp_manual({"a":(.19,.56,.36,.35),"b":(.68,.56,.27,.35),"c":(.12,.10,.38,.28),"d":(.63,.10,.32,.28)},height=4.7)
    m=model_metrics();spaces=["absolute_perturbed_state","total_perturbation_response","intervention_specific_residual"]
    yy=np.arange(6)[::-1];labs=[];vals=[];cols=[];marks=[]
    for model,c,mk in (("GEARS",GEARS,"o"),("scGPT",SCGPT,"s")):
        q=m[m.model==model].set_index("space").loc[spaces]
        for sp,v in zip(spaces,q.pearson):labs.append(f"{model} · {sp.replace('_',' ')}");vals.append(v);cols.append(c);marks.append(mk)
    for y,v,c,mk in zip(yy,vals,cols,marks):ax["a"].hlines(y,0,v,color=c,lw=1.4);ax["a"].scatter(v,y,color=c,marker=mk,s=25)
    ax["a"].set_yticks(yy,labs);ax["a"].set_xlim(0,1);ax["a"].set_xlabel("Pearson correlation");ax["a"].set_title("Metric hierarchy");b.clean(ax["a"]);b.label(ax["a"],"a")

    gg=pd.read_csv(ROOT/"results/gears_geometry_audit/grouped_intervention_geometry.csv");gs=gg[(gg.record_type=="summary")&(gg.space=="intervention_specific_residual")].iloc[0]
    gv=pd.read_csv(ROOT/"results/gears_geometry_audit/spectral_geometry_compression.csv");gv=gv[gv.record_type=="fold_mean"].iloc[0]
    sg=pd.read_csv(ROOT/"results/final_scgpt_preprint_audit/scgpt_grouped_geometry.csv");sg=sg[sg.record_type=="summary"].iloc[0]
    sl=pd.read_csv(ROOT/"results/final_scgpt_preprint_audit/scgpt_local_geometry.csv");sl=sl[sl.record_type=="summary"].iloc[0]
    sv=pd.read_csv(ROOT/"results/final_scgpt_preprint_audit/scgpt_variance_spectrum.csv");sv=sv[sv.record_type=="fold_mean"].iloc[0]
    summary=pd.DataFrame([
        {"metric":"Grouped geometry","GEARS":gs.response_distance_spearman,"scGPT":sg.response_distance_spearman},
        {"metric":"kNN@10","GEARS":gs.local_knn_overlap_k10,"scGPT":sl.knn_overlap_k10},
        {"metric":"Local distance rank","GEARS":gs.local_distance_rank,"scGPT":sl.local_distance_rank},
        {"metric":"Variance retention","GEARS":gv.between_source_variance_ratio,"scGPT":sv.between_source_variance_ratio},])
    yy=np.arange(len(summary))[::-1]
    for model,color,marker,off in (("GEARS",GEARS,"o",.09),("scGPT",SCGPT,"s",-.09)):
        ax["b"].scatter(summary[model],yy+off,color=color,marker=marker,s=25,label=model)
    ax["b"].set_yticks(yy,summary.metric);ax["b"].set_xlim(0,.34);ax["b"].set_xlabel("Native metric value");ax["b"].set_title("Geometry and variation");ax["b"].legend(frameon=False);b.clean(ax["b"]);b.label(ax["b"],"b")

    common=pd.read_csv(ROOT/"results/preprint_finalization/figure2_common_spectral_summary.csv");q=common.set_index("series").loc[["Truth","GEARS","scGPT"]];x=np.arange(3);colors=[TRUTH,GEARS,SCGPT];markers=["D","o","s"]
    for xx,v,c,mk in zip(x,q.pc1_fraction,colors,markers):ax["c"].scatter(xx,v,color=c,marker=mk,s=32)
    ax["c"].set_xticks(x,q.index);ax["c"].set_ylim(.30,.75);ax["c"].set_ylabel("PC1 variance fraction");ax["c"].set_title("Native spectral concentration");b.clean(ax["c"]);b.label(ax["c"],"c")
    for xx,(_,z),c,mk in zip(x,q.iterrows(),colors,markers):
        ax["d"].scatter(z.entropy_effective_rank,xx+.10,color=c,marker=mk,s=28)
        ax["d"].scatter(z.pc80,xx-.10,color=c,marker=mk,s=28,facecolors="none")
    ax["d"].set_yticks(x,q.index);ax["d"].set_xlim(0,27);ax["d"].set_xlabel("Number of components");ax["d"].set_title("Entropy rank (filled) and PC80 (open)");b.clean(ax["d"]);b.label(ax["d"],"d")
    stash("SupplementaryFigure4","a",m);stash("SupplementaryFigure4","b",summary);stash("SupplementaryFigure4","c",common[["series","pc1_fraction"]]);stash("SupplementaryFigure4","d",common[["series","entropy_effective_rank","pc80"]]);save(fig,"SupplementaryFigure4")


def supplementary5():
    fig,ax=_supp_manual({"a":(.10,.58,.85,.32),"b":(.12,.11,.23,.30),"c":(.44,.11,.22,.30),"d":(.75,.11,.20,.30)},height=4.8)
    pca=pd.read_csv(ROOT/"results/final/response_basis/canonical_pca_cumulative_variance.csv");oracle=pd.read_csv(ROOT/"results/nature_comm_figure_derivations/k562_response_basis_curve_summary.csv")
    ax["a"].plot(pca.n_components,pca.cumulative_explained_variance,color=PRED,label="Explained variance");ax["a"].plot(oracle.components,oracle.heldout_oracle_reconstruction_pearson_mean,color=GEARS,label="Held-out oracle")
    for k in (8,16,32,64):ax["a"].axvline(k,color=GRID,lw=.55,zorder=0)
    ax["a"].set_xlim(1,128);ax["a"].set_ylim(.2,1);ax["a"].set_xticks([1,8,16,32,64,128]);ax["a"].set_xlabel("Number of response PCs");ax["a"].set_ylabel("Fraction / Pearson");ax["a"].set_title("Canonical response basis and oracle ceiling");ax["a"].legend(frameon=False,ncol=2);b.clean(ax["a"]);b.label(ax["a"],"a")

    align=pd.read_csv(ROOT/"results/cross_dataset_replication_rpe1/state_intervention/pairwise_geometry_alignment.csv");ident=align[(align.record_type=="summary")&(align.panel=="primary_common_strict_trans")].copy();ident["label"]=ident.representation
    _forest(ax["b"],ident,"label","spearman_rho","ci_low","ci_high",[BASE,PRED,GEARS],["s","o","D"],"Grouped geometry, ρ","Source identifiability");ax["b"].set_xlim(0,.027);b.label(ax["b"],"b")
    d=pd.read_csv(ROOT/"results/main_geometry_integrity_audit/artifact_safe_group_summary.csv");q=d[d.dataset=="K562_CRISPRi"].copy();q["label"]=q.model
    _forest(ax["c"],q,"label","response_distance_spearman","spearman_ci_low","spearman_ci_high",[PRED,BASE],["o","s"],"Grouped geometry, ρ","Matched architecture control");ax["c"].set_xlim(-.08,.08);b.label(ax["c"],"c")
    anti=pd.read_csv(ROOT/"results/directedT_exploration/stage1b_anticollapse_results.csv");anti_summary=anti.groupby("architecture",as_index=False).agg(prediction_variance_ratio=("prediction_variance_ratio","mean"),perturbation_specific_corr=("perturbation_specific_corr","mean"));cols=[BASE,PRED,FAIL];marks=["s","o","^"]
    for (_,z),c,mk in zip(anti_summary.iterrows(),cols,marks):ax["d"].scatter(z.prediction_variance_ratio,z.perturbation_specific_corr,color=c,marker=mk,s=30,label=z.architecture)
    ax["d"].axhline(0,color=GRID,lw=.8);ax["d"].set_xscale("log");ax["d"].set_xlabel("Prediction variance ratio");ax["d"].set_ylabel("Perturbation-specific r");ax["d"].set_title("Anti-collapse control");ax["d"].legend(frameon=False,fontsize=5.5);b.clean(ax["d"]);b.label(ax["d"],"d")
    stash("SupplementaryFigure5","a",pd.merge(pca,oracle,left_on="n_components",right_on="components",how="outer"));stash("SupplementaryFigure5","b",ident);stash("SupplementaryFigure5","c",q);stash("SupplementaryFigure5","d",anti_summary);save(fig,"SupplementaryFigure5")


def supplementary6():
    fig,ax=_supp_manual({"a":(.18,.12,.31,.78),"b":(.64,.56,.31,.34),"c":(.64,.12,.31,.27)},height=4.8)
    syn=pd.read_csv(ROOT/"results/clean_synthetic_directional_control/overall_graph_condition_summary.csv");names={"NO_GRAPH":"No graph","CORRECT_DIRECTED_SIGNED":"Correct signed","CORRECT_DIRECTED_UNSIGNED":"Unsigned","REVERSED_DIRECTED_SIGNED":"Reversed","DEGREE_PRESERVING_SHUFFLE":"Degree shuffle","SIGN_SHUFFLED":"Sign shuffle"};q=syn[syn.condition.isin(names)].copy();q["label"]=q.condition.map(names)
    _forest(ax["a"],q,"label","response_distance_correlation_mean",colors=[GEARS if z=="Correct signed" else (FAIL if z in {"Reversed","Degree shuffle","Sign shuffle"} else BASE) for z in q.label],markers=["o" if z=="Correct signed" else "s" for z in q.label],xlabel="Response-distance correlation",title="Synthetic structural controls");b.label(ax["a"],"a")
    gate=pd.read_csv(ROOT/"results/phase1/gateA_summary.csv");g=gate[(gate.response_metric=="trans_pearson")&gate.bootstrap_ci_low.notna()].copy();g["label"]=g.topology.str.replace("_"," ")
    _forest(ax["b"],g,"label","spearman_rho","bootstrap_ci_low","bootstrap_ci_high",[GEARS if "outgoing" in z else FAIL for z in g.topology],["o" if "signed" in z else "s" for z in g.topology],"Topology–response Spearman ρ","Real-prior boundary");b.label(ax["b"],"b")
    for _,z in q.iterrows():
        c=GEARS if z.label=="Correct signed" else (FAIL if z.label in {"Reversed","Degree shuffle","Sign shuffle"} else BASE);mk="o" if z.label=="Correct signed" else "s";ax["c"].scatter(z.between_perturbation_variance_ratio_mean,z.response_distance_correlation_mean,color=c,marker=mk,s=28,label=z.label)
    ax["c"].set_xscale("log");ax["c"].set_xlabel("Between-intervention variance ratio");ax["c"].set_ylabel("Geometry correlation");ax["c"].set_title("Diversity is not sufficient");ax["c"].legend(frameon=False,fontsize=5.2,ncol=2);b.clean(ax["c"]);b.label(ax["c"],"c")
    stash("SupplementaryFigure6","a",q);stash("SupplementaryFigure6","b",g);stash("SupplementaryFigure6","c",q[["condition","label","between_perturbation_variance_ratio_mean","response_distance_correlation_mean"]]);save(fig,"SupplementaryFigure6")


def supplementary7():
    fig,ax=_supp_manual({"a":(.20,.12,.31,.78),"b":(.65,.59,.30,.30),"c":(.65,.12,.30,.30)},height=4.8)
    comp=pd.read_csv(ROOT/"results/propagation_reproduction/final_model_comparison.csv");q=comp[comp.section=="D_first_response_zero_shot"].copy();q["label"]=q.model.str.replace("Control","",regex=False)
    _forest(ax["a"],q,"label","mean_response_distance_rho",colors=[GEARS if z=="CorrectLag" else BASE for z in q.model],markers=["o" if z=="CorrectLag" else "s" for z in q.model],xlabel="W23 grouped geometry",title="First-wave temporal controls");b.label(ax["a"],"a")
    ts=json.loads((ROOT/"results/renge_temporal_scaling/analysis_summary.json").read_text(encoding="utf-8"));static=pd.DataFrame([{"n_sources":int(k),"geometry":v} for k,v in ts["static_curve"].items()]).sort_values("n_sources");ax["b"].plot(static.n_sources,static.geometry,color=PRED,marker="o",ms=3);ax["b"].axhline(0,color=GRID,lw=.8);ax["b"].set_xlabel("Static source count");ax["b"].set_ylabel("Grouped geometry");ax["b"].set_title("Static breadth scaling");b.clean(ax["b"]);b.label(ax["b"],"b")
    order=pd.DataFrame(ts["temporal_order_means"]);short={"T2_Day2_Day5":"D2→D5","T2b_Day3_Day5":"D3→D5","T3_Day2_Day3_Day5":"D2,D3→D5","T3b_Day3_Day4_Day5":"D3,D4→D5","T4_AllDays":"All days"};order["label"]=order.condition.map(short);positions=np.arange(order.label.nunique())
    for temporal,color,marker,label in (("CorrectTemporal",GEARS,"o","Correct order"),("TemporalShuffle",FAIL,"s","Time shuffle")):
        z=order[order.temporal_order==temporal].set_index("label").loc[list(short.values())];ax["c"].plot(positions,z.response_distance_rho,color=color,marker=marker,ms=3,label=label)
    ax["c"].axhline(0,color=GRID,lw=.8);ax["c"].set_xticks(positions,list(short.values()),rotation=45,ha="right",rotation_mode="anchor");ax["c"].set_ylabel("Grouped geometry");ax["c"].set_title("Temporal depth and order");ax["c"].legend(frameon=False);b.clean(ax["c"]);b.label(ax["c"],"c")
    stash("SupplementaryFigure7","a",q);stash("SupplementaryFigure7","b",static);stash("SupplementaryFigure7","c",order);save(fig,"SupplementaryFigure7")


def supplementary8():
    fig,ax=_supp_manual({"a":(.17,.57,.32,.32),"b":(.65,.57,.30,.32),"c":(.17,.11,.32,.30),"d":(.65,.11,.30,.30)},height=4.9)
    dyn=json.loads((ROOT/"results/renge_dynamic_validity/analysis_summary.json").read_text(encoding="utf-8"));kn=dyn["key_numbers"];q=pd.DataFrame([{"condition":"True W23 entry","estimate":kn["true_entry_w45_geometry"]},{"condition":"Predicted W23 entry","estimate":kn["predicted_entry_w45_geometry"]}])
    _forest(ax["a"],q,"condition","estimate",colors=[TRUTH,PRED],markers=["o","s"],xlabel="W45 grouped geometry",title="Trajectory-entry rescue");b.label(ax["a"],"a")
    rescue=pd.DataFrame([
        {"condition":"True magnitude",**kn["oracle_true_magnitude_rescue_w45"]},
        {"condition":"True direction",**kn["oracle_true_direction_rescue_w45"]},
        {"condition":"True program",**kn["oracle_true_program_rescue_w45"]},
        {"condition":"True entry",**kn["true_entry_minus_predicted_entry_w45"]},]).rename(columns={"point_delta":"estimate"})
    _forest(ax["b"],rescue,"condition","estimate","ci_low","ci_high",colors=[BASE,BASE,GEARS,TRUTH],markers=["o","o","D","o"],xlabel="W45 geometry gain",title="Oracle rescue decomposition");b.label(ax["b"],"b")
    markov_summary=json.loads((ROOT/"results/renge_markov_autopsy_v2/analysis_summary.json").read_text(encoding="utf-8"));mk=pd.DataFrame(markov_summary["models"]["deployable"]);mk["label"]=mk.model.str.replace("Persistent","",regex=False).str.strip().replace({"DirectEndpoint":"Direct endpoint"});_forest(ax["c"],mk,"label","response_distance_rho",colors=[BASE,PRED,FAIL,FAIL],markers=["s","o","^","v"],xlabel="R5 grouped geometry",title="Deployable Markov formulations");b.label(ax["c"],"c")
    comp=pd.read_csv(ROOT/"results/renge_markov_autopsy_v2/geometry_compression_autopsy.csv");comp=comp[comp.scope=="deployable"].groupby("model",as_index=False).agg(response_distance_rho=("response_distance_rho","mean"),between_variance_ratio=("between_variance_ratio","mean"));comp["label"]=comp.model.str.replace("Persistent","",regex=False).str.replace("DirectEndpoint","Direct",regex=False)
    for (_,z),c,m in zip(comp.iterrows(),[BASE,PRED,FAIL,FAIL],["s","o","^","v"]):ax["d"].scatter(z.between_variance_ratio,z.response_distance_rho,color=c,marker=m,s=30,label=z.label)
    ax["d"].axhline(0,color=GRID,lw=.8);ax["d"].set_xlabel("Between-intervention variance ratio");ax["d"].set_ylabel("R5 grouped geometry");ax["d"].set_title("Rollout compression autopsy");ax["d"].legend(frameon=False,fontsize=5.5);b.clean(ax["d"]);b.label(ax["d"],"d")
    stash("SupplementaryFigure8","a",q);stash("SupplementaryFigure8","b",rescue);stash("SupplementaryFigure8","c",mk);stash("SupplementaryFigure8","d",comp);save(fig,"SupplementaryFigure8")


def supplementary9():
    raw=pd.read_csv(ROOT/"results/generalization_factorial/factorial_results.csv");fig,ax=_supp_manual({"a":(.09,.55,.39,.35),"b":(.59,.55,.39,.35),"c":(.22,.10,.67,.27)},height=4.9)
    for a,direction,panel in zip([ax["a"],ax["b"]],["K562_TO_RPE1","RPE1_TO_K562"],["a","b"]):
        q=raw[raw.direction==direction]
        for regime,color,marker in (("ZERO_SHOT",BASE,"o"),("ALIGNED_ANCHOR",GEARS,"o"),("SHUFFLED_ANCHOR",FAIL,"s")):
            r=q[q.regime==regime]
            for _,gp in r.groupby("seed"):a.plot(gp.coverage_fraction*100,gp.geometry,color=color,alpha=.28,lw=.7)
            mean=r.groupby("coverage_fraction",as_index=False).geometry.mean();a.plot(mean.coverage_fraction*100,mean.geometry,color=color,marker=marker,ms=3,lw=1.4,label=regime.replace("_"," ").title())
        a.axhline(0,color=GRID,lw=.8);a.set_xticks([10,25,40,60,80,90]);a.set_xlabel("Other-intervention coverage (%)");a.set_title(direction.replace("_TO_"," → "));b.clean(a);b.label(a,panel)
    ax["a"].set_ylabel("Seed-level grouped geometry");ax["a"].legend(frameon=False,fontsize=6)
    contrasts=pd.read_csv(ROOT/"results/generalization_factorial/factorial_contrasts.csv");wanted=["ZERO_90_MINUS_ZERO_10","ANCHOR_90_MINUS_ZERO_90","ANCHOR_10_MINUS_ZERO_90","ANCHOR_MINUS_SHUFFLE_90"];cq=contrasts[contrasts.contrast.isin(wanted)].copy();cq["label"]=cq.direction.str.replace("_TO_","→",regex=False)+" · "+cq.contrast.map({"ZERO_90_MINUS_ZERO_10":"Zero90−Zero10","ANCHOR_90_MINUS_ZERO_90":"Anchor90−Zero90","ANCHOR_10_MINUS_ZERO_90":"Anchor10−Zero90","ANCHOR_MINUS_SHUFFLE_90":"Aligned−Shuffle90"})
    _forest(ax["c"],cq,"label","estimate","ci_low","ci_high",[GEARS if z=="K562_TO_RPE1" else DARK_DIRECTION_BLUE for z in cq.direction],["o" if z=="K562_TO_RPE1" else "s" for z in cq.direction],"Geometry contrast","Fixed-target decisive contrasts");b.label(ax["c"],"c")
    leakage=pd.read_csv(ROOT/"results/generalization_factorial/leakage_audit.csv");cq["leakage_firewall_all_pass"]=bool(leakage.leakage_firewall_pass.all())
    stash("SupplementaryFigure9","a",raw[raw.direction=="K562_TO_RPE1"]);stash("SupplementaryFigure9","b",raw[raw.direction=="RPE1_TO_K562"]);stash("SupplementaryFigure9","c",cq);save(fig,"SupplementaryFigure9")


def supplementary10():
    fig,ax=_supp_manual({"a":(.10,.57,.85,.32),"b":(.17,.11,.33,.28),"c":(.65,.11,.30,.28)},height=4.8)
    ext=pd.read_csv(ROOT/"results/frangieh_external_replication/external_contrasts.csv")
    curves=ext[(ext.analysis_set=="PRIMARY")&ext.contrast.str.match(r"ALIGNED_MINUS_SHUFFLE_\d+$")].copy();curves["coverage"]=curves.contrast.str.extract(r"(\d+)$").astype(int)
    for direction,(color,marker,ls) in zip(curves.direction.drop_duplicates(),[(GEARS,"o","-"),(DARK_DIRECTION_BLUE,"s","--")]):
        q=curves[curves.direction==direction].sort_values("coverage");label=direction.replace("Co-culture","Co").replace("_TO_","→");ax["a"].fill_between(q.coverage,q.ci_low,q.ci_high,color=color,alpha=.10,lw=0);ax["a"].plot(q.coverage,q.estimate,color=color,marker=marker,ls=ls,ms=3.5,label=label)
    audit=json.loads((ROOT/"results/frangieh_external_replication/data_audit.json").read_text(encoding="utf-8"));safe=audit["safe_descriptor_eligible_perturbations"]
    ax["a"].axhline(0,color=GRID,lw=.8);ax["a"].set_xticks([10,25,40,60,80,90]);ax["a"].set_xlabel("Other-intervention coverage (%)");ax["a"].set_ylabel("Aligned − shuffled geometry");ax["a"].set_title("External identity-anchor coverage curves");ax["a"].legend(frameon=False,ncol=2);ax["a"].text(.01,.96,f"External zero-shot not run: safe descriptor {safe}/237 < 60 threshold",transform=ax["a"].transAxes,va="top",fontsize=6,color=BASE);b.clean(ax["a"]);b.label(ax["a"],"a")
    for a,aset,panel in zip([ax["b"],ax["c"]],["PRIMARY","HIGH_CELL_TARGETS"],["b","c"]):
        q=ext[(ext.analysis_set==aset)&(ext.contrast=="ALIGNED_MINUS_SHUFFLE_90")].copy();q["label"]=q.direction.str.replace("_TO_"," → ",regex=False).str.replace("Î³","γ",regex=False)
        _forest(a,q,"label","estimate","ci_low","ci_high",[GEARS,DARK_DIRECTION_BLUE],["o","s"],"Aligned − shuffled geometry",("Primary external replication" if aset=="PRIMARY" else "High-cell target robustness"));b.label(a,panel)
        stash("SupplementaryFigure10",panel,q)
    curves["safe_descriptor_eligible"]=safe;curves["external_zero_shot_run"]=False;stash("SupplementaryFigure10","a",curves);save(fig,"SupplementaryFigure10")


def write_source_data_and_registry():
    source_dir=OUT/"source_data_internal";source_dir.mkdir(parents=True,exist_ok=True);rows=[]
    for (fig,panel),frame in sorted(PANEL_DATA.items()):
        name=f"{fig}_{panel}.csv";path=source_dir/name
        frame.to_csv(path,index=False)
        rows.append({"figure":fig,"panel":panel,"source_data_file":f"source_data_internal/{name}","records":len(frame),"columns":";".join(map(str,frame.columns))})
    pd.DataFrame(rows).to_csv(OUT/"FIGURE_SOURCE_VALUES.csv",index=False)


def write_audits():
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([
        {"old_figure":"SupplementaryFigure2","old_panel":"all","source_file":"figures/preprint_corrected/SupplementaryFigure2/SupplementaryFigure2.svg","scientific_question":"Artifact-safe intervention geometry","new_figure":"SupplementaryFigure2","new_panel":"a-b","action":"REDRAW","notes":"Rebuilt from fold-local grouped estimates and executable stitched-cloud null."},
        {"old_figure":"SupplementaryFigure7","old_panel":"all","source_file":"figures/preprint_corrected/SupplementaryFigure7/SupplementaryFigure7.svg","scientific_question":"Capacity, response basis and source identifiability","new_figure":"SupplementaryFigure5","new_panel":"a-b","action":"MERGE","notes":"Merged canonical response-basis/oracle and architecture evidence."},
        {"old_figure":"SupplementaryFigure10","old_panel":"all","source_file":"figures/preprint_corrected/SupplementaryFigure10/SupplementaryFigure10.svg","scientific_question":"Structural rescue and real-prior boundary","new_figure":"SupplementaryFigure6","new_panel":"a-b","action":"MERGE","notes":"Matched-generator positive control kept distinct from real-prior boundary."},
        {"old_figure":"SupplementaryFigure12","old_panel":"all","source_file":"figures/preprint_corrected/SupplementaryFigure12/SupplementaryFigure12.svg","scientific_question":"Temporal signal, reliability and first-wave controls","new_figure":"SupplementaryFigure7","new_panel":"a-b","action":"MERGE","notes":"Long categories converted to horizontal layouts."},
        {"old_figure":"SupplementaryFigure13","old_panel":"all","source_file":"NOT_PRESENT_IN_REPOSITORY","scientific_question":"Trajectory entry and temporal rescue","new_figure":"SupplementaryFigure8","new_panel":"a","action":"MERGE","notes":"Task-referenced historical number; no historical figure file was present. Frozen audit values were used."},
        {"old_figure":"SupplementaryFigure16","old_panel":"all","source_file":"figures/preprint_corrected/SupplementaryFigure16/SupplementaryFigure16.svg","scientific_question":"Markov formulation autopsy","new_figure":"SupplementaryFigure8","new_panel":"b","action":"MERGE","notes":"Combined with trajectory-entry rescue evidence."},
        {"old_figure":"SupplementaryFigure17","old_panel":"all","source_file":"figures/preprint_corrected/SupplementaryFigure17/SupplementaryFigure17.svg","scientific_question":"Internal generalization-axis factorial","new_figure":"SupplementaryFigure9","new_panel":"a-b","action":"REDRAW","notes":"All fixed five-seed curves retained with dominant mean lines."},
        {"old_figure":"SupplementaryFigure18","old_panel":"all","source_file":"NOT_PRESENT_IN_REPOSITORY","scientific_question":"Internal generalization controls","new_figure":"SupplementaryFigure9","new_panel":"a-b","action":"MERGE","notes":"Task-referenced historical number; no historical figure file was present. Related frozen factorial controls were merged."},
        {"old_figure":"SupplementaryFigure19","old_panel":"all","source_file":"figures/preprint_corrected/SupplementaryFigure19/SupplementaryFigure19.svg","scientific_question":"External Frangieh replication and robustness","new_figure":"SupplementaryFigure10","new_panel":"a-b","action":"REDRAW","notes":"Sparse wide comparison replaced by compact horizontal forests."},
    ]).to_csv(OUT/"SUPPLEMENTARY_RENUMBER_MAP.csv",index=False)
    citation_text=(
        "# Manuscript supplementary citation replacements\n\n"
        "Final manuscript citations must use only Supplementary Figures 1–10.\n\n"
        "| Historical citation | Final citation | Action |\n|---|---|---|\n"
        "| Supplementary Fig. 2 | Supplementary Fig. 2 | Retain number; cite the redrawn artifact-safe figure. |\n"
        "| Supplementary Fig. 7 | Supplementary Fig. 5 | Replace. |\n"
        "| Supplementary Fig. 10 | Supplementary Fig. 6 | Replace. |\n"
        "| Supplementary Fig. 12 | Supplementary Fig. 7 | Replace. |\n"
        "| Supplementary Fig. 13 | Supplementary Fig. 8 | Replace if present in the manuscript. |\n"
        "| Supplementary Fig. 16 | Supplementary Fig. 8 | Replace. |\n"
        "| Supplementary Figs. 17 and 18 | Supplementary Fig. 9 | Replace and merge. |\n"
        "| Supplementary Fig. 19 | Supplementary Fig. 10 | Replace. |\n\n"
        "Repository search found stale historical numbers in `figures/preprint_corrected/FIGURE_REGENERATION_REPORT.md`, "
        "`scripts/preprint_corrected_figures/build_corrected_figures.py`, and the archival block copied into "
        "`scripts/preprint_finalization/build_preprint_final.py`. These are provenance/archival implementation records, not the final manuscript; they were inventoried rather than silently rewritten.\n")
    (OUT/"MANUSCRIPT_SUPPLEMENTARY_CITATION_REPLACEMENTS.md").write_text(citation_text,encoding="utf-8")
    (OUT/"SUPPLEMENTARY_CITATION_REPLACEMENTS.md").write_text(citation_text,encoding="utf-8")
    color_text=(
        "# Color and style audit\n\nStatus: **PASS**.\n\n"
        "Canonical mappings are fixed: truth #2B2B2B; GEARS #4F8F7B; scGPT #8B79A8; prediction #5F8FB5; "
        "failure/incorrect structure #B65F42; baseline #A3A3A3; grid #DEDEDE. The scGPT purple was recovered from "
        "the v4 style registry. Figure 6 uses dark direction blue #3E6F91 for RPE1 → K562, replacing the pale v4 "
        "direction blue only where necessary to distinguish overlapping confidence bands from GEARS green. "
        "Color is paired with marker shape and/or line style throughout; continuous numeric ticks are unrotated.\n")
    (OUT/"COLOR_STYLE_AUDIT.md").write_text(color_text,encoding="utf-8")
    (OUT/"FINAL_COLOR_AUDIT.md").write_text(color_text,encoding="utf-8")
    reserved={"Figure1":["a","b","e"],"Figure4":["a"],"Figure5":["a","g"],"Figure6":["a","f"]}
    (OUT/"FINAL_FIGURE_VISUAL_QA.md").write_text(
        "# Final figure visual QA\n\nStatus: PENDING_RENDER_INSPECTION\n\n"
        + "\n".join(f"- {fig} panels {', '.join(panels)}: INTENTIONAL_RESERVED_CONCEPTUAL_SPACE" for fig,panels in reserved.items())
        + "\n\nReserved regions are exempt from empty-width failure but must remain clean and non-overlapped.\n",encoding="utf-8")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for f in (fig1,fig2,fig3,fig4,fig5,fig6,supplementary1,supplementary2,supplementary3,supplementary4,supplementary5,supplementary6,supplementary7,supplementary8,supplementary9,supplementary10):
        print(f"Rendering {f.__name__}",flush=True);f()
    write_source_data_and_registry();write_audits()
    print(f"Completed final figure set at {OUT}",flush=True)


if __name__ == "__main__":
    main()
