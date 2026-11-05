"""Render Supplementary Figure 3 from frozen case measurements only.

No fitting, inference, resampling or interval calculation is performed. The only
numeric transformation is a stored technical-half fraction times its denominator.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from settings import INPUT, OUTPUT as OUT, CHECKS as LAYOUT
SOURCE=INPUT/'fig12'
from layout import record_panel_positions

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 6.2, "axes.titlesize": 6.5, "axes.labelsize": 6.2,
    "xtick.labelsize": 6, "ytick.labelsize": 6.2,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .55, "xtick.major.width": .55,
    "ytick.major.width": .55, "xtick.major.size": 2.5,
    "ytick.major.size": 0, "legend.frameon": False,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})

OBSERVED = "#7B9CAF"
PREDICTED = "#C99A70"
MODEL_COLORS = {"PerturbDiff": "#777777", "GEARS": "#BDBDBD", "State": "#A0A0A0"}
TEXT = "#222222"
NEUTRAL = "#222222"

paths = {
    "gene_cases": SOURCE / "main_B_gene_cases_full.csv",
    "gene_vertices": SOURCE / "main_B_gene_vertices.csv",
    "state_vertices": SOURCE / "main_D_state_vertices.csv",
    "state_contexts": SOURCE / "supp3_state_all_five_contexts.csv",
    "state_doses": SOURCE / "supp3_state_all_three_doses.csv",
}
gene = pd.read_csv(paths["gene_cases"])
vertices = pd.read_csv(paths["gene_vertices"])
state_vertices = pd.read_csv(paths["state_vertices"])

def select_state(path):
    frame = pd.read_csv(path)
    out = frame.loc[
        frame.A_drug.eq("Tucidinostat") & frame.B_drug.eq("Temsirolimus")
        & frame.C_drug.eq("Panobinostat")
    ].copy()
    assert (out.A_dose == out.B_dose).all() and (out.B_dose == out.C_dose).all()
    return out

contexts = select_state(paths["state_contexts"])
doses = select_state(paths["state_doses"])
context_order = ["C32", "HOP62", "HepG2-C3A", "Hs 766T", "PANC-1"]
assert len(contexts) == 5 and set(contexts.context) == set(context_order)
assert contexts.A_dose.eq(.5).all()
contexts = contexts.set_index("context").loc[context_order].reset_index()
assert len(doses) == 3 and doses.context.eq("HOP62").all()
assert set(doses.A_dose) == {.05, .5, 5.}
doses = doses.sort_values("A_dose").reset_index(drop=True)
assert len(vertices) == 6 and vertices.all_four_pass.all()
assert len(state_vertices) == 3 and state_vertices.favorable_query.all()
assert gene.model.tolist() == ["PerturbDiff", "GEARS"]
assert gene.all_representation_checks.all()

rep_specs = [
    ("Original residual", "m_true", "m_pred"),
    ("Control-relative", "control_relative_m_true", "control_relative_m_pred"),
    ("Stress removed", "stress_removed_m_true", "stress_removed_m_pred"),
    ("Training-mean\ndirection removed", "training_mean_direction_removed_m_true", "training_mean_direction_removed_m_pred"),
]
margin_rows = []
for row in gene.to_dict("records"):
    for representation, true_field, pred_field in rep_specs:
        for kind, field in [("Observed", true_field), ("Predicted", pred_field)]:
            margin_rows.append({"model": row["model"], "fold": row["fold"],
                "A": row["A"], "B": row["B"], "C": row["C"],
                "representation": representation.replace("\n", " "),
                "kind": kind, "distance_margin": row[field], "source_field": field})
margins = pd.DataFrame(margin_rows)

reliability = gene[["model", "fold", "A", "B", "C", "half_positive_fraction",
                    "stress_removed_half_positive_fraction"]].copy()
reliability["technical_partitions"] = 20
reliability["technical_halves"] = 40
reliability["original_positive_halves"] = (reliability.half_positive_fraction * 40).astype(int)
reliability["stress_removed_positive_halves"] = (reliability.stress_removed_half_positive_fraction * 40).astype(int)
assert reliability.original_positive_halves.eq(40).all()
assert reliability.stress_removed_positive_halves.eq(40).all()

fig = plt.figure(figsize=(7.204724409448819, 7.283464566929134), dpi=300)
axes = []
ids = []

def new_axis(rect, axis_id):
    ax = fig.add_axes(rect)
    ax.set_gid(axis_id)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(NEUTRAL)
    axes.append(ax)
    ids.append(axis_id)
    return ax

def heading(letter, title, x, y):
    fig.text(x, y, letter, fontsize=8, fontweight="bold", color=TEXT, va="top")
    centers={'a':.55,'b':.37,'c':.81,'d':.55,'e':.37,'f':.81}
    fig.text(centers[letter], y, title, fontsize=7, fontweight="bold", color=TEXT, va="top", ha="center")

def quality_row(frame, fields, labels, colors, y0, height, prefix, titles):
    for j, (field, cutoff, xmax, ticks) in enumerate(fields):
        ax = new_axis([.195 + .196 * j, y0, .163, height], f"{prefix}{j + 1}")
        y = np.arange(len(frame))
        # Bar lengths are unmodified values from the frozen vertex tables.
        ax.barh(y, frame[field].to_numpy(), color=colors, height=.62, linewidth=0, zorder=3)
        # Endpoint ticks distinguish exact zero-length bars from missing values.
        zero = frame[field].to_numpy() == 0
        if zero.any():
            ax.scatter(frame[field].to_numpy()[zero], y[zero], marker="|", s=20,
                       c=np.array(colors)[zero], linewidths=1, clip_on=False, zorder=5)
        ax.axvline(cutoff, color="#687580", lw=.65, ls=(0, (3, 2)), zorder=4)
        ax.set_xlim(0, xmax)
        ax.set_xticks(ticks)
        ax.set_ylim(len(frame) - .5, -.5)
        ax.set_yticks(y, labels if j == 0 else [""] * len(frame))
        ax.set_title(titles[j], pad=8, color=TEXT, loc="center")
        ax.grid(False)
        if len(frame) == 6:
            ax.axhline(2.5, color="#CED5DB", lw=.55, zorder=2)

heading("a", "Gene-triplet quality", .028, .978)
fig.legend(handles=[Patch(facecolor=MODEL_COLORS["PerturbDiff"], label="PerturbDiff"),
                    Patch(facecolor=MODEL_COLORS["GEARS"], label="GEARS"),
                    Line2D([0], [0], color="#687580", lw=.65, ls=(0, (3, 2)), label="Eligibility threshold")],
           loc="upper center", bbox_to_anchor=(.57, .956), ncol=3, fontsize=6.2,
           handlelength=1.8, columnspacing=2.2)
gene_fields = [("CA", .75, 1.05, [0, .5, 1]),
               ("PB_original_expression_rank", .5, .55, [0, .25, .5]),
               ("residual_Pearson", .3, 1.05, [0, .5, 1]),
               ("MSE_fractional_improvement", 0, 1.05, [0, .5, 1])]
quality_row(vertices, gene_fields,
            [r.gene for r in vertices.itertuples()],
            [MODEL_COLORS[m] for m in vertices.model], .758, .143, "a",
            ["CA ↑", "PB rank ↓", "Residual r ↑", "MSE improvement ↑"])

fig.legend(handles=[Line2D([0], [0], marker="o", linestyle="none", color=OBSERVED, markerfacecolor="#EAF3F7", markersize=4, label="Observed"),
                    Line2D([0], [0], marker="s", linestyle="none", color=PREDICTED, markerfacecolor="#FAF0E5", markersize=3.8, label="Predicted")],
           loc="upper center", bbox_to_anchor=(.55, .72), ncol=2, fontsize=6.2,
           handlelength=1, columnspacing=2.1)

def margin_plot(ax, labels, observed, predicted, highlight=None):
    y = np.arange(len(labels))
    if highlight is not None:
        ax.axhspan(highlight - .45, highlight + .45, facecolor="#F0F3F5", zorder=0)
    ax.axvline(0, color="#6B7780", lw=.7, ls=(0, (3, 2)), zorder=1)
    ax.hlines(y, observed, predicted, color="#CED5DB", lw=.75, zorder=2)
    ax.scatter(observed, y, facecolors="#EAF3F7", marker="o", s=16, edgecolors=OBSERVED, linewidths=.85, zorder=4)
    ax.scatter(predicted, y, facecolors="#FAF0E5", marker="s", s=15, edgecolors=PREDICTED, linewidths=.85, zorder=4)
    ax.set_xlim(-.65, .45)
    ax.set_xticks([-.6, -.3, 0, .3])
    ax.set_ylim(len(labels) - .55, -.55)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Distance margin", labelpad=4)

for i, model in enumerate(["PerturbDiff", "GEARS"]):
    x_heading = .028 if i == 0 else .518
    heading("bc"[i], model, x_heading, .682)
    ax = new_axis([.224 + .44 * i, .506, .292, .124], "bc"[i])
    row = gene.loc[gene.model.eq(model)].iloc[0]
    margin_plot(ax, [item[0] for item in rep_specs],
                [row[item[1]] for item in rep_specs], [row[item[2]] for item in rep_specs])

heading("d", "Official State: HOP62, 0.5 μM", .028, .44)
state_fields = [("systema_centroid_accuracy", .75, 1.05, [0, .5, 1]),
                ("perturbench_original_rank", .5, .55, [0, .25, .5]),
                ("response_Pearson", .3, 1.05, [0, .5, 1]),
                ("mse_fractional_improvement", 0, 1.05, [0, .5, 1])]
quality_row(state_vertices, state_fields,
            [r.drug for r in state_vertices.itertuples()],
            [MODEL_COLORS["State"]] * 3, .289, .083, "d",
            ["CA ↑", "PB rank ↓", "Response r ↑", "MSE improvement ↑"])

heading("e", "State across contexts", .028, .224)
heading("f", "State across doses", .518, .224)
ax = new_axis([.224, .059, .292, .117], "e")
margin_plot(ax, contexts.context.tolist(), contexts.m_true, contexts.m_pred, highlight=1)
ax = new_axis([.664, .059, .292, .117], "f")
margin_plot(ax, [f"{dose:g} μM" for dose in doses.A_dose], doses.m_true, doses.m_pred, highlight=1)

fig.canvas.draw()
alignment = record_panel_positions(
    fig, axes=axes, panel_ids=ids,
    row_groups=[["a1", "a2", "a3", "a4"], ["b", "c"], ["d1", "d2", "d3", "d4"], ["e", "f"]],
    column_groups=[["a1", "d1"], ["a2", "d2"], ["a3", "d3"], ["a4", "d4"], ["b", "e"], ["c", "f"]],
    json_out=LAYOUT / "alignment.json", overlay_svg=LAYOUT / "alignment.svg",
    tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
)
fig.savefig(OUT / "Supplementary_Figure_3.pdf")
fig.savefig(OUT / "Supplementary_Figure_3.svg")
fig.savefig(OUT / "Supplementary_Figure_3.png", dpi=300)
