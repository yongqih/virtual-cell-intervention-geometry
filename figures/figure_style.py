"""Minimal canonical style and layout helpers for the final preprint figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "preprint_final"

# Frozen from figures_nature_comm_v4/style/nature_comm_style.json before the
# archival v4 tree was removed from the working release.
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

TRUE = C["truth"]
GRID = C["grid"]

LAYOUTS = {
    "Figure4": {
        "figsize": (7.1, 5.7),
        "panels": {
            "a": {"x": .04, "y": .43, "width": .34, "height": .49, "priority": 2, "type": "conceptual_reserved"},
            "b": {"x": .49, "y": .68, "width": .45, "height": .23, "priority": 1, "type": "quantitative"},
            "c": {"x": .49, "y": .45, "width": .45, "height": .13, "priority": 2, "type": "quantitative"},
            "d": {"x": .10, "y": .08, "width": .27, "height": .17, "priority": 3, "type": "quantitative"},
            "e": {"x": .57, "y": .08, "width": .37, "height": .17, "priority": 2, "type": "quantitative"},
        },
    },
    "Figure6": {
        "figsize": (7.1, 6.2),
        "panels": {
            "a": {"x": .04, "y": .63, "width": .16, "height": .29, "priority": 3, "type": "conceptual_reserved"},
            "b": {"x": .27, "y": .63, "width": .30, "height": .29, "priority": 1, "type": "quantitative"},
            "c": {"x": .65, "y": .63, "width": .31, "height": .29, "priority": 1, "type": "quantitative"},
            "d": {"x": .14, "y": .26, "width": .45, "height": .25, "priority": 1, "type": "quantitative"},
            "e": {"x": .64, "y": .26, "width": .32, "height": .25, "priority": 2, "type": "quantitative"},
            "f": {"x": .08, "y": .05, "width": .88, "height": .10, "priority": 3, "type": "conceptual_reserved"},
        },
    },
}

mpl.rcParams.update({
    "axes.prop_cycle": mpl.cycler(color=[C["prediction"], C["gears"], C["failure"], C["baseline"]]),
    "font.family": "Arial",
    "font.size": 7.0,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.15,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def label(ax, letter: str, x: float = -0.17, y: float = 1.10) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9.0,
            fontweight="bold", ha="left", va="top", clip_on=False)


def clean(ax, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.75, length=3)
    if grid:
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)


def rotate_categories(ax) -> None:
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")


def make_main_figure(name: str):
    spec = LAYOUTS[name]
    fig = plt.figure(figsize=spec["figsize"])
    axes = {}
    for panel, pos in spec["panels"].items():
        if pos["type"] == "quantitative":
            axes[panel] = fig.add_axes([pos["x"], pos["y"], pos["width"], pos["height"]])
    return fig, axes


def write_layout_artifacts(name: str) -> None:
    spec = LAYOUTS[name]
    directory = OUT / name
    payload = {
        "figure": name,
        "coordinate_system": "normalized figure coordinates; origin at bottom-left",
        "figsize_inches": list(spec["figsize"]),
        "panels": [dict(panel=panel, **values) for panel, values in spec["panels"].items()],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    (directory / f"{name}_layout_spec.json").write_text(text, encoding="utf-8")
    (directory / "figure_layout_spec.json").write_text(text, encoding="utf-8")


def save(fig, name: str):
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    svg = directory / f"{name}.svg"
    png = directory / f"{name}.png"
    save_kwargs = {} if name in LAYOUTS else {"bbox_inches": "tight"}
    fig.savefig(svg, **save_kwargs)
    fig.savefig(png, dpi=600, **save_kwargs)
    plt.close(fig)
    if name in LAYOUTS:
        write_layout_artifacts(name)
    return svg, png
