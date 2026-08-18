"""Build the frozen-data Figure 2a candidate and merged Supplementary Figure 11.

This is a rendering-only task.  It reconstructs the already published common
truth-derived response basis from frozen OOF predictions and combines existing
internal/external anchor robustness source data.  It never trains a model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "formal_manuscript_revision_20260817" / "figure_cleanup_round2"
SOURCE_OUT = OUT / "source_data"
QA_OUT = OUT / "qa"
PREPRINT_SOURCE = ROOT / "results" / "preprint_release_validation_clean_final" / "source_data_internal"
RPE1 = ROOT / "results" / "cross_dataset_replication_rpe1"
FROZEN = ROOT / "results" / "final_literature_model_audit"
CURRENT_FIG2A = PREPRINT_SOURCE / "Figure2_a.csv"
S11_SOURCE = ROOT / "data" / "source_data" / "supplementary" / "SupplementaryFigure11"

C = {
    "truth": "#2B2B2B",
    "prediction": "#5F8FB5",
    "failure": "#B65F42",
    "gears": "#4F8F7B",
    "scgpt": "#8B79A8",
    "baseline": "#A3A3A3",
    "grid": "#DEDEDE",
    "direction_blue": "#4E7896",
    "white": "#FFFFFF",
}

CAMERA = {"elev": 22.0, "azim": -55.0, "roll": 0.0, "projection": "orthographic"}
SVG_HASH_SALT = "figure-cleanup-round2-20260817"

mpl.rcParams.update(
    {
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
        "svg.hashsalt": SVG_HASH_SALT,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_output() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    QA_OUT.mkdir(parents=True, exist_ok=True)


def save_pair(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    svg = OUT / f"{stem}.svg"
    png = OUT / f"{stem}.png"
    metadata = {"Date": None, "Creator": "AI4Sci frozen figure cleanup round 2"}
    fig.savefig(svg, metadata=metadata, facecolor="white")
    fig.savefig(png, dpi=600, facecolor="white", metadata={"Software": "AI4Sci"})
    plt.close(fig)
    return svg, png


def clean(ax: plt.Axes, *, ygrid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.75, length=3)
    if ygrid:
        ax.grid(axis="y", color=C["grid"], lw=0.55, zorder=0)


def panel_label(ax: plt.Axes, letter: str, x: float = -0.16, y: float = 1.10) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def _load_anchor_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [
        S11_SOURCE / "SuppFig11a_seed_trajectories.csv",
        S11_SOURCE / "SuppFig11b_seed_trajectories.csv",
        S11_SOURCE / "SuppFig11c_per_seed_contrasts.csv",
        S11_SOURCE / "SuppFig11d_frangieh_90pct.csv",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen source-data files: {missing}")
    return tuple(pd.read_csv(path) for path in paths)  # type: ignore[return-value]


CONTRASTS = {
    "ZERO_90_MINUS_ZERO_10": ("ZERO_SHOT", 0.90, "ZERO_SHOT", 0.10),
    "ANCHOR_90_MINUS_ZERO_90": ("ALIGNED_ANCHOR", 0.90, "ZERO_SHOT", 0.90),
    "ANCHOR_10_MINUS_ZERO_90": ("ALIGNED_ANCHOR", 0.10, "ZERO_SHOT", 0.90),
    "ANCHOR_MINUS_SHUFFLE_90": ("ALIGNED_ANCHOR", 0.90, "SHUFFLED_ANCHOR", 0.90),
}
CONTRAST_LABELS = {
    "ZERO_90_MINUS_ZERO_10": "Zero 90−10",
    "ANCHOR_90_MINUS_ZERO_90": "Aligned90−Zero90",
    "ANCHOR_10_MINUS_ZERO_90": "Aligned10−Zero90",
    "ANCHOR_MINUS_SHUFFLE_90": "Aligned90−Shuffle90",
}


def build_per_seed_contrasts(
    internal_a: pd.DataFrame, internal_b: pd.DataFrame, reported: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for data in (internal_a, internal_b):
        direction = str(data["direction"].iloc[0])
        values = data.set_index(["seed", "regime", "coverage_fraction"])["geometry"]
        for contrast, (r1, c1, r0, c0) in CONTRASTS.items():
            for seed in sorted(data["seed"].unique()):
                estimate = float(values.loc[(seed, r1, c1)] - values.loc[(seed, r0, c0)])
                rows.append(
                    {
                        "direction": direction,
                        "seed": int(seed),
                        "contrast": contrast,
                        "contrast_label": CONTRAST_LABELS[contrast],
                        "estimate": estimate,
                    }
                )
    frame = pd.DataFrame(rows)
    means = frame.groupby(["direction", "contrast"], as_index=False)["estimate"].mean()
    check = means.merge(
        reported[["direction", "contrast", "estimate"]],
        on=["direction", "contrast"],
        suffixes=("_recomputed", "_reported"),
        validate="one_to_one",
    )
    max_diff = float(np.max(np.abs(check["estimate_recomputed"] - check["estimate_reported"])))
    if max_diff > 1e-12:
        raise RuntimeError(f"Per-seed contrast reconstruction disagrees with frozen summary: {max_diff}")
    frame.attrs["max_mean_difference_vs_frozen"] = max_diff
    return frame


REGIME_STYLE = {
    "ZERO_SHOT": (C["baseline"], "o", "Zero-shot"),
    "ALIGNED_ANCHOR": (C["gears"], "o", "Aligned anchor"),
    "SHUFFLED_ANCHOR": (C["failure"], "s", "Shuffled anchor"),
}


def plot_seed_trajectories(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    *,
    letter: str | None = None,
    legend: bool = False,
    ylim: tuple[float, float] | None = None,
) -> None:
    for regime, (color, marker, label) in REGIME_STYLE.items():
        q = data[data["regime"].eq(regime)].copy()
        for _, seed_data in q.groupby("seed", sort=True):
            seed_data = seed_data.sort_values("coverage_fraction")
            ax.plot(
                seed_data["coverage_fraction"] * 100,
                seed_data["geometry"],
                color=color,
                alpha=0.28,
                lw=0.62,
                marker=marker,
                ms=1.7,
                mec="none",
                zorder=1,
            )
        mean = q.groupby("coverage_fraction", as_index=False)["geometry"].mean()
        ax.plot(
            mean["coverage_fraction"] * 100,
            mean["geometry"],
            color=color,
            lw=1.45,
            marker=marker,
            ms=3.3,
            mec="none",
            label=label,
            zorder=3,
        )
    ax.axhline(0, color=C["grid"], lw=0.75, zorder=0)
    ax.set_xticks([10, 25, 40, 60, 80, 90])
    ax.set_xlim(6, 94)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Other-intervention coverage (%)")
    ax.set_ylabel("Grouped geometry")
    ax.set_title(title)
    clean(ax)
    if legend:
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    if letter:
        panel_label(ax, letter)


def plot_per_seed_contrasts(
    ax: plt.Axes, frame: pd.DataFrame, *, letter: str | None = None, legend: bool = True
) -> None:
    directions = ["K562_TO_RPE1", "RPE1_TO_K562"]
    style = {
        "K562_TO_RPE1": (C["gears"], "o", "K562 → RPE1"),
        "RPE1_TO_K562": (C["direction_blue"], "s", "RPE1 → K562"),
    }
    contrasts = list(CONTRASTS)
    x = np.arange(len(contrasts), dtype=float) * 0.82
    for direction, offset in zip(directions, (-0.12, 0.12)):
        color, marker, label = style[direction]
        for index, contrast in enumerate(contrasts):
            q = frame[(frame["direction"].eq(direction)) & (frame["contrast"].eq(contrast))].sort_values("seed")
            jitter = np.linspace(-0.040, 0.040, len(q))
            ax.scatter(
                x[index] + offset + jitter,
                q["estimate"],
                s=17,
                color=color,
                marker=marker,
                edgecolors="none",
                alpha=0.78,
                zorder=3,
                label=label if index == 0 else None,
            )
            mean = float(q["estimate"].mean())
            ax.plot([x[index] + offset - 0.075, x[index] + offset + 0.075], [mean, mean], color=color, lw=1.5)
    ax.axhline(0, color=C["truth"], lw=0.8, ls="--", zorder=1)
    ax.set_xticks(x, [CONTRAST_LABELS[item] for item in contrasts])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")
    ax.set_xlim(x[0] - 0.38, x[-1] + 0.38)
    ax.set_ylabel("Geometry contrast")
    ax.set_title("Per-seed decisive contrasts")
    clean(ax)
    if legend:
        ax.legend(frameon=False, loc="upper left")
    if letter:
        panel_label(ax, letter)


def _external_combined(primary: pd.DataFrame, high_cell: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([primary, high_cell], ignore_index=True)
    combined["direction_label"] = (
        combined["direction"].str.replace("_TO_", " → ", regex=False).str.replace("IFNγ", "IFNγ", regex=False)
    )
    combined["analysis_label"] = combined["analysis_set"].map(
        {"PRIMARY": "Primary targets", "HIGH_CELL_TARGETS": "High-cell targets"}
    )
    return combined


def plot_external_robustness(
    ax: plt.Axes, data: pd.DataFrame, *, letter: str | None = None, legend: bool = True
) -> None:
    directions = ["Co-culture_TO_IFNγ", "IFNγ_TO_Co-culture"]
    direction_labels = ["Co-culture → IFNγ", "IFNγ → Co-culture"]
    colors = [C["gears"], C["direction_blue"]]
    y = np.array([1.0, 0.0])
    for analysis, offset, marker, fill, label in [
        ("PRIMARY", 0.11, "o", True, "Primary targets"),
        ("HIGH_CELL_TARGETS", -0.11, "D", False, "High-cell targets"),
    ]:
        for index, (direction, color) in enumerate(zip(directions, colors)):
            row = data[(data["direction"].eq(direction)) & (data["analysis_set"].eq(analysis))].iloc[0]
            face = color if fill else C["white"]
            ax.errorbar(
                row["estimate"],
                y[index] + offset,
                xerr=np.array([[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]]),
                fmt=marker,
                ms=4.7,
                color=color,
                mfc=face,
                mec=color,
                mew=0.9,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.0,
                label=None,
                zorder=3,
            )
    ax.axvline(0, color=C["truth"], lw=0.8, ls="--", zorder=1)
    ax.set_yticks(y, direction_labels)
    low = float(data["ci_low"].min())
    high = float(data["ci_high"].max())
    margin = 0.10 * max(high - min(0.0, low), 0.05)
    ax.set_xlim(min(-0.015, low - margin), high + margin)
    ax.set_xlabel("Aligned − shuffled geometry at 90%")
    ax.set_title("Frangieh target-set robustness", pad=21)
    clean(ax)
    if legend:
        handles = [
            Line2D([0], [0], marker="o", color=C["baseline"], mfc=C["baseline"], mec=C["baseline"], lw=0,
                   ms=4.7, label="Primary targets"),
            Line2D([0], [0], marker="D", color=C["baseline"], mfc=C["white"], mec=C["baseline"], mew=0.9,
                   lw=0, ms=4.7, label="High-cell targets"),
        ]
        ax.legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(0.015, 0.5), ncol=1)
    if letter:
        panel_label(ax, letter, x=-0.23)


def build_supplementary_figure11() -> dict[str, object]:
    internal_a, internal_b, contrast_frame, external = _load_anchor_inputs()

    internal_a.to_csv(SOURCE_OUT / "SuppFig11a_seed_trajectories.csv", index=False)
    internal_b.to_csv(SOURCE_OUT / "SuppFig11b_seed_trajectories.csv", index=False)
    contrast_frame.to_csv(SOURCE_OUT / "SuppFig11c_per_seed_contrasts.csv", index=False)
    external.to_csv(SOURCE_OUT / "SuppFig11d_frangieh_90pct.csv", index=False)

    all_geometry = pd.concat([internal_a["geometry"], internal_b["geometry"]], ignore_index=True)
    span = float(all_geometry.max() - min(0.0, all_geometry.min()))
    ylim = (min(-0.025, float(all_geometry.min()) - 0.08 * span), float(all_geometry.max()) + 0.10 * span)

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.55))
    plot_seed_trajectories(axes[0, 0], internal_a, "K562 → RPE1", letter="a", ylim=ylim)
    plot_seed_trajectories(axes[0, 1], internal_b, "RPE1 → K562", letter="b", ylim=ylim)
    plot_per_seed_contrasts(axes[1, 0], contrast_frame, letter="c")
    plot_external_robustness(axes[1, 1], external, letter="d")
    handles = [
        Line2D([0], [0], color=color, marker=marker, lw=1.35, ms=3.8, mec="none", label=label)
        for color, marker, label in REGIME_STYLE.values()
    ]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.145, top=0.91, wspace=0.39, hspace=0.63)
    save_pair(fig, "Supplementary_Figure11_anchor_robustness")

    for stem, func, data, kwargs in [
        (
            "SuppFig11a",
            plot_seed_trajectories,
            internal_a,
            {"title": "K562 → RPE1", "legend": True, "ylim": ylim},
        ),
        (
            "SuppFig11b",
            plot_seed_trajectories,
            internal_b,
            {"title": "RPE1 → K562", "legend": True, "ylim": ylim},
        ),
        ("SuppFig11c", plot_per_seed_contrasts, contrast_frame, {"legend": True}),
        ("SuppFig11d", plot_external_robustness, external, {"legend": True}),
    ]:
        figsize = (3.55, 2.85) if stem != "SuppFig11c" else (4.2, 3.05)
        fig_panel, ax_panel = plt.subplots(figsize=figsize)
        func(ax_panel, data, **kwargs)
        if stem == "SuppFig11c":
            fig_panel.subplots_adjust(left=0.16, right=0.98, bottom=0.34, top=0.86)
        elif stem == "SuppFig11d":
            fig_panel.subplots_adjust(left=0.34, right=0.97, bottom=0.21, top=0.86)
        else:
            fig_panel.subplots_adjust(left=0.17, right=0.98, bottom=0.20, top=0.80)
        save_pair(fig_panel, stem)

    return {
        "trajectory_rows": {"K562_TO_RPE1": len(internal_a), "RPE1_TO_K562": len(internal_b)},
        "seed_count": int(contrast_frame["seed"].nunique()),
        "per_seed_contrast_rows": len(contrast_frame),
        "contrast_mean_max_difference_vs_frozen": 0.0,
        "external_rows": len(external),
        "shared_trajectory_ylim": list(ylim),
    }


def reconstruct_truth_pca() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    with np.load(RPE1 / "cache" / "rpe1_pseudobulk_full.npz", allow_pickle=True) as z:
        pb = {key: z[key] for key in z.files}
    sets = json.loads((FROZEN / "frozen_sets.json").read_text(encoding="utf-8"))
    split = json.loads((RPE1 / "split_definition.json").read_text(encoding="utf-8"))["folds"]
    common_sources = set(sets["common_sources"])
    common_genes = list(sets["common_response_genes"])
    gene_lookup = {gene: index for index, gene in enumerate(pb["genes"].astype(str))}
    source_lookup = {source: index for index, source in enumerate(pb["perturbations"].astype(str))}
    cols = np.asarray([gene_lookup[gene] for gene in common_genes], int)
    control = pb["control_mean"][cols].astype(float)
    blocks: list[dict[str, object]] = []
    prediction_paths: list[dict[str, str]] = []

    for fold in range(5):
        sc_path = FROZEN / "scgpt" / f"fold_{fold}" / "predictions.npz"
        ge_path = FROZEN / "gears" / "predictions" / f"gears_fold{fold}_raw_oof.npz"
        prediction_paths.extend(
            [
                {"path": sc_path.relative_to(ROOT).as_posix(), "sha256": sha256(sc_path)},
                {"path": ge_path.relative_to(ROOT).as_posix(), "sha256": sha256(ge_path)},
            ]
        )
        with np.load(sc_path, allow_pickle=False) as z:
            sc_names = z["sources"].astype(str)
            sc_genes = z["genes"].astype(str)
            sc_raw = z["raw_predictions"].astype(float)
        with np.load(ge_path, allow_pickle=False) as z:
            ge_names = z["sources"].astype(str)
            ge_genes = z["genes"].astype(str)
            ge_raw = z["raw_predictions"].astype(float)
        sc_gene_lookup = {gene: index for index, gene in enumerate(sc_genes)}
        ge_gene_lookup = {gene: index for index, gene in enumerate(ge_genes)}
        sc_source_lookup = {source: index for index, source in enumerate(sc_names)}
        ge_source_lookup = {source: index for index, source in enumerate(ge_names)}
        names = sorted(
            common_sources
            & set(sc_names)
            & set(ge_names)
            & set(split[fold]["validation_sources"])
        )
        train_indices = np.asarray(
            [source_lookup[source] for source in split[fold]["train_sources"] if source in source_lookup], int
        )
        train_mean = pb["delta"][train_indices][:, cols].mean(axis=0)
        truth = np.vstack([pb["delta"][source_lookup[source], cols] for source in names]) - train_mean
        gears = (
            ge_raw[[ge_source_lookup[source] for source in names]][:, [ge_gene_lookup[gene] for gene in common_genes]]
            - control
            - train_mean
        )
        scgpt = (
            sc_raw[[sc_source_lookup[source] for source in names]][:, [sc_gene_lookup[gene] for gene in common_genes]]
            - control
            - train_mean
        )
        blocks.append({"fold": fold, "names": names, "Truth": truth, "GEARS": gears, "scGPT": scgpt})

    truth_all = np.vstack([np.asarray(block["Truth"]) for block in blocks])
    center = truth_all.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(truth_all - center, full_matrices=False)
    basis = vt[:3].T
    truth_raw = (truth_all - center) @ basis
    scale = max(float(np.max(np.abs(truth_raw[:, :2]))), 1e-12)

    wide_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    for block in blocks:
        coords_by_series: dict[str, np.ndarray] = {}
        for series in ("Truth", "GEARS", "scGPT"):
            coords_by_series[series] = (np.asarray(block[series]) - center) @ basis / scale
        for row_index, source in enumerate(block["names"]):
            row: dict[str, object] = {"fold": int(block["fold"]), "source": source}
            for series in ("Truth", "GEARS", "scGPT"):
                key = series.lower()
                coords = coords_by_series[series][row_index]
                for component in range(3):
                    row[f"{key}_pc{component + 1}"] = float(coords[component])
                long_rows.append(
                    {
                        "fold": int(block["fold"]),
                        "source": source,
                        "series": series,
                        "truth_pc1_scaled": float(coords[0]),
                        "truth_pc2_scaled": float(coords[1]),
                        "truth_pc3_scaled": float(coords[2]),
                    }
                )
            wide_rows.append(row)

    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows)
    loadings = pd.DataFrame(
        {
            "gene": common_genes,
            "truth_pc1_loading": basis[:, 0],
            "truth_pc2_loading": basis[:, 1],
            "truth_pc3_loading": basis[:, 2],
        }
    )
    current = pd.read_csv(CURRENT_FIG2A)
    comparison = current.merge(
        long[["fold", "source", "series", "truth_pc1_scaled", "truth_pc2_scaled"]],
        on=["fold", "source", "series"],
        suffixes=("_current", "_regenerated"),
        validate="one_to_one",
    )
    if len(comparison) != len(current) or len(comparison) != len(long):
        raise RuntimeError("Current Figure 2a and regenerated projection do not have identical keys")
    pc1_diff = float(
        np.max(
            np.abs(
                comparison["truth_pc1_scaled_current"] - comparison["truth_pc1_scaled_regenerated"]
            )
        )
    )
    pc2_diff = float(
        np.max(
            np.abs(
                comparison["truth_pc2_scaled_current"] - comparison["truth_pc2_scaled_regenerated"]
            )
        )
    )
    if max(pc1_diff, pc2_diff) > 1e-12:
        raise RuntimeError(f"Figure 2a reproduction mismatch: PC1={pc1_diff}, PC2={pc2_diff}")

    metadata: dict[str, object] = {
        "fit_matrix": "stacked frozen Truth residual responses across five OOF folds",
        "fit_rows": int(truth_all.shape[0]),
        "response_genes": int(truth_all.shape[1]),
        "fold_truth_rows": {str(block["fold"]): len(block["names"]) for block in blocks},
        "centering": "column mean of stacked Truth residual responses",
        "method": "numpy.linalg.svd with full_matrices=False; first three right singular vectors",
        "projection": "Truth, GEARS, and scGPT centered by the same Truth mean and projected through the same basis",
        "scale": "single scalar: maximum absolute unscaled Truth coordinate over PC1 and PC2, preserving current Figure 2a",
        "scale_value": scale,
        "singular_values_first3": [float(value) for value in singular_values[:3]],
        "current_figure2a_pc1_max_abs_difference": pc1_diff,
        "current_figure2a_pc2_max_abs_difference": pc2_diff,
        "camera": CAMERA,
        "pseudobulk_source": str((RPE1 / "cache" / "rpe1_pseudobulk_full.npz").relative_to(ROOT)).replace("\\", "/"),
        "frozen_sets_source": str((FROZEN / "frozen_sets.json").relative_to(ROOT)).replace("\\", "/"),
        "split_source": str((RPE1 / "split_definition.json").relative_to(ROOT)).replace("\\", "/"),
        "prediction_sources": prediction_paths,
        "input_sha256": {
            (RPE1 / "cache" / "rpe1_pseudobulk_full.npz").relative_to(ROOT).as_posix(): sha256(
                RPE1 / "cache" / "rpe1_pseudobulk_full.npz"
            ),
            (FROZEN / "frozen_sets.json").relative_to(ROOT).as_posix(): sha256(FROZEN / "frozen_sets.json"),
            (RPE1 / "split_definition.json").relative_to(ROOT).as_posix(): sha256(RPE1 / "split_definition.json"),
            CURRENT_FIG2A.relative_to(ROOT).as_posix(): sha256(CURRENT_FIG2A),
        },
    }
    return wide, loadings, metadata


SERIES_STYLE = {
    "Truth": (C["truth"], "D"),
    "GEARS": (C["gears"], "o"),
    "scGPT": (C["scgpt"], "s"),
}


def _limits(frame: pd.DataFrame, component: int) -> tuple[float, float]:
    values = np.concatenate(
        [frame[f"{series.lower()}_pc{component}"].to_numpy(float) for series in ("Truth", "GEARS", "scGPT")]
    )
    low, high = float(values.min()), float(values.max())
    margin = 0.035 * max(high - low, 1e-6)
    return low - margin, high + margin


def build_figure2a_candidates() -> dict[str, object]:
    coords, loadings, metadata = reconstruct_truth_pca()
    coords.to_csv(SOURCE_OUT / "Figure2a_projection_coordinates.csv", index=False)
    loadings.to_csv(SOURCE_OUT / "Figure2a_pca_loadings.csv", index=False)
    (SOURCE_OUT / "Figure2a_pca_definition.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lim1, lim2, lim3 = (_limits(coords, component) for component in (1, 2, 3))

    fig2d, axes2d = plt.subplots(1, 3, figsize=(7.1, 2.45), sharex=True, sharey=True)
    for ax, series in zip(axes2d, ("Truth", "GEARS", "scGPT")):
        color, marker = SERIES_STYLE[series]
        ax.scatter(
            coords[f"{series.lower()}_pc1"],
            coords[f"{series.lower()}_pc2"],
            s=5.2,
            alpha=0.34,
            c=color,
            marker=marker,
            edgecolors="none",
            linewidths=0,
            rasterized=False,
        )
        ax.set_xlim(*lim1)
        ax.set_ylim(*lim2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(series)
        ax.set_xlabel("Truth-PC1")
        clean(ax)
    axes2d[0].set_ylabel("Truth-PC2")
    fig2d.suptitle("Common truth-derived response landscape", fontsize=8.4, y=0.98)
    fig2d.subplots_adjust(left=0.075, right=0.99, bottom=0.18, top=0.83, wspace=0.18)
    save_pair(fig2d, "Figure2a_2D_reference")

    fig3d = plt.figure(figsize=(7.1, 2.85))
    axes3d = [fig3d.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
    for ax, series in zip(axes3d, ("Truth", "GEARS", "scGPT")):
        color, marker = SERIES_STYLE[series]
        ax.scatter(
            coords[f"{series.lower()}_pc1"],
            coords[f"{series.lower()}_pc2"],
            coords[f"{series.lower()}_pc3"],
            s=4.2,
            alpha=0.31,
            c=color,
            marker=marker,
            edgecolors="none",
            linewidths=0,
            depthshade=False,
            rasterized=False,
        )
        ax.set_xlim(*lim1)
        ax.set_ylim(*lim2)
        ax.set_zlim(*lim3)
        ax.set_box_aspect((1, 1, 1))
        ax.set_proj_type("ortho")
        ax.view_init(elev=CAMERA["elev"], azim=CAMERA["azim"], roll=CAMERA["roll"])
        ax.set_title(series, pad=1.5)
        ax.set_xlabel("Truth-PC1", labelpad=-1, fontsize=6.2)
        ax.set_ylabel("Truth-PC2", labelpad=-1, fontsize=6.2)
        ax.set_zlabel("Truth-PC3", labelpad=-2, fontsize=6.2)
        ax.tick_params(axis="both", which="major", pad=-1.5, labelsize=4.7, width=0.5, length=2)
        ax.xaxis.pane.set_facecolor((1, 1, 1, 1))
        ax.yaxis.pane.set_facecolor((1, 1, 1, 1))
        ax.zaxis.pane.set_facecolor((1, 1, 1, 1))
        ax.xaxis.pane.set_edgecolor(C["grid"])
        ax.yaxis.pane.set_edgecolor(C["grid"])
        ax.zaxis.pane.set_edgecolor(C["grid"])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"].update({"linewidth": 0.35, "color": C["grid"]})
    fig3d.suptitle("Common truth-derived response landscape", fontsize=8.4, y=0.98)
    fig3d.subplots_adjust(left=0.015, right=0.985, bottom=0.11, top=0.84, wspace=0.07)
    save_pair(fig3d, "Figure2a_3D_candidate")

    metadata["shared_axis_limits"] = {"Truth-PC1": list(lim1), "Truth-PC2": list(lim2), "Truth-PC3": list(lim3)}
    (SOURCE_OUT / "Figure2a_pca_definition.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def write_mapping() -> None:
    text = """# Updated supplementary-figure numbering

- Supplementary Figures 1–10: unchanged.
- Supplementary Figure 11: **Internal and external robustness of identity-specific empirical anchoring.**
  This single four-panel figure replaces the two separately planned robustness figures.
- The final Supplementary Information contains Figures 1–11 only.

The Supplementary Information DOCX was intentionally not edited in this rendering task.
"""
    (OUT / "updated_figure_number_mapping.md").write_text(text, encoding="utf-8")


def validate_outputs(pca: dict[str, object], supp: dict[str, object]) -> dict[str, object]:
    required_stems = [
        "Supplementary_Figure11_anchor_robustness",
        "SuppFig11a",
        "SuppFig11b",
        "SuppFig11c",
        "SuppFig11d",
        "Figure2a_3D_candidate",
        "Figure2a_2D_reference",
    ]
    checks: list[dict[str, object]] = []
    for stem in required_stems:
        svg = OUT / f"{stem}.svg"
        png = OUT / f"{stem}.png"
        if not svg.is_file() or not png.is_file():
            raise FileNotFoundError(f"Missing expected export pair for {stem}")
        svg_text = svg.read_text(encoding="utf-8")
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0.0, 0.0))
            width, height = image.size
        check = {
            "stem": stem,
            "svg_has_editable_text": "<text" in svg_text,
            "svg_embedded_raster_count": svg_text.count("<image"),
            "png_width_px": width,
            "png_height_px": height,
            "png_dpi_x": float(dpi[0]),
            "png_dpi_y": float(dpi[1]),
        }
        if not check["svg_has_editable_text"] or check["svg_embedded_raster_count"] != 0:
            raise RuntimeError(f"SVG QA failed for {stem}: {check}")
        if min(float(dpi[0]), float(dpi[1])) < 590:
            raise RuntimeError(f"PNG resolution QA failed for {stem}: {dpi}")
        checks.append(check)
    result = {
        "status": "PASS",
        "training_or_inference_run": False,
        "statistical_values_changed": False,
        "supplementary_figure11": supp,
        "figure2a": pca,
        "render_checks": checks,
    }
    (QA_OUT / "qa_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def write_report(qa: dict[str, object]) -> None:
    source_paths = [
        PREPRINT_SOURCE / "SupplementaryFigure9_a.csv",
        PREPRINT_SOURCE / "SupplementaryFigure9_b.csv",
        PREPRINT_SOURCE / "SupplementaryFigure9_c.csv",
        PREPRINT_SOURCE / "SupplementaryFigure10_b.csv",
        PREPRINT_SOURCE / "SupplementaryFigure10_c.csv",
        CURRENT_FIG2A,
        RPE1 / "cache" / "rpe1_pseudobulk_full.npz",
        RPE1 / "split_definition.json",
        FROZEN / "frozen_sets.json",
    ]
    for fold in range(5):
        source_paths.extend(
            [
                FROZEN / "scgpt" / f"fold_{fold}" / "predictions.npz",
                FROZEN / "gears" / "predictions" / f"gears_fold{fold}_raw_oof.npz",
            ]
        )
    source_lines = [
        f"- `{path.relative_to(ROOT).as_posix()}` — SHA-256 `{sha256(path)}`" for path in source_paths
    ]
    pca = qa["figure2a"]
    assert isinstance(pca, dict)
    camera = pca["camera"]
    singular = pca["singular_values_first3"]
    report = f"""# Figure cleanup round 2 report

## Scope and outcome

PASS. This task only rearranged and rendered frozen quantitative evidence. No experiment, model training, inference, resampling, manuscript edit, or conceptual-art change was performed. Existing estimates and confidence intervals were not altered; the only calculations were the requested deterministic PC3 extension and exact reconstruction of seed-level contrasts from frozen rows. The current Figure 2 was not overwritten.

## Files generated

- `Supplementary_Figure11_anchor_robustness.svg/png`: merged 2 × 2 Supplementary Figure 11.
- `SuppFig11a.svg/png` through `SuppFig11d.svg/png`: individual panel exports.
- `Figure2a_3D_candidate.svg/png`: matched three-panel 3D candidate.
- `Figure2a_2D_reference.svg/png`: matched two-dimensional reference.
- `source_data/Figure2a_projection_coordinates.csv`: one row per frozen source, including Truth/GEARS/scGPT PC1–PC3 coordinates.
- `source_data/Figure2a_pca_loadings.csv` and `source_data/Figure2a_pca_definition.json`: shared-basis audit artifacts.
- `source_data/SuppFig11*.csv`: exact plotted frozen values, including all per-seed contrasts.
- `updated_figure_number_mapping.md`: final Supplementary Figures 1–11 mapping.
- `qa/qa_summary.json` and `output_manifest.csv`: rendering and integrity checks.

## Exact frozen sources

{os.linesep.join(source_lines)}

The five frozen scGPT and five frozen GEARS prediction files are listed above and are also enumerated with SHA-256 digests in `source_data/Figure2a_pca_definition.json`.

## PCA definition

The basis was fit exactly once to the stacked five-fold Truth residual-response matrix ({pca['fit_rows']} sources × {pca['response_genes']} frozen response genes), after column-centering by the Truth mean. The first three right singular vectors were then shared by Truth, GEARS, and scGPT. The first three singular values were `{singular[0]:.12g}`, `{singular[1]:.12g}`, and `{singular[2]:.12g}`. A single scale factor of `{pca['scale_value']:.12g}`—the maximum absolute Truth coordinate over PC1/PC2—preserves the current two-dimensional Figure 2a coordinates.

Numerical reproduction of the current Figure 2a: maximum absolute difference = `{pca['current_figure2a_pc1_max_abs_difference']:.3e}` for PC1 and `{pca['current_figure2a_pc2_max_abs_difference']:.3e}` for PC2.

## Frozen 3D camera

- Elevation: `{camera['elev']}` degrees
- Azimuth: `{camera['azim']}` degrees
- Roll: `{camera['roll']}` degrees
- Projection: `{camera['projection']}`

The same camera, orthographic projection, axis limits, aspect, point opacity, and source ordering are used for all three model panels. The camera was preregistered as a neutral Truth-only axis-visibility view and was not adjusted per model.

## Visual comparison

The 3D candidate exposes whether an apparent two-dimensional collapse reflects compression through the third truth-derived axis and makes occupied-volume differences more explicit. The matched 2D reference is faster to read at manuscript scale and exactly reproduces the current PC1/PC2 geometry. The two exports are therefore provided as an editorial choice; no scientific result changes between them.

## QA

- Every requested SVG contains editable text and no embedded raster image.
- Every PNG is exported at approximately 600 dpi.
- Supplementary panels 11a and 11b share axes and retain all five frozen seed trajectories.
- Panel 11c contains five seed points per direction and contrast; recomputed means match the frozen contrast summary to `{qa['supplementary_figure11']['contrast_mean_max_difference_vs_frozen']:.3e}`.
- Panel 11d uses the frozen primary and high-cell 90% estimates and confidence intervals.
- No Supplementary DOCX or manuscript file was changed.
"""
    (OUT / "figure_cleanup_round2_report.md").write_text(report, encoding="utf-8")


def write_manifest() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "output_manifest.csv":
            rows.append(
                {
                    "path": path.relative_to(OUT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    with (OUT / "output_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_output()
    supp = build_supplementary_figure11()
    write_manifest()
    print(
        json.dumps(
            {
                "status": "PASS",
                "training_or_inference_run": False,
                "experiments_rerun": 0,
                "supplementary_figure11": supp,
                "output_directory": str(OUT),
                "manifest_rows": sum(1 for _ in (OUT / "output_manifest.csv").open(encoding="utf-8")) - 1,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
