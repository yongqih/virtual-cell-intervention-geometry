from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results" / "natural_fluctuation_igc_anchor"
DEFAULT_OUTPUT = DEFAULT_INPUT / "supplementary_figure5_source_data"


def _one(frame: pd.DataFrame, **filters) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row for {filters}; found {len(selected)}")
    return selected.iloc[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(actual: float, expected: float, label: str, atol: float = 5e-7) -> None:
    if not np.isclose(actual, expected, atol=atol, rtol=0):
        raise RuntimeError(f"Frozen-value mismatch for {label}: {actual} != {expected}")


def build(input_dir: Path, output_dir: Path) -> None:
    required = [
        "bootstrap_summary.csv",
        "control_cell_audit.csv",
        "data_provenance.json",
        "experiment_manifest.json",
        "geometry_summary.csv",
        "oracle_projection_results.csv",
        "per_target_orientation.csv",
        "spectral_summary.csv",
        "verdict.json",
        "zero_shot_projection_results.csv",
    ]
    missing = [name for name in required if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")

    verdict = json.loads((input_dir / "verdict.json").read_text(encoding="utf-8"))
    if verdict.get("verdict") != "NATURAL_FLUCTUATION_ANCHOR_NOT_SUPPORTED":
        raise RuntimeError("Unexpected frozen natural-fluctuation verdict")

    bootstrap = pd.read_csv(input_dir / "bootstrap_summary.csv")
    control = pd.read_csv(input_dir / "control_cell_audit.csv")
    geometry = pd.read_csv(input_dir / "geometry_summary.csv")
    oracle = pd.read_csv(input_dir / "oracle_projection_results.csv")
    orientation = pd.read_csv(input_dir / "per_target_orientation.csv")
    spectral = pd.read_csv(input_dir / "spectral_summary.csv")
    zero = pd.read_csv(input_dir / "zero_shot_projection_results.csv")

    if int(control.iloc[0].n_control_cells) != 11_485:
        raise RuntimeError("Control-cell count is not the frozen 11,485")
    if int(control.iloc[0].n_common_strict_trans_genes) != 768:
        raise RuntimeError("Strict-trans response panel is not the frozen 768 genes")

    def orientation_mean(estimator: str, truth_space: str) -> pd.Series:
        return _one(
            bootstrap,
            analysis="orientation",
            estimator=estimator,
            truth_space=truth_space,
            metric="cosine",
            statistic="mean",
        )

    raw_r = orientation_mean("raw_corr", "r_control_relative")
    raw_q = orientation_mean("raw_corr", "q_intervention_residual")
    resid_q = orientation_mean("resid_corr", "q_intervention_residual")
    panel_e = pd.DataFrame(
        [
            {
                "order": 0,
                "level": "Full control-relative response r",
                "estimator": "raw_corr",
                "truth_space": "r_control_relative",
                "mean_cosine": raw_r.estimate,
                "ci_low": raw_r.ci_low,
                "ci_high": raw_r.ci_high,
                "n_sources": int(raw_r.n_sources),
            },
            {
                "order": 1,
                "level": "Intervention-specific residual q",
                "estimator": "raw_corr",
                "truth_space": "q_intervention_residual",
                "mean_cosine": raw_q.estimate,
                "ci_low": raw_q.ci_low,
                "ci_high": raw_q.ci_high,
                "n_sources": int(raw_q.n_sources),
            },
            {
                "order": 2,
                "level": "Residualized intervention-specific residual q",
                "estimator": "resid_corr",
                "truth_space": "q_intervention_residual",
                "mean_cosine": resid_q.estimate,
                "ci_low": resid_q.ci_low,
                "ci_high": resid_q.ci_high,
                "n_sources": int(resid_q.n_sources),
            },
        ]
    )
    _close(float(raw_r.estimate), 0.200266, "panel e raw r")
    _close(float(raw_q.estimate), 0.005704, "panel e raw q")
    _close(float(resid_q.estimate), -0.001823, "panel e residualized q")

    resid_targets = orientation[
        (orientation.estimator == "resid_corr")
        & (orientation.truth_space == "q_intervention_residual")
    ].copy()
    if len(resid_targets) != 1_755 or resid_targets.source.nunique() != 1_755:
        raise RuntimeError("Residualized target table is not the frozen 1,755-source evaluation")
    panel_f = pd.DataFrame(
        [
            {
                "order": 0,
                "metric": "Correct orientation",
                "estimate_fraction": float((resid_targets.cosine > 0).mean()),
                "null_fraction": 0.50,
                "definition": "fraction of residualized target cosine values greater than zero",
                "n_sources": len(resid_targets),
                "permutations": 200,
            },
            {
                "order": 1,
                "metric": "Residualized sign agreement",
                "estimate_fraction": float(resid_targets.sign_agreement_top10pct_truth.mean()),
                "null_fraction": 0.50,
                "definition": "mean sign agreement over the top 10% absolute truth coordinates",
                "n_sources": len(resid_targets),
                "permutations": 200,
            },
            {
                "order": 2,
                "metric": "Targets with permutation p < 0.05",
                "estimate_fraction": float((resid_targets.target_permutation_empirical_p_cosine < 0.05).mean()),
                "null_fraction": 0.05,
                "definition": "fraction exceeding the frozen 200-permutation target null",
                "n_sources": len(resid_targets),
                "permutations": 200,
            },
        ]
    )
    _close(float(panel_f.iloc[0].estimate_fraction), 0.480912, "panel f correct orientation")
    _close(float(panel_f.iloc[1].estimate_fraction), 0.500315, "panel f sign agreement")
    _close(float(panel_f.iloc[2].estimate_fraction), 0.049003, "panel f permutation exceedance")
    panel_f_distribution = resid_targets[
        [
            "source",
            "fold",
            "cosine",
            "sign_agreement_top10pct_truth",
            "target_permutation_mean_cosine",
            "target_permutation_empirical_p_cosine",
        ]
    ].sort_values(["fold", "source"])

    raw_oracle = oracle[
        (oracle.estimator == "raw_cov") & (oracle.truth_space == "q_intervention_residual")
    ]
    resid_oracle = oracle[
        (oracle.estimator == "resid_corr") & (oracle.truth_space == "q_intervention_residual")
    ]
    resid_zero = zero[
        (zero.estimator == "resid_corr")
        & (zero.truth_space == "q_intervention_residual")
        & (zero.calibration == "global_scalar")
    ]
    panel_g = pd.DataFrame(
        [
            {
                "order": 0,
                "access": "Held-response oracle",
                "estimator": "raw_cov",
                "median_response_energy_fraction": float(raw_oracle.oracle_energy_fraction.median()),
                "probability_alpha_positive": float(raw_oracle.alpha_oracle_positive.mean()),
                "mean_cosine": np.nan,
                "mean_pearson": np.nan,
                "mean_r2_zero_baseline": np.nan,
                "n_sources": len(raw_oracle),
            },
            {
                "order": 1,
                "access": "Held-response oracle",
                "estimator": "resid_corr",
                "median_response_energy_fraction": float(resid_oracle.oracle_energy_fraction.median()),
                "probability_alpha_positive": float(resid_oracle.alpha_oracle_positive.mean()),
                "mean_cosine": np.nan,
                "mean_pearson": np.nan,
                "mean_r2_zero_baseline": np.nan,
                "n_sources": len(resid_oracle),
            },
            {
                "order": 2,
                "access": "Training-only global scalar",
                "estimator": "resid_corr",
                "median_response_energy_fraction": np.nan,
                "probability_alpha_positive": np.nan,
                "mean_cosine": float(resid_zero.cosine.mean()),
                "mean_pearson": float(resid_zero.pearson.mean()),
                "mean_r2_zero_baseline": float(resid_zero.r2_zero_baseline.mean()),
                "n_sources": len(resid_zero),
            },
        ]
    )
    _close(float(panel_g.iloc[0].median_response_energy_fraction), 0.111741, "panel g raw oracle")
    _close(float(panel_g.iloc[0].probability_alpha_positive), 0.503134, "panel g raw alpha sign")
    _close(float(panel_g.iloc[1].median_response_energy_fraction), 0.000810, "panel g residual oracle")
    _close(float(panel_g.iloc[1].probability_alpha_positive), 0.480912, "panel g residual alpha sign")
    _close(float(panel_g.iloc[2].mean_cosine), 0.000025, "panel g zero-shot cosine")
    _close(float(panel_g.iloc[2].mean_pearson), 0.000692, "panel g zero-shot Pearson")

    established_source = (
        ROOT
        / "results"
        / "cross_dataset_replication_rpe1"
        / "state_intervention"
        / "pairwise_geometry_alignment.csv"
    )
    established_frame = pd.read_csv(established_source)
    established = _one(
        established_frame,
        record_type="summary",
        representation="EstablishedOBS71",
        panel="primary_common_strict_trans",
    )
    residual_geometry = _one(
        geometry,
        record_type="summary",
        estimator="resid_corr",
        prediction_type="orientation_only",
    )
    shuffled_geometry = _one(
        geometry,
        record_type="summary",
        estimator="target_permutation_null",
        prediction_type="source_descriptor",
    )
    scalar_spectrum = _one(
        spectral,
        record_type="fold_mean",
        estimator="resid_corr",
        prediction_type="zero_shot_global_scalar",
    )
    panel_h = pd.DataFrame(
        [
            {
                "order": 0,
                "representation": "EstablishedOBS71",
                "prediction_type": "source_descriptor",
                "grouped_geometry_spearman": established.spearman_rho,
                "ci_low": established.ci_low,
                "ci_high": established.ci_high,
                "knn_overlap_k10": np.nan,
                "target_retrieval_top1": np.nan,
                "target_retrieval_top5": np.nan,
                "between_source_variance_ratio": np.nan,
            },
            {
                "order": 1,
                "representation": "Residualized fluctuation",
                "prediction_type": "orientation_only",
                "grouped_geometry_spearman": residual_geometry.response_distance_spearman,
                "ci_low": residual_geometry.source_bootstrap_ci_low,
                "ci_high": residual_geometry.source_bootstrap_ci_high,
                "knn_overlap_k10": residual_geometry.local_knn_overlap_k10,
                "target_retrieval_top1": residual_geometry.target_retrieval_top1,
                "target_retrieval_top5": residual_geometry.target_retrieval_top5,
                "between_source_variance_ratio": scalar_spectrum.between_source_variance_ratio,
            },
            {
                "order": 2,
                "representation": "Target-permutation null",
                "prediction_type": "source_descriptor",
                "grouped_geometry_spearman": shuffled_geometry.response_distance_spearman,
                "ci_low": shuffled_geometry.source_bootstrap_ci_low,
                "ci_high": shuffled_geometry.source_bootstrap_ci_high,
                "knn_overlap_k10": shuffled_geometry.local_knn_overlap_k10,
                "target_retrieval_top1": np.nan,
                "target_retrieval_top5": np.nan,
                "between_source_variance_ratio": np.nan,
            },
        ]
    )
    _close(float(panel_h.iloc[0].grouped_geometry_spearman), 0.018467, "panel h EstablishedOBS71")
    _close(float(panel_h.iloc[1].grouped_geometry_spearman), 0.007335, "panel h residual geometry")
    _close(float(panel_h.iloc[2].grouped_geometry_spearman), 0.003274, "panel h target null")
    _close(float(panel_h.iloc[1].between_source_variance_ratio), 3.573024e-06, "panel h variance retention", atol=5e-12)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "SupplementaryFigure5_panel_e.csv": panel_e,
        "SupplementaryFigure5_panel_f.csv": panel_f,
        "SupplementaryFigure5_panel_f_target_permutation_distribution.csv": panel_f_distribution,
        "SupplementaryFigure5_panel_g.csv": panel_g,
        "SupplementaryFigure5_panel_h.csv": panel_h,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)

    source_files = []
    for name in sorted(outputs):
        path = output_dir / name
        source_files.append(
            {
                "file": name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "records": len(outputs[name]),
            }
        )
    pd.DataFrame(source_files).to_csv(output_dir / "SOURCE_DATA_MANIFEST.csv", index=False)
    manifest = {
        "figure": "Supplementary Figure 5e-h",
        "verdict": verdict["verdict"],
        "source_only_reproduction_requires_h5ad": False,
        "source_analysis_directory": input_dir.as_posix(),
        "rpe1_h5ad_redistributed": False,
        "rpe1_h5ad_sha256": "25cb5bad6cd7abd834baa191ffa4b0b414dbfe3f5c6b927999380d2d4bc3ae3d",
        "n_control_cells": 11_485,
        "n_sources": 1_755,
        "n_common_strict_trans_genes": 768,
        "fold_sizes": [350, 361, 352, 337, 355],
        "files": source_files,
    }
    (output_dir / "SOURCE_DATA_PROVENANCE.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"PASS: wrote audited Supplementary Figure 5e-h source data to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
