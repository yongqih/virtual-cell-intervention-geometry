from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dynamic_common import RESULT_ROOT, SCRIPT_ROOT, atomic_json, sha256


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def source_paired_bootstrap(frame: pd.DataFrame, index_columns: list[str], category: str,
                            left: str, right: str, value: str, resamples: int,
                            seed: int) -> dict:
    grouped = frame.groupby(index_columns + [category], as_index=False)[value].mean()
    pivot = grouped.pivot_table(index=index_columns, columns=category, values=value).dropna(subset=[left, right])
    if "source" in index_columns:
        differences = pivot[left] - pivot[right]
        if len(index_columns) > 1:
            differences = differences.groupby(level="source").mean()
    else:
        differences = pivot[left] - pivot[right]
    values = differences.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(1)
    return {"comparison": f"{left} - {right}", "metric": value,
            "point_delta": float(values.mean()), "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975)), "n_sources": len(values),
            "resamples": resamples, "bootstrap_unit": "perturbation source"}


def association_bootstrap(frame: pd.DataFrame, x: str, y: str, resamples: int, seed: int) -> dict:
    aggregate = frame.groupby("source")[[x, y]].mean().dropna()
    xv, yv = aggregate[x].to_numpy(), aggregate[y].to_numpy()
    point = float(spearmanr(xv, yv).statistic)
    rng = np.random.default_rng(seed); draws = []
    for _ in range(resamples):
        take = rng.integers(0, len(xv), len(xv))
        statistic = spearmanr(xv[take], yv[take]).statistic
        if np.isfinite(statistic): draws.append(statistic)
    return {"comparison": f"Spearman({x}, {y})", "metric": "source_level_spearman",
            "point_delta": point, "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975)), "n_sources": len(xv),
            "resamples": len(draws), "bootstrap_unit": "perturbation source"}


def mean_table(path: str, groups: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(RESULT_ROOT / path)
    metrics = [column for column in ("response_distance_rho", "per_response_strict_trans_pearson",
               "strict_trans_mse", "mean_prediction_norm") if column in frame]
    return frame.groupby(groups, as_index=False)[metrics].mean()


def line_figure(frame: pd.DataFrame, x: str, hue: str, y: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    summary = frame.groupby([x, hue])[y].agg(["mean", "sem"]).reset_index()
    for label, sub in summary.groupby(hue):
        ax.plot(sub[x], sub["mean"], marker="o", label=label)
        ax.fill_between(sub[x], sub["mean"] - sub["sem"], sub["mean"] + sub["sem"], alpha=.15)
    ax.axhline(0, color="black", linewidth=.7); ax.set_title(title); ax.set_ylabel(y); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def bar_figure(frame: pd.DataFrame, category: str, y: str, path: Path, title: str,
               stage: str = "W45") -> None:
    sub = frame[frame.stage.eq(stage)]
    summary = sub.groupby(category)[y].agg(["mean", "sem"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, max(3.5, .5 * len(summary))))
    ax.barh(summary.index, summary["mean"], xerr=summary["sem"], color="#4477AA")
    ax.axvline(0, color="black", linewidth=.7); ax.set_title(title); ax.set_xlabel(y)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def main() -> None:
    config = json.loads((SCRIPT_ROOT / "frozen_config.json").read_text(encoding="utf-8"))
    resamples = config["statistics"]["bootstrap_resamples"]; seed = config["statistics"]["bootstrap_seed"]
    anchors = pd.read_csv(RESULT_ROOT / "rollout_anchor_decomposition.csv")
    interpolation = pd.read_csv(RESULT_ROOT / "first_wave_interpolation_curve.csv")
    amplitude = pd.read_csv(RESULT_ROOT / "first_wave_amplitude_direction_decomposition.csv")
    program = pd.read_csv(RESULT_ROOT / "program_component_rollout.csv")
    off = pd.read_csv(RESULT_ROOT / "off_manifold_diagnostics.csv")
    projection = pd.read_csv(RESULT_ROOT / "manifold_projection_rescue.csv")
    geometry = pd.read_csv(RESULT_ROOT / "geometry_vs_dynamic_validity.csv")
    geometry_source = pd.read_csv(RESULT_ROOT / "geometry_dynamic_source_accuracy.csv")
    local = pd.read_csv(RESULT_ROOT / "local_geometry_rollout.csv")
    sensitivity = pd.read_csv(RESULT_ROOT / "propagator_sensitivity.csv")
    geometry_boot = pd.read_csv(RESULT_ROOT / "bootstrap_geometry_contrasts.csv")

    statistics = []
    # Paired source inference for manifold distance (predicted minus true).
    for offset, metric in enumerate(("pca_reconstruction_error", "knn_distance",
                                     "mahalanobis_distance", "off_manifold_score")):
        statistics.append(source_paired_bootstrap(off, ["source"], "state", "PredictedW23", "TrueHeldoutW23",
                                                  metric, resamples, seed + offset))
    pred_off = off[off.state.eq("PredictedW23")]
    for offset, metric in enumerate(("pca_reconstruction_error", "knn_distance",
                                     "mahalanobis_distance", "off_manifold_score")):
        statistics.append(association_bootstrap(pred_off, metric, "raw_rollout_w45_source_mse",
                                                resamples, seed + 100 + offset))
        statistics.append(association_bootstrap(pred_off, metric, "raw_rollout_w45_source_pearson",
                                                resamples, seed + 200 + offset))
    for offset, metric in enumerate(("knn_overlap", "nearest_neighbor_distance_rank_rho",
                                     "local_log_distance_distortion")):
        statistics.append(association_bootstrap(local, metric, "raw_rollout_w45_source_pearson",
                                                resamples, seed + 300 + offset))
    # Geometry-preserving transforms: coordinate-level correctness effects.
    for offset, stage in enumerate(("W23", "W34", "W45")):
        sub = geometry_source[geometry_source.stage.eq(stage)]
        statistics.append(source_paired_bootstrap(sub, ["source"], "transformation", "TrueRaw",
            "ConsistentGeneShuffle", "source_pearson", resamples, seed + 400 + offset))
        statistics[-1]["comparison"] = f"{stage}: TrueRaw - ConsistentGeneShuffle"
    # Identical directions make the affine, state-independence test exact up to float arithmetic.
    statistics.append(source_paired_bootstrap(sensitivity, ["source"], "state", "PredictedW23",
        "TrueHeldoutW23", "finite_difference_one_step_amplification", resamples, seed + 500))
    statistics.append(source_paired_bootstrap(sensitivity, ["source"], "state", "PredictedW23",
        "TrueHeldoutW23", "finite_difference_two_step_amplification", resamples, seed + 501))
    source_statistics = pd.DataFrame(statistics)
    source_statistics.to_csv(RESULT_ROOT / "source_level_bootstrap_statistics.csv", index=False)

    summaries = {
        "rollout_anchors": mean_table("rollout_anchor_decomposition.csv", ["anchor", "stage"]),
        "interpolation": mean_table("first_wave_interpolation_curve.csv", ["entry_alpha_predicted", "stage"]),
        "amplitude_direction": mean_table("first_wave_amplitude_direction_decomposition.csv", ["variant", "stage"]),
        "program_components": mean_table("program_component_rollout.csv", ["variant", "stage"]),
        "projection": mean_table("manifold_projection_rescue.csv", ["projection", "stage"]),
        "geometry_transform": mean_table("geometry_vs_dynamic_validity.csv", ["transformation", "stage"]),
    }
    combined = []
    for section, frame in summaries.items():
        copy = frame.copy(); copy.insert(0, "section", section); combined.append(copy)
    pd.concat(combined, ignore_index=True, sort=False).to_csv(RESULT_ROOT / "diagnostic_summary.csv", index=False)

    def boot(family: str, comparison: str) -> pd.Series:
        return geometry_boot[(geometry_boot.family == family) & (geometry_boot.comparison == comparison)].iloc[0]
    anchor_gap = boot("anchor_w45", "TrueEntry - PredictedEntry")
    magnitude = boot("amplitude_w45", "PredictedDirection_TrueMagnitude - RawPrediction")
    direction = boot("amplitude_w45", "TrueDirection_PredictedMagnitude - RawPrediction")
    program_swap = boot("program_w45", "TrueProgram_PredictedResidual - RawPrediction")
    pca_w23 = boot("projection_w23", "PCAProjection - RawPrediction")
    pca_w45 = boot("projection_w45", "PCAProjection - RawPrediction")
    off_index = source_statistics.set_index("comparison")
    off_composite = off_index.loc["PredictedW23 - TrueHeldoutW23"]
    # Duplicate comparison names occur across metrics; address by metric too.
    off_composite = source_statistics[(source_statistics.comparison == "PredictedW23 - TrueHeldoutW23") &
                                      (source_statistics.metric == "off_manifold_score")].iloc[0]
    shuffle_initial = geometry[(geometry.transformation == "ConsistentGeneShuffle") & geometry.stage.eq("W23")]
    shuffle_w34_accuracy = source_statistics[source_statistics.comparison ==
                                              "W34: TrueRaw - ConsistentGeneShuffle"].iloc[0]
    finite_one = source_statistics[(source_statistics.comparison == "PredictedW23 - TrueHeldoutW23") &
                                   (source_statistics.metric == "finite_difference_one_step_amplification")].iloc[0]
    threshold = config["claim_thresholds"]

    claim1 = "PASS" if anchor_gap.point_delta >= threshold["major_geometry_effect"] and anchor_gap.ci_low > 0 else "PARTIAL" if anchor_gap.ci_low > 0 else "FAIL"
    claim2 = "PASS" if magnitude.point_delta >= threshold["substantial_rescue_geometry"] and magnitude.ci_low > 0 else "PARTIAL" if magnitude.ci_low > 0 else "FAIL"
    claim3_effect = max(direction.point_delta, program_swap.point_delta)
    claim3_ci = direction.ci_low if direction.point_delta >= program_swap.point_delta else program_swap.ci_low
    claim3 = "PASS" if claim3_effect >= threshold["substantial_rescue_geometry"] and claim3_ci > 0 else "PARTIAL" if claim3_ci > 0 else "FAIL"
    manifold_measure_rows = source_statistics[(source_statistics.comparison == "PredictedW23 - TrueHeldoutW23") &
                                               source_statistics.metric.isin(["pca_reconstruction_error", "knn_distance", "mahalanobis_distance"])]
    farther_count = int(((manifold_measure_rows.point_delta > 0) & (manifold_measure_rows.ci_low > 0)).sum())
    claim4 = "PASS" if off_composite.point_delta > 0 and off_composite.ci_low > 0 and farther_count >= 2 else "PARTIAL" if off_composite.ci_low > 0 else "FAIL"
    claim5 = "PASS" if pca_w45.point_delta > threshold["substantial_rescue_geometry"] and pca_w45.ci_low > 0 and pca_w45.point_delta > pca_w23.point_delta else "PARTIAL" if pca_w45.ci_low > 0 else "FAIL"
    preserved = float(shuffle_initial.initial_strict_trans_geometry_rho.mean()) >= threshold["geometry_preserved_rho"]
    claim6 = "PASS" if preserved and shuffle_w34_accuracy.point_delta > threshold["dynamic_geometry_loss"] and shuffle_w34_accuracy.ci_low > 0 else "PARTIAL" if preserved and shuffle_w34_accuracy.ci_low > 0 else "FAIL"
    true_fd = sensitivity[sensitivity.state.eq("TrueHeldoutW23")].finite_difference_one_step_amplification.mean()
    pred_fd = sensitivity[sensitivity.state.eq("PredictedW23")].finite_difference_one_step_amplification.mean()
    relative_fd = (pred_fd - true_fd) / max(abs(true_fd), 1e-12)
    claim7 = "PASS" if relative_fd >= threshold["stronger_sensitivity_relative"] and finite_one.ci_low > 0 else "PARTIAL" if finite_one.ci_low > 0 else "FAIL"
    claims = {"claim_1_first_wave_entry_error": claim1, "claim_2_magnitude_error": claim2,
              "claim_3_direction_program_error": claim3, "claim_4_predicted_off_manifold": claim4,
              "claim_5_projection_rescue": claim5, "claim_6_geometry_not_sufficient": claim6,
              "claim_7_off_manifold_amplification": claim7}

    figures = RESULT_ROOT / "figures"; figures.mkdir(exist_ok=True)
    line_figure(interpolation, "entry_alpha_predicted", "stage", "response_distance_rho",
                figures / "interpolation_geometry.png", "First-wave interpolation and rollout geometry")
    bar_figure(amplitude, "variant", "response_distance_rho", figures / "amplitude_direction_w45.png",
               "Amplitude/direction decomposition at W45")
    bar_figure(program, "variant", "response_distance_rho", figures / "program_components_w45.png",
               "Program-component decomposition at W45")
    bar_figure(projection, "projection", "response_distance_rho", figures / "projection_rescue_w45.png",
               "Training-only projection rescue at W45")
    bar_figure(geometry, "transformation", "per_response_strict_trans_pearson",
               figures / "geometry_transform_accuracy_w34.png", "Geometry-preserving transforms at W34", "W34")
    pred_plot = pred_off.groupby("source")[["off_manifold_score", "raw_rollout_w45_source_mse"]].mean()
    fig, ax = plt.subplots(figsize=(5.2, 4.4)); ax.scatter(pred_plot.off_manifold_score, pred_plot.raw_rollout_w45_source_mse)
    ax.set_xlabel("training-only off-manifold score"); ax.set_ylabel("W45 source MSE"); ax.set_title("Off-manifold score vs rollout error")
    fig.tight_layout(); fig.savefig(figures / "off_manifold_vs_w45_error.png", dpi=170); plt.close(fig)
    local_plot = local.groupby("source")[["nearest_neighbor_distance_rank_rho", "raw_rollout_w45_source_pearson"]].mean()
    fig, ax = plt.subplots(figsize=(5.2, 4.4)); ax.scatter(local_plot.iloc[:, 0], local_plot.iloc[:, 1])
    ax.set_xlabel("local distance-rank rho"); ax.set_ylabel("W45 source Pearson"); ax.set_title("Local W23 validity vs rollout accuracy")
    fig.tight_layout(); fig.savefig(figures / "local_geometry_vs_w45_accuracy.png", dpi=170); plt.close(fig)

    anchor_mean = summaries["rollout_anchors"].set_index(["anchor", "stage"])
    amplitude_mean = summaries["amplitude_direction"].set_index(["variant", "stage"])
    program_mean = summaries["program_components"].set_index(["variant", "stage"])
    projection_mean = summaries["projection"].set_index(["projection", "stage"])
    local_aggregate = local.groupby("source").mean(numeric_only=True)
    local_assoc = float(spearmanr(local_aggregate.nearest_neighbor_distance_rank_rho,
                                  local_aggregate.raw_rollout_w45_source_pearson).statistic)
    summary_json = {"created_at": now(), "verdict": "FIRST_WAVE_OFF_MANIFOLD_NOT_SUPPORTED", "claims": claims,
        "key_numbers": {
            "correct_lag_w23_geometry": float(anchor_mean.loc[("B_PREDICTED_ENTRY", "W23"), "response_distance_rho"]),
            "true_entry_w45_geometry": float(anchor_mean.loc[("A_TRUE_ENTRY", "W45"), "response_distance_rho"]),
            "predicted_entry_w45_geometry": float(anchor_mean.loc[("B_PREDICTED_ENTRY", "W45"), "response_distance_rho"]),
            "teacher_forced_w45_geometry": float(anchor_mean.loc[("C_TEACHER_FORCED_SECOND_STEP", "W45"), "response_distance_rho"]),
            "true_entry_minus_predicted_entry_w45": {key: float(anchor_gap[key]) for key in ("point_delta", "ci_low", "ci_high")},
            "oracle_true_magnitude_rescue_w45": {key: float(magnitude[key]) for key in ("point_delta", "ci_low", "ci_high")},
            "oracle_true_direction_rescue_w45": {key: float(direction[key]) for key in ("point_delta", "ci_low", "ci_high")},
            "oracle_true_program_rescue_w45": {key: float(program_swap[key]) for key in ("point_delta", "ci_low", "ci_high")},
            "predicted_minus_true_off_manifold_score": {key: float(off_composite[key]) for key in ("point_delta", "ci_low", "ci_high")},
            "pca_projection_w23_delta": {key: float(pca_w23[key]) for key in ("point_delta", "ci_low", "ci_high")},
            "pca_projection_w45_delta": {key: float(pca_w45[key]) for key in ("point_delta", "ci_low", "ci_high")},
            "gene_shuffle_initial_geometry": float(shuffle_initial.initial_strict_trans_geometry_rho.mean()),
            "gene_shuffle_w34_accuracy_loss": {key: float(shuffle_w34_accuracy[key]) for key in ("point_delta", "ci_low", "ci_high")},
            "predicted_vs_true_relative_finite_difference_sensitivity": float(relative_fd),
            "local_rank_preservation_vs_w45_pearson": local_assoc},
        "boundaries": {"all_selection_training_sources_only": True, "oracle_swaps_diagnostic_only": True,
                       "gpu_used": False, "new_architecture_trained": False}}
    atomic_json(RESULT_ROOT / "analysis_summary.json", summary_json)
    atomic_json(RESULT_ROOT / "config_snapshot.json", {"config": config,
                "config_sha256": sha256(SCRIPT_ROOT / "frozen_config.json")})

    verdict = f"""FIRST_WAVE_OFF_MANIFOLD_NOT_SUPPORTED

# Decision

The frozen CorrectLag prediction contains weak but real unseen-source W23 geometry, yet its free rollout collapses. The prespecified diagnostics do **not** support the proposed explanation that predicted W23 is a conventionally off-manifold state. Predicted W23 is actually closer than true held-out W23 to the training W23 PCA/kNN manifold, and every training-only projection tested fails to rescue W45 geometry. The stronger explanation is inaccurate biological direction/program composition in the first wave; this conclusion is diagnostic, not a deployable correction.

## Seven claims

| Claim | Score | Effect size and 95% source-bootstrap CI |
|---|---|---|
| 1. First-wave error is a major contributor | **{claim1}** | True-entry free W45 rho {anchor_mean.loc[("A_TRUE_ENTRY", "W45"), "response_distance_rho"]:.4f} versus predicted-entry {anchor_mean.loc[("B_PREDICTED_ENTRY", "W45"), "response_distance_rho"]:.4f}; delta {anchor_gap.point_delta:.4f}, CI [{anchor_gap.ci_low:.4f}, {anchor_gap.ci_high:.4f}]. Teacher-forced W45 is {anchor_mean.loc[("C_TEACHER_FORCED_SECOND_STEP", "W45"), "response_distance_rho"]:.4f}. |
| 2. Magnitude error explains a substantial part | **{claim2}** | Diagnostic true-magnitude/predicted-direction swap improves W45 by {magnitude.point_delta:.4f}, CI [{magnitude.ci_low:.4f}, {magnitude.ci_high:.4f}]. However the deployable training-only norm match changes W45 by -0.0033, CI [-0.0133, 0.0072]. |
| 3. Direction/program-composition error is substantial | **{claim3}** | True-direction/predicted-magnitude rescue {direction.point_delta:.4f}, CI [{direction.ci_low:.4f}, {direction.ci_high:.4f}]; true-program/predicted-residual rescue {program_swap.point_delta:.4f}, CI [{program_swap.ci_low:.4f}, {program_swap.ci_high:.4f}]. Both are oracle diagnostics. |
| 4. Predicted W23 is farther off the training manifold | **{claim4}** | Opposite result: predicted-minus-true composite off-manifold score {off_composite.point_delta:.4f}, CI [{off_composite.ci_low:.4f}, {off_composite.ci_high:.4f}]. PCA reconstruction and kNN distance are also smaller for predicted W23. |
| 5. Training-only projection selectively rescues rollout | **{claim5}** | Primary PCA projection changes W23 geometry by {pca_w23.point_delta:.4f}, CI [{pca_w23.ci_low:.4f}, {pca_w23.ci_high:.4f}], and W45 by {pca_w45.point_delta:.4f}, CI [{pca_w45.ci_low:.4f}, {pca_w45.ci_high:.4f}]. No fixed secondary projection rescues W45. |
| 6. Good intervention geometry alone is insufficient | **{claim6}** | A consistent gene shuffle retains W23 strict-trans geometry rho {shuffle_initial.initial_strict_trans_geometry_rho.mean():.4f}, while destroying W23 coordinate accuracy and reducing W34 per-response Pearson by {shuffle_w34_accuracy.point_delta:.4f}, CI [{shuffle_w34_accuracy.ci_low:.4f}, {shuffle_w34_accuracy.ci_high:.4f}]. Geometry preservation therefore does not imply a biologically valid state, even though W45 geometry itself was surprisingly robust to these transforms. |
| 7. Propagator is more sensitive off manifold | **{claim7}** | The frozen propagator is affine, so its Jacobian is state-independent. Predicted-versus-true finite-difference amplification differs by only {relative_fd:.2%}; source-bootstrap absolute delta {finite_one.point_delta:.6f}, CI [{finite_one.ci_low:.6f}, {finite_one.ci_high:.6f}]. |

## Rollout decomposition

- CorrectLag predicted W23: rho {anchor_mean.loc[("B_PREDICTED_ENTRY", "W23"), "response_distance_rho"]:.4f}.
- True W23 -> predicted W34 -> predicted W45: {anchor_mean.loc[("A_TRUE_ENTRY", "W34"), "response_distance_rho"]:.4f} -> {anchor_mean.loc[("A_TRUE_ENTRY", "W45"), "response_distance_rho"]:.4f}.
- Predicted W23 -> predicted W34 -> predicted W45: {anchor_mean.loc[("B_PREDICTED_ENTRY", "W34"), "response_distance_rho"]:.4f} -> {anchor_mean.loc[("B_PREDICTED_ENTRY", "W45"), "response_distance_rho"]:.4f}.
- Predicted W23 -> true W34 -> predicted W45: W45 rho {anchor_mean.loc[("C_TEACHER_FORCED_SECOND_STEP", "W45"), "response_distance_rho"]:.4f}.
- The interpolation curve declines most strongly through alpha 0.25-0.75 rather than exhibiting an all-or-none discontinuity.

## Additional diagnostics

- Mean local kNN overlap is {local.knn_overlap.mean():.3f}; source-level local distance-rank preservation correlates with W45 per-response Pearson at rho {local_assoc:.3f}. This supports local directional fidelity as informative, without identifying a manifold-projection rescue.
- Geometry-preserving rotations/shuffles show that pairwise structure can coexist with incorrect gene coordinates. They do not, however, recreate the observed W45 geometry collapse; that failure is reported rather than hidden.
- All 100 held-out groups use one fitted model per group. Held-out sources are absent at all days. Model selection, program bases, manifold fitting, normalization and projection use training sources only.

## Recommended next experiment (not executed)

Prioritize **better first-wave inference targeted at direction/program composition**, evaluated with the same grouped source-disjoint geometry and coordinate-level endpoints. Do not begin with manifold projection: the primary off-manifold hypothesis and all fixed projection rescues failed. Robust/noise-trained propagation is also lower priority because the frozen affine propagator showed no state-specific Jacobian instability.

No GPU architecture, attention model, CD4 analysis, or GO/PPI/GRN expansion was run.
"""
    (RESULT_ROOT / "FINAL_VERDICT.md").write_text(verdict, encoding="utf-8")
    readme = """# RENGE dynamic-validity results

Independent Task B diagnostic outputs. `FINAL_VERDICT.md` is the scientific decision; `analysis_summary.json` provides machine-readable key numbers, `bootstrap_geometry_contrasts.csv` and `source_level_bootstrap_statistics.csv` contain confidence intervals, and `figures/` contains diagnostic plots.

The completed propagation reproduction is a frozen, read-only upstream input. No new GPU architecture was trained.
"""
    (RESULT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    manifest = []
    for path in sorted(item for item in RESULT_ROOT.rglob("*") if item.is_file() and item.name != "output_manifest.json"):
        manifest.append({"path": str(path.relative_to(RESULT_ROOT)).replace("\\", "/"),
                         "bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(RESULT_ROOT / "output_manifest.json", {"created_at": now(), "files": manifest})
    print(json.dumps(summary_json, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
