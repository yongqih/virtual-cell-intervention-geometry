from __future__ import annotations

import json
import math
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dynamic_common import (FROZEN_CACHE, FROZEN_RESULT_ROOT, FROZEN_SCRIPT_ROOT, RESULT_ROOT,
                            SCRIPT_ROOT, apply_affine, atomic_json, direct_indices, euclidean_distance,
                            fit_dense_transition, full_cosine_distance, geometry_rho,
                            grouped_twofold_splits, prediction_metrics, safe_pearson, safe_spearman,
                            select_ridge_alpha, sha256, source_metrics, standardized_ridge,
                            strict_trans_cosine_distance, transmission_representations,
                            weighted_spearman)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def unit_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12), norm[:, 0]


def quantile_norm_match(prediction: np.ndarray, train_prediction: np.ndarray,
                        train_truth: np.ndarray) -> np.ndarray:
    pred_unit, pred_norm = unit_rows(prediction)
    reference_pred_norm = np.linalg.norm(train_prediction, axis=1)
    reference_true_norm = np.sort(np.linalg.norm(train_truth, axis=1))
    order = np.argsort(reference_pred_norm)
    sorted_pred = reference_pred_norm[order]
    # Fixed empirical quantile map fitted exclusively on outer-training sources.
    quantiles = np.searchsorted(sorted_pred, pred_norm, side="right") / max(len(sorted_pred), 1)
    indices = np.clip(np.rint(quantiles * (len(reference_true_norm) - 1)).astype(int), 0,
                      len(reference_true_norm) - 1)
    return (pred_unit * reference_true_norm[indices, None]).astype(np.float32)


def pca_basis(matrix: np.ndarray, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    mean = matrix.mean(0).astype(np.float32)
    _, _, vt = np.linalg.svd(matrix.astype(float) - mean, full_matrices=False)
    count = min(dimension, len(vt), max(1, len(matrix) - 1))
    return mean, vt[:count].T.astype(np.float32)


def pca_project(matrix: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return ((matrix - mean) @ components @ components.T + mean).astype(np.float32)


def select_program_basis(waves: np.ndarray, train: np.ndarray, split_seed: int,
                         dimensions: tuple[int, ...], alpha_grid: tuple[float, ...]):
    matrix = waves[train].reshape(-1, waves.shape[-1]).astype(float)
    mean = matrix.mean(0); _, _, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    order = np.random.default_rng(split_seed + 57037).permutation(train)
    count = max(2, int(np.ceil(.20 * len(order)))); inner_val, inner_train = order[:count], order[count:]
    scores = {}
    for dimension in dimensions:
        components = vt[:min(dimension, len(vt))].T
        inner_x = np.concatenate([waves[inner_train, 0], waves[inner_train, 1]])
        inner_y = np.concatenate([waves[inner_train, 1], waves[inner_train, 2]])
        val_x = np.concatenate([waves[inner_val, 0], waves[inner_val, 1]])
        val_y = np.concatenate([waves[inner_val, 1], waves[inner_val, 2]])
        x = (inner_x - mean) @ components; y = (inner_y - mean) @ components
        q = (val_x - mean) @ components; target = (val_y - mean) @ components
        pred, _ = standardized_ridge(x, y, q, alpha_grid)
        scores[dimension] = float(np.mean((pred - target) ** 2))
    selected = min(scores, key=scores.get)
    return mean.astype(np.float32), vt[:selected].T.astype(np.float32), selected, scores


def manifold_raw_measures(query: np.ndarray, train: np.ndarray, mean: np.ndarray,
                          components: np.ndarray, neighbors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projected = pca_project(query, mean, components)
    reconstruction = np.linalg.norm(query - projected, axis=1) / math.sqrt(query.shape[1])
    distance = euclidean_distance(np.concatenate([query, train]))[:len(query), len(query):] / math.sqrt(query.shape[1])
    knn = np.mean(np.partition(distance, min(neighbors, len(train)) - 1, axis=1)[:, :min(neighbors, len(train))], axis=1)
    train_z = (train - mean) @ components; query_z = (query - mean) @ components
    latent_variance = np.var(train_z, axis=0, ddof=1)
    ridge = max(float(np.median(latent_variance)) * .1, 1e-6)
    mahalanobis = np.sqrt(np.sum(query_z**2 / (latent_variance + ridge), axis=1))
    return reconstruction, knn, mahalanobis


def manifold_measures(query: np.ndarray, train: np.ndarray, mean: np.ndarray,
                      components: np.ndarray, neighbors: int) -> dict[str, np.ndarray]:
    reconstruction, knn, mahalanobis = manifold_raw_measures(query, train, mean, components, neighbors)
    train_recon, train_knn, train_maha = manifold_raw_measures(train, train, mean, components, neighbors + 1)
    # Remove self (zero) from the train kNN reference explicitly.
    train_distance = euclidean_distance(train) / math.sqrt(train.shape[1])
    np.fill_diagonal(train_distance, np.inf)
    train_knn = np.mean(np.partition(train_distance, min(neighbors, len(train) - 1) - 1, axis=1)
                        [:, :min(neighbors, len(train) - 1)], axis=1)
    reference = (train_recon, train_knn, train_maha)
    measures = (reconstruction, knn, mahalanobis)
    z = []
    for value, ref in zip(measures, reference):
        median = float(np.median(ref)); mad = float(np.median(np.abs(ref - median))) * 1.4826
        z.append((value - median) / max(mad, float(np.std(ref)), 1e-6))
    return {"pca_reconstruction_error": reconstruction, "knn_distance": knn,
            "mahalanobis_distance": mahalanobis, "off_manifold_score": np.mean(z, axis=0)}


def add_group_metrics(rows: list[dict], base: dict, prediction: np.ndarray, truth: np.ndarray,
                      source_names: np.ndarray, genes: np.ndarray) -> None:
    rows.append(base | prediction_metrics(prediction, truth, source_names, genes))


def add_source_accuracy(rows: list[dict], base: dict, prediction: np.ndarray, truth: np.ndarray,
                        source_names: np.ndarray, genes: np.ndarray) -> None:
    correlations, errors = source_metrics(prediction, truth, source_names, genes)
    for index, source in enumerate(source_names):
        rows.append(base | {"source": source, "source_pearson": correlations[index],
                            "source_mse": errors[index]})


def pair_record(store: dict, family: str, split_index: int, test: np.ndarray, direct: np.ndarray,
                truth: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    first, second = np.triu_indices(len(test), 1)
    store[family].append({"split_index": split_index, "test": test.copy(),
                          "pair_sources": (test[first], test[second]),
                          "truth": strict_trans_cosine_distance(truth, direct)[first, second],
                          "predictions": {key: strict_trans_cosine_distance(value, direct)[first, second]
                                          for key, value in predictions.items()}})


def global_source_bootstrap(pair_store: dict, source_count: int, resamples: int, seed: int,
                            comparisons: list[tuple[str, str, str]]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(source_count, np.full(source_count, 1 / source_count), size=resamples)
    output = []
    for family, left_name, right_name in comparisons:
        left_accum = np.zeros(resamples); right_accum = np.zeros(resamples); groups = pair_store[family]
        point_left, point_right = [], []
        for record in groups:
            a, b = record["pair_sources"]
            weights = counts[:, a] * counts[:, b]
            truth = record["truth"]
            left = record["predictions"][left_name]; right = record["predictions"][right_name]
            left_accum += weighted_spearman(left, truth, weights)
            right_accum += weighted_spearman(right, truth, weights)
            point_left.append(safe_spearman(left, truth, True)); point_right.append(safe_spearman(right, truth, True))
        draws = (left_accum - right_accum) / len(groups)
        output.append({"family": family, "comparison": f"{left_name} - {right_name}",
                       "left_mean_geometry_rho": float(np.mean(point_left)),
                       "right_mean_geometry_rho": float(np.mean(point_right)),
                       "point_delta": float(np.mean(point_left) - np.mean(point_right)),
                       "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
                       "resamples": resamples,
                       "bootstrap_unit": "global perturbation-source multinomial weights reused across all heldout groups",
                       "duplicate_handling": "multiplicities weight unique source pairs; duplicate draws create no artificial zero-distance pairs"})
    return pd.DataFrame(output)


def main() -> None:
    config_path = SCRIPT_ROOT / "frozen_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    required_frozen = [FROZEN_CACHE, FROZEN_SCRIPT_ROOT / "common.py", FROZEN_SCRIPT_ROOT / "parts_cde.py",
                       FROZEN_SCRIPT_ROOT / "parts_fg.py", FROZEN_SCRIPT_ROOT / "part_h.py",
                       FROZEN_RESULT_ROOT / "first_responder_ablation.csv",
                       FROZEN_RESULT_ROOT / "program_propagation.csv"]
    missing = [str(path) for path in required_frozen if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen upstream inputs: {missing}")
    provenance = {"created_at": now(), "python": sys.version, "platform": platform.platform(),
                  "frozen_inputs_read_only": True,
                  "inputs": [{"path": str(path.relative_to(Path(__file__).resolve().parents[2])),
                              "sha256": sha256(path), "bytes": path.stat().st_size} for path in required_frozen],
                  "config_sha256": sha256(config_path), "gpu_used": False,
                  "upstream_directories_modified_by_this_task": False}
    atomic_json(RESULT_ROOT / "provenance.json", provenance)
    atomic_json(RESULT_ROOT / "seeds.json", {"outer_split_seed": config["splits"]["outer_split_seed"],
                "split_seed_formula": "outer_split_seed + repeat*1009",
                "transform_seed_offsets": {"rotation": 73001, "noise": 73003, "finite_difference": 73007},
                "bootstrap_seed": config["statistics"]["bootstrap_seed"]})

    with np.load(FROZEN_CACHE, allow_pickle=False) as archive:
        sources = archive["sources"].astype(str); genes = archive["genes"].astype(str)
        waves = archive["waves"].astype(np.float32)
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    source_gene_rows = np.asarray([gene_lookup[source] for source in sources], dtype=int)
    repeats = config["splits"]["source_disjoint_repeats"]
    splits = grouped_twofold_splits(len(sources), repeats, config["splits"]["outer_split_seed"])
    alpha_grid = tuple(config["model"]["ridge_alpha_grid"])
    dims = tuple(config["model"]["program_dimensions"])
    diag = config["diagnostics"]

    anchor_rows: list[dict] = []; interpolation_rows: list[dict] = []
    amplitude_rows: list[dict] = []; program_rows: list[dict] = []
    off_rows: list[dict] = []; projection_rows: list[dict] = []
    geometry_rows: list[dict] = []; geometry_source_rows: list[dict] = []
    local_rows: list[dict] = []; sensitivity_rows: list[dict] = []
    pair_store = defaultdict(list)

    for split_index, split in enumerate(splits):
        train, test = split["train"], split["test"]
        test_sources = sources[test]; direct = direct_indices(test_sources, genes)
        common = {"repeat": split["repeat"], "group": split["group"], "split_seed": split["seed"],
                  "n_train_sources": len(train), "n_test_sources": len(test),
                  "one_model_for_entire_heldout_group": True, "outer_test_used_for_selection": False}

        representation = transmission_representations(waves, sources, genes, train)
        feature = representation[source_gene_rows]
        predicted_all, first_alpha = standardized_ridge(feature[train], waves[train, 0], feature, alpha_grid)
        predicted_w23 = predicted_all[test]; train_predicted_w23 = predicted_all[train]
        transition, transition_alpha = fit_dense_transition(waves, train, split["seed"], alpha_grid)
        common |= {"first_wave_alpha_training_only": first_alpha,
                   "transition_alpha_training_only": transition_alpha}
        propagate = lambda state: apply_affine(transition, state)

        true_w23, true_w34, true_w45 = waves[test, 0], waves[test, 1], waves[test, 2]
        true_entry_w34 = propagate(true_w23); true_entry_w45 = propagate(true_entry_w34)
        predicted_entry_w34 = propagate(predicted_w23); predicted_entry_w45 = propagate(predicted_entry_w34)
        teacher_w45 = propagate(true_w34)
        anchors = {
            "A_TRUE_ENTRY": (("W23", true_w23, true_w23), ("W34", true_entry_w34, true_w34),
                             ("W45", true_entry_w45, true_w45)),
            "B_PREDICTED_ENTRY": (("W23", predicted_w23, true_w23), ("W34", predicted_entry_w34, true_w34),
                                  ("W45", predicted_entry_w45, true_w45)),
            "C_TEACHER_FORCED_SECOND_STEP": (("W23", predicted_w23, true_w23), ("W34", true_w34, true_w34),
                                             ("W45", teacher_w45, true_w45)),
            "D_FULL_ORACLE_REFERENCE": (("W23", true_w23, true_w23), ("W34", true_w34, true_w34),
                                        ("W45", teacher_w45, true_w45))}
        for anchor, stages in anchors.items():
            for stage, prediction, truth in stages:
                add_group_metrics(anchor_rows, common | {"anchor": anchor, "stage": stage,
                                  "oracle_diagnostic": anchor != "B_PREDICTED_ENTRY"}, prediction, truth,
                                  test_sources, genes)
        pair_record(pair_store, "anchor_w45", split_index, test, direct, true_w45,
                    {"TrueEntry": true_entry_w45, "PredictedEntry": predicted_entry_w45,
                     "TeacherForced": teacher_w45})

        manifold_mean, manifold_components = pca_basis(waves[train, 0], diag["manifold_pca_dimension"])
        train_program_mean, program_components, selected_dim, program_scores = select_program_basis(
            waves, train, split["seed"], dims, alpha_grid)

        for alpha in diag["interpolation_alpha"]:
            state = ((1 - alpha) * true_w23 + alpha * predicted_w23).astype(np.float32)
            out34 = propagate(state); out45 = propagate(out34)
            manifold = manifold_measures(state, waves[train, 0], manifold_mean, manifold_components,
                                         diag["knn_neighbors"])
            for stage, prediction, truth in (("W23", state, true_w23), ("W34", out34, true_w34),
                                             ("W45", out45, true_w45)):
                add_group_metrics(interpolation_rows, common | {"entry_alpha_predicted": alpha, "stage": stage,
                                  "oracle_interpolation_diagnostic": alpha < 1,
                                  "mean_off_manifold_score": float(np.mean(manifold["off_manifold_score"])),
                                  "mean_pca_reconstruction_error": float(np.mean(manifold["pca_reconstruction_error"]))},
                                  prediction, truth, test_sources, genes)

        pred_unit, pred_norm = unit_rows(predicted_w23); true_unit, true_norm = unit_rows(true_w23)
        norm_matched = quantile_norm_match(predicted_w23, train_predicted_w23, waves[train, 0])
        train_median_norm = float(np.median(np.linalg.norm(waves[train, 0], axis=1)))
        amplitude_states = {
            "RawPrediction": predicted_w23,
            "TrueDirection_PredictedMagnitude": true_unit * pred_norm[:, None],
            "PredictedDirection_TrueMagnitude": pred_unit * true_norm[:, None],
            "TrainQuantileNormMatchedPrediction": norm_matched,
            "DirectionNormalizedPrediction_TrainMedianMagnitude": pred_unit * train_median_norm,
            "TrueDirection_TrainMedianMagnitude": true_unit * train_median_norm}
        amplitude_outputs = {}
        for variant, state in amplitude_states.items():
            out34 = propagate(state); out45 = propagate(out34); amplitude_outputs[variant] = out45
            for stage, prediction, truth in (("W23", state, true_w23), ("W34", out34, true_w34),
                                             ("W45", out45, true_w45)):
                add_group_metrics(amplitude_rows, common | {"variant": variant, "stage": stage,
                                  "uses_heldout_true_magnitude": variant == "PredictedDirection_TrueMagnitude",
                                  "uses_heldout_true_direction": variant.startswith("TrueDirection")},
                                  prediction.astype(np.float32), truth, test_sources, genes)
        pair_record(pair_store, "amplitude_w45", split_index, test, direct, true_w45, amplitude_outputs)

        def components(state):
            centered = state - train_program_mean
            program = centered @ program_components @ program_components.T
            residual = centered - program
            return program, residual
        true_program, true_residual = components(true_w23)
        pred_program, pred_residual = components(predicted_w23)
        program_states = {
            "RawPrediction": predicted_w23,
            "TrueProgram_PredictedResidual": train_program_mean + true_program + pred_residual,
            "PredictedProgram_TrueResidual": train_program_mean + pred_program + true_residual,
            "PredictedProgramOnly": train_program_mean + pred_program,
            "TrueProgramOnly": train_program_mean + true_program,
            "PredictedResidualOnly": train_program_mean + pred_residual}
        program_outputs = {}
        for variant, state in program_states.items():
            out34 = propagate(state); out45 = propagate(out34); program_outputs[variant] = out45
            for stage, prediction, truth in (("W23", state, true_w23), ("W34", out34, true_w34),
                                             ("W45", out45, true_w45)):
                add_group_metrics(program_rows, common | {"variant": variant, "stage": stage,
                                  "selected_program_dimension_training_only": selected_dim,
                                  "program_selection_scores_json": json.dumps(program_scores, sort_keys=True),
                                  "program_basis_fit_outer_training_sources_only": True,
                                  "oracle_component_swap": variant.startswith("True") or "TrueResidual" in variant},
                                  prediction.astype(np.float32), truth, test_sources, genes)
        pair_record(pair_store, "program_w45", split_index, test, direct, true_w45, program_outputs)

        # Off-manifold diagnostics and per-source downstream losses.
        predicted_manifold = manifold_measures(predicted_w23, waves[train, 0], manifold_mean,
                                               manifold_components, diag["knn_neighbors"])
        true_manifold = manifold_measures(true_w23, waves[train, 0], manifold_mean,
                                          manifold_components, diag["knn_neighbors"])
        w34_corr, w34_mse = source_metrics(predicted_entry_w34, true_w34, test_sources, genes)
        w45_corr, w45_mse = source_metrics(predicted_entry_w45, true_w45, test_sources, genes)
        for state_name, measures in (("PredictedW23", predicted_manifold), ("TrueHeldoutW23", true_manifold)):
            for local_index, source in enumerate(test_sources):
                off_rows.append(common | {"source": source, "state": state_name,
                    "pca_dimension_training_only": manifold_components.shape[1],
                    "pca_reconstruction_error": measures["pca_reconstruction_error"][local_index],
                    "knn_distance": measures["knn_distance"][local_index],
                    "mahalanobis_distance": measures["mahalanobis_distance"][local_index],
                    "off_manifold_score": measures["off_manifold_score"][local_index],
                    "raw_rollout_w34_source_pearson": w34_corr[local_index],
                    "raw_rollout_w34_source_mse": w34_mse[local_index],
                    "raw_rollout_w45_source_pearson": w45_corr[local_index],
                    "raw_rollout_w45_source_mse": w45_mse[local_index]})

        pca_state = pca_project(predicted_w23, manifold_mean, manifold_components)
        residual_shrink = (pca_state + diag["residual_shrinkage"] * (predicted_w23 - pca_state)).astype(np.float32)
        distance_to_train = euclidean_distance(np.concatenate([predicted_w23, waves[train, 0]]))[:len(test), len(test):]
        neighbor_indices = np.argsort(distance_to_train, axis=1)[:, :diag["knn_neighbors"]]
        neighbor_distances = np.take_along_axis(distance_to_train, neighbor_indices, axis=1)
        weights = 1 / np.maximum(neighbor_distances, 1e-6); weights /= weights.sum(1, keepdims=True)
        knn_state = np.sum(waves[train, 0][neighbor_indices] * weights[:, :, None], axis=1).astype(np.float32)
        program_state = (train_program_mean + pred_program).astype(np.float32)
        projection_states = {"RawPrediction": predicted_w23, "PCAProjection": pca_state,
                             "ResidualShrink25": residual_shrink, "KNN3Projection": knn_state,
                             "ProgramReconstruction": program_state}
        projection_outputs_w45 = {}; projection_outputs_w23 = {}
        for method, state in projection_states.items():
            out34 = propagate(state); out45 = propagate(out34)
            projection_outputs_w23[method] = state; projection_outputs_w45[method] = out45
            for stage, prediction, truth in (("W23", state, true_w23), ("W34", out34, true_w34),
                                             ("W45", out45, true_w45)):
                add_group_metrics(projection_rows, common | {"projection": method, "stage": stage,
                    "projection_fit_training_sources_only": method != "RawPrediction",
                    "primary_projection_preregistered": method == config["primary_projection"]},
                    prediction, truth, test_sources, genes)
        pair_record(pair_store, "projection_w23", split_index, test, direct, true_w23, projection_outputs_w23)
        pair_record(pair_store, "projection_w45", split_index, test, direct, true_w45, projection_outputs_w45)

        rng = np.random.default_rng(split["seed"] + 73001)
        q, _ = np.linalg.qr(rng.normal(size=(len(genes), len(genes))))
        noise_rng = np.random.default_rng(split["seed"] + 73003)
        noise = noise_rng.normal(size=true_w23.shape)
        noise -= np.sum(noise * true_w23, axis=1, keepdims=True) / np.maximum(np.sum(true_w23**2, axis=1, keepdims=True), 1e-12) * true_w23
        noise = noise / np.maximum(np.linalg.norm(noise, axis=1, keepdims=True), 1e-12)
        noise *= diag["orthogonal_noise_fraction"] * np.linalg.norm(true_w23, axis=1, keepdims=True)
        permutation = rng.permutation(len(genes))
        geometry_states = {"TrueRaw": true_w23, "OrthogonalRotation": true_w23 @ q,
                           "GlobalScale2": true_w23 * diag["global_scale"],
                           "OrthogonalNoise20Pct": true_w23 + noise,
                           "ConsistentGeneShuffle": true_w23[:, permutation]}
        geometry_outputs = {}
        true_strict = strict_trans_cosine_distance(true_w23, direct); true_full = full_cosine_distance(true_w23)
        true_euclidean = euclidean_distance(true_w23); upper = np.triu_indices(len(test), 1)
        for transformation, state in geometry_states.items():
            state = state.astype(np.float32); out34 = propagate(state); out45 = propagate(out34)
            geometry_outputs[transformation] = out45
            initial = prediction_metrics(state, true_w23, test_sources, genes)
            properties = {"initial_strict_trans_geometry_rho": initial["response_distance_rho"],
                "initial_full_cosine_geometry_rho": safe_spearman(full_cosine_distance(state)[upper], true_full[upper], True),
                "initial_euclidean_geometry_rho": safe_spearman(euclidean_distance(state)[upper], true_euclidean[upper], True),
                "euclidean_distance_max_abs_error": float(np.max(np.abs(euclidean_distance(state) - true_euclidean))),
                "mean_norm_ratio": float(np.mean(np.linalg.norm(state, axis=1)) / max(np.mean(np.linalg.norm(true_w23, axis=1)), 1e-12))}
            for stage, prediction, truth in (("W23", state, true_w23), ("W34", out34, true_w34),
                                             ("W45", out45, true_w45)):
                add_group_metrics(geometry_rows, common | {"transformation": transformation, "stage": stage, **properties},
                                  prediction, truth, test_sources, genes)
                add_source_accuracy(geometry_source_rows, common | {"transformation": transformation, "stage": stage},
                                    prediction, truth, test_sources, genes)
        pair_record(pair_store, "geometry_w45", split_index, test, direct, true_w45, geometry_outputs)

        # Local geometry preservation within the same heldout group.
        true_distance = euclidean_distance(true_w23); pred_distance = euclidean_distance(predicted_w23)
        k = min(diag["knn_neighbors"], len(test) - 1)
        for local_index, source in enumerate(test_sources):
            candidates = np.arange(len(test)) != local_index
            candidate_indices = np.flatnonzero(candidates)
            true_order = candidate_indices[np.argsort(true_distance[local_index, candidates])]
            pred_order = candidate_indices[np.argsort(pred_distance[local_index, candidates])]
            overlap = len(set(true_order[:k]).intersection(pred_order[:k])) / k
            rank_corr = safe_spearman(true_distance[local_index, candidates], pred_distance[local_index, candidates])
            near = true_order[:k]
            distortion = float(np.mean(np.abs(np.log((pred_distance[local_index, near] + 1e-8) /
                                                       (true_distance[local_index, near] + 1e-8)))))
            local_rows.append(common | {"source": source, "k": k, "knn_overlap": overlap,
                "nearest_neighbor_distance_rank_rho": rank_corr, "local_log_distance_distortion": distortion,
                "raw_rollout_w45_source_pearson": w45_corr[local_index],
                "raw_rollout_w45_source_mse": w45_mse[local_index]})

        # The frozen dense propagator is affine: the Jacobian is exactly state-independent.
        x_mean, x_scale, y_mean, coefficient = transition
        jacobian = coefficient / x_scale[:, None]
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        left_vectors = np.linalg.svd(jacobian, full_matrices=False)[0]
        jacobian_two = jacobian @ jacobian
        spectral_one = float(singular_values[0]); spectral_two = float(np.linalg.svd(jacobian_two, compute_uv=False)[0])
        epsilon = diag["finite_difference_relative_epsilon"] * float(np.median(np.linalg.norm(waves[train, 0], axis=1)))
        fd_rng = np.random.default_rng(split["seed"] + 73007)
        finite_directions = fd_rng.normal(size=(len(test), len(genes)))
        finite_directions /= np.maximum(np.linalg.norm(finite_directions, axis=1, keepdims=True), 1e-12)
        for state_name, states in (("TrueHeldoutW23", true_w23), ("PredictedW23", predicted_w23)):
            for local_index, source in enumerate(test_sources):
                # Identical perturbation direction for true and predicted states.  For this affine
                # frozen operator, any residual difference should be numerical roundoff only.
                direction = finite_directions[local_index]
                finite_one = np.linalg.norm(propagate(states[local_index:local_index+1] + epsilon * direction) -
                                            propagate(states[local_index:local_index+1])) / epsilon
                first = propagate(states[local_index:local_index+1])
                finite_two = np.linalg.norm(propagate(propagate(states[local_index:local_index+1] + epsilon * direction)) -
                                            propagate(first)) / epsilon
                error = predicted_w23[local_index] - true_w23[local_index]
                error_norm = max(float(np.linalg.norm(error)), 1e-12)
                actual_one = float(np.linalg.norm(predicted_entry_w34[local_index] - true_entry_w34[local_index]) / error_norm)
                actual_two = float(np.linalg.norm(predicted_entry_w45[local_index] - true_entry_w45[local_index]) / error_norm)
                sensitivity_rows.append(common | {"source": source, "state": state_name,
                    "finite_difference_epsilon": epsilon, "finite_difference_one_step_amplification": finite_one,
                    "finite_difference_two_step_amplification": finite_two,
                    "jacobian_spectral_norm_one_step": spectral_one,
                    "jacobian_spectral_norm_two_step": spectral_two,
                    "actual_entry_error_one_step_amplification": actual_one,
                    "actual_entry_error_two_step_amplification": actual_two,
                    "entry_error_cosine_top_sensitive_direction": float(abs(error @ left_vectors[:, 0]) / error_norm),
                    "affine_jacobian_state_independent": True})

        if (split_index + 1) % 10 == 0:
            print(f"[Dynamic validity] Completed grouped folds {split_index + 1}/{len(splits)}", flush=True)

    tables = {"rollout_anchor_decomposition.csv": anchor_rows,
              "first_wave_interpolation_curve.csv": interpolation_rows,
              "first_wave_amplitude_direction_decomposition.csv": amplitude_rows,
              "program_component_rollout.csv": program_rows,
              "off_manifold_diagnostics.csv": off_rows,
              "manifold_projection_rescue.csv": projection_rows,
              "geometry_vs_dynamic_validity.csv": geometry_rows,
              "geometry_dynamic_source_accuracy.csv": geometry_source_rows,
              "local_geometry_rollout.csv": local_rows,
              "propagator_sensitivity.csv": sensitivity_rows}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(RESULT_ROOT / name, index=False)

    comparisons = [
        ("anchor_w45", "TrueEntry", "PredictedEntry"),
        ("anchor_w45", "TeacherForced", "PredictedEntry"),
        ("amplitude_w45", "PredictedDirection_TrueMagnitude", "RawPrediction"),
        ("amplitude_w45", "TrueDirection_PredictedMagnitude", "RawPrediction"),
        ("amplitude_w45", "TrainQuantileNormMatchedPrediction", "RawPrediction"),
        ("program_w45", "TrueProgram_PredictedResidual", "RawPrediction"),
        ("program_w45", "PredictedProgram_TrueResidual", "RawPrediction"),
        ("projection_w23", "PCAProjection", "RawPrediction"),
        ("projection_w45", "PCAProjection", "RawPrediction"),
        ("projection_w45", "ResidualShrink25", "RawPrediction"),
        ("projection_w45", "KNN3Projection", "RawPrediction"),
        ("projection_w45", "ProgramReconstruction", "RawPrediction"),
        ("geometry_w45", "TrueRaw", "OrthogonalRotation"),
        ("geometry_w45", "TrueRaw", "GlobalScale2"),
        ("geometry_w45", "TrueRaw", "ConsistentGeneShuffle")]
    bootstrap = global_source_bootstrap(pair_store, len(sources), config["statistics"]["bootstrap_resamples"],
                                        config["statistics"]["bootstrap_seed"], comparisons)
    bootstrap.to_csv(RESULT_ROOT / "bootstrap_geometry_contrasts.csv", index=False)
    atomic_json(RESULT_ROOT / "run_complete.json", {"completed_at": now(), "groups": len(splits),
        "sources": len(sources), "genes": len(genes), "tables": {name: len(rows) for name, rows in tables.items()},
        "bootstrap_rows": len(bootstrap), "gpu_used": False, "new_architecture_trained": False})
    print("[Dynamic validity] B2-B10 raw results and source bootstrap are complete.", flush=True)


if __name__ == "__main__":
    main()
