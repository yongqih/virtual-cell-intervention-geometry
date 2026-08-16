from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (CACHE_ROOT, RESULT_ROOT, grouped_twofold_splits, prediction_metrics,
                    ridge_fit_predict, select_ridge_alpha)
from parts_cde import standardized_ridge, transmission_representations
from parts_fg import apply_rrr, choose_model, fit_rrr, transitions


DIMENSIONS = (4, 8, 10, 16)


def basis(waves: np.ndarray, train: np.ndarray, dimension: int):
    matrix = waves[train].reshape(-1, waves.shape[-1]).astype(np.float64)
    mean = matrix.mean(0); _, _, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    return mean.astype(np.float32), vt[:dimension].T.astype(np.float32)


def project(matrix: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (matrix - mean) @ components


def reconstruct(scores: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return scores @ components.T + mean


def fit_program_operator(waves: np.ndarray, rows: np.ndarray, mean: np.ndarray, components: np.ndarray,
                         alpha_grid: tuple[float, ...]):
    current, following = transitions(waves, rows)
    x, y = project(current, mean, components), project(following, mean, components)
    alpha = select_ridge_alpha(x, y, alpha_grid)
    return alpha, x.mean(0), np.maximum(x.std(0), 1e-6), y.mean(0), ridge_fit_predict


def program_predict(train_current: np.ndarray, train_next: np.ndarray, query: np.ndarray,
                    alpha_grid: tuple[float, ...]) -> tuple[np.ndarray, float]:
    prediction, alpha = standardized_ridge(train_current, train_next, query, alpha_grid)
    return prediction, alpha


def main() -> None:
    config = json.loads(Path(__file__).with_name("frozen_config.json").read_text())
    with np.load(CACHE_ROOT / "renge_processed.npz", allow_pickle=False) as archive:
        sources = archive["sources"].astype(str); genes = archive["genes"].astype(str)
        waves = archive["waves"].astype(np.float32); static = archive["static_control_representation"].astype(np.float32)
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    source_gene_rows = np.asarray([gene_lookup[source] for source in sources])
    splits = grouped_twofold_splits(len(sources), config["source_disjoint_repeats"], config["outer_split_seed"])
    alpha_grid = tuple(config["ridge_alpha_grid"]); rows = []
    for split_index, split in enumerate(splits):
        train, test = split["train"], split["test"]
        # First-wave predictor is built only from other-source trajectories.
        representation = transmission_representations(waves, sources, genes, train, "correct_lag", split["seed"])
        feature = representation[source_gene_rows]
        predicted_w23, first_alpha = standardized_ridge(feature[train], waves[train, 0], feature[test], alpha_grid)
        inner_order = np.random.default_rng(split["seed"] + 57037).permutation(train)
        inner_count = max(2, int(np.ceil(.20 * len(inner_order))))
        inner_val, inner_train = inner_order[:inner_count], inner_order[inner_count:]
        inner_scores = {}
        for dimension in DIMENSIONS:
            mean, components = basis(waves, train, dimension)
            inner_x_raw, inner_y_raw = transitions(waves, inner_train)
            val_x_raw, val_y_raw = transitions(waves, inner_val)
            inner_x = project(inner_x_raw, mean, components); inner_y = project(inner_y_raw, mean, components)
            val_x = project(val_x_raw, mean, components); val_y = project(val_y_raw, mean, components)
            val_prediction, _ = program_predict(inner_x, inner_y, val_x, alpha_grid)
            inner_scores[dimension] = float(np.mean((val_prediction - val_y) ** 2))
        selected_dimension = min(inner_scores, key=inner_scores.get)
        for dimension in DIMENSIONS:
            mean, components = basis(waves, train, dimension)
            train_current_raw, train_next_raw = transitions(waves, train)
            train_current = project(train_current_raw, mean, components)
            train_next = project(train_next_raw, mean, components)
            true_w23_z = project(waves[test, 0], mean, components)
            true_w34_z = project(waves[test, 1], mean, components)
            predicted_w23_z = project(predicted_w23, mean, components)
            predicted_w34_z, alpha = program_predict(train_current, train_next, true_w23_z, alpha_grid)
            teacher_w45_z, _ = program_predict(train_current, train_next, true_w34_z, (alpha,))
            free_true_w45_z, _ = program_predict(train_current, train_next, predicted_w34_z, (alpha,))
            free_pred_w34_z, _ = program_predict(train_current, train_next, predicted_w23_z, (alpha,))
            free_pred_w45_z, _ = program_predict(train_current, train_next, free_pred_w34_z, (alpha,))
            evaluations = (("OracleTrueW23_to_W34", reconstruct(predicted_w34_z, mean, components), waves[test, 1]),
                           ("TeacherForcedTrueW34_to_W45", reconstruct(teacher_w45_z, mean, components), waves[test, 2]),
                           ("FreeTrueW23_to_W45", reconstruct(free_true_w45_z, mean, components), waves[test, 2]),
                           ("FreePredictedW23_to_W34", reconstruct(free_pred_w34_z, mean, components), waves[test, 1]),
                           ("FreePredictedW23_to_W45", reconstruct(free_pred_w45_z, mean, components), waves[test, 2]))
            for mode, prediction, truth in evaluations:
                metrics = prediction_metrics(prediction, truth, sources[test], genes)
                rows.append({"repeat": split["repeat"], "group": split["group"], "model": "ProgramRidge",
                             "dimension": dimension, "selected_dimension_training_only": selected_dimension,
                             "is_selected_dimension": dimension == selected_dimension, "mode": mode,
                             "transition_alpha_training_only": alpha, "first_wave_alpha_training_only": first_alpha,
                             "basis_fit_outer_training_sources_only": True, "full_free_rollout": mode.startswith("FreePredicted"),
                             "n_train_sources": len(train), "n_test_sources": len(test), **metrics})
        # Gene-level dense comparator with the same true/predicted first waves.
        dense_alpha, _ = choose_model(waves, train, split["seed"], dense=True)
        train_x, train_y = transitions(waves, train); dense = fit_rrr(train_x, train_y, dense_alpha, None)
        dense_w34 = apply_rrr(dense, waves[test, 0]); dense_teacher_w45 = apply_rrr(dense, waves[test, 1])
        dense_free_true_w45 = apply_rrr(dense, dense_w34)
        dense_pred_w34 = apply_rrr(dense, predicted_w23); dense_pred_w45 = apply_rrr(dense, dense_pred_w34)
        for mode, prediction, truth in (("OracleTrueW23_to_W34", dense_w34, waves[test, 1]),
                                        ("TeacherForcedTrueW34_to_W45", dense_teacher_w45, waves[test, 2]),
                                        ("FreeTrueW23_to_W45", dense_free_true_w45, waves[test, 2]),
                                        ("FreePredictedW23_to_W34", dense_pred_w34, waves[test, 1]),
                                        ("FreePredictedW23_to_W45", dense_pred_w45, waves[test, 2])):
            rows.append({"repeat": split["repeat"], "group": split["group"], "model": "GeneLevelDense",
                         "dimension": len(genes), "selected_dimension_training_only": selected_dimension,
                         "is_selected_dimension": False, "mode": mode, "transition_alpha_training_only": dense_alpha,
                         "first_wave_alpha_training_only": first_alpha, "basis_fit_outer_training_sources_only": True,
                         "full_free_rollout": mode.startswith("FreePredicted"), "n_train_sources": len(train),
                         "n_test_sources": len(test), **prediction_metrics(prediction, truth, sources[test], genes)})
        if (split_index + 1) % 20 == 0:
            print(f"[传播复现] Part H {split_index + 1}/{len(splits)}", flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(RESULT_ROOT / "program_propagation.csv", index=False)
    selected = frame[(frame.model == "ProgramRidge") & frame.is_selected_dimension]
    summary = pd.concat([selected, frame[frame.model == "GeneLevelDense"]]).groupby(["model", "mode"]).response_distance_rho.mean()
    print(summary.to_string())


if __name__ == "__main__":
    main()
