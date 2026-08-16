from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from common import (CACHE_ROOT, RESULT_ROOT, grouped_twofold_splits, prediction_metrics,
                    ridge_fit_predict, select_ridge_alpha)
from parts_cde import standardized_ridge, transmission_representations


RANKS = (2, 4, 8, 10, 16)
ALPHAS = (.01, .1, 1., 10., 100.)


def fit_rrr(x: np.ndarray, y: np.ndarray, alpha: float, rank: int | None):
    x_mean, x_scale = x.mean(0), np.maximum(x.std(0), 1e-6); y_mean = y.mean(0)
    xc = (x - x_mean) / x_scale; yc = y - y_mean
    coefficient = xc.T @ np.linalg.solve(xc @ xc.T + alpha * np.eye(len(xc)), yc)
    if rank is not None:
        _, _, vt = np.linalg.svd(xc @ coefficient, full_matrices=False)
        projection = vt[:min(rank, len(vt))].T @ vt[:min(rank, len(vt))]
        coefficient = coefficient @ projection
    return x_mean, x_scale, y_mean, coefficient


def apply_rrr(model, query: np.ndarray) -> np.ndarray:
    x_mean, x_scale, y_mean, coefficient = model
    return (((query - x_mean) / x_scale) @ coefficient + y_mean).astype(np.float32)


def transitions(waves: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([waves[rows, 0], waves[rows, 1]]), np.concatenate([waves[rows, 1], waves[rows, 2]])


def choose_model(waves: np.ndarray, train: np.ndarray, split_seed: int, dense: bool = False) -> tuple[float, int | None]:
    order = np.random.default_rng(split_seed + 41011).permutation(train)
    count = max(2, int(np.ceil(.20 * len(order)))); inner_val, inner_train = order[:count], order[count:]
    train_x, train_y = transitions(waves, inner_train); val_x, val_y = transitions(waves, inner_val)
    best = (float("inf"), 1., None)
    ranks = (None,) if dense else RANKS
    for alpha in ALPHAS:
        for rank in ranks:
            prediction = apply_rrr(fit_rrr(train_x, train_y, alpha, rank), val_x)
            score = float(np.mean((prediction - val_y) ** 2))
            if score < best[0]: best = (score, alpha, rank)
    return float(best[1]), best[2]


def main() -> None:
    config = json.loads(Path(__file__).with_name("frozen_config.json").read_text())
    gate = json.loads((RESULT_ROOT / "parts_cde_gate.json").read_text())
    if gate["decision"] != "PROCEED_TO_PART_F": raise RuntimeError("Parts C-E gate did not authorize Part F")
    with np.load(CACHE_ROOT / "renge_processed.npz", allow_pickle=False) as archive:
        sources = archive["sources"].astype(str); genes = archive["genes"].astype(str)
        waves = archive["waves"].astype(np.float32); static = archive["static_control_representation"].astype(np.float32)
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    source_gene_rows = np.asarray([gene_lookup[source] for source in sources])
    splits = grouped_twofold_splits(len(sources), config["source_disjoint_repeats"], config["outer_split_seed"])
    operator_rows, alignment_rows = [], []
    for split_index, split in enumerate(splits):
        train, test = split["train"], split["test"]
        train_x, train_y = transitions(waves, train)
        alpha, rank = choose_model(waves, train, split["seed"], dense=False)
        correct_model = fit_rrr(train_x, train_y, alpha, rank)
        dense_alpha, _ = choose_model(waves, train, split["seed"], dense=True)
        dense_model = fit_rrr(train_x, train_y, dense_alpha, None)
        rng = np.random.default_rng(split["seed"] + 42013)
        shuffled_y = train_y.copy()
        for stage in (0, 1):
            loc = np.arange(stage * len(train), (stage + 1) * len(train)); shuffled_y[loc] = train_y[rng.permutation(loc)]
        shuffle_model = fit_rrr(train_x, shuffled_y, alpha, rank)
        same_model = fit_rrr(train_x, train_x, alpha, rank)
        permutation = rng.permutation(len(genes))
        for target_name, current, truth in (("W34", waves[test, 0], waves[test, 1]),
                                            ("W45", waves[test, 1], waves[test, 2])):
            predictions = {"LowRankCorrectLag": apply_rrr(correct_model, current),
                           "DenseCorrectLag": apply_rrr(dense_model, current),
                           "LowRankSameWave": apply_rrr(same_model, current),
                           "LowRankTemporalShuffle": apply_rrr(shuffle_model, current),
                           "LowRankGeneIdentityShuffle": apply_rrr(correct_model, current[:, permutation])}
            for model_name, prediction in predictions.items():
                metrics = prediction_metrics(prediction, truth, sources[test], genes)
                operator_rows.append({"repeat": split["repeat"], "group": split["group"], "model": model_name,
                                      "target": target_name, "selected_rank_training_only": rank if "Dense" not in model_name else len(genes),
                                      "selected_alpha_training_only": dense_alpha if "Dense" in model_name else alpha,
                                      "n_train_sources": len(train), "n_test_sources": len(test), **metrics})

        transmission = transmission_representations(waves, sources, genes, train, "correct_lag", split["seed"])
        feature = transmission[source_gene_rows]
        train_feature, test_feature = feature[train], feature[test]
        x_mean, y_mean = train_feature.mean(0), waves[train, 0].mean(0)
        u, _, vt = np.linalg.svd((train_feature - x_mean).T @ (waves[train, 0] - y_mean), full_matrices=False)
        rotation = u @ vt
        aligned = (test_feature - x_mean) @ rotation + y_mean
        static_prediction, static_alpha = standardized_ridge(static[source_gene_rows][train], waves[train, 0],
                                                             static[source_gene_rows][test], tuple(config["ridge_alpha_grid"]))
        raw_distance = pdist(test_feature, metric="euclidean")
        aligned_distance = pdist((test_feature - x_mean) @ rotation, metric="euclidean")
        invariant_error = float(np.max(np.abs(raw_distance - aligned_distance)))
        for model_name, prediction in (("RawTransmission", test_feature), ("OrthogonalAligned", aligned),
                                       ("StaticControl", static_prediction)):
            metrics = prediction_metrics(prediction, waves[test, 0], sources[test], genes)
            alignment_rows.append({"repeat": split["repeat"], "group": split["group"], "model": model_name,
                                   "target": "heldout_W23", "alignment_fit_training_sources_only": model_name != "RawTransmission",
                                   "euclidean_geometry_invariance_max_abs_error": invariant_error if model_name == "OrthogonalAligned" else np.nan,
                                   "global_scaling_used": False, "static_alpha_training_only": static_alpha if model_name == "StaticControl" else np.nan,
                                   "n_train_sources": len(train), "n_test_sources": len(test), **metrics})
        if (split_index + 1) % 20 == 0:
            print(f"[传播复现] Part F-G {split_index + 1}/{len(splits)}", flush=True)
    operator = pd.DataFrame(operator_rows); alignment = pd.DataFrame(alignment_rows)
    operator.to_csv(RESULT_ROOT / "low_rank_operator.csv", index=False)
    alignment.to_csv(RESULT_ROOT / "alignment_results.csv", index=False)
    print("\nPart F summary\n", operator.groupby(["target", "model"]).response_distance_rho.mean().to_string())
    print("\nPart G summary\n", alignment.groupby("model")[["response_distance_rho", "per_response_strict_trans_pearson"]].mean().to_string())
    print("Max Procrustes Euclidean invariance error:", alignment.euclidean_geometry_invariance_max_abs_error.max())


if __name__ == "__main__":
    main()
