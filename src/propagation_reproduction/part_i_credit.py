from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNet

from common import (CACHE_ROOT, RESULT_ROOT, atomic_json, grouped_twofold_splits, now,
                    prediction_metrics)
from parts_cde import standardized_ridge, transmission_representations
from parts_fg import choose_model, fit_rrr, transitions


ELASTIC_GRID = ((.001, .2), (.001, .5), (.01, .2), (.01, .5), (.1, .2), (.1, .5))


def elastic_fit(x: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: float) -> np.ndarray:
    mean = x.mean(0); scale = np.maximum(x.std(0), 1e-6)
    model = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True,
                                max_iter=1000, tol=1e-3, selection="cyclic")
    model.fit((x - mean) / scale, y)
    return (model.coef_.T / scale[:, None]).astype(np.float32)


def select_elastic(waves: np.ndarray, train: np.ndarray, seed: int,
                   mean: np.ndarray, components: np.ndarray) -> tuple[float, float]:
    order = np.random.default_rng(seed + 61051).permutation(train)
    count = max(2, int(np.ceil(.20 * len(order)))); val, fit = order[:count], order[count:]
    train_x, train_y = transitions(waves, fit); val_x, val_y = transitions(waves, val)
    train_y = (train_y - mean) @ components; val_y = (val_y - mean) @ components
    mean, scale = train_x.mean(0), np.maximum(train_x.std(0), 1e-6)
    best = (float("inf"), .01, .2)
    for alpha, ratio in ELASTIC_GRID:
        model = MultiTaskElasticNet(alpha=alpha, l1_ratio=ratio, fit_intercept=True,
                                    max_iter=1000, tol=1e-3, selection="cyclic")
        model.fit((train_x - mean) / scale, train_y)
        score = float(np.mean((model.predict((val_x - mean) / scale) - val_y) ** 2))
        if score < best[0]: best = (score, alpha, ratio)
    return float(best[1]), float(best[2])


def credit_representation(method: str, waves: np.ndarray, sources: np.ndarray, genes: np.ndarray,
                          train: np.ndarray, dense_alpha: float, lowrank_alpha: float,
                          lowrank_rank: int, elastic: tuple[float, float],
                          program_mean: np.ndarray, program_components: np.ndarray) -> np.ndarray:
    output_dimension = program_components.shape[1] if method == "SparseElasticNetProgramCredit" else len(genes)
    output = np.zeros((len(genes), output_dimension), dtype=np.float32)
    source_lookup = {source: index for index, source in enumerate(sources)}
    for gene_index, gene in enumerate(genes):
        fit_rows = train[train != source_lookup.get(gene, -1)]
        x, y = transitions(waves, fit_rows)
        if method == "DenseRidgeCredit":
            x_mean, x_scale, _, coefficient = fit_rrr(x, y, dense_alpha, None)
            coefficient = coefficient / x_scale[:, None]
        elif method == "LowRankCredit":
            x_mean, x_scale, _, coefficient = fit_rrr(x, y, lowrank_alpha, lowrank_rank)
            coefficient = coefficient / x_scale[:, None]
        elif method == "SparseElasticNetProgramCredit":
            coefficient = elastic_fit(x, (y - program_mean) @ program_components, *elastic)
        else:
            raise ValueError(method)
        output[gene_index] = coefficient[gene_index]
    return output


def main() -> None:
    config = json.loads(Path(__file__).with_name("frozen_config.json").read_text())
    with np.load(CACHE_ROOT / "renge_processed.npz", allow_pickle=False) as archive:
        sources = archive["sources"].astype(str); genes = archive["genes"].astype(str)
        waves = archive["waves"].astype(np.float32); static = archive["static_control_representation"].astype(np.float32)
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    source_gene_rows = np.asarray([gene_lookup[source] for source in sources])
    splits = grouped_twofold_splits(len(sources), config["source_disjoint_repeats"], config["outer_split_seed"])
    methods = ("MarginalLag", "DenseRidgeCredit", "LowRankCredit", "SparseElasticNetProgramCredit", "StaticControl")
    rows = []
    for split_index, split in enumerate(splits):
        train, test = split["train"], split["test"]
        dense_alpha, _ = choose_model(waves, train, split["seed"], dense=True)
        lowrank_alpha, lowrank_rank = choose_model(waves, train, split["seed"], dense=False)
        program_matrix = waves[train].reshape(-1, len(genes)).astype(np.float64)
        program_mean = program_matrix.mean(0); _, _, program_vt = np.linalg.svd(program_matrix - program_mean, full_matrices=False)
        program_components = program_vt[:10].T
        elastic = select_elastic(waves, train, split["seed"], program_mean, program_components)
        marginal = transmission_representations(waves, sources, genes, train, "correct_lag", split["seed"])
        representations = {"MarginalLag": marginal, "StaticControl": static}
        for method in methods[1:4]:
            representations[method] = credit_representation(method, waves, sources, genes, train,
                                                             dense_alpha, lowrank_alpha, lowrank_rank, elastic,
                                                             program_mean, program_components)
        for method in methods:
            feature = representations[method][source_gene_rows]
            prediction, alpha = standardized_ridge(feature[train], waves[train, 0], feature[test],
                                                    tuple(config["ridge_alpha_grid"]))
            rows.append({"repeat": split["repeat"], "group": split["group"], "model": method,
                         "target": "heldout_W23", "mapping_alpha_training_only": alpha,
                         "transition_dense_alpha_training_only": dense_alpha,
                         "transition_lowrank_alpha_training_only": lowrank_alpha,
                         "transition_lowrank_rank_training_only": lowrank_rank,
                         "elastic_alpha_training_only": elastic[0], "elastic_l1_ratio_training_only": elastic[1],
                         "elastic_output_program_dimension": 10 if method == "SparseElasticNetProgramCredit" else np.nan,
                         "heldout_source_absent_all_days": True, "n_train_sources": len(train),
                         "n_test_sources": len(test), **prediction_metrics(prediction, waves[test, 0], sources[test], genes)})
        if (split_index + 1) % 10 == 0:
            print(f"[传播复现] Part I credit baselines {split_index + 1}/{len(splits)}", flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(RESULT_ROOT / "credit_head_results.csv", index=False)
    summary = frame.groupby("model").response_distance_rho.agg(["mean", "median", "std"])
    best_credit = summary.drop(index=["MarginalLag", "StaticControl"])["mean"].idxmax()
    improvement = float(summary.loc[best_credit, "mean"] - summary.loc["MarginalLag", "mean"])
    threshold = config["early_gate"]["minimum_credit_head_minus_marginal_geometry_rho"]
    decision = "ALLOW_TINY_DYNAMIC_CREDIT_ATTENTION" if improvement >= threshold else "STOP_BEFORE_GPU_CREDIT_ATTENTION"
    audit = {"created_at": now(), "decision": decision, "best_credit_head": best_credit,
             "best_credit_minus_marginal_geometry_rho": improvement, "frozen_threshold": threshold,
             "summary": summary.reset_index().to_dict("records"),
             "attention_claim_boundary": "No attention weights have been trained; conditional transition coefficients are predictive credit baselines, not causal edges."}
    atomic_json(RESULT_ROOT / "credit_head_gate.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
