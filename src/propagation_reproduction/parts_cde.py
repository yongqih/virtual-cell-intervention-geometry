from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (CACHE_ROOT, RESULT_ROOT, atomic_json, grouped_twofold_splits, now,
                    prediction_metrics, ridge_fit_predict, select_ridge_alpha)


def standardized_ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
                       grid: tuple[float, ...]) -> tuple[np.ndarray, float]:
    mean = train_x.mean(0); scale = np.maximum(train_x.std(0), 1e-6)
    x = (train_x - mean) / scale; query = (test_x - mean) / scale
    alpha = select_ridge_alpha(x, train_y, grid)
    return ridge_fit_predict(x, train_y, query, alpha), alpha


def transmission_representations(waves: np.ndarray, sources: np.ndarray, genes: np.ndarray,
                                 train: np.ndarray, mode: str, seed: int,
                                 evidence_fraction: float = 1.0) -> np.ndarray:
    """Gene x response-gene coefficients built only from outer-training perturbations."""
    rng = np.random.default_rng(seed)
    source_lookup = {name: index for index, name in enumerate(sources)}
    train_sources = sources[train]
    current = np.concatenate([waves[train, 0], waves[train, 1]], axis=0)
    next_wave = np.concatenate([waves[train, 1], waves[train, 2]], axis=0)
    same_wave = current.copy()
    row_sources = np.concatenate([train_sources, train_sources])
    stage = np.concatenate([np.zeros(len(train), int), np.ones(len(train), int)])
    if mode == "temporal_shuffle":
        shuffled = next_wave.copy()
        for transition in (0, 1):
            loc = np.flatnonzero(stage == transition)
            shuffled[loc] = next_wave[rng.permutation(loc)]
        target = shuffled
    elif mode == "same_wave":
        target = same_wave
    else:
        target = next_wave
    output = np.zeros((len(genes), len(genes)), dtype=np.float32)
    for gene_index, gene in enumerate(genes):
        eligible_sources = np.asarray([name for name in train_sources if name != gene], dtype=str)
        if evidence_fraction <= 0 or not len(eligible_sources):
            continue
        take = len(eligible_sources) if evidence_fraction >= 1 else max(1, int(np.floor(evidence_fraction * len(eligible_sources))))
        selected_sources = eligible_sources if take == len(eligible_sources) else rng.choice(eligible_sources, take, replace=False)
        selected = np.isin(row_sources, selected_sources)
        x = current[selected, gene_index].astype(np.float64); y = target[selected].astype(np.float64)
        x_centered = x - x.mean(); y_centered = y - y.mean(0, keepdims=True)
        denominator = float(x_centered @ x_centered) + 1e-4
        output[gene_index] = (x_centered @ y_centered / denominator).astype(np.float32)
    return output


def representation_for_condition(condition: str, waves: np.ndarray, sources: np.ndarray, genes: np.ndarray,
                                 static: np.ndarray, train: np.ndarray, seed: int) -> np.ndarray:
    if condition == "StaticControl": return static
    correct = transmission_representations(waves, sources, genes, train, "correct_lag", seed)
    if condition == "CorrectLag": return correct
    if condition == "SameWave": return transmission_representations(waves, sources, genes, train, "same_wave", seed)
    if condition == "TemporalShuffle": return transmission_representations(waves, sources, genes, train, "temporal_shuffle", seed)
    if condition == "GeneIdentityShuffle":
        return correct[np.random.default_rng(seed + 991).permutation(len(genes))]
    if condition == "IntermediateMasked": return np.zeros_like(correct)
    raise ValueError(condition)


def add_common(row: dict, split: dict, model: str, target: str, alpha: float,
               train_count: int, test_count: int) -> dict:
    return {"repeat": split["repeat"], "group": split["group"], "split_seed": split["seed"],
            "model": model, "target": target, "selected_alpha_training_only": alpha,
            "n_train_sources": train_count, "n_test_sources": test_count,
            "one_model_for_entire_heldout_group": True, "outer_test_used_for_selection": False, **row}


def main() -> None:
    config = json.loads(Path(__file__).with_name("frozen_config.json").read_text())
    gate = json.loads((RESULT_ROOT / "parts_ab_gate.json").read_text())
    if not gate["parts_cde_allowed"]: raise RuntimeError("Parts A-B reliability gate failed")
    with np.load(CACHE_ROOT / "renge_processed.npz", allow_pickle=False) as archive:
        sources = archive["sources"].astype(str); genes = archive["genes"].astype(str)
        response = archive["response"].astype(np.float32); waves = archive["waves"].astype(np.float32)
        static_all_genes = archive["static_control_representation"].astype(np.float32)
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    source_gene_rows = np.asarray([gene_lookup[source] for source in sources], dtype=np.int64)
    splits = grouped_twofold_splits(len(sources), config["source_disjoint_repeats"], config["outer_split_seed"])
    grid = tuple(config["ridge_alpha_grid"])

    markov_rows = []
    markov_tasks = (("W23_to_W34", waves[:, 0], waves[:, 1]),
                    ("W34_to_W45", waves[:, 1], waves[:, 2]))
    for split in splits:
        train, test = split["train"], split["test"]
        for task, immediate, target in markov_tasks:
            candidates = {"ImmediatePrecedingWaveOracle": immediate,
                          "EarlyR_Day2_Oracle": response[:, 0],
                          "StaticControl": static_all_genes[source_gene_rows]}
            if task == "W34_to_W45": candidates["EarlyW23_Oracle"] = waves[:, 0]
            for model, features in candidates.items():
                prediction, alpha = standardized_ridge(features[train], target[train], features[test], grid)
                metrics = prediction_metrics(prediction, target[test], sources[test], genes)
                markov_rows.append(add_common(metrics, split, model, task, alpha, len(train), len(test)) |
                                   {"interpretation_scope": "oracle temporal-dependence diagnostic; not zero-shot when a true held-out wave is supplied"})
    pd.DataFrame(markov_rows).to_csv(RESULT_ROOT / "markov_wave_test.csv", index=False)

    conditions = ("StaticControl", "CorrectLag", "SameWave", "TemporalShuffle",
                  "GeneIdentityShuffle", "IntermediateMasked")
    ablation_rows = []; prediction_cache = {}
    for split_index, split in enumerate(splits):
        train, test = split["train"], split["test"]
        for condition_index, condition in enumerate(conditions):
            seed = split["seed"] + condition_index * 100003
            representation = representation_for_condition(condition, waves, sources, genes,
                                                            static_all_genes, train, seed)
            features = representation[source_gene_rows]
            prediction, alpha = standardized_ridge(features[train], waves[train, 0], features[test], grid)
            metrics = prediction_metrics(prediction, waves[test, 0], sources[test], genes)
            ablation_rows.append(add_common(metrics, split, condition, "heldout_W23", alpha, len(train), len(test)) |
                                 {"heldout_source_absent_all_days": True,
                                  "test_source_response_used_to_build_representation": False})
            prediction_cache[(split_index, condition)] = prediction
        if (split_index + 1) % 20 == 0:
            print(f"[传播复现] First-responder 消融 {split_index + 1}/{len(splits)}", flush=True)
    ablation = pd.DataFrame(ablation_rows); ablation.to_csv(RESULT_ROOT / "first_responder_ablation.csv", index=False)

    dose_rows = []
    for split_index, split in enumerate(splits):
        train, test = split["train"], split["test"]
        for dose in config["information_doses"]:
            for subsample in range(config["dose_subsamples_per_outer_group"]):
                seed = split["seed"] + int(dose * 1000) * 1009 + subsample * 7919
                representation = transmission_representations(waves, sources, genes, train, "correct_lag", seed, dose)
                features = representation[source_gene_rows]
                prediction, alpha = standardized_ridge(features[train], waves[train, 0], features[test], grid)
                metrics = prediction_metrics(prediction, waves[test, 0], sources[test], genes)
                dose_rows.append(add_common(metrics, split, "CorrectLagDose", "heldout_W23", alpha, len(train), len(test)) |
                                 {"evidence_fraction": dose, "subsample": subsample,
                                  "zero_percent_masks_all_intermediate_evidence": dose == 0})
        if (split_index + 1) % 20 == 0:
            print(f"[传播复现] 信息剂量曲线 {split_index + 1}/{len(splits)}", flush=True)
    dose = pd.DataFrame(dose_rows); dose.to_csv(RESULT_ROOT / "information_dose_curve.csv", index=False)

    markov_summary = pd.DataFrame(markov_rows).groupby(["target", "model"]).response_distance_rho.agg(["mean", "median", "std"]).reset_index()
    ablation_summary = ablation.groupby("model").response_distance_rho.agg(["mean", "median", "std"]).reset_index()
    dose_summary = dose.groupby("evidence_fraction").response_distance_rho.agg(["mean", "median", "std"]).reset_index()
    markov_summary.to_csv(RESULT_ROOT / "markov_wave_summary.csv", index=False)
    ablation_summary.to_csv(RESULT_ROOT / "first_responder_summary.csv", index=False)
    dose_summary.to_csv(RESULT_ROOT / "information_dose_summary.csv", index=False)
    summary = {"created_at": now(), "markov": markov_summary.to_dict("records"),
               "first_responder": ablation_summary.to_dict("records"), "dose": dose_summary.to_dict("records")}
    atomic_json(RESULT_ROOT / "parts_cde_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
