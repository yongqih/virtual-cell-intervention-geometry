from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from benchmark_common import (
    CACHE_ROOT, FROZEN_CACHE, RESULT_ROOT, ROOT, SCRIPT_ROOT, apply_affine, atomic_json,
    atomic_npz, choose_direct_model, complete_metrics, direct_indices, fit_dense_transition,
    geometry_rho, grouped_twofold_splits, rank_metrics, representation, response_cosines,
    safe_pearson, sha256, source_rows, standardized_ridge, strict_trans_cosine_distance,
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def pseudoreplicate_reliability(config, expression, assignment, times, sources, genes):
    path = CACHE_ROOT / "endpoint_pseudoreplicates.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            r5a, r5b, w45a, w45b = z["r5a"], z["r5b"], z["w45a"], z["w45b"]
    else:
        shape = (config["pseudoreplicate_repeats"], len(sources), len(genes))
        r5a, r5b, w45a, w45b = [np.empty(shape, np.float32) for _ in range(4)]
        for repeat in range(config["pseudoreplicate_repeats"]):
            rng = np.random.default_rng(config["pseudoreplicate_seed"] + repeat * 1009)
            halves = {4: [[], []], 5: [[], []]}
            for day in (4, 5):
                cr = np.flatnonzero((assignment == "control") & (times == day)); co = rng.permutation(cr); cut = len(co) // 2
                controls = [expression[co[:cut]].mean(0), expression[co[cut:]].mean(0)]
                for source in sources:
                    rows = np.flatnonzero((assignment == source) & (times == day)); order = rng.permutation(rows); sc = len(order) // 2
                    halves[day][0].append(expression[order[:sc]].mean(0) - controls[0])
                    halves[day][1].append(expression[order[sc:]].mean(0) - controls[1])
            d4a, d4b = np.asarray(halves[4][0]), np.asarray(halves[4][1])
            d5a, d5b = np.asarray(halves[5][0]), np.asarray(halves[5][1])
            r5a[repeat], r5b[repeat] = d5a, d5b; w45a[repeat], w45b[repeat] = d5a - d4a, d5b - d4b
            if (repeat + 1) % 20 == 0: print(f"[endpoint] pseudoreplicates {repeat + 1}/{shape[0]}", flush=True)
        atomic_npz(path, r5a=r5a, r5b=r5b, w45a=w45a, w45b=w45b)
    direct = direct_indices(sources, genes); rows = []
    for target, aa, bb in (("R5", r5a, r5b), ("W45", w45a, w45b)):
        for repeat, (a, b) in enumerate(zip(aa, bb)):
            pearson = []; cosine = []
            for i in range(len(sources)):
                keep = np.ones(len(genes), bool)
                if direct[i] >= 0: keep[direct[i]] = False
                pearson.append(safe_pearson(a[i, keep], b[i, keep]))
                cosine.append(float(a[i, keep] @ b[i, keep] / max(np.linalg.norm(a[i, keep]) * np.linalg.norm(b[i, keep]), 1e-12)))
            rows.append({"target": target, "repeat": repeat,
                         "geometry_reliability": geometry_rho(a, b, direct),
                         "mean_response_pearson": float(np.nanmean(pearson)),
                         "mean_response_cosine": float(np.mean(cosine)),
                         "mean_squared_difference": float(np.mean((a - b) ** 2)),
                         "resampling_unit": "cells within perturbation source and day"})
    pd.DataFrame(rows).to_csv(RESULT_ROOT / "target_reliability.csv", index=False)


def model_row(split, model, target, prediction, truth, sources, genes, k, extra=None):
    test = split["test"]
    return {"repeat": split["repeat"], "group": split["group"], "split_seed": split["seed"],
            "model": model, "target": target, "n_train_sources": len(split["train"]),
            "n_test_sources": len(test), "one_model_for_entire_heldout_group": True,
            "heldout_source_absent_all_days": True, "outer_test_used_for_selection": False,
            **complete_metrics(prediction, truth, sources[test], genes, k), **(extra or {})}


def fit_frozen_chain(waves, response, static, sources, genes, source_gene_rows, train, query, seed, alpha_grid):
    feat = representation("CorrectLag", waves, response, static, sources, genes, source_gene_rows, train, seed)
    w23, a23 = standardized_ridge(feat[train], waves[train, 0], feat[query], alpha_grid)
    r2, ar2 = standardized_ridge(feat[train], response[train, 0], feat[query], alpha_grid)
    transition, at = fit_dense_transition(waves, train, seed, alpha_grid)
    w34 = apply_affine(transition, w23); w45 = apply_affine(transition, w34)
    return {"r2": r2, "w23": w23, "w34": w34, "w45": w45,
            "r5": r2 + w23 + w34 + w45, "transition": transition,
            "alpha_r2": ar2, "alpha_w23": a23, "alpha_transition": at}


def geometry_error_correlation(a, b, truth, source_names, genes):
    direct = direct_indices(source_names, genes); upper = np.triu_indices(len(a), 1)
    td = strict_trans_cosine_distance(truth, direct)[upper]
    ea = strict_trans_cosine_distance(a, direct)[upper] - td
    eb = strict_trans_cosine_distance(b, direct)[upper] - td
    return safe_pearson(ea, eb)


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True); CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    config = json.loads((SCRIPT_ROOT / "config.json").read_text(encoding="utf-8"))
    required = ["target_reliability.csv", "direct_vs_chain_w45.csv", "direct_endpoint_results.csv",
                "propagated_endpoint_results.csv", "endpoint_oracle_ladder.csv",
                "endpoint_shared_response_audit.csv", "direct_chain_complementarity.csv"]
    if all((RESULT_ROOT / name).exists() for name in required):
        print("[endpoint] all scientific tables already cached", flush=True); return
    with np.load(FROZEN_CACHE, allow_pickle=False) as z:
        expression = z["expression"].astype(np.float32); assignment = z["assignment"].astype(str)
        times = z["times"].astype(int); genes = z["genes"].astype(str); sources = z["sources"].astype(str)
        response = z["response"].astype(np.float32); waves = z["waves"].astype(np.float32)
        static = z["static_control_representation"].astype(np.float32)
    gene_lookup = {g: i for i, g in enumerate(genes)}
    source_gene_rows = np.asarray([gene_lookup[s] for s in sources], int)
    pseudoreplicate_reliability(config, expression, assignment, times, sources, genes)
    splits = grouped_twofold_splits(len(sources), config["source_disjoint_repeats"], config["outer_split_seed"])
    names = list(config["direct_representations"]); grid = tuple(config["ridge_alpha_grid"]); k = config["local_neighbors"]
    w45_rows, direct_r5_rows, chain_r5_rows, oracle_rows, shared_rows, comp_rows = [], [], [], [], [], []
    split_audit = []
    for si, split in enumerate(splits):
        train, test = split["train"], split["test"]; seed = split["seed"] + config["inner_split_seed_offset"]
        split_audit.append({"repeat": split["repeat"], "group": split["group"], "seed": split["seed"],
                            "train": sources[train].tolist(), "test": sources[test].tolist()})
        # Training-only selection of the primary direct model, independently for each target.
        selected_w45, alpha_w45, _ = choose_direct_model(waves[:, 2], waves, response, static, sources, genes,
                                                         source_gene_rows, train, seed, names, grid)
        selected_r5, alpha_r5, _ = choose_direct_model(response[:, 3], waves, response, static, sources, genes,
                                                       source_gene_rows, train, seed + 10007, names, grid)
        direct_w45 = {}; direct_r5 = {}
        for mi, name in enumerate(names):
            feat = representation(name, waves, response, static, sources, genes, source_gene_rows, train, seed + mi * 1009)
            pw, aw = standardized_ridge(feat[train], waves[train, 2], feat[test], grid)
            pr, ar = standardized_ridge(feat[train], response[train, 3], feat[test], grid)
            direct_w45[name], direct_r5[name] = pw, pr
            w45_rows.append(model_row(split, f"Direct_{name}", "W45", pw, waves[test, 2], sources, genes, k,
                                       {"selected_alpha_training_only": aw, "primary_selected": name == selected_w45}))
            direct_r5_rows.append(model_row(split, f"Direct_{name}", "R5", pr, response[test, 3], sources, genes, k,
                                            {"selected_alpha_training_only": ar, "primary_selected": name == selected_r5}))
        # Refit the nested-selected direct model with its selected alpha exactly once on outer training.
        fw = representation(selected_w45, waves, response, static, sources, genes, source_gene_rows, train, seed + 70001)
        fr = representation(selected_r5, waves, response, static, sources, genes, source_gene_rows, train, seed + 80021)
        primary_w45, _ = standardized_ridge(fw[train], waves[train, 2], fw[test], (alpha_w45,))
        primary_r5, _ = standardized_ridge(fr[train], response[train, 3], fr[test], (alpha_r5,))
        chain = fit_frozen_chain(waves, response, static, sources, genes, source_gene_rows, train, test, split["seed"], grid)
        w45_rows.append(model_row(split, "Chained_Frozen", "W45", chain["w45"], waves[test, 2], sources, genes, k,
                                   {"selected_alpha_training_only": chain["alpha_w23"], "transition_alpha_training_only": chain["alpha_transition"],
                                    "primary_selected": False}))
        w45_rows.append(model_row(split, "Direct_NestedSelected", "W45", primary_w45, waves[test, 2], sources, genes, k,
                                   {"selected_representation_training_only": selected_w45, "selected_alpha_training_only": alpha_w45,
                                    "primary_selected": True}))
        chain_r5_rows.append(model_row(split, "FullyPredictedMarkov", "R5", chain["r5"], response[test, 3], sources, genes, k,
                                       {"alpha_r2": chain["alpha_r2"], "alpha_w23": chain["alpha_w23"],
                                        "alpha_transition": chain["alpha_transition"]}))
        direct_r5_rows.append(model_row(split, "Direct_NestedSelected", "R5", primary_r5, response[test, 3], sources, genes, k,
                                        {"selected_representation_training_only": selected_r5, "selected_alpha_training_only": alpha_r5,
                                         "primary_selected": True}))
        transition = chain["transition"]
        true_w23 = waves[test, 0]; true_w34 = waves[test, 1]
        oracle_w34 = apply_affine(transition, true_w23); oracle_w45 = apply_affine(transition, oracle_w34)
        teacher_w45 = apply_affine(transition, true_w34)
        ladder = {
            "F0_FullyPredicted": chain["r5"],
            "F1_TrueR2": response[test, 0] + chain["w23"] + chain["w34"] + chain["w45"],
            "F2_TrueR2_TrueW23": response[test, 0] + true_w23 + oracle_w34 + oracle_w45,
            "F3_TrueR2_TrueW23_TrueW34": response[test, 0] + true_w23 + true_w34 + teacher_w45,
            "F4_FullOracle": response[test, 3],
        }
        for level, pred in ladder.items():
            oracle_rows.append(model_row(split, level, "R5", pred, response[test, 3], sources, genes, k,
                                         {"deployable": level == "F0_FullyPredicted"}))
        mu = response[train, 3].mean(0); residual_truth = response[test, 3] - mu
        shared_models = {"MeanTrainR5": np.repeat(mu[None], len(test), axis=0),
                         "PredictedR2Copy": chain["r2"], "Direct_NestedSelected": primary_r5,
                         "FullyPredictedMarkov": chain["r5"]}
        for name, pred in shared_models.items():
            for scale, pp, yy in (("absolute", pred, response[test, 3]), ("intervention_residual", pred - mu, residual_truth)):
                shared_rows.append(model_row(split, name, scale, pp, yy, sources, genes, k))
        # Two-fold OOF predictions on outer training select fusion lambda without test responses.
        order = np.random.default_rng(seed + 33013).permutation(train); inner_folds = np.array_split(order, 2)
        oof_direct = np.zeros((len(train), len(genes)), np.float32); oof_chain = np.zeros_like(oof_direct); loc = {v: i for i, v in enumerate(train)}
        for inner_val in inner_folds:
            inner_fit = np.setdiff1d(train, inner_val)
            ifeat = representation(selected_r5, waves, response, static, sources, genes, source_gene_rows, inner_fit, seed + int(inner_val[0]))
            dp, _ = standardized_ridge(ifeat[inner_fit], response[inner_fit, 3], ifeat[inner_val], (alpha_r5,))
            cp = fit_frozen_chain(waves, response, static, sources, genes, source_gene_rows, inner_fit, inner_val,
                                  seed + int(inner_val[0]) * 193, grid)["r5"]
            for j, source_index in enumerate(inner_val): oof_direct[loc[source_index]], oof_chain[loc[source_index]] = dp[j], cp[j]
        mean_chain = oof_chain.mean(0); centered_oof = oof_chain - mean_chain
        lambda_scores = [(float(np.mean((oof_direct + lam * centered_oof - response[train, 3]) ** 2)), lam)
                         for lam in config["fusion_lambda_grid"]]
        selected_lambda = float(min(lambda_scores)[1]); fused = primary_r5 + selected_lambda * (chain["r5"] - mean_chain)
        ds = pd.DataFrame(source_rows(primary_r5, response[test, 3], sources[test], genes))
        cs = pd.DataFrame(source_rows(chain["r5"], response[test, 3], sources[test], genes))
        correction = chain["r5"] - mean_chain; error = response[test, 3] - primary_r5
        correction_alignment = float(np.mean(response_cosines(correction, error, sources[test], genes)))
        comp_rows.append(model_row(split, "Fusion", "R5", fused, response[test, 3], sources, genes, k,
                                   {"selected_lambda_training_only": selected_lambda,
                                    "source_mse_error_correlation": safe_pearson(ds.mse, cs.mse),
                                    "pairwise_geometry_error_correlation": geometry_error_correlation(primary_r5, chain["r5"], response[test, 3], sources[test], genes),
                                    "residual_correction_alignment": correction_alignment,
                                    "direct_geometry": geometry_rho(primary_r5, response[test, 3], direct_indices(sources[test], genes)),
                                    "chain_geometry": geometry_rho(chain["r5"], response[test, 3], direct_indices(sources[test], genes))}))
        atomic_npz(CACHE_ROOT / f"split_{si:03d}.npz", test=test, primary_w45=primary_w45,
                   chained_w45=chain["w45"], primary_r5=primary_r5, chained_r5=chain["r5"], fused_r5=fused)
        if (si + 1) % 10 == 0: print(f"[endpoint] grouped benchmark {si + 1}/{len(splits)}", flush=True)
    tables = {"direct_vs_chain_w45.csv": w45_rows, "direct_endpoint_results.csv": direct_r5_rows,
              "propagated_endpoint_results.csv": chain_r5_rows, "endpoint_oracle_ladder.csv": oracle_rows,
              "endpoint_shared_response_audit.csv": shared_rows, "direct_chain_complementarity.csv": comp_rows}
    for name, rows in tables.items(): pd.DataFrame(rows).to_csv(RESULT_ROOT / name, index=False)
    atomic_json(RESULT_ROOT / "split_audit.json", {"created_at": now(), "groups": len(splits), "sources": len(sources),
                "heldout_source_absent_all_times": True, "one_model_per_heldout_group": True,
                "outer_test_used_for_fitting_or_selection": False, "splits": split_audit})
    atomic_json(RESULT_ROOT / "provenance.json", {"created_at": now(), "input": str(FROZEN_CACHE.relative_to(ROOT)),
                "input_sha256": sha256(FROZEN_CACHE), "config_sha256": sha256(SCRIPT_ROOT / "config.json"),
                "python": platform.python_version(), "cells": len(expression), "sources": len(sources), "genes": len(genes),
                "gpu_used": False, "previous_results_read_only": ["propagation_reproduction", "renge_dynamic_validity",
                "renge_first_wave_program", "renge_program_identifiability"]})
    atomic_json(RESULT_ROOT / "run_complete.json", {"completed_at": now(), "full_training_started": False,
                "new_architecture_trained": False, "cached_split_predictions": len(splits)})
    print("[endpoint] all scientific tables complete", flush=True)


if __name__ == "__main__": main()
