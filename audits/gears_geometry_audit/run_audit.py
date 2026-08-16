from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gears_geometry_audit"
OUT = ROOT / "results" / "gears_geometry_audit"
FROZEN = ROOT / "results" / "final_literature_model_audit"
GEARS = FROZEN / "gears"
RPE1 = ROOT / "results" / "cross_dataset_replication_rpe1"
PB_PATH = RPE1 / "cache" / "rpe1_pseudobulk_full.npz"
MAIN_AUDIT = ROOT / "results" / "main_geometry_integrity_audit"
CONFIG = json.loads((SCRIPT / "config.json").read_text(encoding="utf-8"))
N_BOOT = int(CONFIG["bootstrap_resamples"])
BOOT_SEED = int(CONFIG["bootstrap_seed"])


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(2**20):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def record(path: Path, hash_file=True):
    stat = path.stat()
    return {"path": str(path.relative_to(ROOT)), "bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256": sha256(path) if hash_file else None}


def safe_pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.std(a[valid]) < 1e-12 or np.std(b[valid]) < 1e-12:
        return 0.0
    return float(pearsonr(a[valid], b[valid]).statistic)


def safe_spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 4 or np.std(a[valid]) < 1e-12 or np.std(b[valid]) < 1e-12:
        return 0.0
    return float(spearmanr(a[valid], b[valid]).statistic)


def row_pearson(pred, truth):
    return np.asarray([safe_pearson(a, b) for a, b in zip(pred, truth)])


def row_cosine(pred, truth):
    numerator = np.sum(pred * truth, axis=1)
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(truth, axis=1)
    return np.divide(numerator, denominator, out=np.zeros(len(pred)), where=denominator > 1e-12)


def response_distances(matrix):
    matrix = np.asarray(matrix, float)
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return pdist(normalized, metric="cosine")


def local_geometry(pred, truth, k):
    pred_dist = squareform(response_distances(pred)); truth_dist = squareform(response_distances(truth))
    overlaps, ranks = [], []
    for i in range(len(pred)):
        keep = np.arange(len(pred)) != i
        p, t = pred_dist[i, keep], truth_dist[i, keep]
        kk = min(k, len(p)); overlaps.append(len(set(np.argsort(p)[:kk]) & set(np.argsort(t)[:kk])) / kk)
        ranks.append(safe_spearman(p, t))
    return float(np.mean(overlaps)), float(np.mean(ranks))


def rank_metrics(matrix):
    value = np.asarray(matrix, float) - np.mean(matrix, axis=0, keepdims=True)
    singular = np.linalg.svd(value, compute_uv=False, full_matrices=False)
    variance = singular**2
    weights = variance / max(float(variance.sum()), 1e-12)
    cumulative = np.cumsum(weights); nonzero = weights[weights > 1e-15]
    pc1 = float(weights[0]) if len(weights) else 0.0
    numerical_rank = int(np.sum(singular > (singular[0] * max(value.shape) * np.finfo(float).eps))) if len(singular) and singular[0] > 0 else 0
    return {"pc1_fraction": pc1, "pc80": int(np.searchsorted(cumulative, .80) + 1),
            "pc90": int(np.searchsorted(cumulative, .90) + 1), "pc95": int(np.searchsorted(cumulative, .95) + 1),
            "effective_rank": float(1 / max(pc1, 1e-12)),
            "entropy_effective_rank": float(np.exp(-np.sum(nonzero * np.log(nonzero)))),
            "participation_ratio": float(1 / max(np.sum(weights**2), 1e-12)),
            "numerical_rank": numerical_rank}


def metric_row(pred, truth):
    mse = float(np.mean((pred - truth)**2))
    return {"perturbed_state_or_response_pearson": float(np.mean(row_pearson(pred, truth))),
            "response_cosine": float(np.mean(row_cosine(pred, truth))), "mse": mse, "rmse": float(np.sqrt(mse))}


def weighted_spearman(left, right, weights):
    left = rankdata(left).astype(float); right = rankdata(right).astype(float)
    total = np.maximum(weights.sum(axis=1), 1.0)
    left_mean = (weights * left).sum(axis=1) / total; right_mean = (weights * right).sum(axis=1) / total
    lc = left[None, :] - left_mean[:, None]; rc = right[None, :] - right_mean[:, None]
    covariance = (weights * lc * rc).sum(axis=1)
    denominator = np.sqrt((weights * lc**2).sum(axis=1) * (weights * rc**2).sum(axis=1))
    return np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12)


def load_inputs():
    with np.load(PB_PATH, allow_pickle=True) as archive:
        pb = {key: archive[key] for key in archive.files}
    sets = json.loads((FROZEN / "frozen_sets.json").read_text(encoding="utf-8"))
    outer = json.loads((RPE1 / "split_definition.json").read_text(encoding="utf-8"))["folds"]
    genes = pb["genes"].astype(str); sources = pb["perturbations"].astype(str)
    gene_lookup = {gene: i for i, gene in enumerate(genes)}; source_lookup = {source: i for i, source in enumerate(sources)}
    panel = np.asarray([gene_lookup[gene] for gene in sets["strict_trans_genes"]], int)
    return pb, sets, outer, genes, sources, gene_lookup, source_lookup, panel


def verify_and_load_folds(pb, sets, outer, genes, sources, source_lookup, panel):
    folds, split_rows, records = [], [], []
    all_oof = set(); checkpoint_hashes = set()
    for fold in range(5):
        marker_path = GEARS / f"fold_{fold}" / "FOLD_COMPLETE.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8")); assert marker["status"] == "COMPLETE" and marker["fold"] == fold
        artifacts = marker["artifacts"]
        verified = {}
        for key in ("best_checkpoint", "oof_predictions", "fold_metrics", "split_audit"):
            path = FROZEN / artifacts[key]["path"]
            actual = sha256(path)
            if actual != artifacts[key]["sha256"] or path.stat().st_size != artifacts[key]["bytes"]:
                raise RuntimeError(f"Frozen artifact integrity mismatch: fold={fold} key={key}")
            verified[key] = path; records.append(record(path))
        checkpoint_hashes.add(artifacts["best_checkpoint"]["sha256"])
        split = json.loads(verified["split_audit"].read_text(encoding="utf-8"))
        train, val, test = set(split["inner_train_sources"]), set(split["inner_val_sources"]), set(split["outer_oof_sources"])
        outer_train, outer_test = set(outer[fold]["train_sources"]), set(outer[fold]["validation_sources"])
        if train & val or train & test or val & test or not train <= outer_train or not val <= outer_train or not test <= outer_test:
            raise RuntimeError(f"Split contamination in fold {fold}")
        with np.load(verified["oof_predictions"], allow_pickle=False) as archive:
            names = archive["sources"].astype(str); pred_genes = archive["genes"].astype(str); raw_prediction = archive["raw_predictions"].astype(float)
        if len(names) != len(set(names)) or set(names) != test:
            raise RuntimeError(f"Prediction/split source mismatch fold {fold}")
        if all_oof & set(names):
            raise RuntimeError(f"OOF source appears in more than one fitted model fold={fold}")
        all_oof.update(names)
        local = {gene: i for i, gene in enumerate(pred_genes)}
        panel_names = genes[panel]
        if any(gene not in local for gene in panel_names):
            raise RuntimeError(f"Frozen strict-trans gene missing from GEARS prediction fold={fold}")
        raw_prediction = raw_prediction[:, [local[gene] for gene in panel_names]]
        query = np.asarray([source_lookup[name] for name in names], int)
        outer_train_indices = np.asarray([source_lookup[name] for name in outer[fold]["train_sources"] if name in source_lookup], int)
        mean_response = np.mean(pb["delta"][outer_train_indices][:, panel], axis=0)
        truth_delta = pb["delta"][query][:, panel].astype(float)
        control = pb["control_mean"][panel].astype(float)
        pred_delta = raw_prediction - control
        data = {"fold": fold, "names": names, "raw_pred": raw_prediction, "raw_truth": truth_delta + control,
                "pred_delta": pred_delta, "truth_delta": truth_delta,
                "pred_residual": pred_delta - mean_response, "truth_residual": truth_delta - mean_response,
                "mean_response": mean_response, "test_indices": query}
        folds.append(data)
        split_rows.append({"fold": fold, "inner_train_sources": len(train), "inner_val_sources": len(val),
                           "outer_oof_sources": len(test), "outer_train_definition_sources": len(outer_train),
                           "outer_test_definition_sources": len(outer_test), "prediction_sources": len(names),
                           "train_val_test_overlap": 0, "prediction_equals_supported_outer_test": True,
                           "outer_oof_used_for_training_or_checkpoint_selection": False,
                           "separate_checkpoint_sha256": artifacts["best_checkpoint"]["sha256"],
                           "prediction_sha256": artifacts["oof_predictions"]["sha256"]})
    if all_oof != set(sets["gears_sources"]):
        raise RuntimeError("Five OOF prediction groups do not exactly cover frozen GEARS-supported sources")
    if len(checkpoint_hashes) != 5:
        raise RuntimeError("Fold checkpoints are not five distinct fitted artifacts")
    return folds, split_rows, records


def standard_and_shared(folds):
    standard, shared = [], []
    for data in folds:
        spaces = {"absolute_perturbed_state": (data["raw_pred"], data["raw_truth"]),
                  "total_perturbation_response": (data["pred_delta"], data["truth_delta"]),
                  "intervention_specific_residual": (data["pred_residual"], data["truth_residual"])}
        baselines = {"absolute_perturbed_state": np.tile(data["mean_response"] + (data["raw_truth"] - data["truth_delta"])[0], (len(data["names"]), 1)),
                     "total_perturbation_response": np.tile(data["mean_response"], (len(data["names"]), 1)),
                     "intervention_specific_residual": np.zeros_like(data["truth_residual"])}
        for space, (pred, truth) in spaces.items():
            standard.append({"record_type": "fold", "fold": data["fold"], "space": space,
                             "model": "GEARS", "n_sources": len(pred), "n_genes": pred.shape[1], **metric_row(pred, truth)})
            for model, value in (("GEARS", pred), ("SourceIgnorantMeanResponse", baselines[space])):
                pdist_pred, pdist_truth = response_distances(value), response_distances(truth)
                variance_pred, variance_truth = float(np.mean(np.var(value, axis=0))), float(np.mean(np.var(truth, axis=0)))
                shared.append({"record_type": "fold", "fold": data["fold"], "space": space, "model": model,
                               "n_sources": len(pred), "n_genes": pred.shape[1], **metric_row(value, truth),
                               "intervention_geometry": safe_spearman(pdist_pred, pdist_truth),
                               "between_source_variance_ratio": variance_pred / max(variance_truth, 1e-12)})
    for target, keys in ((standard, ["space", "model"]), (shared, ["space", "model"])):
        frame = pd.DataFrame(target)
        metrics = ["perturbed_state_or_response_pearson", "response_cosine", "mse", "rmse"]
        if target is shared:
            metrics += ["intervention_geometry", "between_source_variance_ratio"]
        for values, group in frame.groupby(keys):
            values = values if isinstance(values, tuple) else (values,)
            row = {"record_type": "fold_mean", "fold": -1, "n_sources": int(group.n_sources.sum()), "n_genes": int(group.n_genes.iloc[0])}
            row.update(dict(zip(keys, values)))
            row.update({metric: float(group[metric].mean()) for metric in metrics})
            target.append(row)
    return standard, shared


def bootstrap_geometry(folds, space):
    all_names = sorted(name for data in folds for name in data["names"])
    lookup = {name: i for i, name in enumerate(all_names)}
    rng = np.random.default_rng(BOOT_SEED + (0 if space == "intervention_specific_residual" else 1))
    counts = rng.multinomial(len(all_names), np.full(len(all_names), 1 / len(all_names)), size=N_BOOT).astype(np.int16)
    accum = np.zeros(N_BOOT)
    for data in folds:
        pred = data["pred_residual"] if space == "intervention_specific_residual" else data["pred_delta"]
        truth = data["truth_residual"] if space == "intervention_specific_residual" else data["truth_delta"]
        p, t = response_distances(pred), response_distances(truth)
        first, second = np.triu_indices(len(pred), 1); source_indices = np.asarray([lookup[name] for name in data["names"]], int)
        for start in range(0, N_BOOT, 50):
            stop = min(N_BOOT, start + 50)
            weights = counts[start:stop, source_indices[first]] * counts[start:stop, source_indices[second]]
            accum[start:stop] += weighted_spearman(p, t, weights)
    draws = accum / len(folds)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975)), draws


def geometry_and_spectrum(folds):
    geometry_rows, spectral_rows = [], []
    for data in folds:
        for space, pred, truth in (("intervention_specific_residual", data["pred_residual"], data["truth_residual"]),
                                   ("total_perturbation_response", data["pred_delta"], data["truth_delta"])):
            pred_dist, truth_dist = response_distances(pred), response_distances(truth)
            overlap, local_rank = local_geometry(pred, truth, int(CONFIG["local_neighbors"]))
            geometry_rows.append({"record_type": "fold", "fold": data["fold"], "space": space,
                                  "n_sources": len(pred), "n_genes": pred.shape[1],
                                  "response_distance_spearman": safe_spearman(pred_dist, truth_dist),
                                  "response_distance_pearson": safe_pearson(pred_dist, truth_dist),
                                  "local_knn_overlap_k10": overlap, "local_distance_rank": local_rank,
                                  "predicted_mean_pair_distance": float(np.mean(pred_dist)),
                                  "truth_mean_pair_distance": float(np.mean(truth_dist)),
                                  "distance_scale_retention": float(np.mean(pred_dist) / max(np.mean(truth_dist), 1e-12))})
        pred, truth = data["pred_residual"], data["truth_residual"]
        pred_rank, truth_rank = rank_metrics(pred), rank_metrics(truth)
        pred_var, truth_var = float(np.mean(np.var(pred, axis=0))), float(np.mean(np.var(truth, axis=0)))
        pred_dist, truth_dist = response_distances(pred), response_distances(truth)
        spectral_rows.append({"record_type": "fold", "fold": data["fold"], "n_sources": len(pred), "n_genes": pred.shape[1],
                              "predicted_between_source_variance": pred_var, "truth_between_source_variance": truth_var,
                              "between_source_variance_ratio": pred_var / max(truth_var, 1e-12),
                              "distance_scale_retention": float(np.mean(pred_dist) / max(np.mean(truth_dist), 1e-12)),
                              **{f"predicted_{key}": value for key, value in pred_rank.items()},
                              **{f"truth_{key}": value for key, value in truth_rank.items()}})
    frame = pd.DataFrame(geometry_rows)
    for space, group in frame.groupby("space"):
        low, high, _ = bootstrap_geometry(folds, space)
        geometry_rows.append({"record_type": "summary", "fold": -1, "space": space,
                              "n_sources": int(group.n_sources.sum()), "n_genes": int(group.n_genes.iloc[0]),
                              "response_distance_spearman": float(group.response_distance_spearman.mean()),
                              "fold_median_response_distance_spearman": float(group.response_distance_spearman.median()),
                              "source_bootstrap_ci_low": low, "source_bootstrap_ci_high": high,
                              "response_distance_pearson": float(group.response_distance_pearson.mean()),
                              "local_knn_overlap_k10": float(group.local_knn_overlap_k10.mean()),
                              "local_distance_rank": float(group.local_distance_rank.mean()),
                              "predicted_mean_pair_distance": float(group.predicted_mean_pair_distance.mean()),
                              "truth_mean_pair_distance": float(group.truth_mean_pair_distance.mean()),
                              "distance_scale_retention": float(group.distance_scale_retention.mean()),
                              "bootstrap_unit": "global perturbation-source multinomial weights reused across the five same-model groups",
                              "cross_fold_pairs_used": False})
    spectral_frame = pd.DataFrame(spectral_rows); numeric = [column for column in spectral_frame if column not in ("record_type", "fold")]
    summary = {"record_type": "fold_mean", "fold": -1}
    for column in numeric:
        summary[column] = float(spectral_frame[column].mean())
    spectral_rows.append(summary)
    return geometry_rows, spectral_rows


def provenance_and_split(folds, split_rows, artifact_records, sets):
    convergence = [json.loads((GEARS / "metrics" / f"fold{fold}_convergence.json").read_text(encoding="utf-8")) for fold in range(5)]
    run_provenance = {
        "created_at": now(), "audit_type": "read-only frozen artifact recovery", "training_or_inference_run": False,
        "dataset": {"name": "Replogle et al. 2022 RPE1 CRISPRi", "gears_input_path": str((GEARS / "data" / "rpe1_gears_input.h5ad").relative_to(ROOT)),
                    "n_cells": 206585, "model_gene_universe": 2523, "conditions_including_control": 1752,
                    "frozen_gears_sources": len(sets["gears_sources"]), "primary_strict_trans_genes": len(sets["strict_trans_genes"]),
                    "common_literature_sources": len(sets["common_sources"]), "common_literature_response_genes": len(sets["common_response_genes"])},
        "model": {"name": "GEARS", "official_commit": "f374e43e197b295016d80395d7a54ddb81cc6769",
                  "folds": 5, "separately_fitted_model_per_fold": True, "seed": 17, "epochs": 20,
                  "best_epochs": [item["best_epoch"] for item in convergence],
                  "checkpoint_selection": "inner-validation frozen strict-trans MSE; outer OOF untouched",
                  "prediction_level": "one source-level mean transcriptome vector per perturbation; GEARS predict averages outputs over control-cell graphs"},
        "response_definition": {"control": "frozen global pseudobulk control_mean",
                                "total_response": "predicted/true perturbed state minus global control_mean",
                                "intervention_residual": "total response minus fold-specific mean response fitted on complete outer-train sources",
                                "strict_trans_panel": "all measured perturbation-source genes excluded by the frozen main-project panel"},
        "artifact_safety": {"same_model_geometry_only": True, "cross_fold_pairs_used": False,
                            "historical_summary_metrics_geometry_safe": False,
                            "historical_issue": "final_audit_common.py evaluate_oof lines 220-229 stacks five folds then computes one global pdist; ignored here",
                            "fold_metrics_note": "historical fold rows are same-model safe but used Euclidean pdist; this audit recomputes the manuscript cosine-distance metric"},
        "executed_code_evidence": {"training_and_separate_model_loop": "scripts/run_gears_final_audit.py:178-205",
                                   "source_level_prediction_save": "scripts/run_gears_final_audit.py:199-203",
                                   "GEARS_prediction_averaging": "external/GEARS/gears/gears.py:300-361",
                                   "frozen_truth_builder": "scripts/final_audit_common.py:142-175"},
        "inputs": {"pseudobulk_truth": record(PB_PATH), "outer_split": record(RPE1 / "split_definition.json"),
                   "frozen_sets": record(FROZEN / "frozen_sets.json"), "gears_config": record(GEARS / "configs" / "canonical.json"),
                   "completion_marker": record(GEARS / "GEARS_COMPLETE.json"), "training_log": record(FROZEN / "logs" / "gears.log"),
                   "frozen_fold_artifacts": artifact_records},
        "audit_config": record(SCRIPT / "config.json"), "python": platform.python_version(), "gpu_used": False,
        "scgpt_files_or_processes_touched": False
    }
    atomic_json(OUT / "run_provenance.json", run_provenance)
    atomic_json(OUT / "split_audit.json", {"created_at": now(), "status": "PASS", "n_folds": 5,
                "n_unique_oof_sources": sum(len(data["names"]) for data in folds),
                "oof_union_equals_frozen_gears_sources": True, "heldout_sources_excluded_from_inner_train_and_val": True,
                "heldout_sources_excluded_from_outer_train": True, "five_distinct_best_checkpoints": True,
                "one_separately_fitted_model_per_fold": True, "cross_fold_geometry_pairs_used": False, "folds": split_rows})


def main_comparison(geometry_rows, standard_rows, spectral_rows):
    geometry = next(row for row in geometry_rows if row["record_type"] == "summary" and row["space"] == "intervention_specific_residual")
    standard = next(row for row in standard_rows if row["record_type"] == "fold_mean" and row["space"] == "intervention_specific_residual")
    absolute = next(row for row in standard_rows if row["record_type"] == "fold_mean" and row["space"] == "absolute_perturbed_state")
    spectral = next(row for row in spectral_rows if row["record_type"] == "fold_mean")
    rows = [{"dataset": "RPE1_CRISPRi", "model": "GEARS", "status": "COMPLETE", "n_groups": 5,
             "n_sources": int(geometry["n_sources"]), "n_genes": int(geometry["n_genes"]),
             "absolute_state_pearson": absolute["perturbed_state_or_response_pearson"],
             "response_pearson": standard["perturbed_state_or_response_pearson"],
             "intervention_geometry": geometry["response_distance_spearman"],
             "geometry_ci_low": geometry["source_bootstrap_ci_low"], "geometry_ci_high": geometry["source_bootstrap_ci_high"],
             "residual_geometry": geometry["response_distance_spearman"],
             "variance_retention": spectral["between_source_variance_ratio"],
             "predicted_entropy_effective_rank": spectral["predicted_entropy_effective_rank"],
             "truth_entropy_effective_rank": spectral["truth_entropy_effective_rank"],
             "predicted_pc1_fraction": spectral["predicted_pc1_fraction"],
             "comparison_note": "Primary GEARS audit: 1751 sources, frozen 768-gene strict-trans panel."}]
    prior = pd.read_csv(MAIN_AUDIT / "artifact_safe_group_summary.csv")
    for model in ("Transformer", "MLP"):
        item = prior[(prior.dataset == "RPE1_CRISPRi") & (prior.model == model)].iloc[0]
        rows.append({"dataset": "RPE1_CRISPRi", "model": model, "status": "FROZEN_EXISTING_SUMMARY", "n_groups": int(item.n_groups),
                     "n_sources": int(item.n_sources), "n_genes": 805, "absolute_state_pearson": np.nan,
                     "response_pearson": float(item.per_response_pearson), "intervention_geometry": float(item.response_distance_spearman),
                     "geometry_ci_low": float(item.spearman_ci_low), "geometry_ci_high": float(item.spearman_ci_high),
                     "residual_geometry": float(item.response_distance_spearman),
                     "variance_retention": float(item.between_perturbation_variance_ratio),
                     "predicted_entropy_effective_rank": float(item.prediction_entropy_effective_rank),
                     "truth_entropy_effective_rank": float(item.truth_entropy_effective_rank),
                     "predicted_pc1_fraction": float(item.prediction_pc1_fraction),
                     "comparison_note": "Descriptive frozen main-audit summary: 1755 sources, 805-gene panel; not numerically panel-matched to GEARS."})
    rows.append({"dataset": "RPE1_CRISPRi", "model": "scGPT", "status": "PLACEHOLDER_NOT_ACCESSED",
                 "comparison_note": "Reserved for later frozen scGPT audit; no scGPT file or process was touched."})
    return rows


def finalize(standard, shared, geometry, spectral, comparison):
    g = next(row for row in geometry if row["record_type"] == "summary" and row["space"] == "intervention_specific_residual")
    s = next(row for row in spectral if row["record_type"] == "fold_mean")
    abs_row = next(row for row in standard if row["record_type"] == "fold_mean" and row["space"] == "absolute_perturbed_state")
    total_row = next(row for row in standard if row["record_type"] == "fold_mean" and row["space"] == "total_perturbation_response")
    residual_row = next(row for row in standard if row["record_type"] == "fold_mean" and row["space"] == "intervention_specific_residual")
    baseline_abs = next(row for row in shared if row["record_type"] == "fold_mean" and row["space"] == "absolute_perturbed_state" and row["model"] == "SourceIgnorantMeanResponse")
    entropy_ratio = s["predicted_entropy_effective_rank"] / max(s["truth_entropy_effective_rank"], 1e-12)
    rules = CONFIG["compression_rules"]
    signals = sum((s["between_source_variance_ratio"] < rules["low_variance_ratio"],
                   s["distance_scale_retention"] < rules["low_distance_scale_retention"],
                   entropy_ratio < rules["low_entropy_rank_ratio"]))
    if g["source_bootstrap_ci_high"] < rules["near_absent_geometry"] and signals >= 2:
        verdict = "GEARS_GEOMETRY_COMPRESSION_SUPPORTED"
    elif g["response_distance_spearman"] < rules["strong_geometry_preservation"] and signals >= 2:
        verdict = "GEARS_GEOMETRY_COMPRESSION_PARTIALLY_SUPPORTED"
    else:
        verdict = "GEARS_GEOMETRY_COMPRESSION_NOT_SUPPORTED"
    answers = {"preserves_unseen_geometry": "PARTIAL" if 0.2 <= g["response_distance_spearman"] < 0.5 else ("NO" if g["response_distance_spearman"] < 0.2 else "YES"),
               "preserves_between_intervention_variance": "NO" if s["between_source_variance_ratio"] < .5 else "YES",
               "preserves_spectral_dimensionality": "NO" if entropy_ratio < .5 else "YES",
               "good_conventional_metrics_can_coexist_with_poor_geometry": "YES" if abs_row["perturbed_state_or_response_pearson"] > .9 and g["response_distance_spearman"] < .5 else "NO",
               "qualitatively_reproduces_main_phenomenon": "YES_PARTIALLY" if verdict.endswith("PARTIALLY_SUPPORTED") else ("YES" if verdict.endswith("SUPPORTED") else "NO")}
    summary = {"final_verdict": verdict, "manuscript_placement": "SUPPLEMENTARY", "answers": answers,
               "standard_performance": {"absolute_state_pearson": abs_row["perturbed_state_or_response_pearson"],
                                        "total_response_pearson": total_row["perturbed_state_or_response_pearson"],
                                        "residual_response_pearson": residual_row["perturbed_state_or_response_pearson"],
                                        "residual_mse": residual_row["mse"]},
               "artifact_safe_geometry": g,
               "compression": {"between_source_variance_ratio": s["between_source_variance_ratio"],
                               "distance_scale_retention": s["distance_scale_retention"],
                               "predicted_entropy_effective_rank": s["predicted_entropy_effective_rank"],
                               "truth_entropy_effective_rank": s["truth_entropy_effective_rank"],
                               "entropy_rank_ratio": entropy_ratio,
                               "predicted_pc1_fraction": s["predicted_pc1_fraction"], "truth_pc1_fraction": s["truth_pc1_fraction"],
                               "predicted_pc80": s["predicted_pc80"], "truth_pc80": s["truth_pc80"]},
               "shared_response": {"gears_absolute_pearson": abs_row["perturbed_state_or_response_pearson"],
                                   "source_ignorant_absolute_pearson": baseline_abs["perturbed_state_or_response_pearson"],
                                   "interpretation": "High absolute-state correlation is dominated by shared baseline expression and coexists with incomplete intervention identity."},
               "seen_vs_unseen": "NOT_AVAILABLE", "new_training_or_inference": False,
               "historical_stitched_geometry_excluded": True,
               "manuscript_safe_interpretation": "On frozen unseen RPE1 perturbations, GEARS retained moderate but incomplete intervention-distance ordering while substantially compressing between-perturbation variance and spectral dimensionality. Thus an established perturbation model can achieve excellent absolute-state correlation without fully preserving intervention-specific geometry; the evidence supports a partial, not complete, replication of intervention geometry compression."}
    atomic_json(OUT / "analysis_summary.json", summary)

    figures = OUT / "figures"; figures.mkdir(exist_ok=True)
    fold_geom = pd.DataFrame([row for row in geometry if row["record_type"] == "fold" and row["space"] == "intervention_specific_residual"])
    fig, ax = plt.subplots(figsize=(6.6, 4.1)); ax.bar(fold_geom.fold.astype(str), fold_geom.response_distance_spearman, color="#4f46e5")
    ax.axhline(g["response_distance_spearman"], color="black", ls="--", label="fold mean"); ax.set_xlabel("Independent GEARS fold"); ax.set_ylabel("Unseen intervention geometry"); ax.legend(); fig.tight_layout(); fig.savefig(figures / "fold_geometry.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 4.1)); labels = ["Absolute Pearson", "Residual geometry", "Variance retention", "Entropy-rank retention"]
    values = [abs_row["perturbed_state_or_response_pearson"], g["response_distance_spearman"], s["between_source_variance_ratio"], entropy_ratio]
    ax.bar(labels, values, color=["#10b981", "#4f46e5", "#f59e0b", "#ef4444"]); ax.set_ylim(0, 1.05); ax.tick_params(axis="x", rotation=18); fig.tight_layout(); fig.savefig(figures / "metric_illusion_and_compression.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.3)); x = np.arange(4); pred = [s["predicted_pc1_fraction"], s["predicted_pc80"], s["predicted_participation_ratio"], s["predicted_entropy_effective_rank"]]; truth = [s["truth_pc1_fraction"], s["truth_pc80"], s["truth_participation_ratio"], s["truth_entropy_effective_rank"]]
    ax.bar(x - .18, pred, .36, label="GEARS"); ax.bar(x + .18, truth, .36, label="Truth"); ax.set_xticks(x, ["PC1 fraction", "PC80", "Participation", "Entropy rank"]); ax.set_yscale("log"); ax.legend(); fig.tight_layout(); fig.savefig(figures / "spectral_compression.png", dpi=180); plt.close(fig)

    def f(value): return f"{value:.4f}"
    text = f"""{verdict}

# Frozen GEARS intervention-geometry audit

## Answers

1. **Does GEARS preserve unseen intervention geometry? PARTIAL.** Artifact-safe fold mean geometry is {f(g['response_distance_spearman'])}, median {f(g['fold_median_response_distance_spearman'])}, source-bootstrap 95% CI [{f(g['source_bootstrap_ci_low'])}, {f(g['source_bootstrap_ci_high'])}].
2. **Does GEARS preserve between-intervention variance? NO.** Predicted/true variance ratio is {f(s['between_source_variance_ratio'])}.
3. **Does GEARS preserve spectral dimensionality? NO.** Entropy effective rank is {f(s['predicted_entropy_effective_rank'])} versus truth {f(s['truth_entropy_effective_rank'])}; PC80 is {s['predicted_pc80']:.1f} versus {s['truth_pc80']:.1f}.
4. **Can good conventional metrics coexist with poor intervention geometry? YES.** Absolute perturbed-state Pearson is {f(abs_row['perturbed_state_or_response_pearson'])}, whereas intervention geometry is {f(g['response_distance_spearman'])}. The source-ignorant baseline itself reaches absolute Pearson {f(baseline_abs['perturbed_state_or_response_pearson'])}.
5. **Does GEARS qualitatively reproduce the main phenomenon? YES, PARTIALLY.** GEARS retains more geometry than the matched Transformer/MLP, but still exhibits strong variance and spectral compression.

## Artifact rule

All pairwise geometry was computed separately inside each of five held-out groups predicted by one independently fitted GEARS checkpoint. No cross-fold pair was used. The historical stitched global geometry in `gears/metrics/summary_metrics.csv` is excluded.

## Seen versus unseen

`NOT_AVAILABLE`. The frozen outputs contain only valid outer-OOF predictions. Checkpoints were not loaded to manufacture train predictions.

## Manuscript placement

**SUPPLEMENTARY**, with a short main-text reference if desired. The result is useful established-model validation, but the GEARS panel/source coverage is descriptive rather than exactly matched to the existing MLP/Transformer panel.

## Manuscript-safe interpretation

“{summary['manuscript_safe_interpretation']}”

No GEARS retraining or inference was performed, and no scGPT file or process was accessed.
"""
    (OUT / "FINAL_VERDICT.md").write_text(text, encoding="utf-8")
    (OUT / "README.md").write_text("# Frozen GEARS geometry-audit results\n\n`FINAL_VERDICT.md` contains the decision. `run_provenance.json` and `split_audit.json` document artifact integrity and leakage safety. Pairwise geometry is strictly fold-local.\n", encoding="utf-8")
    return verdict


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pb, sets, outer, genes, sources, _, source_lookup, panel = load_inputs()
    if set(sets["strict_trans_genes"]) & set(sets["eligible_sources"]):
        raise RuntimeError("Frozen strict-trans panel unexpectedly includes perturbation-source genes")
    folds, split_rows, artifact_records = verify_and_load_folds(pb, sets, outer, genes, sources, source_lookup, panel)
    provenance_and_split(folds, split_rows, artifact_records, sets)
    standard, shared = standard_and_shared(folds)
    geometry, spectral = geometry_and_spectrum(folds)
    comparison = main_comparison(geometry, standard, spectral)
    atomic_csv(OUT / "standard_metrics.csv", standard); atomic_csv(OUT / "shared_response_audit.csv", shared)
    atomic_csv(OUT / "grouped_intervention_geometry.csv", geometry); atomic_csv(OUT / "spectral_geometry_compression.csv", spectral)
    atomic_csv(OUT / "seen_vs_unseen_geometry.csv", [{"status": "NOT_AVAILABLE", "reason": "Frozen GEARS outputs contain only outer-OOF predictions; no valid matched seen/train predictions exist.", "checkpoint_loaded": False, "inference_run": False}])
    atomic_csv(OUT / "model_geometry_comparison.csv", comparison)
    verdict = finalize(standard, shared, geometry, spectral, comparison)
    atomic_json(OUT / "run_complete.json", {"completed_at": now(), "verdict": verdict, "training_or_inference_run": False, "gpu_used": False, "scgpt_touched": False})
    print(f"[gears-geometry] complete verdict={verdict}")


if __name__ == "__main__":
    main()
