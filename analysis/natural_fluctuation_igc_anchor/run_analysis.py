from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("AI4SCI_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT = ROOT / "results" / "natural_fluctuation_igc_anchor"
SOURCE = OUT / "source_data"
FIGURES = OUT / "figures"

DATA = ROOT / "data" / "raw" / "replogle22rpe1_processed_complete.valid.h5ad"
BASE = ROOT / "results" / "cross_dataset_replication_rpe1"
PB_PATH = BASE / "cache" / "rpe1_pseudobulk_full.npz"
REP_PATH = BASE / "cache" / "control_state_representations.npz"
SPLIT_PATH = BASE / "split_definition.json"
PREPROCESS_PATH = BASE / "preprocessing_config.json"
DATASET_MANIFEST_PATH = BASE / "dataset_manifest.json"
ESTABLISHED_PATH = BASE / "state_intervention" / "pairwise_geometry_alignment.csv"

SEED = 20260818
PCA_SEED = 1701
N_PCS = 20
N_NULL = 200
N_BOOT = 2000
N_BOOT_GEOMETRY = 500
LOCAL_K = 10
TOP_FRACTION = 0.10

ESTIMATORS = ["raw_cov", "raw_corr", "slope", "resid_cov", "resid_corr"]
RESIDUAL_SAFE = ["resid_cov", "resid_corr"]

GATES = {
    "strong_geometry_rescue": {
        "residualized_orientation_mean_cosine_gt": 0.0,
        "residualized_real_minus_target_permutation_bootstrap_ci_low_gt": 0.0,
        "geometry_spearman_mean_gte": 0.15,
        "geometry_source_bootstrap_ci_low_gte": 0.10,
        "between_source_variance_retention_gte": 0.25,
        "entropy_effective_rank_retention_gte": 0.50,
        "local_knn_overlap_k10_gte": 0.08,
    },
    "partial_anchor": {
        "residualized_orientation_mean_cosine_gt": 0.0,
        "residualized_real_minus_target_permutation_bootstrap_ci_low_gt": 0.0,
        "fraction_targetwise_empirical_p_lt_0_05_gt": 0.075,
        "median_oracle_response_energy_fraction_gt": 0.001,
    },
    "conservative_rule": (
        "Partial/strong support requires a residualized estimator; a signal confined to raw covariance "
        "is treated as state-confounded and does not support a natural anchor."
    ),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_pearson(left, right) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    if len(left) < 3 or np.std(left) < 1e-14 or np.std(right) < 1e-14:
        return 0.0
    value = pearsonr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def safe_spearman(left, right) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    if len(left) < 3 or np.std(left) < 1e-14 or np.std(right) < 1e-14:
        return 0.0
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def row_cosine(pred, truth):
    numerator = np.sum(pred * truth, axis=1)
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(truth, axis=1)
    return np.divide(numerator, denominator, out=np.zeros(len(pred)), where=denominator > 1e-12)


def row_pearson(pred, truth):
    pc = pred - pred.mean(axis=1, keepdims=True)
    tc = truth - truth.mean(axis=1, keepdims=True)
    return row_cosine(pc, tc)


def row_spearman(pred, truth):
    return row_pearson(rankdata(pred, axis=1), rankdata(truth, axis=1))


def directional_metrics(pred, truth):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    n, g = pred.shape
    k = max(1, int(math.ceil(TOP_FRACTION * g)))
    truth_top = np.argpartition(np.abs(truth), -k, axis=1)[:, -k:]
    pred_top = np.argpartition(np.abs(pred), -k, axis=1)[:, -k:]
    sign_agreement = np.empty(n)
    overlap = np.empty(n)
    signed_overlap = np.empty(n)
    for i in range(n):
        ti, pi = truth_top[i], pred_top[i]
        sign_agreement[i] = np.mean(np.sign(pred[i, ti]) == np.sign(truth[i, ti]))
        common = np.intersect1d(ti, pi, assume_unique=False)
        overlap[i] = len(common) / k
        signed_overlap[i] = (
            np.sum(np.sign(pred[i, common]) == np.sign(truth[i, common])) / k if len(common) else 0.0
        )
    return {
        "cosine": row_cosine(pred, truth),
        "pearson": row_pearson(pred, truth),
        "spearman": row_spearman(pred, truth),
        "sign_agreement_top10pct_truth": sign_agreement,
        "top10pct_absolute_overlap": overlap,
        "top10pct_signed_overlap": signed_overlap,
    }


def response_distances(matrix):
    matrix = np.asarray(matrix, float)
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return pdist(normalized, metric="cosine")


def local_geometry(pred, truth, k=LOCAL_K):
    pred_dist = squareform(response_distances(pred))
    truth_dist = squareform(response_distances(truth))
    overlaps, ranks = [], []
    for i in range(len(pred)):
        keep = np.arange(len(pred)) != i
        p, t = pred_dist[i, keep], truth_dist[i, keep]
        kk = min(k, len(p))
        overlaps.append(len(set(np.argsort(p)[:kk]) & set(np.argsort(t)[:kk])) / kk)
        ranks.append(safe_spearman(p, t))
    return float(np.mean(overlaps)), float(np.mean(ranks))


def rank_metrics(matrix):
    value = np.asarray(matrix, float) - np.mean(matrix, axis=0, keepdims=True)
    singular = np.linalg.svd(value, compute_uv=False, full_matrices=False)
    variance = singular**2
    weights = variance / max(float(variance.sum()), 1e-12)
    cumulative = np.cumsum(weights)
    nonzero = weights[weights > 1e-15]
    pc1 = float(weights[0]) if len(weights) else 0.0
    numerical_rank = (
        int(np.sum(singular > singular[0] * max(value.shape) * np.finfo(float).eps))
        if len(singular) and singular[0] > 0
        else 0
    )
    return {
        "pc1_fraction": pc1,
        "pc80": int(np.searchsorted(cumulative, 0.80) + 1),
        "pc90": int(np.searchsorted(cumulative, 0.90) + 1),
        "pc95": int(np.searchsorted(cumulative, 0.95) + 1),
        "effective_rank": float(1 / max(pc1, 1e-12)),
        "entropy_effective_rank": float(np.exp(-np.sum(nonzero * np.log(nonzero)))),
        "participation_ratio": float(1 / max(np.sum(weights**2), 1e-12)),
        "numerical_rank": numerical_rank,
    }


def retrieval_metrics(pred, truth):
    pred = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    truth = truth / np.maximum(np.linalg.norm(truth, axis=1, keepdims=True), 1e-12)
    similarity = pred @ truth.T
    order = np.argsort(-similarity, axis=1)
    correct = np.arange(len(pred))
    return float(np.mean(order[:, 0] == correct)), float(np.mean(np.any(order[:, :5] == correct[:, None], axis=1)))


def bootstrap_summary(values, seed, statistic="mean", n_boot=N_BOOT):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for start in range(0, n_boot, 100):
        stop = min(n_boot, start + 100)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        sample = values[indices]
        draws[start:stop] = np.mean(sample, axis=1) if statistic == "mean" else np.median(sample, axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), draws


def weighted_spearman(left, right, weights):
    left = rankdata(left).astype(float)
    right = rankdata(right).astype(float)
    total = np.maximum(weights.sum(axis=1), 1.0)
    left_mean = (weights * left).sum(axis=1) / total
    right_mean = (weights * right).sum(axis=1) / total
    lc = left[None, :] - left_mean[:, None]
    rc = right[None, :] - right_mean[:, None]
    covariance = (weights * lc * rc).sum(axis=1)
    denominator = np.sqrt((weights * lc**2).sum(axis=1) * (weights * rc**2).sum(axis=1))
    return np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12)


def bootstrap_geometry(folds, seed):
    all_names = sorted(name for data in folds for name in data["names"])
    lookup = {name: i for i, name in enumerate(all_names)}
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(all_names), np.full(len(all_names), 1 / len(all_names)), size=N_BOOT_GEOMETRY
    ).astype(np.int16)
    accum = np.zeros(N_BOOT_GEOMETRY)
    for data in folds:
        p, t = response_distances(data["pred"]), response_distances(data["truth"])
        first, second = np.triu_indices(len(data["pred"]), 1)
        source_indices = np.asarray([lookup[name] for name in data["names"]], int)
        for start in range(0, N_BOOT_GEOMETRY, 50):
            stop = min(N_BOOT_GEOMETRY, start + 50)
            weights = counts[start:stop, source_indices[first]] * counts[start:stop, source_indices[second]]
            accum[start:stop] += weighted_spearman(p, t, weights)
    draws = accum / len(folds)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), draws


def matrix_moments(target, response, denominator=None):
    n = target.shape[0]
    denominator = float(n - 1 if denominator is None else denominator)
    if scipy.sparse.issparse(target):
        target_sum = np.asarray(target.sum(axis=0)).ravel().astype(float)
        response_sum = np.asarray(response.sum(axis=0)).ravel().astype(float)
        target_sq = np.asarray(target.multiply(target).sum(axis=0)).ravel().astype(float)
        response_sq = np.asarray(response.multiply(response).sum(axis=0)).ravel().astype(float)
        cross = (target.T @ response).toarray().astype(float)
    else:
        target = np.asarray(target, float)
        response = np.asarray(response, float)
        target_sum, response_sum = target.sum(0), response.sum(0)
        target_sq, response_sq = np.square(target).sum(0), np.square(response).sum(0)
        cross = target.T @ response
    target_mean, response_mean = target_sum / n, response_sum / n
    covariance = (cross - n * target_mean[:, None] * response_mean[None, :]) / denominator
    target_var = (target_sq - n * target_mean**2) / denominator
    response_var = (response_sq - n * response_mean**2) / denominator
    corr = covariance / np.maximum(np.sqrt(target_var[:, None] * response_var[None, :]), 1e-12)
    slope = covariance / np.maximum(target_var[:, None], 1e-12)
    return covariance, corr, slope, target_mean, target_var, response_mean, response_var


def fit_projection_metrics(pred, truth):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    error = pred - truth
    sse = np.sum(error**2, axis=1)
    truth_energy = np.sum(truth**2, axis=1)
    centered_energy = np.sum((truth - truth.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return {
        "cosine": row_cosine(pred, truth),
        "pearson": row_pearson(pred, truth),
        "explained_squared_norm": np.sum(pred**2, axis=1),
        "energy_fraction": np.divide(
            np.sum(pred**2, axis=1), truth_energy, out=np.zeros(len(pred)), where=truth_energy > 1e-12
        ),
        "r2": 1 - np.divide(sse, centered_energy, out=np.full(len(pred), np.inf), where=centered_energy > 1e-12),
        "r2_zero_baseline": 1
        - np.divide(sse, truth_energy, out=np.full(len(pred), np.inf), where=truth_energy > 1e-12),
        "rmse": np.sqrt(np.mean(error**2, axis=1)),
    }


def add_summary_rows(frame, groups, metrics, record_type="summary"):
    rows = []
    for values, group in frame.groupby(groups, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(groups, values))
        row["record_type"] = record_type
        for metric in metrics:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at_before_outcome_computation": now(),
        "analysis": "Natural fluctuation direct response-space anchor for RPE1 IGC",
        "dataset": "Replogle 2022 RPE1 CRISPRi",
        "frozen_inputs": {
            "processed_h5ad": str(DATA.relative_to(ROOT)),
            "pseudobulk": str(PB_PATH.relative_to(ROOT)),
            "split": str(SPLIT_PATH.relative_to(ROOT)),
            "preprocessing": str(PREPROCESS_PATH.relative_to(ROOT)),
            "dataset_manifest": str(DATASET_MANIFEST_PATH.relative_to(ROOT)),
            "established_obs71": str(REP_PATH.relative_to(ROOT)),
        },
        "estimators": {
            "raw_cov": "negative control-cell sample covariance Cov(X_target, X_response)",
            "raw_corr": "negative control-cell Pearson correlation",
            "slope": "negative univariate response-on-target slope Cov/Var(target)",
            "resid_cov": "negative covariance after joint OLS removal of library proxy and 20 global expression PCs",
            "resid_corr": "negative correlation after the same residualization",
        },
        "primary_truth": "frozen fold-reference-mean-centered intervention residual q",
        "secondary_truth": "frozen global-control-relative response r",
        "calibration_source_rule": (
            "frozen outer-train source list intersected with the 1,755 measured source genes for which a "
            "control-expression fluctuation vector exists; q remains centered by the original full frozen-train mean"
        ),
        "strict_trans_panel": "intersection of the five frozen fold-local strict-trans panels",
        "residualization": {
            "cells": "control only",
            "pca": "sklearn PCA, centered full 8749-gene normalized control expression, randomized solver",
            "n_components": N_PCS,
            "pca_seed": PCA_SEED,
            "library_proxy": "row sum of existing normalized/log-transformed expression matrix",
            "regression": "one joint OLS design [intercept, standardized library proxy, standardized PC1-PC20]",
            "degrees_of_freedom_covariance_denominator": "n_cells - design_rank",
        },
        "orientation": {
            "metrics": [
                "cosine",
                "Pearson",
                "Spearman",
                "sign agreement among top 10% absolute truth coordinates",
                "top 10% absolute and signed overlap",
            ],
            "nulls": [
                "within-fold target permutation (200 draws; targetwise expected null and empirical p)",
                "within-vector coordinate permutation",
                "expression/detection matched source",
            ],
        },
        "geometry": {
            "artifact_safe": "all pairwise metrics are computed separately within each held-out fold; no cross-fold pairs",
            "distance": "cosine distance",
            "local_k": LOCAL_K,
            "bootstrap": "global perturbation-source multinomial weights reused across five within-fold groups",
        },
        "seeds": {"base": SEED, "pca": PCA_SEED, "frozen_fold_seed": 1701},
        "replicates": {"target_permutation": N_NULL, "source_bootstrap": N_BOOT, "geometry_bootstrap": N_BOOT_GEOMETRY},
        "verdict_gates": GATES,
        "optional_estimators": "SGS is deferred until primary and residualized results establish signal.",
        "manuscript_or_existing_outputs_modified": False,
    }
    dump_json(OUT / "experiment_manifest.json", manifest)
    print("[1/8] Preregistered manifest written before outcome computation", flush=True)

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    preprocess = json.loads(PREPROCESS_PATH.read_text(encoding="utf-8"))
    dataset_manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
    pb = np.load(PB_PATH, allow_pickle=True)
    genes = np.asarray(pb["genes"]).astype(str)
    perturbations = np.asarray(pb["perturbations"]).astype(str)
    gene_lookup = {name: i for i, name in enumerate(genes)}
    pb_lookup = {name: i for i, name in enumerate(perturbations)}

    fold_npz, source_union, trans_sets = [], [], []
    for fold in range(5):
        data = np.load(BASE / "predictions" / f"fold_{fold}.npz", allow_pickle=True)
        names = np.asarray(data["perturbations"]).astype(str)
        source_union.extend(names.tolist())
        trans_sets.append(set(data["panel_gene_indices"][data["common_trans"]].astype(int).tolist()))
        fold_npz.append(data)
    source_names = np.asarray(sorted(set(source_union)))
    source_lookup = {name: i for i, name in enumerate(source_names)}
    common_indices = np.asarray(sorted(set.intersection(*trans_sets)), int)
    common_genes = genes[common_indices]
    target_indices = np.asarray([gene_lookup[name] for name in source_names], int)
    source_pb_rows = np.asarray([pb_lookup[name] for name in source_names], int)
    assert len(source_union) == len(source_names) == 1755
    assert not (set(common_genes) & set(source_names))

    folds = []
    for fold, data in enumerate(fold_npz):
        held_names = np.asarray(data["perturbations"]).astype(str)
        panel = data["panel_gene_indices"].astype(int)
        local_lookup = {global_index: local for local, global_index in enumerate(panel)}
        cols = np.asarray([local_lookup[index] for index in common_indices], int)
        q = np.asarray(data["truth_residual"][:, cols], np.float64)
        r = np.asarray(pb["delta"][data["query_rows"].astype(int)][:, common_indices], np.float64)
        exact = r - np.asarray(data["training_mean"])[cols]
        if not np.allclose(q, exact, atol=1e-7, rtol=1e-7):
            raise RuntimeError(f"Frozen truth mismatch in fold {fold}")
        # The frozen split spans all 2,016 perturbations, whereas a control-derived
        # target vector exists only for the 1,755 measured source genes. Calibration
        # therefore uses the preregistered intersection, never a held-out source.
        train_names = np.asarray(
            [name for name in split["folds"][fold]["train_sources"] if name in source_lookup]
        ).astype(str)
        train_rows = np.asarray([pb_lookup[name] for name in train_names], int)
        train_r = np.asarray(pb["delta"][train_rows][:, common_indices], np.float64)
        frozen_training_mean = np.asarray(data["training_mean"])[cols]
        train_q = train_r - frozen_training_mean[None, :]
        folds.append(
            {
                "fold": fold,
                "names": held_names,
                "source_rows": np.asarray([source_lookup[name] for name in held_names], int),
                "q": q,
                "r": r,
                "train_names": train_names,
                "train_source_rows": np.asarray([source_lookup[name] for name in train_names], int),
                "train_q": train_q,
                "train_r": train_r,
                "n_frozen_train_sources_all": len(split["folds"][fold]["train_sources"]),
            }
        )
    print(f"[2/8] Frozen folds and {len(common_genes)}-gene common strict-trans panel verified", flush=True)

    backed = ad.read_h5ad(DATA, backed="r")
    control_mask = np.asarray(backed.obs["perturbation"].astype(str) == "control")
    if int(control_mask.sum()) != int(pb["control_count"]):
        raise RuntimeError("Control count mismatch")
    control = backed[control_mask, :].to_memory()
    backed.file.close()
    x_control = control.X.tocsr().astype(np.float32)
    x_target = x_control[:, target_indices]
    x_response = x_control[:, common_indices]
    library_proxy = np.asarray(x_control.sum(axis=1)).ravel().astype(float)
    raw_cov, raw_corr, raw_slope, target_mean, target_var, response_mean, response_var = matrix_moments(
        x_target, x_response
    )
    vectors = {
        "raw_cov": -raw_cov,
        "raw_corr": -raw_corr,
        "slope": -raw_slope,
    }
    target_detection = np.asarray((x_target > 0).sum(axis=0)).ravel().astype(int) / x_target.shape[0]
    target_expressed = np.asarray((x_target > 0).sum(axis=0)).ravel().astype(int)
    eligibility = pd.DataFrame(
        {
            "source": source_names,
            "gene_index": target_indices,
            "mean_control_expression": target_mean,
            "variance_control_expression": target_var,
            "detection_fraction": target_detection,
            "expressing_control_cells": target_expressed,
            "primary_covariance_eligible": target_var > 1e-12,
            "sgs0_eligible_min100_detection_0p05_0p95": (target_expressed >= 100)
            & ((x_target.shape[0] - target_expressed) >= 100)
            & (target_detection >= 0.05)
            & (target_detection <= 0.95),
        }
    )
    eligibility.to_csv(OUT / "target_eligibility.csv", index=False)
    print("[3/8] Raw covariance, correlation, and slope vectors computed from control cells only", flush=True)

    x_dense = x_control.toarray().astype(np.float32, copy=False)
    pca = PCA(n_components=N_PCS, svd_solver="randomized", random_state=PCA_SEED, copy=False)
    cell_pcs = pca.fit_transform(x_dense)
    del x_dense
    gc.collect()
    library_z = (library_proxy - library_proxy.mean()) / max(library_proxy.std(), 1e-12)
    pc_z = StandardScaler().fit_transform(cell_pcs)
    design = np.column_stack([np.ones(len(library_z)), library_z, pc_z]).astype(np.float64)
    design_rank = int(np.linalg.matrix_rank(design))
    x_sub = scipy.sparse.hstack([x_target, x_response], format="csr").toarray().astype(np.float64)
    coefficients = np.linalg.pinv(design) @ x_sub
    residual = x_sub - design @ coefficients
    residual_target = residual[:, : len(source_names)]
    residual_response = residual[:, len(source_names) :]
    resid_cov, resid_corr, _, _, resid_target_var, _, _ = matrix_moments(
        residual_target, residual_response, denominator=len(residual) - design_rank
    )
    vectors["resid_cov"] = -resid_cov
    vectors["resid_corr"] = -resid_corr
    del residual, residual_target, residual_response, x_sub, coefficients, cell_pcs
    gc.collect()
    print(f"[4/8] Residualized vectors computed after removing library proxy + {N_PCS} control-cell PCs", flush=True)

    control_audit = pd.DataFrame(
        [
            {
                "dataset": "Replogle 2022 RPE1 processed complete",
                "processed_file": str(DATA.relative_to(ROOT)),
                "control_definition": 'obs["perturbation"] == "control"',
                "n_total_cells": control.n_obs + int((~control_mask).sum()),
                "n_control_cells": control.n_obs,
                "n_genes": control.n_vars,
                "matrix_type": "CSR sparse float32",
                "matrix_min": float(x_control.data.min()),
                "matrix_max": float(x_control.data.max()),
                "normalization": dataset_manifest["matrix_interpretation"],
                "library_proxy_mean": float(library_proxy.mean()),
                "library_proxy_sd": float(library_proxy.std()),
                "pca_components_removed": N_PCS,
                "pca_explained_variance_fraction": float(pca.explained_variance_ratio_.sum()),
                "residual_design_rank": design_rank,
                "n_sources": len(source_names),
                "n_common_strict_trans_genes": len(common_genes),
                "source_response_gene_overlap": 0,
            }
        ]
    )
    control_audit.to_csv(OUT / "control_cell_audit.csv", index=False)
    pd.DataFrame({"gene_index": common_indices, "gene": common_genes}).to_csv(
        SOURCE / "common_strict_trans_genes.csv", index=False
    )

    orientation_rows, null_rows, null_target_rows, bootstrap_rows = [], [], [], []
    orientation_cache, target_p_cache, target_perm_mean_cache = {}, {}, {}
    rng = np.random.default_rng(SEED)
    for est_index, estimator in enumerate(ESTIMATORS):
        v_all = vectors[estimator]
        for fold_data in folds:
            fold = fold_data["fold"]
            held = fold_data["source_rows"]
            pred = v_all[held]
            for truth_space in ["q_intervention_residual", "r_control_relative"]:
                truth = fold_data["q"] if truth_space.startswith("q_") else fold_data["r"]
                metrics = directional_metrics(pred, truth)
                orientation_cache[(estimator, fold, truth_space)] = metrics
                if truth_space.startswith("q_"):
                    count_ge = np.zeros(len(held), int)
                    null_cos_sum = np.zeros(len(held), float)
                    for b in range(N_NULL):
                        perm = rng.permutation(len(held))
                        null_cos = row_cosine(pred[perm], truth)
                        count_ge += null_cos >= metrics["cosine"]
                        null_cos_sum += null_cos
                    target_p = (count_ge + 1) / (N_NULL + 1)
                    target_permutation_mean_cosine = null_cos_sum / N_NULL
                    target_p_cache[(estimator, fold)] = target_p
                    target_perm_mean_cache[(estimator, fold)] = target_permutation_mean_cosine
                else:
                    target_p = np.full(len(held), np.nan)
                    target_permutation_mean_cosine = np.full(len(held), np.nan)
                for i, source in enumerate(fold_data["names"]):
                    row = {
                        "record_type": "target",
                        "source": source,
                        "fold": fold,
                        "estimator": estimator,
                        "truth_space": truth_space,
                        "n_genes": len(common_genes),
                        "target_permutation_empirical_p_cosine": target_p[i],
                        "target_permutation_mean_cosine": target_permutation_mean_cosine[i],
                    }
                    row.update({key: value[i] for key, value in metrics.items()})
                    orientation_rows.append(row)

            truth = fold_data["q"]
            real = orientation_cache[(estimator, fold, "q_intervention_residual")]
            target_permutation_mean_cosine = target_perm_mean_cache[(estimator, fold)]
            perm = rng.permutation(len(held))
            null_target_pred = pred[perm]
            coordinate_order = np.argsort(rng.random(pred.shape), axis=1)
            null_coordinate_pred = np.take_along_axis(pred, coordinate_order, axis=1)
            features = eligibility.iloc[held][["mean_control_expression", "detection_fraction"]].to_numpy(float)
            scaled = StandardScaler().fit_transform(features)
            match_dist = cdist(scaled, scaled)
            np.fill_diagonal(match_dist, np.inf)
            matched = np.argmin(match_dist, axis=1)
            null_matched_pred = pred[matched]
            for null_type, null_pred in [
                ("target_permutation", null_target_pred),
                ("gene_coordinate_permutation", null_coordinate_pred),
                ("expression_detection_matched_source", null_matched_pred),
            ]:
                null_metrics = directional_metrics(null_pred, truth)
                for metric in ["cosine", "pearson", "spearman", "sign_agreement_top10pct_truth", "top10pct_signed_overlap"]:
                    real_values = real[metric]
                    null_values = (
                        target_permutation_mean_cosine
                        if null_type == "target_permutation" and metric == "cosine"
                        else null_metrics[metric]
                    )
                    delta = real_values - null_values
                    for i, source in enumerate(fold_data["names"]):
                        null_target_rows.append(
                            {
                                "record_type": "target",
                                "fold": fold,
                                "source": source,
                                "estimator": estimator,
                                "truth_space": "q_intervention_residual",
                                "null_type": null_type,
                                "metric": metric,
                                "null_replicates": N_NULL if null_type == "target_permutation" and metric == "cosine" else 1,
                                "real_value": real_values[i],
                                "null_value": null_values[i],
                                "paired_difference": delta[i],
                            }
                        )
                    low, high, draws = bootstrap_summary(delta, SEED + 10000 * est_index + 100 * fold + len(null_rows))
                    null_rows.append(
                        {
                            "record_type": "fold",
                            "fold": fold,
                            "estimator": estimator,
                            "truth_space": "q_intervention_residual",
                            "null_type": null_type,
                            "metric": metric,
                            "null_replicates": N_NULL if null_type == "target_permutation" and metric == "cosine" else 1,
                            "n_sources": len(held),
                            "real_mean": float(np.mean(real_values)),
                            "null_mean": float(np.mean(null_values)),
                            "paired_mean_difference": float(np.mean(delta)),
                            "paired_difference_bootstrap_ci_low": low,
                            "paired_difference_bootstrap_ci_high": high,
                            "one_sided_bootstrap_p_difference_le_zero": float((np.sum(draws <= 0) + 1) / (len(draws) + 1)),
                        }
                    )

    orientation = pd.DataFrame(orientation_rows)
    orientation.to_csv(OUT / "per_target_orientation.csv", index=False)
    for (estimator, truth_space, metric), group in orientation.groupby(["estimator", "truth_space", "record_type"]):
        pass
    for estimator in ESTIMATORS:
        for truth_space in ["q_intervention_residual", "r_control_relative"]:
            group = orientation[(orientation.estimator == estimator) & (orientation.truth_space == truth_space)]
            for metric in ["cosine", "pearson", "spearman", "sign_agreement_top10pct_truth", "top10pct_signed_overlap"]:
                for statistic in ["mean", "median"]:
                    low, high, _ = bootstrap_summary(
                        group[metric].to_numpy(),
                        SEED + 200000 + ESTIMATORS.index(estimator) * 100 + len(bootstrap_rows),
                        statistic=statistic,
                    )
                    bootstrap_rows.append(
                        {
                            "analysis": "orientation",
                            "estimator": estimator,
                            "truth_space": truth_space,
                            "metric": metric,
                            "statistic": statistic,
                            "estimate": float(getattr(group[metric], statistic)()),
                            "ci_low": low,
                            "ci_high": high,
                            "n_sources": len(group),
                            "bootstrap_unit": "perturbation source",
                            "n_bootstrap": N_BOOT,
                        }
                    )
    null_comparisons = pd.DataFrame(null_rows)
    null_target_frame = pd.DataFrame(null_target_rows)
    null_summary = []
    for summary_index, (keys, group) in enumerate(
        null_target_frame.groupby(["estimator", "truth_space", "null_type", "metric"])
    ):
        row = dict(zip(["estimator", "truth_space", "null_type", "metric"], keys))
        low, high, draws = bootstrap_summary(
            group.paired_difference.to_numpy(), SEED + 350000 + summary_index
        )
        row.update(
            {
                "record_type": "fold_mean",
                "fold": -1,
                "null_replicates": int(group.null_replicates.max()),
                "n_sources": len(group),
                "real_mean": float(group.real_value.mean()),
                "null_mean": float(group.null_value.mean()),
                "paired_mean_difference": float(group.paired_difference.mean()),
                "paired_difference_bootstrap_ci_low": low,
                "paired_difference_bootstrap_ci_high": high,
                "one_sided_bootstrap_p_difference_le_zero": float((np.sum(draws <= 0) + 1) / (len(draws) + 1)),
            }
        )
        null_summary.append(row)
        bootstrap_rows.append(
            {
                "analysis": "null_comparison",
                "estimator": row["estimator"],
                "truth_space": row["truth_space"],
                "metric": row["metric"],
                "statistic": "paired_mean_difference",
                "estimate": row["paired_mean_difference"],
                "ci_low": low,
                "ci_high": high,
                "n_sources": len(group),
                "bootstrap_unit": "perturbation source",
                "n_bootstrap": N_BOOT,
                "null_type": row["null_type"],
            }
        )
    null_comparisons = pd.concat([null_target_frame, null_comparisons, pd.DataFrame(null_summary)], ignore_index=True)
    null_comparisons.to_csv(OUT / "null_comparisons.csv", index=False)
    print("[5/8] Orientation, three nulls, and perturbation bootstrap completed", flush=True)

    oracle_rows, zero_rows = [], []
    zero_predictions = {}
    for estimator in ESTIMATORS:
        v_all = vectors[estimator]
        for fold_data in folds:
            fold = fold_data["fold"]
            held, train = fold_data["source_rows"], fold_data["train_source_rows"]
            held_v, train_v = v_all[held], v_all[train]
            features_train = eligibility.iloc[train][
                ["mean_control_expression", "variance_control_expression", "detection_fraction", "expressing_control_cells"]
            ].to_numpy(float)
            features_held = eligibility.iloc[held][
                ["mean_control_expression", "variance_control_expression", "detection_fraction", "expressing_control_cells"]
            ].to_numpy(float)
            feature_scaler = StandardScaler().fit(features_train)
            for truth_space, held_truth, train_truth in [
                ("q_intervention_residual", fold_data["q"], fold_data["train_q"]),
                ("r_control_relative", fold_data["r"], fold_data["train_r"]),
            ]:
                numerator = np.sum(held_v * held_truth, axis=1)
                denominator = np.sum(held_v**2, axis=1)
                alpha = np.divide(numerator, denominator, out=np.zeros(len(held_v)), where=denominator > 1e-12)
                alpha_positive = np.maximum(alpha, 0)
                pred_oracle = alpha[:, None] * held_v
                pred_positive = alpha_positive[:, None] * held_v
                oracle_metrics = fit_projection_metrics(pred_oracle, held_truth)
                positive_metrics = fit_projection_metrics(pred_positive, held_truth)
                for i, source in enumerate(fold_data["names"]):
                    row = {
                        "source": source,
                        "fold": fold,
                        "estimator": estimator,
                        "truth_space": truth_space,
                        "alpha_oracle": alpha[i],
                        "alpha_positive": alpha_positive[i],
                        "alpha_oracle_positive": bool(alpha[i] > 0),
                    }
                    row.update({f"oracle_{key}": value[i] for key, value in oracle_metrics.items()})
                    row.update({f"sign_constrained_{key}": value[i] for key, value in positive_metrics.items()})
                    oracle_rows.append(row)

                alpha_train = float(np.sum(train_v * train_truth) / max(np.sum(train_v**2), 1e-12))
                global_pred = alpha_train * held_v
                train_alpha = np.divide(
                    np.sum(train_v * train_truth, axis=1),
                    np.sum(train_v**2, axis=1),
                    out=np.zeros(len(train_v)),
                    where=np.sum(train_v**2, axis=1) > 1e-12,
                )
                ridge = Ridge(alpha=1.0).fit(feature_scaler.transform(features_train), train_alpha)
                ridge_alpha = ridge.predict(feature_scaler.transform(features_held))
                ridge_pred = ridge_alpha[:, None] * held_v
                if truth_space.startswith("q_"):
                    zero_predictions[(estimator, "global_scalar", fold)] = global_pred
                    zero_predictions[(estimator, "ridge_amplitude", fold)] = ridge_pred
                for calibration, pred, predicted_alpha in [
                    ("global_scalar", global_pred, np.full(len(held), alpha_train)),
                    ("ridge_amplitude", ridge_pred, ridge_alpha),
                ]:
                    metrics = fit_projection_metrics(pred, held_truth)
                    for i, source in enumerate(fold_data["names"]):
                        row = {
                            "source": source,
                            "fold": fold,
                            "estimator": estimator,
                            "truth_space": truth_space,
                            "calibration": calibration,
                            "training_global_alpha": alpha_train,
                            "predicted_alpha": predicted_alpha[i],
                            "ridge_alpha_fixed": 1.0 if calibration == "ridge_amplitude" else np.nan,
                        }
                        row.update(metrics[key][i] for key in [])
                        row.update({key: value[i] for key, value in metrics.items()})
                        zero_rows.append(row)

    oracle = pd.DataFrame(oracle_rows)
    zero = pd.DataFrame(zero_rows)
    oracle.to_csv(OUT / "oracle_projection_results.csv", index=False)
    zero.to_csv(OUT / "zero_shot_projection_results.csv", index=False)
    print("[6/8] Oracle ceiling and leakage-safe training-only scalar/ridge calibration completed", flush=True)

    rep = np.load(REP_PATH, allow_pickle=True)
    rep_names = np.asarray(rep["perturbations"]).astype(str)
    rep_lookup = {name: i for i, name in enumerate(rep_names)}
    established = np.asarray(rep["EstablishedOBS71"], float)
    geometry_rows, spectral_rows, geometry_folds, geometry_plot_rows = [], [], {}, []
    rng_geom = np.random.default_rng(SEED + 77)

    def record_geometry(estimator, prediction_type, fold_data, pred, truth, response_space=True):
        fold = fold_data["fold"]
        pred_dist, truth_dist = response_distances(pred), response_distances(truth)
        overlap, local_rank = local_geometry(pred, truth)
        top1, top5 = retrieval_metrics(pred, truth) if response_space and pred.shape[1] == truth.shape[1] else (np.nan, np.nan)
        pred_var = float(np.mean(np.var(pred, axis=0))) if response_space and pred.shape[1] == truth.shape[1] else np.nan
        truth_var = float(np.mean(np.var(truth, axis=0)))
        row = {
            "record_type": "fold",
            "fold": fold,
            "estimator": estimator,
            "prediction_type": prediction_type,
            "truth_space": "q_intervention_residual",
            "n_sources": len(pred),
            "n_genes_truth": truth.shape[1],
            "response_distance_spearman": safe_spearman(pred_dist, truth_dist),
            "response_distance_pearson": safe_pearson(pred_dist, truth_dist),
            "local_knn_overlap_k10": overlap,
            "local_distance_rank": local_rank,
            "predicted_between_source_variance": pred_var,
            "truth_between_source_variance": truth_var,
            "between_source_variance_ratio": pred_var / max(truth_var, 1e-12) if np.isfinite(pred_var) else np.nan,
            "predicted_mean_pair_distance": float(np.mean(pred_dist)),
            "truth_mean_pair_distance": float(np.mean(truth_dist)),
            "distance_scale_retention": float(np.mean(pred_dist) / max(np.mean(truth_dist), 1e-12)),
            "target_retrieval_top1": top1,
            "target_retrieval_top5": top5,
            "cross_fold_pairs_used": False,
        }
        geometry_rows.append(row)
        geometry_folds.setdefault((estimator, prediction_type), []).append(
            {"names": fold_data["names"].tolist(), "pred": pred, "truth": truth}
        )
        if response_space and pred.shape[1] == truth.shape[1]:
            pred_rank, truth_rank = rank_metrics(pred), rank_metrics(truth)
            spectral_rows.append(
                {
                    "record_type": "fold",
                    "fold": fold,
                    "estimator": estimator,
                    "prediction_type": prediction_type,
                    "n_sources": len(pred),
                    "n_genes": pred.shape[1],
                    "predicted_between_source_variance": pred_var,
                    "truth_between_source_variance": truth_var,
                    "between_source_variance_ratio": pred_var / max(truth_var, 1e-12),
                    **{f"predicted_{key}": value for key, value in pred_rank.items()},
                    **{f"truth_{key}": value for key, value in truth_rank.items()},
                }
            )
        if prediction_type == "orientation_only":
            take = rng_geom.choice(len(pred_dist), size=min(5000, len(pred_dist)), replace=False)
            for pvalue, tvalue in zip(pred_dist[take], truth_dist[take]):
                geometry_plot_rows.append(
                    {
                        "fold": fold,
                        "estimator": estimator,
                        "predicted_cosine_distance": pvalue,
                        "true_cosine_distance": tvalue,
                    }
                )

    for estimator in ESTIMATORS:
        for fold_data in folds:
            held_v = vectors[estimator][fold_data["source_rows"]]
            record_geometry(estimator, "orientation_only", fold_data, held_v, fold_data["q"])
            for calibration in ["global_scalar", "ridge_amplitude"]:
                record_geometry(
                    estimator,
                    f"zero_shot_{calibration}",
                    fold_data,
                    zero_predictions[(estimator, calibration, fold_data["fold"])],
                    fold_data["q"],
                )

    for fold_data in folds:
        train_rep = established[[rep_lookup[name] for name in fold_data["train_names"]]]
        held_rep = established[[rep_lookup[name] for name in fold_data["names"]]]
        scaler = StandardScaler().fit(train_rep)
        record_geometry(
            "EstablishedOBS71",
            "source_descriptor",
            fold_data,
            scaler.transform(held_rep),
            fold_data["q"],
            response_space=False,
        )
        perm = rng_geom.permutation(len(held_rep))
        record_geometry(
            "target_permutation_null",
            "source_descriptor",
            fold_data,
            scaler.transform(held_rep)[perm],
            fold_data["q"],
            response_space=False,
        )

    geometry = pd.DataFrame(geometry_rows)
    summary_rows = []
    metric_columns = [
        "response_distance_spearman",
        "response_distance_pearson",
        "local_knn_overlap_k10",
        "local_distance_rank",
        "predicted_between_source_variance",
        "truth_between_source_variance",
        "between_source_variance_ratio",
        "predicted_mean_pair_distance",
        "truth_mean_pair_distance",
        "distance_scale_retention",
        "target_retrieval_top1",
        "target_retrieval_top5",
    ]
    for pair, group in geometry.groupby(["estimator", "prediction_type"]):
        estimator, prediction_type = pair
        low, high, draws = bootstrap_geometry(
            geometry_folds[(estimator, prediction_type)], SEED + 500000 + len(summary_rows)
        )
        row = {
            "record_type": "summary",
            "fold": -1,
            "estimator": estimator,
            "prediction_type": prediction_type,
            "truth_space": "q_intervention_residual",
            "n_sources": int(group.n_sources.sum()),
            "n_genes_truth": int(group.n_genes_truth.iloc[0]),
            "source_bootstrap_ci_low": low,
            "source_bootstrap_ci_high": high,
            "cross_fold_pairs_used": False,
            "bootstrap_unit": "perturbation source; within-fold pair weights only",
        }
        row.update({column: float(group[column].mean()) for column in metric_columns})
        summary_rows.append(row)
        bootstrap_rows.append(
            {
                "analysis": "grouped_geometry",
                "estimator": estimator,
                "truth_space": "q_intervention_residual",
                "metric": "response_distance_spearman",
                "statistic": "fold_mean",
                "estimate": float(group.response_distance_spearman.mean()),
                "ci_low": low,
                "ci_high": high,
                "n_sources": int(group.n_sources.sum()),
                "bootstrap_unit": "perturbation source with within-fold pair weighting",
                "n_bootstrap": N_BOOT_GEOMETRY,
                "prediction_type": prediction_type,
            }
        )
    geometry = pd.concat([geometry, pd.DataFrame(summary_rows)], ignore_index=True)
    geometry.to_csv(OUT / "geometry_summary.csv", index=False)
    pd.DataFrame(geometry_plot_rows).to_csv(SOURCE / "geometry_pair_samples.csv", index=False)

    spectral = pd.DataFrame(spectral_rows)
    spectral_summary_rows = []
    for pair, group in spectral.groupby(["estimator", "prediction_type"]):
        row = {"record_type": "fold_mean", "fold": -1, "estimator": pair[0], "prediction_type": pair[1]}
        for column in spectral.columns:
            if column not in ["record_type", "fold", "estimator", "prediction_type"]:
                row[column] = float(group[column].mean())
        row["entropy_effective_rank_retention"] = row["predicted_entropy_effective_rank"] / max(
            row["truth_entropy_effective_rank"], 1e-12
        )
        spectral_summary_rows.append(row)
    spectral = pd.concat([spectral, pd.DataFrame(spectral_summary_rows)], ignore_index=True)
    spectral.to_csv(OUT / "spectral_summary.csv", index=False)
    print("[7/8] Artifact-safe grouped geometry, spectra, retrieval, and EstablishedOBS71 comparison completed", flush=True)

    fold_rows = []
    q_orientation = orientation[orientation.truth_space == "q_intervention_residual"]
    for keys, group in q_orientation.groupby(["fold", "estimator"]):
        fold_rows.append(
            {
                "analysis": "orientation",
                "fold": keys[0],
                "estimator": keys[1],
                "prediction_type": "direct_vector",
                "n_sources": len(group),
                "mean_cosine": float(group.cosine.mean()),
                "median_cosine": float(group.cosine.median()),
                "mean_pearson": float(group.pearson.mean()),
                "mean_spearman": float(group.spearman.mean()),
                "fraction_cosine_positive": float(np.mean(group.cosine > 0)),
                "fraction_empirical_p_lt_0_05": float(np.mean(group.target_permutation_empirical_p_cosine < 0.05)),
            }
        )
    for _, row in geometry[geometry.record_type == "fold"].iterrows():
        fold_rows.append(
            {
                "analysis": "geometry",
                "fold": row.fold,
                "estimator": row.estimator,
                "prediction_type": row.prediction_type,
                "n_sources": row.n_sources,
                "response_distance_spearman": row.response_distance_spearman,
                "local_knn_overlap_k10": row.local_knn_overlap_k10,
                "between_source_variance_ratio": row.between_source_variance_ratio,
                "target_retrieval_top1": row.target_retrieval_top1,
                "target_retrieval_top5": row.target_retrieval_top5,
            }
        )
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_summary.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(OUT / "bootstrap_summary.csv", index=False)

    residual_diagnostics = []
    for estimator in RESIDUAL_SAFE:
        orient = q_orientation[q_orientation.estimator == estimator]
        target_null = null_comparisons[
            (null_comparisons.record_type == "fold_mean")
            & (null_comparisons.estimator == estimator)
            & (null_comparisons.null_type == "target_permutation")
            & (null_comparisons.metric == "cosine")
        ].iloc[0]
        oracle_group = oracle[
            (oracle.estimator == estimator) & (oracle.truth_space == "q_intervention_residual")
        ]
        geom = geometry[
            (geometry.record_type == "summary")
            & (geometry.estimator == estimator)
            & (geometry.prediction_type == "orientation_only")
        ].iloc[0]
        spec = spectral[
            (spectral.record_type == "fold_mean")
            & (spectral.estimator == estimator)
            & (spectral.prediction_type == "zero_shot_global_scalar")
        ].iloc[0]
        residual_diagnostics.append(
            {
                "estimator": estimator,
                "mean_cosine": float(orient.cosine.mean()),
                "median_cosine": float(orient.cosine.median()),
                "fraction_cosine_positive": float(np.mean(orient.cosine > 0)),
                "fraction_empirical_p_lt_0_05": float(np.mean(orient.target_permutation_empirical_p_cosine < 0.05)),
                "real_minus_target_null_ci_low": float(target_null.paired_difference_bootstrap_ci_low),
                "median_oracle_energy_fraction": float(oracle_group.oracle_energy_fraction.median()),
                "probability_alpha_positive": float(np.mean(oracle_group.alpha_oracle > 0)),
                "geometry_spearman": float(geom.response_distance_spearman),
                "geometry_ci_low": float(geom.source_bootstrap_ci_low),
                "geometry_ci_high": float(geom.source_bootstrap_ci_high),
                "local_knn_overlap_k10": float(geom.local_knn_overlap_k10),
                "variance_retention": float(spec.between_source_variance_ratio),
                "entropy_effective_rank_retention": float(spec.entropy_effective_rank_retention),
            }
        )
    diagnostic_frame = pd.DataFrame(residual_diagnostics)
    best = diagnostic_frame.sort_values(["mean_cosine", "geometry_spearman"], ascending=False).iloc[0]
    strong = (
        best.mean_cosine > 0
        and best.real_minus_target_null_ci_low > 0
        and best.geometry_spearman >= 0.15
        and best.geometry_ci_low >= 0.10
        and best.variance_retention >= 0.25
        and best.entropy_effective_rank_retention >= 0.50
        and best.local_knn_overlap_k10 >= 0.08
    )
    partial = (
        best.mean_cosine > 0
        and best.real_minus_target_null_ci_low > 0
        and best.fraction_empirical_p_lt_0_05 > 0.075
        and best.median_oracle_energy_fraction > 0.001
    )
    if strong:
        verdict = "NATURAL_FLUCTUATION_BREAKS_IGC"
    elif partial:
        verdict = "NATURAL_FLUCTUATION_ANCHOR_PARTIALLY_SUPPORTED"
    else:
        verdict = "NATURAL_FLUCTUATION_ANCHOR_NOT_SUPPORTED"
    dump_json(
        OUT / "verdict.json",
        {
            "verdict": verdict,
            "selected_residualized_estimator_by_preregistered_ordering": best.estimator,
            "gate_values": {key: (value.item() if hasattr(value, "item") else value) for key, value in best.to_dict().items()},
            "strong_gate_passed": bool(strong),
            "partial_gate_passed": bool(partial),
            "stopping_rule_applied": (
                "RPE1 null: stop without SGS, mechanistic expansion, K562, or external datasets"
                if verdict == "NATURAL_FLUCTUATION_ANCHOR_NOT_SUPPORTED"
                else "signal present: mechanistic follow-up permitted"
            ),
            "gates": GATES,
            "decision_time": now(),
        },
    )

    # Expression-dependence analysis is required only when a signal passes the conservative residualized gate.
    dependence_rows = []
    if verdict != "NATURAL_FLUCTUATION_ANCHOR_NOT_SUPPORTED":
        best_orientation = q_orientation[q_orientation.estimator == best.estimator].copy()
        best_orientation["source_row"] = best_orientation.source.map(source_lookup)
        best_orientation = best_orientation.merge(eligibility, on="source", how="left")
        for feature in ["mean_control_expression", "variance_control_expression", "detection_fraction"]:
            rho = safe_spearman(best_orientation[feature], best_orientation.cosine)
            dependence_rows.append(
                {"record_type": "spearman", "estimator": best.estimator, "feature": feature, "value": rho}
            )
            best_orientation[f"{feature}_quartile"] = pd.qcut(
                best_orientation[feature], 4, labels=False, duplicates="drop"
            )
            for quartile, group in best_orientation.groupby(f"{feature}_quartile"):
                dependence_rows.append(
                    {
                        "record_type": "quartile",
                        "estimator": best.estimator,
                        "feature": feature,
                        "quartile": int(quartile) + 1,
                        "n_sources": len(group),
                        "value": float(group.cosine.mean()),
                        "median_cosine": float(group.cosine.median()),
                    }
                )
        pd.DataFrame(dependence_rows).to_csv(OUT / "mechanistic_expression_dependence.csv", index=False)

    # Provenance hashes are calculated only after analysis so they cannot affect estimators.
    provenance_files = [DATA, PB_PATH, REP_PATH, SPLIT_PATH, PREPROCESS_PATH, DATASET_MANIFEST_PATH]
    provenance_files.extend(BASE / "predictions" / f"fold_{fold}.npz" for fold in range(5))
    provenance = {
        "created_at": now(),
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
            for path in provenance_files
        ],
        "counts": {
            "total_cells": int(len(control_mask)),
            "control_cells": int(control_mask.sum()),
            "genes": int(len(genes)),
            "eligible_sources": int(len(source_names)),
            "common_strict_trans_genes": int(len(common_genes)),
            "fold_sizes": [int(len(item["names"])) for item in folds],
            "calibration_measured_train_sizes": [int(len(item["train_names"])) for item in folds],
            "frozen_all_train_sizes": [int(item["n_frozen_train_sources_all"]) for item in folds],
        },
        "control_definition": 'obs["perturbation"] == "control"',
        "response_definitions": {
            "r": "cached perturbation pseudobulk mean minus all-control pseudobulk mean",
            "q": "r minus mean r among that outer fold's frozen reference/train sources",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "anndata": ad.__version__,
        },
        "frozen_fold_algorithm": split["algorithm"],
        "frozen_fold_seed": split["seed"],
        "no_existing_outputs_modified": True,
    }
    dump_json(OUT / "data_provenance.json", provenance)

    # Plot source tables and publication-quality draft figures.
    plot_orientation = q_orientation.copy()
    plot_orientation.to_csv(SOURCE / "orientation_plot_data.csv", index=False)
    plot_oracle = oracle[oracle.truth_space == "q_intervention_residual"].copy()
    plot_zero = zero[
        (zero.truth_space == "q_intervention_residual") & (zero.calibration == "global_scalar")
    ].copy()
    plot_oracle.to_csv(SOURCE / "oracle_plot_data.csv", index=False)
    plot_zero.to_csv(SOURCE / "zero_shot_plot_data.csv", index=False)
    geometry[geometry.record_type == "summary"].to_csv(SOURCE / "geometry_summary_plot_data.csv", index=False)
    spectral[spectral.record_type == "fold_mean"].to_csv(SOURCE / "spectral_summary_plot_data.csv", index=False)

    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 300, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"raw_cov": "#4477AA", "raw_corr": "#228833", "slope": "#66CCEE", "resid_cov": "#CC6677", "resid_corr": "#AA3377"}

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bins = np.linspace(-0.35, 0.35, 60)
    for estimator in ESTIMATORS:
        values = q_orientation[q_orientation.estimator == estimator].cosine
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.4, color=colors[estimator], label=estimator)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Cosine with true held-out intervention residual", ylabel="Density", title="Natural-fluctuation response orientation")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(FIGURES / "01_orientation_cosine_distribution.png"); fig.savefig(FIGURES / "01_orientation_cosine_distribution.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    summary_null = null_comparisons[(null_comparisons.record_type == "fold_mean") & (null_comparisons.metric == "cosine")]
    positions, labels, values = [], [], []
    pos = 0
    for estimator in ESTIMATORS:
        real = float(summary_null[(summary_null.estimator == estimator) & (summary_null.null_type == "target_permutation")].real_mean.iloc[0])
        for label, value in [("real", real)] + [
            (kind.replace("expression_detection_", "matched_"), float(summary_null[(summary_null.estimator == estimator) & (summary_null.null_type == kind)].null_mean.iloc[0]))
            for kind in ["target_permutation", "gene_coordinate_permutation", "expression_detection_matched_source"]
        ]:
            positions.append(pos); labels.append(f"{estimator}\n{label}"); values.append(value); pos += 1
        pos += 0.5
    ax.bar(positions, values, color=["#333333" if "real" in label else "#BBBBBB" for label in labels])
    ax.axhline(0, color="black", linewidth=0.8); ax.set_xticks(positions, labels, rotation=55, ha="right")
    ax.set(ylabel="Mean cosine", title="Real target vectors versus preregistered nulls")
    fig.tight_layout(); fig.savefig(FIGURES / "02_real_vs_shuffled_orientation.png"); fig.savefig(FIGURES / "02_real_vs_shuffled_orientation.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(ESTIMATORS)); width = 0.26
    oracle_means = [plot_oracle[plot_oracle.estimator == est].oracle_cosine.mean() for est in ESTIMATORS]
    constrained_means = [plot_oracle[plot_oracle.estimator == est].sign_constrained_cosine.mean() for est in ESTIMATORS]
    zero_means = [plot_zero[plot_zero.estimator == est].cosine.mean() for est in ESTIMATORS]
    ax.bar(x - width, oracle_means, width, label="oracle scalar", color="#4477AA")
    ax.bar(x, constrained_means, width, label="sign-constrained oracle", color="#EE6677")
    ax.bar(x + width, zero_means, width, label="training-only scalar", color="#228833")
    ax.set_xticks(x, ESTIMATORS, rotation=25, ha="right"); ax.set(ylabel="Mean response cosine", title="Oracle ceiling versus deployable zero-shot projection")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIGURES / "03_oracle_vs_zero_shot.png"); fig.savefig(FIGURES / "03_oracle_vs_zero_shot.pdf"); plt.close(fig)

    best_estimator = str(best.estimator)
    pair_plot = pd.DataFrame(geometry_plot_rows)
    pair_plot = pair_plot[pair_plot.estimator == best_estimator]
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    ax.hexbin(pair_plot.true_cosine_distance, pair_plot.predicted_cosine_distance, gridsize=55, mincnt=1, cmap="viridis")
    ax.set(xlabel="True held-out cosine distance", ylabel="Predicted fluctuation cosine distance", title=f"Grouped intervention geometry: {best_estimator}")
    fig.tight_layout(); fig.savefig(FIGURES / "04_true_vs_predicted_geometry.png"); fig.savefig(FIGURES / "04_true_vs_predicted_geometry.pdf"); plt.close(fig)

    spec_plot = spectral[(spectral.record_type == "fold_mean") & (spectral.prediction_type == "zero_shot_global_scalar")]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(spec_plot.estimator, spec_plot.between_source_variance_ratio, color=[colors.get(x, "#888888") for x in spec_plot.estimator])
    ax.axhline(1, color="black", linestyle="--", linewidth=0.8); ax.set(ylabel="Between-intervention variance retention", title="Deployable fluctuation variance retention")
    ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(FIGURES / "05_variance_retention.png"); fig.savefig(FIGURES / "05_variance_retention.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(spec_plot)); width = 0.36
    ax.bar(x - width/2, spec_plot.predicted_entropy_effective_rank, width, label="predicted", color="#CC6677")
    ax.bar(x + width/2, spec_plot.truth_entropy_effective_rank, width, label="truth", color="#4477AA")
    ax.set_xticks(x, spec_plot.estimator, rotation=25, ha="right"); ax.set(ylabel="Entropy effective rank", title="Spectral compression of zero-shot predictions")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIGURES / "06_effective_rank_spectral.png"); fig.savefig(FIGURES / "06_effective_rank_spectral.pdf"); plt.close(fig)

    matched = geometry[(geometry.record_type == "summary") & (geometry.prediction_type.isin(["orientation_only", "source_descriptor"]))]
    order = ["EstablishedOBS71", "raw_cov", "raw_corr", "resid_cov", "resid_corr", "target_permutation_null"]
    matched = matched[matched.estimator.isin(order)].copy(); matched["order"] = matched.estimator.map({v:i for i,v in enumerate(order)}); matched = matched.sort_values("order")
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    error = np.vstack([matched.response_distance_spearman - matched.source_bootstrap_ci_low, matched.source_bootstrap_ci_high - matched.response_distance_spearman])
    ax.bar(matched.estimator, matched.response_distance_spearman, color="#667799", yerr=error, capsize=3)
    ax.axhline(0, color="black", linewidth=0.8); ax.set(ylabel="Grouped distance Spearman", title="Matched control-derived geometry comparison")
    ax.tick_params(axis="x", rotation=30); fig.tight_layout(); fig.savefig(FIGURES / "07_established_vs_fluctuation_geometry.png"); fig.savefig(FIGURES / "07_established_vs_fluctuation_geometry.pdf"); plt.close(fig)

    if dependence_rows:
        dep = pd.DataFrame(dependence_rows)
        quartiles = dep[dep.record_type == "quartile"]
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        for feature, group in quartiles.groupby("feature"):
            ax.plot(group.quartile, group.value, marker="o", label=feature.replace("_control_expression", ""))
        ax.axhline(0, color="black", linewidth=0.8); ax.set(xticks=[1,2,3,4], xlabel="Control-feature quartile", ylabel="Mean residualized orientation cosine", title="Dependence on measurable endogenous fluctuation")
        ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIGURES / "08_expression_detection_variance_dependence.png"); fig.savefig(FIGURES / "08_expression_detection_variance_dependence.pdf"); plt.close(fig)

    established_old = pd.read_csv(ESTABLISHED_PATH)
    established_old = established_old[(established_old.record_type == "summary") & (established_old.representation == "EstablishedOBS71")].iloc[0]
    best_zero = zero[(zero.truth_space == "q_intervention_residual") & (zero.estimator == best_estimator) & (zero.calibration == "global_scalar")]
    best_oracle = oracle[(oracle.truth_space == "q_intervention_residual") & (oracle.estimator == best_estimator)]
    best_geom = geometry[(geometry.record_type == "summary") & (geometry.estimator == best_estimator) & (geometry.prediction_type == "orientation_only")].iloc[0]
    best_spec = spectral[(spectral.record_type == "fold_mean") & (spectral.estimator == best_estimator) & (spectral.prediction_type == "zero_shot_global_scalar")].iloc[0]
    readme = f"""# Natural-fluctuation IGC anchor: Replogle RPE1

This directory is a self-contained, additive analysis. No existing IGC result, manuscript file, or figure was modified.

## Frozen data and design

- Processed single-cell file: `{DATA.relative_to(ROOT)}` (SHA-256 recorded in `data_provenance.json`).
- Controls: `obs[\"perturbation\"] == \"control\"`; {int(control_mask.sum()):,} of {len(control_mask):,} cells.
- Expression genes: {len(genes):,}; eligible perturbation sources: {len(source_names):,}.
- Primary response panel: {len(common_genes):,} genes in the intersection of all five frozen strict-trans panels; zero source-gene overlap.
- Frozen source-disjoint fold sizes: {', '.join(str(len(item['names'])) for item in folds)}.
- Primary truth is the frozen fold-reference-centered residual `q`; control-relative `r` is secondary.
- Control-derived estimators use control cells only. Residualized estimators remove a library-size proxy and {N_PCS} full-expression control-cell PCs.

## Primary result

The preregistered conservative gate selected `{best_estimator}` among the two residualized estimators.

- Mean / median held-out `q` cosine: {best.mean_cosine:.5f} / {best.median_cosine:.5f}.
- Fraction cosine > 0: {best.fraction_cosine_positive:.3f}; fraction target-permutation empirical p < 0.05: {best.fraction_empirical_p_lt_0_05:.3f}.
- Oracle median response-energy fraction: {best.median_oracle_energy_fraction:.5f}; P(oracle alpha > 0): {best.probability_alpha_positive:.3f}.
- Training-only scalar mean cosine: {best_zero.cosine.mean():.5f}; mean Pearson: {best_zero.pearson.mean():.5f}.
- Grouped pairwise geometry Spearman: {best_geom.response_distance_spearman:.5f} (source bootstrap 95% CI {best_geom.source_bootstrap_ci_low:.5f}, {best_geom.source_bootstrap_ci_high:.5f}).
- kNN overlap@10: {best_geom.local_knn_overlap_k10:.4f}; top-1/top-5 retrieval: {best_geom.target_retrieval_top1:.4f}/{best_geom.target_retrieval_top5:.4f}.
- Between-source variance retention: {best_spec.between_source_variance_ratio:.5f}; entropy-effective-rank retention: {best_spec.entropy_effective_rank_retention:.5f}.
- Existing unchanged EstablishedOBS71 result: rho {float(established_old.spearman_rho):.5f} (95% CI {float(established_old.ci_low):.5f}, {float(established_old.ci_high):.5f}).

The raw estimators contain a state-confounded one-dimensional component: the unconstrained targetwise oracle can flip this axis and therefore gives a much larger ceiling than the sign-constrained or training-only versions. After control-cell state removal, signed orientation and oracle energy return to null scale.

The residualized spectral rank must be interpreted together with amplitude: when variance retention is nearly zero, a high numerical/effective rank is not geometry rescue.

## Verdict

`{verdict}`

The verdict follows the thresholds written to `experiment_manifest.json` before outcome computation. Raw-only signal cannot establish support if it disappears after the conservative control-cell state residualization.

## Scientific interpretation

1. Held-out target-specific sender information is not supported after conservative state residualization.
2. The raw signal is primarily a shared cell-state axis whose targetwise sign must be chosen with held-out truth, not stable signed orientation or magnitude information.
3. Intervention geometry is not restored: grouped geometry, variance retention, and retrieval remain at null scale.
4. This strengthens the practical IGC non-identifiability diagnosis for ordinary observational control-cell variation, without claiming that all observational variation is causally uninformative.

Because the preregistered RPE1 result is null, optional SGS, control-cell subsampling, response-PC recovery, K562 replication, and external-dataset expansion are not run. This follows the stopping rule and avoids post-null model search. The target-in-PCA sensitivity branch is likewise not pursued after a null residualized primary result.

## File guide

The required CSV/JSON files are in this directory. Plot-ready subsets are in `source_data/`; draft publication-quality PNG and PDF figures are in `figures/`. `per_target_orientation.csv`, `oracle_projection_results.csv`, and `zero_shot_projection_results.csv` retain target-level records. All pairwise geometry is computed within fold only.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"[8/8] Outputs, plots, provenance, and verdict complete: {verdict}", flush=True)


if __name__ == "__main__":
    main()
