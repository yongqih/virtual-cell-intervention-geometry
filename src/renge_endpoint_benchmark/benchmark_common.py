from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = ROOT / "scripts" / "renge_endpoint_benchmark"
RESULT_ROOT = ROOT / "results" / "renge_endpoint_benchmark"
CACHE_ROOT = RESULT_ROOT / "cache"
FROZEN_CACHE = ROOT / "data" / "propagation_reproduction" / "cache" / "renge_processed.npz"

sys.path.insert(0, str(ROOT / "scripts" / "renge_first_wave_program"))
from program_common import (  # noqa: E402
    apply_affine, direct_indices, fit_dense_transition,
    geometry_rho, grouped_twofold_splits, prediction_metrics, safe_pearson,
    safe_spearman, standardized_ridge, strict_trans_cosine_distance,
    transmission_representations,
)
sys.path.insert(0, str(ROOT / "scripts" / "renge_program_identifiability"))
from audit_common import full_dynamic_descriptor  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def response_cosines(prediction: np.ndarray, truth: np.ndarray, sources: np.ndarray,
                     genes: np.ndarray) -> np.ndarray:
    direct = direct_indices(sources, genes); out = []
    for i, (a, b) in enumerate(zip(prediction, truth)):
        keep = np.ones(len(genes), bool)
        if direct[i] >= 0: keep[direct[i]] = False
        out.append(float(a[keep] @ b[keep] / max(np.linalg.norm(a[keep]) * np.linalg.norm(b[keep]), 1e-12)))
    return np.asarray(out)


def rank_metrics(matrix: np.ndarray, sources: np.ndarray, genes: np.ndarray) -> dict:
    value = np.asarray(matrix, float).copy(); direct = direct_indices(sources, genes)
    valid = direct >= 0; value[np.arange(len(value))[valid], direct[valid]] = 0
    value -= value.mean(0, keepdims=True)
    s = np.linalg.svd(value, compute_uv=False, full_matrices=False); q = s * s
    w = q / max(q.sum(), 1e-12); sw = s / max(s.sum(), 1e-12); c = np.cumsum(w)
    nz = sw[sw > 1e-12]
    return {"pc1_fraction": float(w[0]), "pc80": int(np.searchsorted(c, .8) + 1),
            "pc90": int(np.searchsorted(c, .9) + 1), "pc95": int(np.searchsorted(c, .95) + 1),
            "entropy_effective_rank": float(np.exp(-np.sum(nz * np.log(nz)))),
            "participation_ratio": float(1 / max(np.sum(w * w), 1e-12)),
            "between_variance": float(np.mean(np.var(value, axis=0)))}


def local_metrics(prediction: np.ndarray, truth: np.ndarray, sources: np.ndarray,
                  genes: np.ndarray, k: int) -> tuple[float, float]:
    direct = direct_indices(sources, genes)
    pd = strict_trans_cosine_distance(prediction, direct); td = strict_trans_cosine_distance(truth, direct)
    overlaps, ranks = [], []
    for i in range(len(prediction)):
        keep = np.arange(len(prediction)) != i; a, b = pd[i, keep], td[i, keep]
        kk = min(k, len(a)); overlaps.append(len(set(np.argsort(a)[:kk]) & set(np.argsort(b)[:kk])) / kk)
        ranks.append(safe_spearman(a, b, True))
    return float(np.mean(overlaps)), float(np.nanmean(ranks))


def complete_metrics(prediction: np.ndarray, truth: np.ndarray, sources: np.ndarray,
                     genes: np.ndarray, k: int) -> dict:
    base = prediction_metrics(prediction, truth, sources, genes)
    base["response_cosine"] = float(np.mean(response_cosines(prediction, truth, sources, genes)))
    base["local_knn_overlap"] , base["local_distance_rank"] = local_metrics(prediction, truth, sources, genes, k)
    pr, tr = rank_metrics(prediction, sources, genes), rank_metrics(truth, sources, genes)
    for key, value in pr.items(): base[f"predicted_{key}"] = value
    for key, value in tr.items(): base[f"truth_{key}"] = value
    base["between_variance_ratio"] = pr["between_variance"] / max(tr["between_variance"], 1e-12)
    return base


def source_rows(prediction: np.ndarray, truth: np.ndarray, sources: np.ndarray,
                genes: np.ndarray) -> list[dict]:
    direct = direct_indices(sources, genes); rows = []
    for i, (a, b) in enumerate(zip(prediction, truth)):
        keep = np.ones(len(genes), bool)
        if direct[i] >= 0: keep[direct[i]] = False
        rows.append({"source": sources[i], "response_pearson": safe_pearson(a[keep], b[keep]),
                     "response_cosine": float(a[keep] @ b[keep] / max(np.linalg.norm(a[keep]) * np.linalg.norm(b[keep]), 1e-12)),
                     "mse": float(np.mean((a[keep] - b[keep]) ** 2))})
    return rows


def representation(name: str, waves: np.ndarray, response: np.ndarray, static: np.ndarray,
                   sources: np.ndarray, genes: np.ndarray, source_gene_rows: np.ndarray,
                   train: np.ndarray, seed: int) -> np.ndarray:
    if name == "Static": return static[source_gene_rows]
    if name == "CorrectLag": return transmission_representations(waves, sources, genes, train)[source_gene_rows]
    if name == "FullDynamic":
        # Frozen audited descriptor; its PCA basis is fitted on training W23 only.
        x = waves[train, 0]; mean = x.mean(0); _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
        components = vt[:min(8, len(vt))].T.astype(np.float32)
        return full_dynamic_descriptor(waves, sources, genes, train, source_gene_rows, mean, components, seed)
    raise ValueError(name)


def choose_direct_model(target: np.ndarray, waves: np.ndarray, response: np.ndarray,
                        static: np.ndarray, sources: np.ndarray, genes: np.ndarray,
                        source_gene_rows: np.ndarray, train: np.ndarray, seed: int,
                        names: list[str], alpha_grid: tuple[float, ...]) -> tuple[str, float, list[dict]]:
    order = np.random.default_rng(seed).permutation(train); folds = np.array_split(order, 2); scores = []
    for name in names:
        for alpha in alpha_grid:
            metrics = []
            for val in folds:
                fit = np.setdiff1d(train, val)
                feat = representation(name, waves, response, static, sources, genes, source_gene_rows, fit, seed + len(metrics) * 193)
                pred, _ = standardized_ridge(feat[fit], target[fit], feat[val], (alpha,))
                metrics.append(complete_metrics(pred, target[val], sources[val], genes, 3))
            scores.append({"representation": name, "alpha": alpha,
                           "geometry": float(np.mean([m["response_distance_rho"] for m in metrics])),
                           "pearson": float(np.mean([m["per_response_strict_trans_pearson"] for m in metrics])),
                           "mse": float(np.mean([m["strict_trans_mse"] for m in metrics]))})
    best = sorted(scores, key=lambda x: (-x["geometry"], -x["pearson"], x["mse"], x["representation"], x["alpha"]))[0]
    return best["representation"], float(best["alpha"]), scores


def repeat_bootstrap(frame, value: str, seed: int, resamples: int) -> dict:
    by_repeat = frame.groupby("repeat")[value].mean().to_numpy(float)
    rng = np.random.default_rng(seed); draws = np.mean(rng.choice(by_repeat, (resamples, len(by_repeat)), replace=True), axis=1)
    return {"point": float(np.mean(by_repeat)), "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975)), "n_repeats": len(by_repeat)}
