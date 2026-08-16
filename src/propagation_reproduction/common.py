from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "propagation_reproduction"
RESULT_ROOT = ROOT / "results" / "propagation_reproduction"
CACHE_ROOT = DATA_ROOT / "cache"
SEED_BASE = 213069

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or left[valid].std() < 1e-12 or right[valid].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def safe_spearman(left: np.ndarray, right: np.ndarray, constant_prediction_zero: bool = False) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 4 or right[valid].std() < 1e-12:
        return float("nan")
    if left[valid].std() < 1e-12:
        return 0.0 if constant_prediction_zero else float("nan")
    return float(spearmanr(left[valid], right[valid]).statistic)


def direct_indices(source_names: np.ndarray, genes: np.ndarray) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(genes.astype(str))}
    return np.asarray([lookup.get(source, -1) for source in source_names.astype(str)], dtype=np.int64)


def strict_trans_cosine_distance(matrix: np.ndarray, direct: np.ndarray) -> np.ndarray:
    """Pair-specific distance excluding the direct loci of both perturbations."""
    matrix = np.asarray(matrix, np.float64)
    count = len(matrix); first, second = np.triu_indices(count, 1)
    dot = np.sum(matrix[first] * matrix[second], axis=1)
    norm_first = np.sum(matrix[first] ** 2, axis=1)
    norm_second = np.sum(matrix[second] ** 2, axis=1)
    for locus in (direct[first], direct[second]):
        valid = locus >= 0; loc = locus[valid]; a, b = first[valid], second[valid]
        dot[valid] -= matrix[a, loc] * matrix[b, loc]
        norm_first[valid] -= matrix[a, loc] ** 2
        norm_second[valid] -= matrix[b, loc] ** 2
    same = (direct[first] >= 0) & (direct[first] == direct[second])
    if same.any():
        loc = direct[first[same]]; a, b = first[same], second[same]
        dot[same] += matrix[a, loc] * matrix[b, loc]
        norm_first[same] += matrix[a, loc] ** 2
        norm_second[same] += matrix[b, loc] ** 2
    cosine = dot / np.sqrt(np.maximum(norm_first * norm_second, 1e-24))
    distance = np.zeros((count, count), dtype=np.float64)
    distance[first, second] = 1.0 - cosine; distance[second, first] = distance[first, second]
    return distance


def geometry_rho(prediction: np.ndarray, truth: np.ndarray, direct: np.ndarray,
                 constant_prediction_zero: bool = True) -> float:
    predicted = strict_trans_cosine_distance(prediction, direct)
    observed = strict_trans_cosine_distance(truth, direct)
    upper = np.triu_indices(len(prediction), 1)
    return safe_spearman(predicted[upper], observed[upper], constant_prediction_zero)


def rank_metrics(matrix: np.ndarray, direct: np.ndarray | None = None) -> dict[str, float | int]:
    value = np.asarray(matrix, np.float64).copy()
    if direct is not None:
        valid = direct >= 0; value[np.arange(len(value))[valid], direct[valid]] = 0.0
    value -= value.mean(0, keepdims=True)
    singular = np.linalg.svd(value, compute_uv=False, full_matrices=False)
    squared = singular**2; variance_weight = squared / max(squared.sum(), 1e-12)
    singular_weight = singular / max(singular.sum(), 1e-12)
    cumulative = np.cumsum(variance_weight); nz = singular_weight[singular_weight > 0]
    return {
        "pc1_fraction": float(variance_weight[0]), "pc80": int(np.searchsorted(cumulative, .80) + 1),
        "pc90": int(np.searchsorted(cumulative, .90) + 1), "pc95": int(np.searchsorted(cumulative, .95) + 1),
        "participation_ratio": float(1 / max(np.sum(variance_weight**2), 1e-12)),
        "entropy_effective_rank": float(np.exp(-np.sum(nz * np.log(nz)))),
        "between_perturbation_variance": float(np.mean(np.var(value, axis=0))),
    }


def grouped_twofold_splits(source_count: int, repeats: int = 50, seed: int = SEED_BASE) -> list[dict]:
    output = []
    for repeat in range(repeats):
        order = np.random.default_rng(seed + repeat * 1009).permutation(source_count)
        groups = np.array_split(order, 2)
        for group, test in enumerate(groups):
            train = np.setdiff1d(np.arange(source_count), test)
            output.append({"repeat": repeat, "group": group, "seed": seed + repeat * 1009,
                           "train": train, "test": np.sort(test)})
    return output


def ridge_fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
                      alpha: float, fit_intercept: bool = True) -> np.ndarray:
    x, y, q = np.asarray(train_x, float), np.asarray(train_y, float), np.asarray(test_x, float)
    if fit_intercept:
        x_mean, y_mean = x.mean(0), y.mean(0); x = x - x_mean; y = y - y_mean; q = q - x_mean
    else:
        y_mean = np.zeros(y.shape[1])
    if x.shape[1] <= x.shape[0]:
        coefficient = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
    else:
        coefficient = x.T @ np.linalg.solve(x @ x.T + alpha * np.eye(x.shape[0]), y)
    return (q @ coefficient + y_mean).astype(np.float32)


def select_ridge_alpha(x: np.ndarray, y: np.ndarray,
                       grid: tuple[float, ...] = (.01, .1, 1., 10., 100.)) -> float:
    """Training-only leave-one-source-out MSE selection."""
    if len(x) < 4:
        return 1.0
    scores = []
    for alpha in grid:
        error = []
        for index in range(len(x)):
            keep = np.arange(len(x)) != index
            prediction = ridge_fit_predict(x[keep], y[keep], x[index:index + 1], alpha)
            error.append(np.mean((prediction[0] - y[index]) ** 2))
        scores.append(float(np.mean(error)))
    return float(grid[int(np.argmin(scores))])


def prediction_metrics(prediction: np.ndarray, truth: np.ndarray, source_names: np.ndarray,
                       genes: np.ndarray) -> dict[str, float]:
    direct = direct_indices(source_names, genes)
    correlations, errors = [], []
    for index, (predicted, observed) in enumerate(zip(prediction, truth)):
        keep = np.ones(len(genes), bool)
        if direct[index] >= 0: keep[direct[index]] = False
        correlations.append(safe_pearson(predicted[keep], observed[keep]))
        errors.append(float(np.mean((predicted[keep] - observed[keep]) ** 2)))
    predicted_rank, true_rank = rank_metrics(prediction, direct), rank_metrics(truth, direct)
    return {
        "response_distance_rho": geometry_rho(prediction, truth, direct),
        "per_response_strict_trans_pearson": float(np.nanmean(correlations)),
        "strict_trans_mse": float(np.mean(errors)),
        "between_variance_ratio": predicted_rank["between_perturbation_variance"] / max(true_rank["between_perturbation_variance"], 1e-12),
        "predicted_pc1_fraction": predicted_rank["pc1_fraction"], "truth_pc1_fraction": true_rank["pc1_fraction"],
        "predicted_pc80": predicted_rank["pc80"], "truth_pc80": true_rank["pc80"],
        "predicted_pc90": predicted_rank["pc90"], "truth_pc90": true_rank["pc90"],
        "predicted_pc95": predicted_rank["pc95"], "truth_pc95": true_rank["pc95"],
        "predicted_participation_ratio": predicted_rank["participation_ratio"],
        "truth_participation_ratio": true_rank["participation_ratio"],
        "predicted_entropy_effective_rank": predicted_rank["entropy_effective_rank"],
        "truth_entropy_effective_rank": true_rank["entropy_effective_rank"],
    }
