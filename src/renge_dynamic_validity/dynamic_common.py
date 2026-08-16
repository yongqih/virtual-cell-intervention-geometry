from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = ROOT / "scripts" / "renge_dynamic_validity"
RESULT_ROOT = ROOT / "results" / "renge_dynamic_validity"
FROZEN_SCRIPT_ROOT = ROOT / "scripts" / "propagation_reproduction"
FROZEN_RESULT_ROOT = ROOT / "results" / "propagation_reproduction"
FROZEN_CACHE = ROOT / "data" / "propagation_reproduction" / "cache" / "renge_processed.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or left[valid].std() < 1e-12 or right[valid].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def safe_spearman(left: np.ndarray, right: np.ndarray, constant_zero: bool = False) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 4 or right[valid].std() < 1e-12:
        return float("nan")
    if left[valid].std() < 1e-12:
        return 0.0 if constant_zero else float("nan")
    return float(spearmanr(left[valid], right[valid]).statistic)


def direct_indices(source_names: np.ndarray, genes: np.ndarray) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(genes.astype(str))}
    return np.asarray([lookup.get(source, -1) for source in source_names.astype(str)], dtype=np.int64)


def strict_trans_cosine_distance(matrix: np.ndarray, direct: np.ndarray) -> np.ndarray:
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


def full_cosine_distance(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, float)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.maximum(norm, 1e-12)
    return 1.0 - np.clip(unit @ unit.T, -1, 1)


def euclidean_distance(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, float)
    sq = np.sum(matrix * matrix, axis=1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * matrix @ matrix.T, 0))


def geometry_rho(prediction: np.ndarray, truth: np.ndarray, direct: np.ndarray) -> float:
    first, second = np.triu_indices(len(prediction), 1)
    return safe_spearman(strict_trans_cosine_distance(prediction, direct)[first, second],
                         strict_trans_cosine_distance(truth, direct)[first, second], True)


def source_metrics(prediction: np.ndarray, truth: np.ndarray, source_names: np.ndarray,
                   genes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direct = direct_indices(source_names, genes); correlations, errors = [], []
    for index, (predicted, observed) in enumerate(zip(prediction, truth)):
        keep = np.ones(len(genes), bool)
        if direct[index] >= 0: keep[direct[index]] = False
        correlations.append(safe_pearson(predicted[keep], observed[keep]))
        errors.append(float(np.mean((predicted[keep] - observed[keep]) ** 2)))
    return np.asarray(correlations), np.asarray(errors)


def prediction_metrics(prediction: np.ndarray, truth: np.ndarray, source_names: np.ndarray,
                       genes: np.ndarray) -> dict[str, float]:
    direct = direct_indices(source_names, genes)
    correlations, errors = source_metrics(prediction, truth, source_names, genes)
    return {"response_distance_rho": geometry_rho(prediction, truth, direct),
            "per_response_strict_trans_pearson": float(np.nanmean(correlations)),
            "strict_trans_mse": float(np.mean(errors)),
            "mean_prediction_norm": float(np.mean(np.linalg.norm(prediction, axis=1))),
            "mean_truth_norm": float(np.mean(np.linalg.norm(truth, axis=1)))}


def grouped_twofold_splits(source_count: int, repeats: int, seed: int) -> list[dict]:
    output = []
    for repeat in range(repeats):
        order = np.random.default_rng(seed + repeat * 1009).permutation(source_count)
        for group, test in enumerate(np.array_split(order, 2)):
            train = np.setdiff1d(np.arange(source_count), test)
            output.append({"repeat": repeat, "group": group, "seed": seed + repeat * 1009,
                           "train": train, "test": np.sort(test)})
    return output


def ridge_fit_predict(train_x: np.ndarray, train_y: np.ndarray, query: np.ndarray,
                      alpha: float) -> np.ndarray:
    x, y, q = np.asarray(train_x, float), np.asarray(train_y, float), np.asarray(query, float)
    x_mean, y_mean = x.mean(0), y.mean(0); x = x - x_mean; y = y - y_mean; q = q - x_mean
    if x.shape[1] <= x.shape[0]:
        coefficient = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
    else:
        coefficient = x.T @ np.linalg.solve(x @ x.T + alpha * np.eye(x.shape[0]), y)
    return (q @ coefficient + y_mean).astype(np.float32)


def select_ridge_alpha(x: np.ndarray, y: np.ndarray, grid: tuple[float, ...]) -> float:
    scores = []
    for alpha in grid:
        error = []
        for index in range(len(x)):
            keep = np.arange(len(x)) != index
            prediction = ridge_fit_predict(x[keep], y[keep], x[index:index + 1], alpha)
            error.append(np.mean((prediction[0] - y[index]) ** 2))
        scores.append(float(np.mean(error)))
    return float(grid[int(np.argmin(scores))])


def standardized_ridge(train_x: np.ndarray, train_y: np.ndarray, query: np.ndarray,
                       grid: tuple[float, ...]) -> tuple[np.ndarray, float]:
    mean = train_x.mean(0); scale = np.maximum(train_x.std(0), 1e-6)
    x = (train_x - mean) / scale; q = (query - mean) / scale
    alpha = select_ridge_alpha(x, train_y, grid)
    return ridge_fit_predict(x, train_y, q, alpha), alpha


def transmission_representations(waves: np.ndarray, sources: np.ndarray, genes: np.ndarray,
                                 train: np.ndarray) -> np.ndarray:
    train_sources = sources[train]
    current = np.concatenate([waves[train, 0], waves[train, 1]], axis=0)
    following = np.concatenate([waves[train, 1], waves[train, 2]], axis=0)
    row_sources = np.concatenate([train_sources, train_sources])
    output = np.zeros((len(genes), len(genes)), dtype=np.float32)
    for gene_index, gene in enumerate(genes):
        selected = np.isin(row_sources, train_sources[train_sources != gene])
        x = current[selected, gene_index].astype(float); y = following[selected].astype(float)
        xc = x - x.mean(); yc = y - y.mean(0, keepdims=True)
        output[gene_index] = (xc @ yc / (float(xc @ xc) + 1e-4)).astype(np.float32)
    return output


def transitions(waves: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([waves[rows, 0], waves[rows, 1]]), np.concatenate([waves[rows, 1], waves[rows, 2]])


def fit_dense_transition(waves: np.ndarray, train: np.ndarray, split_seed: int,
                         alphas: tuple[float, ...]):
    order = np.random.default_rng(split_seed + 41011).permutation(train)
    count = max(2, int(np.ceil(.20 * len(order))))
    inner_val, inner_train = order[:count], order[count:]
    train_x, train_y = transitions(waves, inner_train); val_x, val_y = transitions(waves, inner_val)
    scores = []
    for alpha in alphas:
        model = fit_affine_rrr(train_x, train_y, alpha)
        scores.append(float(np.mean((apply_affine(model, val_x) - val_y) ** 2)))
    alpha = float(alphas[int(np.argmin(scores))])
    return fit_affine_rrr(*transitions(waves, train), alpha), alpha


def fit_affine_rrr(x: np.ndarray, y: np.ndarray, alpha: float):
    x_mean, x_scale = x.mean(0), np.maximum(x.std(0), 1e-6); y_mean = y.mean(0)
    xc = (x - x_mean) / x_scale; yc = y - y_mean
    coefficient = xc.T @ np.linalg.solve(xc @ xc.T + alpha * np.eye(len(xc)), yc)
    return x_mean.astype(np.float32), x_scale.astype(np.float32), y_mean.astype(np.float32), coefficient.astype(np.float32)


def apply_affine(model, query: np.ndarray) -> np.ndarray:
    x_mean, x_scale, y_mean, coefficient = model
    return (((query - x_mean) / x_scale) @ coefficient + y_mean).astype(np.float32)


def weighted_spearman(left_values: np.ndarray, right_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    left = rankdata(left_values).astype(float); right = rankdata(right_values).astype(float)
    total = np.maximum(weights.sum(1), 1.0)
    lm = (weights * left[None]).sum(1) / total; rm = (weights * right[None]).sum(1) / total
    lc = left[None] - lm[:, None]; rc = right[None] - rm[:, None]
    covariance = (weights * lc * rc).sum(1)
    denominator = np.sqrt((weights * lc**2).sum(1) * (weights * rc**2).sum(1))
    return np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12)
