from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("config.json")
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUT = ROOT / CONFIG["output_dir"]
MASTER_SEED = int(CONFIG["master_seed"])


def child_seed(*parts: object) -> int:
    payload = "|".join([str(MASTER_SEED), *(str(p) for p in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 4 or np.std(a[good]) < 1e-12 or np.std(b[good]) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(a[good], b[good]).statistic)


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3 or np.std(a[good]) < 1e-12 or np.std(b[good]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[good], b[good])[0, 1])


def cosine_distances(rows: np.ndarray) -> np.ndarray:
    x = np.asarray(rows, dtype=float)
    if x.shape[0] < 2:
        return np.empty(0, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-12)
    return pdist(x, metric="cosine")


def geometry(pred: np.ndarray, truth: np.ndarray) -> float:
    return safe_spearman(cosine_distances(pred), cosine_distances(truth))


def weighted_fisher_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    good = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not good.any():
        return float("nan")
    z = np.arctanh(np.clip(v[good], -0.999999, 0.999999))
    return float(np.tanh(np.average(z, weights=w[good])))


def make_round_robin_folds(n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return [np.sort(order[i::n_folds]) for i in range(n_folds)]


def training_only_preprocess_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    med = np.zeros(x.shape[1], dtype=float)
    for j in range(x.shape[1]):
        col = x[:, j]
        finite = col[np.isfinite(col)]
        med[j] = float(np.median(finite)) if finite.size else 0.0
    filled = np.where(np.isfinite(x), x, med[None, :])
    mean = filled.mean(axis=0)
    scale = filled.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (filled - mean) / scale, med, np.vstack([mean, scale])


def training_only_preprocess_apply(x: np.ndarray, med: np.ndarray, stats_: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    filled = np.where(np.isfinite(x), x, med[None, :])
    return (filled - stats_[0][None, :]) / stats_[1][None, :]


def fit_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    xs, med, stats_ = training_only_preprocess_fit(x_train)
    xq = training_only_preprocess_apply(x_query, med, stats_)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(xs, y_train)
    return np.asarray(model.predict(xq), dtype=float), {"median": med, "mean_scale": stats_}


def orient_axes(axes: np.ndarray) -> np.ndarray:
    axes = np.asarray(axes, dtype=float).copy()
    for k in range(axes.shape[0]):
        j = int(np.argmax(np.abs(axes[k])))
        if axes[k, j] < 0:
            axes[k] *= -1
    return axes


def residual_svd_axes(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, singular, vt = np.linalg.svd(np.asarray(residuals, dtype=float), full_matrices=False)
    return orient_axes(vt), singular


def random_axes_from_residual_span(
    residual_axes: np.ndarray,
    q: int,
    rng: np.random.Generator,
) -> np.ndarray:
    rank = residual_axes.shape[0]
    if q > rank:
        raise ValueError(f"q={q} exceeds residual rank={rank}")
    raw = rng.normal(size=(rank, q))
    qmat, _ = np.linalg.qr(raw, mode="reduced")
    axes = qmat.T @ residual_axes
    return orient_axes(axes)


def exact_reconstruction(base: np.ndarray, truth: np.ndarray, axes: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    if q == 0:
        return np.asarray(base, dtype=float).copy(), np.zeros((len(base), 0), dtype=float)
    use = axes[:q]
    coef = (truth - base) @ use.T
    return base + coef @ use, coef


def sign_reconstruction(
    base: np.ndarray,
    oracle_coef: np.ndarray,
    axes: np.ndarray,
    typical_magnitude: np.ndarray,
    q: int,
) -> np.ndarray:
    use = axes[:q]
    signs = np.sign(oracle_coef[:, :q])
    signs[signs == 0] = 1.0
    return base + (signs * typical_magnitude[:q][None, :]) @ use


def fixed_radius_reconstruction(
    base: np.ndarray,
    oracle_coef: np.ndarray,
    axes: np.ndarray,
    typical_radius: float,
) -> np.ndarray:
    direction = oracle_coef[:, :2].copy()
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / np.maximum(norm, 1e-12)
    return base + (direction * float(typical_radius)) @ axes[:2]


def source_bootstrap_geometry(
    pred_by_q: dict[int, np.ndarray],
    truth: np.ndarray,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n = truth.shape[0]
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(idx)) < 3:
            continue
        for q, pred in pred_by_q.items():
            rows.append({"bootstrap": b, "q": q, "rho": geometry(pred[idx], truth[idx])})
    return pd.DataFrame(rows)


def ensure_base_manifests() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig4_orientation_code").mkdir(exist_ok=True)
    (OUT / "fig5_temporal_identifiability").mkdir(exist_ok=True)
    config_target = OUT / "config.json"
    if not config_target.exists():
        config_target.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception as exc:
        commit = f"UNAVAILABLE: {exc}"
    (OUT / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    env = [
        f"python={sys.version.replace(os.linesep, ' ')}",
        f"platform={platform.platform()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
    ]
    try:
        import scipy
        import sklearn

        env.extend([f"scipy={scipy.__version__}", f"scikit-learn={sklearn.__version__}"])
    except Exception as exc:
        env.append(f"package_version_error={exc}")
    (OUT / "environment.txt").write_text("\n".join(env) + "\n", encoding="utf-8")
    manifest = {
        "protocol_version": CONFIG["protocol_version"],
        "master_seed": MASTER_SEED,
        "status": "IN_PROGRESS",
        "script": str(Path(__file__).relative_to(ROOT)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "historical_outputs_overwritten": False,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def append_input_manifest(rows: Iterable[dict[str, object]]) -> None:
    path = OUT / "input_manifest.csv"
    new = pd.DataFrame(list(rows))
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
        new = new.drop_duplicates(subset=["path"], keep="last")
    new.sort_values("path").to_csv(path, index=False)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ci(values: Sequence[float], alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan"), float("nan")
    return float(np.quantile(x, alpha / 2)), float(np.quantile(x, 1 - alpha / 2))
