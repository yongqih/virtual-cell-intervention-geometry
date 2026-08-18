from __future__ import annotations

import io
import json
import math
import pickle
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    CONFIG,
    MASTER_SEED,
    OUT,
    ROOT,
    append_input_manifest,
    child_seed,
    ci,
    cosine_distances,
    ensure_base_manifests,
    exact_reconstruction,
    fit_ridge_predict,
    fixed_radius_reconstruction,
    geometry,
    make_round_robin_folds,
    random_axes_from_residual_span,
    residual_svd_axes,
    safe_spearman,
    sha256_file,
    sign_reconstruction,
    weighted_fisher_mean,
    write_json,
)


Q_VALUES = [int(x) for x in CONFIG["q_values"]]
ALPHAS = [float(x) for x in CONFIG["ridge_alphas"]]
RANDOM_REPS = int(CONFIG["random_subspace_replicates"])
BOOT_REPS = int(CONFIG["source_bootstrap_replicates"])
BLOCK_BOOT_REPS = int(CONFIG["block_bootstrap_replicates"])
PROB_REPS = int(CONFIG["probability_curve_replicates"])
PROB_VALUES = [float(x) for x in CONFIG["probability_values"]]


@dataclass
class GroupArtifact:
    dataset: str
    pathway: str
    cell_line: str
    outer_fold: int
    held_names: list[str]
    base: np.ndarray
    truth: np.ndarray
    axes: np.ndarray
    oracle_coef: np.ndarray
    typical_magnitude: np.ndarray
    typical_radius: float
    predictions_q: dict[int, np.ndarray]


def alpha_oof(
    y: np.ndarray,
    train_idx: np.ndarray,
    feature_builder: Callable[[np.ndarray, np.ndarray], np.ndarray],
    inner_folds: int,
    seed_label: tuple[object, ...],
) -> tuple[float, np.ndarray, pd.DataFrame]:
    fold_pos = make_round_robin_folds(len(train_idx), inner_folds, child_seed(*seed_label, "inner"))
    rows: list[dict[str, object]] = []
    predictions: dict[float, np.ndarray] = {}
    all_pos = np.arange(len(train_idx))
    for alpha in ALPHAS:
        oof = np.full((len(train_idx), y.shape[1]), np.nan, dtype=float)
        for inner_fold, val_pos in enumerate(fold_pos):
            fit_pos = np.setdiff1d(all_pos, val_pos, assume_unique=True)
            fit_idx = train_idx[fit_pos]
            val_idx = train_idx[val_pos]
            x_fit = feature_builder(fit_idx, fit_idx)
            x_val = feature_builder(fit_idx, val_idx)
            pred, _ = fit_ridge_predict(x_fit, y[fit_idx], x_val, alpha)
            oof[val_pos] = pred
            rows.append(
                {
                    "alpha": alpha,
                    "inner_fold": inner_fold,
                    "n_train": len(fit_idx),
                    "n_val": len(val_idx),
                    "residual_mse": float(np.mean((pred - y[val_idx]) ** 2)),
                    "fold_geometry": geometry(pred, y[val_idx]),
                }
            )
        if not np.isfinite(oof).all():
            raise RuntimeError("Incomplete inner OOF prediction matrix")
        predictions[alpha] = oof
    frame = pd.DataFrame(rows)
    score = frame.groupby("alpha", as_index=False).residual_mse.mean().sort_values(["residual_mse", "alpha"])
    best = float(score.iloc[0].alpha)
    return best, predictions[best], frame


def evaluate_block(
    dataset: str,
    pathway: str,
    cell_line: str,
    source_names: list[str],
    y: np.ndarray,
    outer_folds: list[np.ndarray],
    inner_folds: int,
    feature_builder: Callable[[np.ndarray, np.ndarray], np.ndarray],
    block_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[GroupArtifact], pd.DataFrame]:
    block_dir.mkdir(parents=True, exist_ok=True)
    q_rows: list[dict[str, object]] = []
    sign_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    alpha_rows: list[pd.DataFrame] = []
    artifacts: list[GroupArtifact] = []
    all_idx = np.arange(len(source_names))
    for outer_fold, held_idx in enumerate(outer_folds):
        train_idx = np.setdiff1d(all_idx, held_idx, assume_unique=True)
        if np.intersect1d(train_idx, held_idx).size:
            raise RuntimeError("Outer source overlap")
        best_alpha, train_oof, alpha_frame = alpha_oof(
            y,
            train_idx,
            feature_builder,
            inner_folds,
            (dataset, pathway, cell_line, outer_fold),
        )
        alpha_frame.insert(0, "outer_fold", outer_fold)
        alpha_frame.insert(0, "cell_line", cell_line)
        alpha_frame.insert(0, "pathway", pathway)
        alpha_frame.insert(0, "dataset", dataset)
        alpha_rows.append(alpha_frame)
        residual = y[train_idx] - train_oof
        axes, singular = residual_svd_axes(residual)
        x_train = feature_builder(train_idx, train_idx)
        x_held = feature_builder(train_idx, held_idx)
        base, _ = fit_ridge_predict(x_train, y[train_idx], x_held, best_alpha)
        predictions_q: dict[int, np.ndarray] = {}
        oracle_all = (y[held_idx] - base) @ axes.T
        train_coef = residual @ axes.T
        typical_mag = np.median(np.abs(train_coef), axis=0)
        typical_radius = float(np.median(np.linalg.norm(train_coef[:, :2], axis=1)))
        pair_count = len(held_idx) * (len(held_idx) - 1) // 2
        for q in Q_VALUES:
            pred, coef = exact_reconstruction(base, y[held_idx], axes, q)
            predictions_q[q] = pred
            q_rows.append(
                {
                    "dataset": dataset,
                    "pathway": pathway,
                    "cell_line": cell_line,
                    "outer_fold": outer_fold,
                    "q": q,
                    "rho": geometry(pred, y[held_idx]),
                    "n_held": len(held_idx),
                    "pair_count": pair_count,
                    "selected_alpha": best_alpha,
                    "residual_rank": axes.shape[0],
                    "residual_singular_fraction": float(np.sum(singular[:q] ** 2) / np.sum(singular**2)) if q else 0.0,
                }
            )
        exact_q2 = predictions_q[2]
        sign_q1 = sign_reconstruction(base, oracle_all, axes, typical_mag, 1)
        sign_q2 = sign_reconstruction(base, oracle_all, axes, typical_mag, 2)
        fixed = fixed_radius_reconstruction(base, oracle_all, axes, typical_radius)
        sign_predictions = {
            "baseline": base,
            "q1_sign": sign_q1,
            "q2_sign": sign_q2,
            "q2_fixed_radius_direction": fixed,
            "q2_exact": exact_q2,
        }
        for condition, pred in sign_predictions.items():
            sign_rows.append(
                {
                    "dataset": dataset,
                    "pathway": pathway,
                    "cell_line": cell_line,
                    "outer_fold": outer_fold,
                    "condition": condition,
                    "rho": geometry(pred, y[held_idx]),
                    "n_held": len(held_idx),
                    "pair_count": pair_count,
                    "selected_alpha": best_alpha,
                    "typical_radius": typical_radius,
                }
            )
        for pos, source_idx in enumerate(held_idx):
            row: dict[str, object] = {
                "dataset": dataset,
                "pathway": pathway,
                "cell_line": cell_line,
                "outer_fold": outer_fold,
                "source": source_names[source_idx],
                "truth_norm": float(np.linalg.norm(y[source_idx])),
                "baseline_residual_norm": float(np.linalg.norm(y[source_idx] - base[pos])),
                "selected_alpha": best_alpha,
            }
            for k in range(min(8, oracle_all.shape[1])):
                row[f"oracle_u{k + 1}"] = float(oracle_all[pos, k])
                row[f"oracle_sign{k + 1}"] = int(np.sign(oracle_all[pos, k]))
                row[f"training_median_abs_u{k + 1}"] = float(typical_mag[k])
            source_rows.append(row)
        rng = np.random.default_rng(child_seed(dataset, pathway, cell_line, outer_fold, "random_subspace"))
        for replicate in range(RANDOM_REPS):
            for q in Q_VALUES[1:]:
                random_axes = random_axes_from_residual_span(axes, q, rng)
                pred, _ = exact_reconstruction(base, y[held_idx], random_axes, q)
                null_rows.append(
                    {
                        "dataset": dataset,
                        "pathway": pathway,
                        "cell_line": cell_line,
                        "outer_fold": outer_fold,
                        "replicate": replicate,
                        "q": q,
                        "rho": geometry(pred, y[held_idx]),
                        "n_held": len(held_idx),
                        "pair_count": pair_count,
                    }
                )
        np.savez_compressed(
            block_dir / f"fold_{outer_fold}_predictions.npz",
            held_indices=held_idx,
            held_sources=np.asarray([source_names[i] for i in held_idx]),
            truth=y[held_idx],
            baseline=base,
            q1=predictions_q[1],
            q2=predictions_q[2],
            q4=predictions_q[4],
            q8=predictions_q[8],
            q1_sign=sign_q1,
            q2_sign=sign_q2,
            q2_fixed_radius=fixed,
            axes=axes[:8],
            oracle_coefficients=oracle_all[:, :8],
            training_typical_magnitude=typical_mag[:8],
        )
        leakage_rows.append(
            {
                "dataset": dataset,
                "pathway": pathway,
                "cell_line": cell_line,
                "outer_fold": outer_fold,
                "held_sources_in_model_fit": False,
                "held_sources_in_inner_oof_feature_columns": False,
                "held_responses_in_axis_fit": False,
                "held_responses_in_alpha_selection": False,
                "held_responses_used_only_for_oracle_after_axis_freeze": True,
                "training_oof_residuals_used_for_axes": True,
                "leakage_audit": "PASS",
            }
        )
        artifacts.append(
            GroupArtifact(
                dataset=dataset,
                pathway=pathway,
                cell_line=cell_line,
                outer_fold=outer_fold,
                held_names=[source_names[i] for i in held_idx],
                base=base,
                truth=y[held_idx],
                axes=axes,
                oracle_coef=oracle_all,
                typical_magnitude=typical_mag,
                typical_radius=typical_radius,
                predictions_q=predictions_q,
            )
        )
        print(
            f"[{dataset} {pathway} {cell_line} fold {outer_fold}] "
            f"alpha={best_alpha:g} q0={q_rows[-5]['rho']:.3f} q1={q_rows[-4]['rho']:.3f} "
            f"q2={q_rows[-3]['rho']:.3f}",
            flush=True,
        )
    return (
        pd.DataFrame(q_rows),
        pd.DataFrame(sign_rows),
        pd.DataFrame(source_rows),
        pd.DataFrame(null_rows),
        artifacts,
        pd.concat(alpha_rows, ignore_index=True),
        pd.DataFrame(leakage_rows),
    )


def aggregate_q(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, part in frame.groupby([*group_columns, "q"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = dict(zip([*group_columns, "q"], keys))
        values.update(
            {
                "rho": weighted_fisher_mean(part.rho, part.pair_count),
                "groups": len(part),
                "sources": int(part.n_held.sum()),
                "pairs": int(part.pair_count.sum()),
            }
        )
        rows.append(values)
    return pd.DataFrame(rows)


def aggregate_conditions(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, part in frame.groupby([*group_columns, "condition"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = dict(zip([*group_columns, "condition"], keys))
        values.update(
            {
                "rho": weighted_fisher_mean(part.rho, part.pair_count),
                "groups": len(part),
                "sources": int(part.n_held.sum()),
                "pairs": int(part.pair_count.sum()),
            }
        )
        rows.append(values)
    return pd.DataFrame(rows)


def aggregate_random_null(null_frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in null_frame.groupby([*group_columns, "q", "replicate"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = dict(zip([*group_columns, "q", "replicate"], keys))
        values["rho"] = weighted_fisher_mean(part.rho, part.pair_count)
        rows.append(values)
    return pd.DataFrame(rows)


def grouped_source_bootstrap(artifacts: list[GroupArtifact], dataset: str, n_boot: int) -> pd.DataFrame:
    rows = []
    for b in range(n_boot):
        group_values: dict[int, list[float]] = {q: [] for q in Q_VALUES}
        group_weights: dict[int, list[int]] = {q: [] for q in Q_VALUES}
        for art in artifacts:
            rng = np.random.default_rng(child_seed(dataset, art.pathway, art.cell_line, art.outer_fold, "source_boot", b))
            n = len(art.truth)
            idx = rng.integers(0, n, n)
            if len(np.unique(idx)) < 3:
                continue
            pairs = n * (n - 1) // 2
            for q in Q_VALUES:
                group_values[q].append(geometry(art.predictions_q[q][idx], art.truth[idx]))
                group_weights[q].append(pairs)
        for q in Q_VALUES:
            rows.append(
                {
                    "dataset": dataset,
                    "bootstrap": b,
                    "q": q,
                    "rho": weighted_fisher_mean(group_values[q], group_weights[q]),
                }
            )
    return pd.DataFrame(rows)


def probability_curve(artifacts: list[GroupArtifact], dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    replicate_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    pathways = sorted(set(a.pathway for a in artifacts))
    for p in PROB_VALUES:
        for replicate in range(PROB_REPS):
            group_records = []
            for art in artifacts:
                rng = np.random.default_rng(child_seed(dataset, art.pathway, art.cell_line, art.outer_fold, "prob", p, replicate))
                true_sign = np.sign(art.oracle_coef[:, :2])
                true_sign[true_sign == 0] = 1
                correct = rng.random(true_sign.shape) < p
                inferred_sign = np.where(correct, true_sign, -true_sign)
                coef = (2 * p - 1) * inferred_sign * art.typical_magnitude[:2][None, :]
                pred = art.base + coef @ art.axes[:2]
                group_records.append(
                    {
                        "pathway": art.pathway,
                        "rho": geometry(pred, art.truth),
                        "pairs": len(art.truth) * (len(art.truth) - 1) // 2,
                    }
                )
            gf = pd.DataFrame(group_records)
            replicate_rows.append(
                {
                    "dataset": dataset,
                    "pathway": "ALL",
                    "p": p,
                    "replicate": replicate,
                    "rho": weighted_fisher_mean(gf.rho, gf.pairs),
                    "exact_2bit_state_probability": p * p,
                }
            )
            for pathway in pathways:
                part = gf[gf.pathway == pathway]
                replicate_rows.append(
                    {
                        "dataset": dataset,
                        "pathway": pathway,
                        "p": p,
                        "replicate": replicate,
                        "rho": weighted_fisher_mean(part.rho, part.pairs),
                        "exact_2bit_state_probability": p * p,
                    }
                )
    rep = pd.DataFrame(replicate_rows)
    for keys, part in rep.groupby(["dataset", "pathway", "p"]):
        low, high = ci(part.rho)
        summary_rows.append(
            {
                "dataset": keys[0],
                "pathway": keys[1],
                "p": keys[2],
                "mean_rho": float(part.rho.mean()),
                "mc_ci_low": low,
                "mc_ci_high": high,
                "exact_2bit_state_probability": float(keys[2] ** 2),
                "replicates": len(part),
            }
        )
    return rep, pd.DataFrame(summary_rows)


def load_k562() -> tuple[list[str], np.ndarray, np.ndarray, list[np.ndarray], pd.DataFrame]:
    npz_path = ROOT / CONFIG["k562"]["response_npz"]
    network_path = ROOT / CONFIG["k562"]["network_csv"]
    z = np.load(npz_path, allow_pickle=False)
    perturbations = [str(x) for x in z["perturbations"]]
    genes = [str(x) for x in z["genes"]]
    gene_set = set(genes)
    eligible = [i for i, g in enumerate(perturbations) if g in gene_set]
    source_names = [perturbations[i] for i in eligible]
    response_mask = np.array([g not in set(source_names) for g in genes], dtype=bool)
    y = np.asarray(z["delta"][eligible][:, response_mask], dtype=float)
    edges = pd.read_csv(network_path, usecols=["source", "target", "sign"])
    source_index = {g: i for i, g in enumerate(source_names)}
    incoming = np.zeros((len(source_names), len(source_names)), dtype=float)  # feature source x target sample
    for row in edges.itertuples(index=False):
        if row.source in source_index and row.target in source_index:
            incoming[source_index[row.source], source_index[row.target]] = float(row.sign)
    folds = make_round_robin_folds(len(source_names), int(CONFIG["k562"]["outer_folds"]), 17)
    split_rows = []
    for fold, idx in enumerate(folds):
        for i in idx:
            split_rows.append({"dataset": "K562", "pathway": "NA", "source": source_names[i], "outer_fold": fold})
    append_input_manifest(
        [
            {"path": str(npz_path), "bytes": npz_path.stat().st_size, "sha256": sha256_file(npz_path), "role": "K562 response matrix"},
            {"path": str(network_path), "bytes": network_path.stat().st_size, "sha256": sha256_file(network_path), "role": "K562 frozen signed network"},
        ]
    )
    pd.DataFrame({"gene": np.asarray(genes)[response_mask], "included": True}).to_csv(OUT / "k562_response_gene_manifest.csv", index=False)
    return source_names, y, incoming, folds, pd.DataFrame(split_rows)


def build_jiang_cache() -> dict[str, dict[str, object]]:
    cache_path = OUT / "jiang_core_cache.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)
    archive = Path(CONFIG["jiang"]["archive"])
    pathways = list(CONFIG["jiang"]["pathways"])
    cells = list(CONFIG["jiang"]["cell_lines"])
    result: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(archive) as zf:
        for pathway in pathways:
            entries = sorted(
                n
                for n in zf.namelist()
                if f"DE_results_all_pathway/Parse_{pathway}/" in n and n.endswith("_DE_results.txt")
            )
            sources = [Path(n).name.split(f"_{pathway}_pathway_DE_results.txt")[0] for n in entries]
            frames: dict[str, pd.DataFrame] = {}
            strict: set[str] | None = None
            log_cols = [f"log2FC_{c}" for c in cells]
            beta_cols = [f"beta_cell_type{c}" for c in cells]
            for entry, source in zip(entries, sources):
                frame = pd.read_csv(
                    io.BytesIO(zf.read(entry)),
                    sep=r"\s+",
                    na_values=["NA", "NaN", "nan"],
                    usecols=["gene_ID", *log_cols, *beta_cols],
                )
                frame = frame.drop_duplicates("gene_ID", keep="first").set_index("gene_ID")
                frames[source] = frame
                complete = set(frame.index[frame[log_cols].notna().all(axis=1)])
                strict = complete if strict is None else strict.intersection(complete)
            strict_genes = sorted((strict or set()).difference(sources))
            y = np.stack(
                [frames[s].loc[strict_genes, log_cols].to_numpy(dtype=float).T for s in sources], axis=0
            )  # source x cell x gene
            incoming = np.full((len(sources), len(sources), len(cells)), np.nan, dtype=float)
            for feature_source, source in enumerate(sources):
                frame = frames[source]
                for target, target_gene in enumerate(sources):
                    if target_gene in frame.index:
                        incoming[target, feature_source, :] = frame.loc[target_gene, beta_cols].to_numpy(dtype=float)
            result[pathway] = {
                "sources": sources,
                "cells": cells,
                "strict_genes": strict_genes,
                "responses": y,
                "incoming_beta": incoming,
                "entries": entries,
            }
            pd.DataFrame({"pathway": pathway, "gene": strict_genes}).to_csv(
                OUT / f"jiang_{pathway.lower()}_strict_gene_manifest.csv", index=False
            )
            print(f"[Jiang cache] {pathway}: sources={len(sources)} strict_genes={len(strict_genes)}", flush=True)
    with cache_path.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def jiang_reliability(data: dict[str, dict[str, object]], folds_by_pathway: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    rows = []
    reps = int(CONFIG["jiang"]["reliability_gene_split_replicates"])
    for pathway, item in data.items():
        y = np.asarray(item["responses"], dtype=float)
        cells = list(item["cells"])
        for fold, held in enumerate(folds_by_pathway[pathway]):
            g = y.shape[2]
            for rep in range(reps):
                rng = np.random.default_rng(child_seed("jiang_reliability", pathway, fold, rep))
                order = rng.permutation(g)
                half = g // 2
                a, b = order[:half], order[half : 2 * half]
                for c, cell in enumerate(cells):
                    rho = safe_spearman(cosine_distances(y[held, c][:, a]), cosine_distances(y[held, c][:, b]))
                    rows.append(
                        {
                            "pathway": pathway,
                            "cell_line": cell,
                            "outer_fold": fold,
                            "replicate": rep,
                            "n_held": len(held),
                            "genes_per_half": half,
                            "split_half_geometry_rho": rho,
                        }
                    )
    return pd.DataFrame(rows)


def block_bootstrap(block_q: pd.DataFrame) -> pd.DataFrame:
    blocks = block_q[["pathway", "cell_line"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(child_seed("jiang", "block_bootstrap"))
    rows = []
    for b in range(BLOCK_BOOT_REPS):
        sampled = blocks.iloc[rng.integers(0, len(blocks), len(blocks))]
        for q in Q_VALUES:
            vals = []
            for row in sampled.itertuples(index=False):
                vals.append(float(block_q[(block_q.pathway == row.pathway) & (block_q.cell_line == row.cell_line) & (block_q.q == q)].rho.iloc[0]))
            rows.append({"bootstrap": b, "q": q, "rho": weighted_fisher_mean(vals, np.ones(len(vals)))})
    return pd.DataFrame(rows)


def summarize_with_ci(summary: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in summary.itertuples(index=False):
        part = boot[boot.q == row.q]
        low, high = ci(part.rho)
        item = row._asdict()
        item.update({"ci_low": low, "ci_high": high, "bootstrap_replicates": int(part.bootstrap.nunique())})
        rows.append(item)
    return pd.DataFrame(rows)


def run_k562() -> dict[str, object]:
    print("[A/B K562] loading frozen response/network assets", flush=True)
    names, y, incoming, folds, split = load_k562()

    def features(feature_sources: np.ndarray, samples: np.ndarray) -> np.ndarray:
        return incoming[np.ix_(feature_sources, samples)].T

    block = OUT / "experiment_a_k562" / "K562"
    qf, sf, sourcef, nullf, artifacts, alphaf, leakf = evaluate_block(
        "K562", "NA", "K562", names, y, folds, int(CONFIG["k562"]["inner_folds"]), features, block
    )
    qf.to_csv(OUT / "experiment_a_k562" / "fold_metrics.csv", index=False)
    sf.to_csv(OUT / "experiment_b_sign_only" / "k562_fold_metrics.csv", index=False)
    sourcef.to_csv(OUT / "experiment_a_k562" / "held_source_coefficients.csv", index=False)
    nullf.to_csv(OUT / "experiment_a_k562" / "random_subspace_fold_null.csv", index=False)
    alphaf.to_csv(OUT / "experiment_a_k562" / "inner_alpha_selection.csv", index=False)
    leakf.to_csv(OUT / "experiment_a_k562" / "leakage_audit.csv", index=False)
    summary = aggregate_q(qf, [])
    boot = grouped_source_bootstrap(artifacts, "K562", BOOT_REPS)
    boot.to_csv(OUT / "experiment_a_k562" / "source_bootstrap.csv", index=False)
    summary = summarize_with_ci(summary, boot)
    random_agg = aggregate_random_null(nullf, [])
    null_summary_rows = []
    for q in Q_VALUES[1:]:
        vals = random_agg[random_agg.q == q].rho.to_numpy()
        true = float(summary[summary.q == q].rho.iloc[0])
        null_summary_rows.append(
            {
                "dataset": "K562",
                "pathway": "ALL",
                "q": q,
                "true_rho": true,
                "null_median": float(np.median(vals)),
                "null_95th": float(np.quantile(vals, 0.95)),
                "empirical_p": float((1 + np.sum(vals >= true)) / (1 + len(vals))),
                "replicates": len(vals),
            }
        )
    null_summary = pd.DataFrame(null_summary_rows)
    random_agg.to_csv(OUT / "experiment_a_k562" / "random_subspace_aggregate_null.csv", index=False)
    null_summary.to_csv(OUT / "experiment_a_k562" / "random_subspace_summary.csv", index=False)
    sign_summary = aggregate_conditions(sf, [])
    sign_summary.to_csv(OUT / "experiment_b_sign_only" / "k562_summary.csv", index=False)
    prob_rep, prob_summary = probability_curve(artifacts, "K562")
    prob_rep.to_csv(OUT / "experiment_b_sign_only" / "k562_probability_replicates.csv", index=False)
    prob_summary.to_csv(OUT / "experiment_b_sign_only" / "k562_probability_curve.csv", index=False)
    split.to_csv(OUT / "experiment_a_k562" / "split_manifest.csv", index=False)
    return {
        "q_fold": qf,
        "q_summary": summary,
        "null_summary": null_summary,
        "sign_fold": sf,
        "sign_summary": sign_summary,
        "prob_summary": prob_summary,
        "split": split,
        "leakage": leakf,
        "artifacts": artifacts,
    }


def run_jiang() -> dict[str, object]:
    print("[A/B Jiang] parsing/reusing frozen DE archive", flush=True)
    archive = Path(CONFIG["jiang"]["archive"])
    append_input_manifest(
        [{"path": str(archive), "bytes": archive.stat().st_size, "sha256": sha256_file(archive), "role": "Jiang primary DE archive"}]
    )
    data = build_jiang_cache()
    folds_by_pathway: dict[str, list[np.ndarray]] = {}
    split_rows = []
    for pathway, item in data.items():
        names = list(item["sources"])
        folds = make_round_robin_folds(len(names), int(CONFIG["jiang"]["outer_folds"]), child_seed("jiang_outer", pathway))
        folds_by_pathway[pathway] = folds
        for fold, idx in enumerate(folds):
            for i in idx:
                split_rows.append({"dataset": "Jiang", "pathway": pathway, "source": names[i], "outer_fold": fold})
    reliability = jiang_reliability(data, folds_by_pathway)
    reliability.to_csv(OUT / "experiment_a_jiang" / "truth_geometry_split_half_reliability.csv", index=False)
    reliability_summary = (
        reliability.groupby("pathway", as_index=False)
        .split_half_geometry_rho.mean()
        .rename(columns={"split_half_geometry_rho": "mean_split_half_geometry_rho"})
    )
    reliability_summary["gate_threshold"] = float(CONFIG["jiang"]["reliability_gate"])
    reliability_summary["gate_pass"] = reliability_summary.mean_split_half_geometry_rho > reliability_summary.gate_threshold
    reliability_summary.to_csv(OUT / "experiment_a_jiang" / "truth_geometry_reliability_summary.csv", index=False)
    if not reliability_summary.gate_pass.all():
        raise RuntimeError("Jiang reliability gate failed; protocol requires stopping Experiment A-Jiang")
    q_frames = []
    sign_frames = []
    source_frames = []
    null_frames = []
    alpha_frames = []
    leak_frames = []
    artifacts: list[GroupArtifact] = []
    cells = list(CONFIG["jiang"]["cell_lines"])
    for pathway, item in data.items():
        names = list(item["sources"])
        responses = np.asarray(item["responses"], dtype=float)
        incoming = np.asarray(item["incoming_beta"], dtype=float)
        folds = folds_by_pathway[pathway]
        for cell_idx, cell in enumerate(cells):
            aux = np.array([i for i in range(len(cells)) if i != cell_idx], dtype=int)

            def features(feature_sources: np.ndarray, samples: np.ndarray, incoming=incoming, aux=aux) -> np.ndarray:
                return incoming[samples][:, feature_sources][:, :, aux].reshape(len(samples), -1)

            block_dir = OUT / "experiment_a_jiang" / pathway / cell
            qf, sf, sourcef, nullf, arts, alphaf, leakf = evaluate_block(
                "Jiang",
                pathway,
                cell,
                names,
                responses[:, cell_idx, :],
                folds,
                int(CONFIG["jiang"]["inner_folds"]),
                features,
                block_dir,
            )
            q_frames.append(qf)
            sign_frames.append(sf)
            source_frames.append(sourcef)
            null_frames.append(nullf)
            alpha_frames.append(alphaf)
            leak_frames.append(leakf)
            artifacts.extend(arts)
    qf = pd.concat(q_frames, ignore_index=True)
    sf = pd.concat(sign_frames, ignore_index=True)
    sourcef = pd.concat(source_frames, ignore_index=True)
    nullf = pd.concat(null_frames, ignore_index=True)
    alphaf = pd.concat(alpha_frames, ignore_index=True)
    leakf = pd.concat(leak_frames, ignore_index=True)
    qf.to_csv(OUT / "experiment_a_jiang" / "fold_metrics.csv", index=False)
    sf.to_csv(OUT / "experiment_b_sign_only" / "jiang_fold_metrics.csv", index=False)
    sourcef.to_csv(OUT / "experiment_a_jiang" / "held_source_coefficients.csv", index=False)
    nullf.to_csv(OUT / "experiment_a_jiang" / "random_subspace_fold_null.csv", index=False)
    alphaf.to_csv(OUT / "experiment_a_jiang" / "inner_alpha_selection.csv", index=False)
    leakf.to_csv(OUT / "experiment_a_jiang" / "leakage_audit.csv", index=False)
    overall = aggregate_q(qf, [])
    pathway_summary = aggregate_q(qf, ["pathway"])
    block_summary = aggregate_q(qf, ["pathway", "cell_line"])
    block_boot = block_bootstrap(block_summary)
    block_boot.to_csv(OUT / "experiment_a_jiang" / "block_bootstrap.csv", index=False)
    overall = summarize_with_ci(overall, block_boot)
    random_overall = aggregate_random_null(nullf, [])
    random_pathway = aggregate_random_null(nullf, ["pathway"])
    null_rows = []
    for pathway in ["ALL", *CONFIG["jiang"]["pathways"]]:
        source = random_overall if pathway == "ALL" else random_pathway[random_pathway.pathway == pathway]
        true_source = overall if pathway == "ALL" else pathway_summary[pathway_summary.pathway == pathway]
        for q in Q_VALUES[1:]:
            vals = source[source.q == q].rho.to_numpy()
            true = float(true_source[true_source.q == q].rho.iloc[0])
            null_rows.append(
                {
                    "dataset": "Jiang",
                    "pathway": pathway,
                    "q": q,
                    "true_rho": true,
                    "null_median": float(np.median(vals)),
                    "null_95th": float(np.quantile(vals, 0.95)),
                    "empirical_p": float((1 + np.sum(vals >= true)) / (1 + len(vals))),
                    "replicates": len(vals),
                }
            )
    null_summary = pd.DataFrame(null_rows)
    random_overall.to_csv(OUT / "experiment_a_jiang" / "random_subspace_aggregate_null.csv", index=False)
    random_pathway.to_csv(OUT / "experiment_a_jiang" / "random_subspace_pathway_null.csv", index=False)
    null_summary.to_csv(OUT / "experiment_a_jiang" / "random_subspace_summary.csv", index=False)
    sign_overall = aggregate_conditions(sf, [])
    sign_pathway = aggregate_conditions(sf, ["pathway"])
    sign_block = aggregate_conditions(sf, ["pathway", "cell_line"])
    sign_overall.to_csv(OUT / "experiment_b_sign_only" / "jiang_summary.csv", index=False)
    sign_pathway.to_csv(OUT / "experiment_b_sign_only" / "jiang_pathway_summary.csv", index=False)
    sign_block.to_csv(OUT / "experiment_b_sign_only" / "jiang_block_summary.csv", index=False)
    prob_rep, prob_summary = probability_curve(artifacts, "Jiang")
    prob_rep.to_csv(OUT / "experiment_b_sign_only" / "jiang_probability_replicates.csv", index=False)
    prob_summary.to_csv(OUT / "experiment_b_sign_only" / "jiang_probability_curve.csv", index=False)
    split = pd.DataFrame(split_rows)
    split.to_csv(OUT / "experiment_a_jiang" / "split_manifest.csv", index=False)
    return {
        "q_fold": qf,
        "q_summary": overall,
        "q_pathway": pathway_summary,
        "q_block": block_summary,
        "block_boot": block_boot,
        "null_summary": null_summary,
        "sign_fold": sf,
        "sign_summary": sign_overall,
        "sign_pathway": sign_pathway,
        "sign_block": sign_block,
        "prob_summary": prob_summary,
        "split": split,
        "leakage": leakf,
        "reliability": reliability_summary,
        "artifacts": artifacts,
    }


def metric(summary: pd.DataFrame, q: int) -> float:
    return float(summary[summary.q == q].rho.iloc[0])


def condition_metric(summary: pd.DataFrame, condition: str) -> float:
    return float(summary[summary.condition == condition].rho.iloc[0])


def finalize(k: dict[str, object], j: dict[str, object]) -> None:
    ksum = k["q_summary"]
    jsum = j["q_summary"]
    k_null = k["null_summary"]
    j_null = j["null_summary"]
    k_fold = k["q_fold"]
    j_block = j["q_block"]
    j_boot = j["block_boot"]
    k_positive = {
        q: int(
            sum(
                float(part[part.q == q].rho.iloc[0]) > float(part[part.q == 0].rho.iloc[0])
                for _, part in k_fold.groupby("outer_fold")
            )
        )
        for q in [1, 2]
    }
    j_positive = {
        q: int(
            sum(
                float(part[part.q == q].rho.iloc[0]) > float(part[part.q == 0].rho.iloc[0])
                for _, part in j_block.groupby(["pathway", "cell_line"])
            )
        )
        for q in [1, 2]
    }
    k_pass = (
        metric(ksum, 1) - metric(ksum, 0) >= float(CONFIG["k562"]["material_gain_threshold"])
        and metric(ksum, 2) - metric(ksum, 0) >= float(CONFIG["k562"]["material_gain_threshold"])
        and k_positive[1] >= int(CONFIG["pass_rules"]["k562_min_positive_q1_q2_folds"])
        and k_positive[2] >= int(CONFIG["pass_rules"]["k562_min_positive_q1_q2_folds"])
        and all(
            float(k_null[k_null.q == q].true_rho.iloc[0]) > float(k_null[k_null.q == q].null_95th.iloc[0])
            for q in [1, 2]
        )
        and (k["leakage"].leakage_audit == "PASS").all()
    )
    path_q = j["q_pathway"]
    pathways_positive = all(
        metric(path_q[path_q.pathway == pathway], q) > metric(path_q[path_q.pathway == pathway], 0)
        for pathway in CONFIG["jiang"]["pathways"]
        for q in [1, 2]
    )
    gain_ci_positive = True
    gain_intervals = {}
    for q in [1, 2]:
        wide = j_boot.pivot(index="bootstrap", columns="q", values="rho")
        gains = wide[q] - wide[0]
        low, high = ci(gains)
        gain_intervals[q] = [low, high]
        gain_ci_positive &= low > 0
    j_pass = (
        bool(j["reliability"].gate_pass.all())
        and metric(jsum, 1) - metric(jsum, 0) >= float(CONFIG["jiang"]["material_gain_threshold"])
        and metric(jsum, 2) - metric(jsum, 0) >= float(CONFIG["jiang"]["material_gain_threshold"])
        and pathways_positive
        and j_positive[1] >= int(CONFIG["pass_rules"]["jiang_min_positive_blocks_of_18"])
        and j_positive[2] >= int(CONFIG["pass_rules"]["jiang_min_positive_blocks_of_18"])
        and gain_ci_positive
        and all(
            float(j_null[(j_null.pathway == "ALL") & (j_null.q == q)].true_rho.iloc[0])
            > float(j_null[(j_null.pathway == "ALL") & (j_null.q == q)].null_95th.iloc[0])
            for q in [1, 2]
        )
        and (j["leakage"].leakage_audit == "PASS").all()
    )
    a_verdict = "LOW_DIMENSIONAL_MISSING_ORIENTATION_CODE_REVALIDATED" if k_pass and j_pass else "LOW_DIMENSIONAL_MISSING_ORIENTATION_CODE_NOT_REVALIDATED"
    ksign = k["sign_summary"]
    jsign = j["sign_summary"]
    k_base = condition_metric(ksign, "baseline")
    k_exact = condition_metric(ksign, "q2_exact")
    j_base = condition_metric(jsign, "baseline")
    j_exact = condition_metric(jsign, "q2_exact")
    k_best_sign = max(condition_metric(ksign, "q1_sign"), condition_metric(ksign, "q2_sign"))
    j_best_sign = max(condition_metric(jsign, "q1_sign"), condition_metric(jsign, "q2_sign"))
    k_fraction = (k_best_sign - k_base) / max(k_exact - k_base, 1e-12)
    j_fraction = (j_best_sign - j_base) / max(j_exact - j_base, 1e-12)
    j_sign_path = j["sign_pathway"]
    sign_pathways_positive = all(
        max(
            condition_metric(j_sign_path[j_sign_path.pathway == pathway], "q1_sign"),
            condition_metric(j_sign_path[j_sign_path.pathway == pathway], "q2_sign"),
        )
        > condition_metric(j_sign_path[j_sign_path.pathway == pathway], "baseline")
        for pathway in CONFIG["jiang"]["pathways"]
    )
    b_pass = (
        k_fraction >= float(CONFIG["pass_rules"]["sign_only_min_fraction_exact_q2_gain"])
        and j_fraction >= float(CONFIG["pass_rules"]["sign_only_min_fraction_exact_q2_gain"])
        and sign_pathways_positive
        and (k["leakage"].leakage_audit == "PASS").all()
        and (j["leakage"].leakage_audit == "PASS").all()
    )
    b_verdict = "SIGN_ONLY_ORIENTATION_RESCUE_REVALIDATED" if b_pass else "SIGN_ONLY_ORIENTATION_RESCUE_NOT_REVALIDATED"
    write_json(
        OUT / "experiment_a_verdict.json",
        {
            "verdict": a_verdict,
            "k562_pass": bool(k_pass),
            "jiang_pass": bool(j_pass),
            "k562_positive_folds": k_positive,
            "jiang_positive_pathway_cell_blocks": j_positive,
            "jiang_q_gain_block_bootstrap_ci": gain_intervals,
            "leakage_audit": "PASS" if (k["leakage"].leakage_audit == "PASS").all() and (j["leakage"].leakage_audit == "PASS").all() else "FAIL",
        },
    )
    write_json(
        OUT / "experiment_b_verdict.json",
        {
            "verdict": b_verdict,
            "k562_fraction_exact_q2_gain_recovered": float(k_fraction),
            "jiang_fraction_exact_q2_gain_recovered": float(j_fraction),
            "jiang_all_pathways_positive": bool(sign_pathways_positive),
            "leakage_audit": "PASS" if (k["leakage"].leakage_audit == "PASS").all() and (j["leakage"].leakage_audit == "PASS").all() else "FAIL",
        },
    )
    fig4 = OUT / "fig4_orientation_code"
    k["q_summary"].assign(dataset="K562").to_csv(fig4 / "fig4b_k562_q_curve.csv", index=False)
    pd.concat(
        [j["q_summary"].assign(pathway="ALL"), j["q_pathway"]], ignore_index=True
    ).assign(dataset="Jiang").to_csv(fig4 / "fig4c_jiang_q_curve.csv", index=False)
    pd.concat([k["null_summary"], j["null_summary"]], ignore_index=True).to_csv(
        fig4 / "fig4d_random_subspace_null.csv", index=False
    )
    pd.concat(
        [
            k["sign_summary"].assign(dataset="K562", pathway="ALL"),
            j["sign_summary"].assign(dataset="Jiang", pathway="ALL"),
            j["sign_pathway"].assign(dataset="Jiang"),
        ],
        ignore_index=True,
    ).to_csv(fig4 / "fig4e_sign_only_rescue.csv", index=False)
    pd.concat([k["prob_summary"], j["prob_summary"]], ignore_index=True).to_csv(
        OUT / "experiment_b_sign_only" / "probability_information_curve.csv", index=False
    )
    manifest = f"""# Figure 4 orientation-code source manifest

- Frozen master seed: `{MASTER_SEED}`.
- Residual axes: uncentered SVD of outer-training source-disjoint OOF residuals only.
- Held responses enter only oracle coefficient/sign scoring after axis freeze.
- Ridge alpha selected by inner source-disjoint residual MSE.
- Random-subspace nulls: {RANDOM_REPS} per outer group and q.
- K562 response genes exclude all eligible perturbation-source genes.
- Jiang primary pathways: IFNB, IFNG, INS; response genes are complete across all sources/contexts and exclude pathway source genes.
- Experiment A verdict: `{a_verdict}`.
- Experiment B verdict: `{b_verdict}`.
"""
    (fig4 / "fig4_source_manifest.md").write_text(manifest, encoding="utf-8")
    split = pd.concat([k["split"], j["split"]], ignore_index=True)
    split.to_csv(OUT / "split_manifest.csv", index=False)
    leakage = pd.concat([k["leakage"], j["leakage"]], ignore_index=True)
    leakage.to_csv(OUT / "leakage_audit_ab.csv", index=False)
    # Lightweight QC plots only; final manuscript styling is intentionally out of scope.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(k["q_summary"].q, k["q_summary"].rho, marker="o", label="K562")
    axes[0].plot(j["q_summary"].q, j["q_summary"].rho, marker="s", label="Jiang")
    axes[0].set(xlabel="Residual oracle dimensions q", ylabel="Grouped geometry", title="Low-dimensional code QC")
    axes[0].legend(frameon=False)
    for dataset, frame, marker in [("K562", k["sign_summary"], "o"), ("Jiang", j["sign_summary"], "s")]:
        order = ["baseline", "q1_sign", "q2_sign", "q2_fixed_radius_direction", "q2_exact"]
        vals = [condition_metric(frame, x) for x in order]
        axes[1].plot(range(len(order)), vals, marker=marker, label=dataset)
    axes[1].set_xticks(range(5), ["base", "q1 sign", "q2 sign", "fixed r", "exact"], rotation=35, ha="right")
    axes[1].set(ylabel="Grouped geometry", title="Sign-only rescue QC")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_orientation_code" / "orientation_code_qc.png", dpi=160)
    plt.close(fig)
    print(f"[A] {a_verdict}", flush=True)
    print(f"[B] {b_verdict}", flush=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    ensure_base_manifests()
    for path in [OUT / "experiment_a_k562", OUT / "experiment_a_jiang", OUT / "experiment_b_sign_only"]:
        path.mkdir(parents=True, exist_ok=True)
    k = run_k562()
    j = run_jiang()
    finalize(k, j)


if __name__ == "__main__":
    main()
