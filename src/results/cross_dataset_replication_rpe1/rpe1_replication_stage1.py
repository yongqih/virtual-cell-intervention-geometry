from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy import sparse, stats
from scipy.spatial.distance import pdist
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "cross_dataset_replication_rpe1"
DATA = ROOT / "data" / "raw" / "replogle22rpe1_processed_complete.valid.h5ad"
K562_DATA = ROOT / "results" / "directedT_exploration" / "development_data.npz"
K562_BRIDGE = ROOT / "results" / "identifiability_transformer_bridge"
K562_AUTOPSY = ROOT / "results" / "mlp_transformer_matched_autopsy"
K562_ROBUST = ROOT / "results" / "pre_replication_robustness_audit"
QC = OUT / "qc"
METRIC = OUT / "metric_illusion"
GEOM = OUT / "geometry"
STATE = OUT / "state_intervention"
REL = OUT / "reliability"
CROSS = OUT / "cross_context"
FIG = OUT / "figures"
CACHE = OUT / "cache"
PRED = OUT / "predictions"
CKPT = OUT / "checkpoints"

FOLD_SEED = 1701
MODEL_SEED = 17
MODEL_GENES = 2560
PSEUDO_SPLITS = 10
BOOT = 1000
RANDOM_PANEL_SEED = 4701

sys.path.insert(0, str(ROOT))
CANONICAL_SRC = Path(os.environ.get("AI4SCI_CANONICAL_SRC", ROOT / "external" / "Virtual_Cell" / "real_perturbseq_pilot" / "src"))
if not CANONICAL_SRC.is_dir():
    raise RuntimeError("Set AI4SCI_CANONICAL_SRC to the real_perturbseq_pilot/src directory")
sys.path.insert(0, str(CANONICAL_SRC))
from directedT_core import make_direct_mask  # noqa: E402
from tiny_semantic_core import TinyConfig, TinySemanticTransformer, parameter_count, predict, seed_all  # noqa: E402
from real_perturbseq.models.baselines import MLPBaseline  # noqa: E402


def configure() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    for path in (QC, METRIC, GEOM, STATE, REL, CROSS, FIG, CACHE, PRED, CKPT):
        path.mkdir(parents=True, exist_ok=True)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ac = a - a.mean(1, keepdims=True)
    bc = b - b.mean(1, keepdims=True)
    den = np.linalg.norm(ac, axis=1) * np.linalg.norm(bc, axis=1)
    return np.sum(ac * bc, axis=1) / np.maximum(den, 1e-12)


def cosine_safe(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(a[ok], b[ok]).statistic)


def response_distances(x: np.ndarray) -> np.ndarray:
    return pdist(normalize_rows(x), metric="cosine")


def geometry_corr(pred: np.ndarray, truth: np.ndarray) -> float:
    if len(pred) < 4:
        return float("nan")
    return spearman_safe(response_distances(pred), response_distances(truth))


def rank_metrics(matrix: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    x = np.asarray(matrix, np.float64)
    x = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    if s.sum() <= 1e-12:
        empty = {"entropy_effective_rank": 0.0, "participation_ratio": 0.0, "pcs_80": 0,
                 "pcs_90": 0, "pcs_95": 0, "pc1_variance_fraction": 0.0,
                 "top5_variance_fraction": 0.0}
        return empty, np.zeros_like(s)
    p_s = s / s.sum()
    entropy_rank = float(np.exp(-np.sum(p_s[p_s > 0] * np.log(p_s[p_s > 0]))))
    weights = s ** 2 / max(np.sum(s ** 2), 1e-12)
    cum = np.cumsum(weights)
    return {
        "entropy_effective_rank": entropy_rank,
        "participation_ratio": float(1 / max(np.sum(weights ** 2), 1e-12)),
        "pcs_80": int(np.searchsorted(cum, .80) + 1),
        "pcs_90": int(np.searchsorted(cum, .90) + 1),
        "pcs_95": int(np.searchsorted(cum, .95) + 1),
        "pc1_variance_fraction": float(weights[0]),
        "top5_variance_fraction": float(weights[:5].sum()),
    }, weights


def bootstrap_fold_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = np.mean(values[rng.integers(0, len(values), (BOOT, len(values)))], axis=1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def top_fraction_recall(pred: np.ndarray, truth: np.ndarray, fraction: float = .10) -> float:
    k = max(1, int(math.ceil(len(pred) * fraction)))
    pi = set(np.argpartition(np.abs(pred), -k)[-k:].tolist())
    ti = set(np.argpartition(np.abs(truth), -k)[-k:].tolist())
    return len(pi & ti) / k


def pseudobulk_cache() -> dict[str, np.ndarray]:
    cache = CACHE / "rpe1_pseudobulk_full.npz"
    if cache.exists():
        print("  读取已校验的全基因 pseudobulk 缓存。", flush=True)
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    print("  首次流式扫描 240,774 个细胞；不把完整表达矩阵载入内存。", flush=True)
    data = ad.read_h5ad(DATA, backed="r")
    labels = data.obs["perturbation"].astype(str).to_numpy()
    perturbations = np.asarray(sorted(x for x in np.unique(labels) if x != "control"), dtype=str)
    order = np.concatenate([["control"], perturbations])
    lookup = {x: i for i, x in enumerate(order)}
    sums = np.zeros((len(order), data.n_vars), dtype=np.float64)
    observed = np.zeros(len(order), dtype=np.int64)
    chunk = 4096
    for start in range(0, data.n_obs, chunk):
        end = min(start + chunk, data.n_obs)
        x = data.X[start:end]
        if not sparse.issparse(x):
            x = sparse.csr_matrix(x)
        labs = labels[start:end]
        for label in np.unique(labs):
            loc = np.flatnonzero(labs == label)
            pos = lookup[label]
            sums[pos] += np.asarray(x[loc].sum(0)).ravel()
            observed[pos] += len(loc)
        if (start // chunk) % 10 == 0 or end == data.n_obs:
            print(f"    pseudobulk 扫描：{end:,}/{data.n_obs:,} cells", flush=True)
    means = (sums / observed[:, None]).astype(np.float32)
    delta = (means[1:] - means[0]).astype(np.float32)
    genes = data.var_names.astype(str).to_numpy()
    data.file.close()
    np.savez_compressed(cache, perturbations=perturbations, genes=genes, counts=observed[1:],
                        control_count=observed[0], control_mean=means[0], delta=delta,
                        global_mean=(sums.sum(0) / observed.sum()).astype(np.float32))
    print(f"  pseudobulk 缓存已写入：{cache.name}", flush=True)
    return {"perturbations": perturbations, "genes": genes, "counts": observed[1:],
            "control_count": np.asarray(observed[0]), "control_mean": means[0], "delta": delta,
            "global_mean": (sums.sum(0) / observed.sum()).astype(np.float32)}


def write_qc(pb: dict[str, np.ndarray]) -> np.ndarray:
    perturbations = pb["perturbations"].astype(str)
    genes = pb["genes"].astype(str)
    counts = pb["counts"].astype(int)
    covered = np.isin(perturbations, genes)
    delta = pb["delta"].astype(float)
    magnitude = np.linalg.norm(delta, axis=1)
    qc = pd.DataFrame({
        "perturbation": perturbations,
        "cells": counts,
        "label_class": "clean_single_gene",
        "source_gene_measured": covered,
        "primary_model_evaluation_eligible": covered,
        "exclusion_reason": np.where(covered, "", "source locus absent from var_names; retained in folds/QC and canonical missing-source training path"),
        "response_l2_all_genes": magnitude,
        "response_mean_abs_all_genes": np.mean(np.abs(delta), axis=1),
    })
    control = pd.DataFrame([{"perturbation": "control", "cells": int(pb["control_count"]),
                             "label_class": "control", "source_gene_measured": False,
                             "primary_model_evaluation_eligible": False, "exclusion_reason": "control reference",
                             "response_l2_all_genes": 0.0, "response_mean_abs_all_genes": 0.0}])
    pd.concat([control, qc], ignore_index=True).to_csv(QC / "perturbation_qc.csv", index=False)
    summary = {
        "cells": int(counts.sum() + int(pb["control_count"])),
        "genes": int(len(genes)),
        "expressed_genes_global_mean_gt_zero": int((pb["global_mean"] > 0).sum()),
        "perturbations": int(len(perturbations)),
        "control_cells": int(pb["control_count"]),
        "single_gene_perturbations": int(len(perturbations)),
        "ambiguous_or_multiple_perturbations": 0,
        "measured_source_perturbations": int(covered.sum()),
        "unmeasured_source_perturbations": int((~covered).sum()),
        "perturbations_ge_25_cells": int((counts >= 25).sum()),
        "perturbations_ge_50_cells": int((counts >= 50).sum()),
        "perturbations_ge_100_cells": int((counts >= 100).sum()),
        "median_cells_per_perturbation": float(np.median(counts)),
        "response_l2_median": float(np.median(magnitude)),
        "response_l2_q25": float(np.quantile(magnitude, .25)),
        "response_l2_q75": float(np.quantile(magnitude, .75)),
    }
    pd.DataFrame([summary]).to_csv(QC / "dataset_qc_summary.csv", index=False)
    return covered


def control_representations(perturbations: np.ndarray, genes: np.ndarray, covered: np.ndarray) -> dict[str, np.ndarray]:
    cache = CACHE / "control_state_representations.npz"
    if cache.exists():
        print("  读取 control-state 表示缓存。", flush=True)
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files if k != "perturbations"}
    data = ad.read_h5ad(DATA, backed="r")
    labels = data.obs["perturbation"].astype(str).to_numpy()
    control_rows = np.flatnonzero(labels == "control")
    gene_index = {g: i for i, g in enumerate(genes)}
    covered_names = perturbations[covered]
    columns = np.asarray([gene_index[g] for g in covered_names], int)
    block = data[control_rows, :].to_memory()[:, columns].X
    dense = block.toarray() if sparse.issparse(block) else np.asarray(block)
    profiles = dense.T.astype(np.float32)
    stats7 = np.column_stack([
        profiles.mean(1), profiles.std(1), (profiles > 0).mean(1),
        np.quantile(profiles, .25, axis=1), np.quantile(profiles, .50, axis=1),
        np.quantile(profiles, .75, axis=1), np.log1p((profiles > 0).sum(1)),
    ]).astype(np.float32)
    centered = (profiles - profiles.mean(1, keepdims=True)) / (profiles.std(1, keepdims=True) + 1e-6)
    pca64 = TruncatedSVD(n_components=64, random_state=FOLD_SEED).fit_transform(centered).astype(np.float32)
    combined = np.column_stack([stats7, pca64]).astype(np.float32)
    data.file.close()
    np.savez_compressed(cache, perturbations=covered_names, ControlStats7=stats7,
                        ControlProfilePCA64=pca64, EstablishedOBS71=combined)
    return {"ControlStats7": stats7, "ControlProfilePCA64": pca64, "EstablishedOBS71": combined}


def make_panel(pb: dict[str, np.ndarray], reference: np.ndarray, query: np.ndarray) -> dict[str, Any]:
    delta = pb["delta"].astype(np.float32)
    perturbations = pb["perturbations"].astype(str)
    genes = pb["genes"].astype(str)
    control = pb["control_mean"].astype(np.float32)
    gene_index = {g: i for i, g in enumerate(genes)}
    source_cols = list(dict.fromkeys(gene_index[p] for p in perturbations if p in gene_index))
    if len(source_cols) >= MODEL_GENES:
        raise RuntimeError("Frozen panel too small to hold measured source identities")
    response_mean = delta[reference].mean(0)
    residual = delta - response_mean
    selected, selected_set = list(source_cols), set(source_cols)
    variance = np.var(residual[reference], axis=0)
    name_order = np.argsort(genes, kind="stable")
    variance_order = name_order[np.argsort(-variance[name_order], kind="stable")]
    for idx in variance_order:
        if int(idx) not in selected_set:
            selected.append(int(idx)); selected_set.add(int(idx))
        if len(selected) >= MODEL_GENES:
            break
    gene_idx = np.asarray(sorted(selected), dtype=np.int64)
    panel_genes = genes[gene_idx]
    panel_index = {g: i for i, g in enumerate(panel_genes)}
    source_gene_index = np.asarray([panel_index.get(p, -1) for p in perturbations], dtype=np.int64)
    panel_control = control[gene_idx]
    gene_fixed = ((panel_control - panel_control.mean()) / (panel_control.std() + 1e-6))[:, None]
    common_trans = np.ones(len(panel_genes), bool)
    for p in perturbations:
        if p in panel_index:
            common_trans[panel_index[p]] = False
    return {
        "delta": delta[:, gene_idx], "residual": residual[:, gene_idx].astype(np.float32),
        "mean_response": response_mean[gene_idx].astype(np.float32), "genes": panel_genes,
        "control_mean": panel_control.astype(np.float32),
        "perturbations": perturbations, "gene_fixed": gene_fixed.astype(np.float32),
        "source_gene_index": source_gene_index, "common_trans": common_trans,
        "train_idx": reference, "val_idx": query, "gene_idx": gene_idx,
    }


def train_transformer(data: dict[str, Any], query_eval: np.ndarray, fold: int) -> tuple[np.ndarray, dict]:
    seed_all(MODEL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_idx = data["train_idx"]
    train_std = np.maximum(data["residual"][train_idx].std(0), .05).astype(np.float32)
    config = TinyConfig("ID", d_model=64, n_heads=4, n_latents=16, latent_layers=1, dropout=.10)
    model = TinySemanticTransformer(config, data, train_std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    target = torch.as_tensor(data["residual"], dtype=torch.float32, device=device)
    rng = np.random.default_rng(MODEL_SEED)
    losses, started = [], time.time()
    for epoch in range(25):
        model.train(); epoch_losses = []
        order = rng.permutation(train_idx)
        for start in range(0, len(order), 32):
            batch_np = order[start:start + 32]
            batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model(batch)
                mask = make_direct_mask(batch_np, data["perturbations"].tolist(), data["genes"].tolist(), device)
                pred_std, truth_std = prediction / model.train_std, target[batch] / model.train_std
                mse = ((pred_std - truth_std) ** 2)[mask].mean()
                pair_pred, pair_truth = pred_std - pred_std.roll(1, 0), truth_std - truth_std.roll(1, 0)
                pair_mask = mask & mask.roll(1, 0)
                pair_pred = pair_pred.masked_fill(~pair_mask, 0)
                pair_truth = pair_truth.masked_fill(~pair_mask, 0)
                contrast = (1 - F.cosine_similarity(pair_pred, pair_truth, dim=1)).mean()
                pair_mse = ((pair_pred - pair_truth) ** 2)[pair_mask].mean()
                loss = mse + .10 * contrast + .25 * pair_mse
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
        if (epoch + 1) % 5 == 0:
            print(f"      Transformer epoch {epoch+1}/25, train loss={losses[-1]:.4f}", flush=True)
    prediction = predict(model, query_eval, 32, device).astype(np.float32)
    torch.save({"state": model.state_dict(), "fold": fold, "seed": MODEL_SEED,
                "panel_genes": data["genes"], "train_std": train_std}, CKPT / f"transformer_fold{fold}_seed17.pt")
    return prediction, {"fold": fold, "model": "Transformer", "seed": MODEL_SEED, "epochs": 25,
                        "elapsed_seconds": time.time() - started, "final_train_loss": losses[-1],
                        "parameter_count": parameter_count(model), "device": str(device),
                        "query_truth_used_for_selection": False}


def direct_mask_np(rows: np.ndarray, perturbations: np.ndarray, genes: np.ndarray) -> np.ndarray:
    index = {g: i for i, g in enumerate(genes)}
    mask = np.ones((len(rows), len(genes)), bool)
    for i, row in enumerate(rows):
        if perturbations[row] in index:
            mask[i, index[perturbations[row]]] = False
    return mask


def mlp_pass(model: MLPBaseline, rows: np.ndarray, data: dict[str, Any], target: np.ndarray | None,
             optimizer: torch.optim.Optimizer | None, seed: int) -> tuple[np.ndarray | None, float]:
    device = next(model.parameters()).device
    genes, perturbations = data["genes"], data["perturbations"]
    index = {g: i for i, g in enumerate(genes)}
    control = ((data["gene_fixed"].ravel())).astype(np.float32)
    order = np.random.default_rng(seed).permutation(len(rows)) if optimizer is not None else np.arange(len(rows))
    outputs, losses = [], []
    model.train(optimizer is not None)
    for start in range(0, len(order), 32):
        loc = order[start:start + 32]
        source_rows = rows[loc]
        descriptor = np.zeros((len(loc), len(genes)), np.float32)
        missing = np.ones(len(loc), np.float32)
        flags = np.zeros((len(loc), len(genes)), np.int64)
        for j, source in enumerate(source_rows):
            p = perturbations[source]
            if p in index:
                descriptor[j, index[p]] = 1; missing[j] = 0; flags[j, index[p]] = 1
        batch = {
            "control": torch.as_tensor(np.repeat(control[None, :], len(loc), 0), device=device),
            "perturbation_descriptors": torch.as_tensor(descriptor, device=device),
            "perturbation_missing": torch.as_tensor(missing, device=device),
            "target_flags": torch.as_tensor(flags, device=device),
        }
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            pred = model.conditioned(**batch)
            if target is not None:
                truth = torch.as_tensor(target[loc], dtype=torch.float32, device=device)
                mask = torch.as_tensor(direct_mask_np(source_rows, perturbations, genes), device=device)
                loss = ((pred - truth) ** 2)[mask].mean()
        if optimizer is not None:
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        else:
            outputs.append(pred.float().cpu().numpy())
    if optimizer is None:
        return np.concatenate(outputs)[np.argsort(order)], float("nan")
    return None, float(np.mean(losses))


def train_mlp(data: dict[str, Any], query_eval: np.ndarray, fold: int) -> tuple[np.ndarray, dict]:
    seed_all(MODEL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    descriptors = torch.eye(len(data["genes"]), dtype=torch.float32)
    model = MLPBaseline(descriptors, torch.zeros(len(data["genes"])), d_model=48).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    started, losses = time.time(), []
    for epoch in range(4):
        _, loss = mlp_pass(model, data["train_idx"], data, data["residual"][data["train_idx"]], optimizer,
                           MODEL_SEED + epoch * 1009)
        losses.append(loss)
    with torch.no_grad():
        prediction, _ = mlp_pass(model, query_eval, data, None, None, MODEL_SEED)
    torch.save({"state": model.state_dict(), "fold": fold, "seed": MODEL_SEED,
                "panel_genes": data["genes"]}, CKPT / f"mlp_fold{fold}_seed17.pt")
    return prediction.astype(np.float32), {"fold": fold, "model": "MLP", "seed": MODEL_SEED, "epochs": 4,
                                             "elapsed_seconds": time.time() - started, "final_train_loss": losses[-1],
                                             "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                                             "device": str(device), "query_truth_used_for_selection": False}


def model_and_baseline_metrics(fold: int, data: dict[str, Any], query: np.ndarray,
                               predictions: dict[str, np.ndarray]) -> tuple[list[dict], list[dict], list[dict]]:
    mask = data["common_trans"]
    truth_res = data["residual"][query][:, mask].astype(float)
    truth_raw = data["delta"][query][:, mask].astype(float)
    mean = data["mean_response"][mask].astype(float)
    control = data["control_mean"][mask].astype(float)
    truth_abs = control[None, :] + truth_raw
    per_rows, model_rows, baseline_rows = [], [], []
    all_predictions = {
        "CopyControl": np.zeros_like(truth_res),
        "MeanResponse": np.zeros_like(truth_res),
        **{k: np.asarray(v[:, mask], float) for k, v in predictions.items()},
    }
    for model, pred_res in all_predictions.items():
        pred_raw = np.repeat(mean[None, :], len(truth_raw), axis=0) if model == "MeanResponse" else pred_res + mean[None, :]
        if model == "CopyControl":
            pred_raw = np.zeros_like(truth_raw)
        pred_abs = control[None, :] + pred_raw
        abs_r = np.array([pearson_safe(a, b) for a, b in zip(pred_abs, truth_abs)])
        raw_r = np.array([pearson_safe(a, b) for a, b in zip(pred_raw, truth_raw)])
        res_r = np.array([pearson_safe(a, b) for a, b in zip(pred_res, truth_res)])
        recalls = np.array([top_fraction_recall(a, b) for a, b in zip(pred_res, truth_res)])
        for i, source in enumerate(query):
            per_rows.append({"fold": fold, "query_row": int(source), "perturbation": data["perturbations"][source],
                             "model": model, "absolute_state_pearson": abs_r[i], "raw_delta_pearson": raw_r[i],
                             "residual_trans_pearson": res_r[i], "residual_mse": float(np.mean((pred_res[i]-truth_res[i])**2)),
                             "top10pct_residual_recall": recalls[i], "residual_prediction_norm": float(np.linalg.norm(pred_res[i])),
                             "raw_prediction_to_train_mean_cosine": cosine_safe(pred_raw[i], mean)})
        row = {"fold": fold, "model": model, "n_perturbations": len(query), "n_strict_trans_genes": int(mask.sum()),
               "absolute_state_pearson_mean": float(np.nanmean(abs_r)), "absolute_state_pearson_median": float(np.nanmedian(abs_r)),
               "raw_delta_pooled_pearson": pearson_safe(pred_raw.ravel(), truth_raw.ravel()),
               "raw_delta_per_perturbation_pearson_mean": float(np.nanmean(raw_r)),
               "residual_pooled_pearson": pearson_safe(pred_res.ravel(), truth_res.ravel()),
               "residual_per_perturbation_pearson_mean": float(np.nanmean(res_r)),
               "residual_mse": float(np.mean((pred_res-truth_res)**2)),
               "top10pct_residual_recall": float(np.mean(recalls)),
               "mean_response_similarity": float(np.nanmean([cosine_safe(x, mean) for x in pred_raw])),
               "residual_prediction_norm_mean": float(np.mean(np.linalg.norm(pred_res, axis=1)))}
        (baseline_rows if model in ("CopyControl", "MeanResponse") else model_rows).append(row)
    return per_rows, model_rows, baseline_rows


def run_models(pb: dict[str, np.ndarray], covered: np.ndarray) -> tuple[dict[int, dict], pd.DataFrame]:
    split = json.loads((OUT / "split_definition.json").read_text(encoding="utf-8"))
    perturbations = pb["perturbations"].astype(str)
    name_to_index = {x: i for i, x in enumerate(perturbations)}
    fold_cache, training_rows, per_rows, model_rows, baseline_rows = {}, [], [], [], []
    for f in split["folds"]:
        fold = int(f["fold"])
        reference = np.asarray([name_to_index[x] for x in f["train_sources"]], int)
        query_all = np.asarray([name_to_index[x] for x in f["validation_sources"]], int)
        query = query_all[covered[query_all]]
        print(f"  fold {fold+1}/5：构建 training-only 面板；eval sources={len(query)}。", flush=True)
        data = make_panel(pb, reference, query)
        if int(data["common_trans"].sum()) != MODEL_GENES - int(covered.sum()):
            raise RuntimeError("Strict-trans panel count changed unexpectedly")
        tpred, taudit = train_transformer(data, query, fold)
        print(f"    Transformer 完成：{taudit['elapsed_seconds']:.1f}s", flush=True)
        mpred, maudit = train_mlp(data, query, fold)
        print(f"    MLP 完成：{maudit['elapsed_seconds']:.1f}s", flush=True)
        training_rows.extend([taudit, maudit])
        np.savez_compressed(PRED / f"fold_{fold}.npz", perturbations=perturbations[query], query_rows=query,
                            panel_gene_indices=data["gene_idx"], panel_genes=data["genes"],
                            common_trans=data["common_trans"], truth_residual=data["residual"][query],
                            transformer_residual=tpred, mlp_residual=mpred, training_mean=data["mean_response"])
        pr, mr, br = model_and_baseline_metrics(fold, data, query, {"Transformer": tpred, "MLP": mpred})
        per_rows.extend(pr); model_rows.extend(mr); baseline_rows.extend(br)
        fold_cache[fold] = {"reference": reference, "query": query, "data": data,
                            "Truth": data["residual"][query].astype(float),
                            "Transformer": tpred.astype(float), "MLP": mpred.astype(float)}
    pd.DataFrame(training_rows).to_csv(OUT / "model_training_audit.csv", index=False)
    per = pd.DataFrame(per_rows)
    per.to_csv(METRIC / "mean_response_collapse.csv", index=False)
    pd.DataFrame(model_rows).to_csv(METRIC / "model_metric_comparison.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(METRIC / "baseline_metrics.csv", index=False)
    return fold_cache, per


def common_stacks(fold_cache: dict[int, dict]) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    intersection = None
    for fd in fold_cache.values():
        valid = set(fd["data"]["gene_idx"][fd["data"]["common_trans"]].tolist())
        intersection = valid if intersection is None else intersection & valid
    common = np.asarray(sorted(intersection), int)
    stacks = {m: [] for m in ("Truth", "Transformer", "MLP")}
    names = []
    for fold in range(5):
        fd = fold_cache[fold]; loc = {g: i for i, g in enumerate(fd["data"]["gene_idx"])}
        cols = np.asarray([loc[g] for g in common], int)
        names.extend(fd["data"]["perturbations"][fd["query"]].tolist())
        for model in stacks:
            stacks[model].append(fd[model][:, cols])
    return common, {k: np.vstack(v) for k, v in stacks.items()}, np.asarray(names)


def geometry_analysis(pb: dict[str, np.ndarray], fold_cache: dict[int, dict], common: np.ndarray,
                      stacks: dict[str, np.ndarray]) -> pd.DataFrame:
    rank_rows, spectra_rows, distance_rows, variance_rows, sensitivity_rows = [], [], [], [], []
    truth_var = float(np.mean(np.var(stacks["Truth"], axis=0)))
    for model in ("Truth", "Transformer", "MLP"):
        metrics, spectrum = rank_metrics(stacks[model])
        rank_rows.append({"panel": "primary_common_strict_trans", "model": model,
                          "n_perturbations": len(stacks[model]), "n_genes": stacks[model].shape[1], **metrics})
        spectra_rows.extend({"panel": "primary_common_strict_trans", "model": model, "component": i+1,
                             "variance_fraction": float(x), "cumulative_variance": float(spectrum[:i+1].sum())}
                            for i, x in enumerate(spectrum))
        variance_rows.append({"panel": "primary_common_strict_trans", "model": model,
                              "between_perturbation_variance": float(np.mean(np.var(stacks[model], axis=0))),
                              "variance_ratio_to_truth": 1.0 if model == "Truth" else float(np.mean(np.var(stacks[model], axis=0))/max(truth_var,1e-12))})
        distance_rows.append({"scope": "combined", "fold": "all", "panel": "primary_common_strict_trans",
                              "model": model, "n_perturbations": len(stacks[model]),
                              "n_pairs": len(stacks[model])*(len(stacks[model])-1)//2,
                              "response_distance_spearman": 1.0 if model == "Truth" else geometry_corr(stacks[model], stacks["Truth"])})
    # Primary and high-control-expression sensitivity on common stacked OOF data.
    high = np.flatnonzero(pb["control_mean"][common] >= np.median(pb["control_mean"][common]))
    for panel, cols in (("primary_common_strict_trans", np.arange(len(common))),
                        ("high_control_expression_half", high)):
        truth = stacks["Truth"][:, cols]
        for model in ("Truth", "Transformer", "MLP"):
            metrics, _ = rank_metrics(stacks[model][:, cols])
            sensitivity_rows.append({"panel": panel, "fold": "combined", "model": model,
                                     "n_perturbations": len(truth), "n_genes": len(cols),
                                     "response_distance_spearman": 1.0 if model == "Truth" else geometry_corr(stacks[model][:, cols], truth), **metrics})
    # Fold-local broader and one matched random panel.
    for fold, fd in fold_cache.items():
        trans = np.flatnonzero(fd["data"]["common_trans"])
        rng = np.random.default_rng(RANDOM_PANEL_SEED + fold)
        random_cols = np.sort(rng.choice(trans, size=min(len(common), len(trans)), replace=False))
        for panel, cols in (("fold_local_broader_strict_trans", trans),
                            ("random_response_blind_matched_size", random_cols)):
            truth = fd["Truth"][:, cols]
            for model in ("Truth", "Transformer", "MLP"):
                metrics, _ = rank_metrics(fd[model][:, cols])
                row = {"panel": panel, "fold": fold, "model": model, "n_perturbations": len(truth),
                       "n_genes": len(cols), "response_distance_spearman": 1.0 if model == "Truth" else geometry_corr(fd[model][:, cols], truth), **metrics}
                sensitivity_rows.append(row)
                if panel == "fold_local_broader_strict_trans":
                    distance_rows.append({"scope": "fold", "fold": fold, "panel": panel, "model": model,
                                          "n_perturbations": len(truth), "n_pairs": len(truth)*(len(truth)-1)//2,
                                          "response_distance_spearman": row["response_distance_spearman"]})
    rank = pd.DataFrame(rank_rows); rank.to_csv(GEOM / "rank_estimators.csv", index=False)
    pd.DataFrame(spectra_rows).to_csv(GEOM / "variance_spectra.csv", index=False)
    pd.DataFrame(distance_rows).to_csv(GEOM / "response_distance_geometry.csv", index=False)
    pd.DataFrame(variance_rows).to_csv(GEOM / "between_perturbation_variance.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(GEOM / "gene_panel_sensitivity.csv", index=False)
    return rank


def state_analysis(pb: dict[str, np.ndarray], covered: np.ndarray, reps: dict[str, np.ndarray],
                   fold_cache: dict[int, dict], common: np.ndarray) -> pd.DataFrame:
    perturbations, counts = pb["perturbations"].astype(str), pb["counts"].astype(int)
    covered_rows = np.flatnonzero(covered); covered_pos = {row: i for i, row in enumerate(covered_rows)}
    pair_rows, high_rows, nn_rows, sensitivity_rows = [], [], [], []
    high_common = np.flatnonzero(pb["control_mean"][common] >= np.median(pb["control_mean"][common]))
    for rep_name, rep in reps.items():
        fold_values = []
        for fold, fd in fold_cache.items():
            q = fd["query"]; r = fd["reference"][covered[fd["reference"]]]
            qpos = np.asarray([covered_pos[int(x)] for x in q]); rpos = np.asarray([covered_pos[int(x)] for x in r])
            scaler = StandardScaler().fit(rep[rpos])
            rq, rr = normalize_rows(scaler.transform(rep[qpos])), normalize_rows(scaler.transform(rep[rpos]))
            loc = {g: i for i, g in enumerate(fd["data"]["gene_idx"])}
            cols = np.asarray([loc[g] for g in common])
            response_q = fd["Truth"][:, cols]
            response_r = fd["data"]["residual"][r][:, cols].astype(float)
            state_sim = 1 - pdist(rq, metric="cosine")
            response_sim = 1 - pdist(normalize_rows(response_q), metric="cosine")
            rho = spearman_safe(state_sim, response_sim); fold_values.append(rho)
            top = state_sim >= np.quantile(state_sim, .95)
            divergent = response_sim <= np.quantile(response_sim, .25)
            pair_rows.append({"record_type": "fold", "representation": rep_name, "panel": "primary_common_strict_trans",
                              "cell_threshold": "all", "fold": fold, "n_sources": len(q), "n_pairs": len(state_sim),
                              "spearman_rho": rho})
            broader_cols = np.flatnonzero(fd["data"]["common_trans"])
            random_cols = np.sort(np.random.default_rng(RANDOM_PANEL_SEED + fold).choice(
                broader_cols, size=min(len(common), len(broader_cols)), replace=False
            ))
            panel_responses = {
                "high_control_expression_half": response_q[:, high_common],
                "fold_local_broader_strict_trans": fd["Truth"][:, broader_cols],
                "random_response_blind_matched_size": fd["Truth"][:, random_cols],
            }
            for panel_name, panel_response in panel_responses.items():
                panel_response_sim = 1 - pdist(normalize_rows(panel_response), metric="cosine")
                pair_rows.append({"record_type": "panel_sensitivity", "representation": rep_name,
                                  "panel": panel_name, "cell_threshold": "all", "fold": fold,
                                  "n_sources": len(q), "n_pairs": len(state_sim),
                                  "spearman_rho": spearman_safe(state_sim, panel_response_sim)})
            high_rows.append({"representation": rep_name, "panel": "primary_common_strict_trans", "cell_threshold": "all",
                              "fold": fold, "n_top5_pairs": int(top.sum()),
                              "median_outgoing_response_similarity": float(np.median(response_sim[top])),
                              "divergent_pair_fraction": float(divergent[top].mean())})
            similarity = rq @ rr.T
            nearest = similarity.argmax(1)
            nn_resp = row_pearson(response_q, response_r[nearest])
            oracle_mat = normalize_rows(response_q - response_q.mean(1, keepdims=True)) @ normalize_rows(response_r - response_r.mean(1, keepdims=True)).T
            oracle = oracle_mat.max(1)
            for local, source in enumerate(q):
                nn_rows.append({"record_type": "source", "representation": rep_name, "fold": fold,
                                "query_source": perturbations[source], "query_cells": counts[source],
                                "internal_neighbor": perturbations[r[nearest[local]]],
                                "internal_nn_response_similarity": nn_resp[local],
                                "oracle_nn_response_similarity": oracle[local], "oracle_gap": oracle[local]-nn_resp[local]})
            # Cell-count sensitivity for Claim C.
            for threshold in (50, 100):
                qkeep = counts[q] >= threshold
                if qkeep.sum() < 10:
                    continue
                ss = 1 - pdist(rq[qkeep], metric="cosine")
                rs = 1 - pdist(normalize_rows(response_q[qkeep]), metric="cosine")
                sensitivity_rows.append({"representation": rep_name, "fold": fold, "cell_threshold": threshold,
                                         "n_sources": int(qkeep.sum()), "spearman_rho": spearman_safe(ss, rs),
                                         "panel": "primary_common_strict_trans"})
        point, lo, hi = bootstrap_fold_mean(np.asarray(fold_values), FOLD_SEED + len(pair_rows))
        pair_rows.append({"record_type": "summary", "representation": rep_name, "panel": "primary_common_strict_trans",
                          "cell_threshold": "all", "fold": "all", "n_sources": int(covered.sum()),
                          "n_pairs": sum(x["n_pairs"] for x in pair_rows if x.get("record_type") == "fold" and x.get("representation") == rep_name),
                          "spearman_rho": point, "ci_low": lo, "ci_high": hi, "bootstrap_unit": "five source-disjoint folds"})
    sensitivity_frame = pd.DataFrame([x for x in pair_rows if x["record_type"] == "panel_sensitivity"])
    for (name, panel_name), part in sensitivity_frame.groupby(["representation", "panel"]):
        point, lo, hi = bootstrap_fold_mean(part.spearman_rho.to_numpy(),
                                            FOLD_SEED + len(pair_rows) + len(name) + len(panel_name))
        pair_rows.append({"record_type": "panel_summary", "representation": name, "panel": panel_name,
                          "cell_threshold": "all", "fold": "all", "n_sources": int(covered.sum()),
                          "n_pairs": int(part.n_pairs.sum()), "spearman_rho": point,
                          "ci_low": lo, "ci_high": hi, "bootstrap_unit": "five source-disjoint folds"})
    pair = pd.DataFrame(pair_rows); pair.to_csv(STATE / "pairwise_geometry_alignment.csv", index=False)
    pd.DataFrame(high_rows).to_csv(STATE / "high_similarity_divergent_pairs.csv", index=False)
    nn = pd.DataFrame(nn_rows)
    summary_rows = []
    for name, part in nn.groupby("representation"):
        summary_rows.append({"record_type": "summary", "representation": name, "fold": "all",
                             "n_sources": len(part), "internal_nn_response_similarity": part.internal_nn_response_similarity.median(),
                             "oracle_nn_response_similarity": part.oracle_nn_response_similarity.median(),
                             "oracle_gap": part.oracle_gap.median()})
    pd.concat([nn, pd.DataFrame(summary_rows)], ignore_index=True, sort=False).to_csv(STATE / "nearest_neighbor_oracle_gap.csv", index=False)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sens_summaries = []
    for (name, threshold), part in sensitivity.groupby(["representation", "cell_threshold"]):
        point, lo, hi = bootstrap_fold_mean(part.spearman_rho.to_numpy(), FOLD_SEED + int(threshold) + len(sens_summaries))
        sens_summaries.append({"representation": name, "fold": "all", "cell_threshold": threshold,
                               "n_sources": int(part.n_sources.sum()), "spearman_rho": point,
                               "ci_low": lo, "ci_high": hi, "panel": "primary_common_strict_trans"})
    pd.concat([sensitivity, pd.DataFrame(sens_summaries)], ignore_index=True, sort=False).to_csv(STATE / "cell_count_sensitivity.csv", index=False)
    pd.DataFrame([{"representation": k, "dimensions": v.shape[1], "source": "control cells only",
                   "response_labels_used": False, "fold_scaling": "fit on reference source genes only"}
                  for k, v in reps.items()]).to_csv(STATE / "representation_summary.csv", index=False)
    return pair[(pair.record_type == "summary") & (pair.panel == "primary_common_strict_trans")].copy()


def mean_sparse_rows(matrix, indices: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[indices].mean(0)).ravel().astype(np.float64)


def reliability_analysis(pb: dict[str, np.ndarray], covered: np.ndarray, common: np.ndarray) -> pd.DataFrame:
    perturbations, genes, counts = pb["perturbations"].astype(str), pb["genes"].astype(str), pb["counts"].astype(int)
    covered_names = perturbations[covered]
    data = ad.read_h5ad(DATA, backed="r")
    labels_all = data.obs["perturbation"].astype(str).to_numpy()
    selected_blocks, selected_labels = [], []
    chunk = 4096
    wanted = set(covered_names.tolist())
    for start in range(0, data.n_obs, chunk):
        end = min(start + chunk, data.n_obs)
        chunk_labels = labels_all[start:end]
        keep = np.asarray([x in wanted for x in chunk_labels])
        if keep.any():
            x = data.X[start:end]
            x = sparse.csr_matrix(x) if not sparse.issparse(x) else x.tocsr()
            selected_blocks.append(x[keep][:, common])
            selected_labels.append(chunk_labels[keep])
    block = sparse.vstack(selected_blocks, format="csr")
    labels = np.concatenate(selected_labels)
    data.file.close()
    groups = [np.flatnonzero(labels == p) for p in covered_names]
    group_counts = np.asarray([len(x) for x in groups])
    control = pb["control_mean"][common].astype(float)
    per_rows, geometry_rows = [], []
    for split in range(PSEUDO_SPLITS):
        a, b = [], []
        for source, indices in enumerate(groups):
            rng = np.random.default_rng(FOLD_SEED * 100000 + split * 1000 + source)
            order = rng.permutation(indices); half = len(order) // 2
            a.append(mean_sparse_rows(block, order[:half]) - control)
            b.append(mean_sparse_rows(block, order[half:]) - control)
        a, b = np.vstack(a), np.vstack(b)
        for threshold in (50, 100):
            keep = group_counts >= threshold
            ra, rb = a[keep] - a[keep].mean(0), b[keep] - b[keep].mean(0)
            raw_r = row_pearson(a[keep], b[keep]); residual_r = row_pearson(ra, rb)
            for p, n, x, y in zip(covered_names[keep], group_counts[keep], raw_r, residual_r):
                per_rows.append({"split": split, "cell_threshold": threshold, "perturbation": p,
                                 "cell_count": n, "split_half_raw_response_pearson": x,
                                 "split_half_residual_response_pearson": y})
            geometry_rows.append({"record_type": "split", "split": split, "cell_threshold": threshold,
                                  "n_perturbations": int(keep.sum()),
                                  "response_geometry_reproducibility": geometry_corr(ra, rb),
                                  "raw_response_pearson_median": float(np.nanmedian(raw_r)),
                                  "residual_response_pearson_median": float(np.nanmedian(residual_r))})
        print(f"    reliability split {split+1}/{PSEUDO_SPLITS}", flush=True)
    per = pd.DataFrame(per_rows); per.to_csv(REL / "split_or_replicate_reliability.csv", index=False)
    geom = pd.DataFrame(geometry_rows)
    summaries = []
    for threshold, part in geom.groupby("cell_threshold"):
        row = {"record_type": "summary", "split": "all", "cell_threshold": threshold,
               "n_perturbations": int(part.n_perturbations.iloc[0])}
        for i, metric in enumerate(("response_geometry_reproducibility", "raw_response_pearson_median", "residual_response_pearson_median")):
            point, lo, hi = bootstrap_fold_mean(part[metric].to_numpy(), FOLD_SEED + int(threshold)*10 + i)
            row[metric], row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = point, lo, hi
        summaries.append(row)
    frame = pd.concat([geom, pd.DataFrame(summaries)], ignore_index=True, sort=False)
    frame.to_csv(REL / "response_geometry_reproducibility.csv", index=False)
    metadata = {"dataset": str(DATA.relative_to(ROOT)), "available_metadata_fields": ["perturbation"],
                "genuine_biological_replicate_available": False, "method": "deterministic split-cell pseudoreplicates",
                "warning": "conditional measurement reliability, not biological replicate reproducibility",
                "split_count": PSEUDO_SPLITS, "thresholds": [50, 100], "gene_panel": "primary common strict-trans",
                "n_genes": len(common), "sealed_k562_test_open_count": 0}
    (REL / "reliability_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame[frame.record_type == "summary"].copy()


def k562_reference() -> dict[str, Any]:
    z = np.load(K562_DATA, allow_pickle=True)
    delta, perturbations, genes = z["delta"].astype(float), z["perturbations"].astype(str), z["genes"].astype(str)
    control = z["control_mean"].astype(float); gene_index = {g: i for i, g in enumerate(genes)}
    covered = np.asarray([p in gene_index for p in perturbations])
    folds = list(KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED).split(np.arange(len(perturbations))))
    common_set = None
    for fold in range(5):
        f = np.load(K562_BRIDGE / f"_oof_predictions_fold{fold}_seed17.npz", allow_pickle=True)
        idx = f["panel_gene_indices"].astype(int); names = genes[idx]; mask = np.ones(len(idx), bool)
        pindex = {g: i for i, g in enumerate(names)}
        for p in perturbations:
            if p in pindex: mask[pindex[p]] = False
        valid = set(idx[mask]); common_set = valid if common_set is None else common_set & valid
    common = np.asarray(sorted(common_set), int)
    copy_values, truths, mean_preds = [], [], []
    for reference, query in folds:
        q = query[covered[query]]; mean = delta[reference].mean(0)[common]
        truth_raw = delta[q][:, common]; truth_abs = control[common][None, :] + truth_raw
        pred_abs = np.repeat(control[common][None, :], len(q), axis=0)
        copy_values.extend(row_pearson(pred_abs, truth_abs)); truths.append(truth_raw); mean_preds.append(np.repeat(mean[None, :], len(q), 0))
    overall = pd.read_csv(K562_AUTOPSY / "overall_performance.csv").set_index("model")
    rank = pd.read_csv(K562_ROBUST / "rank" / "rank_estimators.csv").set_index("model")
    panel = pd.read_csv(K562_ROBUST / "rank" / "gene_panel_sensitivity.csv")
    panel = panel[(panel.panel == "current_common_strict_trans") & (panel.fold.astype(str) == "combined")].set_index("model")
    obs = pd.read_csv(K562_ROBUST / "obs_representation" / "representation_comparison.csv")
    obs = obs[obs.record_type == "summary"].set_index("representation")
    rel = pd.read_csv(K562_ROBUST / "reliability" / "geometry_noise_ceiling.csv")
    rel = rel[(rel.record_type == "summary") & (rel.cell_threshold == 50)].iloc[0]
    return {"copy_control_absolute_pearson": float(np.mean(copy_values)),
            "mean_response_raw_pooled_pearson": pearson_safe(np.vstack(mean_preds).ravel(), np.vstack(truths).ravel()),
            "transformer_residual": float(overall.loc["Transformer", "pooled_residual_pearson"]),
            "mlp_residual": float(overall.loc["MLP", "pooled_residual_pearson"]),
            "truth_pc1": float(rank.loc["Truth", "pc1_variance_fraction"]),
            "transformer_pc1": float(rank.loc["Transformer", "pc1_variance_fraction"]),
            "mlp_pc1": float(rank.loc["MLP", "pc1_variance_fraction"]),
            "truth_pc80": int(rank.loc["Truth", "pcs_80"]), "transformer_pc80": int(rank.loc["Transformer", "pcs_80"]),
            "mlp_pc80": int(rank.loc["MLP", "pcs_80"]),
            "transformer_distance": float(panel.loc["Transformer", "response_distance_correlation"]),
            "mlp_distance": float(panel.loc["MLP", "response_distance_correlation"]),
            "state_rho_min": float(obs.similarity_to_response_rho.min()), "state_rho_max": float(obs.similarity_to_response_rho.max()),
            "geometry_reliability": float(rel.response_geometry_reproducibility),
            "internal_nn": float(obs.loc["EstablishedOBS71", "nearest_neighbor_response_similarity"]),
            "oracle_nn": float(obs.loc["EstablishedOBS71", "oracle_response_similarity"])}


def verdict_and_cross(pb: dict[str, np.ndarray], rank: pd.DataFrame, pair_summary: pd.DataFrame,
                      rel_summary: pd.DataFrame, per: pd.DataFrame) -> tuple[str, dict, pd.DataFrame]:
    model_summary = pd.read_csv(METRIC / "model_metric_comparison.csv").groupby("model").mean(numeric_only=True)
    base_summary = pd.read_csv(METRIC / "baseline_metrics.csv").groupby("model").mean(numeric_only=True)
    rank_i = rank.set_index("model")
    dist = pd.read_csv(GEOM / "response_distance_geometry.csv")
    dist = dist[(dist.scope == "combined") & (dist.panel == "primary_common_strict_trans")].set_index("model")
    nn = pd.read_csv(STATE / "nearest_neighbor_oracle_gap.csv")
    nn = nn[nn.record_type == "summary"].set_index("representation")
    pair = pair_summary.set_index("representation")
    rel50 = rel_summary[rel_summary.cell_threshold == 50].iloc[0]
    # Frozen gates.
    a_signatures = [
        base_summary.loc["CopyControl", "absolute_state_pearson_mean"] >= .20,
        base_summary.loc["MeanResponse", "raw_delta_pooled_pearson"] - (base_summary.loc["MeanResponse", "residual_pooled_pearson"] if np.isfinite(base_summary.loc["MeanResponse", "residual_pooled_pearson"]) else 0) >= .10,
        all(model_summary.loc[m, "raw_delta_pooled_pearson"] - model_summary.loc[m, "residual_pooled_pearson"] >= .10 for m in ("Transformer", "MLP")),
    ]
    claim_a = "PASS" if sum(a_signatures) >= 2 else "PARTIAL" if sum(a_signatures) == 1 else "FAIL"
    primary_compress = {m: bool(rank_i.loc[m, "pc1_variance_fraction"] >= rank_i.loc["Truth", "pc1_variance_fraction"] + .10 and
                                rank_i.loc[m, "pcs_80"] <= .5 * rank_i.loc["Truth", "pcs_80"]) for m in ("Transformer", "MLP")}
    sensitivity = pd.read_csv(GEOM / "gene_panel_sensitivity.csv")
    sens_ok = {}
    for model in ("Transformer", "MLP"):
        good_panels = 0
        for panel_name, part in sensitivity[sensitivity.panel != "primary_common_strict_trans"].groupby("panel"):
            model_part, truth_part = part[part.model == model], part[part.model == "Truth"]
            merged = model_part.merge(truth_part, on="fold", suffixes=("_m", "_t"))
            if len(merged) and ((merged.pc1_variance_fraction_m >= merged.pc1_variance_fraction_t + .10) &
                                (merged.pcs_80_m <= .5 * merged.pcs_80_t)).mean() >= .6:
                good_panels += 1
        sens_ok[model] = bool(good_panels >= 2)
    claim_b = "PASS" if all(primary_compress.values()) and all(sens_ok.values()) else "PARTIAL" if any(primary_compress.values()) else "FAIL"
    state_ok = ((pair.spearman_rho.abs() < .10) & (nn.oracle_gap >= .15))
    reliability_ok = rel50.response_geometry_reproducibility >= .30
    claim_c = "PASS" if reliability_ok and state_ok.sum() >= 2 else "PARTIAL" if reliability_ok else "FAIL"
    if all(x == "PASS" for x in (claim_a, claim_b, claim_c)):
        verdict = "CORE_DIAGNOSIS_REPLICATED_IN_RPE1"
    elif claim_a == "PASS" and claim_b == "FAIL" and claim_c == "FAIL":
        verdict = "METRIC_ILLUSION_ONLY_REPLICATED"
    elif any(x in ("PASS", "PARTIAL") for x in (claim_a, claim_b, claim_c)):
        verdict = "CORE_DIAGNOSIS_PARTIALLY_REPLICATED_IN_RPE1"
    else:
        verdict = "RPE1_DOES_NOT_SUPPORT_CORE_DIAGNOSIS"
    k = k562_reference()
    r = {"copy_control_absolute_pearson": float(base_summary.loc["CopyControl", "absolute_state_pearson_mean"]),
         "mean_response_raw_pooled_pearson": float(base_summary.loc["MeanResponse", "raw_delta_pooled_pearson"]),
         "transformer_residual": float(model_summary.loc["Transformer", "residual_pooled_pearson"]),
         "mlp_residual": float(model_summary.loc["MLP", "residual_pooled_pearson"]),
         "truth_pc1": float(rank_i.loc["Truth", "pc1_variance_fraction"]), "transformer_pc1": float(rank_i.loc["Transformer", "pc1_variance_fraction"]),
         "mlp_pc1": float(rank_i.loc["MLP", "pc1_variance_fraction"]), "truth_pc80": int(rank_i.loc["Truth", "pcs_80"]),
         "transformer_pc80": int(rank_i.loc["Transformer", "pcs_80"]), "mlp_pc80": int(rank_i.loc["MLP", "pcs_80"]),
         "transformer_distance": float(dist.loc["Transformer", "response_distance_spearman"]),
         "mlp_distance": float(dist.loc["MLP", "response_distance_spearman"]),
         "state_rho_min": float(pair.spearman_rho.min()), "state_rho_max": float(pair.spearman_rho.max()),
         "geometry_reliability": float(rel50.response_geometry_reproducibility),
         "internal_nn": float(nn.loc["EstablishedOBS71", "internal_nn_response_similarity"]),
         "oracle_nn": float(nn.loc["EstablishedOBS71", "oracle_nn_response_similarity"])}
    rows = []
    labels = {
        "copy_control_absolute_pearson": "Copy-control absolute Pearson", "mean_response_raw_pooled_pearson": "Mean-response raw metric",
        "transformer_residual": "Transformer residual metric", "mlp_residual": "MLP residual metric",
        "truth_pc1": "Truth PC1 fraction", "transformer_pc1": "Transformer PC1 fraction", "mlp_pc1": "MLP PC1 fraction",
        "truth_pc80": "Truth PC80 dimension", "transformer_pc80": "Transformer PC80 dimension", "mlp_pc80": "MLP PC80 dimension",
        "transformer_distance": "Transformer response-distance corr", "mlp_distance": "MLP response-distance corr",
        "geometry_reliability": "Response-geometry reliability", "internal_nn": "Internal NN response similarity",
        "oracle_nn": "Oracle NN response similarity",
    }
    for key, label in labels.items():
        rows.append({"diagnostic": label, "k562": k[key], "rpe1": r[key]})
    rows.append({"diagnostic": "State→intervention rho range", "k562": f"{k['state_rho_min']:.3f} to {k['state_rho_max']:.3f}",
                 "rpe1": f"{r['state_rho_min']:.3f} to {r['state_rho_max']:.3f}"})
    cross = pd.DataFrame(rows); cross.to_csv(CROSS / "k562_rpe1_comparison.csv", index=False)
    claims = {"A": claim_a, "B": claim_b, "C": claim_c, "a_signatures": [bool(x) for x in a_signatures],
              "primary_compression": primary_compress, "sensitivity_compression": sens_ok,
              "state_representations_passing": int(state_ok.sum()), "response_geometry_reliable": bool(reliability_ok)}
    return verdict, claims, cross


def figures(rank: pd.DataFrame, pair: pd.DataFrame, rel: pd.DataFrame, cross: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    # Metric illusion.
    model = pd.read_csv(METRIC / "model_metric_comparison.csv").groupby("model", as_index=False).mean(numeric_only=True)
    base = pd.read_csv(METRIC / "baseline_metrics.csv").groupby("model", as_index=False).mean(numeric_only=True)
    metric_plot = pd.concat([base, model], ignore_index=True)
    melt = metric_plot.melt(id_vars="model", value_vars=["absolute_state_pearson_mean", "raw_delta_pooled_pearson", "residual_pooled_pearson"],
                            var_name="metric", value_name="pearson")
    melt["metric"] = melt["metric"].map({"absolute_state_pearson_mean": "Absolute state",
                                           "raw_delta_pooled_pearson": "Raw delta",
                                           "residual_pooled_pearson": "Residual trans"})
    plt.figure(figsize=(11, 6)); sns.barplot(data=melt, x="model", y="pearson", hue="metric")
    plt.axhline(0, color="black", lw=1); plt.ylabel("Pearson"); plt.xlabel(""); plt.xticks(rotation=12)
    plt.title("RPE1 conventional versus intervention-specific metrics")
    plt.legend(title="", loc="upper left", bbox_to_anchor=(1.01, 1)); plt.tight_layout()
    plt.savefig(FIG / "metric_illusion.png", dpi=180, bbox_inches="tight"); plt.close()
    # Spectra.
    spectra = pd.read_csv(GEOM / "variance_spectra.csv")
    plt.figure(figsize=(9, 6))
    for name in ("Truth", "Transformer", "MLP"):
        z = spectra[(spectra.model == name) & (spectra.component <= 40)]
        plt.plot(z.component, z.variance_fraction, marker="o" if name != "Truth" else None, ms=3, label=name)
    plt.yscale("log"); plt.xlabel("Principal component"); plt.ylabel("Variance fraction (log)")
    plt.title("RPE1 residual intervention spectra"); plt.legend(); plt.tight_layout(); plt.savefig(FIG / "variance_spectra.png", dpi=180); plt.close()
    # Rank summary.
    rm = rank.melt(id_vars="model", value_vars=["pc1_variance_fraction", "pcs_80", "participation_ratio"], var_name="metric", value_name="value")
    rm["metric"] = rm["metric"].map({"pc1_variance_fraction": "PC1 variance", "pcs_80": "PCs for 80%",
                                       "participation_ratio": "Participation ratio"})
    g = sns.catplot(data=rm, x="model", y="value", col="metric", kind="bar", sharey=False, height=4.5, aspect=.9, hue="model", legend=False)
    g.set_axis_labels("", "Value"); g.set_titles("{col_name}"); g.figure.suptitle("RPE1 geometry-compression diagnostics", y=1.04)
    g.figure.savefig(FIG / "rank_diagnostics.png", dpi=180, bbox_inches="tight"); plt.close(g.figure)
    # State alignment versus reliability.
    ps = pair[["representation", "spearman_rho", "ci_low", "ci_high"]].copy()
    x = np.arange(len(ps)); yerr = np.vstack([ps.spearman_rho-ps.ci_low, ps.ci_high-ps.spearman_rho])
    plt.figure(figsize=(9, 6)); plt.errorbar(x, ps.spearman_rho, yerr=yerr, fmt="o", capsize=5, label="State→response alignment")
    rel50 = rel[rel.cell_threshold == 50].iloc[0]; plt.axhline(rel50.response_geometry_reproducibility, color="#D55E00", lw=2, label="Split-cell geometry reliability")
    plt.axhline(0, color="black", lw=1); plt.xticks(x, ps.representation, rotation=15); plt.ylabel("Spearman rho")
    plt.title("State alignment relative to measurable response geometry"); plt.legend(); plt.tight_layout(); plt.savefig(FIG / "state_alignment_vs_reliability.png", dpi=180); plt.close()
    # NN oracle gap.
    nn = pd.read_csv(STATE / "nearest_neighbor_oracle_gap.csv"); nn = nn[nn.record_type == "summary"]
    nplot = nn.melt(id_vars="representation", value_vars=["internal_nn_response_similarity", "oracle_nn_response_similarity"], var_name="neighbor", value_name="response_similarity")
    nplot["neighbor"] = nplot["neighbor"].map({"internal_nn_response_similarity": "Internal state NN",
                                                "oracle_nn_response_similarity": "Response oracle NN"})
    plt.figure(figsize=(10, 6)); sns.barplot(data=nplot, x="representation", y="response_similarity", hue="neighbor")
    plt.ylabel("Median outgoing-response similarity"); plt.xlabel(""); plt.xticks(rotation=12); plt.title("RPE1 nearest-state-neighbor versus response oracle")
    plt.legend(title="", loc="upper left", bbox_to_anchor=(1.01, 1)); plt.tight_layout()
    plt.savefig(FIG / "nearest_neighbor_oracle_gap.png", dpi=180, bbox_inches="tight"); plt.close()
    # Reliability thresholds.
    plt.figure(figsize=(8, 6)); rr = rel.copy(); plt.errorbar(rr.cell_threshold, rr.response_geometry_reproducibility,
        yerr=np.vstack([rr.response_geometry_reproducibility-rr.response_geometry_reproducibility_ci_low,
                        rr.response_geometry_reproducibility_ci_high-rr.response_geometry_reproducibility]), fmt="o-", capsize=5)
    plt.xlabel("Minimum cells per perturbation"); plt.ylabel("Split-cell geometry reproducibility"); plt.title("RPE1 response-geometry reliability")
    plt.tight_layout(); plt.savefig(FIG / "response_geometry_reliability.png", dpi=180); plt.close()


def report(verdict: str, claims: dict, rank: pd.DataFrame, pair: pd.DataFrame, rel: pd.DataFrame, cross: pd.DataFrame) -> None:
    rank_i = rank.set_index("model"); pair_i = pair.set_index("representation")
    nn = pd.read_csv(STATE / "nearest_neighbor_oracle_gap.csv"); nn = nn[nn.record_type == "summary"].set_index("representation")
    model = pd.read_csv(METRIC / "model_metric_comparison.csv").groupby("model").mean(numeric_only=True)
    base = pd.read_csv(METRIC / "baseline_metrics.csv").groupby("model").mean(numeric_only=True)
    dist = pd.read_csv(GEOM / "response_distance_geometry.csv"); dist = dist[dist.scope == "combined"].set_index("model")
    rel50 = rel[rel.cell_threshold == 50].iloc[0]; rel100 = rel[rel.cell_threshold == 100].iloc[0]
    k = cross.set_index("diagnostic").k562; r = cross.set_index("diagnostic").rpe1
    metric_replicated = "YES" if claims["A"] == "PASS" else "PARTIAL" if claims["A"] == "PARTIAL" else "NO"
    text = f"""{verdict}

# Final RPE1 Stage 1 cross-dataset replication verdict

Claim A — Shared-response / metric illusion: **{claims['A']}**

Claim B — Intervention geometry compression: **{claims['B']}**

Claim C — Audited state/intervention geometry mismatch: **{claims['C']}**

| Diagnostic | K562 | RPE1 | Replicated? |
|---|---:|---:|---|
| Metric illusion | raw > residual | copy={base.loc['CopyControl','absolute_state_pearson_mean']:.3f}; mean raw={base.loc['MeanResponse','raw_delta_pooled_pearson']:.3f} | {metric_replicated} |
| Truth PC1 fraction | {float(k['Truth PC1 fraction']):.3f} | {rank_i.loc['Truth','pc1_variance_fraction']:.3f} | context metric |
| Transformer PC1 fraction | {float(k['Transformer PC1 fraction']):.3f} | {rank_i.loc['Transformer','pc1_variance_fraction']:.3f} | {'YES' if rank_i.loc['Transformer','pc1_variance_fraction']>rank_i.loc['Truth','pc1_variance_fraction'] else 'NO'} |
| MLP PC1 fraction | {float(k['MLP PC1 fraction']):.3f} | {rank_i.loc['MLP','pc1_variance_fraction']:.3f} | {'YES' if rank_i.loc['MLP','pc1_variance_fraction']>rank_i.loc['Truth','pc1_variance_fraction'] else 'NO'} |
| Truth PCs for 80% | {int(float(k['Truth PC80 dimension']))} | {int(rank_i.loc['Truth','pcs_80'])} | context metric |
| Transformer PCs for 80% | {int(float(k['Transformer PC80 dimension']))} | {int(rank_i.loc['Transformer','pcs_80'])} | {'YES' if rank_i.loc['Transformer','pcs_80']<rank_i.loc['Truth','pcs_80'] else 'NO'} |
| MLP PCs for 80% | {int(float(k['MLP PC80 dimension']))} | {int(rank_i.loc['MLP','pcs_80'])} | {'YES' if rank_i.loc['MLP','pcs_80']<rank_i.loc['Truth','pcs_80'] else 'NO'} |
| Transformer response-distance corr | {float(k['Transformer response-distance corr']):.3f} | {dist.loc['Transformer','response_distance_spearman']:.3f} | diagnostic |
| MLP response-distance corr | {float(k['MLP response-distance corr']):.3f} | {dist.loc['MLP','response_distance_spearman']:.3f} | diagnostic |
| State→intervention rho range | {k['State→intervention rho range']} | {r['State→intervention rho range']} | {'YES' if (pair_i.spearman_rho.abs()<.1).sum()>=2 else 'NO'} |
| Response-geometry reliability | {float(k['Response-geometry reliability']):.3f} | {rel50.response_geometry_reproducibility:.3f} | {'YES' if rel50.response_geometry_reproducibility>=.3 else 'NO'} |
| Internal NN response sim | {float(k['Internal NN response similarity']):.3f} | {nn.loc['EstablishedOBS71','internal_nn_response_similarity']:.3f} | diagnostic |
| Oracle NN response sim | {float(k['Oracle NN response similarity']):.3f} | {nn.loc['EstablishedOBS71','oracle_nn_response_similarity']:.3f} | diagnostic |

## Direct answers

1. Does shared-response metric inflation reproduce in RPE1? **{'YES' if claims['A']=='PASS' else 'PARTIAL' if claims['A']=='PARTIAL' else 'NO'}**.
2. Do Transformer/MLP again compress intervention-specific geometry? **{'YES' if claims['B']=='PASS' else 'PARTIAL' if claims['B']=='PARTIAL' else 'NO'}**.
3. Is compression robust across spectral measures and gene panels? **{'YES' if claims['B']=='PASS' else 'NO'}**.
4. Is RPE1 outgoing intervention geometry itself reliably measurable? **{'YES' if claims['response_geometry_reliable'] else 'NO'}** — split-cell pseudoreplicates, not biological replicates; rho={rel50.response_geometry_reproducibility:.3f} at >=50 and {rel100.response_geometry_reproducibility:.3f} at >=100 cells.
5. Are audited RPE1 state representations poorly aligned with outgoing intervention geometry? **{'YES' if claims['state_representations_passing']>=2 else 'NO'}**.
6. Does the nearest-state-neighbor vs oracle gap reproduce? **{'YES' if (nn.oracle_gap>=.15).sum()>=2 else 'NO'}**.
7. Which claims replicated? **A={claims['A']}; B={claims['B']}; C={claims['C']}**.
8. What differed materially from K562? **RPE1 has 2,016 perturbations and 1,755 measured sources, requiring a 2,560-gene panel; its hosted H5AD has no replicate/batch metadata; Stage 1 used one frozen seed and 10 split-cell pseudoreplicates.**
9. Does any result require weakening the manuscript story? **{'NO' if all(claims[x]=='PASS' for x in ('A','B','C')) else 'YES'}**.
10. Should we proceed to the orthogonal Norman K562 gain-of-function dataset? **{'YES' if sum(claims[x]=='PASS' for x in ('A','B','C'))>=2 else 'NO'}** — Norman was not opened or started.

No GO, GRN, DirectedT, external semantics, architecture search, performance-driven retuning, Norman data, or sealed K562 test was used. Sealed K562 test open count: **0**.
"""
    (OUT / "FINAL_RPE1_REPLICATION_VERDICT.md").write_text(text, encoding="utf-8")
    log = f"""# Research log

- Frozen plan/config/split timestamps precede all model predictions.
- scPertEval RPE1 H5AD SHA-256: `{json.loads((OUT/'dataset_manifest.json').read_text())['sha256']}`.
- Full pseudobulk was generated by chunked backed-H5AD accumulation; no full dense cell-by-gene matrix was loaded.
- Five source-disjoint folds, seed 1701; canonical model seed 17 only for lightweight Stage 1.
- Model dimension was mechanically enlarged to 2,560 genes to include all 1,755 measurable intervention identities.
- Training mean was fit on fold-reference perturbations only; query responses were never used in training or selection.
- Reliability used 10 deterministic split-cell pseudoreplicates because no biological replicate/batch field exists.
- Execution correction 1: the first fold completed before a MeanResponse row-broadcast shape bug stopped metric assembly; the bug was corrected without changing any frozen setting and all five model folds were rerun from the start.
- Execution correction 2: all analyses completed before a `numpy.bool_` JSON-serialization error stopped log writing; only explicit Python-bool conversion was added, then reports/figures/manifests were finalized from the already frozen CSV outputs without retraining.
- Claim verdicts: `{json.dumps(claims)}`.
- Final verdict: `{verdict}`.
- Norman and the sealed K562 test were not opened.
"""
    (OUT / "RESEARCH_LOG.md").write_text(log, encoding="utf-8")
    metadata = {"verdict": verdict, "claims": claims, "fold_seed": FOLD_SEED, "model_seed": MODEL_SEED,
                "model_gene_panel": MODEL_GENES, "pseudoreplicate_splits": PSEUDO_SPLITS,
                "sealed_k562_test_open_count": 0, "norman_open_count": 0,
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                "cuda": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def artifact_manifest() -> None:
    rows = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "artifact_manifest.csv" and "cache" not in p.parts and p.suffix != ".pt":
            rows.append({"path": str(p.relative_to(OUT)), "size_bytes": p.stat().st_size,
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    pd.DataFrame(rows).to_csv(OUT / "artifact_manifest.csv", index=False)


def main() -> None:
    configure()
    print("[1/9] 载入冻结计划；确认不读取 GO/GRN/Norman/K562 sealed test。", flush=True)
    for required in ("REPLICATION_PLAN.md", "dataset_manifest.json", "preprocessing_config.json", "model_configs.json", "split_definition.json"):
        if not (OUT / required).exists():
            raise RuntimeError(f"Missing preregistration artifact: {required}")
    print("[2/9] 构建/读取 RPE1 pseudobulk 并写 QC。", flush=True)
    pb = pseudobulk_cache(); covered = write_qc(pb)
    print(f"  QC：2016 clean singles；measured sources={covered.sum()}；>=50 cells={(pb['counts']>=50).sum()}；>=100={(pb['counts']>=100).sum()}。", flush=True)
    print("[3/9] 仅用 control cells 构建三种 observational state 表示。", flush=True)
    reps = control_representations(pb["perturbations"].astype(str), pb["genes"].astype(str), covered)
    print("[4/9] 运行 frozen canonical Transformer/MLP 的五折轻量 Stage 1。", flush=True)
    fold_cache, per = run_models(pb, covered)
    common, stacks, oof_names = common_stacks(fold_cache)
    print(f"[5/9] 计算 Claim A/B；primary common strict-trans panel={len(common)} genes，OOF={len(oof_names)}。", flush=True)
    rank = geometry_analysis(pb, fold_cache, common, stacks)
    print("[6/9] 计算 Claim C：state→response、top5% divergent pairs 与 NN-oracle gap。", flush=True)
    pair = state_analysis(pb, covered, reps, fold_cache, common)
    print("[7/9] 运行 10 次 split-cell pseudoreplicate 可靠性审计。", flush=True)
    rel = reliability_analysis(pb, covered, common)
    print("[8/9] 冻结 RPE1 后生成 K562/RPE1 横向表、图和 claim gates。", flush=True)
    verdict, claims, cross = verdict_and_cross(pb, rank, pair, rel, per)
    figures(rank, pair, rel, cross); report(verdict, claims, rank, pair, rel, cross); artifact_manifest()
    print(f"[9/9] 完成：{verdict}；A={claims['A']} B={claims['B']} C={claims['C']}；sealed K562 test=0；Norman=0。", flush=True)


if __name__ == "__main__":
    main()
