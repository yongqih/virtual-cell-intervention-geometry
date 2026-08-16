from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "phase1"
DATA = ROOT / "data" / "raw"
ADATA_PATH = DATA / "replogle22k562_processed_complete.valid.h5ad"
CORE_PATH = DATA / "omnipath_core.tsv"
COLLECTRI_PATH = DATA / "omnipath_collectri.tsv"
CONFIG_PATH = OUT / "preregistered_config.json"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def md5(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def join_unique(values: pd.Series) -> str:
    tokens: set[str] = set()
    for value in values.dropna().astype(str):
        for token in value.split(";"):
            token = token.strip()
            if token:
                tokens.add(token)
    return ";".join(sorted(tokens))


def freeze_network() -> tuple[pd.DataFrame, dict]:
    parts = []
    for origin, path in (("OmniPath_core", CORE_PATH), ("CollecTRI", COLLECTRI_PATH)):
        frame = pd.read_csv(path, sep="\t")
        frame = frame[
            frame["consensus_direction"].astype(bool)
            & (
                frame["consensus_stimulation"].astype(bool)
                ^ frame["consensus_inhibition"].astype(bool)
            )
        ].copy()
        frame["source_gene"] = frame["source_genesymbol"].astype(str).str.strip()
        frame["target_gene"] = frame["target_genesymbol"].astype(str).str.strip()
        frame["sign"] = np.where(frame["consensus_stimulation"].astype(bool), 1, -1)
        frame["origin"] = origin
        frame = frame[
            frame["source_gene"].ne("")
            & frame["target_gene"].ne("")
            & frame["source_gene"].ne("nan")
            & frame["target_gene"].ne("nan")
            & frame["source_gene"].ne(frame["target_gene"])
        ]
        parts.append(frame)

    combined = pd.concat(parts, ignore_index=True)
    sign_counts = combined.groupby(["source_gene", "target_gene"])["sign"].nunique()
    conflict_pairs = sign_counts[sign_counts > 1].index
    conflicts = set(conflict_pairs.tolist())
    keep = np.fromiter(
        ((s, t) not in conflicts for s, t in zip(combined.source_gene, combined.target_gene)),
        dtype=bool,
        count=len(combined),
    )
    combined = combined.loc[keep].copy()
    combined["evidence/source"] = (
        combined["origin"].astype(str)
        + "|resources="
        + combined["sources"].fillna("").astype(str)
        + "|references="
        + combined["references"].fillna("").astype(str)
    )
    frozen = (
        combined.groupby(["source_gene", "target_gene", "sign"], as_index=False)
        .agg({"evidence/source": join_unique})
        .rename(columns={"source_gene": "source", "target_gene": "target"})
        .sort_values(["source", "target", "sign"], kind="stable")
        .reset_index(drop=True)
    )
    frozen.to_csv(OUT / "frozen_directed_signed_network.csv", index=False)
    nodes = set(frozen.source) | set(frozen.target)
    stats_dict = {
        "raw_rows": int(sum(len(x) for x in parts)),
        "sign_conflict_pairs_dropped": int(len(conflicts)),
        "nodes": int(len(nodes)),
        "edges": int(len(frozen)),
        "fraction_positive": float((frozen.sign == 1).mean()),
        "fraction_negative": float((frozen.sign == -1).mean()),
    }
    return frozen, stats_dict


def cosine_similarity_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    denominator = np.outer(norms, norms)
    similarities = matrix @ matrix.T
    np.divide(similarities, denominator, out=similarities, where=denominator > 0)
    similarities[denominator == 0] = 0.0
    np.fill_diagonal(similarities, 1.0)
    return similarities


def jaccard_similarity_rows(matrix: np.ndarray) -> np.ndarray:
    binary = np.asarray(matrix != 0, dtype=np.float64)
    intersection = binary @ binary.T
    degrees = binary.sum(axis=1)
    union = degrees[:, None] + degrees[None, :] - intersection
    out = np.zeros_like(intersection)
    np.divide(intersection, union, out=out, where=union > 0)
    np.fill_diagonal(out, 1.0)
    return out


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return np.nan
    return float(stats.spearmanr(x[mask], y[mask]).statistic)


def safe_cosine(x: np.ndarray, y: np.ndarray) -> float:
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else np.nan


def topology_matrices(
    frozen: pd.DataFrame, perturbations: list[str], shared_genes: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    gene_to_idx = {gene: idx for idx, gene in enumerate(shared_genes)}
    pert_to_idx = {gene: idx for idx, gene in enumerate(perturbations)}
    signed_out = np.zeros((len(perturbations), len(shared_genes)), dtype=np.float32)
    signed_in = np.zeros_like(signed_out)
    for row in frozen.itertuples(index=False):
        src = row.source
        tgt = row.target
        if src in pert_to_idx and tgt in gene_to_idx:
            signed_out[pert_to_idx[src], gene_to_idx[tgt]] = row.sign
        if tgt in pert_to_idx and src in gene_to_idx:
            signed_in[pert_to_idx[tgt], gene_to_idx[src]] = row.sign
    features = {
        "outgoing_signed": signed_out,
        "outgoing_unsigned": np.abs(signed_out),
        "incoming_signed": signed_in,
        "incoming_unsigned": np.abs(signed_in),
        "reversed_signed": signed_in.copy(),
    }
    similarities = {name: cosine_similarity_rows(value) for name, value in features.items()}
    similarities["outgoing_unsigned_jaccard"] = jaccard_similarity_rows(signed_out)
    similarities["incoming_unsigned_jaccard"] = jaccard_similarity_rows(signed_in)
    return features, similarities


def build_response_similarities(
    delta: np.ndarray, perturbations: list[str], shared_genes: list[str]
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    n = len(perturbations)
    gene_to_idx = {gene: idx for idx, gene in enumerate(shared_genes)}
    pearson = np.eye(n, dtype=np.float64)
    spearman = np.eye(n, dtype=np.float64)
    cosine = np.eye(n, dtype=np.float64)
    records = []
    all_mask = np.ones(len(shared_genes), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            mask = all_mask.copy()
            if perturbations[i] in gene_to_idx:
                mask[gene_to_idx[perturbations[i]]] = False
            if perturbations[j] in gene_to_idx:
                mask[gene_to_idx[perturbations[j]]] = False
            x, y = delta[i, mask], delta[j, mask]
            p = safe_pearson(x, y)
            s = safe_spearman(x, y)
            c = safe_cosine(x, y)
            pearson[i, j] = pearson[j, i] = p
            spearman[i, j] = spearman[j, i] = s
            cosine[i, j] = cosine[j, i] = c
            records.append(
                {
                    "perturbation_i": perturbations[i],
                    "perturbation_j": perturbations[j],
                    "response_pearson_trans": p,
                    "response_spearman_trans": s,
                    "response_cosine_trans": c,
                    "n_response_genes": int(mask.sum()),
                }
            )
    return pd.DataFrame(records), {"pearson": pearson, "spearman": spearman, "cosine": cosine}


def upper_values(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(matrix.shape[0], 1)
    return np.asarray(matrix[idx], dtype=np.float64)


def mantel_permutation(
    topology: np.ndarray,
    response: np.ndarray,
    n_perm: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    observed = safe_spearman(upper_values(topology), upper_values(response))
    rng = np.random.default_rng(seed)
    values = np.empty(n_perm, dtype=np.float64)
    for b in range(n_perm):
        order = rng.permutation(topology.shape[0])
        shuffled = topology[np.ix_(order, order)]
        values[b] = safe_spearman(upper_values(shuffled), upper_values(response))
    p_one_sided = (1 + np.sum(values >= observed)) / (n_perm + 1)
    return observed, float(p_one_sided), values


def bootstrap_gene_associations(
    similarities: dict[str, np.ndarray],
    response: np.ndarray,
    topology_names: list[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = response.shape[0]
    records = []
    for b in range(n_boot):
        sampled = rng.integers(0, n, size=n)
        a, c = np.triu_indices(n, 1)
        left, right = sampled[a], sampled[c]
        keep = left != right
        resp = response[left[keep], right[keep]]
        for name in topology_names:
            top = similarities[name][left[keep], right[keep]]
            records.append(
                {"bootstrap": b, "topology": name, "spearman_rho": safe_spearman(top, resp)}
            )
    return pd.DataFrame(records)


def partial_spearman(y: np.ndarray, x: np.ndarray, covariates: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(y)), covariates])
    valid = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(design), axis=1)
    if valid.sum() < design.shape[1] + 3:
        return np.nan
    design = design[valid]
    y_resid = y[valid] - design @ np.linalg.lstsq(design, y[valid], rcond=None)[0]
    x_resid = x[valid] - design @ np.linalg.lstsq(design, x[valid], rcond=None)[0]
    return safe_spearman(x_resid, y_resid)


def directed_edge_swap(
    source: np.ndarray,
    target: np.ndarray,
    sign: np.ndarray,
    n_nodes: int,
    seed: int,
    success_multiplier: int,
    attempt_multiplier: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    src = source.copy()
    dst = target.copy()
    sgn = sign.copy()
    edge_codes = set((src.astype(np.int64) * n_nodes + dst).tolist())
    required = success_multiplier * len(src)
    max_attempts = attempt_multiplier * len(src)
    successes = 0
    attempts = 0
    while successes < required and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, len(src), size=2)
        if i == j:
            continue
        a, b, c, d = int(src[i]), int(dst[i]), int(src[j]), int(dst[j])
        if a == c or b == d or a == d or c == b:
            continue
        old1, old2 = a * n_nodes + b, c * n_nodes + d
        new1, new2 = a * n_nodes + d, c * n_nodes + b
        if new1 == new2 or new1 in edge_codes or new2 in edge_codes:
            continue
        edge_codes.remove(old1)
        edge_codes.remove(old2)
        edge_codes.add(new1)
        edge_codes.add(new2)
        dst[i], dst[j] = d, b
        successes += 1
    return src, dst, sgn, successes, attempts


def shuffled_outgoing_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    sign: np.ndarray,
    node_names: list[str],
    perturbations: list[str],
    shared_genes: list[str],
    signed: bool = True,
) -> np.ndarray:
    name_to_node = {name: idx for idx, name in enumerate(node_names)}
    node_to_pert = {name_to_node[p]: i for i, p in enumerate(perturbations) if p in name_to_node}
    gene_node_to_feature = {
        name_to_node[g]: i for i, g in enumerate(shared_genes) if g in name_to_node
    }
    matrix = np.zeros((len(perturbations), len(shared_genes)), dtype=np.float32)
    for u, v, effect in zip(src, dst, sign):
        ui = node_to_pert.get(int(u))
        vi = gene_node_to_feature.get(int(v))
        if ui is not None and vi is not None:
            matrix[ui, vi] = effect if signed else 1
    return cosine_similarity_rows(matrix)


def prototype_predict(
    delta: np.ndarray,
    similarity: np.ndarray,
    train: np.ndarray,
    test_index: int,
    k: int,
) -> tuple[np.ndarray, int, bool]:
    sims = similarity[test_index, train].copy()
    positive = np.flatnonzero(sims > 0)
    fallback = len(positive) == 0
    if fallback:
        return delta[train].mean(axis=0), 0, True
    order = positive[np.argsort(sims[positive], kind="stable")[::-1][:k]]
    neighbors = train[order]
    weights = sims[order]
    weights /= weights.sum()
    return weights @ delta[neighbors], len(neighbors), False


def prediction_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
    direct_gene: str,
    gene_to_idx: dict[str, int],
) -> dict[str, float]:
    mask = np.ones(len(truth), dtype=bool)
    if direct_gene in gene_to_idx:
        mask[gene_to_idx[direct_gene]] = False
    pred, true = predicted[mask], truth[mask]
    n_top = max(1, math.ceil(0.10 * len(true)))
    true_top = np.argpartition(np.abs(true), -n_top)[-n_top:]
    pred_top = np.argpartition(np.abs(pred), -n_top)[-n_top:]
    recall = len(set(true_top.tolist()) & set(pred_top.tolist())) / n_top
    return {
        "pearson": safe_pearson(pred, true),
        "spearman": safe_spearman(pred, true),
        "mse": float(np.mean((pred - true) ** 2)),
        "cosine": safe_cosine(pred, true),
        "top10_abs_recall": float(recall),
        "n_trans_genes": int(mask.sum()),
    }


def evaluate_prototypes(
    delta: np.ndarray,
    perturbations: list[str],
    shared_genes: list[str],
    similarities: dict[str, np.ndarray],
    folds: np.ndarray,
    k: int,
    names: list[str],
) -> pd.DataFrame:
    records = []
    gene_to_idx = {gene: idx for idx, gene in enumerate(shared_genes)}
    all_indices = np.arange(len(perturbations))
    for name in names:
        sim = similarities[name]
        for test_idx in all_indices:
            fold = int(folds[test_idx])
            train = all_indices[folds != fold]
            prediction, n_neighbors, fallback = prototype_predict(delta, sim, train, test_idx, k)
            metric = prediction_metrics(prediction, delta[test_idx], perturbations[test_idx], gene_to_idx)
            records.append(
                {
                    "topology": name,
                    "fold": fold + 1,
                    "perturbation": perturbations[test_idx],
                    "n_neighbors": n_neighbors,
                    "fallback_training_mean": fallback,
                    **metric,
                }
            )
    return pd.DataFrame(records)


def summarize_folds(per_pert: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (name, fold), group in per_pert.groupby(["topology", "fold"], sort=False):
        pearson = np.clip(group.pearson.to_numpy(float), -0.999999, 0.999999)
        fisher = np.arctanh(pearson)
        records.append(
            {
                "topology": name,
                "fold": int(fold),
                "n_perturbations": len(group),
                "fisher_z_mean": float(np.nanmean(fisher)),
                "pearson_fisher_mean": float(np.tanh(np.nanmean(fisher))),
                "pearson_median": float(np.nanmedian(group.pearson)),
                "spearman_mean": float(np.nanmean(group.spearman)),
                "mse_mean": float(np.nanmean(group.mse)),
                "cosine_mean": float(np.nanmean(group.cosine)),
                "top10_abs_recall_mean": float(np.nanmean(group.top10_abs_recall)),
                "fallback_fraction": float(np.mean(group.fallback_training_mean)),
            }
        )
    return pd.DataFrame(records)


def summarize_gate_b(per_pert: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for name, group in per_pert.groupby("topology", sort=False):
        pearson = np.clip(group.pearson.to_numpy(float), -0.999999, 0.999999)
        fisher = np.arctanh(pearson)
        boot = np.empty(n_boot)
        for b in range(n_boot):
            sample = rng.integers(0, len(group), size=len(group))
            boot[b] = np.nanmean(fisher[sample])
        records.append(
            {
                "topology": name,
                "n_perturbations": len(group),
                "fisher_z_mean": float(np.nanmean(fisher)),
                "fisher_z_ci_low": float(np.nanpercentile(boot, 2.5)),
                "fisher_z_ci_high": float(np.nanpercentile(boot, 97.5)),
                "pearson_fisher_mean": float(np.tanh(np.nanmean(fisher))),
                "pearson_median": float(np.nanmedian(group.pearson)),
                "spearman_mean": float(np.nanmean(group.spearman)),
                "mse_mean": float(np.nanmean(group.mse)),
                "cosine_mean": float(np.nanmean(group.cosine)),
                "top10_abs_recall_mean": float(np.nanmean(group.top10_abs_recall)),
            }
        )
    return pd.DataFrame(records)


def save_placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if md5(ADATA_PATH) != config["dataset"]["expected_md5"]:
        raise RuntimeError("Dataset MD5 does not match preregistration; aborting before analysis")

    log("Freezing external directed signed network")
    frozen, network_stats = freeze_network()
    graph_nodes = set(frozen.source) | set(frozen.target)

    log("Reading K562 metadata and applying objective eligibility rules")
    adata = ad.read_h5ad(ADATA_PATH, backed="r")
    measured_genes = adata.var_names.astype(str).tolist()
    measured_set = set(measured_genes)
    shared_genes = sorted(measured_set & graph_nodes)
    cell_counts = adata.obs["perturbation"].astype(str).value_counts().sort_index()
    labels = cell_counts.index.tolist()
    out_usable = (
        frozen[frozen.target.isin(shared_genes)]
        .groupby("source")["target"]
        .nunique()
        .to_dict()
    )
    out_degree = frozen.groupby("source")["target"].nunique().to_dict()
    in_degree = frozen.groupby("target")["source"].nunique().to_dict()
    eligible = []
    qc_records = []
    for label in labels:
        count = int(cell_counts[label])
        is_control = label == config["dataset"]["control_label"]
        is_combo = "+" in label
        reasons = []
        if not is_control:
            if is_combo:
                reasons.append("combinatorial_perturbation")
            if count < config["dataset"]["min_cells_per_perturbation"]:
                reasons.append("fewer_than_50_cells")
            if int(out_usable.get(label, 0)) < config["coverage"]["min_usable_outgoing_signed_edges"]:
                reasons.append("fewer_than_5_usable_outgoing_signed_edges")
            if not reasons:
                eligible.append(label)
        qc_records.append(
            {
                "record_type": "label",
                "perturbation": label,
                "cells": count,
                "is_control": is_control,
                "is_combinatorial": is_combo,
                "in_frozen_graph": label in graph_nodes,
                "usable_outgoing_degree": int(out_usable.get(label, 0)),
                "status": "control" if is_control else ("retained" if not reasons else "removed"),
                "removal_reason": ";".join(reasons),
                "dataset_total_cells": int(adata.n_obs),
                "dataset_total_perturbations": int(len(labels) - int(config["dataset"]["control_label"] in labels)),
                "dataset_control_cells": int(cell_counts.get(config["dataset"]["control_label"], 0)),
                "measured_genes": int(adata.n_vars),
                "shared_genes_retained": int(len(shared_genes)),
            }
        )
    eligible = sorted(eligible)
    pd.DataFrame(qc_records).to_csv(OUT / "data_qc.csv", index=False)

    total_perts = [x for x in labels if x != config["dataset"]["control_label"] and "+" not in x]
    perts_ge_cells = [x for x in total_perts if cell_counts[x] >= config["dataset"]["min_cells_per_perturbation"]]
    coverage_records = [
        {"metric": "total_single_gene_perturbations", "value": len(total_perts)},
        {"metric": "perturbations_with_graph_node", "value": sum(x in graph_nodes for x in total_perts)},
        {
            "metric": "perturbations_with_at_least_5_usable_outgoing_edges",
            "value": sum(out_usable.get(x, 0) >= 5 for x in total_perts),
        },
        {"metric": "perturbations_with_at_least_50_cells", "value": len(perts_ge_cells)},
        {"metric": "perturbations_used_in_final_audit", "value": len(eligible)},
        {"metric": "warning_below_100_eligible", "value": bool(len(eligible) < 100)},
    ]
    pd.DataFrame(coverage_records).to_csv(OUT / "graph_coverage.csv", index=False)
    if len(eligible) == 0:
        raise RuntimeError("No eligible perturbations under preregistered thresholds")

    log(f"Aggregating trans-response pseudobulks for {len(eligible)} eligible perturbations")
    obs_labels = adata.obs["perturbation"].astype(str).to_numpy()
    wanted = [config["dataset"]["control_label"], *eligible]
    means = {}
    for idx, label in enumerate(wanted):
        rows = np.flatnonzero(obs_labels == label)
        expression = adata[rows, :].X
        means[label] = np.asarray(expression.mean(axis=0)).ravel().astype(np.float32)
        if idx % 20 == 0:
            log(f"Pseudobulk {idx + 1}/{len(wanted)}")
    adata.file.close()
    shared_positions = np.array([measured_genes.index(gene) for gene in shared_genes], dtype=int)
    control_mean = means[config["dataset"]["control_label"]][shared_positions]
    delta = np.vstack([means[p][shared_positions] - control_mean for p in eligible]).astype(np.float32)
    np.savez_compressed(
        OUT / "perturbation_response_matrix.npz",
        delta=delta,
        perturbations=np.asarray(eligible),
        genes=np.asarray(shared_genes),
        control_mean=control_mean,
    )

    metadata = pd.DataFrame(
        {
            "perturbation": eligible,
            "cells": [int(cell_counts[p]) for p in eligible],
            "outgoing_degree": [int(out_degree.get(p, 0)) for p in eligible],
            "usable_outgoing_degree": [int(out_usable.get(p, 0)) for p in eligible],
            "incoming_degree": [int(in_degree.get(p, 0)) for p in eligible],
            "perturbation_measured": [p in measured_set for p in eligible],
        }
    )
    metadata.to_csv(OUT / "perturbation_metadata.csv", index=False)

    log("Constructing frozen topology representations and trans-only pairwise response metrics")
    features, similarities = topology_matrices(frozen, eligible, shared_genes)
    pairwise, response_matrices = build_response_similarities(delta, eligible, shared_genes)
    tri = np.triu_indices(len(eligible), 1)
    for name, matrix in similarities.items():
        pairwise[f"topology_{name}"] = matrix[tri]
    pairwise.to_csv(OUT / "gateA_pairwise_metrics.csv", index=False)

    primary_names = [
        "outgoing_signed",
        "outgoing_unsigned",
        "incoming_signed",
        "incoming_unsigned",
        "reversed_signed",
    ]
    gate_a_records = []
    permutation_records = []
    for offset, name in enumerate(primary_names):
        observed, p_value, null = mantel_permutation(
            similarities[name], response_matrices["pearson"], config["gate_a"]["label_permutations"], 17 + offset
        )
        gate_a_records.append(
            {
                "topology": name,
                "topology_metric": "cosine",
                "response_metric": "trans_pearson",
                "spearman_rho": observed,
                "mantel_permutation_p_one_sided": p_value,
            }
        )
        for b, value in enumerate(null):
            permutation_records.append({"topology": name, "permutation": b, "rho": value})
        for response_name in ("spearman", "cosine"):
            gate_a_records.append(
                {
                    "topology": name,
                    "topology_metric": "cosine",
                    "response_metric": f"trans_{response_name}",
                    "spearman_rho": safe_spearman(
                        upper_values(similarities[name]), upper_values(response_matrices[response_name])
                    ),
                    "mantel_permutation_p_one_sided": np.nan,
                }
            )
    for name in ("outgoing_unsigned_jaccard", "incoming_unsigned_jaccard"):
        gate_a_records.append(
            {
                "topology": name,
                "topology_metric": "jaccard",
                "response_metric": "trans_pearson",
                "spearman_rho": safe_spearman(
                    upper_values(similarities[name]), upper_values(response_matrices["pearson"])
                ),
                "mantel_permutation_p_one_sided": np.nan,
            }
        )

    log("Running perturbation-level Gate A bootstrap")
    bootstrap = bootstrap_gene_associations(
        similarities,
        response_matrices["pearson"],
        primary_names,
        config["gate_a"]["bootstrap_replicates"],
        29,
    )
    bootstrap.to_csv(OUT / "gateA_bootstrap.csv", index=False)
    boot_summary = bootstrap.groupby("topology").spearman_rho.agg(
        bootstrap_median="median",
        bootstrap_ci_low=lambda x: np.nanpercentile(x, 2.5),
        bootstrap_ci_high=lambda x: np.nanpercentile(x, 97.5),
    )
    gate_a = pd.DataFrame(gate_a_records).merge(
        boot_summary, how="left", left_on="topology", right_index=True
    )

    log("Running degree and response-strength confound audits")
    pair_i, pair_j = tri
    degrees_out = metadata.outgoing_degree.to_numpy(float)
    degrees_in = metadata.incoming_degree.to_numpy(float)
    usable_degrees = metadata.usable_outgoing_degree.to_numpy(float)
    shared_index = {gene: idx for idx, gene in enumerate(shared_genes)}
    pert_control_expr = np.array(
        [control_mean[shared_index[p]] if p in shared_index else np.nan for p in eligible], dtype=float
    )
    if np.isnan(pert_control_expr).any():
        pert_control_expr[np.isnan(pert_control_expr)] = np.nanmedian(pert_control_expr)
    covariates = np.column_stack(
        [
            np.abs(np.log1p(usable_degrees[pair_i]) - np.log1p(usable_degrees[pair_j])),
            np.log1p(usable_degrees[pair_i] + usable_degrees[pair_j]),
            0.5 * (pert_control_expr[pair_i] + pert_control_expr[pair_j]),
        ]
    )
    degree_audit = []
    response_pair = response_matrices["pearson"][tri]
    for name in primary_names:
        topology_pair = similarities[name][tri]
        degree_audit.append(
            {
                "topology": name,
                "raw_spearman_rho": safe_spearman(topology_pair, response_pair),
                "partial_spearman_rho": partial_spearman(response_pair, topology_pair, covariates),
                "covariates": "abs_log1p_usable_outdegree_diff;log1p_pair_usable_outdegree_sum;mean_control_expression",
            }
        )
    pd.DataFrame(degree_audit).to_csv(OUT / "degree_confound_audit.csv", index=False)

    response_strength = pd.DataFrame(
        {
            "perturbation": eligible,
            "outgoing_degree": degrees_out.astype(int),
            "usable_outgoing_degree": usable_degrees.astype(int),
            "incoming_degree": degrees_in.astype(int),
            "l2_norm": np.linalg.norm(delta, axis=1),
            "strong_response_genes_abs_delta_ge_0_25": (np.abs(delta) >= 0.25).sum(axis=1),
            "mean_absolute_delta": np.mean(np.abs(delta), axis=1),
            "control_expression_of_perturbed_gene": pert_control_expr,
        }
    )
    for measure in ("l2_norm", "strong_response_genes_abs_delta_ge_0_25", "mean_absolute_delta"):
        response_strength[f"global_spearman_{measure}_vs_usable_outdegree"] = safe_spearman(
            response_strength[measure], response_strength["usable_outgoing_degree"]
        )
    response_strength.to_csv(OUT / "response_strength_audit.csv", index=False)

    log("Generating 100 directed degree-preserving signed shuffled-network nulls")
    node_names = sorted(graph_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(node_names)}
    src = frozen.source.map(node_to_idx).to_numpy(np.int32)
    dst = frozen.target.map(node_to_idx).to_numpy(np.int32)
    signs = frozen.sign.to_numpy(np.int8)
    shuffle_records = []
    shuffled_sims = np.empty((config["topology"]["n_shuffled_networks"], len(eligible), len(eligible)), dtype=np.float32)
    seed_basis = config["random_seeds"]
    for shuffle_idx in range(config["topology"]["n_shuffled_networks"]):
        seed = seed_basis[shuffle_idx % len(seed_basis)] * 10000 + shuffle_idx
        ss, tt, sg, successes, attempts = directed_edge_swap(
            src,
            dst,
            signs,
            len(node_names),
            seed,
            config["topology"]["successful_swaps_per_shuffle_multiplier"],
            config["topology"]["swap_attempt_multiplier"],
        )
        sim = shuffled_outgoing_similarity(ss, tt, sg, node_names, eligible, shared_genes, signed=True)
        shuffled_sims[shuffle_idx] = sim.astype(np.float32)
        rho = safe_spearman(upper_values(sim), upper_values(response_matrices["pearson"]))
        shuffle_records.append(
            {
                "shuffle": shuffle_idx,
                "seed": seed,
                "successful_swaps": successes,
                "attempts": attempts,
                "required_swaps": config["topology"]["successful_swaps_per_shuffle_multiplier"] * len(src),
                "complete": successes
                >= config["topology"]["successful_swaps_per_shuffle_multiplier"] * len(src),
                "spearman_rho": rho,
            }
        )
        if (shuffle_idx + 1) % 10 == 0:
            log(f"Shuffled networks {shuffle_idx + 1}/100")
    shuffle_null = pd.DataFrame(shuffle_records)
    shuffle_null.to_csv(OUT / "gateA_shuffle_null.csv", index=False)
    np.savez_compressed(OUT / "shuffled_topology_similarities.npz", similarities=shuffled_sims)

    shuffled_summary = {
        "topology": "shuffled_null",
        "topology_metric": "signed_cosine",
        "response_metric": "trans_pearson",
        "spearman_rho": float(shuffle_null.spearman_rho.median()),
        "mantel_permutation_p_one_sided": np.nan,
        "bootstrap_median": np.nan,
        "bootstrap_ci_low": float(np.percentile(shuffle_null.spearman_rho, 2.5)),
        "bootstrap_ci_high": float(np.percentile(shuffle_null.spearman_rho, 97.5)),
    }
    gate_a = pd.concat([gate_a, pd.DataFrame([shuffled_summary])], ignore_index=True)
    gate_a.to_csv(OUT / "gateA_summary.csv", index=False)
    pd.DataFrame(permutation_records).to_csv(OUT / "gateA_label_permutation_null.csv", index=False)

    primary_gate_a = gate_a[(gate_a.topology == "outgoing_signed") & (gate_a.response_metric == "trans_pearson")].iloc[0]
    shuffle_p95 = float(np.percentile(shuffle_null.spearman_rho, 95))
    gate_a_pass = bool(
        primary_gate_a.spearman_rho > shuffle_p95 and primary_gate_a.bootstrap_ci_low > 0
    )
    observed = {
        row.topology: float(row.spearman_rho)
        for row in gate_a.itertuples()
        if row.response_metric == "trans_pearson"
    }
    boot_pivot = bootstrap.pivot(index="bootstrap", columns="topology", values="spearman_rho")
    sign_support = bool(
        observed["outgoing_signed"] > observed["outgoing_unsigned"]
        and np.mean(boot_pivot.outgoing_signed > boot_pivot.outgoing_unsigned) > 0.5
    )
    direction_support = bool(
        observed["outgoing_signed"] > observed["incoming_signed"]
        and observed["outgoing_signed"] > observed["reversed_signed"]
        and np.mean(boot_pivot.outgoing_signed > boot_pivot.incoming_signed) > 0.5
    )

    # Gate A figures are generated before any Gate B decision.
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x = pairwise["topology_outgoing_signed"].to_numpy(float)
    y = pairwise["response_pearson_trans"].to_numpy(float)
    ax.hexbin(x, y, gridsize=30, mincnt=1, cmap="viridis", linewidths=0)
    bins = pd.qcut(x, q=min(10, len(np.unique(x))), duplicates="drop")
    trend = pd.DataFrame({"x": x, "y": y, "bin": bins}).groupby("bin", observed=True).mean()
    ax.plot(trend.x, trend.y, color="#d62728", marker="o", lw=2, label="Decile mean")
    ax.set(xlabel="Signed outgoing topology cosine", ylabel="Trans-only response Pearson", title="Topology–response relationship")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "plot1_topology_vs_response.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    plot_names = primary_names
    values = [observed[n] for n in plot_names]
    lows = [float(boot_summary.loc[n, "bootstrap_ci_low"]) for n in plot_names]
    highs = [float(boot_summary.loc[n, "bootstrap_ci_high"]) for n in plot_names]
    positions = np.arange(len(plot_names))
    ax.bar(positions, values, color=["#2166ac", "#67a9cf", "#b2182b", "#ef8a62", "#999999"])
    ax.errorbar(positions, values, yerr=[np.array(values) - lows, np.array(highs) - values], fmt="none", color="black", capsize=4)
    ax.axhspan(np.percentile(shuffle_null.spearman_rho, 2.5), np.percentile(shuffle_null.spearman_rho, 97.5), color="gray", alpha=0.2, label="Shuffled 95% interval")
    ax.set_xticks(positions, [x.replace("_", "\n") for x in plot_names])
    ax.set_ylabel("Pairwise Spearman association")
    ax.set_title("Gate A topology effect sizes")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "plot2_gateA_effect_sizes.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(shuffle_null.spearman_rho, bins=20, color="#bdbdbd", edgecolor="white")
    ax.axvline(observed["outgoing_signed"], color="#2166ac", lw=2.5, label="Correct signed outgoing")
    ax.axvline(shuffle_p95, color="#d62728", ls="--", label="Shuffled 95th percentile")
    ax.set(xlabel="Gate A Spearman association", ylabel="Shuffled networks", title="Correct graph versus degree-preserving null")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "plot4_correct_vs_shuffled_null.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(metadata.usable_outgoing_degree, bins=20, color="#4daf4a", edgecolor="white")
    axes[0].axvline(5, color="black", ls="--", label="Eligibility threshold")
    axes[0].set(xlabel="Usable outgoing degree", ylabel="Eligible perturbations", title=f"Eligible n={len(eligible)}")
    axes[0].legend(frameon=True)
    cov_vals = [len(total_perts), sum(x in graph_nodes for x in total_perts), len(eligible)]
    axes[1].bar(["All single", "Graph node", "Final audit"], cov_vals, color=["#bdbdbd", "#80b1d3", "#4daf4a"])
    axes[1].set(ylabel="Perturbations", title="Graph coverage funnel")
    fig.tight_layout()
    fig.savefig(OUT / "plot5_graph_coverage_degree.png", dpi=220)
    plt.close(fig)

    folds = np.empty(len(eligible), dtype=int)
    order = np.random.default_rng(17).permutation(len(eligible))
    folds[order] = np.arange(len(eligible)) % config["gate_b"]["folds"]
    gate_b_pass = False
    fold_wins = 0
    chosen_outgoing = None
    gate_b_ran = False

    if not gate_a_pass:
        log("Gate A failed the frozen minimum rule; stopping before Gate B")
        pd.DataFrame(
            columns=["topology", "fold", "perturbation", "pearson", "spearman", "mse", "cosine", "top10_abs_recall", "skip_reason"]
        ).assign(skip_reason="Gate A failed; Gate B not run per protocol").to_csv(OUT / "gateB_per_perturbation.csv", index=False)
        pd.DataFrame(columns=["topology", "fold", "fisher_z_mean", "mse_mean", "skip_reason"]).assign(
            skip_reason="Gate A failed; Gate B not run per protocol"
        ).to_csv(OUT / "gateB_per_fold.csv", index=False)
        pd.DataFrame([{"status": "NOT_RUN", "reason": "Gate A failed; Gate B not run per protocol"}]).to_csv(
            OUT / "gateB_summary.csv", index=False
        )
        pd.DataFrame(columns=["shuffle", "fold", "fisher_z_mean", "mse_mean", "top10_abs_recall_mean", "skip_reason"]).assign(
            skip_reason="Gate A failed; Gate B not run per protocol"
        ).to_csv(OUT / "gateB_shuffle_null.csv", index=False)
        save_placeholder(OUT / "plot3_prototype_transfer.png", "Gate B not run", "Gate A failed the frozen minimum support rule; the protocol requires stopping before transfer testing.")
        save_placeholder(OUT / "plot6_per_perturbation_prototype.png", "Per-perturbation transfer not available", "No prototype predictions were computed after the Gate A stop decision.")
    else:
        gate_b_ran = True
        log("Gate A passed; running fixed five-fold non-neural prototype transfer")
        prototype_names = ["outgoing_signed", "outgoing_unsigned", "incoming_signed", "reversed_signed"]
        per_pert = evaluate_prototypes(delta, eligible, shared_genes, similarities, folds, config["gate_b"]["k_neighbors"], prototype_names)
        per_pert.to_csv(OUT / "gateB_per_perturbation.csv", index=False)
        per_fold = summarize_folds(per_pert)
        per_fold.to_csv(OUT / "gateB_per_fold.csv", index=False)
        gate_b_summary = summarize_gate_b(per_pert, config["gate_b"]["bootstrap_replicates"], 43)

        shuffle_b_records = []
        for shuffle_idx, sim in enumerate(shuffled_sims):
            shuffled_per_pert = evaluate_prototypes(
                delta,
                eligible,
                shared_genes,
                {"shuffled_signed": sim},
                folds,
                config["gate_b"]["k_neighbors"],
                ["shuffled_signed"],
            )
            for row in summarize_folds(shuffled_per_pert).itertuples(index=False):
                shuffle_b_records.append(
                    {
                        "shuffle": shuffle_idx,
                        "fold": row.fold,
                        "fisher_z_mean": row.fisher_z_mean,
                        "pearson_fisher_mean": row.pearson_fisher_mean,
                        "mse_mean": row.mse_mean,
                        "top10_abs_recall_mean": row.top10_abs_recall_mean,
                    }
                )
        shuffle_b = pd.DataFrame(shuffle_b_records)
        shuffle_b.to_csv(OUT / "gateB_shuffle_null.csv", index=False)
        shuffle_overall = shuffle_b.groupby("shuffle").agg(
            fisher_z_mean=("fisher_z_mean", "mean"),
            mse_mean=("mse_mean", "mean"),
            top10_abs_recall_mean=("top10_abs_recall_mean", "mean"),
        )
        gate_b_summary = pd.concat(
            [
                gate_b_summary,
                pd.DataFrame(
                    [
                        {
                            "topology": "shuffled_null_median",
                            "n_perturbations": len(eligible),
                            "fisher_z_mean": float(shuffle_overall.fisher_z_mean.median()),
                            "fisher_z_ci_low": float(np.percentile(shuffle_overall.fisher_z_mean, 2.5)),
                            "fisher_z_ci_high": float(np.percentile(shuffle_overall.fisher_z_mean, 97.5)),
                            "pearson_fisher_mean": float(np.tanh(shuffle_overall.fisher_z_mean.median())),
                            "pearson_median": np.nan,
                            "spearman_mean": np.nan,
                            "mse_mean": float(shuffle_overall.mse_mean.median()),
                            "cosine_mean": np.nan,
                            "top10_abs_recall_mean": float(shuffle_overall.top10_abs_recall_mean.median()),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        gate_b_summary.to_csv(OUT / "gateB_summary.csv", index=False)

        fold_null = shuffle_b.groupby("fold").fisher_z_mean.median()
        candidate_wins = {}
        for name in ("outgoing_signed", "outgoing_unsigned"):
            correct = per_fold[per_fold.topology == name].set_index("fold").fisher_z_mean
            candidate_wins[name] = int(sum(correct.loc[f] > fold_null.loc[f] for f in range(1, 6)))
        chosen_outgoing = max(candidate_wins, key=lambda x: (candidate_wins[x], float(gate_b_summary.set_index("topology").loc[x, "fisher_z_mean"])))
        fold_wins = candidate_wins[chosen_outgoing]
        chosen_mse = float(gate_b_summary.set_index("topology").loc[chosen_outgoing, "mse_mean"])
        null_mse = float(shuffle_overall.mse_mean.median())
        gate_b_pass = bool(fold_wins >= 4 and chosen_mse <= 1.10 * null_mse)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        summary_plot = gate_b_summary[gate_b_summary.topology != "shuffled_null_median"]
        ax.bar(summary_plot.topology.str.replace("_", "\n"), summary_plot.pearson_fisher_mean, color=["#2166ac", "#67a9cf", "#b2182b", "#999999"])
        ax.axhline(float(np.tanh(shuffle_overall.fisher_z_mean.median())), color="black", ls="--", label="Shuffled median")
        ax.set(ylabel="Fisher-z aggregated Pearson", title="Five-fold prototype transfer")
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(OUT / "plot3_prototype_transfer.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for name, color in (("outgoing_signed", "#2166ac"), ("outgoing_unsigned", "#67a9cf"), ("incoming_signed", "#b2182b")):
            group = per_pert[per_pert.topology == name]
            ax.scatter(group.perturbation, group.pearson, s=18, alpha=0.75, label=name, color=color)
        ax.axhline(0, color="black", lw=0.8)
        ax.set(xlabel="Held-out perturbation", ylabel="Trans-only Pearson", title="Per-perturbation prototype performance")
        ax.tick_params(axis="x", labelrotation=90, labelsize=5)
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(OUT / "plot6_per_perturbation_prototype.png", dpi=220)
        plt.close(fig)

        # Replace null plot with a two-panel Gate A/Gate B version.
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        axes[0].hist(shuffle_null.spearman_rho, bins=20, color="#bdbdbd", edgecolor="white")
        axes[0].axvline(observed["outgoing_signed"], color="#2166ac", lw=2.5)
        axes[0].set(xlabel="Gate A Spearman", ylabel="Shuffled networks", title="Gate A null")
        axes[1].hist(shuffle_overall.fisher_z_mean, bins=20, color="#bdbdbd", edgecolor="white")
        correct_z = float(gate_b_summary.set_index("topology").loc[chosen_outgoing, "fisher_z_mean"])
        axes[1].axvline(correct_z, color="#2166ac", lw=2.5)
        axes[1].set(xlabel="Gate B Fisher-z mean", ylabel="Shuffled networks", title="Gate B null")
        fig.tight_layout()
        fig.savefig(OUT / "plot4_correct_vs_shuffled_null.png", dpi=220)
        plt.close(fig)

    if gate_a_pass and gate_b_pass:
        verdict = "PHASE1_REAL_DIRECTIONAL_SIGNAL_SUPPORTED"
    elif gate_a_pass:
        verdict = "PHASE1_DESCRIPTIVE_SIGNAL_ONLY"
    else:
        verdict = "PHASE1_DIRECTIONAL_SIGNAL_NOT_SUPPORTED"

    network_report = {
        **network_stats,
        "download_date": config["external_network"]["download_date"],
        "source_database": "OmniPath core + CollecTRI via OmniPath web service",
    }
    with (OUT / "network_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(network_report, handle, indent=2)
    with (OUT / "software_versions.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "anndata": ad.__version__,
                "dataset_md5": md5(ADATA_PATH),
                "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
            },
            handle,
            indent=2,
        )

    gate_b_sentence = (
        f"Gate B {'passed' if gate_b_pass else 'failed'}: {chosen_outgoing} beat the fold-specific shuffled median in {fold_wins}/5 folds."
        if gate_b_ran
        else "Gate B was not run because Gate A failed the frozen minimum support rule."
    )
    verdict_text = f"""{verdict}

## Dataset and sample sizes

- Replogle et al. K562 CRISPRi Perturb-seq (processed scPertEval build): {int(sum(cell_counts)):,} cells, {len(total_perts):,} single-gene perturbation labels, {int(cell_counts.get('control', 0)):,} controls, and {len(measured_genes):,} measured genes.
- The fixed response/network intersection contained {len(shared_genes):,} genes.
- {len(eligible)} perturbations met both frozen criteria (at least 50 cells and at least 5 usable outgoing signed edges). This is below 100 and is a prominent coverage limitation.

## External graph and coverage

- Frozen external graph: OmniPath core plus CollecTRI, downloaded 2026-08-10, with {network_stats['nodes']:,} nodes and {network_stats['edges']:,} directed signed edges after dropping {network_stats['sign_conflict_pairs_dropped']:,} sign-conflict pairs.
- Positive-edge fraction: {network_stats['fraction_positive']:.3f}; negative-edge fraction: {network_stats['fraction_negative']:.3f}.

## Gate A

- Signed outgoing topology-response Spearman association: {observed['outgoing_signed']:.4f}.
- Degree-preserving shuffled null median: {float(shuffle_null.spearman_rho.median()):.4f}; 95th percentile: {shuffle_p95:.4f}.
- Bootstrap 95% interval for signed outgoing: [{float(primary_gate_a.bootstrap_ci_low):.4f}, {float(primary_gate_a.bootstrap_ci_high):.4f}].
- Gate A {'PASSED' if gate_a_pass else 'FAILED'} the frozen minimum rule.

## Sign and direction comparisons

- Signed outgoing: {observed['outgoing_signed']:.4f}; unsigned outgoing: {observed['outgoing_unsigned']:.4f}. Sign claim: {'SUPPORTED' if sign_support else 'NOT SUPPORTED'}.
- Incoming signed: {observed['incoming_signed']:.4f}; reversed signed: {observed['reversed_signed']:.4f}. Direction claim: {'SUPPORTED' if direction_support else 'NOT SUPPORTED'}.
- Incoming and reversed are algebraically identical under the preregistered one-step representation, so their equality is expected rather than independent replication.

## Gate B and fold consistency

- {gate_b_sentence}

## Major caveats

- Only {len(eligible)} perturbations were eligible, below the preregistered warning level of 100.
- The processed public file retains target-gene labels but not guide identifiers, so guide-level concordance could not be audited.
- The graph is activation-skewed and context-agnostic; external curation does not guarantee K562 activity.
- Pseudobulk deltas use the provided normalized log1p expression and a shared global control mean; batch covariates are unavailable in the trimmed file.

## Phase II decision

{'Phase II Transformer ablation is scientifically justified by the frozen Phase I gates, but must not begin without explicit instruction.' if verdict == 'PHASE1_REAL_DIRECTIONAL_SIGNAL_SUPPORTED' else 'Phase II Transformer testing is not scientifically justified by this Phase I result.'}
"""
    (OUT / "PHASE1_VERDICT.md").write_text(verdict_text, encoding="utf-8")
    log(f"Completed: {verdict}")


if __name__ == "__main__":
    main()
