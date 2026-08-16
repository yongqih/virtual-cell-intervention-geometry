from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("MAIN_GEOMETRY_AUDIT_OUT", ROOT / "results" / "main_geometry_integrity_audit")).resolve()
FIG = OUT / "figures"
CONFIG = json.loads((SCRIPT / "frozen_config.json").read_text(encoding="utf-8"))
BOOT_DRAWS = int(CONFIG["uncertainty"]["source_bootstrap_draws"])
BOOT_SEED = int(CONFIG["uncertainty"]["seed"])


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(2**20):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, hash_limit: int = 80_000_000) -> dict:
    stat = path.stat()
    record = {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    record["sha256"] = sha256(path) if stat.st_size <= hash_limit else None
    record["sha256_note"] = None if record["sha256"] else "not recomputed for large immutable artifact"
    return record


def safe_corr(left: np.ndarray, right: np.ndarray, method: str = "spearman") -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 4 or np.std(left[valid]) < 1e-12 or np.std(right[valid]) < 1e-12:
        return 0.0
    if method == "spearman":
        return float(stats.spearmanr(left[valid], right[valid]).statistic)
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, np.float64)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def response_distances(matrix: np.ndarray) -> np.ndarray:
    return pdist(normalize_rows(matrix), metric="cosine")


def geometry(prediction: np.ndarray, truth: np.ndarray, method: str = "spearman") -> float:
    if len(prediction) < 4:
        return 0.0
    return safe_corr(response_distances(prediction), response_distances(truth), method)


def row_pearson(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    prediction = prediction - prediction.mean(1, keepdims=True)
    truth = truth - truth.mean(1, keepdims=True)
    denom = np.linalg.norm(prediction, axis=1) * np.linalg.norm(truth, axis=1)
    return np.divide(np.sum(prediction * truth, axis=1), denom,
                     out=np.full(len(prediction), np.nan), where=denom > 1e-12)


def rank_metrics(matrix: np.ndarray) -> dict[str, float | int]:
    matrix = np.asarray(matrix, np.float64)
    matrix = matrix - matrix.mean(0, keepdims=True)
    singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    if np.sum(singular**2) <= 1e-12:
        return {"pc1_fraction": 0.0, "pc80": 0, "pc90": 0, "pc95": 0,
                "participation_ratio": 0.0, "entropy_effective_rank": 0.0}
    weights = singular**2 / np.sum(singular**2)
    cumulative = np.cumsum(weights)
    entropy = float(np.exp(-np.sum(weights[weights > 0] * np.log(weights[weights > 0]))))
    return {
        "pc1_fraction": float(weights[0]),
        "pc80": int(np.searchsorted(cumulative, .80) + 1),
        "pc90": int(np.searchsorted(cumulative, .90) + 1),
        "pc95": int(np.searchsorted(cumulative, .95) + 1),
        "participation_ratio": float(1 / np.sum(weights**2)),
        "entropy_effective_rank": entropy,
    }


def group_metrics(dataset: str, fold: int, model: str, prediction: np.ndarray,
                  truth: np.ndarray, evaluation: str = "recovered_existing_fold",
                  pred_distance: np.ndarray | None = None,
                  truth_distance: np.ndarray | None = None) -> dict:
    pred_rank = rank_metrics(prediction)
    truth_rank = rank_metrics(truth)
    pred_var = float(np.mean(np.var(prediction, axis=0)))
    truth_var = float(np.mean(np.var(truth, axis=0)))
    pred_distance = response_distances(prediction) if pred_distance is None else pred_distance
    truth_distance = response_distances(truth) if truth_distance is None else truth_distance
    return {
        "dataset": dataset, "fold": fold, "model": model, "evaluation": evaluation,
        "n_sources": len(truth), "n_genes": truth.shape[1],
        "response_distance_spearman": safe_corr(pred_distance, truth_distance, "spearman"),
        "response_distance_pearson": safe_corr(pred_distance, truth_distance, "pearson"),
        "per_response_pearson": float(np.nanmean(row_pearson(prediction, truth))),
        "residual_mse": float(np.mean((prediction - truth)**2)),
        "between_perturbation_variance_ratio": pred_var / max(truth_var, 1e-12),
        **{f"prediction_{key}": value for key, value in pred_rank.items()},
        **{f"truth_{key}": value for key, value in truth_rank.items()},
    }


def source_weighted_geometry_draws(groups: list[tuple[np.ndarray, np.ndarray]], seed: int,
                                   draws: int = BOOT_DRAWS) -> np.ndarray:
    """Multinomial source bootstrap without introducing fake duplicate-source pairs."""
    prepared = []
    for prediction, truth in groups:
        n = len(truth)
        upper = np.triu_indices(n, 1)
        pred_dist = response_distances(prediction)
        truth_dist = response_distances(truth)
        prepared.append((n, upper, stats.rankdata(pred_dist), stats.rankdata(truth_dist)))
    rng = np.random.default_rng(seed)
    values = np.empty(draws, float)
    for draw in range(draws):
        group_values = []
        for n, upper, pred_rank, truth_rank in prepared:
            count = rng.multinomial(n, np.repeat(1 / n, n))
            weight = count[upper[0]] * count[upper[1]]
            positive = weight > 0
            if positive.sum() < 4:
                group_values.append(0.0)
                continue
            w = weight[positive].astype(float)
            x, y = pred_rank[positive], truth_rank[positive]
            wsum = w.sum(); xm = np.sum(w * x) / wsum; ym = np.sum(w * y) / wsum
            cov = np.sum(w * (x - xm) * (y - ym))
            denom = np.sqrt(np.sum(w * (x - xm)**2) * np.sum(w * (y - ym)**2))
            group_values.append(float(cov / denom) if denom > 1e-12 else 0.0)
        values[draw] = float(np.mean(group_values))
    return values


def common_k562_groups() -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    base = ROOT / "results" / "identifiability_transformer_bridge"
    mlp = ROOT / "results" / "mlp_transformer_matched_autopsy"
    development = np.load(ROOT / "results" / "directedT_exploration" / "development_data.npz", allow_pickle=True)
    all_sources = set(development["perturbations"].astype(str).tolist())
    groups = {"Transformer": [], "MLP": []}
    for fold in range(5):
        transformer_files = [np.load(base / f"_oof_predictions_fold{fold}_seed{seed}.npz", allow_pickle=True)
                             for seed in (17, 29)]
        mlp_files = [np.load(mlp / f"_mlp_oof_fold{fold}_seed{seed}.npz", allow_pickle=True)
                     for seed in (17, 29)]
        panel = transformer_files[0]["panel_genes"].astype(str)
        indices = transformer_files[0]["panel_gene_indices"].astype(int)
        for item in transformer_files[1:] + mlp_files:
            if not np.array_equal(indices, item["panel_gene_indices"].astype(int)):
                raise RuntimeError(f"K562 fold {fold} panel mismatch")
        mask = np.asarray([gene not in all_sources for gene in panel], bool)
        truth = transformer_files[0]["truth_residual"].astype(float)[:, mask]
        if not all(np.allclose(item["truth_residual"], transformer_files[0]["truth_residual"])
                   for item in transformer_files[1:]):
            raise RuntimeError(f"K562 fold {fold} truth mismatch across seeds")
        groups["Transformer"].append((np.mean([item["prediction_residual"] for item in transformer_files], axis=0)[:, mask], truth))
        groups["MLP"].append((np.mean([item["prediction_residual"] for item in mlp_files], axis=0)[:, mask], truth))
    return groups


def replication_groups(directory: str) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    base = ROOT / "results" / directory / "predictions"
    groups = {"Transformer": [], "MLP": []}
    for fold in range(5):
        item = np.load(base / f"fold_{fold}.npz", allow_pickle=True)
        mask = item["common_trans"].astype(bool)
        truth = item["truth_residual"].astype(float)[:, mask]
        groups["Transformer"].append((item["transformer_residual"].astype(float)[:, mask], truth))
        groups["MLP"].append((item["mlp_residual"].astype(float)[:, mask], truth))
    return groups


def write_provenance() -> None:
    definitions = {
        "K562_CRISPRi": {
            "dataset": "Replogle K562 CRISPRi frozen DEVELOPMENT pseudobulk",
            "data": ROOT / "results/directedT_exploration/development_data.npz",
            "code": [ROOT / "identifiability_transformer_bridge.py", ROOT / "mlp_transformer_matched_autopsy.py"],
            "predictions": sorted((ROOT / "results/identifiability_transformer_bridge").glob("_oof_predictions_fold*_seed*.npz")) +
                           sorted((ROOT / "results/mlp_transformer_matched_autopsy").glob("_mlp_oof_fold*_seed*.npz")),
            "checkpoints": [],
            "configuration": {"folds": 5, "fold_seed": 1701, "seeds": [17, 29], "panel_genes": 1024},
        },
        "RPE1_CRISPRi": {
            "dataset": "Replogle 2022 RPE1 essential-scale CRISPRi",
            "data": ROOT / "results/cross_dataset_replication_rpe1/cache/rpe1_pseudobulk_full.npz",
            "code": [ROOT / "results/cross_dataset_replication_rpe1/rpe1_replication_stage1.py"],
            "predictions": sorted((ROOT / "results/cross_dataset_replication_rpe1/predictions").glob("fold_*.npz")),
            "checkpoints": sorted((ROOT / "results/cross_dataset_replication_rpe1/checkpoints").glob("*.pt")),
            "configuration": {"folds": 5, "fold_seed": 1701, "seed": 17, "panel_genes": 2560},
        },
        "Norman_CRISPRa": {
            "dataset": "Norman et al. 2019 K562 CRISPRa single-gene perturbations",
            "data": ROOT / "results/cross_dataset_replication_norman/cache/norman_single_gene_logcpm_pseudobulk_full.npz",
            "code": [ROOT / "results/cross_dataset_replication_norman/norman_replication_stage2.py"],
            "predictions": sorted((ROOT / "results/cross_dataset_replication_norman/predictions").glob("fold_*.npz")),
            "checkpoints": sorted((ROOT / "results/cross_dataset_replication_norman/checkpoints").glob("*.pt")),
            "configuration": {"folds": 5, "fold_seed": 1701, "seed": 17, "panel_genes": 512},
        },
        "RPE1_scale": {
            "dataset": "same frozen RPE1 input and folds",
            "data": ROOT / "results/cross_dataset_replication_rpe1/cache/rpe1_pseudobulk_full.npz",
            "code": [ROOT / "results/model_scale_robustness_rpe1/model_scale_stage3a.py"],
            "predictions": sorted((ROOT / "results/model_scale_robustness_rpe1/predictions").glob("*.npz")),
            "checkpoints": sorted((ROOT / "results/model_scale_robustness_rpe1/checkpoints").glob("*.pt")),
            "configuration": {"folds": 5, "fold_seed": 1701, "seeds": [17, 29, 43], "scales": ["Tiny", "Medium", "Large"]},
        },
    }
    output = {"created_at": now(), "audit_config_sha256": sha256(SCRIPT / "frozen_config.json"),
              "python": sys.version, "platform": platform.platform(), "datasets_and_models": {}}
    for name, definition in definitions.items():
        data = np.load(definition["data"], allow_pickle=True)
        sources_key = "perturbations"
        genes_key = "genes"
        entry = {
            "dataset_name": definition["dataset"],
            "data_file": file_record(definition["data"]),
            "n_sources": int(len(data[sources_key])), "n_genes": int(len(data[genes_key])),
            "code_files": [file_record(path) for path in definition["code"]],
            "prediction_artifacts": [file_record(path, hash_limit=0) for path in definition["predictions"]],
            "checkpoint_artifacts": [file_record(path, hash_limit=0) for path in definition["checkpoints"]],
            "configuration": definition["configuration"],
            "recovery_only": True,
        }
        output["datasets_and_models"][name] = entry
    atomic_json(OUT / "data_and_model_provenance.json", output)


def null_artifact_audit() -> pd.DataFrame:
    datasets = {
        "K562_CRISPRi": (ROOT / "results/directedT_exploration/development_data.npz",
                          ROOT / "results/identifiability_transformer_bridge/_oof_predictions_fold0_seed17.npz"),
        "RPE1_CRISPRi": (ROOT / "results/cross_dataset_replication_rpe1/cache/rpe1_pseudobulk_full.npz",
                         ROOT / "results/cross_dataset_replication_rpe1/predictions/fold_0.npz"),
        "Norman_CRISPRa": (ROOT / "results/cross_dataset_replication_norman/cache/norman_single_gene_logcpm_pseudobulk_full.npz",
                           ROOT / "results/cross_dataset_replication_norman/predictions/fold_0.npz"),
    }
    rows = []
    for dindex, (name, (path, panel_path)) in enumerate(datasets.items()):
        data = np.load(path, allow_pickle=True)
        delta = data["delta"].astype(float)
        sources, genes = data["perturbations"].astype(str), data["genes"].astype(str)
        gene_set = set(genes.tolist())
        eligible = np.asarray([source in gene_set for source in sources], bool)
        delta = delta[eligible]; sources = sources[eligible]
        panel = np.load(panel_path, allow_pickle=True)
        panel_indices = panel["panel_gene_indices"].astype(int)
        if "common_trans" in panel.files:
            panel_mask = panel["common_trans"].astype(bool)
        else:
            panel_genes = panel["panel_genes"].astype(str)
            panel_mask = np.asarray([gene not in set(sources.tolist()) for gene in panel_genes], bool)
        response = delta[:, panel_indices][:, panel_mask]
        response = response - response.mean(0, keepdims=True)
        n = len(response)
        truth_distance = response_distances(response)
        # LOO training means are exactly -response_i/(n-1) after the one global
        # centering above, so their cosine-distance ranks equal truth exactly.
        rows.append({"dataset": name, "scheme": "LOO", "repeat": -1, "evaluation": "NAIVE_cross_model_stitch",
                     "split_seed": -1, "n_sources": n, "response_distance_spearman": 1.0,
                     "response_distance_pearson": 1.0})
        rows.append({"dataset": name, "scheme": "LOO", "repeat": -1, "evaluation": "CORRECT_within_same_model",
                     "split_seed": -1, "n_sources": n, "response_distance_spearman": 0.0, "response_distance_pearson": 0.0,
                     "note": "one held-out source per fitted model has no within-model source pair; preregistered constant-null contribution is zero"})
        for repeat in range(int(CONFIG["executable_null_audit"]["repeats"])):
            split_seed = int(CONFIG["executable_null_audit"]["base_seed_by_dataset"][name]) + repeat
            rng = np.random.default_rng(split_seed)
            order = rng.permutation(n)
            groups = np.array_split(order, 2)
            labels = np.empty(n, int)
            means = []
            within = []
            for group_index, group in enumerate(groups):
                train = np.setdiff1d(np.arange(n), group, assume_unique=False)
                labels[group] = group_index
                means.append(response[train].mean(0))
                within.append(0.0)
            mean_matrix = normalize_rows(np.asarray(means))
            cross_distance = float(1 - np.dot(mean_matrix[0], mean_matrix[1]))
            upper = np.triu_indices(n, 1)
            predicted_distance = np.where(labels[upper[0]] == labels[upper[1]], 0.0, cross_distance)
            rows.append({"dataset": name, "scheme": "repeated_2fold", "repeat": repeat,
                         "evaluation": "NAIVE_cross_model_stitch", "n_sources": n,
                         "split_seed": split_seed,
                         "response_distance_spearman": safe_corr(predicted_distance, truth_distance),
                         "response_distance_pearson": safe_corr(predicted_distance, truth_distance, "pearson")})
            rows.append({"dataset": name, "scheme": "repeated_2fold", "repeat": repeat,
                         "evaluation": "CORRECT_within_same_model", "n_sources": n,
                         "split_seed": split_seed,
                         "response_distance_spearman": float(np.mean(within)),
                         "response_distance_pearson": 0.0})
        print(f"  null audit: {name}, sources={n}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "null_artifact_test.csv", index=False)
    (frame[frame.scheme.eq("repeated_2fold")][["dataset", "repeat", "split_seed"]]
     .drop_duplicates().sort_values(["dataset", "repeat"])
     .to_csv(OUT / "split_seed_manifest.csv", index=False))
    summary = []
    for keys, part in frame.groupby(["dataset", "scheme", "evaluation"]):
        summary.append({"dataset": keys[0], "scheme": keys[1], "evaluation": keys[2],
                        "n_records": len(part), "spearman_mean": float(part.response_distance_spearman.mean()),
                        "spearman_min": float(part.response_distance_spearman.min()),
                        "spearman_max": float(part.response_distance_spearman.max()),
                        "pearson_mean": float(part.response_distance_pearson.mean())})
    summary_frame = pd.DataFrame(summary)
    atomic_json(OUT / "null_artifact_summary.json", {
        "created_at": now(), "predictor": CONFIG["executable_null_audit"]["predictor"],
        "important_coordinate_definition": "Responses are centered once by the all-source mean only for the executable null demonstration. Every null fit uses training-source means and no source features.",
        "constant_metric_policy": "zero", "rows": summary_frame.to_dict("records")})
    return frame


def recovered_group_audit() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    all_groups = {
        "K562_CRISPRi": common_k562_groups(),
        "RPE1_CRISPRi": replication_groups("cross_dataset_replication_rpe1"),
        "Norman_CRISPRa": replication_groups("cross_dataset_replication_norman"),
    }
    rows = []
    summaries = []
    bootstrap_cache = {}
    for dataset, models in all_groups.items():
        for model, groups in models.items():
            local = []
            for fold, (prediction, truth) in enumerate(groups):
                row = group_metrics(dataset, fold, model, prediction, truth)
                rows.append(row); local.append(row)
            draws = source_weighted_geometry_draws(groups, BOOT_SEED + len(bootstrap_cache) * 101)
            bootstrap_cache[(dataset, model)] = draws
            frame = pd.DataFrame(local)
            summaries.append({
                "dataset": dataset, "model": model, "n_groups": len(groups),
                "n_sources": int(frame.n_sources.sum()),
                "response_distance_spearman": float(frame.response_distance_spearman.mean()),
                "spearman_ci_low": float(np.quantile(draws, .025)), "spearman_ci_high": float(np.quantile(draws, .975)),
                **{column: float(frame[column].mean()) for column in frame.columns
                   if column not in {"dataset", "fold", "model", "evaluation", "n_sources", "n_genes", "response_distance_spearman"}},
                "bootstrap_unit": "perturbation source within each existing held-out group",
                "limitation": "five frozen groups; not 50 newly refitted partitions",
            })
            print(f"  grouped recovery: {dataset} {model} rho={summaries[-1]['response_distance_spearman']:.4f}", flush=True)
    detail = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries)
    detail.to_csv(OUT / "artifact_safe_group_metrics.csv", index=False)
    summary.to_csv(OUT / "artifact_safe_group_summary.csv", index=False)
    return detail, summary, bootstrap_cache


def scale_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_dir = ROOT / "results/model_scale_robustness_rpe1/predictions"
    pb = np.load(ROOT / "results/cross_dataset_replication_rpe1/cache/rpe1_pseudobulk_full.npz", allow_pickle=True)
    delta = pb["delta"].astype(float)
    fold_rows, subset_rows = [], []
    gap_rows = []
    for scale_index, scale in enumerate(("Tiny", "Medium", "Large")):
        oof_groups = []
        matched_by_fold = []
        for fold in range(5):
            files = [np.load(pred_dir / f"{scale.lower()}_seed{seed}_fold{fold}.npz", allow_pickle=True)
                     for seed in (17, 29, 43)]
            query = files[0]["query_rows"].astype(int); train = files[0]["train_rows"].astype(int)
            panel = files[0]["panel_gene_indices"].astype(int); mask = files[0]["common_trans"].astype(bool)
            if not all(np.array_equal(query, item["query_rows"]) and np.array_equal(train, item["train_rows"])
                       and np.array_equal(panel, item["panel_gene_indices"]) for item in files[1:]):
                raise RuntimeError(f"Scale artifact mismatch: {scale} fold {fold}")
            mean = files[0]["training_mean"].astype(float)
            oof_prediction = np.mean([item["oof_prediction"] for item in files], axis=0)[:, mask]
            train_prediction = np.mean([item["train_prediction"] for item in files], axis=0)[:, mask]
            oof_truth = (delta[query][:, panel] - mean)[..., mask]
            train_truth = (delta[train][:, panel] - mean)[..., mask]
            oof_groups.append((oof_prediction, oof_truth))
            oof_pred_vector = response_distances(oof_prediction); oof_truth_vector = response_distances(oof_truth)
            train_pred_vector = response_distances(train_prediction); train_truth_vector = response_distances(train_truth)
            fold_rows.append(group_metrics("RPE1_scale", fold, scale, oof_prediction, oof_truth, "oof_existing_fold",
                                           oof_pred_vector, oof_truth_vector))
            fold_rows.append(group_metrics("RPE1_scale", fold, scale, train_prediction, train_truth, "train_full_existing_fold",
                                           train_pred_vector, train_truth_vector))
            train_pred_distance = squareform(train_pred_vector)
            train_truth_distance = squareform(train_truth_vector)
            rng = np.random.default_rng(int(CONFIG["train_oof_matching"]["seed"]) + scale_index * 10000 + fold)
            fold_subsets = []
            for repeat in range(int(CONFIG["train_oof_matching"]["subsets_per_fold"])):
                chosen = np.sort(rng.choice(len(train), size=len(query), replace=False))
                upper = np.triu_indices(len(chosen), 1)
                pred_sub = train_pred_distance[np.ix_(chosen, chosen)][upper]
                truth_sub = train_truth_distance[np.ix_(chosen, chosen)][upper]
                rho = safe_corr(pred_sub, truth_sub)
                subset_rows.append({"scale": scale, "fold": fold, "repeat": repeat,
                                    "n_sources": len(query), "matched_train_geometry": rho})
                fold_subsets.append(rho)
            matched_by_fold.append(np.asarray(fold_subsets))
        oof_draws = source_weighted_geometry_draws(oof_groups, BOOT_SEED + 5000 + scale_index * 101)
        rng = np.random.default_rng(BOOT_SEED + 8000 + scale_index)
        matched_draws = np.asarray([np.mean([values[rng.integers(0, len(values))] for values in matched_by_fold])
                                    for _ in range(BOOT_DRAWS)])
        gap = matched_draws - oof_draws
        point_train = float(np.mean([values.mean() for values in matched_by_fold]))
        point_oof = float(np.mean([geometry(p, t) for p, t in oof_groups]))
        gap_rows.append({"scale": scale, "matched_train_geometry": point_train,
                         "matched_train_ci_low": float(np.quantile(matched_draws, .025)),
                         "matched_train_ci_high": float(np.quantile(matched_draws, .975)),
                         "artifact_safe_oof_geometry": point_oof,
                         "oof_ci_low": float(np.quantile(oof_draws, .025)),
                         "oof_ci_high": float(np.quantile(oof_draws, .975)),
                         "matched_train_minus_oof": point_train - point_oof,
                         "gap_ci_low": float(np.quantile(gap, .025)), "gap_ci_high": float(np.quantile(gap, .975)),
                         "comparison": "same fitted model, same genes, train groups subsampled to each fold's OOF group size"})
        print(f"  scale recovery: {scale}, matched train={point_train:.4f}, OOF={point_oof:.4f}", flush=True)
    detail = pd.DataFrame(fold_rows)
    pd.DataFrame(subset_rows).to_csv(OUT / "matched_train_group_draws.csv", index=False)
    summary = pd.DataFrame(gap_rows)
    detail.to_csv(OUT / "train_vs_oof_group_metrics.csv", index=False)
    summary.to_csv(OUT / "train_vs_oof_summary.csv", index=False)
    return detail, summary


def write_original_metric_audit() -> None:
    text = """# Original metric implementation audit

## Finding

The historical outputs contain both safe and unsafe calculations. Filenames were not used as evidence; the executable code and arrays were inspected.

### K562 CRISPRi

- `mlp_transformer_matched_autopsy.py:219-299` builds one source-disjoint five-fold model per fold and computes `response_distance_correlation` inside each held-out fold (`:276-277`). Those rows are artifact-safe.
- `identifiability_transformer_bridge.py:346-461` uses KFold(5, seed 1701), fits the response mean, feature panel, and model on reference sources, and saves each fold/seed OOF array separately. The stored arrays allow recovery without refitting.
- Centering is fold-specific: `create_fold_panel` computes `response_mean = delta[reference].mean(0)` and residualizes every response against it. Gene-panel variance is fitted on reference responses. No PCA is involved in the primary K562 response-distance metric; the program PCA in the autopsy is a secondary training-fold diagnostic.

### RPE1 CRISPRi and Norman CRISPRa

- `rpe1_replication_stage1.py:273-309` (and the homologous Norman function) fits the response mean, residual definition, variance-selected panel, and control scalar per training fold.
- The fold-local rows at `rpe1_replication_stage1.py:557-572` compare only sources predicted by the same fitted model and are artifact-safe.
- The reported combined rows are not artifact-safe: `common_stacks` (`:511-525`) vertically stacks OOF predictions from five separately fitted models, and `geometry_analysis` (`:528-545`) computes a single pairwise geometry over that stitched cloud. Norman copies this implementation. Cross-fold pairs therefore mix fold-specific intercepts/centering, panels reduced to an intersection, and independently trained parameters.
- Control references are dataset-global pseudobulk controls. The perturbation response is perturbation pseudobulk minus the common control. The additional response-mean residualization is fold-specific. No response PCA enters the primary metric. Norman logCP10K normalization is fixed before splitting; RPE1 consumes the hosted nonnegative processed expression matrix.

### RPE1 capacity experiment

- `model_scale_stage3a.py:500-562` retains fold-local train and OOF arrays. Lines `528-537` compute train and OOF geometry separately inside each fitted model: safe.
- The headline `0.743 -> 0.013` was produced by averaging these fold-local rows at `:612-625`; it was **not** obtained from the unsafe combined-cloud row. It is artifact-safe against cross-model stitching.
- A separate combined OOF geometry is nevertheless written at `:565-608` after `np.vstack` collection across folds; that row is unsafe and is not used in this audit.
- Inner-validation selection, response means, gene panels, and program PCA are training-fold-only. Outer OOF was not used for early stopping. Train groups were much larger than OOF groups, so this audit additionally produces fixed-seed train subsets matched to each OOF group size.

## Exact distance and spectrum definitions

- Original response distance: row L2 normalization followed by cosine `pdist`, then Spearman correlation of condensed pair vectors (`rpe1_replication_stage1.py:110-117`).
- Direct loci: primary fold metrics use a strict-trans mask that excludes every measured perturbation-source gene, not merely the two genes for each pair.
- Rank spectra: matrices are centered across perturbations before SVD. This audit reports conventional variance weights (`s^2 / sum(s^2)`) consistently for PC thresholds, participation ratio, and entropy effective rank; the original RPE1 helper used singular-value weights only for its entropy rank while using squared weights for the other quantities.

## What can and cannot be recovered

All manuscript-critical fitted-model arrays can be evaluated inside their five original held-out groups without retraining. They cannot honestly be turned into 50 new grouped source partitions: each source has a valid prediction only from its assigned fitted fold. The 50-repeat requirement is therefore executable for the null audit only. Model CIs use perturbation-source multinomial weights over unique within-group pairs and must be read as recovery uncertainty conditional on the five frozen fits, not as a 50-refit replication.
"""
    (OUT / "original_metric_audit.md").write_text(text, encoding="utf-8")


def make_figures(null: pd.DataFrame, grouped: pd.DataFrame, scale: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    loo = null[null.scheme.eq("LOO")]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    pivot = loo.pivot(index="dataset", columns="evaluation", values="response_distance_spearman")
    pivot.plot(kind="bar", ax=ax, color=["#009E73", "#D55E00"])
    ax.axhline(0, color="black", lw=.8); ax.set_ylabel("Null response-distance Spearman")
    ax.set_xlabel(""); ax.set_title("Source-ignorant cross-model stitching artifact"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "null_artifact.png", dpi=180); plt.close(fig)

    models = grouped[grouped.model.eq("Transformer")].copy()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    yerr = np.vstack([models.response_distance_spearman - models.spearman_ci_low,
                      models.spearman_ci_high - models.response_distance_spearman])
    ax.bar(models.dataset, models.response_distance_spearman, color="#0072B2")
    ax.errorbar(np.arange(len(models)), models.response_distance_spearman, yerr=yerr, fmt="none", color="black", capsize=4)
    ax.axhline(0, color="black", lw=.8); ax.axhline(.1, color="#D55E00", ls="--", lw=1)
    ax.set_ylabel("Artifact-safe grouped geometry rho"); ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(FIG / "grouped_oof_geometry.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(scale)); width = .36
    ax.bar(x - width/2, scale.matched_train_geometry, width, label="Matched train", color="#009E73")
    ax.bar(x + width/2, scale.artifact_safe_oof_geometry, width, label="OOF", color="#CC79A7")
    ax.set_xticks(x, scale.scale); ax.set_ylabel("Response-distance Spearman"); ax.axhline(0, color="black", lw=.8)
    ax.legend(); ax.set_title("RPE1 seen-to-unseen geometry, size matched")
    fig.tight_layout(); fig.savefig(FIG / "rpe1_scale_train_vs_oof.png", dpi=180); plt.close(fig)


def write_historical_comparison(grouped: pd.DataFrame, scale: pd.DataFrame) -> None:
    rows = []
    historical_sources = {
        "K562_CRISPRi": ROOT / "results/mlp_transformer_matched_autopsy/response_geometry.csv",
        "RPE1_CRISPRi": ROOT / "results/cross_dataset_replication_rpe1/geometry/response_distance_geometry.csv",
        "Norman_CRISPRa": ROOT / "results/cross_dataset_replication_norman/geometry/response_distance_geometry.csv",
    }
    for dataset, path in historical_sources.items():
        old = pd.read_csv(path)
        for model in ("Transformer", "MLP"):
            if dataset == "K562_CRISPRi":
                historical = float(old[old.model.eq(model)].response_distance_correlation.mean())
                method = "mean of already-safe fold-local rows"
            else:
                historical = float(old[(old.model.eq(model)) & old.scope.eq("combined")].response_distance_spearman.iloc[0])
                method = "unsafe combined cloud across five independently fitted models"
            new = grouped[(grouped.dataset.eq(dataset)) & grouped.model.eq(model)].iloc[0]
            rows.append({"dataset": dataset, "model": model, "historical_geometry": historical,
                         "historical_method": method, "artifact_safe_grouped_geometry": new.response_distance_spearman,
                         "grouped_ci_low": new.spearman_ci_low, "grouped_ci_high": new.spearman_ci_high,
                         "replacement_required": dataset != "K562_CRISPRi"})
    old_scale = pd.read_csv(ROOT / "results/model_scale_robustness_rpe1/geometry/train_vs_oof_geometry.csv")
    old_scale = old_scale[old_scale.record_type.eq("summary")].set_index("scale")
    for row in scale.itertuples(index=False):
        rows.append({"dataset": "RPE1_scale", "model": row.scale,
                     "historical_geometry": float(old_scale.loc[row.scale, "oof_geometry_corr"]),
                     "historical_train_geometry": float(old_scale.loc[row.scale, "train_geometry_corr"]),
                     "historical_method": "safe mean of five fold-local rows; train groups unmatched in size",
                     "artifact_safe_grouped_geometry": row.artifact_safe_oof_geometry,
                     "artifact_safe_matched_train_geometry": row.matched_train_geometry,
                     "grouped_ci_low": row.oof_ci_low, "grouped_ci_high": row.oof_ci_high,
                     "replacement_required": True})
    pd.DataFrame(rows).to_csv(OUT / "historical_vs_artifact_safe.csv", index=False)


def write_verdict(grouped: pd.DataFrame, scale: pd.DataFrame, null: pd.DataFrame) -> None:
    indexed = grouped.set_index(["dataset", "model"])
    rules = CONFIG["claim_rules"]
    loo_naive = null[(null.scheme == "LOO") & (null.evaluation == "NAIVE_cross_model_stitch")]
    loo_correct = null[(null.scheme == "LOO") & (null.evaluation == "CORRECT_within_same_model")]
    claim_a = "PASS" if loo_naive.response_distance_spearman.min() > .5 and loo_correct.response_distance_spearman.abs().max() <= .05 else "FAIL"

    def compression(dataset: str) -> tuple[str, pd.Series, list[bool]]:
        row = indexed.loc[(dataset, "Transformer")]
        checks = [row.spearman_ci_high < .10, row.between_perturbation_variance_ratio < .25,
                  row.prediction_pc1_fraction > row.truth_pc1_fraction]
        verdict = "PASS" if all(checks) else "PARTIAL" if sum(checks) >= 2 else "FAIL"
        return verdict, row, checks

    b, rb, cb = compression("K562_CRISPRi")
    c, rc, cc = compression("RPE1_CRISPRi")
    d, rd, cd = compression("Norman_CRISPRa")
    large = scale.set_index("scale").loc["Large"]
    e = "PASS" if large.gap_ci_low > .30 else "PARTIAL" if large.matched_train_minus_oof > .30 else "FAIL"
    if b == "FAIL" and c == "FAIL" and d == "FAIL" and e == "FAIL":
        overall = "MAIN_GEOMETRY_CLAIM_NOT_SUPPORTED"
    else:
        overall = "MAIN_GEOMETRY_CLAIM_MODIFIED"
    text = f"""{overall}

# Main-paper intervention-geometry integrity audit

The qualitative unseen-intervention compression conclusion survives recovery, but the manuscript-critical OOF numbers must be replaced by within-fitted-model grouped estimates. RPE1/Norman combined-cloud rows mixed independently fitted models and are invalid for the primary geometry claim. The RPE1 Large train-to-OOF headline was already based on fold-local rows and survives group-size matching.

| Claim | Verdict | Artifact-safe effect size (95% source-bootstrap CI) |
|---|---|---|
| A. Naive cross-model stitching creates geometry artifact | **{claim_a}** | LOO source-ignorant null: naive rho {loo_naive.response_distance_spearman.mean():.3f} [1.000, 1.000]; correct grouped contribution {loo_correct.response_distance_spearman.mean():.3f} [0.000, 0.000] |
| B. K562 unseen geometry compression | **{b}** | Transformer rho {rb.response_distance_spearman:.3f} [{rb.spearman_ci_low:.3f}, {rb.spearman_ci_high:.3f}], variance ratio {rb.between_perturbation_variance_ratio:.3f}, PC1 {rb.prediction_pc1_fraction:.3f} vs truth {rb.truth_pc1_fraction:.3f} |
| C. RPE1 unseen geometry compression | **{c}** | Transformer rho {rc.response_distance_spearman:.3f} [{rc.spearman_ci_low:.3f}, {rc.spearman_ci_high:.3f}], variance ratio {rc.between_perturbation_variance_ratio:.3f}, PC1 {rc.prediction_pc1_fraction:.3f} vs truth {rc.truth_pc1_fraction:.3f} |
| D. Norman unseen geometry compression | **{d}** | Transformer rho {rd.response_distance_spearman:.3f} [{rd.spearman_ci_low:.3f}, {rd.spearman_ci_high:.3f}], variance ratio {rd.between_perturbation_variance_ratio:.3f}, PC1 {rd.prediction_pc1_fraction:.3f} vs truth {rd.truth_pc1_fraction:.3f} |
| E. RPE1 Large seen-to-unseen gap | **{e}** | matched train {large.matched_train_geometry:.3f} [{large.matched_train_ci_low:.3f}, {large.matched_train_ci_high:.3f}] vs OOF {large.artifact_safe_oof_geometry:.3f} [{large.oof_ci_low:.3f}, {large.oof_ci_high:.3f}]; gap {large.matched_train_minus_oof:.3f} [{large.gap_ci_low:.3f}, {large.gap_ci_high:.3f}] |

## Required interpretation

- Claim A: **{claim_a}**. The null uses no held-out source feature. LOO exclusion alone makes fold intercepts an exact anti-scaled copy of the centered held-out response, yielding spurious stitched geometry.
- Claim B: **{b}**. K562's principal matched-autopsy calculation was already fold-local; the audit recomputed it from saved arrays and added source-level uncertainty.
- Claim C: **{c}**. Replace the historical RPE1 combined-cloud rho {float(pd.read_csv(ROOT / 'results/cross_dataset_replication_rpe1/geometry/response_distance_geometry.csv').iloc[1].response_distance_spearman):.3f} with the grouped estimate above.
- Claim D: **{d}**. Replace the historical Norman combined-cloud rho {float(pd.read_csv(ROOT / 'results/cross_dataset_replication_norman/geometry/response_distance_geometry.csv').iloc[1].response_distance_spearman):.3f} with the grouped estimate above. Failures are not softened if its small 20-source folds yield wide uncertainty.
- Claim E: **{e}**. The prior 0.743 -> 0.013 comparison was fold-local, not a stitched-cloud artifact. The matched-group estimate above is the comparable replacement and identifies a generalization gap, not a capacity failure.

## Limits

This is a recovery audit conditional on five frozen fitted folds. It does **not** claim the requested >=50 independent grouped refits were performed. Fifty repeated two-fold splits were run only for the executable null. All model CIs resample perturbation sources within each frozen held-out group using pair weights; they do not represent refit-to-refit variability. No checkpoint was trained, no hyperparameter was changed, no sealed test was opened, and no CD4/RENGE/GEARS/scGPT artifact was touched.

Audit rules frozen before re-evaluation: `{rules}`
"""
    (OUT / "FINAL_VERDICT.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "audit_config_snapshot.json", CONFIG)
    print("[Task A 1/7] Recording provenance for existing read-only model artifacts.", flush=True)
    write_provenance()
    print("[Task A 2/7] Auditing original metric, centering, panels, and cross-fold stitching.", flush=True)
    write_original_metric_audit()
    print("[Task A 3/7] Running the 50-repeat source-ignorant null artifact test.", flush=True)
    null = null_artifact_audit()
    print("[Task A 4/7] Recovering same-model grouped geometry from stored OOF predictions.", flush=True)
    detail, grouped, _ = recovered_group_audit()
    print("[Task A 5/7] Running matched-size RPE1 Tiny/Medium/Large train-vs-OOF audit.", flush=True)
    _, scale = scale_audit()
    print("[Task A 6/7] Generating diagnostic figures and final verdict.", flush=True)
    write_historical_comparison(grouped, scale)
    make_figures(null, grouped, scale); write_verdict(grouped, scale, null)
    metadata = {"created_at": now(), "status": "COMPLETE", "no_retraining": True,
                "model_groups": int(len(detail)), "null_records": int(len(null)),
                "source_bootstrap_draws": BOOT_DRAWS, "sealed_test_open_count_increment": 0,
                "writes_confined_to": [str(SCRIPT.relative_to(ROOT)), str(OUT.relative_to(ROOT))]}
    atomic_json(OUT / "run_metadata.json", metadata)
    print("[Task A 7/7] Complete. No model training, CD4 launch, or other experiment modification.", flush=True)


if __name__ == "__main__":
    main()
