from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.gears_geometry_audit.run_audit import (
    local_geometry,
    metric_row,
    rank_metrics,
    response_distances,
    safe_pearson,
    safe_spearman,
    weighted_spearman,
)


FROZEN = ROOT / "results" / "final_literature_model_audit"
SCGPT = FROZEN / "scgpt"
RPE1 = ROOT / "results" / "cross_dataset_replication_rpe1"
PB_PATH = RPE1 / "cache" / "rpe1_pseudobulk_full.npz"
OUT = ROOT / "results" / "final_scgpt_preprint_audit"
CONFIG = json.loads((ROOT / "scripts" / "gears_geometry_audit" / "config.json").read_text(encoding="utf-8"))
N_BOOT = int(CONFIG["bootstrap_resamples"])
BOOT_SEED = int(CONFIG["bootstrap_seed"]) + 101


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(2**20):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def artifact_record(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_inputs():
    with np.load(PB_PATH, allow_pickle=True) as z:
        pb = {key: z[key] for key in z.files}
    sets = json.loads((FROZEN / "frozen_sets.json").read_text(encoding="utf-8"))
    outer = json.loads((RPE1 / "split_definition.json").read_text(encoding="utf-8"))["folds"]
    genes = pb["genes"].astype(str)
    sources = pb["perturbations"].astype(str)
    gene_lookup = {g: i for i, g in enumerate(genes)}
    source_lookup = {s: i for i, s in enumerate(sources)}
    # scGPT's frozen vocabulary lacks 11 legacy gene symbols from the 768-gene
    # strict-trans panel. The preregistered common response panel is exactly the
    # 757-gene leakage-safe intersection used for matched literature-model audit.
    panel_names = np.asarray(sets["common_response_genes"], str)
    if set(panel_names) & set(sets["eligible_sources"]):
        raise RuntimeError("Strict-trans panel contains an eligible perturbation source")
    panel = np.asarray([gene_lookup[g] for g in panel_names], int)
    return pb, sets, outer, genes, source_lookup, panel_names, panel


def verify_folds(pb, sets, outer, genes, source_lookup, panel_names, panel):
    manifest = json.loads((SCGPT / "experiment_manifest.json").read_text(encoding="utf-8"))
    experiment_hash = manifest["experiment_hash"]
    folds, split_rows, records, all_oof, checkpoint_hashes = [], [], [], set(), set()
    for fold in range(5):
        base = SCGPT / f"fold_{fold}"
        marker_path = base / "FOLD_COMPLETE.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") != "COMPLETE" or marker.get("fold") != fold or marker.get("model") != "scgpt":
            raise RuntimeError(f"Invalid completion marker for fold {fold}")
        if marker.get("experiment_hash") != experiment_hash:
            raise RuntimeError(f"Experiment hash mismatch fold {fold}")
        verified = {}
        for key in ("best_checkpoint", "predictions", "metrics", "split_audit", "convergence"):
            item = marker["artifacts"][key]
            path = base / Path(item["path"])
            if not path.is_file() or path.stat().st_size != int(item["bytes"]) or sha256(path) != item["sha256"]:
                raise RuntimeError(f"Artifact integrity mismatch fold={fold} key={key}")
            verified[key] = path
            records.append(artifact_record(path))
        checkpoint_hashes.add(marker["artifacts"]["best_checkpoint"]["sha256"])
        split = json.loads(verified["split_audit"].read_text(encoding="utf-8"))
        if split.get("experiment_hash") != experiment_hash:
            raise RuntimeError(f"Split hash mismatch fold {fold}")
        train, val, test = set(split["inner_train_sources"]), set(split["inner_val_sources"]), set(split["outer_oof_sources"])
        outer_train = set(outer[fold]["train_sources"])
        outer_test = set(outer[fold]["validation_sources"])
        if train & val or train & test or val & test or not train <= outer_train or not val <= outer_train or not test <= outer_test:
            raise RuntimeError(f"Split contamination fold {fold}")
        convergence = json.loads(verified["convergence"].read_text(encoding="utf-8"))
        with np.load(verified["predictions"], allow_pickle=False) as z:
            names = z["sources"].astype(str)
            pred_genes = z["genes"].astype(str)
            raw = z["raw_predictions"].astype(float)
        if len(names) != len(set(names)) or set(names) != test:
            raise RuntimeError(f"Prediction/source ordering mismatch fold {fold}")
        if all_oof & set(names):
            raise RuntimeError(f"OOF source duplicated across fitted folds: {fold}")
        all_oof.update(names)
        local = {g: i for i, g in enumerate(pred_genes)}
        if any(g not in local for g in panel_names):
            raise RuntimeError(f"Strict-trans prediction gene missing fold {fold}")
        raw = raw[:, [local[g] for g in panel_names]]
        query = np.asarray([source_lookup[n] for n in names], int)
        train_idx = np.asarray([source_lookup[n] for n in outer[fold]["train_sources"] if n in source_lookup], int)
        mean_response = np.mean(pb["delta"][train_idx][:, panel], axis=0)
        truth_delta = pb["delta"][query][:, panel].astype(float)
        control = pb["control_mean"][panel].astype(float)
        pred_delta = raw - control
        folds.append({"fold": fold, "names": names, "raw_pred": raw, "raw_truth": truth_delta + control,
                      "pred_delta": pred_delta, "truth_delta": truth_delta,
                      "pred_residual": pred_delta - mean_response, "truth_residual": truth_delta - mean_response,
                      "mean_response": mean_response})
        split_rows.append({"fold": fold, "inner_train_sources": len(train), "inner_val_sources": len(val),
                           "outer_oof_sources": len(test), "prediction_sources": len(names),
                           "train_val_test_overlap": 0, "outer_oof_used_for_training": False,
                           "outer_oof_used_for_checkpoint_selection": False,
                           "best_epoch": convergence["best_epoch"], "final_epoch": convergence["final_epoch"],
                           "early_stopping": convergence["early_stopping"],
                           "loaded_pretrained_tensors": convergence.get("loaded_pretrained_tensors"),
                           "checkpoint_sha256": marker["artifacts"]["best_checkpoint"]["sha256"],
                           "prediction_sha256": marker["artifacts"]["predictions"]["sha256"]})
    if all_oof != set(sets["scgpt_sources"]):
        raise RuntimeError("OOF union does not equal frozen scGPT source set")
    if len(checkpoint_hashes) != 5:
        raise RuntimeError("Five folds do not have five distinct best checkpoints")
    return folds, split_rows, records, manifest


def hierarchy_and_baselines(folds):
    hierarchy, baseline = [], []
    for data in folds:
        spaces = {
            "absolute_perturbed_state": (data["raw_pred"], data["raw_truth"]),
            "total_perturbation_response": (data["pred_delta"], data["truth_delta"]),
            "intervention_specific_residual": (data["pred_residual"], data["truth_residual"]),
        }
        baselines = {
            "absolute_perturbed_state": np.tile(data["mean_response"] + (data["raw_truth"] - data["truth_delta"])[0], (len(data["names"]), 1)),
            "total_perturbation_response": np.tile(data["mean_response"], (len(data["names"]), 1)),
            "intervention_specific_residual": np.zeros_like(data["truth_residual"]),
        }
        for space, (pred, truth) in spaces.items():
            hierarchy.append({"record_type": "fold", "fold": data["fold"], "model": "scGPT", "space": space,
                              "n_sources": len(pred), "n_genes": pred.shape[1], **metric_row(pred, truth)})
            for model, values in (("scGPT", pred), ("SourceIgnorantMeanResponse", baselines[space])):
                pred_var = float(np.mean(np.var(values, axis=0)))
                truth_var = float(np.mean(np.var(truth, axis=0)))
                baseline.append({"record_type": "fold", "fold": data["fold"], "model": model, "space": space,
                                 "n_sources": len(pred), "n_genes": pred.shape[1], **metric_row(values, truth),
                                 "same_model_geometry": safe_spearman(response_distances(values), response_distances(truth)),
                                 "between_source_variance_ratio": pred_var / max(truth_var, 1e-12)})
    for rows, keys, metrics in (
        (hierarchy, ["model", "space"], ["perturbed_state_or_response_pearson", "response_cosine", "mse", "rmse"]),
        (baseline, ["model", "space"], ["perturbed_state_or_response_pearson", "response_cosine", "mse", "rmse", "same_model_geometry", "between_source_variance_ratio"]),
    ):
        frame = pd.DataFrame(rows)
        for values, group in frame.groupby(keys):
            row = {"record_type": "fold_mean", "fold": -1, "n_sources": int(group.n_sources.sum()), "n_genes": int(group.n_genes.iloc[0])}
            row.update(dict(zip(keys, values)))
            row.update({metric: float(group[metric].mean()) for metric in metrics})
            rows.append(row)
    return hierarchy, baseline


def bootstrap_geometry(folds):
    names = sorted(name for data in folds for name in data["names"])
    lookup = {name: i for i, name in enumerate(names)}
    rng = np.random.default_rng(BOOT_SEED)
    counts = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=N_BOOT).astype(np.int16)
    accum = np.zeros(N_BOOT)
    for data in folds:
        pred_dist = response_distances(data["pred_residual"])
        truth_dist = response_distances(data["truth_residual"])
        first, second = np.triu_indices(len(data["names"]), 1)
        idx = np.asarray([lookup[name] for name in data["names"]], int)
        for start in range(0, N_BOOT, 50):
            stop = min(N_BOOT, start + 50)
            weights = counts[start:stop, idx[first]] * counts[start:stop, idx[second]]
            accum[start:stop] += weighted_spearman(pred_dist, truth_dist, weights)
    draws = accum / len(folds)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975)), draws


def geometry_local_spectrum(folds):
    geometry, local_rows, spectrum = [], [], []
    for data in folds:
        pred, truth = data["pred_residual"], data["truth_residual"]
        pdist_pred, pdist_truth = response_distances(pred), response_distances(truth)
        rho = safe_spearman(pdist_pred, pdist_truth)
        knn, local_rank = local_geometry(pred, truth, int(CONFIG["local_neighbors"]))
        geometry.append({"record_type": "fold", "fold": data["fold"], "n_sources": len(pred), "n_genes": pred.shape[1],
                         "response_distance_spearman": rho, "response_distance_pearson": safe_pearson(pdist_pred, pdist_truth),
                         "cross_fold_pairs_used": False})
        local_rows.append({"record_type": "fold", "fold": data["fold"], "n_sources": len(pred), "k": 10,
                           "knn_overlap_k10": knn, "local_distance_rank": local_rank})
        pred_rank, truth_rank = rank_metrics(pred), rank_metrics(truth)
        pred_var = float(np.mean(np.var(pred, axis=0))); truth_var = float(np.mean(np.var(truth, axis=0)))
        spectrum.append({"record_type": "fold", "fold": data["fold"], "n_sources": len(pred), "n_genes": pred.shape[1],
                         "predicted_between_source_variance": pred_var, "truth_between_source_variance": truth_var,
                         "between_source_variance_ratio": pred_var / max(truth_var, 1e-12),
                         "distance_scale_retention": float(np.mean(pdist_pred) / max(np.mean(pdist_truth), 1e-12)),
                         **{f"predicted_{k}": v for k, v in pred_rank.items()},
                         **{f"truth_{k}": v for k, v in truth_rank.items()}})
    low, high, draws = bootstrap_geometry(folds)
    gf = pd.DataFrame(geometry); lf = pd.DataFrame(local_rows); sf = pd.DataFrame(spectrum)
    geometry.append({"record_type": "summary", "fold": -1, "n_sources": int(gf.n_sources.sum()), "n_genes": int(gf.n_genes.iloc[0]),
                     "response_distance_spearman": float(gf.response_distance_spearman.mean()),
                     "fold_median_response_distance_spearman": float(gf.response_distance_spearman.median()),
                     "source_bootstrap_ci_low": low, "source_bootstrap_ci_high": high,
                     "response_distance_pearson": float(gf.response_distance_pearson.mean()),
                     "cross_fold_pairs_used": False, "bootstrap_unit": "perturbation source"})
    local_rows.append({"record_type": "summary", "fold": -1, "n_sources": int(lf.n_sources.sum()), "k": 10,
                       "knn_overlap_k10": float(lf.knn_overlap_k10.mean()),
                       "local_distance_rank": float(lf.local_distance_rank.mean())})
    numeric = [c for c in sf.columns if c not in ("record_type", "fold")]
    spectrum.append({"record_type": "fold_mean", "fold": -1, **{c: float(sf[c].mean()) for c in numeric}})
    return geometry, local_rows, spectrum, draws


def response_landscape(folds):
    truth = np.vstack([data["truth_residual"] for data in folds])
    center = truth - truth.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(center, full_matrices=False)
    basis = vt[:2].T
    scale = max(float(np.max(np.abs(center @ basis))), 1e-12)
    rows = []
    for data in folds:
        for kind, matrix in (("Truth", data["truth_residual"]), ("scGPT", data["pred_residual"])):
            coords = (matrix - truth.mean(axis=0, keepdims=True)) @ basis / scale
            for source, (pc1, pc2) in zip(data["names"], coords):
                rows.append({"fold": data["fold"], "source": source, "series": kind,
                             "truth_pc1_scaled": float(pc1), "truth_pc2_scaled": float(pc2),
                             "basis": "truth-derived strict-trans residual PCA shared across scGPT folds"})
    return rows


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing audit: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    pb, sets, outer, genes, source_lookup, panel_names, panel = load_inputs()
    folds, split_rows, records, manifest = verify_folds(pb, sets, outer, genes, source_lookup, panel_names, panel)
    hierarchy, baseline = hierarchy_and_baselines(folds)
    geometry, local_rows, spectrum, draws = geometry_local_spectrum(folds)
    landscape = response_landscape(folds)
    atomic_csv(OUT / "scgpt_metric_hierarchy.csv", hierarchy)
    atomic_csv(OUT / "scgpt_source_ignorant_audit.csv", baseline)
    atomic_csv(OUT / "scgpt_grouped_geometry.csv", geometry)
    atomic_csv(OUT / "scgpt_local_geometry.csv", local_rows)
    atomic_csv(OUT / "scgpt_variance_spectrum.csv", spectrum)
    atomic_csv(OUT / "scgpt_response_landscape.csv", landscape)
    atomic_csv(OUT / "scgpt_split_audit.csv", split_rows)
    np.savez_compressed(OUT / "scgpt_geometry_source_bootstrap.npz", draws=draws, seed=BOOT_SEED)
    g = next(row for row in geometry if row["record_type"] == "summary")
    l = next(row for row in local_rows if row["record_type"] == "summary")
    s = next(row for row in spectrum if row["record_type"] == "fold_mean")
    h = {(row["space"]): row for row in hierarchy if row["record_type"] == "fold_mean"}
    b = {(row["space"]): row for row in baseline if row["record_type"] == "fold_mean" and row["model"] == "SourceIgnorantMeanResponse"}
    entropy_ratio = s["predicted_entropy_effective_rank"] / max(s["truth_entropy_effective_rank"], 1e-12)
    signals = sum((s["between_source_variance_ratio"] < .5, s["distance_scale_retention"] < .75, entropy_ratio < .5))
    if g["source_bootstrap_ci_high"] < .2 and signals >= 2:
        verdict = "SCGPT_INTERVENTION_GEOMETRY_COMPRESSION_SUPPORTED"
    elif g["response_distance_spearman"] < .5 and signals >= 2:
        verdict = "SCGPT_INTERVENTION_GEOMETRY_COMPRESSION_PARTIALLY_SUPPORTED"
    else:
        verdict = "SCGPT_INTERVENTION_GEOMETRY_COMPRESSION_NOT_SUPPORTED"
    provenance = {
        "created_at": now(), "audit_type": "read-only frozen artifact audit", "training_run": False,
        "inference_run": False, "gpu_used": False, "dataset": "Replogle et al. RPE1 CRISPRi",
        "n_cells": 206585, "n_oof_sources": len(sets["scgpt_sources"]), "strict_trans_genes": len(panel_names),
        "model_prediction_genes": 2489, "experiment_hash": manifest["experiment_hash"],
        "model": {"name": "scGPT", "five_independent_fold_checkpoints": True,
                  "pretrained_checkpoint": "external/scGPT/checkpoints/scGPT_human/best_model.pt",
                  "checkpoint_selection": "inner-validation MSE; outer OOF untouched",
                  "loaded_pretrained_tensors_by_fold": [r["loaded_pretrained_tensors"] for r in split_rows]},
        "response_definition": {"control": "frozen global pseudobulk control_mean",
                                "total_response": "predicted/true perturbed state minus global control_mean",
                                "intervention_residual": "total response minus fold-specific outer-train mean response",
                                "strict_trans": "frozen panel excludes every eligible perturbation-source gene"},
        "geometry": {"distance": "cosine after row L2 normalization", "association": "Spearman distance-rank correlation",
                     "same_fitted_model_only": True, "cross_fold_pairs_used": False,
                     "bootstrap": f"{N_BOOT} perturbation-source multinomial resamples; seed {BOOT_SEED}"},
        "inputs": {"pseudobulk": artifact_record(PB_PATH), "frozen_sets": artifact_record(FROZEN / "frozen_sets.json"),
                   "outer_split": artifact_record(RPE1 / "split_definition.json"),
                   "experiment_manifest": artifact_record(SCGPT / "experiment_manifest.json"), "fold_artifacts": records},
        "python": platform.python_version(), "verdict": verdict,
    }
    atomic_json(OUT / "scgpt_provenance.json", provenance)
    report = f"""{verdict}

# Final frozen scGPT preprint audit

No model was trained or loaded and no inference was performed. All metrics use the five completed frozen OOF prediction files.

## Integrity and leakage

- Five folds: **PASS**; {len(sets['scgpt_sources'])} unique OOF sources and five distinct best-checkpoint hashes.
- Artifact hashes and byte sizes: **PASS**.
- Inner train, inner validation, and outer OOF source disjointness: **PASS**.
- Outer OOF used for checkpoint selection: **NO**.
- Strict-trans filtering: **PASS**; {len(panel_names)} genes, with all eligible perturbation-source genes excluded.
- Geometry firewall: **PASS**; pairwise distances were computed only within each independently fitted fold and summarized across folds.

## Canonical results

- Absolute perturbed-state Pearson: **{h['absolute_perturbed_state']['perturbed_state_or_response_pearson']:.6f}**.
- Total control-relative response Pearson: **{h['total_perturbation_response']['perturbed_state_or_response_pearson']:.6f}**.
- Intervention-specific residual Pearson: **{h['intervention_specific_residual']['perturbed_state_or_response_pearson']:.6f}**.
- Source-ignorant absolute-state Pearson: **{b['absolute_perturbed_state']['perturbed_state_or_response_pearson']:.6f}**.
- Same-model grouped intervention geometry: **{g['response_distance_spearman']:.6f}**, source-bootstrap 95% CI **[{g['source_bootstrap_ci_low']:.6f}, {g['source_bootstrap_ci_high']:.6f}]**.
- kNN@10 overlap: **{l['knn_overlap_k10']:.6f}**; local distance-rank statistic: **{l['local_distance_rank']:.6f}**.
- Between-intervention variance retention: **{s['between_source_variance_ratio']:.6f}**.
- PC1 variance fraction: prediction **{s['predicted_pc1_fraction']:.6f}**, truth **{s['truth_pc1_fraction']:.6f}**.
- Entropy effective rank: prediction **{s['predicted_entropy_effective_rank']:.6f}**, truth **{s['truth_entropy_effective_rank']:.6f}**.
- PC80: prediction **{s['predicted_pc80']:.1f}**, truth **{s['truth_pc80']:.1f}**.

## Required interpretation

- Metric illusion replicates: **{'YES' if h['absolute_perturbed_state']['perturbed_state_or_response_pearson'] > .9 and h['intervention_specific_residual']['perturbed_state_or_response_pearson'] < h['absolute_perturbed_state']['perturbed_state_or_response_pearson'] else 'NO'}**.
- Global geometry is compressed: **{'YES' if g['response_distance_spearman'] < .5 else 'NO'}**.
- Local geometry is degraded: **{'YES' if l['local_distance_rank'] < .5 else 'NO'}**.
- Between-intervention variance contracts: **{'YES' if s['between_source_variance_ratio'] < .5 else 'NO'}**.
- Spectral dimensionality contracts: **{'YES' if entropy_ratio < .5 else 'NO'}**.
- Qualitative Intervention Geometry Compression phenotype: **{verdict.replace('SCGPT_INTERVENTION_GEOMETRY_COMPRESSION_', '')}**.
"""
    (OUT / "SCGPT_FINAL_AUDIT.md").write_text(report, encoding="utf-8")
    atomic_json(OUT / "run_complete.json", {"completed_at": now(), "verdict": verdict, "training": False, "inference": False})
    print(json.dumps({"verdict": verdict, "geometry": g["response_distance_spearman"],
                      "geometry_ci": [g["source_bootstrap_ci_low"], g["source_bootstrap_ci_high"]],
                      "variance_retention": s["between_source_variance_ratio"]}, indent=2))


if __name__ == "__main__":
    main()
