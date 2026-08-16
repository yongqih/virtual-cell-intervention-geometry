from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.gears_geometry_audit.run_audit import rank_metrics, safe_spearman

OUT = ROOT / "results" / "preprint_finalization"
RPE1 = ROOT / "results" / "cross_dataset_replication_rpe1"
FROZEN = ROOT / "results" / "final_literature_model_audit"
PAIR_SEED = 26081503


def normalize_rows(x):
    x = np.asarray(x, float)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def atomic_csv(path: Path, frame: pd.DataFrame, compression=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.gz" if str(path).endswith(".gz") else ".tmp"
    tmp = path.with_name(path.name + suffix)
    frame.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)


def build_pairwise():
    cache = np.load(RPE1 / "cache" / "control_state_representations.npz", allow_pickle=True)
    rep_names = cache["perturbations"].astype(str)
    rep = cache["EstablishedOBS71"].astype(float)
    rep_lookup = {name: i for i, name in enumerate(rep_names)}
    split = json.loads((RPE1 / "split_definition.json").read_text(encoding="utf-8"))["folds"]
    fold_files = [np.load(RPE1 / "predictions" / f"fold_{fold}.npz", allow_pickle=True) for fold in range(5)]
    common = None
    for z in fold_files:
        idx = z["panel_gene_indices"].astype(int)
        valid = set(idx[z["common_trans"].astype(bool)].tolist())
        common = valid if common is None else common & valid
    common = np.asarray(sorted(common), int)
    rows, fold_summary = [], []
    rng = np.random.default_rng(PAIR_SEED)
    sampled = []
    for fold, z in enumerate(fold_files):
        query_names = z["perturbations"].astype(str)
        query_rep = np.asarray([rep_lookup[name] for name in query_names], int)
        reference_names = [name for name in split[fold]["train_sources"] if name in rep_lookup]
        reference_rep = np.asarray([rep_lookup[name] for name in reference_names], int)
        scaler = StandardScaler().fit(rep[reference_rep])
        rq = normalize_rows(scaler.transform(rep[query_rep]))
        idx = z["panel_gene_indices"].astype(int)
        loc = {gene: i for i, gene in enumerate(idx)}
        cols = np.asarray([loc[gene] for gene in common], int)
        response = z["truth_residual"].astype(float)[:, cols]
        state_sim = 1 - pdist(rq, metric="cosine")
        response_sim = 1 - pdist(normalize_rows(response), metric="cosine")
        first, second = np.triu_indices(len(query_names), 1)
        frame = pd.DataFrame({"fold": fold, "source_i": query_names[first], "source_j": query_names[second],
                              "establishedobs71_cosine_similarity": state_sim,
                              "true_response_cosine_similarity": response_sim})
        rows.append(frame)
        take = min(2000, len(frame))
        sampled.append(frame.iloc[np.sort(rng.choice(len(frame), size=take, replace=False))].assign(display_sample_seed=PAIR_SEED))
        fold_summary.append({"fold": fold, "n_sources": len(query_names), "n_pairs": len(frame),
                             "spearman_rho": safe_spearman(state_sim, response_sim), "n_response_genes": len(common)})
    full = pd.concat(rows, ignore_index=True)
    display = pd.concat(sampled, ignore_index=True)
    atomic_csv(OUT / "figure3d_pairwise_full.csv.gz", full, compression="gzip")
    atomic_csv(OUT / "figure3d_pairwise_display_sample.csv", display)
    atomic_csv(OUT / "figure3d_pairwise_fold_summary.csv", pd.DataFrame(fold_summary))
    return fold_summary


def build_common_landscape():
    with np.load(RPE1 / "cache" / "rpe1_pseudobulk_full.npz", allow_pickle=True) as z:
        pb = {k: z[k] for k in z.files}
    sets = json.loads((FROZEN / "frozen_sets.json").read_text(encoding="utf-8"))
    split = json.loads((RPE1 / "split_definition.json").read_text(encoding="utf-8"))["folds"]
    common_sources = set(sets["common_sources"])
    common_genes = sets["common_response_genes"]
    pg = {g: i for i, g in enumerate(pb["genes"].astype(str))}
    ps = {s: i for i, s in enumerate(pb["perturbations"].astype(str))}
    cols = np.asarray([pg[g] for g in common_genes], int)
    control = pb["control_mean"][cols].astype(float)
    blocks = []
    spectral_rows = []
    for fold in range(5):
        sc_path = FROZEN / "scgpt" / f"fold_{fold}" / "predictions.npz"
        ge_path = FROZEN / "gears" / "predictions" / f"gears_fold{fold}_raw_oof.npz"
        with np.load(sc_path, allow_pickle=False) as z:
            sc_names, sc_genes, sc_raw = z["sources"].astype(str), z["genes"].astype(str), z["raw_predictions"].astype(float)
        with np.load(ge_path, allow_pickle=False) as z:
            ge_names, ge_genes, ge_raw = z["sources"].astype(str), z["genes"].astype(str), z["raw_predictions"].astype(float)
        sc_loc, ge_loc = {g: i for i, g in enumerate(sc_genes)}, {g: i for i, g in enumerate(ge_genes)}
        sc_src, ge_src = {s: i for i, s in enumerate(sc_names)}, {s: i for i, s in enumerate(ge_names)}
        names = sorted(common_sources & set(sc_names) & set(ge_names) & set(split[fold]["validation_sources"]))
        train = np.asarray([ps[s] for s in split[fold]["train_sources"] if s in ps], int)
        mean = pb["delta"][train][:, cols].mean(axis=0)
        truth = np.vstack([pb["delta"][ps[s], cols] for s in names]) - mean
        gears = ge_raw[[ge_src[s] for s in names]][:, [ge_loc[g] for g in common_genes]] - control - mean
        scgpt = sc_raw[[sc_src[s] for s in names]][:, [sc_loc[g] for g in common_genes]] - control - mean
        blocks.append({"fold": fold, "names": names, "Truth": truth, "GEARS": gears, "scGPT": scgpt})
        for series, matrix in (("Truth", truth), ("GEARS", gears), ("scGPT", scgpt)):
            spectral_rows.append({"fold": fold, "series": series, "n_sources": len(names), "n_genes": len(common_genes), **rank_metrics(matrix)})
    truth_all = np.vstack([b["Truth"] for b in blocks])
    center_mean = truth_all.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(truth_all - center_mean, full_matrices=False)
    basis = vt[:2].T
    raw_truth_coords = (truth_all - center_mean) @ basis
    scale = max(float(np.max(np.abs(raw_truth_coords))), 1e-12)
    rows = []
    for block in blocks:
        for series in ("Truth", "GEARS", "scGPT"):
            coords = (block[series] - center_mean) @ basis / scale
            for source, (pc1, pc2) in zip(block["names"], coords):
                rows.append({"fold": block["fold"], "source": source, "series": series,
                             "truth_pc1_scaled": float(pc1), "truth_pc2_scaled": float(pc2)})
    atomic_csv(OUT / "figure2_common_landscape.csv", pd.DataFrame(rows))
    sf = pd.DataFrame(spectral_rows)
    summary = sf.groupby("series", as_index=False).mean(numeric_only=True).assign(record_type="fold_mean")
    atomic_csv(OUT / "figure2_common_spectral_fold.csv", sf)
    atomic_csv(OUT / "figure2_common_spectral_summary.csv", summary)
    return len(rows), summary.to_dict("records")


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing prepared data: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    fold_summary = build_pairwise()
    n_landscape, spectral = build_common_landscape()
    metadata = {"pair_display_seed": PAIR_SEED, "pair_full_rows": int(sum(r["n_pairs"] for r in fold_summary)),
                "pair_fold_summary": fold_summary, "common_landscape_rows": n_landscape,
                "common_spectral_summary": spectral, "training_or_inference": False}
    (OUT / "preparation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
