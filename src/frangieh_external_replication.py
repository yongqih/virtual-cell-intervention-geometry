from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import platform
import sys
import time
from ctypes import wintypes
from itertools import combinations
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/frangieh_external_replication.json"
OUT = ROOT / "results/frangieh_external_replication"
DATA_PATH = ROOT / "data/raw/frangieh_2021_rna.h5ad"
INTERNAL_GENES = ROOT / "results/generalization_factorial/common_response_genes.txt"
DESCRIPTOR_PATH = ROOT / "results/cross_dataset_replication_rpe1/cache/control_state_representations.npz"
GEOMETRY_CODE = ROOT / "scripts/main_geometry_integrity_audit/run_audit.py"
RIDGE_CODE = ROOT / "scripts/propagation_reproduction/common.py"
CACHE = OUT / "pseudobulk_cache.npz"


def load_utilities():
    spec = importlib.util.spec_from_file_location("frozen_geometry", GEOMETRY_CODE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.path.insert(0, str(RIDGE_CODE.parent))
    from common import ridge_fit_predict

    return module.geometry, module.row_pearson, module.response_distances, module.safe_corr, ridge_fit_predict


GEOMETRY, ROW_PEARSON, RESPONSE_DISTANCES, SAFE_CORR, RIDGE_PREDICT = load_utilities()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def peak_working_set_mb() -> float:
    if platform.system() != "Windows":
        return float("nan")

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return float(counters.PeakWorkingSetSize / 2**20) if ok else float("nan")


def derangement(size: int, seed: int) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(size)
    donor = np.empty(size, dtype=int)
    donor[order] = np.roll(order, 1)
    if np.any(donor == np.arange(size)):
        raise RuntimeError("Derangement construction failed")
    return donor


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def metadata_audit(config: dict[str, Any]) -> dict[str, Any]:
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size != int(config["dataset"]["expected_bytes"]):
        raise RuntimeError("EXTERNAL_DATA_ACCESS_BLOCKED: processed RNA H5AD is missing or has the wrong size")
    previous = {}
    if (OUT / "data_audit.json").exists():
        previous = json.loads((OUT / "data_audit.json").read_text(encoding="utf-8"))
    file_hash = previous.get("sha256")
    if file_hash != config["dataset"]["sha256"]:
        file_hash = sha256(DATA_PATH)
    if file_hash != config["dataset"]["sha256"]:
        raise RuntimeError("EXTERNAL_DATA_ACCESS_BLOCKED: processed RNA H5AD hash mismatch")

    data = ad.read_h5ad(DATA_PATH, backed="r")
    context_column = config["dataset"]["context_column"]
    perturbation_column = config["dataset"]["perturbation_column"]
    control = config["dataset"]["control_label"]
    obs = data.obs[[context_column, perturbation_column, "ncounts"]].copy()
    n_sgrna_categories = int(data.obs["sgRNA"].nunique(dropna=True))
    contexts = [str(value) for value in obs[context_column].cat.categories]
    counts = obs.groupby([context_column, perturbation_column], observed=True).size().unstack(fill_value=0)
    all_perturbations = sorted(str(value) for value in counts.columns if str(value) != control)
    threshold = int(config["preprocessing"]["minimum_cells_per_perturbation_context"])
    pair_records = []
    for left, right in combinations(contexts, 2):
        shared = sorted(p for p in all_perturbations if counts.loc[left, p] >= threshold and counts.loc[right, p] >= threshold)
        pair_records.append({"contexts": [left, right], "shared_perturbations": len(shared)})
    pair_records.sort(key=lambda row: (-row["shared_perturbations"], row["contexts"]))
    primary = config["context_pair_selection"]["primary_contexts"]
    if pair_records[0]["contexts"] != primary:
        raise RuntimeError(f"Frozen primary pair {primary} is not the maximum-overlap pair {pair_records[0]['contexts']}")
    paired = sorted(p for p in all_perturbations if counts.loc[primary[0], p] >= threshold and counts.loc[primary[1], p] >= threshold)
    internal = set(INTERNAL_GENES.read_text(encoding="utf-8").splitlines())
    var_names = set(data.var_names.astype(str))
    response_genes = sorted((internal & var_names) - set(paired))
    data.file.close()
    with np.load(DESCRIPTOR_PATH, allow_pickle=True) as archive:
        descriptor_names = set(archive["perturbations"].astype(str))
    descriptor_eligible = sorted(set(paired) & descriptor_names)
    if len(descriptor_eligible) != int(config["zero_shot"]["observed_eligible_interventions"]):
        raise RuntimeError("Frozen zero-shot descriptor accessibility count changed")

    rng = np.random.default_rng(int(config["target_selection"]["seed"]))
    order = rng.permutation(len(paired))
    targets = sorted(paired[i] for i in order[: int(config["target_selection"]["target_count"])])
    remaining = sorted(set(paired) - set(targets))
    minimum_counts = {p: int(min(counts.loc[primary[0], p], counts.loc[primary[1], p])) for p in paired}
    robustness_threshold = int(np.quantile(list(minimum_counts.values()), .25, method="higher"))
    high_cell_targets = sorted(p for p in targets if minimum_counts[p] >= robustness_threshold)

    subsets: dict[int, dict[float, list[str]]] = {}
    serialized_subsets: dict[str, Any] = {}
    derangements: dict[str, dict[int, np.ndarray]] = {}
    serialized_derangements: dict[str, Any] = {}
    directions = [f"{primary[0]}_TO_{primary[1]}", f"{primary[1]}_TO_{primary[0]}"]
    for seed in config["coverage_seeds"]:
        permutation = np.random.default_rng(int(seed)).permutation(len(remaining))
        subsets[int(seed)] = {}
        serialized_subsets[str(seed)] = {}
        for coverage in config["coverage_fractions"]:
            n = max(1, int(np.floor(float(coverage) * len(remaining))))
            names = [remaining[i] for i in permutation[:n]]
            subsets[int(seed)][float(coverage)] = names
            serialized_subsets[str(seed)][f"{float(coverage):.2f}"] = {"n": n, "identities": names}
    for direction_index, direction in enumerate(directions):
        derangements[direction] = {}; serialized_derangements[direction] = {}
        for seed in config["coverage_seeds"]:
            used_seed = int(config["derangement_seed_base"]) + direction_index * 1000 + int(seed)
            donor = derangement(len(targets), used_seed)
            derangements[direction][int(seed)] = donor
            serialized_derangements[direction][str(seed)] = {
                "seed": used_seed,
                "fixed_points": int(np.sum(donor == np.arange(len(targets)))),
                "mapping": {targets[i]: targets[int(donor[i])] for i in range(len(targets))},
            }

    target_audit = {
        "status": "FROZEN_BEFORE_PSEUDOBULK_OR_PERFORMANCE_INSPECTION",
        "eligible_universe_count": len(paired),
        "eligible_universe": paired,
        "target_count": len(targets),
        "targets": targets,
        "target_seed": config["target_selection"]["seed"],
        "remaining_training_universe_count": len(remaining),
        "target_training_intersection": sorted(set(targets) & set(remaining)),
        "nested_training_sets": serialized_subsets,
        "derangements": serialized_derangements,
        "high_cell_threshold": robustness_threshold,
        "high_cell_target_count": len(high_cell_targets),
        "high_cell_targets": high_cell_targets,
        "config_sha256": sha256(CONFIG_PATH),
    }
    atomic_text(OUT / "fixed_test_targets.txt", "\n".join(targets) + "\n")
    atomic_json(OUT / "target_selection_audit.json", target_audit)
    cell_rows = []
    for perturbation in paired:
        cell_rows.append({
            "perturbation": perturbation,
            **{f"cells_{context}": int(counts.loc[context, perturbation]) for context in contexts},
            "minimum_cells_primary_pair": minimum_counts[perturbation],
            "role": "TARGET" if perturbation in set(targets) else "TRAINING_POOL",
            "high_cell_robustness_eligible": minimum_counts[perturbation] >= robustness_threshold,
        })
    pd.DataFrame(cell_rows).to_csv(OUT / "cell_count_audit.csv", index=False)

    audit = {
        "dataset_source": config["dataset"]["source_url"],
        "dataset_reference": config["dataset"]["source_reference"],
        "local_path": str(DATA_PATH.relative_to(ROOT)),
        "sha256": file_hash,
        "n_cells": int(len(obs)),
        "n_perturbations_total": len(all_perturbations),
        "perturbation_unit": "gene-level harmonized perturbation identity",
        "n_sgrna_categories": n_sgrna_categories,
        "contexts": contexts,
        "n_contexts": len(contexts),
        "perturbation_column": perturbation_column,
        "context_column": context_column,
        "control_label": control,
        "control_cells_by_context": {context: int(counts.loc[context, control]) for context in contexts},
        "response_gene_count": len(response_genes),
        "shared_perturbations_by_context": {" | ".join(row["contexts"]): row["shared_perturbations"] for row in pair_records},
        "usable_context_pairs": pair_records,
        "download_size": DATA_PATH.stat().st_size,
        "preprocessing_required": config["preprocessing"],
        "primary_pair_eligible_perturbations": len(paired),
        "safe_descriptor_eligible_perturbations": len(descriptor_eligible),
        "zero_shot_access_status": config["zero_shot"]["access_status"],
        "input_hashes": {
            str(CONFIG_PATH.relative_to(ROOT)): sha256(CONFIG_PATH),
            str(INTERNAL_GENES.relative_to(ROOT)): sha256(INTERNAL_GENES),
            str(GEOMETRY_CODE.relative_to(ROOT)): sha256(GEOMETRY_CODE),
            str(RIDGE_CODE.relative_to(ROOT)): sha256(RIDGE_CODE),
        },
    }
    if previous.get("input_hashes", {}).get(str(CONFIG_PATH.relative_to(ROOT))) == audit["input_hashes"][str(CONFIG_PATH.relative_to(ROOT))] and "smoke_gate" in previous:
        audit["smoke_gate"] = previous["smoke_gate"]
    atomic_json(OUT / "data_audit.json", audit)
    selection = {
        "selection_status": "FROZEN_BEFORE_PERFORMANCE_INSPECTION",
        "frozen_minimum_cells": threshold,
        "candidate_pairs": pair_records,
        "selected_context_pair": primary,
        "selected_shared_perturbations": len(paired),
        "selection_rule": config["context_pair_selection"]["rule"],
        "selection_reason": "maximum clean perturbation overlap at the frozen threshold, large matched controls, and distinct IFN-gamma versus T-cell co-culture immune-pressure conditions",
        "directions": directions,
        "performance_used_for_selection": False,
    }
    atomic_json(OUT / "context_pair_selection.json", selection)
    return {
        "audit": audit, "contexts": contexts, "pair": primary, "directions": directions,
        "paired": paired, "response_genes": response_genes, "targets": targets, "remaining": remaining,
        "subsets": subsets, "derangements": derangements, "minimum_counts": minimum_counts,
        "robustness_threshold": robustness_threshold, "high_cell_targets": high_cell_targets,
    }


def build_pseudobulk(config: dict[str, Any], design: dict[str, Any]) -> dict[str, np.ndarray]:
    expected_config_hash = sha256(CONFIG_PATH)
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=True) as archive:
            if str(archive["config_sha256"].item()) == expected_config_hash:
                print("Using verified pseudobulk cache", flush=True)
                return {key: archive[key].copy() for key in archive.files}
    started = time.perf_counter()
    data = ad.read_h5ad(DATA_PATH, backed="r")
    obs = data.obs[[config["dataset"]["context_column"], config["dataset"]["perturbation_column"], "ncounts"]].copy()
    gene_index = {name: i for i, name in enumerate(data.var_names.astype(str))}
    gene_columns = np.asarray([gene_index[name] for name in design["response_genes"]], dtype=int)
    print(f"Loading {len(gene_columns)} frozen RNA genes from backed H5AD", flush=True)
    expression = data.X[:, gene_columns]
    if hasattr(expression, "to_memory"):
        expression = expression.to_memory()
    expression = expression.tocsr().astype(np.float32)
    data.file.close()
    contexts = design["pair"]
    perturbations = design["paired"]
    labels = obs[config["dataset"]["perturbation_column"]].astype(str).to_numpy()
    context_values = obs[config["dataset"]["context_column"]].astype(str).to_numpy()
    keep = np.isin(context_values, contexts) & np.isin(labels, [config["dataset"]["control_label"], *perturbations])
    expression = expression[keep]
    totals = obs["ncounts"].to_numpy(float)[keep]
    expression = sparse.diags((10000.0 / np.maximum(totals, 1.0)).astype(np.float32)) @ expression
    expression.data = np.log1p(expression.data)
    labels = labels[keep]; context_values = context_values[keep]
    group_names = [(context, label) for context in contexts for label in [config["dataset"]["control_label"], *perturbations]]
    group_index = {group: i for i, group in enumerate(group_names)}
    codes = np.asarray([group_index[(context, label)] for context, label in zip(context_values, labels)], dtype=int)
    assignment = sparse.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))), shape=(len(group_names), len(codes)))
    cell_counts = np.asarray(assignment.sum(axis=1)).ravel().astype(int)
    group_means = (assignment @ expression).toarray() / np.maximum(cell_counts[:, None], 1)
    responses = np.empty((len(contexts), len(perturbations), len(design["response_genes"])), np.float32)
    counts = np.empty((len(contexts), len(perturbations)), int)
    control_counts = np.empty(len(contexts), int)
    for context_index, context in enumerate(contexts):
        control_row = group_index[(context, config["dataset"]["control_label"])]
        control_mean = group_means[control_row]
        control_counts[context_index] = cell_counts[control_row]
        for perturbation_index, perturbation in enumerate(perturbations):
            row = group_index[(context, perturbation)]
            responses[context_index, perturbation_index] = group_means[row] - control_mean
            counts[context_index, perturbation_index] = cell_counts[row]
    np.savez_compressed(
        CACHE, contexts=np.asarray(contexts), perturbations=np.asarray(perturbations),
        genes=np.asarray(design["response_genes"]), response=responses, cell_count=counts,
        control_count=control_counts, config_sha256=np.asarray(expected_config_hash),
        source_sha256=np.asarray(config["dataset"]["sha256"]),
    )
    print(f"Pseudobulk complete in {time.perf_counter()-started:.1f}s; peak={peak_working_set_mb():.1f} MB", flush=True)
    return {
        "contexts": np.asarray(contexts), "perturbations": np.asarray(perturbations),
        "genes": np.asarray(design["response_genes"]), "response": responses,
        "cell_count": counts, "control_count": control_counts,
        "config_sha256": np.asarray(expected_config_hash), "source_sha256": np.asarray(config["dataset"]["sha256"]),
    }


def fixed_ridge(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, alpha: float) -> np.ndarray:
    scaler = StandardScaler().fit(train_x)
    return RIDGE_PREDICT(scaler.transform(train_x), train_y, scaler.transform(query_x), alpha, fit_intercept=True)


def bootstrap_geometry(prediction: np.ndarray, truth: np.ndarray, bootstrap_rows: np.ndarray) -> np.ndarray:
    pred_distance = squareform(RESPONSE_DISTANCES(prediction))
    truth_distance = squareform(RESPONSE_DISTANCES(truth))
    upper = np.triu_indices(bootstrap_rows.shape[1], 1)
    values = np.empty(len(bootstrap_rows), float)
    for index, rows in enumerate(bootstrap_rows):
        values[index] = SAFE_CORR(
            pred_distance[np.ix_(rows, rows)][upper], truth_distance[np.ix_(rows, rows)][upper]
        )
    return values


def score(prediction: np.ndarray, truth: np.ndarray, bootstrap_rows: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    row_corr = ROW_PEARSON(prediction, truth)
    truth_variance = float(np.mean(np.var(truth, axis=0)))
    boot = bootstrap_geometry(prediction, truth, bootstrap_rows)
    return {
        "geometry": float(GEOMETRY(prediction, truth)),
        "geometry_ci_low": float(np.quantile(boot, .025)),
        "geometry_ci_high": float(np.quantile(boot, .975)),
        "mean_response_pearson": float(np.mean(row_corr)),
        "median_response_pearson": float(np.median(row_corr)),
        "variance_retention": float(np.mean(np.var(prediction, axis=0)) / max(truth_variance, 1e-12)),
    }, boot


def run_condition(
    config: dict[str, Any], design: dict[str, Any], pseudobulk: dict[str, np.ndarray],
    direction: str, seed: int, coverage: float, bootstrap_primary: np.ndarray, bootstrap_high: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    contexts = pseudobulk["contexts"].astype(str).tolist()
    source_context, recipient_context = direction.split("_TO_")
    source = pseudobulk["response"][contexts.index(source_context)].astype(np.float32)
    recipient = pseudobulk["response"][contexts.index(recipient_context)].astype(np.float32)
    p_index = {name: i for i, name in enumerate(pseudobulk["perturbations"].astype(str))}
    targets = design["targets"]
    train_names = design["subsets"][seed][coverage]
    if set(targets) & set(train_names):
        raise RuntimeError("Leakage firewall: targets intersect training identities")
    target_rows = np.asarray([p_index[name] for name in targets], int)
    train_rows = np.asarray([p_index[name] for name in train_names], int)
    donor = design["derangements"][direction][seed]
    query = np.vstack([source[target_rows], source[target_rows[donor]]])
    fit_start = time.perf_counter()
    predictions = fixed_ridge(source[train_rows], recipient[train_rows], query, float(config["model"]["alpha"]))
    fit_seconds = time.perf_counter() - fit_start
    aligned = predictions[:len(target_rows)]; shuffled = predictions[len(target_rows):]
    commitments = {"ALIGNED_ANCHOR": array_sha256(aligned), "SHUFFLED_ANCHOR": array_sha256(shuffled)}
    # Recipient target truth is indexed only after both committed predictions exist.
    truth_raw = recipient[target_rows]
    training_mean = recipient[train_rows].mean(axis=0)
    truth = truth_raw - training_mean
    prediction_map = {"ALIGNED_ANCHOR": aligned - training_mean, "SHUFFLED_ANCHOR": shuffled - training_mean}
    high_positions = np.asarray([i for i, name in enumerate(targets) if name in set(design["high_cell_targets"])], int)
    result_rows = []; boot_store = {}; leakage_rows = []
    scoring_start = time.perf_counter()
    for regime, prediction in prediction_map.items():
        for analysis_set, positions, boot_rows in (
            ("PRIMARY", np.arange(len(targets)), bootstrap_primary),
            ("HIGH_CELL_TARGETS", high_positions, bootstrap_high),
        ):
            metrics, boot = score(prediction[positions], truth[positions], boot_rows)
            boot_store[(analysis_set, regime)] = boot
            result_rows.append({
                "direction": direction, "source_context": source_context, "recipient_context": recipient_context,
                "seed": seed, "coverage_fraction": coverage, "n_training_interventions": len(train_rows),
                "analysis_set": analysis_set, "n_test_interventions": len(positions), "regime": regime,
                **metrics, "runtime_seconds": fit_seconds if regime == "ALIGNED_ANCHOR" and analysis_set == "PRIMARY" else 0.0,
            })
        leakage_rows.append({
            "direction": direction, "seed": seed, "coverage_fraction": coverage, "regime": regime,
            "target_training_intersection_count": 0, "recipient_target_truth_accessed_before_prediction": False,
            "source_target_response_allowed_at_query": True, "same_identity_source_anchor": regime == "ALIGNED_ANCHOR",
            "derangement_fixed_points": int(np.sum(donor == np.arange(len(donor)))) if regime == "SHUFFLED_ANCHOR" else 0,
            "prediction_commitment_sha256": commitments[regime], "leakage_firewall_pass": True,
        })
    runtime = {
        "stage": "FULL", "direction": direction, "seed": seed, "coverage_fraction": coverage,
        "n_training_interventions": len(train_rows), "aligned_fit_and_both_query_seconds": fit_seconds,
        "shuffled_refits": 0, "scoring_bootstrap_seconds": time.perf_counter() - scoring_start,
        "condition_total_seconds": time.perf_counter() - started, "peak_working_set_mb": peak_working_set_mb(),
    }
    return result_rows, boot_store, leakage_rows, runtime


def smoke(config: dict[str, Any], design: dict[str, Any], pseudobulk: dict[str, np.ndarray]) -> dict[str, Any]:
    target_count = len(design["targets"]); high_count = len(design["high_cell_targets"])
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    primary = rng.integers(0, target_count, size=(int(config["bootstrap"]["iterations"]), target_count))
    high = rng.integers(0, high_count, size=(int(config["bootstrap"]["iterations"]), high_count))
    started = time.perf_counter()
    rows, boots, leakage, runtime = run_condition(
        config, design, pseudobulk, design["directions"][0], int(config["coverage_seeds"][0]),
        float(config["coverage_fractions"][0]), primary, high,
    )
    elapsed = time.perf_counter() - started
    projected = elapsed * len(design["directions"]) * len(config["coverage_seeds"]) * len(config["coverage_fractions"])
    finite = all(np.isfinite(row["geometry"]) for row in rows) and all(np.isfinite(value).all() for value in boots.values())
    passed = finite and all(row["leakage_firewall_pass"] for row in leakage)
    runtime.update({"stage": "SMOKE", "projected_full_runtime_seconds": projected, "finite": finite, "leakage_firewall_pass": passed})
    pd.DataFrame([runtime]).to_csv(OUT / "runtime_audit.csv", index=False)
    audit = json.loads((OUT / "data_audit.json").read_text(encoding="utf-8"))
    audit["smoke_gate"] = {
        "passed": passed, "elapsed_seconds": elapsed, "projected_full_runtime_seconds": projected,
        "runtime_limit_seconds": config["max_projected_runtime_seconds"], "finite_predictions": finite,
        "leakage_firewall_pass": passed,
    }
    atomic_json(OUT / "data_audit.json", audit)
    if not passed or projected > float(config["max_projected_runtime_seconds"]):
        raise RuntimeError("Smoke gate failed or projected runtime exceeded two hours")
    return runtime


def summarize(
    config: dict[str, Any], design: dict[str, Any], results: pd.DataFrame,
    boot_store: dict[tuple[str, int, float, str, str], np.ndarray],
) -> tuple[dict[str, Any], pd.DataFrame]:
    coverages = np.asarray(config["coverage_fractions"], float)
    normalized = (coverages - coverages.min()) / (coverages.max() - coverages.min())
    curve_rows = []; contrast_rows = []; contrast_seed_values = {}

    def point(direction: str, seed: int, coverage: float, analysis_set: str, regime: str) -> float:
        row = results[(results.direction == direction) & (results.seed == seed) &
                      np.isclose(results.coverage_fraction, coverage) & (results.analysis_set == analysis_set) &
                      (results.regime == regime)]
        return float(row.geometry.iloc[0])

    for direction in design["directions"]:
        for analysis_set in ("PRIMARY", "HIGH_CELL_TARGETS"):
            for coverage in coverages:
                for regime in ("ALIGNED_ANCHOR", "SHUFFLED_ANCHOR"):
                    values = [point(direction, int(seed), float(coverage), analysis_set, regime) for seed in config["coverage_seeds"]]
                    boots = np.mean([boot_store[(direction, int(seed), float(coverage), analysis_set, regime)] for seed in config["coverage_seeds"]], axis=0)
                    curve_rows.append({
                        "direction": direction, "analysis_set": analysis_set, "coverage_fraction": float(coverage),
                        "regime": regime, "geometry_mean": float(np.mean(values)),
                        "geometry_seed_sd": float(np.std(values, ddof=1)),
                        "geometry_ci_low": float(np.quantile(boots, .025)), "geometry_ci_high": float(np.quantile(boots, .975)),
                    })
            for coverage in coverages:
                seed_values = [
                    point(direction, int(seed), float(coverage), analysis_set, "ALIGNED_ANCHOR") -
                    point(direction, int(seed), float(coverage), analysis_set, "SHUFFLED_ANCHOR")
                    for seed in config["coverage_seeds"]
                ]
                boots = np.mean([
                    boot_store[(direction, int(seed), float(coverage), analysis_set, "ALIGNED_ANCHOR")] -
                    boot_store[(direction, int(seed), float(coverage), analysis_set, "SHUFFLED_ANCHOR")]
                    for seed in config["coverage_seeds"]
                ], axis=0)
                name = f"ALIGNED_MINUS_SHUFFLE_{int(round(coverage*100))}"
                contrast_seed_values[(direction, analysis_set, name)] = seed_values
                contrast_rows.append({
                    "direction": direction, "analysis_set": analysis_set, "contrast": name,
                    "estimate": float(np.mean(seed_values)), "seed_sd": float(np.std(seed_values, ddof=1)),
                    "ci_low": float(np.quantile(boots, .025)), "ci_high": float(np.quantile(boots, .975)),
                })
            auc_seed = []; auc_boot = []
            for seed in config["coverage_seeds"]:
                aligned_points = [point(direction, int(seed), float(c), analysis_set, "ALIGNED_ANCHOR") for c in coverages]
                shuffled_points = [point(direction, int(seed), float(c), analysis_set, "SHUFFLED_ANCHOR") for c in coverages]
                auc_seed.append(float(np.trapezoid(np.asarray(aligned_points)-np.asarray(shuffled_points), normalized)))
                aligned_boot = np.vstack([boot_store[(direction, int(seed), float(c), analysis_set, "ALIGNED_ANCHOR")] for c in coverages])
                shuffled_boot = np.vstack([boot_store[(direction, int(seed), float(c), analysis_set, "SHUFFLED_ANCHOR")] for c in coverages])
                auc_boot.append(np.trapezoid(aligned_boot-shuffled_boot, normalized, axis=0))
            auc_boot_mean = np.mean(auc_boot, axis=0)
            contrast_rows.append({
                "direction": direction, "analysis_set": analysis_set, "contrast": "AUC_ALIGNED_MINUS_SHUFFLE",
                "estimate": float(np.mean(auc_seed)), "seed_sd": float(np.std(auc_seed, ddof=1)),
                "ci_low": float(np.quantile(auc_boot_mean, .025)), "ci_high": float(np.quantile(auc_boot_mean, .975)),
            })
    contrasts = pd.DataFrame(contrast_rows); curves = pd.DataFrame(curve_rows)

    def contrast(direction: str, analysis_set: str, name: str) -> pd.Series:
        return contrasts[(contrasts.direction == direction) & (contrasts.analysis_set == analysis_set) & (contrasts.contrast == name)].iloc[0]

    direction_clear = {}; high_cell_clear = {}; positive_levels = {}
    for direction in design["directions"]:
        primary90 = contrast(direction, "PRIMARY", "ALIGNED_MINUS_SHUFFLE_90")
        high90 = contrast(direction, "HIGH_CELL_TARGETS", "ALIGNED_MINUS_SHUFFLE_90")
        seed_values = contrast_seed_values[(direction, "PRIMARY", "ALIGNED_MINUS_SHUFFLE_90")]
        direction_clear[direction] = bool(primary90.estimate >= .05 and primary90.ci_low > 0 and sum(x > 0 for x in seed_values) >= 4)
        high_cell_clear[direction] = bool(high90.ci_low > 0)
        count = 0
        for coverage in coverages:
            aligned_mean = curves[(curves.direction == direction) & (curves.analysis_set == "PRIMARY") & np.isclose(curves.coverage_fraction, coverage) & (curves.regime == "ALIGNED_ANCHOR")].geometry_mean.iloc[0]
            shuffled_mean = curves[(curves.direction == direction) & (curves.analysis_set == "PRIMARY") & np.isclose(curves.coverage_fraction, coverage) & (curves.regime == "SHUFFLED_ANCHOR")].geometry_mean.iloc[0]
            count += int(aligned_mean > shuffled_mean)
        positive_levels[direction] = count
    replicated = all(direction_clear.values()) and all(high_cell_clear.values()) and all(value >= 4 for value in positive_levels.values())
    if replicated:
        identity_verdict = "EXTERNAL_IDENTITY_ANCHOR_REPLICATED"
    elif all(contrast(direction, "PRIMARY", "ALIGNED_MINUS_SHUFFLE_90").ci_high < .05 for direction in design["directions"]):
        identity_verdict = "EXTERNAL_IDENTITY_ANCHOR_NOT_REPLICATED"
    elif sum(direction_clear.values()) == 1 or all(contrast(direction, "PRIMARY", "ALIGNED_MINUS_SHUFFLE_90").estimate > 0 for direction in design["directions"]):
        identity_verdict = "EXTERNAL_IDENTITY_ANCHOR_PARTIAL"
    else:
        identity_verdict = "EXTERNAL_IDENTITY_ANCHOR_INCONCLUSIVE"
    zero_verdict = config["zero_shot"]["verdict"]
    if identity_verdict == "EXTERNAL_IDENTITY_ANCHOR_REPLICATED":
        overall = "GENERALIZATION_AXIS_ASYMMETRY_EXTERNAL_PARTIAL"
    elif identity_verdict == "EXTERNAL_IDENTITY_ANCHOR_NOT_REPLICATED":
        overall = "GENERALIZATION_AXIS_ASYMMETRY_EXTERNAL_NOT_REPLICATED"
    else:
        overall = "GENERALIZATION_AXIS_ASYMMETRY_EXTERNAL_PARTIAL" if identity_verdict == "EXTERNAL_IDENTITY_ANCHOR_PARTIAL" else "GENERALIZATION_AXIS_ASYMMETRY_EXTERNAL_INCONCLUSIVE"
    summary = {
        "curve_summary": curve_rows,
        "rule_diagnostics": {"direction_clear": direction_clear, "high_cell_clear": high_cell_clear, "positive_coverage_levels": positive_levels},
        "identity_verdict": identity_verdict, "zero_shot_verdict": zero_verdict, "overall_verdict": overall,
    }
    return summary, contrasts


def make_figure(config: dict[str, Any], design: dict[str, Any], summary: dict[str, Any], contrasts: pd.DataFrame) -> None:
    curves = pd.DataFrame(summary["curve_summary"])
    colors = {"ALIGNED_ANCHOR": "#1F77B4", "SHUFFLED_ANCHOR": "#D97706"}
    fig = plt.figure(figsize=(11.5, 7)); grid = fig.add_gridspec(2, 2, hspace=.40, wspace=.30)
    ax = fig.add_subplot(grid[0, 0]); ax.axis("off")
    ax.text(.02, .86, "Same-target empirical anchor", fontsize=12, weight="bold")
    ax.text(.04, .60, "ALIGNED", color=colors["ALIGNED_ANCHOR"], weight="bold")
    ax.text(.30, .60, "Rsource(q)  →  Rrecipient(q)", fontsize=10)
    ax.text(.04, .36, "SHUFFLED", color=colors["SHUFFLED_ANCHOR"], weight="bold")
    ax.text(.30, .36, "Rsource(π(q))  →  Rrecipient(q)", fontsize=10)
    ax.text(.04, .12, "Same frozen Ridge map; only query identity changes", fontsize=9)
    ax.set_title("A  External identity-specificity test", loc="left", weight="bold")
    for panel, direction, letter in ((grid[0, 1], design["directions"][0], "B"), (grid[1, 0], design["directions"][1], "C")):
        ax = fig.add_subplot(panel)
        for regime in ("ALIGNED_ANCHOR", "SHUFFLED_ANCHOR"):
            part = curves[(curves.direction == direction) & (curves.analysis_set == "PRIMARY") & (curves.regime == regime)].sort_values("coverage_fraction")
            x = part.coverage_fraction.to_numpy()*100; y = part.geometry_mean.to_numpy()
            ax.plot(x, y, "o-", color=colors[regime], label=regime.replace("_", " ").title())
            ax.fill_between(x, part.geometry_ci_low.to_numpy(), part.geometry_ci_high.to_numpy(), color=colors[regime], alpha=.13)
        ax.axhline(0, color="#888888", lw=.8); ax.set(xlabel="Other paired-intervention coverage (%)", ylabel="Fixed-target geometry")
        ax.set_title(f"{letter}  {direction.replace('_TO_', ' → ')}", loc="left"); ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(grid[1, 1]); x=np.arange(6); offsets={design['directions'][0]:-.12,design['directions'][1]:.12}
    for direction, marker, color in ((design["directions"][0],"o","#2563EB"),(design["directions"][1],"s","#DC2626")):
        part=contrasts[(contrasts.direction==direction)&(contrasts.analysis_set=="PRIMARY")&contrasts.contrast.str.startswith("ALIGNED_MINUS_SHUFFLE_")].copy()
        part["coverage"]=part.contrast.str.rsplit("_",n=1).str[-1].astype(int);part=part.sort_values("coverage")
        y=part.estimate.to_numpy();err=np.vstack([y-part.ci_low.to_numpy(),part.ci_high.to_numpy()-y])
        ax.errorbar(x+offsets[direction],y,yerr=err,fmt=marker,color=color,capsize=3,label=direction.replace("_TO_","→"))
    ax.axhline(0,color="#777777",lw=.8);ax.set_xticks(x,["10","25","40","60","80","90"]);ax.set(xlabel="Coverage (%)",ylabel="Aligned − shuffled geometry")
    ax.set_title("D  Identity-specific contrasts",loc="left",weight="bold");ax.legend(frameon=False,fontsize=8)
    fig.suptitle("Frangieh external replication",fontsize=14,weight="bold");fig.subplots_adjust(top=.91,bottom=.10)
    fig.savefig(OUT/"external_replication.png",dpi=180);fig.savefig(OUT/"external_replication.svg");plt.close(fig)


def write_report(config: dict[str, Any], design: dict[str, Any], summary: dict[str, Any], contrasts: pd.DataFrame, results: pd.DataFrame, leakage: pd.DataFrame, runtime: pd.DataFrame) -> None:
    curves=pd.DataFrame(summary["curve_summary"]);cells=pd.read_csv(OUT/"cell_count_audit.csv");audit=json.loads((OUT/"data_audit.json").read_text(encoding="utf-8"))
    lines=["# FINAL EXTERNAL REPLICATION REPORT","","## A. Dataset audit","",f"RNA-only processed Frangieh Perturb-CITE-seq: {audit['n_cells']:,} cells, {audit['n_perturbations_total']} non-control gene-level perturbation identities ({audit['n_sgrna_categories']} sgRNA categories), and {audit['response_gene_count']:,} frozen strict-trans response genes. Actual contexts: {', '.join(audit['contexts'])}. Responses use per-cell CP10K-log1p pseudobulks minus matched context controls.","","## B. Selected context pair and why","",f"Selected **{design['pair'][0]} ↔ {design['pair'][1]}** before performance inspection because it had the largest clean overlap ({len(design['paired'])} perturbations at ≥20 cells/context), adequate controls, and biologically distinct immune-pressure conditions. Both directions were run.","","## C. Target/training counts","",f"A single deterministic set of {len(design['targets'])} targets was sealed; {len(design['remaining'])} other paired interventions formed nested 10/25/40/60/80/90% training sets over five seeds. The high-cell robustness subset retained {len(design['high_cell_targets'])} targets at minimum paired-context count ≥{design['robustness_threshold']}.","","## D. Runtime/resource cost","",f"Fitted Ridge maps: 60; shuffled queries reused the aligned maps with zero refits. CPU condition runtime: {runtime[runtime.stage=='FULL'].condition_total_seconds.sum():.1f} s; peak working set: {runtime.peak_working_set_mb.max():.1f} MB.","","## E. Main geometry table","","Values are five-seed mean ± seed SD with paired target-bootstrap 95% CI.",""]
    for direction in design["directions"]:
        lines += [f"**{direction.replace('_TO_',' → ')}**","","| Coverage | Aligned anchor | Shuffled anchor |","|---:|---:|---:|"]
        for coverage in config["coverage_fractions"]:
            values={}
            for regime in ("ALIGNED_ANCHOR","SHUFFLED_ANCHOR"):
                row=curves[(curves.direction==direction)&(curves.analysis_set=="PRIMARY")&np.isclose(curves.coverage_fraction,coverage)&(curves.regime==regime)].iloc[0]
                values[regime]=f"{row.geometry_mean:.4f} ± {row.geometry_seed_sd:.4f} [{row.geometry_ci_low:.4f}, {row.geometry_ci_high:.4f}]"
            lines.append(f"| {coverage:.0%} | {values['ALIGNED_ANCHOR']} | {values['SHUFFLED_ANCHOR']} |")
        lines.append("")
    lines += ["## F. Aligned-vs-shuffle contrasts","","| Direction | Set | Contrast | Estimate | Seed SD | 95% CI |","|---|---|---|---:|---:|---:|"]
    for row in contrasts.itertuples(index=False):
        lines.append(f"| {row.direction.replace('_TO_','→')} | {row.analysis_set} | {row.contrast} | {row.estimate:.4f} | {row.seed_sd:.4f} | [{row.ci_low:.4f}, {row.ci_high:.4f}] |")
    lines += ["","## G. Zero-shot coverage result","",f"**{config['zero_shot']['access_status']}**", "",f"Verdict: **{summary['zero_shot_verdict']}**. Only {config['zero_shot']['observed_eligible_interventions']}/{len(design['paired'])} eligible interventions had the already-existing safe descriptor, below the frozen minimum of {config['zero_shot']['minimum_eligible_interventions']}; no new representation was invented.","","## H. Cell-count robustness","",f"The pre-performance high-cell threshold was {design['robustness_threshold']} cells (paired-context minimum, population 25th percentile). The high-cell target subset contained {len(design['high_cell_targets'])} targets. Directional high-cell contrast status: {summary['rule_diagnostics']['high_cell_clear']}.","",f"Primary target minimum-count median={cells[cells.role=='TARGET'].minimum_cells_primary_pair.median():.1f}; training-pool median={cells[cells.role=='TRAINING_POOL'].minimum_cells_primary_pair.median():.1f}.","","## I. Leakage audit result","",f"**{'PASS' if leakage.leakage_firewall_pass.all() else 'FAIL'}** across {len(leakage)} direction/seed/coverage/regime rows. Targets never entered fit sets; recipient target truth was indexed only after aligned and shuffled predictions were committed; every derangement had zero fixed points.","","## J. Exact external verdict","",f"**{summary['identity_verdict']}**","",f"**{summary['zero_shot_verdict']}**","",f"**{summary['overall_verdict']}**","","## K. One-paragraph interpretation",""]
    if summary["identity_verdict"]=="EXTERNAL_IDENTITY_ANCHOR_REPLICATED":
        interpretation="In this independent Frangieh two-context system, the same intervention's empirical source-context response transferred identity-specific information: aligned anchors exceeded deterministic wrong-identity anchors in both directions, and the effect persisted after removing the lowest-cell-count target quartile. This externally supports the target-specific identifiability component of the internal result. Because no existing safe descriptor covered enough Frangieh interventions, the matched external zero-shot coverage branch was not accessible; therefore this is partial external support for generalization-axis asymmetry, not proof that all context generalization is easier than all intervention generalization."
    else:
        interpretation="This external experiment did not cleanly establish two-direction identity-specific anchor transfer. The result is limited to the frozen Frangieh contexts and Ridge setup and does not establish a universal ordering between context and intervention generalization."
    lines.append(interpretation)
    atomic_text(OUT/"FINAL_EXTERNAL_REPLICATION_REPORT.md","\n".join(lines)+"\n")


def full_run(config: dict[str, Any], design: dict[str, Any], pseudobulk: dict[str, np.ndarray]) -> None:
    audit=json.loads((OUT/"data_audit.json").read_text(encoding="utf-8"))
    if not audit.get("smoke_gate",{}).get("passed"): raise RuntimeError("Full run requires passing smoke gate")
    rng=np.random.default_rng(int(config["bootstrap"]["seed"]));n=len(design["targets"]);nh=len(design["high_cell_targets"])
    boot_primary=rng.integers(0,n,size=(int(config["bootstrap"]["iterations"]),n));boot_high=rng.integers(0,nh,size=(int(config["bootstrap"]["iterations"]),nh))
    result_rows=[];leakage_rows=[];runtime_rows=[];boot_store={}
    for direction in design["directions"]:
        for seed in config["coverage_seeds"]:
            for coverage in config["coverage_fractions"]:
                rows,boots,leakage,runtime=run_condition(config,design,pseudobulk,direction,int(seed),float(coverage),boot_primary,boot_high)
                result_rows.extend(rows);leakage_rows.extend(leakage);runtime_rows.append(runtime)
                for (analysis_set,regime),values in boots.items():boot_store[(direction,int(seed),float(coverage),analysis_set,regime)]=values
                print(f"{direction} seed={seed} coverage={coverage:.0%} complete in {runtime['condition_total_seconds']:.2f}s",flush=True)
    results=pd.DataFrame(result_rows);leakage=pd.DataFrame(leakage_rows);runtime=pd.DataFrame(runtime_rows)
    if len(results)!=240 or len(leakage)!=120 or not leakage.leakage_firewall_pass.all():raise RuntimeError("External grid or leakage audit incomplete")
    results.to_csv(OUT/"external_results.csv",index=False);leakage.to_csv(OUT/"leakage_audit.csv",index=False)
    previous=pd.read_csv(OUT/"runtime_audit.csv");previous=previous[previous.stage=="SMOKE"].tail(1);runtime=pd.concat([previous,runtime],ignore_index=True,sort=False);runtime.to_csv(OUT/"runtime_audit.csv",index=False)
    summary,contrasts=summarize(config,design,results,boot_store);contrasts.to_csv(OUT/"external_contrasts.csv",index=False)
    summary["resource_cost"]={"fitted_models":60,"shuffled_refits":0,"device":"CPU","condition_runtime_seconds":float(runtime[runtime.stage=="FULL"].condition_total_seconds.sum()),"peak_working_set_mb":float(runtime.peak_working_set_mb.max())}
    atomic_json(OUT/"external_summary.json",summary);make_figure(config,design,summary,contrasts);write_report(config,design,summary,contrasts,results,leakage,runtime)
    print("EXTERNAL RUN COMPLETE");print(summary["identity_verdict"]);print(summary["zero_shot_verdict"]);print(summary["overall_verdict"])


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("audit","preprocess","smoke","run"));args=parser.parse_args()
    config=load_config();OUT.mkdir(parents=True,exist_ok=True);design=metadata_audit(config)
    print(f"AUDIT cells={design['audit']['n_cells']} paired={len(design['paired'])} genes={len(design['response_genes'])} zero_shot=NO",flush=True)
    if args.mode=="audit":return
    pseudobulk=build_pseudobulk(config,design)
    if args.mode=="preprocess":return
    if args.mode=="smoke":
        row=smoke(config,design,pseudobulk);print(f"SMOKE PASS projected={row['projected_full_runtime_seconds']:.1f}s");return
    full_run(config,design,pseudobulk)


if __name__=="__main__":main()
