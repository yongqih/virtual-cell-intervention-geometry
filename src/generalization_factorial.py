from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any
from ctypes import wintypes

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/generalization_factorial.json"
OUT = ROOT / "results/generalization_factorial"
RPE1_PATH = ROOT / "results/cross_dataset_replication_rpe1/cache/rpe1_pseudobulk_full.npz"
K562_PATH = ROOT / "results/directedT_exploration/development_data.npz"
DESCRIPTOR_PATH = ROOT / "results/cross_dataset_replication_rpe1/cache/control_state_representations.npz"
SPLIT_PATH = ROOT / "results/cross_dataset_replication_rpe1/split_definition.json"
PILOT_CODE = ROOT / "analysis/minimum_generalization_pilot.py"
PILOT_DESIGN = ROOT / "results/minimum_generalization_pilot/pilot_design.json"
PILOT_RESOURCE = ROOT / "results/minimum_generalization_pilot/resource_cost.json"
GEOMETRY_CODE = ROOT / "scripts/main_geometry_integrity_audit/run_audit.py"
RIDGE_CODE = ROOT / "scripts/propagation_reproduction/common.py"
RAW_K562 = ROOT / "data/raw/replogle22k562_processed_complete.valid.h5ad"
RAW_RPE1 = ROOT / "data/raw/replogle22rpe1_processed_complete.valid.h5ad"


def load_utilities():
    spec = importlib.util.spec_from_file_location("frozen_geometry", GEOMETRY_CODE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.path.insert(0, str(RIDGE_CODE.parent))
    from common import ridge_fit_predict

    return module.geometry, module.row_pearson, module.response_distances, module.safe_corr, ridge_fit_predict


GEOMETRY, ROW_PEARSON, RESPONSE_DISTANCES, SAFE_CORR, RIDGE_PREDICT = load_utilities()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def peak_working_set_mb() -> float:
    if platform.system() != "Windows":
        return float("nan")

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return float(counters.PeakWorkingSetSize / 2**20) if ok else float("nan")


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_data() -> dict[str, Any]:
    with np.load(RPE1_PATH, allow_pickle=True) as archive:
        rpe1 = {key: archive[key].copy() for key in archive.files}
    with np.load(K562_PATH, allow_pickle=True) as archive:
        k562 = {key: archive[key].copy() for key in archive.files}
    with np.load(DESCRIPTOR_PATH, allow_pickle=True) as archive:
        descriptor = {key: archive[key].copy() for key in archive.files}
    return {"RPE1": rpe1, "K562": k562, "descriptor": descriptor}


def eligibility(data: dict[str, Any]) -> dict[str, Any]:
    r_names = data["RPE1"]["perturbations"].astype(str)
    k_names = data["K562"]["perturbations"].astype(str)
    d_names = data["descriptor"]["perturbations"].astype(str)
    paired = sorted(set(r_names) & set(k_names) & set(d_names))
    r_genes = data["RPE1"]["genes"].astype(str)
    k_genes = data["K562"]["genes"].astype(str)
    common_genes = sorted((set(r_genes) & set(k_genes)) - set(paired))
    fold = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["folds"][0]
    fold0_shared = sorted(set(fold["validation_sources"]) & set(paired))
    return {"paired": paired, "common_genes": common_genes, "fold0_shared": fold0_shared}


def write_repo_audit(config: dict[str, Any], data: dict[str, Any], eligible: dict[str, Any]) -> dict[str, Any]:
    pilot_runtime = None
    if PILOT_RESOURCE.exists():
        pilot_runtime = json.loads(PILOT_RESOURCE.read_text(encoding="utf-8")).get("total_runtime_seconds")
    audit = {
        "k562_response_cache": str(K562_PATH.relative_to(ROOT)),
        "rpe1_response_cache": str(RPE1_PATH.relative_to(ROOT)),
        "context_local_control_means": {
            "K562": f"{K562_PATH.relative_to(ROOT)}::control_mean",
            "RPE1": f"{RPE1_PATH.relative_to(ROOT)}::control_mean",
        },
        "shared_intervention_count": len(eligible["paired"]),
        "shared_response_gene_count": len(eligible["common_genes"]),
        "safe_source_descriptor": f"{DESCRIPTOR_PATH.relative_to(ROOT)}::EstablishedOBS71 (control-only; no perturbation-response labels)",
        "existing_ridge_alpha": config["model"]["alpha"],
        "existing_ridge_code": str(RIDGE_CODE.relative_to(ROOT)),
        "existing_geometry_code": str(GEOMETRY_CODE.relative_to(ROOT)),
        "existing_bootstrap_code": str(PILOT_CODE.relative_to(ROOT)),
        "existing_fold0_targets": str(SPLIT_PATH.relative_to(ROOT)) + "::folds[0].validation_sources",
        "fold0_shared_eligible_count": len(eligible["fold0_shared"]),
        "response_gene_panels": {
            "RPE1": "cached fold-specific strict-trans panels plus deterministic K562/RPE1 intersection",
            "K562": "cached development_data genes with manuscript strict-trans target exclusion",
            "factorial": "deterministic cached gene intersection minus all paired perturbation targets",
        },
        "cell_count_control_accessible": False,
        "cell_count_control_reason": "RPE1 cache has per-source counts, but paired K562 source-level cache has none. Raw H5ADs exist, but equal-budget pseudobulk reconstruction would require a new multi-gigabyte cell-level pipeline prohibited by the task.",
        "raw_cell_files_present": {"K562": RAW_K562.exists(), "RPE1": RAW_RPE1.exists()},
        "estimated_runtime": "approximately 5-15 CPU minutes, subject to mandatory smoke projection",
        "pilot_reference_runtime_seconds": pilot_runtime,
        "large_model_training_required": False,
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (K562_PATH, RPE1_PATH, DESCRIPTOR_PATH, SPLIT_PATH, CONFIG_PATH, GEOMETRY_CODE, RIDGE_CODE)
        },
    }
    if (OUT / "repo_audit.json").exists():
        previous = json.loads((OUT / "repo_audit.json").read_text(encoding="utf-8"))
        same_config = previous.get("input_hashes", {}).get(str(CONFIG_PATH.relative_to(ROOT))) == sha256(CONFIG_PATH)
        if same_config and "smoke_gate" in previous:
            audit["smoke_gate"] = previous["smoke_gate"]
    atomic_json(OUT / "repo_audit.json", audit)
    return audit


def select_targets(config: dict[str, Any], eligible: dict[str, Any]) -> tuple[list[str], list[str], str]:
    rule = config["target_selection"]
    if len(eligible["fold0_shared"]) >= int(rule["minimum_fold0_shared_targets"]):
        targets = list(eligible["fold0_shared"])
        reason = "existing RPE1 fold-0 targets met the frozen minimum"
    else:
        rng = np.random.default_rng(int(rule["fallback_seed"]))
        order = rng.permutation(len(eligible["paired"]))
        targets = sorted(eligible["paired"][i] for i in order[: int(rule["target_count"])])
        reason = (
            f"existing fold 0 had only {len(eligible['fold0_shared'])} shared eligible targets; "
            f"selected {len(targets)} once from sorted paired universe with seed {rule['fallback_seed']}"
        )
    remaining = sorted(set(eligible["paired"]) - set(targets))
    return targets, remaining, reason


def derangement(size: int, seed: int) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(size)
    donor = np.empty(size, dtype=int)
    donor[order] = np.roll(order, 1)
    if np.any(donor == np.arange(size)):
        raise RuntimeError("Derangement construction failed")
    return donor


def prepare_design(config: dict[str, Any], data: dict[str, Any], eligible: dict[str, Any]) -> dict[str, Any]:
    targets, remaining, reason = select_targets(config, eligible)
    atomic_text(OUT / "common_response_genes.txt", "\n".join(eligible["common_genes"]) + "\n")
    atomic_text(OUT / "fixed_test_targets.txt", "\n".join(targets) + "\n")
    subsets: dict[int, dict[float, list[str]]] = {}
    set_dir = OUT / "training_identity_sets"
    set_dir.mkdir(parents=True, exist_ok=True)
    for seed in config["coverage_seeds"]:
        order = np.random.default_rng(int(seed)).permutation(len(remaining))
        nested: dict[float, list[str]] = {}
        serializable: dict[str, Any] = {"seed": seed, "remaining_universe_size": len(remaining), "coverage_sets": {}}
        for coverage in config["coverage_fractions"]:
            count = max(1, int(np.floor(float(coverage) * len(remaining))))
            names = [remaining[i] for i in order[:count]]
            nested[float(coverage)] = names
            serializable["coverage_sets"][f"{float(coverage):.2f}"] = {"n": count, "identities": names}
        subsets[int(seed)] = nested
        atomic_json(set_dir / f"seed_{seed}.json", serializable)

    derangements: dict[str, dict[int, np.ndarray]] = {}
    derangement_audit: dict[str, Any] = {}
    base = int(config["derangement_seed_base"])
    for direction_index, direction in enumerate(config["directions"]):
        derangements[direction] = {}
        derangement_audit[direction] = {}
        for seed in config["coverage_seeds"]:
            used_seed = base + direction_index * 1000 + int(seed)
            donor = derangement(len(targets), used_seed)
            derangements[direction][int(seed)] = donor
            derangement_audit[direction][str(seed)] = {
                "seed": used_seed,
                "mapping": {targets[i]: targets[int(donor[i])] for i in range(len(targets))},
                "fixed_points": int(np.sum(donor == np.arange(len(targets)))),
            }
    target_audit = {
        "selection_status": "FROZEN_BEFORE_SMOKE_OR_FULL_FITS",
        "paired_universe_count": len(eligible["paired"]),
        "paired_universe": eligible["paired"],
        "existing_fold0_shared_eligible_count": len(eligible["fold0_shared"]),
        "existing_fold0_shared_eligible": eligible["fold0_shared"],
        "selection_reason": reason,
        "fixed_target_count": len(targets),
        "fixed_targets": targets,
        "remaining_training_universe_count": len(remaining),
        "common_response_gene_count": len(eligible["common_genes"]),
        "target_training_intersection": sorted(set(targets) & set(remaining)),
        "config_sha256": sha256(CONFIG_PATH),
        "derangements": derangement_audit,
    }
    atomic_json(OUT / "target_selection_audit.json", target_audit)
    return {"targets": targets, "remaining": remaining, "subsets": subsets, "derangements": derangements}


def align_arrays(data: dict[str, Any], eligible: dict[str, Any]) -> dict[str, np.ndarray]:
    paired = eligible["paired"]
    genes = eligible["common_genes"]
    r_names = data["RPE1"]["perturbations"].astype(str)
    k_names = data["K562"]["perturbations"].astype(str)
    d_names = data["descriptor"]["perturbations"].astype(str)
    r_genes = data["RPE1"]["genes"].astype(str)
    k_genes = data["K562"]["genes"].astype(str)
    rpi, kpi, dpi = ({name: i for i, name in enumerate(values)} for values in (r_names, k_names, d_names))
    rgi, kgi = ({name: i for i, name in enumerate(values)} for values in (r_genes, k_genes))
    return {
        "RPE1": data["RPE1"]["delta"].astype(np.float32)[[rpi[x] for x in paired]][:, [rgi[x] for x in genes]],
        "K562": data["K562"]["delta"].astype(np.float32)[[kpi[x] for x in paired]][:, [kgi[x] for x in genes]],
        "descriptor": data["descriptor"]["EstablishedOBS71"].astype(np.float32)[[dpi[x] for x in paired]],
    }


def fixed_ridge(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, alpha: float) -> np.ndarray:
    scaler = StandardScaler().fit(train_x)
    return RIDGE_PREDICT(
        scaler.transform(train_x), train_y, scaler.transform(query_x), alpha, fit_intercept=True
    )


def bootstrap_geometry(prediction: np.ndarray, truth: np.ndarray, bootstrap_rows: np.ndarray) -> np.ndarray:
    pred_distance = squareform(RESPONSE_DISTANCES(prediction))
    truth_distance = squareform(RESPONSE_DISTANCES(truth))
    values = np.empty(len(bootstrap_rows), dtype=float)
    upper = np.triu_indices(bootstrap_rows.shape[1], 1)
    for index, rows in enumerate(bootstrap_rows):
        pred_values = pred_distance[np.ix_(rows, rows)][upper]
        truth_values = truth_distance[np.ix_(rows, rows)][upper]
        values[index] = SAFE_CORR(pred_values, truth_values)
    return values


def score_regime(prediction: np.ndarray, truth: np.ndarray, bootstrap_rows: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    row_corr = ROW_PEARSON(prediction, truth)
    truth_variance = float(np.mean(np.var(truth, axis=0)))
    point = float(GEOMETRY(prediction, truth))
    boot = bootstrap_geometry(prediction, truth, bootstrap_rows)
    metrics = {
        "geometry": point,
        "geometry_ci_low": float(np.quantile(boot, 0.025)),
        "geometry_ci_high": float(np.quantile(boot, 0.975)),
        "mean_response_pearson": float(np.mean(row_corr)),
        "median_response_pearson": float(np.median(row_corr)),
        "variance_retention": float(np.mean(np.var(prediction, axis=0)) / max(truth_variance, 1e-12)),
    }
    return metrics, boot


def run_condition(
    config: dict[str, Any],
    aligned: dict[str, np.ndarray],
    eligible: dict[str, Any],
    design: dict[str, Any],
    direction: str,
    seed: int,
    coverage: float,
    bootstrap_rows: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    condition_start = time.perf_counter()
    paired_index = {name: i for i, name in enumerate(eligible["paired"])}
    target_index = np.asarray([paired_index[name] for name in design["targets"]], dtype=int)
    train_names = design["subsets"][seed][coverage]
    train_index = np.asarray([paired_index[name] for name in train_names], dtype=int)
    if set(design["targets"]) & set(train_names):
        raise RuntimeError("Leakage firewall: held targets intersect training identities")
    source_name, recipient_name = direction.split("_TO_")
    source = aligned[source_name]
    recipient = aligned[recipient_name]
    descriptor = aligned["descriptor"]
    alpha = float(config["model"]["alpha"])

    zero_start = time.perf_counter()
    zero_prediction = fixed_ridge(descriptor[train_index], recipient[train_index], descriptor[target_index], alpha)
    zero_seconds = time.perf_counter() - zero_start
    donor = design["derangements"][direction][seed]
    anchor_start = time.perf_counter()
    anchor_queries = np.vstack([source[target_index], source[target_index[donor]]])
    anchor_predictions = fixed_ridge(source[train_index], recipient[train_index], anchor_queries, alpha)
    aligned_prediction = anchor_predictions[: len(target_index)]
    shuffled_prediction = anchor_predictions[len(target_index):]
    anchor_fit_seconds = time.perf_counter() - anchor_start
    shuffled_seconds = 0.0
    predictions = {
        "ZERO_SHOT": zero_prediction,
        "ALIGNED_ANCHOR": aligned_prediction,
        "SHUFFLED_ANCHOR": shuffled_prediction,
    }
    prediction_hashes = {regime: array_sha256(value) for regime, value in predictions.items()}

    # Recipient target truth is indexed only after all three predictions and their commitments exist.
    truth_raw = recipient[target_index]
    training_mean = recipient[train_index].mean(axis=0)
    truth = truth_raw - training_mean
    predictions = {regime: value - training_mean for regime, value in predictions.items()}
    scoring_start = time.perf_counter()
    result_rows: list[dict[str, Any]] = []
    boot: dict[str, np.ndarray] = {}
    leakage_rows: list[dict[str, Any]] = []
    runtime_by_regime = {
        "ZERO_SHOT": zero_seconds,
        "ALIGNED_ANCHOR": anchor_fit_seconds,
        "SHUFFLED_ANCHOR": shuffled_seconds,
    }
    for regime in config["regimes"]:
        metrics, boot_values = score_regime(predictions[regime], truth, bootstrap_rows)
        boot[regime] = boot_values
        result_rows.append({
            "direction": direction,
            "seed": seed,
            "coverage_fraction": coverage,
            "n_training_interventions": len(train_index),
            "regime": regime,
            **metrics,
            "runtime_seconds": runtime_by_regime[regime],
        })
        leakage_rows.append({
            "direction": direction,
            "seed": seed,
            "coverage_fraction": coverage,
            "regime": regime,
            "target_training_intersection_count": 0,
            "target_validation_intersection_count": 0,
            "recipient_target_truth_accessed_before_prediction": False,
            "target_response_used_as_zero_shot_input": False,
            "source_target_response_allowed_at_query": regime != "ZERO_SHOT",
            "same_identity_source_anchor": regime == "ALIGNED_ANCHOR",
            "derangement_fixed_points": int(np.sum(donor == np.arange(len(donor)))) if regime == "SHUFFLED_ANCHOR" else 0,
            "prediction_commitment_sha256": prediction_hashes[regime],
            "leakage_firewall_pass": True,
        })
    scoring_seconds = time.perf_counter() - scoring_start
    runtime = {
        "stage": "FULL",
        "direction": direction,
        "seed": seed,
        "coverage_fraction": coverage,
        "n_training_interventions": len(train_index),
        "zero_fit_predict_seconds": zero_seconds,
        "anchor_fit_aligned_predict_seconds": anchor_fit_seconds,
        "shuffle_query_refit_seconds": shuffled_seconds,
        "scoring_bootstrap_seconds": scoring_seconds,
        "condition_total_seconds": time.perf_counter() - condition_start,
        "peak_working_set_mb": peak_working_set_mb(),
    }
    return result_rows, boot, leakage_rows, runtime


def smoke(config: dict[str, Any], data: dict[str, Any], eligible: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    aligned = align_arrays(data, eligible)
    target_count = len(design["targets"])
    rows = np.random.default_rng(int(config["bootstrap"]["seed"])).integers(
        0, target_count, size=(int(config["bootstrap"]["iterations"]), target_count)
    )
    started = time.perf_counter()
    result, boot, leakage, runtime = run_condition(
        config, aligned, eligible, design, config["directions"][0], int(config["coverage_seeds"][0]),
        float(config["coverage_fractions"][0]), rows
    )
    elapsed = time.perf_counter() - started
    projected = elapsed * len(config["directions"]) * len(config["coverage_seeds"]) * len(config["coverage_fractions"])
    finite = all(np.isfinite(row["geometry"]) for row in result) and all(np.isfinite(values).all() for values in boot.values())
    passed = finite and all(row["leakage_firewall_pass"] for row in leakage)
    smoke_row = {
        **runtime,
        "stage": "SMOKE",
        "projected_full_runtime_seconds": projected,
        "predictions_and_metrics_finite": finite,
        "leakage_firewall_pass": passed,
    }
    pd.DataFrame([smoke_row]).to_csv(OUT / "runtime_audit.csv", index=False)
    audit = json.loads((OUT / "repo_audit.json").read_text(encoding="utf-8"))
    audit["smoke_gate"] = {
        "passed": passed,
        "elapsed_seconds": elapsed,
        "projected_full_runtime_seconds": projected,
        "runtime_limit_seconds": config["max_projected_runtime_seconds"],
        "finite_predictions": finite,
        "leakage_firewall_pass": all(row["leakage_firewall_pass"] for row in leakage),
        "geometry_values": {row["regime"]: row["geometry"] for row in result},
    }
    atomic_json(OUT / "repo_audit.json", audit)
    if not passed:
        raise RuntimeError("Smoke gate failed")
    if projected > float(config["max_projected_runtime_seconds"]):
        raise RuntimeError(f"Projected runtime {projected:.1f}s exceeds frozen limit")
    return smoke_row


def ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def summarize(
    config: dict[str, Any], results: pd.DataFrame, boot_store: dict[tuple[str, int, float, str], np.ndarray]
) -> tuple[dict[str, Any], pd.DataFrame]:
    coverages = np.asarray(config["coverage_fractions"], dtype=float)
    normalized_coverage = (coverages - coverages.min()) / (coverages.max() - coverages.min())
    curve_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    contrast_seed_values: dict[tuple[str, str], list[float]] = {}

    def point(direction: str, seed: int, coverage: float, regime: str) -> float:
        row = results[(results.direction == direction) & (results.seed == seed) &
                      np.isclose(results.coverage_fraction, coverage) & (results.regime == regime)]
        return float(row.geometry.iloc[0])

    def add_contrast(direction: str, name: str, seed_points: list[float], boot_values: np.ndarray) -> None:
        low, high = ci(boot_values)
        contrast_seed_values[(direction, name)] = seed_points
        contrast_rows.append({
            "direction": direction,
            "contrast": name,
            "estimate": float(np.mean(seed_points)),
            "ci_low": low,
            "ci_high": high,
        })

    for direction in config["directions"]:
        for coverage in coverages:
            for regime in config["regimes"]:
                points = [point(direction, seed, float(coverage), regime) for seed in config["coverage_seeds"]]
                boots = np.mean(
                    [boot_store[(direction, int(seed), float(coverage), regime)] for seed in config["coverage_seeds"]], axis=0
                )
                low, high = ci(boots)
                curve_rows.append({
                    "direction": direction,
                    "coverage_fraction": float(coverage),
                    "regime": regime,
                    "geometry_mean": float(np.mean(points)),
                    "geometry_seed_sd": float(np.std(points, ddof=1)),
                    "geometry_ci_low": low,
                    "geometry_ci_high": high,
                })

        seeds = [int(seed) for seed in config["coverage_seeds"]]
        p = lambda seed, cov, regime: point(direction, seed, cov, regime)
        b = lambda seed, cov, regime: boot_store[(direction, seed, cov, regime)]
        contrast_specs = [
            ("ZERO_90_MINUS_ZERO_10", lambda s: p(s, .90, "ZERO_SHOT") - p(s, .10, "ZERO_SHOT"),
             lambda s: b(s, .90, "ZERO_SHOT") - b(s, .10, "ZERO_SHOT")),
            ("ANCHOR_90_MINUS_ZERO_90", lambda s: p(s, .90, "ALIGNED_ANCHOR") - p(s, .90, "ZERO_SHOT"),
             lambda s: b(s, .90, "ALIGNED_ANCHOR") - b(s, .90, "ZERO_SHOT")),
            ("ANCHOR_10_MINUS_ZERO_90", lambda s: p(s, .10, "ALIGNED_ANCHOR") - p(s, .90, "ZERO_SHOT"),
             lambda s: b(s, .10, "ALIGNED_ANCHOR") - b(s, .90, "ZERO_SHOT")),
        ]
        for coverage in coverages:
            label = f"ANCHOR_MINUS_SHUFFLE_{int(round(coverage * 100))}"
            contrast_specs.append((
                label,
                lambda s, c=float(coverage): p(s, c, "ALIGNED_ANCHOR") - p(s, c, "SHUFFLED_ANCHOR"),
                lambda s, c=float(coverage): b(s, c, "ALIGNED_ANCHOR") - b(s, c, "SHUFFLED_ANCHOR"),
            ))
        for name, point_function, boot_function in contrast_specs:
            seed_points = [float(point_function(seed)) for seed in seeds]
            boots = np.mean([boot_function(seed) for seed in seeds], axis=0)
            add_contrast(direction, name, seed_points, boots)

        auc_point: dict[str, list[float]] = {regime: [] for regime in config["regimes"]}
        auc_boot: dict[str, list[np.ndarray]] = {regime: [] for regime in config["regimes"]}
        for seed in seeds:
            for regime in config["regimes"]:
                y = np.asarray([p(seed, float(c), regime) for c in coverages])
                yb = np.vstack([b(seed, float(c), regime) for c in coverages])
                auc_point[regime].append(float(np.trapezoid(y, normalized_coverage)))
                auc_boot[regime].append(np.trapezoid(yb, normalized_coverage, axis=0))
        for name, left, right in (
            ("AUC_ANCHOR_MINUS_ZERO", "ALIGNED_ANCHOR", "ZERO_SHOT"),
            ("AUC_ANCHOR_MINUS_SHUFFLE", "ALIGNED_ANCHOR", "SHUFFLED_ANCHOR"),
        ):
            seed_points = (np.asarray(auc_point[left]) - np.asarray(auc_point[right])).tolist()
            boots = np.mean(np.vstack(auc_boot[left]) - np.vstack(auc_boot[right]), axis=0)
            add_contrast(direction, name, seed_points, boots)

    contrasts = pd.DataFrame(contrast_rows)
    curves = pd.DataFrame(curve_rows)

    def contrast(direction: str, name: str) -> pd.Series:
        return contrasts[(contrasts.direction == direction) & (contrasts.contrast == name)].iloc[0]

    directions = config["directions"]
    coverage_strong_by_direction = {}
    coverage_weak_by_direction = {}
    anchor_clear_by_direction = {}
    aligned_zero_clear = {}
    coverage_pair_counts = {}
    for direction in directions:
        c1 = contrast(direction, "ZERO_90_MINUS_ZERO_10")
        c2 = contrast(direction, "ANCHOR_90_MINUS_ZERO_90")
        c4 = contrast(direction, "ANCHOR_MINUS_SHUFFLE_90")
        c1_seeds = contrast_seed_values[(direction, "ZERO_90_MINUS_ZERO_10")]
        c2_seeds = contrast_seed_values[(direction, "ANCHOR_90_MINUS_ZERO_90")]
        c4_seeds = contrast_seed_values[(direction, "ANCHOR_MINUS_SHUFFLE_90")]
        coverage_strong_by_direction[direction] = bool(c1.estimate >= .10 and c1.ci_low > 0 and sum(x > 0 for x in c1_seeds) >= 4)
        coverage_weak_by_direction[direction] = bool(c1.estimate < .05 and c1.ci_high < .10)
        aligned_zero_clear[direction] = bool(c2.estimate >= .05 and c2.ci_low > 0 and sum(x > 0 for x in c2_seeds) >= 4)
        clear = bool(aligned_zero_clear[direction] and c4.estimate >= .05 and c4.ci_low > 0 and sum(x > 0 for x in c4_seeds) >= 4)
        mean_positive_levels = 0
        for coverage in coverages:
            a = curves[(curves.direction == direction) & np.isclose(curves.coverage_fraction, coverage) & (curves.regime == "ALIGNED_ANCHOR")].geometry_mean.iloc[0]
            z = curves[(curves.direction == direction) & np.isclose(curves.coverage_fraction, coverage) & (curves.regime == "ZERO_SHOT")].geometry_mean.iloc[0]
            s = curves[(curves.direction == direction) & np.isclose(curves.coverage_fraction, coverage) & (curves.regime == "SHUFFLED_ANCHOR")].geometry_mean.iloc[0]
            mean_positive_levels += int(a > z and a > s)
        coverage_pair_counts[direction] = mean_positive_levels
        anchor_clear_by_direction[direction] = bool(clear and mean_positive_levels >= 4)

    if all(coverage_strong_by_direction.values()):
        coverage_verdict = "OTHER_INTERVENTION_COVERAGE_STRONG_RESCUE"
    elif all(coverage_weak_by_direction.values()):
        coverage_verdict = "OTHER_INTERVENTION_COVERAGE_WEAK_RESCUE"
    else:
        coverage_verdict = "OTHER_INTERVENTION_COVERAGE_INCONCLUSIVE"

    if all(anchor_clear_by_direction.values()):
        anchor_verdict = "TARGET_SPECIFIC_ANCHOR_RESCUE_SUPPORTED"
    elif all(contrast(direction, "ANCHOR_90_MINUS_ZERO_90").ci_high < .05 for direction in directions):
        anchor_verdict = "TARGET_SPECIFIC_ANCHOR_RESCUE_NOT_SUPPORTED"
    elif sum(anchor_clear_by_direction.values()) == 1 or all(aligned_zero_clear.values()):
        anchor_verdict = "TARGET_SPECIFIC_ANCHOR_RESCUE_PARTIAL"
    else:
        anchor_verdict = "TARGET_SPECIFIC_ANCHOR_RESCUE_INCONCLUSIVE"

    if coverage_verdict == "OTHER_INTERVENTION_COVERAGE_WEAK_RESCUE" and anchor_verdict == "TARGET_SPECIFIC_ANCHOR_RESCUE_SUPPORTED":
        overall = "GENERALIZATION_AXIS_ASYMMETRY_SUPPORTED"
    elif coverage_verdict == "OTHER_INTERVENTION_COVERAGE_STRONG_RESCUE" and anchor_verdict == "TARGET_SPECIFIC_ANCHOR_RESCUE_NOT_SUPPORTED":
        overall = "GENERALIZATION_AXIS_ASYMMETRY_NOT_SUPPORTED"
    elif anchor_verdict in ("TARGET_SPECIFIC_ANCHOR_RESCUE_SUPPORTED", "TARGET_SPECIFIC_ANCHOR_RESCUE_PARTIAL"):
        overall = "GENERALIZATION_AXIS_ASYMMETRY_PARTIALLY_SUPPORTED"
    else:
        overall = "GENERALIZATION_AXIS_ASYMMETRY_INCONCLUSIVE"
    external = "EXTERNAL_REPLICATION_JUSTIFIED" if anchor_verdict == "TARGET_SPECIFIC_ANCHOR_RESCUE_SUPPORTED" else "EXTERNAL_REPLICATION_NOT_JUSTIFIED_YET"
    summary = {
        "curve_summary": curve_rows,
        "contrast_seed_sd": {
            f"{direction}::{name}": float(np.std(values, ddof=1))
            for (direction, name), values in contrast_seed_values.items()
        },
        "rule_diagnostics": {
            "coverage_strong_by_direction": coverage_strong_by_direction,
            "coverage_weak_by_direction": coverage_weak_by_direction,
            "anchor_clear_by_direction": anchor_clear_by_direction,
            "aligned_zero_clear_by_direction": aligned_zero_clear,
            "coverage_levels_aligned_above_both_controls": coverage_pair_counts,
        },
        "component_verdicts": {"coverage": coverage_verdict, "anchor": anchor_verdict},
        "overall_verdict": overall,
        "external_replication_decision": external,
    }
    return summary, contrasts


def make_figure(config: dict[str, Any], summary: dict[str, Any], contrasts: pd.DataFrame) -> None:
    curves = pd.DataFrame(summary["curve_summary"])
    colors = {"ZERO_SHOT": "#6B7280", "ALIGNED_ANCHOR": "#1F77B4", "SHUFFLED_ANCHOR": "#D97706"}
    labels = {"ZERO_SHOT": "Zero-shot", "ALIGNED_ANCHOR": "Aligned anchor", "SHUFFLED_ANCHOR": "Shuffled anchor"}
    fig = plt.figure(figsize=(12, 7.2))
    grid = fig.add_gridspec(2, 2, hspace=.38, wspace=.30)
    ax = fig.add_subplot(grid[0, 0])
    ax.axis("off")
    ax.text(.02, .90, "Other-intervention coverage", fontsize=12, weight="bold")
    ax.annotate("", xy=(.92, .76), xytext=(.08, .76), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(.08, .68, "10%", ha="center"); ax.text(.92, .68, "90%", ha="center")
    ax.text(.03, .48, "ZERO-SHOT", color=colors["ZERO_SHOT"], weight="bold")
    ax.text(.30, .48, "descriptor(q)  →  recipient response(q)", fontsize=9)
    ax.text(.03, .30, "ALIGNED", color=colors["ALIGNED_ANCHOR"], weight="bold")
    ax.text(.30, .30, "source response(q)  →  recipient response(q)", fontsize=9)
    ax.text(.03, .12, "SHUFFLED", color=colors["SHUFFLED_ANCHOR"], weight="bold")
    ax.text(.30, .12, "source response(π(q))  →  recipient response(q)", fontsize=9)
    ax.set_title("A  Matched information regimes", loc="left", fontsize=12, weight="bold")

    for panel, direction, title in ((grid[0, 1], "K562_TO_RPE1", "B  K562 → RPE1"),
                                    (grid[1, 0], "RPE1_TO_K562", "C  RPE1 → K562")):
        ax = fig.add_subplot(panel)
        for regime in config["regimes"]:
            part = curves[(curves.direction == direction) & (curves.regime == regime)].sort_values("coverage_fraction")
            x = part.coverage_fraction.to_numpy() * 100
            y = part.geometry_mean.to_numpy()
            ax.plot(x, y, "o-", color=colors[regime], label=labels[regime])
            ax.fill_between(x, part.geometry_ci_low.to_numpy(), part.geometry_ci_high.to_numpy(), color=colors[regime], alpha=.12)
        ax.axhline(0, color="#BBBBBB", linewidth=.8)
        ax.set(xlabel="Other-intervention coverage (%)", ylabel="Fixed-target geometry", title=title)
        ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(grid[1, 1])
    names = ["ZERO_90_MINUS_ZERO_10", "ANCHOR_90_MINUS_ZERO_90", "ANCHOR_10_MINUS_ZERO_90", "ANCHOR_MINUS_SHUFFLE_90"]
    short = ["Zero 90−10", "Anchor90−Zero90", "Anchor10−Zero90", "Aligned−Shuffle90"]
    x = np.arange(len(names), dtype=float)
    offsets = {"K562_TO_RPE1": -.12, "RPE1_TO_K562": .12}
    for direction, marker, color in (("K562_TO_RPE1", "o", "#2563EB"), ("RPE1_TO_K562", "s", "#DC2626")):
        part = contrasts.set_index(["direction", "contrast"]).loc[[(direction, name) for name in names]].reset_index()
        estimates = part.estimate.to_numpy()
        errors = np.vstack([estimates - part.ci_low.to_numpy(), part.ci_high.to_numpy() - estimates])
        ax.errorbar(x + offsets[direction], estimates, yerr=errors, fmt=marker, color=color, capsize=3, label=direction.replace("_TO_", "→"))
    ax.axhline(0, color="#777777", linewidth=.8)
    ax.set_xticks(x, short, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Geometry contrast")
    ax.set_title("D  Prespecified high-value contrasts", loc="left", fontsize=12, weight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Generalization-axis factorial experiment", fontsize=14, weight="bold")
    fig.subplots_adjust(top=.91, bottom=.12)
    fig.savefig(OUT / "figure6_internal_factorial.png", dpi=180)
    fig.savefig(OUT / "figure6_internal_factorial.svg")
    plt.close(fig)


def report(
    config: dict[str, Any], audit: dict[str, Any], design: dict[str, Any], results: pd.DataFrame,
    contrasts: pd.DataFrame, summary: dict[str, Any], runtime: pd.DataFrame, leakage: pd.DataFrame
) -> None:
    curves = pd.DataFrame(summary["curve_summary"])
    lines = [
        "# FINAL FACTORIAL REPORT",
        "",
        "## A. DATA / ELIGIBILITY AUDIT",
        "",
        f"- Shared eligible perturbations: **{audit['shared_intervention_count']}**",
        f"- Frozen targets: **{len(design['targets'])}**; common strict-trans response genes: **{audit['shared_response_gene_count']}**",
        f"- Source descriptor: `{audit['safe_source_descriptor']}`",
        f"- Ridge: direct multi-output affine regression, alpha={config['model']['alpha']}, training-only input scaling, no PCA.",
        f"- Leakage firewall: **{'PASS' if leakage.leakage_firewall_pass.all() else 'FAIL'}** across {len(leakage)} regime rows.",
        "",
        "## B. EXPERIMENT DESIGN",
        "",
        "One fixed target set T was held out from every fit. Five seeds used nested 10/25/40/60/80/90% subsets of the same paired remaining universe. Zero-shot used the control-only source descriptor; aligned transfer used the same target's empirical source-context response; shuffled transfer reused the aligned model but supplied a deterministic wrong-target source response. Both K562→RPE1 and RPE1→K562 used identical targets, coverage sets, response genes, model family, and scoring.",
        "",
        "## C. RESOURCE COST",
        "",
        f"- Fitted models: **{2 * len(config['directions']) * len(config['coverage_seeds']) * len(config['coverage_fractions'])}** (shuffled queries reused aligned maps).",
        f"- Device: CPU; full runtime: **{runtime[runtime.stage == 'FULL'].condition_total_seconds.sum():.1f} s**; observed peak working set: **{runtime.peak_working_set_mb.max():.1f} MB**.",
        f"- Cell-budget control: **{config['cell_budget_control']}**",
        "",
        "## D. CORE CURVES",
        "",
    ]
    for direction in config["directions"]:
        lines += [f"**{direction.replace('_TO_', ' → ')}**", "", "Entries are mean across five coverage seeds ± seed SD, followed by the 95% paired target-bootstrap CI.", "", "| Coverage | Zero-shot geometry | Aligned anchor | Shuffled anchor |", "|---:|---:|---:|---:|"]
        for coverage in config["coverage_fractions"]:
            values = {}
            for regime in config["regimes"]:
                row = curves[(curves.direction == direction) & np.isclose(curves.coverage_fraction, coverage) & (curves.regime == regime)].iloc[0]
                values[regime] = f"{row.geometry_mean:.4f} ± {row.geometry_seed_sd:.4f} [{row.geometry_ci_low:.4f}, {row.geometry_ci_high:.4f}]"
            lines.append(f"| {coverage:.0%} | {values['ZERO_SHOT']} | {values['ALIGNED_ANCHOR']} | {values['SHUFFLED_ANCHOR']} |")
        lines.append("")
    lines += ["## E. PRIMARY CONTRASTS", "", "| Direction | Contrast | Estimate | Seed SD | 95% paired target-bootstrap CI |", "|---|---|---:|---:|---:|"]
    for row in contrasts.itertuples(index=False):
        seed_sd = summary["contrast_seed_sd"][f"{row.direction}::{row.contrast}"]
        lines.append(f"| {row.direction.replace('_TO_', '→')} | {row.contrast} | {row.estimate:.4f} | {seed_sd:.4f} | [{row.ci_low:.4f}, {row.ci_high:.4f}] |")
    lines += [
        "",
        "C1 is ZERO_90_MINUS_ZERO_10; C2 is ANCHOR_90_MINUS_ZERO_90; C3 is ANCHOR_10_MINUS_ZERO_90; C4 is the six ANCHOR_MINUS_SHUFFLE contrasts; C5 is their independent replication across both directions.",
        f"C5 directional replication: anchor-clear criteria passed K562→RPE1={summary['rule_diagnostics']['anchor_clear_by_direction']['K562_TO_RPE1']} and RPE1→K562={summary['rule_diagnostics']['anchor_clear_by_direction']['RPE1_TO_K562']}.",
        "",
        "## F. COMPONENT VERDICTS",
        "",
        f"**{summary['component_verdicts']['coverage']}**",
        "",
        f"**{summary['component_verdicts']['anchor']}**",
        "",
        "## G. OVERALL VERDICT",
        "",
        f"**{summary['overall_verdict']}**",
        "",
        "## H. LIMITATIONS",
        "",
        "1. This does not prove zero-shot gene prediction is impossible.",
        "2. Cross-context source-response transfer itself is not novel.",
        "3. The experiment tests relative information efficiency under a lightweight, matched Ridge setup.",
        "4. K562 and RPE1 are only two contexts.",
        "5. External replication is required before a field-level atlas-design claim.",
        "6. Cell-budget matching was not possible, so coverage and experimental measurement budget remain partially coupled.",
        "",
        "## I. EXTERNAL REPLICATION DECISION",
        "",
        f"**{summary['external_replication_decision']}**",
        "",
    ]
    if summary["external_replication_decision"] == "EXTERNAL_REPLICATION_JUSTIFIED":
        lines.append("Short plan: in a separate preregistered task, use an existing Frangieh multi-context paired perturbation cache, freeze a common strict-trans response axis and held targets, and repeat the same direct Ridge zero-shot/aligned/shuffled comparison without implementing PerturbMap or tuning alpha.")
        lines.append("")
    coverage_verdict = summary["component_verdicts"]["coverage"]
    anchor_verdict = summary["component_verdicts"]["anchor"]
    if anchor_verdict == "TARGET_SPECIFIC_ANCHOR_RESCUE_SUPPORTED" and coverage_verdict == "OTHER_INTERVENTION_COVERAGE_WEAK_RESCUE":
        answer = "For these fixed unseen targets, one empirical response of the same intervention in the other context was more informative than exposure to many additional other perturbations: zero-shot coverage produced only limited recovery, while identity-aligned anchors produced a larger, identity-specific gain in both directions. This is evidence about relative sample efficiency in this matched Ridge experiment, not an impossibility result or a universal claim."
    elif anchor_verdict in ("TARGET_SPECIFIC_ANCHOR_RESCUE_SUPPORTED", "TARGET_SPECIFIC_ANCHOR_RESCUE_PARTIAL"):
        answer = "For these fixed unseen targets, the same intervention's empirical response in the other context was generally more informative than adding other perturbations, but the combined coverage and directional evidence was mixed. The result supports at most a conditional relative-sample-efficiency interpretation for this matched Ridge setup, not a universal or impossibility claim."
    else:
        answer = "For these fixed unseen targets, the experiment did not establish that a same-intervention empirical anchor was more informative than broad exposure to other perturbations. The result is limited to this matched Ridge setup and does not establish either universal transferability or impossibility of zero-shot intervention prediction."
    lines.append("**Final scientific question.** " + answer)
    atomic_text(OUT / "FINAL_FACTORIAL_REPORT.md", "\n".join(lines) + "\n")


def full_run(config: dict[str, Any], data: dict[str, Any], eligible: dict[str, Any], design: dict[str, Any]) -> None:
    audit = json.loads((OUT / "repo_audit.json").read_text(encoding="utf-8"))
    if not audit.get("smoke_gate", {}).get("passed"):
        raise RuntimeError("Full run requires a passing smoke gate")
    if audit["large_model_training_required"]:
        raise RuntimeError("Large-model training is forbidden")
    aligned = align_arrays(data, eligible)
    target_count = len(design["targets"])
    bootstrap_rows = np.random.default_rng(int(config["bootstrap"]["seed"])).integers(
        0, target_count, size=(int(config["bootstrap"]["iterations"]), target_count)
    )
    result_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    boot_store: dict[tuple[str, int, float, str], np.ndarray] = {}
    for direction in config["directions"]:
        for seed in config["coverage_seeds"]:
            for coverage in config["coverage_fractions"]:
                started = time.perf_counter()
                rows, boots, leakage, runtime = run_condition(
                    config, aligned, eligible, design, direction, int(seed), float(coverage), bootstrap_rows
                )
                result_rows.extend(rows); leakage_rows.extend(leakage); runtime_rows.append(runtime)
                for regime, values in boots.items():
                    boot_store[(direction, int(seed), float(coverage), regime)] = values
                print(f"{direction} seed={seed} coverage={coverage:.0%} complete in {time.perf_counter()-started:.2f}s", flush=True)
    results = pd.DataFrame(result_rows)
    leakage = pd.DataFrame(leakage_rows)
    runtime = pd.DataFrame(runtime_rows)
    if len(results) != 180 or len(leakage) != 180 or not leakage.leakage_firewall_pass.all():
        raise RuntimeError("Full grid or leakage audit is incomplete")
    results.to_csv(OUT / "factorial_results.csv", index=False)
    leakage.to_csv(OUT / "leakage_audit.csv", index=False)
    previous_runtime = pd.read_csv(OUT / "runtime_audit.csv")
    previous_runtime = previous_runtime[previous_runtime.stage == "SMOKE"].tail(1)
    runtime = pd.concat([previous_runtime, runtime], ignore_index=True, sort=False)
    runtime.to_csv(OUT / "runtime_audit.csv", index=False)
    summary, contrasts = summarize(config, results, boot_store)
    contrasts.to_csv(OUT / "factorial_contrasts.csv", index=False)
    summary["resource_cost"] = {
        "fitted_models": 120,
        "shuffled_anchor_refits": 0,
        "device": config["device"],
        "full_condition_runtime_seconds": float(runtime[runtime.stage == "FULL"].condition_total_seconds.sum()),
        "peak_working_set_mb": float(runtime.peak_working_set_mb.max()),
        "cell_budget_control_status": "CELL_BUDGET_CONTROL_NOT_ACCESSIBLE",
    }
    atomic_json(OUT / "factorial_summary.json", summary)
    make_figure(config, summary, contrasts)
    report(config, audit, design, results, contrasts, summary, runtime, leakage)
    print("FULL GRID COMPLETE")
    print(summary["component_verdicts"]["coverage"])
    print(summary["component_verdicts"]["anchor"])
    print(summary["overall_verdict"])
    print(summary["external_replication_decision"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "smoke", "run"))
    args = parser.parse_args()
    config = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    eligible = eligibility(data)
    audit = write_repo_audit(config, data, eligible)
    if audit["large_model_training_required"]:
        raise SystemExit("STOP: large-model training would be required")
    if audit["shared_intervention_count"] < 40 or audit["shared_response_gene_count"] < 200:
        raise SystemExit("STOP: paired K562/RPE1 data are insufficient")
    print(f"AUDIT shared={audit['shared_intervention_count']} genes={audit['shared_response_gene_count']} large_model=NO")
    if args.mode == "audit":
        return
    design = prepare_design(config, data, eligible)
    if args.mode == "smoke":
        smoke_row = smoke(config, data, eligible, design)
        print(f"SMOKE PASS projected_full_runtime={smoke_row['projected_full_runtime_seconds']:.1f}s")
        return
    full_run(config, data, eligible, design)


if __name__ == "__main__":
    main()
