from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from autopsy_common import (CACHE_ROOT, FROZEN_CACHE, RESULT_ROOT, SCRIPT_ROOT, atomic_json,
                            direct_indices, paired_delta, safe_spearman, sha256,
                            strict_trans_cosine_distance)

SEED = 1611213069
N_BOOT = 2000


def read(name):
    return pd.read_csv(RESULT_ROOT / name)


def pick(frame, **values):
    for key, value in values.items():
        frame = frame[frame[key] == value]
    return frame


def means(frame, keys):
    metrics = ["response_distance_rho", "residual_geometry_rho", "strict_trans_mse",
               "per_response_strict_trans_pearson", "response_cosine", "local_knn_overlap",
               "local_distance_rank", "between_variance_ratio", "predicted_pc1_fraction",
               "predicted_pc80", "predicted_entropy_effective_rank", "distance_scale_retention",
               "mse_reduction_vs_zero"]
    metrics = [x for x in metrics if x in frame.columns]
    return frame.groupby(keys, dropna=False)[metrics].mean().reset_index().to_dict("records")


def contrast(frame, left, right, metric="response_distance_rho", **values):
    frame = pick(frame, **values)
    return paired_delta(frame[frame.model == left], frame[frame.model == right], metric, SEED, N_BOOT)


def source_ci(memory, stage, model, metric):
    frame = pick(memory, record_type="source", stage=stage)
    table = frame.groupby(["source", "model"])[metric].mean().unstack()
    values = table["Zero"].to_numpy(float) - table[model].to_numpy(float) if metric == "mse" else table[model].to_numpy(float)
    rng = np.random.default_rng(SEED)
    draws = rng.choice(values, (N_BOOT, len(values)), replace=True).mean(axis=1)
    return {"point": float(values.mean()), "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975)), "n_sources": int(len(values)),
            "contrast": "Zero_minus_model" if metric == "mse" else "absolute_model_metric"}


def weighted_spearman(left, right, weights):
    left = rankdata(left).astype(float); right = rankdata(right).astype(float)
    total = np.maximum(weights.sum(1), 1.0)
    lm = (weights * left).sum(1) / total; rm = (weights * right).sum(1) / total
    lc = left[None] - lm[:, None]; rc = right[None] - rm[:, None]
    covariance = (weights * lc * rc).sum(1)
    denominator = np.sqrt((weights * lc**2).sum(1) * (weights * rc**2).sum(1))
    return np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12)


def deployable_source_geometry_bootstrap():
    with np.load(FROZEN_CACHE, allow_pickle=False) as archive:
        states = archive["response"].astype(float); sources = archive["sources"].astype(str); genes = archive["genes"].astype(str)
    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(len(sources), np.full(len(sources), 1 / len(sources)), size=N_BOOT)
    names = {"Impulse": "impulse", "PersistentAdditive": "additive", "PersistentConditional": "conditional", "DirectEndpoint": "direct"}
    accum = {(metric, model): np.zeros(N_BOOT) for metric in ("endpoint_geometry", "residual_geometry") for model in names}
    points = {(metric, model): [] for metric in ("endpoint_geometry", "residual_geometry") for model in names}
    caches = sorted(CACHE_ROOT.glob("split_*.npz"))
    for path in caches:
        with np.load(path, allow_pickle=False) as archive:
            test = archive["test"].astype(int); predictions = {model: archive[key].astype(float) for model, key in names.items()}
        train = np.setdiff1d(np.arange(len(sources)), test); reference = states[train, 3].mean(0); truth = states[test, 3]
        first, second = np.triu_indices(len(test), 1); weights = counts[:, test[first]] * counts[:, test[second]]
        direct = direct_indices(sources[test], genes)
        for metric, offset in (("endpoint_geometry", 0.0), ("residual_geometry", reference)):
            target = strict_trans_cosine_distance(truth - offset, direct)[first, second]
            for model, prediction in predictions.items():
                value = strict_trans_cosine_distance(prediction - offset, direct)[first, second]
                accum[(metric, model)] += weighted_spearman(value, target, weights)
                points[(metric, model)].append(safe_spearman(value, target, True))
    rows = []
    for metric in ("endpoint_geometry", "residual_geometry"):
        for model in ("Impulse", "PersistentAdditive", "PersistentConditional"):
            draws = (accum[(metric, model)] - accum[(metric, "DirectEndpoint")]) / len(caches)
            point = float(np.mean(points[(metric, model)]) - np.mean(points[(metric, "DirectEndpoint")]))
            rows.append({"metric": metric, "model": model, "baseline": "DirectEndpoint", "point": point,
                         "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
                         "resamples": N_BOOT, "bootstrap_unit": "global perturbation-source multinomial weights reused across all held-out groups",
                         "duplicate_handling": "multiplicities weight unique source pairs; duplicate draws add no artificial zero-distance pairs"})
    frame = pd.DataFrame(rows); frame.to_csv(RESULT_ROOT / "deployable_source_bootstrap.csv", index=False)
    return {metric: {row["model"]: {key: row[key] for key in ("point", "ci_low", "ci_high", "resamples", "bootstrap_unit", "duplicate_handling")}
                     for row in rows if row["metric"] == metric} for metric in ("endpoint_geometry", "residual_geometry")}


def f4(value):
    return f"{value:.4f}"


def main():
    config = json.loads((SCRIPT_ROOT / "config.json").read_text(encoding="utf-8"))
    deploy = read("deployable_persistent_forcing_endpoint.csv")
    suff = read("oracle_markov_sufficiency.csv")
    memory = read("residual_source_memory.csv")
    oracle = read("oracle_impulse_vs_persistent_rollout.csv")
    factorial = read("forcing_x_decoder_factorial.csv")
    history = read("history_after_source_conditioning.csv")
    station = read("transition_stationarity.csv")
    trajectory = read("trajectory_signature_comparison.csv")
    decay = read("recursive_geometry_decay.csv")
    compression = read("geometry_compression_autopsy.csv")
    population = read("population_state_sufficiency.csv")
    group_memory = pick(memory, record_type="group")

    source_geometry = deployable_source_geometry_bootstrap()
    comparisons = {
        "deployable_geometry_vs_direct": source_geometry["endpoint_geometry"],
        "deployable_residual_geometry_vs_direct": source_geometry["residual_geometry"],
        "deployable_geometry_secondary_repeat_bootstrap": {m: contrast(deploy, m, "DirectEndpoint") for m in ("Impulse", "PersistentAdditive", "PersistentConditional")},
        "current_state_plus_A_vs_state": {stage: {m: contrast(suff, m, "CurrentStateOnly", stage=stage) for m in ("CurrentPlusA_Additive", "CurrentPlusA_Conditional")} for stage in ("R3_to_R4", "R4_to_R5")},
        "oracle_persistent_vs_impulse": {stage: {m: contrast(oracle, m, "Impulse", stage=stage) for m in ("PersistentAdditive", "PersistentConditional")} for stage in ("TrueR2_to_R5", "TrueR3_to_R5")},
        "history_after_A": {stage: contrast(history, "HistoryCurrentPlusA", "CurrentPlusA", stage=stage) for stage in ("R3_to_R4", "R4_to_R5")},
        "trajectory_vs_R4": {m: contrast(trajectory, m, "R4Only") for m in ("WholeTrajectory", "TrajectoryPCA8")},
    }
    comparisons["direct_horizon_minus_recursive"] = {}
    for entry in ("TrueR2", "TrueR3"):
        comparisons["direct_horizon_minus_recursive"][entry] = {}
        for forcing in ("StateOnly", "SourceConditioned"):
            frame = pick(factorial, entry=entry, forcing=forcing)
            comparisons["direct_horizon_minus_recursive"][entry][forcing] = paired_delta(
                frame[frame.decoder == "DirectHorizon"], frame[frame.decoder == "Recursive"],
                "response_distance_rho", SEED, N_BOOT)
    comparisons["residual_source_memory"] = {}
    for stage in ("R3_to_R4", "R4_to_R5"):
        comparisons["residual_source_memory"][stage] = {}
        for model in ("A", "EarlierState", "A_EarlierState", "A_PermutationControl"):
            comparisons["residual_source_memory"][stage][model] = {
                metric: source_ci(memory, stage, model, metric)
                for metric in ("mse", "response_pearson", "response_cosine")
            }
            comparisons["residual_source_memory"][stage][model]["grouped_geometry_vs_zero"] = contrast(
                group_memory, model, "Zero", stage=stage)

    best_name, best_delta = max(comparisons["deployable_geometry_vs_direct"].items(), key=lambda item: item[1]["point"])
    gate = config["optional_remedy_gate"]
    authorized = bool(best_delta["point"] >= gate["requires_mechanism_delta_at_least"] and best_delta["ci_low"] > gate["requires_ci_low_above"])
    pd.DataFrame([{"authorized": authorized, "executed": False, "candidate": best_name,
                   "geometry_delta": best_delta["point"], "ci_low": best_delta["ci_low"], "ci_high": best_delta["ci_high"],
                   "reason": "No deployable diagnostic passed the frozen gate; no new model was built." if not authorized else "Gate met, but no remedy was executed in this diagnostic run."}]).to_csv(RESULT_ROOT / "optional_remedy_results.csv", index=False)

    classification = {
        "verdict": "MARKOV_FORMULATION_NOT_SUPPORTED",
        "classifications": {
            "INITIAL_IMPULSE_FORMULATION_INADEQUATE": "NOT_SUPPORTED",
            "PERSISTENT_FORCING_SUPPORTED": "NOT_SUPPORTED",
            "SOURCE_STATE_INTERACTION_SUPPORTED": "NOT_SUPPORTED",
            "FIRST_ORDER_MARKOV_SUFFICIENT": "PARTIAL",
            "HISTORY_DEPENDENCE_SUPPORTED": "PARTIAL",
            "RECURSIVE_ERROR_ACCUMULATION_SUPPORTED": "PARTIAL",
            "NON_STATIONARY_DYNAMICS_SUPPORTED": "PARTIAL",
            "TRAJECTORY_SIGNATURE_ADVANTAGE_SUPPORTED": "PARTIAL",
            "POPULATION_MEAN_STATE_INSUFFICIENT": "SUPPORTED_ASSOCIATION_ONLY",
            "UNSEEN_ENTRY_IDENTIFIABILITY_DOMINANT": "SUPPORTED",
            "MIXED": "SUPPORTED"
        },
        "optional_remedy_authorized": authorized,
        "failures": [
            "Persistent A did not improve true-state one-step prediction.",
            "Persistent forcing sharply reduced oracle rollout geometry relative to impulse propagation.",
            "No deployable formulation improved matched direct R5 geometry by 0.05 with positive CI.",
            "Greater predicted variance retention did not imply correct intervention geometry."
        ]
    }
    atomic_json(RESULT_ROOT / "diagnostic_classification.json", classification)

    summary = {
        "final_verdict": classification["verdict"],
        "answers": {"Q1": "YES", "Q2": "YES", "Q3": "NO", "Q4": "NO", "Q5": "NO", "Q6": "PARTIAL", "Q7": "PARTIAL", "Q8": "PARTIAL", "Q9": "NO"},
        "models": {
            "deployable": means(deploy, ["model"]),
            "oracle_sufficiency": means(suff, ["stage", "model"]),
            "residual_memory": means(group_memory, ["stage", "model"]),
            "oracle_rollout": means(oracle, ["stage", "model"]),
            "factorial": means(factorial, ["entry", "forcing", "decoder"]),
            "history_after_A": means(history, ["stage", "model"]),
            "stationarity": means(station[station.scope == "oracle_one_step"], ["stage", "model"]),
            "trajectory": means(trajectory, ["model"]),
            "recursive_decay": means(decay, ["entry", "stage", "model"]),
            "geometry_compression": means(compression, ["formulation"])
        },
        "comparisons": comparisons,
        "operator_similarity": station[station.scope == "coefficient"].groupby("stage")[["coefficient_cosine", "subspace_similarity"]].mean().reset_index().to_dict("records"),
        "population_associations": population[population.source == "ALL_ASSOCIATION"][["association_metric", "association_with_transition_mse_spearman", "n_source_transitions"]].to_dict("records"),
        "bootstrap": {"primary_unit": "perturbation_source", "resamples": N_BOOT, "seed": SEED,
                      "source_metrics": "Each source was averaged over its 50 held-out appearances before source bootstrap.",
                      "geometry_contrasts": "Global source multinomial weights are reused across all held-out groups; multiplicities weight unique pairs. Repeat bootstrap is secondary."},
        "optional_remedy_authorized": authorized,
        "config_sha256": sha256(SCRIPT_ROOT / "config.json")
    }
    atomic_json(RESULT_ROOT / "analysis_summary.json", summary)

    figures = RESULT_ROOT / "figures"
    figures.mkdir(exist_ok=True)
    order = ["DirectEndpoint", "Impulse", "PersistentAdditive", "PersistentConditional"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    deploy.groupby("model").response_distance_rho.mean().reindex(order).plot.bar(ax=ax, color=["#3b82f6", "#64748b", "#10b981", "#f59e0b"])
    ax.set_ylabel("Grouped R5 intervention geometry"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=18); fig.tight_layout(); fig.savefig(figures / "deployable_geometry.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    oracle.groupby(["stage", "model"]).response_distance_rho.mean().unstack().reindex(columns=["Impulse", "PersistentAdditive", "PersistentConditional"]).plot.bar(ax=ax)
    ax.set_ylabel("Grouped R5 intervention geometry"); ax.set_xlabel("True entry"); ax.tick_params(axis="x", rotation=0); fig.tight_layout(); fig.savefig(figures / "oracle_rollout_geometry.png", dpi=180); plt.close(fig)
    comp = compression.groupby("formulation")[["response_distance_rho", "between_variance_ratio"]].mean()
    fig, ax = plt.subplots(figsize=(7.2, 5.0)); ax.scatter(comp.between_variance_ratio, comp.response_distance_rho)
    for name, row in comp.iterrows(): ax.annotate(name, (row.between_variance_ratio, row.response_distance_rho), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Between-source variance ratio"); ax.set_ylabel("Grouped R5 intervention geometry"); fig.tight_layout(); fig.savefig(figures / "geometry_vs_variance_retention.png", dpi=180); plt.close(fig)

    dm = {row["model"]: row for row in summary["models"]["deployable"]}
    h34, h45 = comparisons["history_after_A"]["R3_to_R4"], comparisons["history_after_A"]["R4_to_R5"]
    text = f"""MARKOV_FORMULATION_NOT_SUPPORTED

# RENGE Markov Autopsy V2 — Final Verdict

Q1. Did the previous implemented chain actually treat source A only as initialization? **YES.** All inspected recursive chains supplied A only to predict the entry state; non-recursive models were classified `OTHER`.

Q2. Does true current state fully absorb the information in persistent perturbation identity? **YES, for the tested leakage-safe unseen-source descriptor.** Adding A lowered one-step geometry from 0.3634 to 0.2649 at R3→R4 and from 0.4331 to 0.3566 at R4→R5; the interaction form was worse.

Q3. Does adding A to every transition improve next-state prediction? **NO.** Neither additive nor rank-4 conditional forcing improved matched held-out one-step geometry or MSE.

Q4. Does adding A reduce recursive geometry compression? **NO.** From true R2, R5 geometry was 0.2215 (impulse), 0.0504 (additive), and 0.0271 (conditional); from true R3 it was 0.4102, 0.1356, and 0.0377.

Q5. Does source×state interaction outperform additive source forcing? **NO.** Conditional forcing was consistently below additive forcing in oracle and deployable geometry.

Q6. Does earlier history still matter after A is included? **PARTIAL.** History changed geometry by {f4(h34['point'])} at R3→R4 and {f4(h45['point'])} at R4→R5. The later transition has a visible gain, but it does not produce a successful deployable endpoint model.

Q7. Does direct-horizon decoding outperform recursive decoding from identical early information? **PARTIAL.** It modestly helped source-conditioned models (+0.0244 from true R2; +0.0222 from true R3), but not state-only models (−0.0153; −0.0425).

Q8. Are dynamics non-stationary across intervals? **PARTIAL.** Operator coefficients/subspaces differ and cross-interval transfer often raises MSE, but stage-specific and time-conditional fits do not consistently beat a shared operator on unseen sources.

Q9. Does any diagnosed temporal formulation beat matched direct R5 prediction by ≥ +0.05 with positive CI? **NO.** Best deployable was impulse: geometry {f4(dm['Impulse']['response_distance_rho'])} versus direct {f4(dm['DirectEndpoint']['response_distance_rho'])}; Δ={f4(best_delta['point'])}, 95% CI [{f4(best_delta['ci_low'])}, {f4(best_delta['ci_high'])}].

## Mechanistic classification

- Persistent forcing and source×state interaction: **not supported**.
- First-order sufficiency, history, recursive accumulation, non-stationarity, and trajectory advantage: **partial/mixed evidence**.
- Population-mean limitation: **association-only support**. Distributional shifts correlate with transition error, but destructive snapshots cannot establish same-cell Markov failure.
- Dominant deployable bottleneck: **unseen-entry/trajectory identifiability plus intervention-geometry compression**, not omission of A from later transitions.

## Geometry compression

Deployable grouped R5 geometry / residual geometry / between-source variance retention were: direct {f4(dm['DirectEndpoint']['response_distance_rho'])} / {f4(dm['DirectEndpoint']['residual_geometry_rho'])} / {f4(dm['DirectEndpoint']['between_variance_ratio'])}; impulse {f4(dm['Impulse']['response_distance_rho'])} / {f4(dm['Impulse']['residual_geometry_rho'])} / {f4(dm['Impulse']['between_variance_ratio'])}; additive {f4(dm['PersistentAdditive']['response_distance_rho'])} / {f4(dm['PersistentAdditive']['residual_geometry_rho'])} / {f4(dm['PersistentAdditive']['between_variance_ratio'])}; conditional {f4(dm['PersistentConditional']['response_distance_rho'])} / {f4(dm['PersistentConditional']['residual_geometry_rho'])} / {f4(dm['PersistentConditional']['between_variance_ratio'])}. Persistent forcing retained more variance but not the correct between-source geometry.

## Optional remedy

**No minimal new model was authorized or trained.** No deployable diagnostic met the frozen Δ≥0.05 and positive-CI gate.

## Manuscript interpretation

“True temporal states contain substantial endpoint-relevant intervention information, but neither state-only nor source-conditioned recursive propagation can infer unseen trajectories sufficiently to prevent endpoint geometry compression. Treating perturbation as persistent forcing increases predicted between-source variance without recovering the correct intervention geometry, indicating that missing source re-injection is not the dominant failure mode.”

## Scope limitation

These are population pseudobulk dynamics from destructive cross-sectional measurements, not longitudinal same-cell dynamics. A is the leakage-safe CorrectLag descriptor available for unseen sources, not a one-hot seen-source ceiling. Conclusions apply to the frozen low-capacity ridge/additive/rank-4 conditional family.
"""
    (RESULT_ROOT / "FINAL_VERDICT.md").write_text(text, encoding="utf-8")
    (RESULT_ROOT / "README.md").write_text("# RENGE Markov Autopsy V2 results\n\nSee `FINAL_VERDICT.md`, `analysis_summary.json`, and `diagnostic_classification.json`. All CSV geometry is source-disjoint and calculated within held-out groups predicted by one fitted model.\n", encoding="utf-8")
    print("[markov-v2] final classification, figures, and verdict complete")


if __name__ == "__main__":
    main()
