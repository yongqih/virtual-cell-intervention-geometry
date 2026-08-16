from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from common import (CACHE_ROOT, RESULT_ROOT, atomic_json, direct_indices, grouped_twofold_splits,
                    now, safe_spearman, strict_trans_cosine_distance)
from parts_cde import representation_for_condition, standardized_ridge


CONDITIONS = ("StaticControl", "CorrectLag", "SameWave", "TemporalShuffle",
              "GeneIdentityShuffle", "IntermediateMasked")
COMPARISONS = ("StaticControl", "SameWave", "TemporalShuffle", "GeneIdentityShuffle", "IntermediateMasked")


def weighted_spearman(left_values: np.ndarray, right_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply source multiplicities to unique pairs without inventing duplicate zero distances."""
    left = rankdata(left_values).astype(np.float64); right = rankdata(right_values).astype(np.float64)
    total = np.maximum(weights.sum(1), 1.0)
    left_mean = (weights * left[None]).sum(1) / total
    right_mean = (weights * right[None]).sum(1) / total
    left_centered = left[None] - left_mean[:, None]; right_centered = right[None] - right_mean[:, None]
    covariance = (weights * left_centered * right_centered).sum(1)
    denominator = np.sqrt((weights * left_centered**2).sum(1) * (weights * right_centered**2).sum(1))
    output = np.zeros(len(weights), dtype=np.float64)
    valid = denominator > 1e-12; output[valid] = covariance[valid] / denominator[valid]
    return output


def main() -> None:
    config = json.loads(Path(__file__).with_name("frozen_config.json").read_text())
    with np.load(CACHE_ROOT / "renge_processed.npz", allow_pickle=False) as archive:
        sources = archive["sources"].astype(str); genes = archive["genes"].astype(str)
        waves = archive["waves"].astype(np.float32); static = archive["static_control_representation"].astype(np.float32)
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    source_gene_rows = np.asarray([gene_lookup[source] for source in sources])
    splits = grouped_twofold_splits(len(sources), config["source_disjoint_repeats"], config["outer_split_seed"])
    grid = tuple(config["ridge_alpha_grid"]); distance_records = []
    for split_index, split in enumerate(splits):
        train, test = split["train"], split["test"]
        direct = direct_indices(sources[test], genes)
        truth_distance = strict_trans_cosine_distance(waves[test, 0], direct)
        model_distances = {}
        for condition_index, condition in enumerate(CONDITIONS):
            seed = split["seed"] + condition_index * 100003
            representation = representation_for_condition(condition, waves, sources, genes, static, train, seed)
            features = representation[source_gene_rows]
            prediction, _ = standardized_ridge(features[train], waves[train, 0], features[test], grid)
            model_distances[condition] = strict_trans_cosine_distance(prediction, direct)
        distance_records.append((split, truth_distance, model_distances))
        if (split_index + 1) % 20 == 0:
            print(f"[传播复现] Source-bootstrap 预测恢复 {split_index + 1}/{len(splits)}", flush=True)

    resamples = config["bootstrap_resamples"]; rng = np.random.default_rng(880213069)
    accumulated = {baseline: np.zeros(resamples, np.float64) for baseline in COMPARISONS}
    for split, truth_distance, model_distances in distance_records:
        count = len(split["test"]); counts = rng.multinomial(count, np.full(count, 1 / count), size=resamples)
        upper = np.triu_indices(count, 1)
        weights = counts[:, upper[0]] * counts[:, upper[1]]
        truth_values = truth_distance[upper]
        correct_rho = weighted_spearman(model_distances["CorrectLag"][upper], truth_values, weights)
        for baseline in COMPARISONS:
            baseline_rho = weighted_spearman(model_distances[baseline][upper], truth_values, weights)
            accumulated[baseline] += correct_rho - baseline_rho
    source_bootstrap_rows = []
    group_metrics = pd.read_csv(RESULT_ROOT / "first_responder_ablation.csv")
    means = group_metrics.groupby("model").response_distance_rho.mean()
    for baseline in COMPARISONS:
        draws = accumulated[baseline] / len(distance_records)
        source_bootstrap_rows.append({"comparison": f"CorrectLag - {baseline}",
            "point_delta_mean_group_geometry_rho": float(means["CorrectLag"] - means[baseline]),
            "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
            "resamples": resamples, "bootstrap_unit": "perturbation source multinomial weights within each held-out group",
            "duplicate_handling": "multiplicities weight unique source-pairs; duplicate draws do not create artificial zero-distance pairs",
            "heldout_groups_averaged_per_resample": len(distance_records)})
    source_bootstrap = pd.DataFrame(source_bootstrap_rows)
    source_bootstrap.to_csv(RESULT_ROOT / "first_responder_source_bootstrap.csv", index=False)

    # Global source-deletion influence audit for CorrectLag - StaticControl.
    base_differences = []
    for split, truth_distance, model_distances in distance_records:
        upper = np.triu_indices(len(split["test"]), 1)
        base_differences.append(safe_spearman(model_distances["CorrectLag"][upper], truth_distance[upper], True) -
                                safe_spearman(model_distances["StaticControl"][upper], truth_distance[upper], True))
    base_mean = float(np.mean(base_differences)); influence_rows = []
    for global_source, source_name in enumerate(sources):
        differences = []
        for split, truth_distance, model_distances in distance_records:
            test = split["test"]; local = np.flatnonzero(test == global_source)
            keep = np.ones(len(test), bool)
            if len(local): keep[local[0]] = False
            correct = model_distances["CorrectLag"][np.ix_(keep, keep)]
            static_distance = model_distances["StaticControl"][np.ix_(keep, keep)]
            truth = truth_distance[np.ix_(keep, keep)]; upper = np.triu_indices(keep.sum(), 1)
            differences.append(safe_spearman(correct[upper], truth[upper], True) -
                               safe_spearman(static_distance[upper], truth[upper], True))
        leave_out_mean = float(np.mean(differences))
        influence_rows.append({"source": source_name, "full_mean_delta": base_mean,
                               "leave_source_out_mean_delta": leave_out_mean,
                               "change_after_leaving_source_out": leave_out_mean - base_mean})
    influence = pd.DataFrame(influence_rows)
    influence.to_csv(RESULT_ROOT / "source_influence_audit.csv", index=False)

    dose = pd.read_csv(RESULT_ROOT / "information_dose_curve.csv")
    per_repeat = dose.groupby(["repeat", "evidence_fraction"]).response_distance_rho.mean().unstack()
    dose_rng = np.random.default_rng(881213069); dose_rows = []
    dose_values = np.asarray(config["information_doses"], float)
    repeat_indices = dose_rng.integers(0, len(per_repeat), size=(resamples, len(per_repeat)))
    sampled_curves = per_repeat.to_numpy()[repeat_indices].mean(1)
    trend_draws = np.asarray([spearmanr(dose_values, curve).statistic for curve in sampled_curves])
    delta_draws = sampled_curves[:, -1] - sampled_curves[:, 0]
    for index, evidence in enumerate(dose_values):
        values = sampled_curves[:, index]
        dose_rows.append({"evidence_fraction": evidence, "mean_geometry_rho": float(per_repeat[evidence].mean()),
                          "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975)),
                          "bootstrap_unit": "outer repeat (each repeat averages both grouped source-disjoint folds and evidence subsamples)",
                          "resamples": resamples})
    pd.DataFrame(dose_rows).to_csv(RESULT_ROOT / "information_dose_curve_bootstrap.csv", index=False)
    dose_test = {"delta_100_minus_0": float(per_repeat[1.0].mean() - per_repeat[0.0].mean()),
                 "delta_ci": [float(np.quantile(delta_draws, .025)), float(np.quantile(delta_draws, .975))],
                 "dose_mean_spearman": float(spearmanr(dose_values, per_repeat.mean(0)).statistic),
                 "trend_bootstrap_ci": [float(np.quantile(trend_draws, .025)), float(np.quantile(trend_draws, .975))]}

    markov = pd.read_csv(RESULT_ROOT / "markov_wave_test.csv")
    markov_paired = markov.pivot_table(index=["repeat", "group", "target"], columns="model", values="response_distance_rho")
    markov_summary = {}
    for target in markov.target.unique():
        table = markov_paired.xs(target, level="target")
        immediate = table["ImmediatePrecedingWaveOracle"]
        baselines = [column for column in table if column != "ImmediatePrecedingWaveOracle" and table[column].notna().any()]
        markov_summary[target] = {baseline: float((immediate - table[baseline]).mean()) for baseline in baselines}

    boot = source_bootstrap.set_index("comparison")
    frozen = config["early_gate"]
    checks = {
        "correct_minus_static_point": float(means.CorrectLag - means.StaticControl) >= frozen["minimum_correct_lag_minus_static_geometry_rho"],
        "correct_minus_static_ci": float(boot.loc["CorrectLag - StaticControl", "ci_low"]) > 0,
        "correct_minus_same_wave": float(means.CorrectLag - means.SameWave) > frozen["minimum_correct_lag_minus_same_wave_geometry_rho"],
        "correct_minus_temporal_shuffle": float(means.CorrectLag - means.TemporalShuffle) > frozen["minimum_correct_lag_minus_temporal_shuffle_geometry_rho"],
        "correct_minus_identity_shuffle": float(means.CorrectLag - means.GeneIdentityShuffle) > frozen["minimum_correct_lag_minus_identity_shuffle_geometry_rho"],
        "dose_100_minus_0_ci": dose_test["delta_ci"][0] > 0,
        "dose_monotonic_means": dose_test["dose_mean_spearman"] > .8,
    }
    decision = "PROCEED_TO_PART_F" if all(checks.values()) else "STOP_AFTER_PART_E"
    result = {"created_at": now(), "decision": decision, "checks": checks,
              "source_bootstrap": source_bootstrap_rows, "dose_test": dose_test, "markov_paired_deltas": markov_summary,
              "source_influence": {"maximum_absolute_change": float(influence.change_after_leaving_source_out.abs().max()),
                                   "minimum_leave_one_source_out_delta": float(influence.leave_source_out_mean_delta.min()),
                                   "maximum_leave_one_source_out_delta": float(influence.leave_source_out_mean_delta.max())},
              "claim_boundary": "Parts C oracle-wave models assess predictive temporal dependence, not zero-shot inference. Part D is the zero-shot all-days-heldout experiment."}
    atomic_json(RESULT_ROOT / "parts_cde_gate.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
