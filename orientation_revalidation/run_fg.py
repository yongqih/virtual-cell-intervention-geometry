"""Experiments F/G after the RENGE fidelity-gated Experiment E stop."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from common import (CONFIG, CONFIG_PATH, MASTER_SEED, OUT, ROOT, child_seed,
                    fit_ridge_predict, make_round_robin_folds, residual_svd_axes,
                    sha256_file)


E_DIR = OUT / "experiment_e_temporal_sign"
F_DIR = OUT / "experiment_f_sign_reliability"
G_DIR = OUT / "experiment_g_early_anchor"
FIG5 = OUT / "fig5_temporal_identifiability"
CACHE = ROOT / "data" / "propagation_reproduction" / "cache" / "renge_processed.npz"
GRID = tuple(CONFIG["ridge_alphas"])


def select_alpha(x: np.ndarray, y: np.ndarray) -> tuple[float, list[dict]]:
    rows = []
    for alpha in GRID:
        errors = []
        for i in range(len(x)):
            keep = np.arange(len(x)) != i
            pred, _ = fit_ridge_predict(x[keep], y[keep], x[i:i + 1], alpha)
            errors.append(float(np.mean((pred[0] - y[i]) ** 2)))
        rows.append({"alpha": alpha, "inner_loo_mse": float(np.mean(errors))})
    return float(min(rows, key=lambda row: row["inner_loo_mse"])["alpha"]), rows


def fold_axes(train: np.ndarray, test: np.ndarray, static: np.ndarray,
              endpoint: np.ndarray, fold: int):
    oof = np.zeros_like(endpoint[train], dtype=float)
    alpha_rows = []
    for position, source_index in enumerate(train):
        fit_idx = train[train != source_index]
        alpha, scores = select_alpha(static[fit_idx], endpoint[fit_idx])
        oof[position], _ = fit_ridge_predict(static[fit_idx], endpoint[fit_idx], static[source_index:source_index + 1], alpha)
        for row in scores:
            alpha_rows.append({"fold": fold, "purpose": "training_oof_axis", "held_training_source_index": int(source_index),
                               "selected": row["alpha"] == alpha, **row})
    residual = endpoint[train] - oof
    axes, singular = residual_svd_axes(residual)
    alpha, scores = select_alpha(static[train], endpoint[train])
    baseline_test, _ = fit_ridge_predict(static[train], endpoint[train], static[test], alpha)
    for row in scores:
        alpha_rows.append({"fold": fold, "purpose": "outer_query_baseline", "held_training_source_index": -1,
                           "selected": row["alpha"] == alpha, **row})
    return axes, singular, baseline_test, alpha_rows


def sign(values: np.ndarray) -> np.ndarray:
    out = np.sign(values).astype(int)
    out[out == 0] = 1
    return out


def bootstrap_g(frame: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(child_seed("G", "source_bootstrap"))
    for day, part in frame.groupby("early_day"):
        part = part.reset_index(drop=True)
        for b in range(n_boot):
            sample = part.iloc[rng.integers(0, len(part), len(part))]
            row = {"bootstrap": b, "early_day": int(day)}
            for mode in (1, 2):
                row[f"p{mode}_accuracy"] = float(sample[f"p{mode}_correct"].mean())
                row[f"p{mode}_balanced_accuracy"] = float(balanced_accuracy_score(sample[f"p{mode}_true_sign"], sample[f"p{mode}_early_sign"]))
                high = sample[sample[f"p{mode}_reliability"] >= .8]
                row[f"p{mode}_high_reliability_accuracy"] = float(high[f"p{mode}_correct"].mean()) if len(high) else np.nan
            row["exact_state_accuracy"] = float(sample["exact_correct"].mean())
            high_both = sample[(sample.p1_reliability >= .8) & (sample.p2_reliability >= .8)]
            row["exact_high_reliability_accuracy"] = float(high_both.exact_correct.mean()) if len(high_both) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    fidelity = json.loads((E_DIR / "fidelity_gate.json").read_text(encoding="utf-8"))
    if fidelity["gate"] != "FAIL":
        raise RuntimeError("run_fg.py is the fidelity-fail path; Experiment E status was not FAIL")
    F_DIR.mkdir(parents=True, exist_ok=False)
    G_DIR.mkdir(parents=True, exist_ok=False)
    FIG5.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_PATH, OUT / "config.json")

    with np.load(CACHE, allow_pickle=False) as z:
        expression = z["expression"].astype(float)
        assignment = z["assignment"].astype(str)
        times = z["times"].astype(int)
        genes = z["genes"].astype(str)
        sources = z["sources"].astype(str)
        response = z["response"].astype(float)
        controls = z["matched_controls"].astype(float)
        static_all = z["static_control_representation"].astype(float)
        cell_count = z["cell_count"].astype(int)
    lookup = {g: i for i, g in enumerate(genes)}
    source_rows = np.asarray([lookup[s] for s in sources])
    trans = np.asarray([i for i, g in enumerate(genes) if g not in set(sources)], dtype=int)
    static = static_all[source_rows]
    endpoint = response[:, 3][:, trans]
    folds = make_round_robin_folds(len(sources), CONFIG["renge"]["outer_folds"], CONFIG["renge"]["efg_seed"])

    split_rows, alpha_rows, axis_rows, coeff_rows = [], [], [], []
    fold_cache = {}
    for fold, test in enumerate(folds):
        train = np.setdiff1d(np.arange(len(sources)), test)
        axes, singular, baseline, fold_alpha = fold_axes(train, test, static, endpoint, fold)
        alpha_rows.extend(fold_alpha)
        fold_cache[fold] = {"train": train, "test": test, "axes": axes[:2], "baseline": baseline}
        np.savez_compressed(F_DIR / f"fold_{fold}_frozen_axes.npz", axes=axes[:2], singular_values=singular,
                            train_indices=train, test_indices=test, trans_gene_indices=trans, baseline_test=baseline)
        for role, idx in [("train", train), ("test", test)]:
            for i in idx:
                split_rows.append({"fold": fold, "role": role, "source_index": int(i), "source": sources[i],
                                   "all_days_removed_if_test": role == "test"})
        full_coef = (endpoint[test] - baseline) @ axes[:2].T
        for local, source_index in enumerate(test):
            coeff_rows.append({"fold": fold, "source_index": int(source_index), "source": sources[source_index],
                               "p1_full_coefficient": full_coef[local, 0], "p2_full_coefficient": full_coef[local, 1],
                               "p1_full_sign": sign(full_coef[local:local + 1, 0])[0],
                               "p2_full_sign": sign(full_coef[local:local + 1, 1])[0]})
        for mode in (0, 1):
            for gene_index, loading in zip(trans, axes[mode]):
                axis_rows.append({"fold": fold, "mode": mode + 1, "gene": genes[gene_index], "loading": loading})

    pd.DataFrame(split_rows).to_csv(F_DIR / "split_manifest.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(F_DIR / "baseline_alpha_audit.csv", index=False)
    pd.DataFrame(axis_rows).to_csv(F_DIR / "endpoint_axes.csv", index=False)
    coeff = pd.DataFrame(coeff_rows).sort_values("source_index")
    coeff.to_csv(F_DIR / "full_endpoint_coefficients.csv", index=False)

    reliability_rows = []
    reps = CONFIG["renge"]["sign_reliability_replicates"]
    for fold, item in fold_cache.items():
        for local, source_index in enumerate(item["test"]):
            cells = np.flatnonzero((assignment == sources[source_index]) & (times == 5))
            full = coeff[coeff.source_index == source_index].iloc[0]
            rng = np.random.default_rng(child_seed("F", fold, sources[source_index]))
            for rep in range(reps):
                order = rng.permutation(cells)
                half1, half2 = np.array_split(order, 2)
                half_resp = [expression[h].mean(0)[trans] - controls[3, trans] for h in (half1, half2)]
                half_coef = [(value - item["baseline"][local]) @ item["axes"].T for value in half_resp]
                for mode in (0, 1):
                    h1, h2 = sign(np.asarray([half_coef[0][mode], half_coef[1][mode]]))
                    fs = int(full[f"p{mode + 1}_full_sign"])
                    reliability_rows.append({"fold": fold, "source_index": int(source_index), "source": sources[source_index],
                                             "replicate": rep, "mode": mode + 1, "n_day5_cells": len(cells),
                                             "full_coefficient": float(full[f"p{mode + 1}_full_coefficient"]),
                                             "half1_sign": h1, "half2_sign": h2, "full_sign": fs,
                                             "half1_vs_full": int(h1 == fs), "half2_vs_full": int(h2 == fs),
                                             "half1_vs_half2": int(h1 == h2)})
    rel = pd.DataFrame(reliability_rows)
    rel.to_csv(F_DIR / "split_half_replicates.csv", index=False)
    per_target = rel.groupby(["fold", "source_index", "source", "mode", "n_day5_cells", "full_coefficient"], as_index=False)[
        ["half1_vs_full", "half2_vs_full", "half1_vs_half2"]].mean()
    per_target["coefficient_abs"] = per_target.full_coefficient.abs()
    per_target.to_csv(F_DIR / "per_target_sign_reliability.csv", index=False)
    f_summary = per_target.groupby("mode").agg(
        half1_vs_full=("half1_vs_full", "mean"), half2_vs_full=("half2_vs_full", "mean"),
        half1_vs_half2=("half1_vs_half2", "mean"),
        fraction_targets_half_vs_half_above_0_8=("half1_vs_half2", lambda x: float(np.mean(x >= .8))),
        targets=("source", "size")).reset_index()
    f_summary.to_csv(F_DIR / "sign_reliability_summary.csv", index=False)
    f_record = {"status": "COMPLETE", "interpretation": "P2_PARTIALLY_NOISE_LIMITED" if float(f_summary.loc[f_summary['mode'] == 2, 'half1_vs_half2'].iloc[0]) < .8 else "BOTH_MODES_RELIABLE",
                "cache_sha256": sha256_file(CACHE), "response_axis_genes": len(trans), "source_genes_excluded": len(sources),
                "replicates_per_target": reps, "summary": f_summary.to_dict("records")}
    (F_DIR / "experiment_f_verdict.json").write_text(json.dumps(f_record, indent=2), encoding="utf-8")

    rel_lookup = {(int(r.source_index), int(r.mode)): float(r.half1_vs_half2) for r in per_target.itertuples()}
    g_rows = []
    for fold, item in fold_cache.items():
        for local, source_index in enumerate(item["test"]):
            full = coeff[coeff.source_index == source_index].iloc[0]
            for day_index, day in enumerate([2, 3, 4]):
                early_coef = (response[source_index, day_index, trans] - item["baseline"][local]) @ item["axes"].T
                es = sign(early_coef)
                ts = np.asarray([int(full.p1_full_sign), int(full.p2_full_sign)])
                g_rows.append({"fold": fold, "source_index": int(source_index), "source": sources[source_index], "early_day": day,
                               "p1_true_sign": ts[0], "p2_true_sign": ts[1], "p1_early_sign": es[0], "p2_early_sign": es[1],
                               "p1_early_coefficient": early_coef[0], "p2_early_coefficient": early_coef[1],
                               "p1_correct": int(es[0] == ts[0]), "p2_correct": int(es[1] == ts[1]),
                               "exact_correct": int(np.array_equal(es, ts)),
                               "p1_reliability": rel_lookup[(int(source_index), 1)],
                               "p2_reliability": rel_lookup[(int(source_index), 2)]})
    g = pd.DataFrame(g_rows)
    g.to_csv(G_DIR / "early_source_predictions.csv", index=False)
    boots = bootstrap_g(g, CONFIG["renge"]["source_bootstrap_replicates"])
    boots.to_csv(G_DIR / "source_bootstrap.csv", index=False)
    summary_rows = []
    for day, part in g.groupby("early_day"):
        row = {"early_day": int(day), "sources": len(part)}
        for mode in (1, 2):
            row[f"p{mode}_accuracy"] = float(part[f"p{mode}_correct"].mean())
            row[f"p{mode}_balanced_accuracy"] = float(balanced_accuracy_score(part[f"p{mode}_true_sign"], part[f"p{mode}_early_sign"]))
            high = part[part[f"p{mode}_reliability"] >= .8]
            row[f"p{mode}_high_reliability_accuracy"] = float(high[f"p{mode}_correct"].mean()) if len(high) else np.nan
            row[f"p{mode}_high_reliability_sources"] = len(high)
        row["exact_state_accuracy"] = float(part.exact_correct.mean())
        hb = part[(part.p1_reliability >= .8) & (part.p2_reliability >= .8)]
        row["exact_high_reliability_accuracy"] = float(hb.exact_correct.mean()) if len(hb) else np.nan
        row["exact_high_reliability_sources"] = len(hb)
        bpart = boots[boots.early_day == day]
        for metric in ["p1_accuracy", "p2_accuracy", "p1_balanced_accuracy", "p2_balanced_accuracy", "exact_state_accuracy"]:
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = np.quantile(bpart[metric], [.025, .975])
        summary_rows.append(row)
    g_summary = pd.DataFrame(summary_rows)
    g_summary.to_csv(G_DIR / "early_day_summary.csv", index=False)
    # Protocol G is explicitly conditioned on reliable dominant labels and asks
    # for a high-reliability restricted analysis; P2-unreliable targets must not
    # be used to turn that reliability-qualified criterion into a universal one.
    g_pass = bool((g_summary.p1_high_reliability_accuracy >= .8).any() and
                  (g_summary.exact_high_reliability_accuracy >= .8).any())
    g_record = {"verdict": "EARLY_TARGET_ANCHOR_EXPOSES_ORIENTATION_REVALIDATED" if g_pass else "EARLY_TARGET_ANCHOR_EXPOSES_ORIENTATION_NOT_REVALIDATED",
                "pass": g_pass, "empirical_anchor_not_zero_shot": True, "self_gene_excluded": True,
                "summary": g_summary.to_dict("records")}
    (G_DIR / "experiment_g_verdict.json").write_text(json.dumps(g_record, indent=2), encoding="utf-8")

    pd.DataFrame([{"status": "NOT_RUN", "reason": "RENGE implementation-fidelity gate failed before source-disjoint fitting"}]).to_csv(
        FIG5 / "fig5c_renge_sign_controls.csv", index=False)
    f_summary.to_csv(FIG5 / "fig5_supp_sign_reliability.csv", index=False)
    g_summary.to_csv(FIG5 / "fig5d_early_anchor.csv", index=False)
    pd.DataFrame(split_rows).to_csv(G_DIR / "split_manifest.csv", index=False)
    print(json.dumps({"experiment_f": f_record, "experiment_g": g_record}, indent=2))


if __name__ == "__main__":
    main()
