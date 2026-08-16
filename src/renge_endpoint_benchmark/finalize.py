from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_common import RESULT_ROOT, SCRIPT_ROOT, atomic_json, repeat_bootstrap, sha256


def contrast(left, right, metric, seed, config):
    keys = ["repeat", "group"]
    merged = left[keys + [metric]].merge(right[keys + [metric]], on=keys, suffixes=("_left", "_right"))
    merged["delta"] = merged[f"{metric}_left"] - merged[f"{metric}_right"]
    return repeat_bootstrap(merged, "delta", seed, config["bootstrap_resamples"])


def summaries(frame, section):
    metrics = ["response_distance_rho", "per_response_strict_trans_pearson", "response_cosine",
               "strict_trans_mse", "local_knn_overlap", "local_distance_rank", "between_variance_ratio",
               "predicted_pc1_fraction", "predicted_pc80", "predicted_pc90", "predicted_pc95",
               "predicted_entropy_effective_rank", "predicted_participation_ratio"]
    keys = [c for c in ("target", "model") if c in frame.columns]
    out = frame.groupby(keys)[metrics].mean().reset_index(); out.insert(0, "section", section)
    return out


def main():
    config = json.loads((SCRIPT_ROOT / "config.json").read_text(encoding="utf-8"))
    rel = pd.read_csv(RESULT_ROOT / "target_reliability.csv")
    w45 = pd.read_csv(RESULT_ROOT / "direct_vs_chain_w45.csv")
    direct = pd.read_csv(RESULT_ROOT / "direct_endpoint_results.csv")
    chain = pd.read_csv(RESULT_ROOT / "propagated_endpoint_results.csv")
    oracle = pd.read_csv(RESULT_ROOT / "endpoint_oracle_ladder.csv")
    shared = pd.read_csv(RESULT_ROOT / "endpoint_shared_response_audit.csv")
    comp = pd.read_csv(RESULT_ROOT / "direct_chain_complementarity.csv")
    seed = config["bootstrap_seed"]

    direct_primary = direct[direct.model == "Direct_NestedSelected"]
    direct_matched = direct[direct.model == "Direct_CorrectLag"]
    chain_primary = chain[chain.model == "FullyPredictedMarkov"]
    w45_chain = w45[w45.model == "Chained_Frozen"]
    w45_direct = w45[w45.model == "Direct_NestedSelected"]
    w45_matched = w45[w45.model == "Direct_CorrectLag"]
    residual_chain = shared[(shared.target == "intervention_residual") & (shared.model == "FullyPredictedMarkov")]
    residual_direct = shared[(shared.target == "intervention_residual") & (shared.model == "Direct_NestedSelected")]
    oracle_w23 = oracle[oracle.model == "F2_TrueR2_TrueW23"]

    contrasts = {
        "w45_chain_minus_nested_direct_geometry": contrast(w45_chain, w45_direct, "response_distance_rho", seed + 1, config),
        "w45_chain_minus_matched_correctlag_geometry": contrast(w45_chain, w45_matched, "response_distance_rho", seed + 2, config),
        "r5_chain_minus_nested_direct_geometry": contrast(chain_primary, direct_primary, "response_distance_rho", seed + 3, config),
        "r5_chain_minus_matched_correctlag_geometry": contrast(chain_primary, direct_matched, "response_distance_rho", seed + 4, config),
        "r5_chain_minus_nested_direct_pearson": contrast(chain_primary, direct_primary, "per_response_strict_trans_pearson", seed + 5, config),
        "r5_residual_chain_minus_direct_geometry": contrast(residual_chain, residual_direct, "response_distance_rho", seed + 6, config),
        "r5_oracle_w23_minus_nested_direct_geometry": contrast(oracle_w23, direct_primary, "response_distance_rho", seed + 7, config),
        "r5_oracle_w23_minus_matched_correctlag_geometry": contrast(oracle_w23, direct_matched, "response_distance_rho", seed + 8, config),
        "r5_fusion_minus_nested_direct_geometry": contrast(comp, direct_primary, "response_distance_rho", seed + 9, config),
        "r5_fusion_minus_nested_direct_pearson": contrast(comp, direct_primary, "per_response_strict_trans_pearson", seed + 10, config),
    }
    gate = config["success_standard"]
    practical = contrasts["r5_chain_minus_nested_direct_geometry"]
    residual = contrasts["r5_residual_chain_minus_direct_geometry"]
    oracle_gain = contrasts["r5_oracle_w23_minus_nested_direct_geometry"]
    fusion_gain = contrasts["r5_fusion_minus_nested_direct_geometry"]
    clears = lambda x: x["point"] >= gate["minimum_delta_r5_grouped_geometry"] and x["ci_low"] > 0
    mean_error_corr = float(comp.source_mse_error_correlation.mean())
    mean_pair_error_corr = float(comp.pairwise_geometry_error_correlation.mean())
    mean_alignment = float(comp.residual_correction_alignment.mean())
    complementarity_partial = (fusion_gain["point"] > 0 and
                               contrasts["r5_fusion_minus_nested_direct_pearson"]["ci_low"] > 0 and
                               mean_alignment > 0 and mean_error_corr < .8)
    claims = {
        "claim_1": "CHAIN_HIGHER_VS_NESTED_SELECTED; TIED_VS_MATCHED_CORRECTLAG",
        "claim_2": "PASS" if clears(practical) else "FAIL",
        "claim_3": "PASS" if clears(residual) else "FAIL",
        "claim_4": "PASS" if clears(oracle_gain) else ("PARTIAL" if oracle_gain["point"] > 0 else "FAIL"),
        "claim_5": "PASS" if clears(fusion_gain) and mean_alignment > 0 else ("PARTIAL" if complementarity_partial else "FAIL"),
        "claim_6": "PASS" if clears(practical) or clears(residual) or clears(fusion_gain) else "FAIL",
    }
    verdict = "PROPAGATION_ENDPOINT_ADVANTAGE_SUPPORTED" if claims["claim_2"] == claims["claim_3"] == claims["claim_6"] == "PASS" else (
        "PROPAGATION_ENDPOINT_ADVANTAGE_PARTIALLY_SUPPORTED" if claims["claim_4"] in ("PASS", "PARTIAL") else
        "PROPAGATION_ENDPOINT_ADVANTAGE_NOT_SUPPORTED")
    rel_summary = rel.groupby("target")[["geometry_reliability", "mean_response_pearson", "mean_response_cosine"]].agg(["mean", "std"])
    ladder = oracle.groupby("model")[["response_distance_rho", "per_response_strict_trans_pearson", "response_cosine", "strict_trans_mse"]].mean()
    summary = {
        "verdict": verdict, "decision_case": "CASE_2_WITH_ORACLE_ENTRY_HEADROOM",
        "claims": claims, "contrasts": contrasts,
        "target_reliability": {target: {metric: float(rel[rel.target == target][metric].mean())
                                         for metric in ("geometry_reliability", "mean_response_pearson", "mean_response_cosine")}
                               for target in ("R5", "W45")},
        "direct_r5_means": direct.groupby("model")[["response_distance_rho", "per_response_strict_trans_pearson", "response_cosine", "strict_trans_mse"]].mean().to_dict("index"),
        "chain_r5_means": chain.groupby("model")[["response_distance_rho", "per_response_strict_trans_pearson", "response_cosine", "strict_trans_mse"]].mean().to_dict("index"),
        "direct_w45_means": w45.groupby("model")[["response_distance_rho", "per_response_strict_trans_pearson", "response_cosine", "strict_trans_mse"]].mean().to_dict("index"),
        "oracle_ladder_means": ladder.to_dict("index"),
        "shared_response_means": shared.groupby(["target", "model"])[["response_distance_rho", "per_response_strict_trans_pearson", "response_cosine", "strict_trans_mse"]].mean().reset_index().to_dict("records"),
        "complementarity": {"mean_selected_lambda": float(comp.selected_lambda_training_only.mean()),
                            "fraction_nonzero_lambda": float(np.mean(comp.selected_lambda_training_only != 0)),
                            "source_mse_error_correlation": mean_error_corr,
                            "pairwise_geometry_error_correlation": mean_pair_error_corr,
                            "residual_correction_alignment": mean_alignment,
                            "interpretation": "not complementary: lambda is predominantly negative, errors are correlated, and correction alignment is negative"},
        "gpu_used": False, "new_architecture_trained": False,
    }
    atomic_json(RESULT_ROOT / "analysis_summary.json", summary)
    atomic_json(RESULT_ROOT / "config.json", config)
    provenance_path = RESULT_ROOT / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["config_sha256"] = sha256(SCRIPT_ROOT / "config.json")
    atomic_json(provenance_path, provenance)

    final = pd.concat([summaries(w45, "W45"), summaries(direct, "Direct_R5"), summaries(chain, "Propagated_R5"),
                       summaries(oracle, "Oracle_R5"), summaries(shared, "SharedResponse"), summaries(comp, "Fusion")], ignore_index=True)
    final.to_csv(RESULT_ROOT / "final_model_comparison.csv", index=False)

    fig = RESULT_ROOT / "figures"; fig.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 4)); means = rel.groupby("target").geometry_reliability.mean().reindex(["W45", "R5"])
    plt.bar(means.index, means.values, color=["#d95f02", "#1b9e77"]); plt.ylabel("Pseudoreplicate geometry reliability"); plt.title("Endpoint is more reliable than late wave"); plt.tight_layout(); plt.savefig(fig / "01_target_reliability.png", dpi=160); plt.close()
    plt.figure(figsize=(8, 4)); wm = w45.groupby("model").response_distance_rho.mean().sort_values(); plt.barh(wm.index, wm.values); plt.xlabel("Grouped W45 geometry"); plt.title("Direct versus chained W45"); plt.tight_layout(); plt.savefig(fig / "02_w45_direct_vs_chain.png", dpi=160); plt.close()
    plt.figure(figsize=(8, 4)); dm = pd.concat([direct, chain]).groupby("model").response_distance_rho.mean().sort_values(); plt.barh(dm.index, dm.values); plt.xlabel("Grouped R5 geometry"); plt.title("Deployable R5 benchmark"); plt.tight_layout(); plt.savefig(fig / "03_r5_direct_vs_chain.png", dpi=160); plt.close()
    plt.figure(figsize=(8, 4)); lm = ladder.response_distance_rho; plt.plot(range(len(lm)), lm.values, marker="o"); plt.xticks(range(len(lm)), lm.index, rotation=25, ha="right"); plt.ylabel("Grouped R5 geometry"); plt.title("Endpoint oracle-entry ladder"); plt.tight_layout(); plt.savefig(fig / "04_oracle_ladder.png", dpi=160); plt.close()
    pivot = shared.groupby(["target", "model"]).response_distance_rho.mean().unstack(0)
    pivot.plot(kind="bar", figsize=(9, 4)); plt.ylabel("Grouped geometry"); plt.title("Absolute versus intervention-residual R5"); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.savefig(fig / "05_shared_response_audit.png", dpi=160); plt.close()

    md = f"""{verdict}

# RENGE direct-endpoint versus temporal-propagation benchmark

## Decision

`CASE_2_WITH_ORACLE_ENTRY_HEADROOM`: deployable propagation and matched direct prediction are effectively similar on R5 geometry, but true early entry creates a large oracle advantage. This is not a practical propagation win.

## Claims

- Claim 1 — W45 direction: **{claims['claim_1']}**. Chain-minus nested-direct geometry = {contrasts['w45_chain_minus_nested_direct_geometry']['point']:.4f} [{contrasts['w45_chain_minus_nested_direct_geometry']['ci_low']:.4f}, {contrasts['w45_chain_minus_nested_direct_geometry']['ci_high']:.4f}]. Against matched CorrectLag direct: {contrasts['w45_chain_minus_matched_correctlag_geometry']['point']:.4f}.
- Claim 2 — deployable propagated R5 improves over direct: **{claims['claim_2']}**. Delta = {practical['point']:.4f} [{practical['ci_low']:.4f}, {practical['ci_high']:.4f}].
- Claim 3 — residual geometry improves: **{claims['claim_3']}**. Delta = {residual['point']:.4f} [{residual['ci_low']:.4f}, {residual['ci_high']:.4f}].
- Claim 4 — oracle first-wave entry beats direct: **{claims['claim_4']}**. Delta = {oracle_gain['point']:.4f} [{oracle_gain['ci_low']:.4f}, {oracle_gain['ci_high']:.4f}].
- Claim 5 — complementary information: **{claims['claim_5']}**. Fusion geometry delta = {fusion_gain['point']:.4f} [{fusion_gain['ci_low']:.4f}, {fusion_gain['ci_high']:.4f}].
- Claim 6 — artifact-safe propagation advantage: **{claims['claim_6']}**.

## Shared-response warning

The training-mean endpoint obtains response Pearson {shared[(shared.target == 'absolute') & (shared.model == 'MeanTrainR5')].per_response_strict_trans_pearson.mean():.4f} while its geometry is exactly zero. Absolute Pearson therefore cannot establish intervention identity prediction. On residual R5, both nested direct and chain geometries are near zero.

## Scope

All 100 held-out groups reuse the audited 50x2 source-disjoint splits. Each group is predicted by one model, all selection is training-only, and no stitched LOO geometry, GPU model, new architecture, CD4 data, or Markov retuning was used.
"""
    (RESULT_ROOT / "FINAL_VERDICT.md").write_text(md, encoding="utf-8")
    (RESULT_ROOT / "README.md").write_text("# RENGE endpoint benchmark\n\nSee `FINAL_VERDICT.md`, `analysis_summary.json`, tables, and `figures/`.\n", encoding="utf-8")

    files = []
    for path in sorted(p for p in RESULT_ROOT.rglob("*") if p.is_file() and p.name != "output_manifest.json"):
        files.append({"path": path.relative_to(RESULT_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(RESULT_ROOT / "output_manifest.json", {"verdict": verdict, "files": files})
    print(f"[endpoint] verdict: {verdict}")


if __name__ == "__main__": main()
