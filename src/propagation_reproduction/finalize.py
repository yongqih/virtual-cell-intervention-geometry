from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULT_ROOT, atomic_json, now


METRICS = (
    "response_distance_rho",
    "per_response_strict_trans_pearson",
    "strict_trans_mse",
    "between_variance_ratio",
    "predicted_pc1_fraction",
    "truth_pc1_fraction",
    "predicted_pc80",
    "truth_pc80",
    "predicted_pc90",
    "truth_pc90",
    "predicted_pc95",
    "truth_pc95",
    "predicted_participation_ratio",
    "truth_participation_ratio",
    "predicted_entropy_effective_rank",
    "truth_entropy_effective_rank",
)


def aggregate(frame: pd.DataFrame, section: str, keys: list[str]) -> pd.DataFrame:
    available = [metric for metric in METRICS if metric in frame.columns]
    rows: list[dict] = []
    for values, group in frame.groupby(keys, dropna=False, sort=False):
        if not isinstance(values, tuple):
            values = (values,)
        row = {"section": section, **dict(zip(keys, values)), "n_outer_groups": len(group)}
        for metric in available:
            numbers = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"mean_{metric}"] = float(numbers.mean())
            row[f"median_{metric}"] = float(numbers.median())
            row[f"std_{metric}"] = float(numbers.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    figure_root = RESULT_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    ablation = pd.read_csv(RESULT_ROOT / "first_responder_ablation.csv")
    markov = pd.read_csv(RESULT_ROOT / "markov_wave_test.csv")
    low_rank = pd.read_csv(RESULT_ROOT / "low_rank_operator.csv")
    alignment = pd.read_csv(RESULT_ROOT / "alignment_results.csv")
    program = pd.read_csv(RESULT_ROOT / "program_propagation.csv")
    credit = pd.read_csv(RESULT_ROOT / "credit_head_results.csv")
    temporal = pd.read_csv(RESULT_ROOT / "temporal_structure.csv")
    reliability = pd.read_csv(RESULT_ROOT / "pseudoreplicate_reliability.csv")
    dose = pd.read_csv(RESULT_ROOT / "information_dose_curve_bootstrap.csv")

    selected_program = program[
        (program["model"] == "GeneLevelDense")
        | (program["is_selected_dimension"].astype(str).str.lower() == "true")
    ].copy()

    comparisons = pd.concat(
        [
            aggregate(ablation, "D_first_response_zero_shot", ["model", "target"]),
            aggregate(markov, "C_oracle_temporal_dependence", ["model", "target"]),
            aggregate(low_rank, "F_transition_operator", ["model", "target"]),
            aggregate(alignment, "G_alignment", ["model", "target"]),
            aggregate(selected_program, "H_rollout", ["model", "mode", "dimension"]),
            aggregate(credit, "I_credit_head", ["model", "target"]),
        ],
        ignore_index=True,
        sort=False,
    )
    comparisons.to_csv(RESULT_ROOT / "final_model_comparison.csv", index=False)

    rollout = selected_program.copy()
    interpretation = {
        "OracleTrueW23_to_W34": "transition only; true first wave supplied",
        "TeacherForcedTrueW34_to_W45": "late transition only; true preceding wave supplied",
        "FreeTrueW23_to_W45": "two transitions; true first wave supplied",
        "FreePredictedW23_to_W34": "first-wave inference plus one transition",
        "FreePredictedW23_to_W45": "full free rollout: inferred first wave plus two transitions",
    }
    rollout.insert(3, "decomposition_component", rollout["mode"].map(interpretation))
    rollout.to_csv(RESULT_ROOT / "rollout_decomposition.csv", index=False)

    pd.DataFrame(
        [{
            "status": "NOT_RUN_GATE_STOP",
            "stage": "K_noise_dose_rollout_and_L_gpu_propagator",
            "reason": "Best conditional credit head did not improve marginal lag by the preregistered +0.03 geometry threshold.",
            "best_credit_head": "DenseRidgeCredit",
            "best_credit_minus_marginal_geometry_rho": -0.0038765471675919483,
            "frozen_threshold": 0.03,
            "gpu_training_started": False,
            "gears_gpu_monitoring_performed": False,
        }]
    ).to_csv(RESULT_ROOT / "noise_dose_rollout.csv", index=False)

    # Figure 1: observed response/wave geometry.
    cross = temporal[temporal["record_type"].isin(["cross_time_geometry", "cross_wave_geometry"])]
    labels = [f"{a}–{b}" for a, b in zip(cross["matrix_a"], cross["matrix_b"])]
    plt.figure(figsize=(9, 4))
    plt.bar(labels, cross["value"], color=["#4C78A8"] * 6 + ["#F58518"] * 3)
    plt.axhline(0, color="black", linewidth=.8)
    plt.ylabel("Strict-trans distance Spearman ρ")
    plt.title("Observed temporal response geometry")
    plt.xticks(rotation=35, ha="right")
    save_figure(figure_root / "01_temporal_geometry.png")

    # Figure 2: cell pseudoreplicate reliability.
    matrix_order = ["W23", "W34", "W45"]
    values = [reliability.loc[reliability["matrix"] == matrix, "geometry_reliability_rho"] for matrix in matrix_order]
    plt.figure(figsize=(6, 4))
    plt.boxplot(values, tick_labels=matrix_order, showfliers=False)
    plt.axhline(.1, color="#E45756", linestyle="--", linewidth=1, label="frozen gate")
    plt.ylabel("Pseudoreplicate geometry reliability ρ")
    plt.title("Wave reliability across 50 cell splits")
    plt.legend(frameon=False)
    save_figure(figure_root / "02_pseudoreplicate_reliability.png")

    # Figure 3: zero-shot first-wave ablations.
    ablation_means = ablation.groupby("model", sort=False)["response_distance_rho"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    colors = ["#54A24B" if name == "CorrectLag" else "#B9B9B9" for name in ablation_means.index]
    plt.bar(ablation_means.index, ablation_means.values, color=colors)
    plt.axhline(0, color="black", linewidth=.8)
    plt.ylabel("Grouped held-out geometry ρ")
    plt.title("Unseen-source first-wave prediction and controls")
    plt.xticks(rotation=30, ha="right")
    save_figure(figure_root / "03_first_responder_ablation.png")

    # Figure 4: evidence dose response with repeat-bootstrap intervals.
    plt.figure(figsize=(6, 4))
    plt.plot(dose["evidence_fraction"], dose["mean_geometry_rho"], marker="o", color="#4C78A8")
    plt.fill_between(dose["evidence_fraction"], dose["ci_low"], dose["ci_high"], color="#4C78A8", alpha=.2)
    plt.xlabel("Fraction of intermediate evidence retained")
    plt.ylabel("Grouped held-out geometry ρ")
    plt.title("Intermediate-evidence dose curve")
    save_figure(figure_root / "04_information_dose_curve.png")

    # Figure 5: selected program and gene-level rollout decomposition.
    rollout_means = selected_program.groupby(["model", "mode"])["response_distance_rho"].mean().unstack(0)
    rollout_means = rollout_means.reindex(list(interpretation))
    rollout_means.plot(kind="bar", figsize=(10, 4), color=["#4C78A8", "#F58518"])
    plt.axhline(0, color="black", linewidth=.8)
    plt.ylabel("Grouped held-out geometry ρ")
    plt.xlabel("")
    plt.title("Teacher-forced versus free rollout")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Model", frameon=False)
    save_figure(figure_root / "05_rollout_decomposition.png")

    # Figure 6: conditional credit gate.
    credit_means = credit.groupby("model", sort=False)["response_distance_rho"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    colors = ["#54A24B" if name == "MarginalLag" else "#B9B9B9" for name in credit_means.index]
    plt.bar(credit_means.index, credit_means.values, color=colors)
    plt.ylabel("Grouped held-out geometry ρ")
    plt.title("Conditional credit heads do not clear the frozen gate")
    plt.xticks(rotation=30, ha="right")
    save_figure(figure_root / "06_credit_head_gate.png")

    verdict = """PROPAGATION_PRINCIPLE_PARTIALLY_SUPPORTED

## Decision

The independent RENGE reproduction supports a narrow perturbation-wave principle: other perturbations' correctly ordered intermediate responses contain transferable information about the first population response of a completely unseen perturbation source. It does **not** support an end-to-end learned wave propagator that preserves held-out response geometry through free rollout.

The primary endpoint is the within-held-out-group, pair-specific strict-trans cosine-distance Spearman correlation. Each of 50 repeated two-fold source-disjoint splits holds the source out at every day, and every group is predicted by one fitted model. A synthetic audit showed why concatenated leave-one-source-out geometry is invalid: a no-source mean predictor scored 1.000 naively but 0.000 under the grouped metric.

## Six preregistered claims

| # | Claim | Score | Evidence |
|---|---|---|---|
| 1 | Intermediate behavior predicts the unseen source's first response | **PASS** | Correct lag ρ=0.1721 versus static 0.0556; source-bootstrap delta 0.1165, 95% CI [0.0506, 0.1540]. Held-out source absent at Days 2–5. |
| 2 | Correct temporal order matters beyond same-wave/shuffled controls | **PASS** | Correct−same-wave delta 0.0459, CI [0.0222, 0.0623]; correct−temporal-shuffle 0.0576, CI [0.0260, 0.0823]. |
| 3 | Signal grows with intermediate-evidence dose | **PASS** | Mean ρ: 0%,25%,50%,75%,100% = 0.0000, 0.0743, 0.1176, 0.1375, 0.1721; dose-rank ρ=1.0; 100%−0% CI [0.1436, 0.2003]. |
| 4 | Conditional credit assignment improves the marginal propagator | **FAIL** | Best credit head 0.1682 versus marginal 0.1721 (delta −0.0039), below the frozen +0.03 gate. |
| 5 | A learned propagator preserves geometry in free rollout | **FAIL** | Dense teacher-forced W34→W45 ρ=0.4284, but full free W23→W45 ρ=0.0427. The dominant problem is first-wave inference/exposure, not merely one-step transition capacity. |
| 6 | A final dynamic propagated endpoint model improves over direct/static baselines | **FAIL** | Not demonstrated: no candidate passed the credit gate, so GPU attention and endpoint benchmarking were not authorized. |

## Supporting and limiting observations

- Wave pseudoreplicate median reliabilities were 0.691 (W23), 0.570 (W34), and 0.558 (W45), so the basic wave geometry is measurable.
- Oracle temporal prediction supports predictive dependence, not causality or true zero-shot inference: immediate-wave ρ was 0.374 for W23→W34 and 0.415 for W34→W45.
- The dense transition operator worked better than the low-rank operator. The tested low-rank parameterization did not consistently beat identity/same-wave/shuffled controls.
- Orthogonal Procrustes alignment preserved Euclidean geometry numerically but did not restore held-out response-distance geometry (ρ≈0.001).
- Leave-one-source influence analysis retained a positive correct-lag minus static gap for every omitted source (minimum 0.0755), arguing against a single-source artifact.
- These are population pseudobulk response waves, not within-cell trajectories. The result is predictive and mechanistic in ordering/control structure, but not proof of causal molecular transmission.

## Stop decision

The frozen conditional-credit gate stopped Parts K–L. `noise_dose_rollout.csv` records `NOT_RUN_GATE_STOP`; no GPU model was launched and no GEARS monitoring was performed, as GEARS had already completed.

## Data and provenance

- Dataset: GEO [GSE213069](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213069), 14,945 cells, 103 modeled genes, 23 KO sources, Days 2–5.
- Paper: Ishikawa et al., *Communications Biology* (2023), [DOI 10.1038/s42003-023-05594-4](https://doi.org/10.1038/s42003-023-05594-4).
- Official code/data example: [masastat/RENGE](https://github.com/masastat/RENGE) pinned at `ca0d636ae47311fb5ce501f4a0a835b55379d9fa`; input SHA-256 values are recorded in `data_provenance.json`.
"""
    (RESULT_ROOT / "FINAL_VERDICT.md").write_text(verdict, encoding="utf-8")

    readme = """# Perturbation-wave propagation reproduction

This directory contains the complete CPU-first audit generated from the official pinned RENGE example. See `FINAL_VERDICT.md` for the scientific decision, `final_model_comparison.csv` for matched metrics, and `figures/` for compact diagnostics.

Population waves are adjacent-day differences of matched-control pseudobulk responses, not single-cell trajectories. The primary evaluation is 50 repeated two-fold perturbation-source-disjoint splits with one fitted model per held-out group. The GPU stage was gated off after conditional credit heads failed to improve the marginal lag model.
"""
    (RESULT_ROOT / "README.md").write_text(readme, encoding="utf-8")

    manifest_files = sorted(
        path for path in RESULT_ROOT.rglob("*")
        if path.is_file() and path.name != "output_manifest.json"
    )
    manifest = []
    for path in manifest_files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append({"path": path.relative_to(RESULT_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest})
    atomic_json(RESULT_ROOT / "output_manifest.json", {
        "created_at": now(),
        "verdict": "PROPAGATION_PRINCIPLE_PARTIALLY_SUPPORTED",
        "gpu_training_started": False,
        "files": manifest,
    })
    print("[完成] 最终汇总、图形、门控记录和科学判定已写入", RESULT_ROOT)


if __name__ == "__main__":
    main()
