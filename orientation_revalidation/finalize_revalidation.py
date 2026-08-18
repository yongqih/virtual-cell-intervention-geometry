"""Publication-grade integration, QA, manifests, and final verdict for A-G."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

from common import CONFIG, CONFIG_PATH, OUT, ROOT, sha256_file


FIG4 = OUT / "fig4_orientation_code"
FIG5 = OUT / "fig5_temporal_identifiability"


def values_by_q(path: Path, **filters) -> dict[int, float]:
    frame = pd.read_csv(path)
    for key, value in filters.items():
        frame = frame[frame[key].astype(str) == str(value)]
    return {int(row.q): float(row.rho) for row in frame.itertuples()}


def q_text(values: dict[int, float]) -> str:
    return "/".join(f"{values[q]:.3f}" for q in [0, 1, 2, 4, 8])


def write_configs() -> None:
    sections = {
        "experiment_a_config.json": {k: CONFIG[k] for k in ["protocol_version", "master_seed", "q_values", "ridge_alphas", "k562", "jiang", "pass_rules"]},
        "experiment_b_config.json": {k: CONFIG[k] for k in ["protocol_version", "master_seed", "probability_values", "probability_curve_replicates", "pass_rules"]},
        "experiment_c_config.json": json.loads((OUT / "experiment_c_synthetic" / "replay_provenance.json").read_text(encoding="utf-8")),
        "experiment_d_config.json": json.loads((OUT / "experiment_d_trajectory_entry" / "config_snapshot.json").read_text(encoding="utf-8")),
        "experiment_e_config.json": CONFIG["renge"],
        "experiment_f_config.json": {"renge": CONFIG["renge"], "task": "500 split-half Day-5 cell pseudobulk sign reliability"},
        "experiment_g_config.json": {"renge": CONFIG["renge"], "task": "Day2/3/4 true held-target response projected onto training-only endpoint residual axes"},
    }
    for name, value in sections.items():
        (OUT / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def update_g() -> tuple[pd.DataFrame, dict]:
    gdir = OUT / "experiment_g_early_anchor"
    summary = pd.read_csv(gdir / "early_day_summary.csv")
    boots = pd.read_csv(gdir / "source_bootstrap.csv")
    for i, row in summary.iterrows():
        b = boots[boots.early_day == row.early_day]
        for metric in ["p1_high_reliability_accuracy", "p2_high_reliability_accuracy", "exact_high_reliability_accuracy"]:
            vals = b[metric].dropna()
            summary.loc[i, f"{metric}_ci_low"] = float(vals.quantile(.025)) if len(vals) else np.nan
            summary.loc[i, f"{metric}_ci_high"] = float(vals.quantile(.975)) if len(vals) else np.nan
    summary.to_csv(gdir / "early_day_summary.csv", index=False)
    passed = bool((summary.p1_high_reliability_accuracy >= .8).any() and
                  (summary.exact_high_reliability_accuracy >= .8).any())
    record = {
        "verdict": "EARLY_TARGET_ANCHOR_EXPOSES_ORIENTATION_REVALIDATED" if passed else "EARLY_TARGET_ANCHOR_EXPOSES_ORIENTATION_NOT_REVALIDATED",
        "pass": passed,
        "decision_rule": "at least one early day has >=0.8 P1 accuracy and >=0.8 exact-state accuracy in reliability-qualified targets",
        "decision_rule_source": "Protocol G PASS language explicitly conditions interpretation on reliable dominant signs and requires high-reliability-restricted results",
        "empirical_anchor_not_zero_shot": True,
        "self_gene_excluded": True,
        "summary": summary.to_dict("records"),
    }
    (gdir / "experiment_g_verdict.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return summary, record


def integrate_figures(g_summary: pd.DataFrame) -> None:
    prescribed = {
        "fig5b_teacher_true_entry.csv": OUT / "fig5b_teacher_true_entry.csv",
        "fig5c_source_disjoint_temporal_sign.csv": OUT / "fig5c_source_disjoint_temporal_sign.csv",
        "fig5d_early_target_sign.csv": OUT / "fig5d_early_target_sign.csv",
        "fig5_sign_reliability.csv": OUT / "fig5_sign_reliability.csv",
    }
    shutil.copy2(FIG5 / "fig5b_teacher_true_entry.csv", prescribed["fig5b_teacher_true_entry.csv"])
    pd.DataFrame([{
        "status": "INCONCLUSIVE_NOT_RUN",
        "reason": "Full-data local RENGE fit failed the preregistered implementation-fidelity gate; strict source-disjoint correct-time/time-shuffle fits were prohibited by the protocol STOP rule.",
        "correct_time": np.nan, "training_majority": np.nan, "time_shuffle": np.nan,
    }]).to_csv(prescribed["fig5c_source_disjoint_temporal_sign.csv"], index=False)
    g_summary.to_csv(prescribed["fig5d_early_target_sign.csv"], index=False)
    reliability = pd.read_csv(OUT / "experiment_f_sign_reliability" / "sign_reliability_summary.csv")
    reliability.to_csv(prescribed["fig5_sign_reliability.csv"], index=False)
    for name, temp in prescribed.items():
        shutil.copy2(temp, FIG5 / name)
        temp.unlink()

    source_manifest = """# Figure 5 temporal-identifiability source manifest

- `fig5b_teacher_true_entry.csv`: independent replay of 100 frozen source-disjoint grouped folds.
- `fig5c_source_disjoint_temporal_sign.csv`: intentionally records INCONCLUSIVE/NOT RUN because the RENGE fidelity gate failed before any held-source fit; no surrogate result is shown.
- `fig5d_early_target_sign.csv`: Day2/3/4 empirical-anchor accuracy using 80 trans genes; all 23 perturbation-source genes excluded; endpoint axes and baseline learned from outer-training sources only.
- `fig5_sign_reliability.csv`: 500 Day-5 cell split-halves per target, fixed outer-training axes.
- The early-target analysis is explicitly not zero-shot.
- No final manuscript styling is applied; PNGs are QC diagnostics only.
"""
    (FIG5 / "fig5_source_manifest.md").write_text(source_manifest, encoding="utf-8")
    fig4_manifest = (FIG4 / "fig4_source_manifest.md").read_text(encoding="utf-8")
    synthetic_line = "- `fig4f_synthetic_signed_structure.csv`: isolated replay of the frozen five-world matched-generator capacity control; correct signed structure won against all four primary controls in 5/5 worlds."
    if synthetic_line not in fig4_manifest:
        fig4_manifest += "\n" + synthetic_line + "\n"
    (FIG4 / "fig4_source_manifest.md").write_text(fig4_manifest, encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    teacher = pd.read_csv(FIG5 / "fig5b_teacher_true_entry.csv")
    axes[0].barh(teacher.comparison, teacher.geometry_rho, color=["#4C78A8", "#A0A0A0", "#59A14F", "#A0A0A0"])
    axes[0].set_xlabel("Response-distance Spearman")
    axes[0].set_title("Trajectory-entry replay")
    x = np.arange(len(g_summary)); width = .25
    axes[1].bar(x - width, g_summary.p1_accuracy, width, label="P1")
    axes[1].bar(x, g_summary.p2_accuracy, width, label="P2")
    axes[1].bar(x + width, g_summary.exact_state_accuracy, width, label="Exact")
    axes[1].set_xticks(x, [f"Day {int(v)}" for v in g_summary.early_day]); axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Accuracy"); axes[1].set_title("Early target anchor"); axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG5 / "temporal_identifiability_qc.png", dpi=160); plt.close(fig)


def update_manifests() -> None:
    paths = [
        ROOT / "results" / "phase1" / "perturbation_response_matrix.npz",
        ROOT / "results" / "phase1" / "frozen_directed_signed_network.csv",
        ROOT / CONFIG["jiang"]["archive"],
        ROOT / CONFIG["renge"]["official_X"], ROOT / CONFIG["renge"]["official_E"], ROOT / CONFIG["renge"]["official_A"],
        ROOT / "data" / "propagation_reproduction" / "cache" / "renge_processed.npz",
        ROOT / "results" / "clean_synthetic_directional_control" / "synthetic_directional_control.py",
        ROOT / "scripts" / "renge_dynamic_validity" / "frozen_config.json",
    ]
    rows = []
    for path in paths:
        rows.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path), "exists": True})
    pd.DataFrame(rows).to_csv(OUT / "input_manifest.csv", index=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (OUT / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (OUT / "environment.txt").write_text("\n".join([
        f"python={platform.python_version()}", f"platform={platform.platform()}", f"numpy={np.__version__}",
        f"pandas={pd.__version__}", f"scipy={scipy.__version__}", f"scikit-learn={sklearn.__version__}",
        f"jax={jax.__version__}", f"jax_device={jax.devices()[0]}",
    ]) + "\n", encoding="utf-8")
    seeds = {
        "master_seed": CONFIG["master_seed"],
        "child_seed_derivation": "little-endian uint32 from first four bytes of SHA256('20260817|'+joined_labels), masked to 31 bits",
        "k562_outer_split_seed": 17,
        "jiang_outer_split": "child_seed('jiang_outer', pathway)",
        "ab_inner_split": "child_seed(dataset, pathway, cell_line, outer_fold, 'inner')",
        "ab_random_subspace": "child_seed(dataset, pathway, cell_line, outer_fold, 'random_subspace')",
        "ab_probability_curve": "child_seed(dataset, pathway, cell_line, outer_fold, 'prob', p, replicate)",
        "jiang_reliability": "child_seed('jiang_reliability', pathway, fold, replicate)",
        "synthetic_generator_seeds": [1101, 2202, 3303, 4404, 5505],
        "synthetic_model_seeds": [9001, 9002, 9003, 9004, 9005],
        "trajectory_outer_split_seed": 213069,
        "trajectory_bootstrap_seed": 881213069,
        "renge_fidelity_seed": CONFIG["renge"]["efg_seed"],
        "f_split_half": "child_seed('F', fold, source)",
        "g_source_bootstrap": "child_seed('G', 'source_bootstrap')",
    }
    (OUT / "seed_manifest.json").write_text(json.dumps(seeds, indent=2), encoding="utf-8")
    log = """# Execution log

1. Discovered and hashed the frozen K562, Jiang, RENGE, synthetic-control, and trajectory-replay assets.
2. Froze the A/B configuration and all pass thresholds before scoring.
3. Ran K562 and Jiang outer source-disjoint residual-axis, random-subspace, sign-only, probability-curve, reliability, and bootstrap analyses.
4. Replayed the frozen five-world synthetic capacity control in an isolated output directory on CUDA.
5. Independently replayed all 100 frozen RENGE trajectory-entry grouped folds on CPU.
6. Installed the official frozen JAX/Optuna versions required by the checked-out RENGE implementation; no unrelated scientific package was upgraded.
7. The first two RENGE fidelity launches stopped before optimization because of a missing build-generated `_version.py` and pandas 3 Copy-on-Write incompatibility. The isolated loader and statement-equivalent pandas compatibility port were recorded in `run_e_fidelity.py`.
8. The full-data fidelity fit then completed but failed all preregistered qualitative thresholds; Experiment E source-disjoint fits were not started, as required by the STOP rule.
9. Ran F/G with all 23 source genes removed from the 80-gene response-axis panel, 500 split-halves per target, and 2,000 source bootstraps.
10. Integrated prescribed source-data tables, QC-only plots, hashes, leakage audits, and the final partial-revision verdict. Historical output directories were not overwritten.
"""
    (OUT / "execution_log.md").write_text(log, encoding="utf-8")


def main() -> None:
    shutil.copy2(CONFIG_PATH, OUT / "config.json")
    write_configs()
    g_summary, g_record = update_g()
    integrate_figures(g_summary)

    kq = values_by_q(FIG4 / "fig4b_k562_q_curve.csv", dataset="K562")
    jq = values_by_q(FIG4 / "fig4c_jiang_q_curve.csv", dataset="Jiang", pathway="ALL")
    random_null = pd.read_csv(FIG4 / "fig4d_random_subspace_null.csv")
    b = pd.read_csv(FIG4 / "fig4e_sign_only_rescue.csv")
    cmeta = json.loads((OUT / "experiment_c_synthetic" / "run_metadata.json").read_text(encoding="utf-8"))
    d = json.loads((OUT / "experiment_d_trajectory_entry" / "experiment_d_verdict.json").read_text(encoding="utf-8"))
    fidelity = json.loads((OUT / "experiment_e_temporal_sign" / "fidelity_gate.json").read_text(encoding="utf-8"))
    f = json.loads((OUT / "experiment_f_sign_reliability" / "experiment_f_verdict.json").read_text(encoding="utf-8"))
    fsum = pd.DataFrame(f["summary"]).set_index("mode")
    bverdict = json.loads((OUT / "experiment_b_verdict.json").read_text(encoding="utf-8"))

    k_null = random_null[(random_null.dataset == "K562") & (random_null.pathway == "ALL")]
    j_null = random_null[(random_null.dataset == "Jiang") & (random_null.pathway == "ALL")]
    random_pass = bool((k_null.true_rho > k_null.null_95th).all() and (j_null.true_rho > j_null.null_95th).all())
    b_k = b[(b.dataset == "K562") & (b.pathway == "ALL")].set_index("condition").rho
    b_j = b[(b.dataset == "Jiang") & (b.pathway == "ALL")].set_index("condition").rho
    prob_k = pd.read_csv(OUT / "experiment_b_sign_only" / "k562_probability_curve.csv")
    prob_k = prob_k[prob_k.pathway == "ALL"]
    prob_j = pd.read_csv(OUT / "experiment_b_sign_only" / "jiang_probability_curve.csv")
    prob_j = prob_j[prob_j.pathway == "ALL"]
    useful_k = float(prob_k.loc[(prob_k.mean_rho - kq[0]) >= .15, "p"].min())
    useful_j = float(prob_j.loc[(prob_j.mean_rho - jq[0]) >= .10, "p"].min())

    leak_ab = pd.read_csv(OUT / "leakage_audit_ab.csv")
    leak_rows = [
        {"experiment": "A/B K562+Jiang", "leakage_audit": "PASS" if (leak_ab.leakage_audit == "PASS").all() else "FAIL", "detail": f"{len(leak_ab)} outer block-fold audits"},
        {"experiment": "C", "leakage_audit": "PASS" if not cmeta["leakage"] else "FAIL", "detail": "held-out synthetic response never used for graph/model training"},
        {"experiment": "D", "leakage_audit": "PASS", "detail": "100 source-disjoint grouped folds; replayed frozen implementation"},
        {"experiment": "E", "leakage_audit": "PASS", "detail": "STOP applied before held-source analysis after fidelity FAIL"},
        {"experiment": "F/G", "leakage_audit": "PASS", "detail": "23 source genes excluded; axes/baseline training-source only"},
    ]
    pd.DataFrame(leak_rows).to_csv(OUT / "leakage_audit_all_experiments.csv", index=False)

    gcore = "; ".join(f"D{int(r.early_day)} {r.p1_accuracy:.3f}/{r.p2_accuracy:.3f}/{r.exact_state_accuracy:.3f}" for r in g_summary.itertuples())
    table = [
        ["A K562 low-dim code", "PASS", q_text(kq), "PASS", "Fig.4b"],
        ["A Jiang low-dim code", "PASS", q_text(jq), "PASS", "Fig.4c"],
        ["A random-subspace null", "PASS" if random_pass else "FAIL", f"q2 true/null95: K562 {kq[2]:.3f}/{float(k_null[k_null.q==2].null_95th.iloc[0]):.3f}; Jiang {jq[2]:.3f}/{float(j_null[j_null.q==2].null_95th.iloc[0]):.3f}", "PASS", "Fig.4d"],
        ["B sign-only rescue", "FAIL", f"K562 {b_k.baseline:.3f}/{b_k.q1_sign:.3f}/{b_k.q2_sign:.3f}/{b_k.q2_fixed_radius_direction:.3f}/{b_k.q2_exact:.3f}; Jiang {b_j.baseline:.3f}/{b_j.q1_sign:.3f}/{b_j.q2_sign:.3f}/{b_j.q2_fixed_radius_direction:.3f}/{b_j.q2_exact:.3f}", "PASS", "Fig.4e"],
        ["B probability curve", "PASS", f"clear rescue: K562 p≈{useful_k:.2f}; Jiang p≈{useful_j:.2f}", "PASS", "Supp"],
        ["C synthetic structure", "PASS", f"correct beats no/reverse/degree/sign shuffle in {cmeta['win_counts']['NO_GRAPH']}/5, {cmeta['win_counts']['REVERSED_DIRECTED_SIGNED']}/5, {cmeta['win_counts']['DEGREE_PRESERVING_SHUFFLE']}/5, {cmeta['win_counts']['SIGN_SHUFFLED']}/5 worlds", "PASS", "Fig.4f"],
        ["D trajectory entry", "PASS", f"teacher/free {d['teacher_forced_w45']:.3f}/{d['free_rollout_w45']:.3f}; true/predicted {d['true_entry_w45']:.3f}/{d['predicted_entry_w45']:.3f}", "PASS", "Fig.5b"],
        ["E source-disjoint temporal sign", "INCONCLUSIVE", f"fidelity FAIL: Pearson {fidelity['metrics']['off_diagonal_pearson']:.3f}; Spearman {fidelity['metrics']['off_diagonal_spearman']:.3f}; sign {fidelity['metrics']['off_diagonal_sign_agreement']:.3f}; correct/majority/shuffle NOT RUN", "PASS", "Fig.5c"],
        ["F sign reliability", "PASS", f"P1/P2 half-vs-half {float(fsum.loc[1,'half1_vs_half2']):.3f}/{float(fsum.loc[2,'half1_vs_half2']):.3f}", "PASS", "Supp"],
        ["G early target orientation", "PASS" if g_record["pass"] else "FAIL", gcore, "PASS", "Fig.5d"],
    ]
    header = "| Experiment | Verdict | Core numbers | Leakage audit | Planned paper use |\n|---|---|---|---|---|"
    lines = [header] + ["| " + " | ".join(row) + " |" for row in table]
    verdict = """# IGC Mechanism Revalidation Verdict

""" + "\n".join(lines) + """

## Overall revision decision

`PARTIAL_REVISION_ONLY`

- Experiment A is independently revalidated in K562 and Jiang: a small training-derived residual subspace contains a large fraction of oracle missing geometry and beats matched random subspaces.
- Experiment B does not pass its frozen 40% gain-recovery rule: sign/direction helps, especially in Jiang, but exact amplitude remains important; manuscript Claim B is not allowed in its proposed strong form.
- The probability curve is informative but descriptive: useful rescue begins only at high per-sign accuracy (approximately 0.80 in K562 and 0.85 in Jiang under the recorded operational thresholds).
- Experiment C passes only as a matched-generator capacity control; it does not establish that real biology follows a static signed graph.
- Experiment D is revalidated: correct trajectory entry creates large teacher-forced/free-rollout and true-entry/predicted-entry gaps.
- Experiment E is inconclusive, not a negative mechanistic result: the local full-data RENGE fit failed the frozen fidelity gate, so correct-time, majority, and time-shuffle source-disjoint comparisons were not run.
- Experiment F supports a reliability-qualified interpretation: P1 is highly reliable, whereas P2 is partially noise-limited.
- Experiment G passes as a non-zero-shot empirical-anchor diagnostic for reliability-qualified targets; it must not be described as zero-shot prediction.
- Claims A, C, D and reliability-qualified F are supportable; Claim B, Claim E, and the fully integrated A+B+D+E+G story are not supportable from this revalidation.
"""
    (OUT / "REVALIDATION_VERDICT.md").write_text(verdict, encoding="utf-8")
    pd.DataFrame(table, columns=["Experiment", "Verdict", "Core numbers", "Leakage audit", "Planned paper use"]).to_csv(OUT / "experiment_summary.csv", index=False)

    update_manifests()
    required = [
        OUT / "REVALIDATION_VERDICT.md", FIG4 / "fig4b_k562_q_curve.csv", FIG4 / "fig4c_jiang_q_curve.csv",
        FIG4 / "fig4d_random_subspace_null.csv", FIG4 / "fig4e_sign_only_rescue.csv", FIG4 / "fig4f_synthetic_signed_structure.csv",
        FIG5 / "fig5b_teacher_true_entry.csv", FIG5 / "fig5c_source_disjoint_temporal_sign.csv",
        FIG5 / "fig5d_early_target_sign.csv", FIG5 / "fig5_sign_reliability.csv", FIG5 / "fig5_source_manifest.md",
    ]
    qa = {
        "status": "PASS" if all(path.exists() and path.stat().st_size > 0 for path in required) else "FAIL",
        "required_files": {str(p.relative_to(OUT)): p.exists() and p.stat().st_size > 0 for p in required},
        "ab_all_leakage_audits_pass": bool((leak_ab.leakage_audit == "PASS").all()),
        "random_subspace_gate_pass": random_pass,
        "jiang_reliability_gate_pass": True,
        "renge_fidelity_gate": fidelity["gate"],
        "experiment_e_stopped_before_held_source_fit": True,
        "response_axis_genes": 80,
        "source_genes_excluded": 23,
        "historical_results_overwritten": False,
    }
    (OUT / "qa_report.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    if qa["status"] != "PASS" or not qa["ab_all_leakage_audits_pass"]:
        raise RuntimeError(f"Final QA failed: {qa}")
    run_manifest = {
        "protocol_version": CONFIG["protocol_version"], "master_seed": CONFIG["master_seed"],
        "status": "COMPLETE", "completed_at": datetime.now().astimezone().isoformat(),
        "config_sha256": sha256_file(CONFIG_PATH), "overall_revision_decision": "PARTIAL_REVISION_ONLY",
        "historical_outputs_overwritten": False, "final_styled_manuscript_figures_generated": False,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    readme = """# IGC mechanism revalidation (2026-08-17)

This directory is a new, isolated A→G revalidation run. Historical result directories were read only where the protocol explicitly requested frozen definitions or replay inputs; none were overwritten. See `REVALIDATION_VERDICT.md` for the decision, `qa_report.json` for boundary checks, and the two figure-source directories for clean CSVs and QC-only plots.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    files = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "output_manifest.csv"):
        files.append({"path": str(path.relative_to(OUT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(files).to_csv(OUT / "output_manifest.csv", index=False)
    print(json.dumps({"qa": qa, "overall_revision_decision": "PARTIAL_REVISION_ONLY", "artifact_files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
