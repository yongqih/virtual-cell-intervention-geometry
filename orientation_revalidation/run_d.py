"""Independent, isolated replay of the frozen RENGE trajectory-entry audit."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd

from common import OUT, ROOT, sha256_file


SCRIPT_DIR = ROOT / "scripts" / "renge_dynamic_validity"
DEST = OUT / "experiment_d_trajectory_entry"
FIG5 = OUT / "fig5_temporal_identifiability"


def main() -> None:
    if DEST.exists():
        raise RuntimeError(f"Refusing to overwrite replay output: {DEST}")
    DEST.mkdir(parents=True)
    FIG5.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SCRIPT_DIR))
    dynamic_common = importlib.import_module("dynamic_common")
    dynamic_common.RESULT_ROOT = DEST
    diagnostics = importlib.import_module("run_diagnostics")
    diagnostics.RESULT_ROOT = DEST
    diagnostics.main()
    finalize = importlib.import_module("finalize")
    finalize.RESULT_ROOT = DEST
    finalize.main()

    anchors = pd.read_csv(DEST / "rollout_anchor_decomposition.csv")
    means = anchors.groupby(["anchor", "stage"], as_index=False)["response_distance_rho"].mean()
    table = means.pivot(index="anchor", columns="stage", values="response_distance_rho")
    values = {
        "teacher_forced_w45": float(table.loc["C_TEACHER_FORCED_SECOND_STEP", "W45"]),
        "free_rollout_w45": float(table.loc["B_PREDICTED_ENTRY", "W45"]),
        "true_entry_w45": float(table.loc["A_TRUE_ENTRY", "W45"]),
        "predicted_entry_w45": float(table.loc["B_PREDICTED_ENTRY", "W45"]),
    }
    values["teacher_minus_free"] = values["teacher_forced_w45"] - values["free_rollout_w45"]
    values["true_minus_predicted_entry"] = values["true_entry_w45"] - values["predicted_entry_w45"]
    passed = values["teacher_minus_free"] >= 0.20 and values["true_minus_predicted_entry"] >= 0.20
    verdict = "TRAJECTORY_ENTRY_BOTTLENECK_REVALIDATED" if passed else "TRAJECTORY_ENTRY_BOTTLENECK_NOT_REVALIDATED"
    pd.DataFrame([
        {"comparison": "teacher_forced", "geometry_rho": values["teacher_forced_w45"]},
        {"comparison": "free_rollout", "geometry_rho": values["free_rollout_w45"]},
        {"comparison": "true_entry", "geometry_rho": values["true_entry_w45"]},
        {"comparison": "predicted_entry", "geometry_rho": values["predicted_entry_w45"]},
    ]).to_csv(FIG5 / "fig5b_teacher_true_entry.csv", index=False)
    record = {
        "verdict": verdict,
        "pass": passed,
        "prespecified_large_gap_threshold": 0.20,
        **values,
        "independent_replay": True,
        "source_cache": str(dynamic_common.FROZEN_CACHE.resolve()),
        "source_cache_sha256": sha256_file(dynamic_common.FROZEN_CACHE),
        "frozen_diagnostic_config_sha256": sha256_file(SCRIPT_DIR / "frozen_config.json"),
        "historical_result_directory_modified": False,
    }
    (DEST / "experiment_d_verdict.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
