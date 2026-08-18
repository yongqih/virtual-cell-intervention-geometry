"""Isolated replay of the frozen clean synthetic directional capacity control."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from common import OUT, ROOT, sha256_file


SOURCE = ROOT / "results" / "clean_synthetic_directional_control"
DEST = OUT / "experiment_c_synthetic"
FIG4 = OUT / "fig4_orientation_code"


def load_frozen_module():
    script = SOURCE / "synthetic_directional_control.py"
    spec = importlib.util.spec_from_file_location("frozen_synthetic_directional_control", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen script: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in ["SYNTHETIC_CONTROL_PLAN.md", "INFORMATION_BOUNDARY.md", "oracle_information_boundary.json"]:
        shutil.copy2(SOURCE / name, DEST / name)
    module = load_frozen_module()
    module.OUT = DEST
    module.WORLDS = DEST / "worlds"
    module.FIG = DEST / "figures"
    module.main()

    metrics = pd.read_csv(DEST / "per_world_metrics.csv")
    summary = pd.read_csv(DEST / "overall_graph_condition_summary.csv")
    correct = metrics[metrics.condition == "CORRECT_DIRECTED_SIGNED"].set_index("world")
    controls = ["NO_GRAPH", "REVERSED_DIRECTED_SIGNED", "DEGREE_PRESERVING_SHUFFLE", "SIGN_SHUFFLED"]
    rows = []
    for control in controls:
        other = metrics[metrics.condition == control].set_index("world")
        for world in correct.index:
            rows.append({
                "world": int(world),
                "comparison": f"CORRECT_DIRECTED_SIGNED_minus_{control}",
                "geometry_difference": float(correct.loc[world, "response_distance_correlation"] - other.loc[world, "response_distance_correlation"]),
            })
    pd.DataFrame(rows).to_csv(FIG4 / "fig4f_synthetic_signed_structure.csv", index=False)
    shutil.copy2(DEST / "figures" / "geometry_recovery_by_condition.png", FIG4 / "fig4f_synthetic_signed_structure_qc.png")

    metadata = json.loads((DEST / "run_metadata.json").read_text(encoding="utf-8"))
    provenance = {
        "execution": "isolated replay of frozen historical implementation",
        "source_script": str((SOURCE / "synthetic_directional_control.py").resolve()),
        "source_script_sha256": sha256_file(SOURCE / "synthetic_directional_control.py"),
        "historical_results_read": False,
        "historical_results_modified": False,
        "frozen_generator_seeds": module.GENERATOR_SEEDS,
        "frozen_conditions": module.CONDITIONS,
        "frozen_generator_config": vars(module.GC),
        "frozen_model_config": vars(module.MC),
        "verdict": metadata["verdict"],
        "summary_rows": len(summary),
    }
    (DEST / "replay_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
