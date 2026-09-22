"""Evaluate user-supplied response matrices with the released IGC geometry function."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


def evaluate(input_path: Path, output_path: Path) -> dict:
    code = Path(__file__).resolve().parents[1]
    source = code / "audits/main_geometry_integrity_audit/run_audit.py"
    spec = importlib.util.spec_from_file_location("released_igc_geometry", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(input_path, allow_pickle=False) as data:
        required = {"prediction", "truth", "group", "source_id", "gene_names"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Missing arrays: {sorted(missing)}")
        pred = np.asarray(data["prediction"], dtype=float)
        truth = np.asarray(data["truth"], dtype=float)
        groups = np.asarray(data["group"]).astype(str)
        sources = np.asarray(data["source_id"]).astype(str)
        genes = np.asarray(data["gene_names"]).astype(str)
    if pred.ndim != 2 or pred.shape != truth.shape or pred.shape[0] < 4 or pred.shape[1] < 2:
        raise ValueError("prediction and truth must be aligned N x G matrices, N >= 4 and G >= 2")
    if groups.shape != (len(pred),) or sources.shape != (len(pred),):
        raise ValueError("group and source_id must each contain N entries")
    if genes.shape != (pred.shape[1],) or len(set(genes)) != len(genes):
        raise ValueError("gene_names must contain G unique entries")
    if set(sources) & set(genes):
        raise ValueError("Strict-trans readout must exclude every evaluated target gene")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("Response matrices must contain finite values")
    if (np.linalg.norm(pred, axis=1) <= 1e-12).any() or (np.linalg.norm(truth, axis=1) <= 1e-12).any():
        raise ValueError("Zero response vectors have undefined cosine geometry")
    rows = []
    for group in np.unique(groups):
        selected = groups == group
        count = int(selected.sum())
        if count < 4 or len(set(sources[selected])) != count:
            raise ValueError(f"Group {group!r} needs at least four distinct interventions")
        score = module.geometry(pred[selected], truth[selected])
        rows.append({"group": group, "n_interventions": count,
                     "n_within_group_pairs": count * (count - 1) // 2,
                     "geometry_spearman": score})
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "grouped_geometry.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"n_groups": len(rows), "n_interventions": len(pred),
               "n_genes": pred.shape[1], "cross_group_pairs_used": False,
               "mean_group_geometry": float(np.mean([r["geometry_spearman"] for r in rows]))}
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.input, args.output), indent=2))
