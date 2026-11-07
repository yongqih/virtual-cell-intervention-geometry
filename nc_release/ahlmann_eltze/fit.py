"""Fit the no-anchor linear model using only the RPE1 training conditions in each fold."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent


def training_matrix(folder, allowed):
    names, rows = [], []
    genes = None
    for path in sorted(folder.glob("rpe1_means_*.h5")):
        with h5py.File(path) as part:
            axis = part["genes"].asstr()[:]
            if genes is not None and not np.array_equal(genes, axis):
                raise ValueError(f"Inconsistent gene axis in {path}")
            genes = axis
            conditions = part["conditions"].asstr()[:]
            indices = np.flatnonzero(np.isin(conditions, list(allowed)))
            if len(indices):
                names.extend(conditions[indices])
                # Select training rows before loading any measured expression.
                counts = part["cell_counts"][indices]
                rows.append(part["expression_sum"][indices, :] / counts[:, None])
    if len(names) != len(allowed) or set(names) != allowed:
        raise ValueError("Missing or duplicated RPE1 training conditions")
    return genes, names, np.concatenate(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Extracted v2.0.0 data directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--profiles", type=Path, default=HERE / "profiles")
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--fold", type=int, choices=range(5), help="Default: fit all five folds")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    folds = json.loads((args.data / "model_configurations/rpe1_folds.json").read_text())["folds"]
    for fold in folds:
        f = fold["fold"]
        if args.fold is not None and args.fold != f:
            continue
        train, valid, test = (set(fold[k]) for k in ("train", "validation", "test"))
        with np.load(args.data / f"transfer/fold{f}_identities.npz") as design:
            queries, readouts = design["held"], design["readout"]
        if train & (valid | test) or valid & test or not set(queries) <= test:
            raise ValueError(f"Fold {f}: inconsistent intervention split")
        genes, names, expression = training_matrix(args.profiles, train | {"control"})
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            names[names.index("control")] = "ctrl"
            for label, values in [("genes", genes), ("conditions", names), ("queries", queries),
                                  ("readouts", readouts)]:
                (work / f"{label}.txt").write_text("\n".join(values) + "\n", encoding="utf-8")
            expression.astype("<f8", copy=False).tofile(work / "expression.f64")
            subprocess.run([args.rscript, "--vanilla", str(HERE / "fit.R"), str(work)], check=True)
            prediction = np.fromfile(work / "response.f64", dtype="<f8").reshape(len(queries), len(readouts))
        np.savez_compressed(args.output / f"fold{f}.npz", names=queries, genes=readouts, response=prediction)
        print(f"Fold {f}: {len(train)} training interventions, {len(queries)} predictions", flush=True)


if __name__ == "__main__":
    main()
