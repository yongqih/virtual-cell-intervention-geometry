"""Prepare condition means from the processed Replogle RPE1 single-cell matrix."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def prepare(source, out):
    out.mkdir(parents=True, exist_ok=True)
    config = Path(__file__).resolve().parents[1] / "model_configurations/rpe1_folds.json"
    folds = json.loads(config.read_text())["folds"]
    included = {"control"}.union(*(set(fold["train"]) for fold in folds))
    with h5py.File(source) as h5:
        obs = h5["obs/perturbation"]
        labels = obs["categories"].asstr()[:][obs["codes"][:]]
        genes = h5["var/_index"].asstr()[:]
        conditions = [label for label in dict.fromkeys(labels) if label in included]
        if set(conditions) != included:
            raise ValueError("The source matrix is missing training conditions")
        position = {name: i for i, name in enumerate(conditions)}
        counts = np.zeros(len(conditions), dtype=int)
        sums = np.zeros((len(conditions), len(genes)), dtype=np.float64)
        x = h5["X"]
        ptr = x["indptr"][:]
        edges = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
        for first, last in zip(edges[:-1], edges[1:]):
            if labels[first] not in position:
                continue
            row = position[labels[first]]
            counts[row] += last - first
            for start in range(first, last, 512):
                a, b = ptr[start], ptr[min(last, start + 512)]
                sums[row] += np.bincount(x["indices"][a:b], weights=x["data"][a:b],
                                         minlength=len(genes))
    # Keep original condition order for the truncated PCA. Shards are independently readable.
    for part, rows in enumerate(np.array_split(np.arange(len(conditions)), 4)):
        with h5py.File(out / f"rpe1_means_{part}.h5", "w") as block:
            block["conditions"] = np.array(conditions, dtype="S")[rows]
            block["genes"] = np.array(genes, dtype="S")
            block["cell_counts"] = counts[rows]
            block.create_dataset("expression_sum", data=sums[rows],
                                 chunks=(1, len(genes)), compression="gzip", shuffle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.h5ad, args.output)
