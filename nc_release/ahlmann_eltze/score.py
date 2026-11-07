"""Recalculate the scores and paired uncertainty for Supplementary Table 5."""
import argparse
from pathlib import Path

from numba import njit, prange, set_num_threads
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from table5 import AE, TX, TRANSFER, PRIMARY, SECONDARY, HERE, METRICS, build


def geometry(x):
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ValueError("Cosine similarity is undefined for a zero response")
    normalized = x / norms[:, None]
    similarity = normalized @ normalized.T
    pairs = similarity[np.triu_indices(len(x), 1)]
    order = np.argsort(pairs, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(pairs[order])) + 1, len(order)]
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argsort(-similarity, axis=1, kind="stable")[:, :10]
    return pairs, order, starts, neighbors


@njit
def midpoint_ranks(order, starts, weights, total):
    # Each tied group receives its weighted midrank, centered on the grand mean.
    ranks = np.empty(len(weights))
    previous = 0.
    for group in range(len(starts) - 1):
        begin, end = starts[group], starts[group + 1]
        mass = 0.
        for j in range(begin, end):
            mass += weights[order[j]]
        rank = previous + (mass - total) / 2
        for j in range(begin, end):
            ranks[order[j]] = rank
        previous += mass
    return ranks


@njit(parallel=True)
def paired_rho(weights, first, second, truth_order, truth_starts, pred_order, pred_starts):
    values = np.empty(len(weights))
    for b in prange(len(weights)):
        pair_weights = weights[b, first] * weights[b, second]
        total = pair_weights.sum()
        a = midpoint_ranks(truth_order, truth_starts, pair_weights, total)
        y = midpoint_ranks(pred_order, pred_starts, pair_weights, total)
        covariance, xvar, yvar = 0., 0., 0.
        for j in range(len(pair_weights)):
            mass = pair_weights[j]
            covariance += mass * a[j] * y[j]
            xvar += mass * a[j] * a[j]
            yvar += mass * y[j] * y[j]
        values[b] = covariance / np.sqrt(xvar * yvar)
    return values


def score(pred, truth, reference, control, weights, measured):
    a, order, starts, neighbors = geometry(pred - reference)
    t, torder, tstarts, expected = measured
    overlap = (neighbors[:, :, None] == expected[:, None, :]).any(axis=2).mean(axis=1)
    distance = cdist(pred + control, truth + control)
    correspondence = (distance > distance.diagonal()[:, None]).sum(axis=1) / (len(pred) - 1)
    point = [spearmanr(a, t).statistic, overlap.mean(), correspondence.mean()]
    first, second = np.triu_indices(len(pred), 1)
    draws = [paired_rho(weights, first, second, torder, tstarts, order, starts),
             weights @ overlap / len(pred), weights @ correspondence / len(pred)]
    return np.asarray(point), np.stack(draws)


def evaluate(data, predictions, output):
    rows, all_draws = [], {}
    with np.load(data / "transfer/BOOTSTRAP_WEIGHTS.npz") as bootstrap:
        for f in range(5):
            with (np.load(data / f"transfer/TxPert_fold{f}_response.npz") as tx,
                  np.load(data / f"subspaces/biology_fold{f}.npz") as bio,
                  np.load(predictions / f"fold{f}.npz") as ae):
                names, genes, cols = tx["names"], tx["genes"], tx["readout_columns"]
                for actual, expected in [(ae["names"], names), (ae["genes"], genes),
                                         (bio["genes"][cols], genes), (bootstrap[f"f{f}_names"], names)]:
                    np.testing.assert_array_equal(actual, expected)
                idx = pd.Index(bio["held_names"]).get_indexer(names)
                if np.any(idx < 0):
                    raise ValueError(f"Fold {f}: missing measured responses")
                truth = bio["held_truth"][idx][:, cols] + bio["reference"][cols]
                weights = bootstrap[f"f{f}_weights"]
                if weights.shape != (1000, len(names)) or np.any(weights.sum(axis=1) != len(names)):
                    raise ValueError(f"Fold {f}: invalid bootstrap weights")
                control = tx["control"]
                methods = {AE: ae["response"], TX: tx["response"]}
            reused, fold_points = {}, {}
            for panel in range(5):
                with np.load(data / f"transfer/direct/fold{f}_panel{panel}.npz") as z:
                    np.testing.assert_array_equal(z["names"], names)
                    np.testing.assert_array_equal(z["genes"], genes)
                    if len(z["training_names"]) != 32 or set(names) & set(z["training_names"]):
                        raise ValueError(f"Fold {f}: unexpected calibration set")
                    methods[TRANSFER] = z["ResidualLinearCV"]
                    refs = {PRIMARY: z["reference11"], SECONDARY: z["reference_full"]}
                for label, ref in refs.items():
                    measured = geometry(truth - ref)
                    for method, pred in methods.items():
                        key = (label, method)
                        # Fixed zero-shot predictions share one full-training reference across panels.
                        reuse_key = (method, ref.tobytes())
                        if method != TRANSFER and reuse_key in reused:
                            point, draws = reused[reuse_key]
                        else:
                            point, draws = score(pred, truth, ref, control, weights, measured)
                            if method != TRANSFER:
                                reused[reuse_key] = point, draws
                        fold_points.setdefault(key, []).append(point)
                        for metric, draw in zip(METRICS, draws):
                            all_draws.setdefault("|".join((label, method, metric)), []).append(draw)
            for (ref, method), values in fold_points.items():
                rows.append(dict(fold=f, reference=ref, method=method, n_query=len(names),
                                 n_readout=len(genes), **dict(zip(METRICS, np.mean(values, axis=0)))))
            print(f"Scored fold {f}: {len(names)} interventions", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["fold", "reference", "method"]).to_csv(output / "fold_scores.csv", index=False)
    np.savez_compressed(output / "paired_draws.npz", **{k: np.mean(v, axis=0) for k, v in all_draws.items()})
    build(output).to_csv(output / "Table_5.csv", index=False, float_format="%.12g")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Extracted v2.0.0 data directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--predictions", type=Path, default=HERE / "predictions")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for the paired bootstrap")
    args = parser.parse_args()
    set_num_threads(args.threads)
    with threadpool_limits(limits=1, user_api="blas"):
        evaluate(args.data, args.predictions, args.output)
