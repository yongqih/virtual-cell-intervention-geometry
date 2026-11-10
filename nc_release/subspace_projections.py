"""Reproduce the rank-eight projections and model-level tests in Supplementary Fig. 7."""
import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from analysis_metrics import cosine_similarity
from replay import MODELS, ca


RANK = 8
RANDOM_DRAWS = 1000
OVERLAP = "overlap_with_measured_training_subspace"
ENDPOINTS = {"rho": "projected_global_rho", "top10": "projected_Top10", "CA": "projected_CA"}
TRAIN_MODELS = ("GEARS", "Scouter", "TxPert-public", "Biology")


def exact_tests(scores):
    if len(scores) != 7 or set(scores["model"]) != set(MODELS):
        raise ValueError("The exact test requires one mean for each of the seven models.")
    values = scores[[OVERLAP, *ENDPOINTS.values()]].to_numpy()
    if not np.isfinite(values).all() or (np.ptp(values, axis=0) == 0).any():
        raise ValueError("Each endpoint and the overlap must contain finite, nonconstant values.")
    # Twice the centered average ranks are integers, including in the presence of ties.
    ranks = np.rint(2 * rankdata(values, axis=0) - 8).astype(int)
    x = ranks[:, 0]
    permuted_x = np.array(list(permutations(x)))
    rows = []
    for endpoint, y in zip(ENDPOINTS, ranks[:, 1:].T):
        observed = x @ y
        count = np.count_nonzero(abs(permuted_x @ y) >= abs(observed))
        rows.append(dict(endpoint=endpoint, n_models=len(x),
                         spearman_r=observed / (np.linalg.norm(x) * np.linalg.norm(y)),
                         extreme_permutations=count, n_permutations=len(permuted_x),
                         p_raw=count / len(permuted_x)))
    table = pd.DataFrame(rows)
    order = table["p_raw"].sort_values().index
    adjusted = table.loc[order, "p_raw"].to_numpy() * np.arange(len(rows), 0, -1)
    table.loc[order, "p_holm"] = np.minimum(1, np.maximum.accumulate(adjusted))
    return table


def response_basis(profiles):
    centered = profiles - profiles.mean(axis=0)
    _, singular_values, directions = svd(centered, full_matrices=False, check_finite=False)
    if singular_values[RANK - 1] <= 1e-10 * singular_values[0]:
        raise ValueError("Fewer than eight supported response directions.")
    return directions[:RANK].T


def geometry(profiles):
    similarity = cosine_similarity(profiles)
    ranks = rankdata(similarity[np.triu_indices(len(profiles), 1)])
    ranks -= ranks.mean()
    distances = -similarity
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :10]
    return ranks / np.linalg.norm(ranks), neighbors


def projection_scores(responses, axes, origin, observed, reference_geometry):
    coordinates = responses @ axes
    reconstructed = coordinates @ axes.T
    ranks, neighbors = geometry(coordinates)
    true_ranks, true_neighbors = reference_geometry
    common = (neighbors[:, :, None] == true_neighbors[:, None, :]).any(axis=2)
    total_energy = np.sum(responses ** 2)
    return dict(rho=ranks @ true_ranks, top10=common.mean(),
                relative_mse=np.sum((responses - reconstructed) ** 2) / total_energy,
                energy=np.sum(coordinates ** 2) / total_energy,
                CA=ca(origin + reconstructed, observed))


def reproduce(data, output):
    by_model, training, random_scores = [], [], []
    random_seeds = np.random.SeedSequence([20261109, 0]).spawn(2)
    for fold in range(5):
        with np.load(data / f"subspaces/biology_fold{fold}.npz") as bio:
            truth, origin = bio["held_truth"], bio["origin"]
            names, genes = bio["held_names"], bio["genes"]
            biology_axes = response_basis(bio["train_truth"])
        reference_geometry = geometry(truth)
        for model in MODELS:
            with np.load(data / f"benchmark/{model}_fold{fold}.npz") as predictions:
                np.testing.assert_array_equal(predictions["names"], names)
                np.testing.assert_array_equal(predictions["genes"], genes)
                observed = predictions["truth"]
                np.testing.assert_array_equal(observed - origin, truth)
                axes = response_basis(predictions["pred"] - origin)
            scores = projection_scores(truth, axes, origin, observed, reference_geometry)
            by_model.append({"fold": fold, "model": model,
                             OVERLAP: np.sum((axes.T @ biology_axes) ** 2) / RANK,
                             **{column: scores[key] for key, column in ENDPOINTS.items()}})

        if fold < 2:
            with np.load(data / f"subspaces/train_bases_fold{fold}_full.npz") as saved:
                np.testing.assert_array_equal(saved["genes"], genes)
                for model in TRAIN_MODELS:
                    scores = projection_scores(truth, saved[model][:, :RANK], origin,
                                               observed, reference_geometry)
                    training.append(dict(scope=f"fold{fold}", basis=model, **scores))
            rng = np.random.default_rng(random_seeds[fold])
            for draw in range(RANDOM_DRAWS):
                axes, triangular = np.linalg.qr(rng.standard_normal((len(genes), RANK)))
                axes *= np.sign(np.diag(triangular))
                scores = projection_scores(truth, axes, origin, observed, reference_geometry)
                random_scores.append(dict(scope=f"fold{fold}", draw=draw, **scores))
        print(f"Completed fold {fold + 1} of 5", flush=True)

    folds = pd.DataFrame(by_model)
    means = folds.drop(columns="fold").groupby("model", as_index=False).mean()
    folds.to_csv(output / "s7_projection_by_fold.csv", index=False)
    means.to_csv(output / "s7_seven_model_projections.csv", index=False)
    trained = pd.DataFrame(training)
    average = trained.drop(columns="scope").groupby("basis", as_index=False).mean()
    pd.concat([trained, average.assign(scope="equal_fold_mean")]).to_csv(
        output / "s7_training_axis_scores.csv", index=False)
    draws = pd.DataFrame(random_scores)
    paired = draws.drop(columns="scope").groupby("draw", as_index=False).mean()
    draws = pd.concat([draws, paired.assign(scope="equal_fold_mean")])
    summaries = []
    for scope, group in draws.groupby("scope", sort=False):
        for metric in ["rho", "top10", "relative_mse", "energy", "CA"]:
            values = group[metric]
            summaries.append(dict(scope=scope, metric=metric, mean=values.mean(),
                                  p025=values.quantile(.025), p975=values.quantile(.975),
                                  minimum=values.min(), maximum=values.max(), draws=len(values)))
    pd.DataFrame(summaries).to_csv(output / "s7_random_axis_reference.csv", index=False)
    return means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data", type=Path, help="Extracted Zenodo data directory")
    inputs.add_argument("--model-scores", type=Path, help="Seven-model projection means CSV")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1):
        scores = (reproduce(args.data, args.out) if args.data is not None
                  else pd.read_csv(args.model_scores))
        tests = exact_tests(scores)
    tests.to_csv(args.out / "s7_axis_alignment_tests.csv", index=False)
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
