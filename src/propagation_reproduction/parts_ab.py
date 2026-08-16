from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd

from common import (CACHE_ROOT, RESULT_ROOT, atomic_json, atomic_npz, direct_indices, now, rank_metrics,
                    safe_pearson, safe_spearman, strict_trans_cosine_distance)


MATRICES = ("R_Day2", "R_Day3", "R_Day4", "R_Day5", "W23", "W34", "W45")


def geometry_similarity(left: np.ndarray, right: np.ndarray, direct: np.ndarray) -> float:
    left_distance = strict_trans_cosine_distance(left, direct)
    right_distance = strict_trans_cosine_distance(right, direct)
    upper = np.triu_indices(len(left), 1)
    return safe_spearman(left_distance[upper], right_distance[upper])


def split_means(values: np.ndarray, assignment: np.ndarray, times: np.ndarray, sources: np.ndarray,
                seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed); output = []
    for replicate in range(2):
        response = np.empty((len(sources), 4, values.shape[1]), dtype=np.float32)
        for time_index, day in enumerate((2, 3, 4, 5)):
            control_rows = np.flatnonzero((assignment == "control") & (times == day))
            shuffled = rng.permutation(control_rows); halves = np.array_split(shuffled, 2)
            control = values[halves[replicate]].mean(0)
            for source_index, source in enumerate(sources):
                rows = np.flatnonzero((assignment == source) & (times == day))
                shuffled = rng.permutation(rows); halves = np.array_split(shuffled, 2)
                response[source_index, time_index] = values[halves[replicate]].mean(0) - control
        output.append(response)
    return output[0], output[1]


def main() -> None:
    config = json.loads((__import__("pathlib").Path(__file__).with_name("frozen_config.json")).read_text())
    with np.load(CACHE_ROOT / "renge_processed.npz", allow_pickle=False) as archive:
        values = archive["expression"].astype(np.float32); assignment = archive["assignment"].astype(str)
        times = archive["times"].astype(int); sources = archive["sources"].astype(str)
        genes = archive["genes"].astype(str); response = archive["response"].astype(np.float32)
        waves = archive["waves"].astype(np.float32); cell_count = archive["cell_count"].copy()
    matrices = {**{f"R_Day{day}": response[:, index] for index, day in enumerate((2, 3, 4, 5))},
                **{name: waves[:, index] for index, name in enumerate(("W23", "W34", "W45"))}}
    direct = direct_indices(sources, genes); rows = []; distances = {}
    for name, matrix in matrices.items():
        metrics = rank_metrics(matrix, direct)
        distance = strict_trans_cosine_distance(matrix, direct); distances[name] = distance.astype(np.float32)
        upper = np.triu_indices(len(matrix), 1)
        metrics.update({"mean_pairwise_distance": float(distance[upper].mean()),
                        "median_pairwise_distance": float(np.median(distance[upper]))})
        rows.extend({"record_type": "matrix_structure", "matrix_a": name, "matrix_b": "",
                     "metric": metric, "value": value, "n_sources": len(sources)} for metric, value in metrics.items())
    response_names = [f"R_Day{day}" for day in (2, 3, 4, 5)]
    for first, second in combinations(response_names, 2):
        upper = np.triu_indices(len(sources), 1)
        rows.append({"record_type": "cross_time_geometry", "matrix_a": first, "matrix_b": second,
                     "metric": "strict_trans_distance_spearman",
                     "value": safe_spearman(distances[first][upper], distances[second][upper]), "n_sources": len(sources)})
    for first, second in combinations(("W23", "W34", "W45"), 2):
        upper = np.triu_indices(len(sources), 1)
        rows.append({"record_type": "cross_wave_geometry", "matrix_a": first, "matrix_b": second,
                     "metric": "strict_trans_distance_spearman",
                     "value": safe_spearman(distances[first][upper], distances[second][upper]), "n_sources": len(sources)})
    pd.DataFrame(rows).to_csv(RESULT_ROOT / "temporal_structure.csv", index=False)
    atomic_npz(CACHE_ROOT / "true_distance_matrices.npz", sources=sources, **distances)

    reliability_rows = []
    for repeat in range(config["pseudoreplicate_repeats"]):
        seed = config["pseudoreplicate_seed"] + repeat * 1009
        first, second = split_means(values, assignment, times, sources, seed)
        first_matrices = [first[:, index] for index in range(4)] + [np.diff(first, axis=1)[:, index] for index in range(3)]
        second_matrices = [second[:, index] for index in range(4)] + [np.diff(second, axis=1)[:, index] for index in range(3)]
        for name, left, right in zip(MATRICES, first_matrices, second_matrices):
            per_source = []
            for source_index in range(len(sources)):
                keep = np.ones(len(genes), bool)
                if direct[source_index] >= 0: keep[direct[source_index]] = False
                per_source.append(safe_pearson(left[source_index, keep], right[source_index, keep]))
            reliability_rows.append({"repeat": repeat, "seed": seed, "matrix": name,
                                     "geometry_reliability_rho": geometry_similarity(left, right, direct),
                                     "median_per_source_response_pearson": float(np.nanmedian(per_source)),
                                     "n_sources": len(sources)})
        if (repeat + 1) % 10 == 0:
            print(f"[传播复现] 伪重复可靠性 {repeat + 1}/{config['pseudoreplicate_repeats']}", flush=True)
    reliability = pd.DataFrame(reliability_rows)
    reliability.to_csv(RESULT_ROOT / "pseudoreplicate_reliability.csv", index=False)
    wave_medians = reliability[reliability.matrix.str.startswith("W")].groupby("matrix").geometry_reliability_rho.median()
    gate = config["early_gate"]
    checks = {"w23_reliable": float(wave_medians["W23"]) >= gate["minimum_w23_pseudoreplicate_median_geometry_rho"],
              "at_least_two_waves_reliable": int((wave_medians >= .1).sum()) >= gate["minimum_count_reliable_waves_above_rho_0_1"]}
    summary = {"created_at": now(), "wave_median_geometry_reliability": wave_medians.to_dict(),
               "checks": checks, "parts_cde_allowed": all(checks.values()),
               "cell_count_minimum": int(cell_count.min()), "cell_count_median": float(np.median(cell_count))}
    atomic_json(RESULT_ROOT / "parts_ab_gate.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
