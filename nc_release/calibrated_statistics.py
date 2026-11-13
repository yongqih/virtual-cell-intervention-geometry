"""Reproduce the Fig. 6 learning-curve intervals, sparse-bridge intervals and S17 triplet rates."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calibrated_transfer import HERE, INPUT, NAMES, direct_replay
from analysis_metrics import cosine_similarity
from replay import row_indices

PLOTS = HERE / "plot_inputs/fig6/calibrated_transfer"


def interval_family(rows, draws):
    point = rows.delta.to_numpy()
    se = draws.std(axis=1, ddof=1)
    varying = se > 1e-14
    centered = draws - point[:, None]
    np.testing.assert_allclose(centered[~varying], 0, atol=1e-12)
    cutoff = np.quantile(np.max(abs(centered[varying] / se[varying, None]), axis=0), .95)
    for sign, col in [(-1,"simultaneous_low"), (1,"simultaneous_high")]:
        np.testing.assert_allclose(point + sign * cutoff * se, rows[col], atol=1e-9, rtol=0)


def intervals():
    table = pd.read_csv(INPUT / "direct_intervals.csv")
    with np.load(INPUT / "direct_bootstrap.npz") as data:
        for (method, comparator, n), rows in table.groupby(["method","comparison","family_endpoints"]):
            if n != 90:
                continue
            draws = [data[f"{r.k}/{method}"][NAMES.index(r.metric)] -
                     data[f"{r.k}/{comparator}"][NAMES.index(r.metric)] for r in rows.itertuples()]
            interval_family(rows, np.stack(draws))
    main = table.query("method=='JointPolicy_scaled' and comparison=='TxPert' and family_endpoints==90")
    passes = main.groupby("k").simultaneous_low.min().gt(0)
    print("First size with positive simultaneous lower bounds for all three scores:", passes[passes].index.min())
    table = pd.read_csv(INPUT / "sparse_bridge_intervals.csv")
    with np.load(INPUT / "sparse_bridge_bootstrap.npz") as data:
        for label, rows in table.groupby("family"):
            draws = [data[f"{r.route}/{r.method}/{r.metric}"] - data[f"{r.route}/{r.comparator}/{r.metric}"]
                     for r in rows.itertuples()]
            interval_family(rows, np.stack(draws))
            print(label, len(rows), "paired intervals")


def triplet_rates(data):
    pools = pd.read_csv(data / "triplet_repair/ELIGIBLE_COHORTS.csv.gz")
    pools = pools.loc[pools.quality_mode.eq("all_three_four_criteria") & pools.eligibility.eq("fully_rematched_rule")]
    fixed = pd.read_csv(HERE / "plot_inputs/fig6/fixed_26_original.csv")
    expected = pd.read_csv(PLOTS / "triplet_repair_by_case.csv")
    rates = pd.read_csv(PLOTS / "triplet_population_rates.csv").query(
        "quality_mode=='all_three_four_criteria' and eligibility=='fully_rematched_rule'").set_index("reference")
    total = {ref:np.zeros((3,1001)) for ref in rates.index}
    counts = np.zeros(5, int)
    for f in range(5):
        with np.load(data / "transfer/BOOTSTRAP_WEIGHTS.npz") as z:
            w, order = z[f"f{f}_weights"], z[f"f{f}_names"]
        with np.load(data / f"transfer/TxPert_fold{f}_response.npz") as z:
            tx, targets = z["response"], z["names"]
        assert np.array_equal(order,targets)
        for panel in range(5):
            names, genes, prediction = direct_replay(data,f,panel,32,False,"joint")
            assert np.array_equal(names,order)
            with np.load(data / f"transfer/direct/fold{f}_panel{panel}.npz") as z:
                centers = {"TxPert_train_secondary":z["reference_full"], "paired11_primary":z["reference11"]}
                assert np.array_equal(genes,z["genes"])
            def indices(frame):
                return [row_indices(frame[k],names) for k in ("A","B","C")]
            for ref, center in centers.items():
                sim, baseline = cosine_similarity(prediction-center), cosine_similarity(tx-center)
                local = fixed.loc[fixed.fold.eq(f)]
                a,b,c = indices(local)
                margin = sim[a,c] - sim[a,b]
                want = expected.query("fold==@f and panel==@panel and reference==@ref").set_index(["A","B","C"])
                keys = pd.MultiIndex.from_frame(local[["A","B","C"]])
                np.testing.assert_allclose(margin,want.loc[keys,"joint_margin"],atol=1e-7,rtol=0)
                if ref == "TxPert_train_secondary":
                    counts[panel] += (margin > 1e-10).sum()
                part = pools.query("fold==@f and panel==@panel and reference==@ref")
                a,b,c = indices(part)
                wrong = baseline[a,c]-baseline[a,b] < -1e-10
                np.testing.assert_array_equal(wrong,part.baseline_wrong)
                after = sim[a,c]-sim[a,b] < -1e-10
                weights = w[:,a].astype(float)*w[:,b]*w[:,c]
                events = np.stack([np.ones(len(a)),wrong,after])
                total[ref] += np.column_stack([events.sum(1), events @ weights.T])
    print("Fixed errors corrected by panel:", counts, "out of",len(fixed))
    for ref,(n,initial,final) in total.items():
        for key,num in [("baseline_inversion_rate",initial),("inversion_rate",final),("net_error_reduction_rate",initial-final)]:
            v = np.divide(num,n,out=np.full_like(n,np.nan),where=n>0)
            reported = rates.loc[ref,[key,key+"_low",key+"_high"]].to_numpy(float)
            np.testing.assert_allclose(np.r_[v[0],np.nanquantile(v[1:],[.025,.975])],reported,atol=1e-10)
        print(ref,"net reduction (percentage points)",100*rates.loc[ref,"net_error_reduction_rate"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets", type=Path, metavar="DATA", help="Refit the 25 maps and resample the eligible triplets")
    args = parser.parse_args()
    with threadpool_limits(1):
        intervals()
        if args.triplets:
            triplet_rates(args.triplets)
