"""Refit the calibrated Fig. 6 maps from paired training responses; score held queries separately."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from analysis_metrics import cosine_similarity, discrimination, spearman
from fit_transfer import predict
from replay import row_indices

HERE = Path(__file__).resolve().parent
INPUT = HERE / "calibrated_inputs"
ROUTES = ("K562_RPE1_Jurkat", "Jurkat_RPE1_K562")
NAMES = ("rho", "top10", "CA")
DIRECT = [(beta, alpha) for alpha in (.0003, .003, .03, .3, 3.) for beta in (0., .5, 1.)]
BRIDGE = [(beta, alpha) for beta in (0., .5, 1.) for alpha in (.001, .01, .1, 1., 10.)]


def metrics(prediction, observed, origin):
    n = len(observed)
    def relationships(x):
        sim = cosine_similarity(x - origin)
        pairs = sim[np.triu_indices(n, 1)]
        np.fill_diagonal(sim, -np.inf)
        return pairs, np.argsort(-sim, axis=1, kind="stable")[:, :10]
    a, neighbors_a = relationships(prediction)
    b, neighbors_b = relationships(observed)
    top = (neighbors_a[:, :, None] == neighbors_b[:, None, :]).any(2).mean()
    return np.array([spearman(a, b), top, discrimination(prediction, observed)[0].mean()])


def hashed_groups(names, prefix, balanced):
    def key(name):
        digest = hashlib.sha256((prefix + name).encode()).hexdigest()
        return int(digest if prefix.startswith("onehop") else digest[:12], 16)
    keys = [key(n) for n in names]
    if not balanced:
        return np.array([k % 3 for k in keys])
    groups = np.empty(len(names), int)
    groups[sorted(range(len(names)), key=keys.__getitem__)] = np.arange(len(names)) % 3
    return groups


def edge_predict(x, y, query, cx, cq, beta, alpha):
    mean, sd = x.mean(0), x.std(0)
    sd[sd < 1e-6] = 1.
    z = (x - mean) / sd
    residual = y - beta * cx
    mu = residual.mean(0)
    yc = residual - mu
    penalty = alpha * max(np.sum(z * z) / len(z), 1e-12)
    dual = len(x) < x.shape[1]
    vals, vec = eigh(z @ z.T if dual else z.T @ z, check_finite=False)
    denominator = (np.maximum(vals, 0) + penalty)[:, None]
    if dual:
        coef = z.T @ vec @ ((vec.T @ yc) / denominator)
    else:
        coef = vec @ ((vec.T @ z.T @ yc) / denominator)
    coef /= sd[:, None]
    return beta * cq + query @ coef + (mu - mean @ coef)


def direct_predict(x, y, query, cx, cq, beta, alpha):
    return predict(x, y, query, cx, cq, alpha, beta, True)


def select(x, y, cx, names, design, seed=0, rule="joint"):
    direct = design == "direct"
    grid = DIRECT if direct else BRIDGE
    fit = direct_predict if direct else edge_predict
    if direct:
        groups = np.arange(len(y)) if len(y) < 6 else np.random.default_rng(seed).permutation(len(y)) % 3
    else:
        groups = hashed_groups(names, "source-inner::", False)
    profiles = np.empty((len(grid), *y.shape))
    loss = np.zeros(len(grid))
    scores = np.zeros((len(grid), 3))
    for g in np.unique(groups):
        test = groups == g
        train = ~test
        for j, params in enumerate(grid):
            pv = fit(x[train], y[train], x[test], cx[train], cx[test], *params)
            profiles[j, test] = pv
            error = np.square(pv - y[test])
            loss[j] += error.sum() / y.size if direct else error.mean() / 3
            if design == "full":
                scores[j] += metrics(pv, y[test], y[train].mean(0)) / 3
    ranks = np.zeros(len(grid))
    if rule == "joint" and len(y) >= 12:
        if design != "full":
            origin = y[:11].mean(0) if direct else y.mean(0)
            try:
                scores = np.array([metrics(pv, y, origin) for pv in profiles]).round(12)
            except ValueError:
                if not direct:
                    raise
                scores.fill(np.nan)
        if np.isfinite(scores).all():
            ranks = rankdata(-scores, axis=0, method="average").mean(1)
    idx = min(range(len(grid)), key=lambda j: (ranks[j], loss[j], j))
    return grid[idx]


def cross_fitted(x, y, cx, names, design, seed, rule):
    prefix = "onehop-rms-v1::" if design == "direct" else "radius-outer-v1::"
    outer = hashed_groups(names, prefix, True)
    fit = direct_predict if design == "direct" else edge_predict
    pred = np.empty_like(y)
    for g in range(3):
        mask = outer == g
        tr = ~mask
        params = select(x[tr], y[tr], cx[tr], names[tr], design, seed, rule)
        pred[mask] = fit(x[tr], y[tr], x[mask], cx[tr], cx[mask], *params)
    return pred


def radius(y, oof, center):
    s = np.sqrt(np.square(y - center).sum(1).mean() / np.square(oof - center).sum(1).mean())
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Cannot calibrate the response radius")
    return s


def verify(scores, expected, tag):
    np.testing.assert_allclose(scores, expected, atol=2e-6, rtol=0, err_msg=tag)
    print(tag + "  " + "  ".join(f"{m}={v:.6f}" for m, v in zip(NAMES, scores)))


def direct_replay(data, f, panel, size, matched, rule):
    if matched and (size != 32 or rule != "joint"):
        raise ValueError("Matched-input scores are supplied for the jointly selected 32-pair map")
    archive = data / "transfer"
    with np.load(archive / f"direct/fold{f}_panel{panel}.npz") as d, np.load(archive / "K562_3536_ALL_TARGETS.npz") as z, np.load(data / f"subspaces/biology_fold{f}.npz") as b:
        ids, queries, genes, ref = d["training_names"][:size], d["names"], d["genes"], d["reference11"]
        assert len(set(ids)) == size and not set(ids) & set(queries)
        cols = row_indices(genes, b["genes"])
        y = b["train_truth"][row_indices(ids, b["train_names"])][:, cols] + b["reference"][cols]
        measured = b["held_truth"][row_indices(queries, b["held_names"])][:, cols] + b["reference"][cols]
        srcgenes, srcids, matrix = z["genes"], z["names"], z["delta"]
    all_ids = np.concatenate([ids, queries])
    xall = matrix[row_indices(all_ids, srcids)].astype(float)
    if matched:
        with np.load(INPUT / "source_readouts.npz") as z:
            assert z["names"][0] == "control"
            vals = z["means"][row_indices(all_ids, z["names"])].astype(float) - z["means"][0]
            c = vals[:, row_indices(genes, z["genes"])]
            extra = [g for g in genes if g not in set(srcgenes)]
            xall = np.column_stack([xall, vals[:, row_indices(extra, z["genes"])]])
    else:
        c = np.zeros((len(all_ids), len(genes)))
        for j, gene in enumerate(genes):
            if gene in srcgenes:
                c[:, j] = xall[:, np.flatnonzero(srcgenes == gene)[0]]
    x, q = xall[:size], xall[size:]
    cx, cq = c[:size], c[size:]
    seed = 261102 + 100*f + 10*panel + size
    params = select(x, y, cx, ids, "direct", seed, rule)
    raw = direct_predict(x, y, q, cx, cq, *params).astype(np.float32).astype(float)
    center = y[:min(size,11)].mean(0)
    if size >= 3:
        s = radius(y, cross_fitted(x, y, cx, ids, "direct", seed, rule), center)
        prediction = center + s * (raw - center)
    else:
        prediction = raw
    if matched:
        table = pd.read_csv(INPUT / "matched_scores.csv").query("protocol=='full32' and method=='Joint'")
    else:
        method = "UnscaledMSE_k2" if size == 2 else ("JointPolicy_scaled" if rule == "joint" else "MSE_scaled")
        table = pd.read_csv(INPUT / "direct_scores.csv").query("method==@method")
    expected = table.query("fold==@f and panel==@panel and k==@size").iloc[0]
    verify(metrics(prediction, measured, ref), expected[list(NAMES)].to_numpy(float), f"direct fold={f} panel={panel} pairs={size}")
    return queries, genes, prediction


def bridge_replay(data, f, route, full):
    path = data / "bridges" / route
    with np.load(path / f"training_inputs_fold{f}.npz") as a:
        ids1, ids2 = a["first_names"], a["second_names"]
        x1, y1, x2, y2 = a["first_source"], a["first_A"], a["second_A"], a["second_target"]
        names, source, tx, genes, origin = a["query_names"], a["source_query"], a["TxPert_A_query"], a["genes"], a["target_reference"]
    design = "full" if full else "sparse"
    if full:
        forward = route == ROUTES[0]
        edge1, edge2 = ("AB","CA") if forward else ("CA","AB")
        c1, c2 = ("B","C") if forward else ("C","B")
        with np.load(INPUT / f"full_edges_fold{f}.npz") as a:
            assert np.array_equal(genes, a["genes"])
            def ordered(key):
                ids = a[key+"_names"]
                ix = np.argsort(ids, kind="stable")
                return ids[ix], ix
            ids1, ix1 = ordered(edge1)
            ids2, ix2 = ordered(edge2)
            x1, y1 = a[c1+"_"+edge1][ix1], a["A_"+edge1][ix1]
            x2, y2 = a["A_"+edge2][ix2], a[c2+"_"+edge2][ix2]
            origin = y2[row_indices(sorted(a[edge2+"_names"][:50]), ids2)].mean(0)
    assert not set(ids1)&set(ids2) and not set(names)&(set(ids1)|set(ids2))
    a1 = select(x1,y1,x1,ids1,design)
    a2 = select(x2,y2,x2,ids2,design)
    def first(q): return edge_predict(x1,y1,q,x1,q,*a1)
    def last(q): return edge_predict(x2,y2,q,x2,q,*a2)
    if full:
        oof = last(cross_fitted(x1,y1,x1,ids1,design,0,"joint"))
    else:
        oof = cross_fitted(x2,y2,x2,ids2,design,0,"joint")
    s = radius(y2,oof,origin)
    learned = origin + s * (last(first(source)) - origin)
    with np.load(path / f"fold{f}.npz") as z:
        assert np.array_equal(names,z["query_names"])
        truth = z["target_truth"]
    if full:
        table = pd.read_csv(INPUT / "full_bridge_scores.csv").query("route==@route and fold==@f and method=='joint'").set_index("metric")
        verify(metrics(learned,truth,origin),table.loc[list(NAMES),"value"],f"full {route} fold={f}")
    else:
        table = pd.read_csv(HERE / "plot_inputs/fig6/calibrated_transfer/bridge_before_after_scaling.csv").query("route==@route and fold==@f").set_index("method")
        for key,pred in [("learned",learned),("txroute",origin+s*(last(tx)-origin))]:
            values = metrics(pred,truth,origin)
            np.testing.assert_allclose(values[[0,2]],table.loc[key,["rho","CA"]].to_numpy(float),atol=2e-6,rtol=0)
            print(f"{route} {key} fold={f} rho={values[0]:.6f} CA={values[2]:.6f}")
    return names, genes, learned


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("design", choices=["direct","matched","bridge","full-bridge"])
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--panel", type=int, choices=range(5), default=0)
    p.add_argument("--pairs", type=int, choices=range(2,33), default=32)
    p.add_argument("--rule", choices=["joint","mse"], default="joint")
    p.add_argument("--route", choices=ROUTES, default=ROUTES[0])
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    with threadpool_limits(1):
        if args.design in ("direct","matched"):
            result = direct_replay(args.data,args.fold,args.panel,args.pairs,args.design=="matched",args.rule)
        else:
            result = bridge_replay(args.data,args.fold,args.route,args.design=="full-bridge")
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.out,names=result[0],genes=result[1],prediction=result[2])


if __name__ == "__main__":
    main()
