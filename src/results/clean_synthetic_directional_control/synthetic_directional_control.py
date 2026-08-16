from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
from scipy import stats
from scipy.spatial.distance import pdist
import sklearn
import torch
from torch import nn


OUT = Path(__file__).resolve().parent
WORLDS = OUT / "worlds"
FIG = OUT / "figures"
GENERATOR_SEEDS = [1101, 2202, 3303, 4404, 5505]
CONDITIONS = [
    "NO_GRAPH",
    "CORRECT_DIRECTED_SIGNED",
    "CORRECT_DIRECTED_UNSIGNED",
    "REVERSED_DIRECTED_SIGNED",
    "DEGREE_PRESERVING_SHUFFLE",
    "SIGN_SHUFFLED",
]


@dataclass(frozen=True)
class GeneratorConfig:
    genes: int = 64
    train_sources: int = 44
    validation_sources: int = 20
    edge_probability: float = 0.08
    max_degree: int = 7
    min_magnitude: float = 0.3
    max_magnitude: float = 1.0
    target_spectral_radius: float = 0.75
    alpha: float = 0.7
    steps: int = 3
    noise_sd: float = 0.02
    train_observations: int = 12
    validation_observations: int = 16


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 64
    layers: int = 2
    heads: int = 4
    ff_hidden: int = 128
    graph_bias_lambda: float = 4.0
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    epochs: int = 250


GC = GeneratorConfig()
MC = ModelConfig()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def spectral_radius(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(a))))


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 4 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def top10_recall(pred: np.ndarray, truth: np.ndarray, source: int) -> float:
    mask = np.ones(len(pred), bool); mask[source] = False
    idx = np.flatnonzero(mask)
    k = min(10, len(idx))
    p = set(idx[np.argpartition(np.abs(pred[idx]), -k)[-k:]])
    t = set(idx[np.argpartition(np.abs(truth[idx]), -k)[-k:]])
    return len(p & t) / k


def effective_rank(matrix: np.ndarray) -> tuple[float, float, int, float]:
    x = matrix - matrix.mean(0, keepdims=True)
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    w = s ** 2 / max(float(np.sum(s ** 2)), 1e-12)
    nz = w[w > 0]
    erank = float(np.exp(-np.sum(nz * np.log(nz))))
    pcs80 = int(np.searchsorted(np.cumsum(w), .8) + 1)
    return erank, float(w[0]), pcs80, float(w[:5].sum())


def generate_graph(seed: int):
    rng = np.random.default_rng(seed)
    g = GC.genes
    edges: set[tuple[int, int]] = set()
    indeg = np.zeros(g, int); outdeg = np.zeros(g, int)

    def add(source: int, target: int) -> bool:
        if source == target or (source, target) in edges:
            return False
        if outdeg[source] >= GC.max_degree or indeg[target] >= GC.max_degree:
            return False
        edges.add((source, target)); outdeg[source] += 1; indeg[target] += 1
        return True

    # Response-blind asymmetric backbone guarantees bounded nonzero coverage.
    for source in range(g):
        add(source, (source + 1) % g)
        add(source, (source + 7) % g)
    target_edges = int(round(GC.edge_probability * g * (g - 1)))
    attempts = 0
    while len(edges) < target_edges and attempts < 200000:
        attempts += 1
        add(int(rng.integers(g)), int(rng.integers(g)))
    if len(edges) != target_edges:
        raise RuntimeError(f"Could not create target graph density: {len(edges)}/{target_edges}")
    ordered = sorted(edges)
    signs = np.ones(len(ordered), int)
    signs[: len(signs) // 2] = -1
    rng.shuffle(signs)
    magnitudes = rng.uniform(GC.min_magnitude, GC.max_magnitude, len(ordered))
    raw = np.zeros((g, g), float)  # [target, source]
    for (source, target), sign, magnitude in zip(ordered, signs, magnitudes):
        raw[target, source] = sign * magnitude
    before = spectral_radius(raw)
    scale = GC.target_spectral_radius / max(before, 1e-12)
    weighted = raw * scale
    after = spectral_radius(weighted)
    table = pd.DataFrame([
        {"source": source, "target": target, "sign": int(sign), "unscaled_magnitude": float(magnitude),
         "unscaled_weight": float(sign * magnitude), "scaled_weight": float(weighted[target, source])}
        for (source, target), sign, magnitude in zip(ordered, signs, magnitudes)
    ])
    return weighted, table, {"spectral_radius_before": before, "spectral_radius_after": after,
                             "scale_factor": scale, "edges": len(table), "density": len(table) / (g * (g - 1)),
                             "min_in_degree": int(indeg.min()), "max_in_degree": int(indeg.max()),
                             "min_out_degree": int(outdeg.min()), "max_out_degree": int(outdeg.max())}


def degree_preserving_shuffle(table: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    src = table.source.to_numpy(int).copy()
    tgt = table.target.to_numpy(int).copy()
    sign = table.sign.to_numpy(int).copy()
    edge_set = set(zip(src.tolist(), tgt.tolist()))
    successes = 0
    target_successes = 20 * len(src)
    for _ in range(300000):
        if successes >= target_successes:
            break
        i, j = rng.choice(len(src), 2, replace=False)
        a, b, c, d = src[i], tgt[i], src[j], tgt[j]
        if a == c or b == d or a == d or c == b:
            continue
        if (a, d) in edge_set or (c, b) in edge_set:
            continue
        edge_set.remove((a, b)); edge_set.remove((c, d))
        edge_set.add((a, d)); edge_set.add((c, b))
        tgt[i], tgt[j] = d, b
        successes += 1
    if successes < 5 * len(src):
        raise RuntimeError(f"Insufficient directed edge swaps: {successes}")
    result = pd.DataFrame({"source": src, "target": tgt, "sign": sign})
    result["shuffle_swaps"] = successes
    return result.sort_values(["source", "target"], kind="stable").reset_index(drop=True)


def sign_matrix(table: pd.DataFrame) -> np.ndarray:
    result = np.zeros((GC.genes, GC.genes), np.float32)
    for r in table.itertuples(index=False):
        result[int(r.target), int(r.source)] = int(r.sign)
    return result


def build_controls(graph_table: pd.DataFrame, seed: int):
    base = graph_table[["source", "target", "sign"]].copy()
    reverse = base.rename(columns={"source": "target", "target": "source"})[["source", "target", "sign"]]
    degree = degree_preserving_shuffle(base, seed + 71)
    rng = np.random.default_rng(seed + 91)
    sign_shuffle = base.copy(); sign_shuffle["sign"] = rng.permutation(sign_shuffle.sign.to_numpy())
    controls = {
        "NO_GRAPH": np.zeros((GC.genes, GC.genes), np.float32),
        "CORRECT_DIRECTED_SIGNED": sign_matrix(base),
        "CORRECT_DIRECTED_UNSIGNED": (sign_matrix(base) != 0).astype(np.float32),
        "REVERSED_DIRECTED_SIGNED": sign_matrix(reverse),
        "DEGREE_PRESERVING_SHUFFLE": sign_matrix(degree),
        "SIGN_SHUFFLED": sign_matrix(sign_shuffle),
    }
    return controls, reverse, degree, sign_shuffle


def generate_responses(weighted: np.ndarray, seed: int, train: np.ndarray, val: np.ndarray):
    identity = np.eye(GC.genes)
    operator = identity.copy()
    power = identity.copy()
    for step in range(1, GC.steps + 1):
        power = power @ weighted
        operator += (GC.alpha ** step) * power
    latent = -operator.T  # [source, target]
    rng = np.random.default_rng(seed + 313)
    train_obs = latent[train, None, :] + rng.normal(0, GC.noise_sd, (len(train), GC.train_observations, GC.genes))
    val_obs = latent[val, None, :] + rng.normal(0, GC.noise_sd, (len(val), GC.validation_observations, GC.genes))
    return latent.astype(np.float32), train_obs.astype(np.float32), val_obs.astype(np.float32)


class GraphAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        d, h = MC.d_model, MC.heads
        self.h = h; self.hd = d // h
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, MC.ff_hidden), nn.GELU(), nn.Linear(MC.ff_hidden, d))

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        z = self.norm1(x)
        qkv = self.qkv(z).reshape(b, n, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hd)
        attention = torch.softmax(logits + bias[None], dim=-1)
        message = torch.matmul(attention, v).transpose(1, 2).reshape(b, n, d)
        x = x + self.out(message)
        return x + self.ff(self.norm2(x))


class GraphBiasTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gene = nn.Parameter(torch.randn(GC.genes, MC.d_model) * 0.02)
        self.source_flag = nn.Embedding(2, MC.d_model)
        self.layers = nn.ModuleList([GraphAttentionLayer() for _ in range(MC.layers)])
        self.readout = nn.Sequential(nn.LayerNorm(MC.d_model), nn.Linear(MC.d_model, 1))

    def forward(self, sources: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        batch = len(sources)
        flags = torch.zeros((batch, GC.genes), dtype=torch.long, device=sources.device)
        flags[torch.arange(batch, device=sources.device), sources] = 1
        x = self.gene[None].expand(batch, -1, -1) + self.source_flag(flags)
        for layer in self.layers:
            x = layer(x, bias)
        return self.readout(x).squeeze(-1)


def graph_bias(condition: str, matrix: np.ndarray, device: torch.device) -> torch.Tensor:
    edge = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    if condition == "NO_GRAPH":
        return torch.zeros((MC.heads, GC.genes, GC.genes), device=device)
    if condition == "CORRECT_DIRECTED_UNSIGNED":
        return MC.graph_bias_lambda * edge.abs()[None].expand(MC.heads, -1, -1)
    polarity = torch.tensor([1, 1, -1, -1], dtype=torch.float32, device=device)[:, None, None]
    return MC.graph_bias_lambda * polarity * edge[None]


def train_condition(condition: str, matrix: np.ndarray, train_sources: np.ndarray, train_target: np.ndarray,
                    val_sources: np.ndarray, model_seed: int, world_dir: Path):
    seed_all(model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphBiasTransformer().to(device)
    bias = graph_bias(condition, matrix, device)
    src = torch.as_tensor(train_sources, dtype=torch.long, device=device)
    target = torch.as_tensor(train_target, dtype=torch.float32, device=device)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[torch.arange(len(src), device=device), src] = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=MC.learning_rate, weight_decay=MC.weight_decay)
    history = []
    model.train()
    for epoch in range(1, MC.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(src, bias)
        loss = torch.mean((prediction[mask] - target[mask]) ** 2)
        loss.backward(); optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == MC.epochs:
            history.append({"condition": condition, "epoch": epoch, "train_trans_mse": float(loss.detach().cpu())})
    model.eval()
    with torch.no_grad():
        val_src = torch.as_tensor(val_sources, dtype=torch.long, device=device)
        pred = model(val_src, bias).cpu().numpy()
    torch.save(model.state_dict(), world_dir / "models" / f"{condition}.pt")
    return pred, history, sum(p.numel() for p in model.parameters())


def metrics(condition: str, prediction: np.ndarray, truth: np.ndarray, val_sources: np.ndarray, world: int, seed: int):
    per = []
    flat_p, flat_t = [], []
    for i, source in enumerate(val_sources):
        mask = np.ones(GC.genes, bool); mask[source] = False
        flat_p.append(prediction[i, mask]); flat_t.append(truth[i, mask])
        per.append(safe_pearson(prediction[i, mask], truth[i, mask]))
    flat_p, flat_t = np.concatenate(flat_p), np.concatenate(flat_t)
    common = np.ones(GC.genes, bool); common[val_sources] = False
    pt, tt = prediction[:, common], truth[:, common]
    true_cos, pred_cos = pdist(normalize_rows(tt), "cosine"), pdist(normalize_rows(pt), "cosine")
    true_euc, pred_euc = pdist(tt, "euclidean"), pdist(pt, "euclidean")
    er, pc1, pcs80, top5 = effective_rank(pt)
    ter, tpc1, tpcs80, ttop5 = effective_rank(tt)
    truth_var = float(np.mean(np.var(tt, axis=0)))
    return {
        "world": world, "generator_seed": seed, "condition": condition, "n_train_sources": GC.train_sources,
        "n_validation_sources": GC.validation_sources, "common_geometry_genes": int(common.sum()),
        "trans_pearson": float(np.nanmean(per)), "pooled_trans_pearson": safe_pearson(flat_p, flat_t),
        "trans_mse": float(np.mean((flat_p - flat_t) ** 2)),
        "top10_recall": float(np.mean([top10_recall(prediction[i], truth[i], int(source)) for i, source in enumerate(val_sources)])),
        "response_distance_correlation": safe_spearman(pred_cos, true_cos),
        "swap_distance_correlation": safe_spearman(pred_euc, true_euc),
        "between_perturbation_variance_ratio": float(np.mean(np.var(pt, axis=0)) / max(truth_var, 1e-12)),
        "effective_rank": er, "pc1_variance_fraction": pc1, "pcs_80": pcs80, "top5_variance_fraction": top5,
        "truth_effective_rank": ter, "truth_pc1_variance_fraction": tpc1, "truth_pcs_80": tpcs80,
        "truth_top5_variance_fraction": ttop5,
    }


def graph_table_for_matrix(matrix: np.ndarray) -> pd.DataFrame:
    target, source = np.nonzero(matrix)
    return pd.DataFrame({"source": source, "target": target, "sign": matrix[target, source].astype(int)})


def write_readme(device: str):
    text = f"""# Clean synthetic directional control

This directory is self-contained. It does not use real K562 responses or the real sealed test.

## Reproduce

From this directory, run:

```powershell
python synthetic_directional_control.py
```

The script recreates all five worlds, graph controls, response arrays, checkpoints, predictions, metrics, figures, reports, and the SHA-256 artifact manifest from the frozen JSON/Markdown configuration files.

## Interpretation boundary

This is a matched-generator realizability test. The generator graph is intentionally the same structure supplied to the correct condition. The model receives topology/direction/sign only and never receives generator edge magnitudes or held-out response information.

## Runtime used

- Python: {platform.python_version()}
- NumPy: {np.__version__}
- SciPy: {scipy.__version__}
- pandas: {pd.__version__}
- scikit-learn: {sklearn.__version__}
- PyTorch: {torch.__version__}
- Device: {device}

See `SYNTHETIC_CONTROL_PLAN.md`, `INFORMATION_BOUNDARY.md`, and `oracle_information_boundary.json` before interpreting results.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def create_figures(metrics_frame: pd.DataFrame, world_one: dict):
    sns.set_theme(style="whitegrid", context="talk")
    order = CONDITIONS
    labels = ["No graph", "Correct signed", "Unsigned", "Reverse", "Degree shuffle", "Sign shuffle"]
    palette = {c: color for c, color in zip(order, ["#999999", "#D55E00", "#E69F00", "#0072B2", "#56B4E9", "#CC79A7"])}

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=metrics_frame, x="condition", y="response_distance_correlation", order=order, hue="condition",
                palette=palette, legend=False, showfliers=False)
    sns.stripplot(data=metrics_frame, x="condition", y="response_distance_correlation", order=order, color="black", size=6)
    plt.xticks(range(len(labels)), labels, rotation=18); plt.xlabel(""); plt.ylabel("Response-distance Spearman")
    plt.title("Geometry recovery across five synthetic worlds"); plt.tight_layout(); plt.savefig(FIG / "geometry_recovery_by_condition.png", dpi=180); plt.close()

    correct = metrics_frame[metrics_frame.condition == "CORRECT_DIRECTED_SIGNED"].set_index("world")
    comps = ["NO_GRAPH", "REVERSED_DIRECTED_SIGNED", "DEGREE_PRESERVING_SHUFFLE", "SIGN_SHUFFLED"]
    rows = []
    for comp in comps:
        other = metrics_frame[metrics_frame.condition == comp].set_index("world")
        for world in correct.index:
            rows.append({"world": world, "comparator": comp, "improvement": correct.loc[world, "response_distance_correlation"] - other.loc[world, "response_distance_correlation"]})
    imp = pd.DataFrame(rows)
    plt.figure(figsize=(11, 6)); sns.pointplot(data=imp, x="comparator", y="improvement", hue="world", markers="o", linestyles="-", errorbar=None)
    plt.axhline(0, color="black", lw=1); plt.xticks(range(4), ["No graph", "Reverse", "Degree shuffle", "Sign shuffle"], rotation=12)
    plt.xlabel(""); plt.ylabel("Correct-signed geometry improvement"); plt.title("Per-world structural advantage")
    plt.legend(title="World", ncol=5, fontsize=10); plt.tight_layout(); plt.savefig(FIG / "per_world_correct_advantage.png", dpi=180); plt.close()

    summary = metrics_frame.groupby("condition", as_index=False)[["trans_pearson", "response_distance_correlation", "swap_distance_correlation"]].mean()
    melt = summary.melt(id_vars="condition", var_name="metric", value_name="value")
    g = sns.catplot(data=melt, x="condition", y="value", col="metric", kind="bar", order=order, hue="condition",
                    palette=palette, legend=False, sharey=False, height=4.5, aspect=1.0)
    for ax in g.axes.flat:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=28, ha="right")
    g.set_axis_labels("", "Mean across worlds"); g.set_titles("{col_name}"); g.figure.suptitle("Prediction and geometry metrics", y=1.05)
    g.figure.savefig(FIG / "metric_summary.png", dpi=180, bbox_inches="tight"); plt.close(g.figure)

    rank = metrics_frame.groupby("condition", as_index=False)[["pc1_variance_fraction", "pcs_80"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=rank, x="condition", y="pc1_variance_fraction", order=order, hue="condition", palette=palette, legend=False, ax=axes[0])
    sns.barplot(data=rank, x="condition", y="pcs_80", order=order, hue="condition", palette=palette, legend=False, ax=axes[1])
    for ax in axes:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right"); ax.set_xlabel("")
    axes[0].set_ylabel("PC1 variance fraction"); axes[1].set_ylabel("PCs for 80% variance")
    fig.suptitle("Prediction rank structure"); fig.tight_layout(); fig.savefig(FIG / "rank_structure_by_condition.png", dpi=180); plt.close(fig)

    matrix = world_one["correct_matrix"]
    plt.figure(figsize=(8, 7)); sns.heatmap(matrix, cmap="coolwarm", center=0, square=True, cbar_kws={"label": "Edge sign"})
    plt.xlabel("Source gene"); plt.ylabel("Target gene"); plt.title("World 1 directed signed topology")
    plt.tight_layout(); plt.savefig(FIG / "world01_graph.png", dpi=180); plt.close()

    truth = world_one["truth"]
    common = world_one["common"]
    true_d = pdist(normalize_rows(truth[:, common]), "cosine")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, condition, color in zip(axes, ["NO_GRAPH", "CORRECT_DIRECTED_SIGNED"], ["#999999", "#D55E00"]):
        pred_d = pdist(normalize_rows(world_one["predictions"][condition][:, common]), "cosine")
        ax.scatter(true_d, pred_d, s=22, alpha=.65, color=color)
        ax.set_xlabel("True cosine distance"); ax.set_ylabel("Predicted cosine distance")
        ax.set_title(f"{condition}\nSpearman={safe_spearman(true_d, pred_d):.3f}")
    fig.suptitle("World 1 held-out response geometry"); fig.tight_layout(); fig.savefig(FIG / "world01_distance_recovery.png", dpi=180); plt.close(fig)


def artifact_manifest():
    rows = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.csv":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": digest})
    pd.DataFrame(rows).to_csv(OUT / "artifact_manifest.csv", index=False)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    WORLDS.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    write_readme(device)
    print(f"[1/8] 冻结配置已载入；设备={device}；不读取任何真实响应。", flush=True)
    all_metrics, all_history, split_rows, graph_audit = [], [], [], []
    world_one = None
    for world, generator_seed in enumerate(GENERATOR_SEEDS, 1):
        world_dir = WORLDS / f"world_{world:02d}"
        (world_dir / "models").mkdir(parents=True, exist_ok=True)
        (world_dir / "predictions").mkdir(parents=True, exist_ok=True)
        print(f"[world {world}/5] 先生成图与所有 controls，再生成响应。", flush=True)
        weighted, graph_table, audit = generate_graph(generator_seed)
        controls, reverse, degree, sign_shuffle = build_controls(graph_table, generator_seed)
        rng = np.random.default_rng(generator_seed + 17)
        perm = rng.permutation(GC.genes)
        train, val = np.sort(perm[:GC.train_sources]), np.sort(perm[GC.train_sources:])
        if np.intersect1d(train, val).size:
            raise RuntimeError("Source split overlap")
        latent, train_obs, val_obs = generate_responses(weighted, generator_seed, train, val)
        train_target, val_target = train_obs.mean(1), val_obs.mean(1)

        graph_table.to_csv(world_dir / "true_graph.csv", index=False)
        reverse.to_csv(world_dir / "reversed_graph.csv", index=False)
        degree.to_csv(world_dir / "shuffled_graph.csv", index=False)
        sign_shuffle.to_csv(world_dir / "sign_shuffled_graph.csv", index=False)
        graph_table_for_matrix(controls["CORRECT_DIRECTED_UNSIGNED"]).to_csv(world_dir / "unsigned_graph.csv", index=False)
        graph_table_for_matrix(controls["NO_GRAPH"]).to_csv(world_dir / "no_graph.csv", index=False)
        np.save(world_dir / "true_responses.npy", latent)
        np.save(world_dir / "observed_train_responses.npy", train_obs)
        np.save(world_dir / "observed_validation_responses.npy", val_obs)
        np.save(world_dir / "validation_target_mean.npy", val_target)
        np.save(world_dir / "generator_weight_matrix.npy", weighted)
        (world_dir / "split.json").write_text(json.dumps({"generator_seed": generator_seed,
                                                           "train_sources": train.tolist(), "validation_sources": val.tolist(),
                                                           "intersection": [], **audit}, indent=2), encoding="utf-8")
        for source in train:
            split_rows.append({"world": world, "generator_seed": generator_seed, "source": int(source), "split": "train"})
        for source in val:
            split_rows.append({"world": world, "generator_seed": generator_seed, "source": int(source), "split": "validation"})
        graph_audit.append({"world": world, "generator_seed": generator_seed, **audit,
                            "degree_shuffle_in_degree_exact": bool(np.array_equal((controls["CORRECT_DIRECTED_SIGNED"] != 0).sum(1), (controls["DEGREE_PRESERVING_SHUFFLE"] != 0).sum(1))),
                            "degree_shuffle_out_degree_exact": bool(np.array_equal((controls["CORRECT_DIRECTED_SIGNED"] != 0).sum(0), (controls["DEGREE_PRESERVING_SHUFFLE"] != 0).sum(0))),
                            "sign_counts_preserved": bool(np.array_equal(np.sort(controls["CORRECT_DIRECTED_SIGNED"][controls["CORRECT_DIRECTED_SIGNED"] != 0]), np.sort(controls["DEGREE_PRESERVING_SHUFFLE"][controls["DEGREE_PRESERVING_SHUFFLE"] != 0]))),
                            "source_split_disjoint": True})
        world_metrics, predictions = [], {}
        model_seed = 9000 + world
        for condition in CONDITIONS:
            pred, history, params = train_condition(condition, controls[condition], train, train_target, val, model_seed, world_dir)
            predictions[condition] = pred
            np.save(world_dir / "predictions" / f"{condition}.npy", pred)
            row = metrics(condition, pred, val_target, val, world, generator_seed)
            row.update({"model_seed": model_seed, "parameter_count": params,
                        "final_train_trans_mse": history[-1]["train_trans_mse"]})
            world_metrics.append(row); all_metrics.append(row)
            for h in history:
                all_history.append({"world": world, "generator_seed": generator_seed, **h})
            print(f"  {condition}: geometry={row['response_distance_correlation']:.3f}, trans-r={row['trans_pearson']:.3f}", flush=True)
        pd.DataFrame(world_metrics).to_csv(world_dir / "metrics.csv", index=False)
        if world == 1:
            common = np.ones(GC.genes, bool); common[val] = False
            world_one = {"correct_matrix": controls["CORRECT_DIRECTED_SIGNED"], "truth": val_target,
                         "predictions": predictions, "common": common}

    metrics_frame = pd.DataFrame(all_metrics)
    metrics_frame.to_csv(OUT / "per_world_metrics.csv", index=False)
    pd.DataFrame(all_history).to_csv(OUT / "training_history.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUT / "train_val_sources.csv", index=False)
    pd.DataFrame(graph_audit).to_csv(OUT / "graph_generation_audit.csv", index=False)
    print("[7/8] 汇总跨世界 geometry、swap、variance/rank 与胜出计数。", flush=True)
    correct = metrics_frame[metrics_frame.condition == "CORRECT_DIRECTED_SIGNED"].set_index("world")
    summary_rows = []
    numeric = ["trans_pearson", "pooled_trans_pearson", "trans_mse", "top10_recall", "response_distance_correlation",
               "swap_distance_correlation", "between_perturbation_variance_ratio", "effective_rank",
               "pc1_variance_fraction", "pcs_80", "top5_variance_fraction"]
    for condition in CONDITIONS:
        part = metrics_frame[metrics_frame.condition == condition].set_index("world")
        row = {"condition": condition, "worlds": len(part)}
        for metric in numeric:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_median"] = float(part[metric].median())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1))
        if condition != "CORRECT_DIRECTED_SIGNED":
            improvement = correct.response_distance_correlation - part.response_distance_correlation
            row["correct_signed_geometry_improvement_mean"] = float(improvement.mean())
            row["correct_signed_wins_worlds"] = int((improvement > 0).sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "overall_graph_condition_summary.csv", index=False)
    metrics_frame[["world", "generator_seed", "condition", "response_distance_correlation",
                   "truth_effective_rank", "truth_pc1_variance_fraction", "truth_pcs_80"]].to_csv(OUT / "geometry_recovery.csv", index=False)
    metrics_frame[["world", "generator_seed", "condition", "between_perturbation_variance_ratio", "effective_rank",
                   "pc1_variance_fraction", "pcs_80", "top5_variance_fraction", "truth_effective_rank",
                   "truth_pc1_variance_fraction", "truth_pcs_80", "truth_top5_variance_fraction"]].to_csv(OUT / "variance_rank_audit.csv", index=False)
    metrics_frame[["world", "generator_seed", "condition", "swap_distance_correlation"]].to_csv(OUT / "swap_audit.csv", index=False)
    pd.DataFrame([{"status": "NOT_RUN", "reason": "Optional mismatch curve omitted to keep the preregistered primary control lightweight and frozen."}]).to_csv(OUT / "optional_graph_mismatch.csv", index=False)

    forbidden = json.loads((OUT / "oracle_information_boundary.json").read_text(encoding="utf-8"))
    leakage = any(forbidden[k] for k in ["graph_uses_validation_response_values", "graph_uses_validation_response_ranking",
                                         "model_sees_validation_response_values", "model_sees_validation_program_coefficients",
                                         "graph_weights_fit_to_validation_responses", "heldout_source_rows_used_in_training"])
    comps = ["NO_GRAPH", "REVERSED_DIRECTED_SIGNED", "DEGREE_PRESERVING_SHUFFLE", "SIGN_SHUFFLED"]
    win_counts = {}
    mean_improvements = {}
    for comp in comps:
        other = metrics_frame[metrics_frame.condition == comp].set_index("world")
        diff = correct.response_distance_correlation - other.response_distance_correlation
        win_counts[comp] = int((diff > 0).sum()); mean_improvements[comp] = float(diff.mean())
    if leakage:
        verdict = "SYNTHETIC_CONTROL_LEAKAGE_CONCERN"
    elif all(win_counts[c] >= 4 and mean_improvements[c] > 0 for c in comps):
        verdict = "SYNTHETIC_DIRECTIONAL_STRUCTURE_CAPACITY_SUPPORTED"
    elif win_counts["NO_GRAPH"] >= 3 and sum(win_counts[c] >= 3 for c in comps[1:]) >= 2:
        verdict = "SYNTHETIC_STRUCTURE_PARTIALLY_SUPPORTED"
    else:
        verdict = "SYNTHETIC_DIRECTIONAL_STRUCTURE_NOT_SUPPORTED"
    create_figures(metrics_frame, world_one)
    s = summary.set_index("condition")
    short = {"NO_GRAPH": "No graph", "CORRECT_DIRECTED_SIGNED": "Correct signed", "CORRECT_DIRECTED_UNSIGNED": "Unsigned",
             "REVERSED_DIRECTED_SIGNED": "Reverse", "DEGREE_PRESERVING_SHUFFLE": "Degree shuffle", "SIGN_SHUFFLED": "Sign shuffle"}
    table_metrics = [("Trans Pearson", "trans_pearson_mean", ".3f"), ("MSE", "trans_mse_mean", ".5f"),
                     ("Response-distance corr", "response_distance_correlation_mean", ".3f"),
                     ("Swap-distance corr", "swap_distance_correlation_mean", ".3f"),
                     ("Variance ratio", "between_perturbation_variance_ratio_mean", ".3f"),
                     ("PC1 fraction", "pc1_variance_fraction_mean", ".3f")]
    header = "| Metric | " + " | ".join(short[c] for c in CONDITIONS) + " |\n|---|" + "---:|" * len(CONDITIONS)
    lines = [header]
    for label, col, fmt in table_metrics:
        vals = [format(float(s.loc[c, col]), fmt) for c in CONDITIONS]
        lines.append("| " + label + " | " + " | ".join(vals) + " |")
    unsigned = metrics_frame[metrics_frame.condition == "CORRECT_DIRECTED_UNSIGNED"].set_index("world")
    signsh = metrics_frame[metrics_frame.condition == "SIGN_SHUFFLED"].set_index("world")
    reverse = metrics_frame[metrics_frame.condition == "REVERSED_DIRECTED_SIGNED"].set_index("world")
    degree = metrics_frame[metrics_frame.condition == "DEGREE_PRESERVING_SHUFFLE"].set_index("world")
    no = metrics_frame[metrics_frame.condition == "NO_GRAPH"].set_index("world")
    signed_beats_unsigned = int((correct.response_distance_correlation > unsigned.response_distance_correlation).sum())
    recommendation = "MAIN_TEXT_SMALL_PANEL" if verdict == "SYNTHETIC_DIRECTIONAL_STRUCTURE_CAPACITY_SUPPORTED" else "SUPPLEMENT_ONLY" if verdict == "SYNTHETIC_STRUCTURE_PARTIALLY_SUPPORTED" else "REMOVE"
    report = f"""{verdict}

# Final clean synthetic directional capacity verdict

{chr(10).join(lines)}

## Direct answers

1. Does correct directed+signed structure improve intervention-response prediction? **{'YES' if s.loc['CORRECT_DIRECTED_SIGNED','trans_pearson_mean'] > s.loc['NO_GRAPH','trans_pearson_mean'] else 'NO'}**.
2. Does it improve response geometry, not only average error? **{'YES' if win_counts['NO_GRAPH'] >= 4 and mean_improvements['NO_GRAPH'] > 0 else 'PARTIAL' if mean_improvements['NO_GRAPH'] > 0 else 'NO'}**.
3. Does correct direction beat reversed direction? **{'YES' if win_counts['REVERSED_DIRECTED_SIGNED'] >= 4 else 'PARTIAL' if mean_improvements['REVERSED_DIRECTED_SIGNED'] > 0 else 'NO'}** ({win_counts['REVERSED_DIRECTED_SIGNED']}/5 worlds).
4. Does real sign information beat sign-shuffled/unsigned controls? **{'YES' if win_counts['SIGN_SHUFFLED'] >= 4 and signed_beats_unsigned >= 4 else 'PARTIAL' if mean_improvements['SIGN_SHUFFLED'] > 0 else 'NO'}** (sign-shuffled {win_counts['SIGN_SHUFFLED']}/5; unsigned {signed_beats_unsigned}/5).
5. Does correct structure beat degree-preserving shuffled topology? **{'YES' if win_counts['DEGREE_PRESERVING_SHUFFLE'] >= 4 else 'PARTIAL' if mean_improvements['DEGREE_PRESERVING_SHUFFLE'] > 0 else 'NO'}** ({win_counts['DEGREE_PRESERVING_SHUFFLE']}/5 worlds).
6. Is the result consistent across independent synthetic worlds? **{'YES' if verdict == 'SYNTHETIC_DIRECTIONAL_STRUCTURE_CAPACITY_SUPPORTED' else 'PARTIAL' if verdict == 'SYNTHETIC_STRUCTURE_PARTIALLY_SUPPORTED' else 'NO'}**.
7. Is there any held-out-response leakage? **{'YES' if leakage else 'NO'}**.
8. Is this properly interpreted as a matched-generator capacity control? **YES**.
9. Manuscript placement: **{recommendation}**.

## Matched-generator disclosure

The same underlying graph family generates the synthetic response operator and supplies the correct oracle topology/direction/sign. The model does not receive edge magnitudes or any response-derived graph information. This is intentionally realizable and supports only a capacity statement; it is not evidence that real biology follows the graph.

Across-world correct-signed geometry wins: NoGraph {win_counts['NO_GRAPH']}/5, Reverse {win_counts['REVERSED_DIRECTED_SIGNED']}/5, DegreeShuffle {win_counts['DEGREE_PRESERVING_SHUFFLE']}/5, SignShuffle {win_counts['SIGN_SHUFFLED']}/5.

No real K562 response, real-data replication, or sealed real test was opened. Sealed real-test open count: **0**.
"""
    (OUT / "FINAL_SYNTHETIC_CAPACITY_VERDICT.md").write_text(report, encoding="utf-8")
    log = f"""# Research log

- Frozen plan and JSON configs were written before generation/training.
- Five generator seeds: {GENERATOR_SEEDS}.
- Six graph conditions generated before training in each world.
- Model seeds matched within world: {[9000 + x for x in range(1, 6)]}.
- Fixed 250 epochs; no early stopping or condition-specific tuning.
- Graph passed exact degree/sign-count audits in every world: {bool(pd.DataFrame(graph_audit)[['degree_shuffle_in_degree_exact','degree_shuffle_out_degree_exact','sign_counts_preserved']].all().all())}.
- Held-out response leakage: {leakage}.
- Verdict: `{verdict}`.
- Sealed real-test open count: 0.
"""
    (OUT / "RESEARCH_LOG.md").write_text(log, encoding="utf-8")
    (OUT / "run_metadata.json").write_text(json.dumps({"verdict": verdict, "recommendation": recommendation,
                                                        "generator_seeds": GENERATOR_SEEDS, "conditions": CONDITIONS,
                                                        "win_counts": win_counts, "mean_geometry_improvements": mean_improvements,
                                                        "device": device, "leakage": leakage,
                                                        "sealed_real_test_open_count": 0}, indent=2), encoding="utf-8")
    artifact_manifest()
    print(f"[8/8] 完成：{verdict}；真实 sealed test 打开次数=0。", flush=True)


if __name__ == "__main__":
    main()
