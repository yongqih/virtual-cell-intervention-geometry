"""Build Supplementary Table 5 from fold estimates and paired bootstrap draws."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIMARY = "paired11_primary"
SECONDARY = "TxPert_train_secondary"
AE = "Ahlmann_Eltze_no_anchor"
TX = "TxPert_frozen"
TRANSFER = "ResidualLinearCV"
LABELS = {AE: "Ahlmann-Eltze linear model", TX: "TxPert", TRANSFER: "Residual transfer"}
METRICS = ("rho", "top10", "CA")
FAMILY = [(PRIMARY, "rho"), (PRIMARY, "top10"), (SECONDARY, "rho"),
          (SECONDARY, "top10"), (PRIMARY, "CA")]


def build(folder):
    fold = pd.read_csv(folder / "fold_scores.csv")
    index = ["fold", "reference", "method"]
    if len(fold) != 30 or fold.duplicated(index).any():
        raise ValueError("Expected five fold estimates for each method and reference")
    means = fold.groupby(["reference", "method"])[list(METRICS)].mean()
    result = []

    def add(section, method, metric, value, low, high, interval):
        result.append(dict(section=section, method=LABELS[method], metric=metric,
                           estimate=value, ci_low=low, ci_high=high, interval=interval))

    with np.load(folder / "paired_draws.npz") as draws:
        for method in LABELS:
            for metric in METRICS:
                point = means.loc[(PRIMARY, method), metric]
                bounds = np.quantile(draws[f"{PRIMARY}|{method}|{metric}"], [.025, .975])
                add("Performance", method, metric, point, *bounds, "Pointwise 95%")

        for method in [TX, TRANSFER]:
            points, paired = [], []
            for ref, metric in FAMILY:
                points.append(means.loc[(ref, method), metric] - means.loc[(ref, AE), metric])
                paired.append(draws[f"{ref}|{method}|{metric}"] - draws[f"{ref}|{AE}|{metric}"])
            points = np.array(points)
            paired = np.stack(paired)
            se = paired.std(axis=1, ddof=1)
            critical = np.quantile(np.max(np.abs((paired - points[:, None]) / se[:, None]), axis=0), .95)
            for i, (ref, metric) in enumerate(FAMILY):
                if ref == PRIMARY:
                    width = critical * se[i]
                    add("Gain over Ahlmann-Eltze", method, metric, points[i],
                        points[i] - width, points[i] + width, "Simultaneous 95%")
    return pd.DataFrame(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=HERE, help="Directory with fold_scores.csv and paired_draws.npz")
    parser.add_argument("--out", type=Path, required=True, help="Destination CSV")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build(args.scores).to_csv(args.out, index=False, float_format="%.12g")
