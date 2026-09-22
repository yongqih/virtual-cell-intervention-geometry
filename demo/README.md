# Grouped-geometry demonstration

`example_responses.npz` is a simulated, fixed-seed example containing 12 interventions, 8 readout genes and 2 independent groups. It demonstrates the evaluator and is not a biological result or an additional experiment.

From the directory containing `code/` and `data/`, run:

```text
python code/reproduction/evaluate_responses.py --input code/demo/example_responses.npz --output output/geometry_demo
```

Expected results:

| Group | Interventions | Within-group pairs | Geometry Spearman correlation |
| --- | ---: | ---: | ---: |
| fold_0 | 6 | 15 | 0.11071428571428568 |
| fold_1 | 6 | 15 | 0.682142857142857 |

`summary.json` reports 2 groups, 12 interventions, 8 genes, `cross_group_pairs_used: false`, and `mean_group_geometry: 0.39642857142857135`. Numerical comparisons can use an absolute tolerance of 1e-10. Reference output files are provided in `expected/`.

The command uses the unchanged `geometry` function in the released analysis source. It normalizes response rows, calculates cosine distances within each group, and computes the Spearman correlation between predicted and observed distance vectors. The summary is the unweighted mean of group correlations. It does not pool pairs across independently trained models or groups. This demo reports point estimates; it does not compute confidence intervals or change the manuscript bootstrap procedure.

## Your own data

Prepare an NPZ file with these arrays. All arrays must be loadable with `allow_pickle=False`; use ordinary Unicode arrays for names.

| Array | Shape | Meaning |
| --- | --- | --- |
| `prediction` | N x G | Predicted, control-relative responses |
| `truth` | N x G | Observed, control-relative responses in exactly the same row and gene order |
| `group` | N | Evaluation-group labels: each group uses one model fit and one biological context |
| `source_id` | N | Intervention target symbols, unique within a group |
| `gene_names` | G | Unique readout gene symbols, shared by prediction and truth |

Use matched controls and a common, predefined normalization and gene panel. For residual geometry, supply the residual responses after applying the study's shared-response subtraction convention to both matrices. Exclude all perturbed target genes from the readout panel before saving; the wrapper additionally checks for overlap with the evaluated `source_id` values. At least four interventions are required per group. Non-finite and zero response vectors are rejected because their cosine geometry is undefined. Identical distance vectors have correlation 1; constant distance vectors follow the released evaluator's convention of returning 0.

For example, if the two response CSV files have the same intervention symbols as their row index and the same readout genes as their columns, and `groups.csv` contains `source_id,group`:

```python
import numpy as np
import pandas as pd

truth = pd.read_csv("truth_responses.csv", index_col=0)
prediction = pd.read_csv("predicted_responses.csv", index_col=0)
metadata = pd.read_csv("groups.csv").set_index("source_id")
assert truth.index.is_unique and metadata.index.is_unique
assert prediction.index.equals(truth.index)
assert prediction.columns.equals(truth.columns)
np.savez_compressed(
    "my_responses.npz",
    prediction=prediction.to_numpy(dtype=float),
    truth=truth.to_numpy(dtype=float),
    source_id=truth.index.to_numpy(dtype=str),
    group=metadata.loc[truth.index, "group"].to_numpy(dtype=str),
    gene_names=truth.columns.to_numpy(dtype=str),
)
```

Run:

```text
python code/reproduction/evaluate_responses.py --input my_responses.npz --output output/my_evaluation
```

Observed responses enter this command only for evaluation. Predictions must be generated independently, using the intended intervention-disjoint training and validation design. The wrapper is an evaluator, not an unseen-response predictor, and does not establish whether an externally supplied prediction was obtained without leakage.
