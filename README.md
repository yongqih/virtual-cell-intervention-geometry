# Intervention geometry compression reveals a measurement strategy for virtual cells

This repository contains the analysis and figure code for the study by Yongqi Huang, Hanzhi Wang and Parker Wilson. Version 2.0.0 covers six main figures, twenty supplementary figures and four supplementary tables. The figure source tables are included in the repository. Individual predictions, measured responses and paired bootstrap draws are distributed in the companion data archive.

## Reproduce the figures

Use Python 3.12 and run these commands from the root of the repository or extracted code archive:

```sh
python -m venv .venv
# Activate the virtual environment in your shell.
python -m pip install -r nc_release/requirements.txt
python nc_release/render_figures.py F1 F3 F4 F6 --out rendered
```

Omit the figure names to draw all 26 figures. Supplementary figures are identified by S1 through S20. The command writes PNG, PDF and SVG files. Each figure runs in its own process. A GPU, network connection and model checkpoints are unnecessary for rendering. Arial was used for the reference figures. Matplotlib uses DejaVu Sans when Arial is unavailable.

The primary plotting environment used Python 3.12.14 and Matplotlib 3.11.1. Figures F3, F4, S6, S8, S9, S10, S13, S16 and S17 were drawn with Python 3.10.20 and Matplotlib 3.10.9. For exact raster comparisons, use the corresponding requirements file (`nc_release/requirements-py310.txt` for the second group). The numerical source tables are common to both environments.

## Recalculate the results

Extract the matching **v2.0.0 data archive** next to the repository. The published v2.0.0 archives are available on Zenodo for the [source data](https://doi.org/10.5281/zenodo.23990921) and [analysis code](https://doi.org/10.5281/zenodo.23990945). Source tables for plotting are also available directly in this tag.

```sh
python nc_release/replay.py ../data --table point_estimates.csv
python nc_release/uncertainty.py ../data
python nc_release/fit_transfer.py ../data --fold 0 --panel 0
```

The first command calculates 381 point estimates from saved arrays, including response correspondence, global ordering, local neighborhood overlap, effective rank, repeatable variation, subspace alignment, coordinate assignment, temporal prediction and transfer. The second reconstructs 319 simultaneous confidence intervals from paired bootstrap draws. The third reproduces the ordinary Ridge and linear residual predictions for one of the original 32-pair calibration sets. The inputs for fitting and the recipient responses for evaluation are identified separately. The replay starts from study predictions and measured pseudobulks. Model configurations and original fold assignments accompany the data.

| Figure | Question | Data folders in the companion archive |
| --- | --- | --- |
| 1, S1–S3 | Do accurate individual responses preserve intervention relationships? | benchmark, triplets, spectrum, plot_inputs/fig12 |
| 2, S4–S5, Tables 3–4 | How much reproducible response variation do predictions retain? | benchmark, spectrum, subspaces |
| 3, S6–S7 | Do training predictions span the same response directions across architectures? | subspaces, plot_inputs/fig34 |
| 4, S8–S10 | Which inputs place unseen interventions correctly within a fixed response space? | coordinate_inputs, subspaces, plot_inputs/fig34 |
| 5, S11–S13 | How does a measured early response affect later relationship prediction? | temporal, plot_inputs/protein, plot_inputs/renge |
| 6, S14–S20 | How can reference measurements support direct and indirect context transfer? | transfer, bridges, triplet_repair, multicontext, plot_inputs/fig6 |

Supplementary Tables 1 and 2 describe the cohorts, readouts, fold sizes and the information supplied to each predictor. All four tables are available in `nc_release/supplementary_tables/`.

## Upstream data

- Replogle Perturb-seq in RPE1, K562 and Jurkat: [Replogle et al.](https://doi.org/10.1016/j.cell.2022.05.013).
- Tahoe drug-response predictions: [State study and released resources](https://doi.org/10.1016/j.cell.2026.07.052).
- Time-resolved drug perturbation proteomics: [ProteinTalks](https://doi.org/10.1038/s41586-026-11001-9).
- Genetic time course: [RENGE](https://doi.org/10.1038/s42003-023-05594-4), [GSE213069](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213069).
- Melanoma context-transfer analysis: [Frangieh et al.](https://doi.org/10.1038/s41588-021-00779-1).

Project code is distributed under the MIT License. Upstream software and source datasets retain their own terms. The [earlier analysis release](https://github.com/yongqih/virtual-cell-intervention-geometry/tree/1b43e24ae09048af03630c4ef99e216fbe887742) remains accessible by its commit.
