# Intervention geometry compression reveals a measurement strategy for virtual cells

This repository contains the analysis and figure code for the study by Yongqi Huang, Hanzhi Wang and Parker Wilson. Version 2.0.0 covers six main figures, twenty supplementary figures and Supplementary Tables 1–4. The current repository also includes the rank-eight projection comparisons in Supplementary Fig. 7 and the Ahlmann-Eltze baseline in Supplementary Table 5. The figure source tables are included in the repository. Individual predictions, measured responses and paired bootstrap draws are distributed in the companion data archive.

## Reproduce the figures

Use Python 3.12 and run these commands from the root of the repository or extracted code archive:

```sh
python -m venv .venv
# Activate the virtual environment in your shell.
python -m pip install -r nc_release/requirements.txt
python nc_release/render_figures.py F1 F3 F4 F6 --out rendered
```

Omit the figure names to draw all 26 figures. Supplementary figures are identified by S1 through S20. The command writes PNG, PDF and SVG files. Each figure runs in its own process. A GPU, network connection and model checkpoints are unnecessary for rendering. Arial was used for the reference figures. Matplotlib uses DejaVu Sans when Arial is unavailable.

The primary plotting environment used Python 3.12.14 and Matplotlib 3.11.1. Figures F3, F4, S6, S7, S8, S9, S10, S13, S16 and S17 use Matplotlib 3.10.9. For exact raster comparisons, use the corresponding Python 3.10 requirements file (`nc_release/requirements-py310.txt` for this group). The numerical source tables are common to both environments.

## Recalculate the results

Extract the **v2.0.0 data archive** next to the repository. The published v2.0.0 archives are available on Zenodo for the [source data](https://doi.org/10.5281/zenodo.23990921) and [analysis code](https://doi.org/10.5281/zenodo.23990945). Use the plotting tables in the current checkout to reproduce the revised figures.

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
| 3, S6–S7 | How much response structure do model subspaces retain, and how accurately do models assign coordinates? | benchmark, subspaces, plot_inputs/fig34 |
| 4, S8–S10 | Which inputs place unseen interventions correctly within a fixed response space? | coordinate_inputs, subspaces, plot_inputs/fig34 |
| 5, S11–S13 | How does a measured early response affect later relationship prediction? | temporal, plot_inputs/protein, plot_inputs/renge |
| 6, S14–S20 | How can reference measurements support direct and indirect context transfer? | transfer, bridges, triplet_repair, multicontext, plot_inputs/fig6 |

Supplementary Tables 1 and 2 describe the cohorts, readouts, fold sizes and the information supplied to each predictor. All five tables are available in [nc_release/supplementary_tables/](nc_release/supplementary_tables/).

## Reproduce Supplementary Figure 7

Panels a–f use eight axes fitted separately to each model’s predictions for unseen interventions in each fold. Each measured response is projected onto those axes. Global ρ and Top-10 measure recovery of the observed ordering and neighborhoods. CA compares reconstructed expression with the original measured profiles. The model-level analysis relates subspace overlap to each projected score using seven model means, each weighted equally across five folds.

Recalculate the three correlations and their two-sided exact *P* values from the [seven-model table](nc_release/plot_inputs/fig34/s7_seven_model_projections.csv). The test enumerates all 5,040 permutations of the model labels and applies Holm correction across the three endpoints. The [fold scores](nc_release/plot_inputs/fig34/s7_projection_by_fold.csv) and [exact test results](nc_release/plot_inputs/fig34/s7_axis_alignment_tests.csv) are provided alongside the figure inputs.

```sh
python nc_release/subspace_projections.py --model-scores nc_release/plot_inputs/fig34/s7_seven_model_projections.csv --out results/s7
```

To recompute the projection scores from the v2.0.0 data archive, use the command below. It also reproduces panels g–j, which compare fixed training axes from GEARS, Scouter, TxPert and measured responses with 1,000 random eight-dimensional subspaces per fold. This comparison uses the two shared folds (339 and 355 held-out interventions) and all 757 scoring genes. Each random draw uses the same held-out responses as the training-axis comparison. The plotted range spans the 2.5th and 97.5th percentiles of the two-fold mean across draws.

```sh
python nc_release/subspace_projections.py --data ../data --out results/s7
python nc_release/render_figures.py S7 --out rendered
```

## Reproduce Supplementary Table 5

[Supplementary Table 5](nc_release/supplementary_tables/Table_5.csv) compares the no-anchor Ahlmann-Eltze linear model, frozen TxPert and 32-pair residual transfer. The two zero-shot predictors used all 1,225–1,254 RPE1 training interventions in each fold. The transfer map used 32 interventions measured in both cell lines for calibration and one measured K562 response per new target. All three methods were evaluated on the same 1,432 held-out RPE1 interventions.

| Method | Global ρ | Top-10 | CA |
| --- | ---: | ---: | ---: |
| Ahlmann-Eltze linear model | 0.050 | 0.069 | 0.535 |
| TxPert | 0.255 | 0.143 | 0.650 |
| Residual transfer | 0.389 | 0.202 | 0.651 |

**Rebuild the table.** From the repository root, the following command reconstructs the estimates and confidence intervals from the [fold scores](nc_release/ahlmann_eltze/fold_scores.csv) and 1,000 paired bootstrap draws included here. It requires only NumPy and pandas.

```sh
python nc_release/ahlmann_eltze/table5.py --out results/Table_5.csv
```

**Recalculate from predictions.** Extract `IGC_data_v2.0.0.zip` so that `../data` points to its `data/` directory. The [Zenodo data archive](https://doi.org/10.5281/zenodo.23990921) supplies the measured RPE1 responses, frozen comparator predictions, calibration panels and intervention resampling weights. This repository supplies the five sets of [Ahlmann-Eltze predictions](nc_release/ahlmann_eltze/predictions). For the full calculation, use a separate Python 3.10 environment with the analysis dependencies:

```sh
python -m pip install -r nc_release/ahlmann_eltze/requirements.txt
python nc_release/ahlmann_eltze/score.py ../data results/from_predictions
```

This writes `Table_5.csv`, fold-level scores and paired bootstrap draws. Global ρ and Top-10 use the same scoring genes and references as Fig. 6b. CA compares each predicted response with measured candidates in its fold. The table reports pointwise 95% intervals for individual methods and simultaneous 95% intervals for paired gains over Ahlmann-Eltze. Each paired interval family covers ρ and Top-10 under both scoring references and CA.

**Refit the linear baseline.** We used the single-gene model in which both embeddings are learned from RPE1 training expression. Each fold selects only its training conditions and unperturbed controls before PCA. The unseen gene is identified by its expression across these training conditions. The fit uses 10 principal components per embedding, ridge penalty 0.1 on each side and seed 1, following the [paper](https://doi.org/10.1038/s41592-025-02772-6) and [authors’ implementation](https://github.com/const-ae/linear_perturbation_prediction-Paper/blob/bfa6eeea2bd145a1af2ec0127a2e808cc38456a9/benchmark/src/run_linear_pretrained_model.R).

The input files under [profiles/](nc_release/ahlmann_eltze/profiles) store the expression sums and cell counts needed to reconstruct each condition mean for all 8,749 genes. The four files partition conditions in their original order. The fit was run on CPU with R 4.6.1, Matrix 1.7-5 and irlba 2.3.7. With Rscript on your PATH, run:

```sh
Rscript -e "install.packages(c('Matrix', 'irlba'), repos='https://cloud.r-project.org')"
python nc_release/ahlmann_eltze/fit.py ../data results/refitted
python nc_release/ahlmann_eltze/score.py ../data results/from_refit --predictions results/refitted
```

To refit one fold, add `--fold 0` to the fit command. Both scoring commands use the published TxPert and residual-transfer predictions. To regenerate the condition profiles from the original [scPertEval RPE1 matrix](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/docs/user-guide/datasets.md), run `python nc_release/ahlmann_eltze/prepare_profiles.py replogle22rpe1_processed_complete.h5ad results/profiles` and pass `--profiles results/profiles` to the fit command. The processed H5AD used here has SHA-256 `25cb5bad6cd7abd834baa191ffa4b0b414dbfe3f5c6b927999380d2d4bc3ae3d`.

## Upstream data

- The K562 and RPE1 Perturb-seq data come from [Replogle et al. (2022)](https://doi.org/10.1016/j.cell.2022.05.013).
- The Jurkat Perturb-seq data come from [Nadig et al. (2025)](https://doi.org/10.1038/s41588-025-02169-3) and are available from GEO under [GSE264667](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667).
- Tahoe drug-response predictions: [State study and released resources](https://doi.org/10.1016/j.cell.2026.07.052).
- Time-resolved drug perturbation proteomics: [ProteinTalks](https://doi.org/10.1038/s41586-026-11001-9).
- Genetic time course: [RENGE](https://doi.org/10.1038/s42003-023-05594-4), [GSE213069](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213069).
- Melanoma context-transfer analysis: [Frangieh et al.](https://doi.org/10.1038/s41588-021-00779-1).

Project code is distributed under the MIT License. Upstream software and source datasets retain their own terms. The [earlier analysis release](https://github.com/yongqih/virtual-cell-intervention-geometry/tree/1b43e24ae09048af03630c4ef99e216fbe887742) remains accessible by its commit.
