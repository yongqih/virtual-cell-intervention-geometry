# Intervention geometry analysis code

This package contains implementations for perturbation-response evaluation, residual orientation coordinates, temporal propagation and same-target cross-context transfer. It includes the RPE1 control-cell fluctuation analysis and figure generation from derived tables.

## Published archives

- Code: [10.5281/zenodo.22692691](https://doi.org/10.5281/zenodo.22692691)
- Matching source data: [10.5281/zenodo.22692573](https://doi.org/10.5281/zenodo.22692573)

The analysis code and configurations match the published code archive. This repository updates the archive links and accompanying documentation.

## Reproduce figures from source data

Extract the code and data archives into sibling `code/` and `data/` directories. Alternatively, clone this repository into a directory named `code` and extract the data archive alongside it. Run the commands below from their shared parent directory after installing the dependencies in `code/requirements/requirements-figures.txt`.

```bash
python code/reproduction/make_all_figures.py --data data --output output/base_figures
python code/reproduction/validate_release.py --data data --figures output/base_figures
python code/figure_revision/build_revision_figures.py
python code/figure_revision/build_figure_cleanup_round2.py
```

The first command generates the established-model base figures. The second checks their source records, checksums and image outputs. The third generates the orientation and temporal figures (main Figures 4 and 5; Supplementary Figures 7–10). The fourth generates Supplementary Figure 11 from the paired-context source tables. The latter two commands write to `results/formal_manuscript_revision_20260817/` relative to the directory containing `code/` and `data/`.

The base collection contains six main and ten supplementary figures. Use the orientation and temporal outputs in place of the corresponding base figures, and include Supplementary Figure 11. The data package contains source data for the final set of six main and eleven supplementary figures. Figure generation uses saved response summaries and does not train GEARS or scGPT.

## Analysis organization

- `analysis/natural_fluctuation_igc_anchor/`: target-gene covariance and correlation in unperturbed cells, state residualization, target permutations, oracle projections and training-only scalar calibration.
- `orientation_revalidation/`: source-disjoint baseline residuals, training-derived axes, matched random subspaces, sign decomposition and early-response orientation.
- `src/generalization_factorial.py` and `src/frangieh_external_replication.py`: paired-context response transfer and identity-shuffle comparisons.
- `src/propagation_reproduction/` and the RENGE modules: response waves, entry-state comparisons and propagation formulations.
- `audits/`: metric calculations, intervention-split checks and analyses of saved established-model predictions.

Some program paths and machine-readable field names retain their original identifiers so that imports, source tables and checksum records remain compatible. The experimental definitions and input requirements are given in the associated configurations. Full analyses require the public datasets and, where applicable, saved model predictions identified by those records; the figure-only commands use the released derived tables.

Code is licensed under the MIT License.
