# Canonical audited reproduction code

This repository contains the curated analysis and figure-reproduction code retained in the final audited manuscript. It includes the established-model figure pipeline; low-dimensional orientation coding; out-of-fold residual-axis construction; the matched random-subspace control; Jiang pathway confirmation; sign-only and fixed-radius decompositions; and the RENGE temporal, early same-target orientation and sign-reliability analyses.

## Archived release

- Code: [10.5281/zenodo.21998676](https://doi.org/10.5281/zenodo.21998676)
- Matching source data: [10.5281/zenodo.21998614](https://doi.org/10.5281/zenodo.21998614)

The versioned Zenodo archives contain the frozen manifests, derived source data, exact audited figure assets and the package-level one-command reproduction wrapper. Extract the matching code and data archives into the same directory before running the packaged workflows.

## Reproduce the established-model figures

```bash
python reproduction/make_all_figures.py --data <path-to-data-directory> --output reproduced_base_figures
python reproduction/validate_release.py --data <path-to-data-directory> --figures reproduced_base_figures
```

The retained orientation implementations are under `orientation_revalidation/`, and the revised Figure 4, Figure 5 and Supplementary Figure 7-11 builders are under `figure_revision/`. No command in the figure-only workflow loads checkpoints or retrains GEARS or scGPT. Optional audit replays requiring large public inputs use only repository-relative locations documented in the frozen configurations.
