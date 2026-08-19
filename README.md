# Canonical audited reproduction code

This repository contains the curated analysis and figure-reproduction code retained in the final audited manuscript. It includes the established-model figure pipeline; low-dimensional orientation coding; out-of-fold residual-axis construction; the matched random-subspace control; Jiang pathway confirmation; sign-only and fixed-radius decompositions; the RENGE temporal, early same-target orientation and sign-reliability analyses; and the frozen RPE1 natural-fluctuation source-anchor audit used for Supplementary Figure 5e–h.

## Verified archives

- Code: [10.5281/zenodo.21998676](https://doi.org/10.5281/zenodo.21998676)
- Matching source data: [10.5281/zenodo.21998614](https://doi.org/10.5281/zenodo.21998614)

These are the currently published, checksum-verified Zenodo records. The repository update adds the natural-fluctuation audit, its frozen configuration, the machine-checked panel-data builder and the revised Supplementary Figure 5a–h renderer. A new Zenodo version must be cited only after Zenodo publishes it and its checksum has been verified.

## Reproduce the established-model figures

```bash
python reproduction/make_all_figures.py --data <path-to-data-directory> --output reproduced_base_figures
python reproduction/validate_release.py --data <path-to-data-directory> --figures reproduced_base_figures
```

The natural-fluctuation implementation and panel-data builder are under `analysis/natural_fluctuation_igc_anchor/`; its frozen specification is under `configs/natural_fluctuation_igc_anchor/`. The matching data package supplies derived panel source tables through `data/source_data/supplementary/SupplementaryFigure5/`, so figure regeneration does not load or redistribute the third-party RPE1 H5AD. The full audit can be rerun separately after placing the public input at the repository-relative location recorded in the frozen specification.

The retained orientation implementations are under `orientation_revalidation/`, and the revised Figure 4, Figure 5 and Supplementary Figure 7–11 builders are under `figure_revision/`. No command in the figure-only workflow loads checkpoints or retrains GEARS or scGPT. The final publication topology remains 6 main figures and exactly Supplementary Figures 1–11; Supplementary Figure 5 contains panels a–h and no supplementary figure was added or renumbered.
