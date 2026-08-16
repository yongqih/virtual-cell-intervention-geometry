# Canonical preprint reproduction code

This curated package reproduces the six main and ten supplementary figures from frozen released source data without loading checkpoints or retraining models. Exploratory branches, raw datasets, weights, results, credentials and machine-specific paths are excluded.

Run `python reproduction/make_all_figures.py --data <data_availability> --output <output>` and then `python reproduction/validate_release.py --data <data_availability> --figures <output>`.

Link to Zenodo: DOI: 10.5281/zenodo.21963187
