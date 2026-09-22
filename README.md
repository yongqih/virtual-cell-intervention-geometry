# Intervention geometry analysis code

This repository accompanies "Virtual-cell models compress unseen intervention geometry through a target-specific generalization bottleneck". It contains the released analysis sources, installation instructions, recorded environments and a small executable demonstration. Derived manuscript source data are available from the matching data archive below.

Source repository: https://github.com/yongqih/virtual-cell-intervention-geometry

Published code: https://doi.org/10.5281/zenodo.22692691

Published source data: https://doi.org/10.5281/zenodo.22692573

The analysis and figure-generation Python files and original configurations match the published code archive. This repository also provides additional documentation, environment records and a command-line interface to the published geometry function. These additions are versioned here and are not asserted to be included in the earlier DOI archive.

## 1. System requirements and tested versions

The review workflows below were tested on **Windows 11 Pro, version 10.0.26200 (64-bit), Python 3.12.14**, using an AMD Ryzen AI 9 HX 370 CPU (12 cores, 24 logical processors) and 32 GB installed RAM. The CPU demo and figure reproduction require no GPU or other specialized hardware. Allow approximately 1 GB of free space for the extracted package, environment and generated outputs; this is a practical allowance, not a measured minimum hardware requirement. Linux and macOS were not tested in this submission check.

The directly used review dependencies are:

| Package | Tested version | Use |
| --- | --- | --- |
| NumPy | 2.2.6 | Response matrices and numerical operations |
| pandas | 2.2.3 | Source tables and outputs |
| SciPy | 1.15.3 | Cosine distances and Spearman correlation |
| Matplotlib | 3.10.3 | Figure generation; Agg backend |
| Pillow | 11.2.1 | Image operations in figure scripts |

`code/requirements/requirements-review.lock.txt` pins every installed runtime dependency, including transitive dependencies. `requirements-review.txt` lists the five direct dependencies. Python's standard library supplies the remaining demo imports.

Original study environments are recorded separately in `code/environments/`. The GEARS/scGPT records specify Python 3.10.20, PyTorch 2.11.0 with CUDA 12.8, torch-geometric 2.8.0.post1, cell-gears 0.1.2 and scGPT 0.2.5; associated NumPy, pandas, SciPy, scikit-learn, Scanpy, AnnData, Seaborn and other dependency versions are included in the lock files and conda YAML files. Those runs used an NVIDIA GeForce RTX 5070 Ti Laptop GPU. The orientation-analysis record reports Windows 11 build 26200, Python 3.12.13, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.0, scikit-learn 1.9.0 and JAX 0.4.35 on CPU. These are historical execution records, not claims that model training was repeated for this submission check.

A module-level import list is provided in `code/environments/module_dependencies.json`. Dependencies such as AnnData, scikit-learn, Seaborn, PyTorch and JAX belong to the original analysis workflows and are not needed for the review commands below. Use separate environments for the recorded model pipelines; their package versions differ from the lightweight demo environment.

## 2. Installation

Install Python 3.12. Clone this repository into a directory named `code` within a new working directory. Run all commands from that working directory (the parent of `code/`). On Windows PowerShell:

```powershell
mkdir igc_reproduction
cd igc_reproduction
git clone https://github.com/yongqih/virtual-cell-intervention-geometry.git code
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r code/requirements/requirements-review.lock.txt
```

The commands below use `python` for readability. On Windows, replace it with `.venv\Scripts\python.exe`, or activate the environment before running them. No package build, compiler, GPU toolkit or model download is required for the review workflows. The original model environments are supplied as records and are not installed by this procedure.

On the stated desktop, creating a clean virtual environment took **6.66 seconds** and downloading/installing dependencies without a pip cache took **61.74 seconds**, or **68.40 seconds total**. Python was already installed; its installation and code/data download are excluded. Allow roughly **1-5 minutes** after Python is available, depending on network and disk speed. The exact tested dependency set is preserved in the lock file.

## 3. Small demonstration

```text
python code/reproduction/evaluate_responses.py --input code/demo/example_responses.npz --output output/geometry_demo
```

The small simulated dataset contains 12 interventions, 8 readout genes and 2 groups. The wrapper calls the released geometry function without changing it. Expected files are `output/geometry_demo/grouped_geometry.csv` and `output/geometry_demo/summary.json`. The group correlations are **0.11071428571428568** and **0.682142857142857**; their mean is **0.39642857142857135**. Each group contributes 15 within-group pairs, with no cross-group pairs. Expected files and the complete input schema are in `code/demo/`.

The first invocation took **7.00 seconds**, including Python/import startup. Allow about **5-15 seconds** on a comparable desktop. These simulated outputs demonstrate software behavior and are not manuscript results.

## 4. Use with your own data

See [demo/README.md](demo/README.md) for the complete NPZ schema and an example that converts aligned response CSV files. Supply predicted and observed control-relative responses with matching intervention and gene axes, evaluation-group labels, intervention target symbols and a strict-trans readout panel. Each group must use a single fitted model and biological context and contain at least four interventions. Apply the intended normalization and shared-response subtraction before evaluation. The software calculates evaluation metrics; it does not train a predictor or infer missing responses.

```text
python code/reproduction/evaluate_responses.py --input my_responses.npz --output output/my_evaluation
```

The output has one geometry value per group and their unweighted mean. Zero vectors, non-finite values, inconsistent array dimensions, duplicate targets within a group and evaluated target genes in the readout are rejected. Supplied predictions must have been generated independently of held response truth.

## 5. Reproduce manuscript figures from released data

Download the source-data archive from [Zenodo](https://doi.org/10.5281/zenodo.22692573) and extract it so that `data/` sits alongside `code/`. The small demonstration above does not require this download. Run the following commands from their shared parent directory:

```text
python code/reproduction/make_all_figures.py --data data --output output/base_figures
python code/reproduction/validate_release.py --data data --figures output/base_figures
python code/figure_revision/build_revision_figures.py
python code/figure_revision/build_figure_cleanup_round2.py
```

All four commands passed in the clean review environment. Measured wall-clock times were:

| Command | Seconds | Expected result |
| --- | ---: | --- |
| Base figures | 8.62 | Six main and ten supplementary figures, PNG and SVG, plus source tables |
| Release validation | 1.72 | `RELEASE VALIDATION PASS` |
| Orientation and temporal figures | 5.87 | Main Figures 4-5 and Supplementary Figures 7-10 |
| Supplementary Figure 11 | 1.72 | Paired-context transfer figure and panels |

The four commands took **17.94 seconds** in total. Allow approximately **20-60 seconds** on a comparable desktop; startup, fonts and storage affect timing. The base command writes to `output/base_figures`. The orientation/temporal command writes to `results/formal_manuscript_revision_20260817/`. The last command writes within its `figure_cleanup_round2/` subdirectory. All paths are relative to the working directory containing `code/` and `data/`.

For the manuscript figure set, use the base outputs except replace main Figures 4-5 and Supplementary Figures 7-10 with the corresponding orientation/temporal outputs, then add Supplementary Figure 11. This yields six main and eleven supplementary figures. Figure generation uses the released source tables and does not fit models, recalculate bootstrap intervals or alter the source data. Running in this separate working directory keeps the outputs separate from manuscript files.

## 6. Analysis sources and scope

The source directories contain perturbation-response metrics, residual orientation analyses, temporal propagation and same-target cross-context transfer implementations. `code/configs/` records their analysis definitions. `data/DATASET_PROVENANCE.csv`, preprocessing records and source tables identify the underlying public resources. The released figure workflow and generic response evaluator are the executable entry points tested here. Original dataset-specific workflows additionally require their documented raw data, saved predictions and preprocessing artifacts; large third-party H5AD files and model checkpoints are not included in this repository or the derived-data archive. Their historical environment records should not be interpreted as a tested, single-command reconstruction of all model training.

The numerical analysis methods are described in the manuscript Methods. This repository provides the unchanged study source files for inspection and the commands above for figure reproduction and evaluation on new response matrices.

## License

Project-authored code, including the demonstration wrapper, is distributed under the MIT License in `code/LICENSE`. Third-party packages and public datasets retain their respective licenses. Software compatibility and timing records are in [reproduction/TESTED_WORKFLOWS.md](reproduction/TESTED_WORKFLOWS.md).
