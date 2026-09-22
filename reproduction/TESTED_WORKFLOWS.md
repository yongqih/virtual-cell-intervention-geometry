# Tested software workflows

Date: 2026-09-21. Platform: Windows 11 Pro 10.0.26200, 64-bit; Python 3.12.14; AMD Ryzen AI 9 HX 370 CPU; 32 GB installed RAM. All review workflows ran on CPU in a newly created virtual environment.

| Check | Result |
| --- | --- |
| Dependency installation from PyPI without pip cache | PASS; 61.74 s |
| Virtual environment creation | PASS; 6.66 s |
| `pip check` | PASS; no broken requirements |
| Simulated grouped-geometry demo | PASS; 7.00 s |
| Published base figure generation | PASS; 8.62 s |
| Released source/image validation | PASS; 1.72 s |
| Published orientation/temporal figure generation | PASS; 5.87 s |
| Published Supplementary Figure 11 generation | PASS; 1.72 s |
| Identical predicted/observed responses | Geometry equals 1 |
| Group separation | Two groups of six: 15 pairs each; no cross-group pairs |
| Target included in readout panel | Rejected with a clear error |

The expected numerical demo outputs are included in `code/demo/expected/`. The direct and complete installed dependency specifications are in `code/requirements/`. Timings include process startup and exclude Python installation and archive download. First-use font initialization can affect the demo runtime.

The recorded original GEARS/scGPT and orientation environments are provided for provenance. Model training and raw-data preprocessing were not repeated for this software check. All original analysis Python files, figure-generation Python files, configurations and data were retained byte-for-byte. Only documentation, environment records, the generic evaluator wrapper, its simulated input and an adjacent copy of the geometry configuration were added or updated.
