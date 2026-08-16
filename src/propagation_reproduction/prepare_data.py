from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from common import CACHE_ROOT, DATA_ROOT, RESULT_ROOT, atomic_json, atomic_npz, now, sha256


COMMIT = "ca0d636ae47311fb5ce501f4a0a835b55379d9fa"
FILES = {
    "E_renge_d2_80.csv": ("d4282200b36e03bf4af90a786727b4ca44ecbdbb1a65ee74b7a8fb932b9152d7", 21014434),
    "X_renge_d2_80.csv": ("149b1f59c8f1e81190fc296862674dce3a0f84983e03a0119813a83d08c02498", 6531595),
}


def download(name: str, checksum: str, size: int) -> Path:
    destination = DATA_ROOT / "source" / name; destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == size and sha256(destination) == checksum:
        return destination
    url = f"https://raw.githubusercontent.com/masastat/RENGE/{COMMIT}/examples/data/{name}"
    temporary = destination.with_name(f".{name}.{os.getpid()}.tmp")
    with urlopen(Request(url, headers={"User-Agent": "AI4Sci-propagation-reproduction/1.0"}), timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(8 << 20): output.write(chunk)
        output.flush(); os.fsync(output.fileno())
    if temporary.stat().st_size != size or sha256(temporary) != checksum:
        raise RuntimeError(f"Official file integrity failure: {name}")
    os.replace(temporary, destination); return destination


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True); CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {name: download(name, *specification) for name, specification in FILES.items()}
    expression = pd.read_csv(paths["E_renge_d2_80.csv"], index_col=0)
    design = pd.read_csv(paths["X_renge_d2_80.csv"], index_col=0)
    if not expression.index.equals(design.index): raise RuntimeError("E/X cell order mismatch")
    ko = design.drop(columns="t"); genes = np.asarray(expression.columns.astype(str), dtype=str)
    if set(genes) != set(ko.columns.astype(str)): raise RuntimeError("E/X gene-set mismatch")
    expression = expression.reindex(sorted(expression.columns), axis=1)
    ko = ko.reindex(expression.columns, axis=1); genes = np.asarray(expression.columns.astype(str), dtype=str)
    row_sum = ko.sum(1).to_numpy(); gene_array = np.asarray(ko.columns.astype(str), dtype=str)
    if not np.all(np.isin(row_sum, [0., 1.])): raise RuntimeError("Design rows must be control or one KO")
    assignment = np.full(len(ko), "control", dtype=f"<U{max(map(len, gene_array))}")
    perturbed = row_sum == 1; assignment[perturbed] = gene_array[np.argmax(ko.to_numpy()[perturbed], axis=1)]
    times = design["t"].astype(int).to_numpy(); unique_times = np.asarray([2, 3, 4, 5])
    sources = np.asarray(sorted(set(assignment) - {"control"}), dtype=str)
    values = expression.to_numpy(np.float32)
    response = np.empty((len(sources), len(unique_times), len(genes)), np.float32)
    cell_count = np.zeros((len(sources), len(unique_times)), np.int64)
    controls = np.empty((len(unique_times), len(genes)), np.float32)
    static = np.empty((len(genes), len(unique_times) * 3), np.float32)
    for time_index, day in enumerate(unique_times):
        control_cells = values[(assignment == "control") & (times == day)]
        controls[time_index] = control_cells.mean(0)
        static[:, time_index * 3:(time_index + 1) * 3] = np.column_stack([
            control_cells.mean(0), control_cells.std(0), (control_cells > 0).mean(0)])
        for source_index, source in enumerate(sources):
            cells = values[(assignment == source) & (times == day)]
            if not len(cells): raise RuntimeError(f"Missing source/day: {source}/{day}")
            response[source_index, time_index] = cells.mean(0) - controls[time_index]
            cell_count[source_index, time_index] = len(cells)
    waves = np.diff(response, axis=1)
    output = CACHE_ROOT / "renge_processed.npz"
    atomic_npz(output, expression=values, assignment=assignment, times=times, genes=genes, sources=sources,
               days=unique_times, response=response, waves=waves, matched_controls=controls,
               static_control_representation=static, cell_count=cell_count)
    numerical_audit = {
        "all_values_nonnegative": bool((values >= 0).all()),
        "fraction_expm1_values_within_1e_8_of_integer": float((np.abs(np.expm1(values) - np.rint(np.expm1(values))) < 1e-8).mean()),
        "note": "The author example is used without additional normalization. Numerical audit is reported because its nonnegative log-like scale should not be conflated with raw counts or re-normalized silently."
    }
    provenance = {
        "accessed_at": now(),
        "paper": {"title": "RENGE infers gene regulatory networks using time-series single-cell RNA-seq data with CRISPR perturbations",
                  "authors_short": "Ishikawa et al.", "journal": "Communications Biology", "year": 2023,
                  "doi": "10.1038/s42003-023-05594-4", "pmcid": "PMC10754834"},
        "dataset": {"name": "single cell CRISPR analysis in human pluripotent stem cells",
                    "geo_accession": "GSE213069", "bioproject": "PRJNA879051",
                    "geo_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213069",
                    "days": unique_times.tolist(), "ko_sources": len(sources), "modeled_genes": len(genes),
                    "cells": len(values), "controls": int((assignment == "control").sum()),
                    "cells_by_day": {str(day): int((times == day).sum()) for day in unique_times},
                    "control_cells_by_day": {str(day): int(((times == day) & (assignment == "control")).sum()) for day in unique_times},
                    "minimum_cells_per_source_day": int(cell_count.min()), "median_cells_per_source_day": float(np.median(cell_count))},
        "official_code": {"repository": "https://github.com/masastat/RENGE", "commit": COMMIT,
                          "zenodo_doi": "10.5281/zenodo.10114567"},
        "files": [{"filename": name, "source_url": f"https://raw.githubusercontent.com/masastat/RENGE/{COMMIT}/examples/data/{name}",
                   "bytes": paths[name].stat().st_size, "sha256": sha256(paths[name]),
                   "line_ending_note": "Hash is for GitHub raw LF bytes; a Windows git checkout may be larger and hash differently after CRLF conversion."} for name in FILES],
        "processing": {"published_paper_normalization": "Seurat 4.0.3 sctransform after cell QC; single-gRNA cells retained",
                       "analysis_input": "official RENGE author example E/X matrices, used unchanged",
                       "response_definition": "source/day cell mean minus matched same-day control cell mean",
                       "wave_definition": "population response difference between adjacent days; not a single-cell trajectory",
                       "numerical_audit": numerical_audit},
    }
    atomic_json(RESULT_ROOT / "data_provenance.json", provenance)
    print(json.dumps({"processed_cache": str(output), **provenance["dataset"], "numerical_audit": numerical_audit}, indent=2))


if __name__ == "__main__":
    main()
