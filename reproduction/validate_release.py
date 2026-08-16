from __future__ import annotations
import argparse, csv, hashlib, re
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--figures",required=True);a=p.parse_args();data=Path(a.data);figs=Path(a.figures)
required=[f"Figure{i}" for i in range(1,7)]+[f"SupplementaryFigure{i}" for i in range(1,11)]
errors=[]
for name in required:
    for ext in ("svg","png"):
        q=figs/name/f"{name}.{ext}"
        if not q.exists():errors.append(f"missing {q}")
extra=[p.name for p in figs.glob("SupplementaryFigure*") if re.fullmatch(r"SupplementaryFigure(1[1-9]|[2-9][0-9]+)",p.name)]
if extra:errors.append(f"unexpected supplementary figures: {extra}")
manifest=data/"SOURCE_DATA_MANIFEST.csv"
if not manifest.exists():errors.append("missing SOURCE_DATA_MANIFEST.csv")
else:
    for row in csv.DictReader(manifest.open(encoding="utf-8")):
        source=data/row["path"]
        if not source.exists():errors.append(f"unresolved source {row['path']}")
        else:
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            if row.get("sha256") and digest != row["sha256"]:errors.append(f"hash mismatch {row['path']}")
for svg in figs.glob("**/*.svg"):
    text=svg.read_text(encoding="utf-8",errors="replace")
    if "<text" not in text:errors.append(f"non-editable or missing SVG text: {svg}")
    if re.search(r"AUDIT REQUIRED|scGPT pending|placeholder",text,re.I):errors.append(f"placeholder string: {svg}")
    if re.search(r"C:[\\/]Users[\\/]",text,re.I):errors.append(f"absolute Windows path: {svg}")
    if re.search(r"(?<![A-Za-z])nan(?![A-Za-z])",text,re.I):errors.append(f"NaN rendered in SVG: {svg}")
registry=figs/"FIGURE_SOURCE_VALUES.csv"
if not registry.exists():errors.append("missing FIGURE_SOURCE_VALUES.csv")
else:
    rows=list(csv.DictReader(registry.open(encoding="utf-8")))
    expected={(f"Figure{i}",p) for i, panels in {1:"cd",2:"abcdef",3:"abcdef",4:"bcde",5:"bcdef",6:"bcde"}.items() for p in panels}
    expected|={(f"SupplementaryFigure{i}",p) for i in range(1,11) for p in "ab"}
    actual={(r["figure"],r["panel"]) for r in rows}
    if expected-actual:errors.append(f"missing panel source registrations: {sorted(expected-actual)}")
    for row in rows:
        if not (figs/row["source_data_file"]).exists():errors.append(f"unresolved panel source {row['source_data_file']}")
for name in ("Figure1","Figure2"):
    text=(figs/name/f"{name}.svg").read_text(encoding="utf-8",errors="replace") if (figs/name/f"{name}.svg").exists() else ""
    if "scGPT" not in text:errors.append(f"missing scGPT label in {name}")
for rel in ("results/gears_geometry_audit/grouped_intervention_geometry.csv","results/final_scgpt_preprint_audit/scgpt_grouped_geometry.csv"):
    source=data/"original_audit_sources"/rel
    if source.exists():
        for row in csv.DictReader(source.open(encoding="utf-8")):
            value=str(row.get("cross_fold_pairs_used","")).strip().lower()
            if value not in ("","false","0","nan"):errors.append(f"cross-model/cross-fold geometry flag in {rel}")
code_root=Path(__file__).resolve().parents[1];code_manifest=code_root/"manifests/CODE_MANIFEST.csv"
if not code_manifest.exists():errors.append("missing CODE_MANIFEST.csv")
else:
    for row in csv.DictReader(code_manifest.open(encoding="utf-8")):
        source=code_root/row["path"]
        if not source.exists():errors.append(f"missing released code {row['path']}")
        elif hashlib.sha256(source.read_bytes()).hexdigest()!=row["sha256"]:errors.append(f"code hash mismatch {row['path']}")
if errors:raise SystemExit("RELEASE VALIDATION FAIL\n"+"\n".join(errors))
print("RELEASE VALIDATION PASS")
