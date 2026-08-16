from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--output",default="reproduced_figures");a=p.parse_args()
data=Path(a.data).resolve();out=Path(a.output).resolve();root=data/"original_audit_sources"
script=Path(__file__).resolve().parents[1]/"figures"/"generate_final_figures.py"
env=os.environ.copy();env["AI4SCI_RELEASE_ROOT"]=str(root);env["AI4SCI_FIGURE_OUT"]=str(out);env["PYTHONDONTWRITEBYTECODE"]="1"
subprocess.run([sys.executable,"-B",str(script)],check=True,env=env)
print(f"Generated 6 main and 10 supplementary figures in {out}")
