from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--output", default="reproduced_figures")
args = parser.parse_args()

data = Path(args.data).resolve()
output = Path(args.output).resolve()
root = data / "original_audit_sources"
script = Path(__file__).resolve().parents[1] / "figures" / "generate_final_figures.py"
natural = data / "source_data" / "supplementary" / "SupplementaryFigure5"

environment = os.environ.copy()
environment["AI4SCI_RELEASE_ROOT"] = str(root)
environment["AI4SCI_FIGURE_OUT"] = str(output)
environment["AI4SCI_NATURAL_SOURCE_DATA"] = str(natural)
environment["PYTHONDONTWRITEBYTECODE"] = "1"
subprocess.run([sys.executable, "-B", str(script)], check=True, env=environment)
print(
    "Generated the established-model base set "
    f"(6 main and 10 supplementary figures) in {output}"
)
