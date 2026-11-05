"""Input and output locations for one figure render."""
import os
from pathlib import Path

INPUT = Path(os.environ['IGC_PLOT_INPUTS'])
OUTPUT = Path(os.environ['IGC_FIGURE_OUTPUT'])
CHECKS = OUTPUT / 'layout'
OUTPUT.mkdir(parents=True, exist_ok=True)
CHECKS.mkdir(parents=True, exist_ok=True)
