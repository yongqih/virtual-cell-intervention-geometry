"""Redraw any final figure using frozen tabular inputs. No model weights or GPU required."""
import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
JOBS = {
    'F1': ('fig1', None), 'F2': ('fig2', 'main_figure'),
    'F3': ('fig34', 'main3_new'), 'F4': ('fig34', 'main4_new'),
    'F5': ('fig5', 'export_figure'), 'F6': ('fig6', 'main'),
    'S1': ('quality', 'supplementary1'), 'S2': ('supplement_controls', 'render_s2'),
    'S3': ('triplet_examples', None), 'S4': ('fig2', 'supplementary_spectra'),
    'S5': ('fig2', 'supplementary_maps'), 'S6': ('subspaces', 'supp6'),
    'S7': ('subspaces', 'supp33'), 'S8': ('subspaces', 'supp41'),
    'S9': ('subspaces', 'supp42'), 'S10': ('subspaces', 'supp43'),
    'S11': ('temporal', 'supplement11'), 'S12': ('temporal', 'supplement12'),
    'S13': ('temporal', 'supplement13'), 'S14': ('fig6', 'old.supplement1'),
    'S15': ('supplement_controls', 'render_s15'), 'S16': ('fig6', 'old.supplement2'),
    'S17': ('fig6', 'old.supplement3'), 'S18': ('fig6', 'old.supplement4'),
    'S19': ('fig6', 's19'), 'S20': ('fig6', 's20')
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figures', nargs='*', help='F1 through F6 or S1 through S20. Default: all.')
    parser.add_argument('--data', type=Path, default=HERE / 'plot_inputs', help='Directory containing the per-figure source tables.')
    parser.add_argument('--out', type=Path, default=Path('rendered'))
    parser.add_argument('--single', choices=JOBS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.data.is_dir():
        parser.error(f'Input directory missing: {args.data}')
    if args.single:
        os.environ['IGC_PLOT_INPUTS'] = str(args.data.resolve())
        os.environ['IGC_FIGURE_OUTPUT'] = str(args.out.resolve())
        sys.path.insert(0, str(HERE / 'plotting'))
        if args.single in ('S6','S8','S9','S10','S13','S16','S17'):
            import reader_labels
        mod, func = JOBS[args.single]
        obj = importlib.import_module(mod)
        if func:
            for name in func.split('.'):
                obj = getattr(obj, name)
            obj()
        return
    for pick in args.figures or JOBS:
        if pick not in JOBS:
            parser.error(f'Unknown figure: {pick}')
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--single', pick,
                        '--data', str(args.data.resolve()), '--out', str(args.out.resolve())], check=True)

if __name__ == '__main__':
    main()
