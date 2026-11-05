"""Record panel bounds in normalized figure coordinates for layout inspection."""
import json
from pathlib import Path

def record_panel_positions(figure, *, axes, panel_ids, json_out=None, **options):
    if len(axes) != len(panel_ids) or len(set(panel_ids)) != len(panel_ids):
        raise ValueError('Each plotted axis must have a distinct panel identifier.')
    bounds = {}
    for name, axis in zip(panel_ids, axes):
        x, y, w, h = axis.get_position().bounds
        if min(x, y) < -1e-6 or max(x + w, y + h) > 1.000001:
            raise ValueError(f'Panel lies outside the figure canvas: {name}')
        bounds[name] = [float(v) for v in (x, y, w, h)]
    report = {'status': 'within canvas', 'panel_bounds': bounds}
    if json_out is not None:
        Path(json_out).write_text(json.dumps(report, indent=2), encoding='utf8')
    return report
