"""
render_script_live.py — Callbacks DAT for graph_render Script TOP.

This file is loaded into TouchDesigner via:
    op('/project1/render_script').text = open(r'<path>/render_script_live.py').read()

The cook(scriptOp) function is called by TD every frame (triggered by noise_trigger input).
"""
import sys as _sys
_root = r'/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought'
_venv = r'/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought/.venv_td/lib/python3.11/site-packages'
for _p in (_root, _venv):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

# Clear cached graph_renderer so the next cook re-imports from disk.
# This runs whenever TD reloads/recompiles the DAT content.
_sys.modules.pop('touchdesigner.graph_renderer', None)


def cook(scriptOp):
    """Called by TouchDesigner each frame to render the knowledge graph."""
    g = op('/project1').fetch('graph', None)
    if g is None:
        # No graph yet — output black frame
        import numpy as np
        scriptOp.copyNumpyArray(
            np.zeros((scriptOp.height, scriptOp.width, 4), dtype=np.float32)
        )
        return
    try:
        import numpy as np
        from touchdesigner.graph_renderer import render_to_rgba
        # Use the operator's reported dimensions, but enforce a minimum of 1280×720
        # so the graph renders at full quality regardless of TD's internal resolution state.
        out_w = max(scriptOp.width,  1280)
        out_h = max(scriptOp.height,  720)
        rgba = render_to_rgba(g, width=out_w, height=out_h)
        scriptOp.copyNumpyArray(np.ascontiguousarray(rgba, dtype=np.float32))
    except Exception as e:
        print(f'[render] ERROR: {e}')
        import traceback
        traceback.print_exc()
