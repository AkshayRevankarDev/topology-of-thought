"""
positions_chop.py — Script CHOP callback for the 3D node-positions stream.

Attach this file to a Script CHOP (``node_positions``) inside ``/project1``.
Each cook the script reads the live GraphState held by the physics engine and
emits one sample per node across the following channels:

    tx, ty, tz      — world-space position in TD units (sphere physics output
                      is in 1920×1080×R space; we recentre to the origin and
                      scale to a tidy radius for the 3D viewport).
    r,  g,  b       — RGB colour derived from node.confidence.
    scale           — per-instance scale (currently constant; used later for
                      selected/expanded emphasis).
    selected        — 0/1 flag for the currently-selected node.

The physics engine writes positions in a frame centred on
(960, 540, 0) with sphere radius 480 (see td_auto_setup.py).  We map that into
TD world space centred on the origin at radius ``SCENE_RADIUS``.
"""
from __future__ import annotations

import sys as _sys

# ---------------------------------------------------------------------------
# Project import path bootstrap — keep this in sync with td_auto_setup.py.
# ---------------------------------------------------------------------------
_root = r'/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought'
_venv = r'/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought/.venv_td/lib/python3.11/site-packages'
for _p in (_root, _venv):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# World-space mapping
# ---------------------------------------------------------------------------
# Physics frame:  origin at (960, 540, 0), sphere radius = 480 units.
# TD scene frame: origin at (0, 0, 0), radius = SCENE_RADIUS units.
PHYS_CENTER = (960.0, 540.0, 0.0)
PHYS_RADIUS = 480.0
SCENE_RADIUS = 4.0   # TD world units; tweak for "around me" sizing


def _scale_xyz(nx: float, ny: float, nz: float):
    """Map a physics-space (x, y, z) to TD world coordinates centred on origin."""
    s = SCENE_RADIUS / PHYS_RADIUS
    return (
        (nx - PHYS_CENTER[0]) * s,
        (ny - PHYS_CENTER[1]) * s,
        (nz - PHYS_CENTER[2]) * s,
    )


def _confidence_to_rgb(c: float):
    """Dim blue-grey → bright sky-blue along the confidence axis."""
    c = max(0.0, min(1.0, float(c)))
    r = 0.10 + 0.30 * c
    g = 0.40 + 0.50 * c
    b = 0.70 + 0.30 * c
    return r, g, b


# ---------------------------------------------------------------------------
# TouchDesigner callbacks
# ---------------------------------------------------------------------------
def onSetupParameters(scriptOp):
    """Called once when the Script CHOP is created or reset."""
    return


def _get_graph():
    """Resolve the live GraphState — held on the graph_store Text DAT module."""
    store = op('/project1/graph_store')
    if store is None:
        return None
    try:
        return mod(store).graph
    except Exception:
        return None


def onCook(scriptOp):
    """Called every cook — populate channels from the live graph."""
    scriptOp.clear()

    graph = _get_graph()
    if graph is None or not graph.nodes:
        # Emit a single zero sample so downstream operators have valid shape.
        for name in ('tx', 'ty', 'tz', 'r', 'g', 'b', 'scale', 'selected'):
            scriptOp.appendChan(name)
        scriptOp.numSamples = 1
        for ch in scriptOp.chans():
            ch[0] = 0.0
        return

    nodes = graph.nodes
    n = len(nodes)

    # Allocate channels.
    chans = {
        name: scriptOp.appendChan(name)
        for name in ('tx', 'ty', 'tz', 'r', 'g', 'b', 'scale', 'selected')
    }
    scriptOp.numSamples = n

    for i, nd in enumerate(nodes):
        x, y, z = _scale_xyz(nd.x, nd.y, nd.z)
        r, g, b = _confidence_to_rgb(getattr(nd, 'confidence', 1.0))
        chans['tx'][i] = x
        chans['ty'][i] = y
        chans['tz'][i] = z
        chans['r'][i] = r
        chans['g'][i] = g
        chans['b'][i] = b
        chans['scale'][i] = 1.0 if not getattr(nd, 'selected', False) else 1.8
        chans['selected'][i] = 1.0 if getattr(nd, 'selected', False) else 0.0
