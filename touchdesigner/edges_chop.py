"""
edges_chop.py — Script CHOP callback exposing edge endpoints to TD.

Attach this file to a Script CHOP (``edge_positions``) inside ``/project1``.
Each cook emits one sample per graph edge with channels:

    p1x, p1y, p1z   — source-node world-space position
    p2x, p2y, p2z   — target-node world-space position
    weight          — edge weight (0..1), used to drive line alpha/thickness
    confidence      — edge confidence (0..1)

Downstream a Line SOP (or Add SOP via Script SOP) instances a unit segment
between each (p1, p2) pair to render edges in 3D.

World-space mapping is kept identical to ``positions_chop.py`` so node and
edge endpoints stay locked together.
"""
from __future__ import annotations

import sys as _sys

_root = r'/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought'
_venv = r'/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought/.venv_td/lib/python3.11/site-packages'
for _p in (_root, _venv):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

PHYS_CENTER = (960.0, 540.0, 0.0)
PHYS_RADIUS = 480.0
SCENE_RADIUS = 4.0


def _scale_xyz(nx: float, ny: float, nz: float):
    s = SCENE_RADIUS / PHYS_RADIUS
    return (
        (nx - PHYS_CENTER[0]) * s,
        (ny - PHYS_CENTER[1]) * s,
        (nz - PHYS_CENTER[2]) * s,
    )


def onSetupParameters(scriptOp):
    return


def onCook(scriptOp):
    scriptOp.clear()

    graph = op('/project1').fetch('graph', None)
    chan_names = ('p1x', 'p1y', 'p1z', 'p2x', 'p2y', 'p2z', 'weight', 'confidence')

    if graph is None or not graph.edges or not graph.nodes:
        for name in chan_names:
            scriptOp.appendChan(name)
        scriptOp.numSamples = 1
        for ch in scriptOp.chans():
            ch[0] = 0.0
        return

    node_by_id = {nd.id: nd for nd in graph.nodes}
    valid_edges = [
        e for e in graph.edges
        if e.source_id in node_by_id and e.target_id in node_by_id
    ]

    if not valid_edges:
        for name in chan_names:
            scriptOp.appendChan(name)
        scriptOp.numSamples = 1
        for ch in scriptOp.chans():
            ch[0] = 0.0
        return

    chans = {name: scriptOp.appendChan(name) for name in chan_names}
    scriptOp.numSamples = len(valid_edges)

    for i, edge in enumerate(valid_edges):
        a = node_by_id[edge.source_id]
        b = node_by_id[edge.target_id]
        ax, ay, az = _scale_xyz(a.x, a.y, a.z)
        bx, by, bz = _scale_xyz(b.x, b.y, b.z)
        chans['p1x'][i] = ax
        chans['p1y'][i] = ay
        chans['p1z'][i] = az
        chans['p2x'][i] = bx
        chans['p2y'][i] = by
        chans['p2z'][i] = bz
        chans['weight'][i] = float(getattr(edge, 'weight', 1.0))
        chans['confidence'][i] = float(getattr(edge, 'confidence', 1.0))
