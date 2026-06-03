"""
setup_3d.py — Builds the 3D ("Iron Man") graph viewer inside TouchDesigner.

WHAT IT BUILDS
==============
Under /project1 (placed below the existing 2D pipeline):

    node_positions   Script CHOP   one sample per node  → tx,ty,tz,r,g,b,scale
    edge_positions   Script CHOP   one sample per edge  → p1xyz, p2xyz, weight

    nodes_geo        Geometry COMP   sphere SOP instanced over node_positions
    edges_geo        Geometry COMP   line SOP   instanced over edge_positions
    light1           Light COMP      single point light
    cam1             Camera COMP     orbit/dolly controlled from cam_ctl CHOP

    cam_ctl          Math/Expression CHOPs driven by Mouse In →
                       azimuth, elevation, distance

    render3d         Render TOP      cam1 + nodes_geo + edges_geo + light1
    webcam3d_in      Video Device In TOP  (re-uses webcam_in if it exists)
    composite3d      Over TOP        render3d over webcam (AR passthrough)
    out3d            Out TOP

PREREQUISITES
=============
- td_auto_setup.py has already been run (creates /project1/graph_store and the
  physics engine with sphere_mode=True).
- The two callback files live next to this one:
      touchdesigner/positions_chop.py
      touchdesigner/edges_chop.py

HOW TO RUN (from TD Textport)
=============================
    from touchdesigner import setup_3d
    setup_3d.build_3d_scene()

Re-running is safe — every operator is recreated (destroy-then-create).

WORLD-SPACE MAPPING
===================
positions_chop.py / edges_chop.py map the physics-space sphere (radius 480,
centred at (960, 540, 0)) onto a TD world-space sphere of radius SCENE_RADIUS
centred at the origin. Camera defaults sit just outside that radius so the
viewer initially sees the whole globe; pinch/scroll to dolly inside it for an
"around me" feeling.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TD_DIR = PROJECT_ROOT / 'touchdesigner'


# ---------------------------------------------------------------------------
# Helpers (mirroring td_auto_setup conventions)
# ---------------------------------------------------------------------------
def _make(BASE, op_type, name, x=0, y=0):
    existing = BASE.op(name)
    if existing is not None:
        existing.destroy()
    n = BASE.create(op_type, name)
    n.nodeX = x
    n.nodeY = y
    return n


def _set_par(op_obj, par_name, value, *fallbacks):
    for name in (par_name,) + fallbacks:
        try:
            setattr(op_obj.par, name, value)
            return True
        except Exception:
            continue
    print(f'[setup_3d] WARN: could not set {op_obj.name}.par.{par_name} = {value!r}')
    return False


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def build_3d_scene(base_path: str = '/project1') -> None:
    BASE = op(base_path)  # noqa: F821 — `op` injected by TD
    if BASE is None:
        raise RuntimeError(f'Base path not found: {base_path}')

    print('[setup_3d] Building 3D scene under', base_path)

    # ------------------------------------------------------------------ #
    # 1. Script CHOPs that surface the live 3D positions                  #
    # ------------------------------------------------------------------ #
    # positions_chop.py
    pos_script_dat = _make(BASE, textDAT, 'positions_chop_script', x=600, y=400)  # noqa: F821
    pos_script_dat.text = (TD_DIR / 'positions_chop.py').read_text()

    node_pos = _make(BASE, scriptCHOP, 'node_positions', x=800, y=400)  # noqa: F821
    _set_par(node_pos, 'callbacks', pos_script_dat.path, 'dat', 'callbackdat')

    # edges_chop.py
    edge_script_dat = _make(BASE, textDAT, 'edges_chop_script', x=600, y=300)  # noqa: F821
    edge_script_dat.text = (TD_DIR / 'edges_chop.py').read_text()

    edge_pos = _make(BASE, scriptCHOP, 'edge_positions', x=800, y=300)  # noqa: F821
    _set_par(edge_pos, 'callbacks', edge_script_dat.path, 'dat', 'callbackdat')

    # ------------------------------------------------------------------ #
    # 2. Nodes geometry — Sphere SOP instanced over node_positions        #
    # ------------------------------------------------------------------ #
    nodes_geo = _make(BASE, geometryCOMP, 'nodes_geo', x=1000, y=400)  # noqa: F821

    # Inside the Geometry COMP, build a small sphere SOP that becomes the
    # per-instance mesh.
    inner = nodes_geo
    # remove default torus
    default_sop = inner.op('torus1')
    if default_sop is not None:
        default_sop.destroy()
    node_sphere = inner.create(sphereSOP, 'node_sphere')  # noqa: F821
    _set_par(node_sphere, 'rad', 0.08, 'radx')   # small marker sphere
    _set_par(node_sphere, 'rady', 0.08)
    _set_par(node_sphere, 'radz', 0.08)
    _set_par(node_sphere, 'rows', 12)
    _set_par(node_sphere, 'cols', 18)

    # Enable instancing on the Geometry COMP itself.
    _set_par(nodes_geo, 'instanceop', node_pos.path, 'instancechop')
    _set_par(nodes_geo, 'instancing', True)
    _set_par(nodes_geo, 'instancetx', 'tx')
    _set_par(nodes_geo, 'instancety', 'ty')
    _set_par(nodes_geo, 'instancetz', 'tz')
    _set_par(nodes_geo, 'instancesx', 'scale')
    _set_par(nodes_geo, 'instancesy', 'scale')
    _set_par(nodes_geo, 'instancesz', 'scale')
    # Per-instance colour via channels r/g/b → instance shader sees as Cd.
    _set_par(nodes_geo, 'instancecr', 'r')
    _set_par(nodes_geo, 'instancecg', 'g')
    _set_par(nodes_geo, 'instancecb', 'b')

    # ------------------------------------------------------------------ #
    # 3. Edges geometry — Line SOP instanced over edge_positions          #
    # ------------------------------------------------------------------ #
    edges_geo = _make(BASE, geometryCOMP, 'edges_geo', x=1000, y=300)  # noqa: F821
    default_sop = edges_geo.op('torus1')
    if default_sop is not None:
        default_sop.destroy()
    # A unit line from (0,0,0) to (1,0,0); per-instance translate+rotate+scale
    # will not by itself draw the segment between two arbitrary endpoints, so
    # for the MVP we drop a Script SOP that builds the polyline directly from
    # the live graph instead of instancing.
    edge_script_sop_dat = _make(BASE, textDAT, 'edges_sop_script', x=600, y=200)  # noqa: F821
    edge_script_sop_dat.text = _EDGES_SOP_CODE
    line_sop = edges_geo.create(scriptSOP, 'edge_lines')  # noqa: F821
    _set_par(line_sop, 'callbacks', edge_script_sop_dat.path, 'dat', 'callbackdat')

    # ------------------------------------------------------------------ #
    # 4. Camera + lights                                                  #
    # ------------------------------------------------------------------ #
    cam = _make(BASE, cameraCOMP, 'cam1', x=1200, y=400)  # noqa: F821
    # Initial camera transform — distance from origin along +Z.
    _set_par(cam, 'tz', 10.0, 'tz3')

    light = _make(BASE, lightCOMP, 'light1', x=1200, y=300)  # noqa: F821
    _set_par(light, 'tx', 6.0)
    _set_par(light, 'ty', 6.0)
    _set_par(light, 'tz', 8.0)

    # ------------------------------------------------------------------ #
    # 5. Orbit/dolly camera controller from Mouse In CHOP                 #
    # ------------------------------------------------------------------ #
    mouse_chop = _make(BASE, mouseinCHOP, 'cam_mouse', x=1000, y=500)  # noqa: F821
    # tx/ty in mouse CHOP report normalised mouse position; wheel is on 'mw'.

    cam_ctl_dat = _make(BASE, textDAT, 'cam_ctl_script', x=1100, y=500)  # noqa: F821
    cam_ctl_dat.text = _CAM_CTL_CODE

    cam_exec = _make(BASE, executeDAT, 'cam_exec', x=1300, y=500)  # noqa: F821
    cam_exec.text = _CAM_EXEC_CODE
    _set_par(cam_exec, 'framestart', True, 'onframestart')

    # ------------------------------------------------------------------ #
    # 6. Render TOP + AR composite                                        #
    # ------------------------------------------------------------------ #
    render3d = _make(BASE, renderTOP, 'render3d', x=1400, y=400)  # noqa: F821
    _set_par(render3d, 'camera', cam.path)
    _set_par(render3d, 'lights', light.path)
    _set_par(render3d, 'geometry', f'{nodes_geo.path} {edges_geo.path}')
    _set_par(render3d, 'resolutionw', 1280)
    _set_par(render3d, 'resolutionh', 720)

    # Reuse existing webcam_in TOP if present; otherwise create one.
    webcam = BASE.op('webcam_in') or _make(BASE, videodeviceinTOP, 'webcam_in', x=1400, y=300)  # noqa: F821

    composite = _make(BASE, overTOP, 'composite3d', x=1600, y=400)  # noqa: F821
    # over TOP: input0 = top layer, input1 = bottom. We want render3d on top
    # of webcam.
    render3d.outputConnectors[0].connect(composite.inputConnectors[0])
    webcam.outputConnectors[0].connect(composite.inputConnectors[1])

    out3d = _make(BASE, outTOP, 'out3d', x=1800, y=400)  # noqa: F821
    composite.outputConnectors[0].connect(out3d.inputConnectors[0])

    print('[setup_3d] 3D scene built. View /project1/composite3d for AR passthrough.')
    print('[setup_3d] Drag in cam_mouse area to orbit, scroll to dolly.')


# ---------------------------------------------------------------------------
# Embedded Script SOP code — builds the edge polylines each cook.
# ---------------------------------------------------------------------------
_EDGES_SOP_CODE = '''"""edges_sop — Builds 3D polylines for every graph edge each cook."""
import sys as _sys
_root = r\'''' + str(PROJECT_ROOT) + '''\'
_venv = r\'''' + str(PROJECT_ROOT / '.venv_td/lib/python3.11/site-packages') + '''\'
for _p in (_root, _venv):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

PHYS_CENTER = (960.0, 540.0, 0.0)
PHYS_RADIUS = 480.0
SCENE_RADIUS = 4.0

def _scale(nx, ny, nz):
    s = SCENE_RADIUS / PHYS_RADIUS
    return ((nx - PHYS_CENTER[0]) * s,
            (ny - PHYS_CENTER[1]) * s,
            (nz - PHYS_CENTER[2]) * s)

def onCook(scriptOp):
    scriptOp.clear()
    graph = op('/project1').fetch('graph', None)
    if graph is None:
        return
    node_by_id = {n.id: n for n in graph.nodes}
    for edge in graph.edges:
        a = node_by_id.get(edge.source_id)
        b = node_by_id.get(edge.target_id)
        if a is None or b is None:
            continue
        ax, ay, az = _scale(a.x, a.y, a.z)
        bx, by, bz = _scale(b.x, b.y, b.z)
        p1 = scriptOp.appendPoint()
        p2 = scriptOp.appendPoint()
        p1.x, p1.y, p1.z = ax, ay, az
        p2.x, p2.y, p2.z = bx, by, bz
        poly = scriptOp.appendPoly(2, closed=False, addPoints=False)
        poly[0].point = p1
        poly[1].point = p2
'''


# ---------------------------------------------------------------------------
# Camera control DAT — orbit/dolly state machine
# ---------------------------------------------------------------------------
_CAM_CTL_CODE = '''"""cam_ctl_script — internal state for orbit/dolly camera."""
# State is held on the module so cam_exec onFrameStart can mutate it.
azimuth   = 30.0   # degrees around Y
elevation = 15.0   # degrees above XZ plane
distance  = 10.0   # camera distance from origin (TD units)

# Drag state
_dragging   = False
_last_mx    = 0.0
_last_my    = 0.0

# Sensitivity
ORBIT_SENS  = 180.0   # degrees per normalised-screen unit
DOLLY_SENS  = 1.5     # units per scroll tick
MIN_DIST    = 1.5
MAX_DIST    = 30.0
'''


_CAM_EXEC_CODE = '''"""cam_exec — onFrameStart: poll mouse, update cam1 transform."""
import math

def onFrameStart(frame):
    mouse = op('cam_mouse')
    ctl   = mod(op('cam_ctl_script'))
    cam   = op('cam1')
    if mouse is None or ctl is None or cam is None:
        return

    mx = float(mouse['tx'])
    my = float(mouse['ty'])
    lb = float(mouse['lselect']) if 'lselect' in [c.name for c in mouse.chans()] else 0.0
    mw = float(mouse['mw']) if 'mw' in [c.name for c in mouse.chans()] else 0.0

    # ---- Orbit on left-drag ----
    if lb > 0.5:
        if not ctl._dragging:
            ctl._dragging = True
            ctl._last_mx  = mx
            ctl._last_my  = my
        else:
            dx = mx - ctl._last_mx
            dy = my - ctl._last_my
            ctl.azimuth   += dx * ctl.ORBIT_SENS
            ctl.elevation += dy * ctl.ORBIT_SENS
            ctl.elevation = max(-85.0, min(85.0, ctl.elevation))
            ctl._last_mx  = mx
            ctl._last_my  = my
    else:
        ctl._dragging = False

    # ---- Dolly on scroll / pinch ----
    if abs(mw) > 1e-4:
        ctl.distance -= mw * ctl.DOLLY_SENS
        ctl.distance = max(ctl.MIN_DIST, min(ctl.MAX_DIST, ctl.distance))

    # ---- Apply spherical coords to cam1 ----
    az = math.radians(ctl.azimuth)
    el = math.radians(ctl.elevation)
    d  = ctl.distance
    cx = d * math.cos(el) * math.sin(az)
    cy = d * math.sin(el)
    cz = d * math.cos(el) * math.cos(az)
    cam.par.tx = cx
    cam.par.ty = cy
    cam.par.tz = cz
    # Look at origin
    cam.par.lookat = ''   # clear any lookat target ref
    # Compute Euler so camera points to origin (simple azimuth/elevation aim).
    cam.par.rx = -ctl.elevation
    cam.par.ry =  ctl.azimuth
    cam.par.rz =  0.0
'''
