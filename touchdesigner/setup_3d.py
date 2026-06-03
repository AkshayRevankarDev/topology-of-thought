"""
setup_3d.py — Builds the JARVIS-style 3D holographic graph view in TD.

VISUAL DIRECTION
================
- Pure virtual space (no webcam). Deep-space background.
- Neon-cyan / electric-blue palette.  Additive bloom for the glow halo.
- Perspective grid floor for spatial reference (Iron Man HUD).
- Radial vignette focuses attention on the centre globe.
- Idle auto-rotation when the user isn't dragging — feels alive.
- Per-frame data driven: nodes / edges read live 3D positions from the
  sphere-mode PhysicsEngine via the two Script CHOPs.

WHAT IT BUILDS UNDER /project1
==============================
    node_positions   Script CHOP    one sample per node  (tx/ty/tz/r/g/b/scale)
    edge_positions   Script CHOP    one sample per edge  (p1xyz, p2xyz, ...)

    node_mat         Constant MAT   emit cyan, uses instance colour
    edge_mat         Constant MAT   emit electric blue, additive blend
    grid_mat         Constant MAT   wireframe cyan, low alpha

    nodes_geo        Geometry COMP  Sphere SOP instanced via node_positions
    edges_geo        Geometry COMP  Script SOP polylines from live edges
    grid_geo         Geometry COMP  large Grid SOP, XZ plane (floor)

    cam1             Camera COMP    orbit/dolly target (origin)
    light1           Light COMP     (kept for any non-constant materials)

    cam_mouse        Mouse In CHOP
    cam_ctl_script   Text DAT       module — orbit/dolly/idle-spin state
    cam_exec         Execute DAT    onFrameStart updates cam1 transform

    render3d         Render TOP     1280×720, deep-space bg
    glow_blur        Blur TOP       wide blur for bloom
    glow_add         Add TOP        bloom = render + blurred render
    vignette_ramp    Ramp TOP       radial dark vignette
    hud_mult         Multiply TOP   vignette * bloom
    out3d            Out TOP

HOW TO RUN
==========
After td_auto_setup has built the base graph + physics:

    import importlib, touchdesigner.setup_3d as s
    importlib.reload(s)
    s.build_3d_scene()

Then view  /project1/out3d  (or hud_mult).
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TD_DIR = PROJECT_ROOT / 'touchdesigner'


# ---------------------------------------------------------------------------
# Helpers
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
    print(f'[setup_3d] WARN: could not set {op_obj.name}.par.{par_name}')
    return False


def _maybe_destroy(BASE, name):
    existing = BASE.op(name)
    if existing is not None:
        try:
            existing.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def build_3d_scene(base_path: str = '/project1') -> None:
    # Grab TD globals from the td module (this file is imported, so `op` etc.
    # aren't injected automatically).
    import td
    op               = td.op
    textDAT          = td.textDAT
    scriptCHOP       = td.scriptCHOP
    scriptSOP        = td.scriptSOP
    sphereSOP        = td.sphereSOP
    gridSOP          = td.gridSOP
    geometryCOMP     = td.geometryCOMP
    cameraCOMP       = td.cameraCOMP
    lightCOMP        = td.lightCOMP
    nullCOMP         = td.nullCOMP
    constantMAT      = td.constantMAT
    renderTOP        = td.renderTOP
    blurTOP          = td.blurTOP
    rampTOP          = td.rampTOP
    multiplyTOP      = td.multiplyTOP
    addTOP           = td.addTOP
    outTOP           = td.outTOP
    mouseinCHOP      = td.mouseinCHOP
    executeDAT       = td.executeDAT

    BASE = op(base_path)
    if BASE is None:
        raise RuntimeError(f'Base path not found: {base_path}')

    print('[setup_3d] Building JARVIS HUD under', base_path)

    # ------------------------------------------------------------------ #
    # 0. Clean up stale ops from previous builds                          #
    # ------------------------------------------------------------------ #
    for stale in ('composite3d', 'hud_mult', 'vignette_ramp'):
        _maybe_destroy(BASE, stale)

    # ------------------------------------------------------------------ #
    # 1. Script CHOPs surfacing live 3D positions                         #
    # ------------------------------------------------------------------ #
    pos_script_dat = _make(BASE, textDAT, 'positions_chop_script', x=600, y=500)
    pos_script_dat.text = (TD_DIR / 'positions_chop.py').read_text()
    node_pos = _make(BASE, scriptCHOP, 'node_positions', x=800, y=500)
    _set_par(node_pos, 'callbacks', pos_script_dat.path, 'dat', 'callbackdat')

    edge_script_dat = _make(BASE, textDAT, 'edges_chop_script', x=600, y=400)
    edge_script_dat.text = (TD_DIR / 'edges_chop.py').read_text()
    edge_pos = _make(BASE, scriptCHOP, 'edge_positions', x=800, y=400)
    _set_par(edge_pos, 'callbacks', edge_script_dat.path, 'dat', 'callbackdat')

    # ------------------------------------------------------------------ #
    # 2. Materials — neon cyan, emissive, additive                        #
    # ------------------------------------------------------------------ #
    node_mat = _make(BASE, constantMAT, 'node_mat', x=800, y=700)
    _set_par(node_mat, 'colorr', 0.35)
    _set_par(node_mat, 'colorg', 0.95)
    _set_par(node_mat, 'colorb', 1.00)
    _set_par(node_mat, 'alpha', 1.0)
    # Honour per-instance Cd. Par name varies between TD versions / MAT pages.
    _set_par(node_mat, 'pointcolor', True,
             'usepointcolor', 'usepointcolour',
             'usepointcolors', 'pointcolors',
             'instancecolor', 'useinstancecolor')

    edge_mat = _make(BASE, constantMAT, 'edge_mat', x=900, y=700)
    _set_par(edge_mat, 'colorr', 0.15)
    _set_par(edge_mat, 'colorg', 0.70)
    _set_par(edge_mat, 'colorb', 1.00)
    _set_par(edge_mat, 'alpha', 0.85)

    grid_mat = _make(BASE, constantMAT, 'grid_mat', x=1000, y=700)
    _set_par(grid_mat, 'colorr', 0.08)
    _set_par(grid_mat, 'colorg', 0.45)
    _set_par(grid_mat, 'colorb', 0.85)
    _set_par(grid_mat, 'alpha', 0.35)

    # ------------------------------------------------------------------ #
    # 3. Nodes geometry                                                   #
    # ------------------------------------------------------------------ #
    nodes_geo = _make(BASE, geometryCOMP, 'nodes_geo', x=1000, y=500)
    default_sop = nodes_geo.op('torus1')
    if default_sop is not None:
        default_sop.destroy()
    node_sphere = nodes_geo.create(sphereSOP, 'node_sphere')
    # Bigger spheres — radius ~0.25 reads clearly at camera distance ~12.
    _set_par(node_sphere, 'radx', 0.25)
    _set_par(node_sphere, 'rady', 0.25)
    _set_par(node_sphere, 'radz', 0.25)
    _set_par(node_sphere, 'rows', 16)
    _set_par(node_sphere, 'cols', 24)
    # Critical: mark this SOP as the render+display target of the Geo COMP.
    node_sphere.render  = True
    node_sphere.display = True

    _set_par(nodes_geo, 'instanceop', node_pos.path, 'instancechop')
    _set_par(nodes_geo, 'instancing', True)
    _set_par(nodes_geo, 'instancetx', 'tx')
    _set_par(nodes_geo, 'instancety', 'ty')
    _set_par(nodes_geo, 'instancetz', 'tz')
    _set_par(nodes_geo, 'instancesx', 'scale')
    _set_par(nodes_geo, 'instancesy', 'scale')
    _set_par(nodes_geo, 'instancesz', 'scale')
    # Per-instance colour channels. TD spells these differently across builds.
    _set_par(nodes_geo, 'instancecolorr', 'r', 'instancecr', 'instr')
    _set_par(nodes_geo, 'instancecolorg', 'g', 'instancecg', 'instg')
    _set_par(nodes_geo, 'instancecolorb', 'b', 'instancecb', 'instb')
    _set_par(nodes_geo, 'instancecolormode', 'rgb',
             'instancecolormethod', 'instcolormode')
    _set_par(nodes_geo, 'material', node_mat.path)
    _set_par(nodes_geo, 'render',  True)
    _set_par(nodes_geo, 'display', True)

    # ------------------------------------------------------------------ #
    # 4. Edges geometry — Script SOP builds live polylines                #
    # ------------------------------------------------------------------ #
    edges_geo = _make(BASE, geometryCOMP, 'edges_geo', x=1000, y=400)
    default_sop = edges_geo.op('torus1')
    if default_sop is not None:
        default_sop.destroy()
    edge_sop_script_dat = _make(BASE, textDAT, 'edges_sop_script', x=600, y=300)
    edge_sop_script_dat.text = _EDGES_SOP_CODE
    edge_lines = edges_geo.create(scriptSOP, 'edge_lines')
    _set_par(edge_lines, 'callbacks', edge_sop_script_dat.path, 'dat', 'callbackdat')
    edge_lines.render  = True
    edge_lines.display = True
    _set_par(edges_geo, 'material', edge_mat.path)
    _set_par(edges_geo, 'render',  True)
    _set_par(edges_geo, 'display', True)

    # ------------------------------------------------------------------ #
    # 5. Floor grid (HUD perspective grid)                                #
    # ------------------------------------------------------------------ #
    grid_geo = _make(BASE, geometryCOMP, 'grid_geo', x=1000, y=300)
    default_sop = grid_geo.op('torus1')
    if default_sop is not None:
        default_sop.destroy()
    grid_sop = grid_geo.create(gridSOP, 'grid_sop')
    _set_par(grid_sop, 'sizex', 30.0)
    _set_par(grid_sop, 'sizey', 30.0)
    _set_par(grid_sop, 'rows', 31)
    _set_par(grid_sop, 'cols', 31)
    _set_par(grid_sop, 'orient', 0, 'orientation')   # XY plane
    grid_sop.render  = True
    grid_sop.display = True
    # Drop the grid below the globe and orient as floor (rotate 90° around X).
    _set_par(grid_geo, 'ty', -4.5)
    _set_par(grid_geo, 'rx', 90.0)
    _set_par(grid_geo, 'material', grid_mat.path)
    _set_par(grid_geo, 'render',  True)
    _set_par(grid_geo, 'display', True)

    # ------------------------------------------------------------------ #
    # 6. Camera + light                                                   #
    # ------------------------------------------------------------------ #
    # Null COMP at origin gives the camera a stable look-at target so we don't
    # have to compute Euler angles by hand from spherical coordinates.
    cam_target = _make(BASE, nullCOMP, 'cam_target', x=1200, y=600)
    _set_par(cam_target, 'tx', 0.0)
    _set_par(cam_target, 'ty', 0.0)
    _set_par(cam_target, 'tz', 0.0)

    cam = _make(BASE, cameraCOMP, 'cam1', x=1200, y=500)
    _set_par(cam, 'tz', 12.0)
    _set_par(cam, 'fov',  45.0, 'fovx', 'angle')
    # Point at the origin regardless of where we place tx/ty/tz.
    _set_par(cam, 'lookat', cam_target.path)

    light = _make(BASE, lightCOMP, 'light1', x=1200, y=400)
    _set_par(light, 'tx', 6.0)
    _set_par(light, 'ty', 6.0)
    _set_par(light, 'tz', 8.0)

    # ------------------------------------------------------------------ #
    # 7. Orbit/dolly/idle-spin camera controller                          #
    # ------------------------------------------------------------------ #
    _make(BASE, mouseinCHOP, 'cam_mouse', x=1000, y=600)

    cam_ctl_dat = _make(BASE, textDAT, 'cam_ctl_script', x=1100, y=600)
    cam_ctl_dat.text = _CAM_CTL_CODE

    cam_exec = _make(BASE, executeDAT, 'cam_exec', x=1300, y=600)
    cam_exec.text = _CAM_EXEC_CODE
    _set_par(cam_exec, 'framestart', True, 'onframestart')

    # ------------------------------------------------------------------ #
    # 8. Render + bloom + vignette                                        #
    # ------------------------------------------------------------------ #
    render3d = _make(BASE, renderTOP, 'render3d', x=1400, y=500)
    _set_par(render3d, 'camera',   cam.path)
    _set_par(render3d, 'lights',   light.path)
    _set_par(render3d, 'geometry', f'{nodes_geo.path} {edges_geo.path} {grid_geo.path}')
    _set_par(render3d, 'resolutionw', 1280)
    _set_par(render3d, 'resolutionh', 720)
    # Deep space bg colour (very dark navy).
    _set_par(render3d, 'bgcolorr', 0.005)
    _set_par(render3d, 'bgcolorg', 0.015)
    _set_par(render3d, 'bgcolorb', 0.040)
    _set_par(render3d, 'bgalpha', 1.0, 'bga', 'bgcoloralpha', 'bgcolora')

    glow_blur = _make(BASE, blurTOP, 'glow_blur', x=1500, y=500)
    render3d.outputConnectors[0].connect(glow_blur.inputConnectors[0])
    # Modest blur — large kernels wipe sparse cyan dots into invisibility.
    _set_par(glow_blur, 'size', 6.0, 'blursize')

    glow_add = _make(BASE, addTOP, 'glow_add', x=1600, y=500)
    render3d.outputConnectors[0].connect(glow_add.inputConnectors[0])
    glow_blur.outputConnectors[0].connect(glow_add.inputConnectors[1])

    # Vignette deferred — its default ramp can zero everything out. Skip the
    # multiply for now and pipe the bloomed render straight to the output.
    out3d = _make(BASE, outTOP, 'out3d', x=1800, y=500)
    glow_add.outputConnectors[0].connect(out3d.inputConnectors[0])

    # Verify render flags actually took — silent par failures here would
    # produce a completely black render even with all wiring otherwise right.
    for g in (nodes_geo, edges_geo, grid_geo):
        r = g.par.render.eval() if hasattr(g.par, 'render') else 'NA'
        d = g.par.display.eval() if hasattr(g.par, 'display') else 'NA'
        print(f'[setup_3d] {g.name}: render={r} display={d}')

    print('[setup_3d] JARVIS HUD ready. View /project1/out3d')
    print('[setup_3d] Drag in cam_mouse to orbit, scroll/pinch to dolly.')
    print('[setup_3d] Idle for ~2s and the globe will auto-rotate.')


# ---------------------------------------------------------------------------
# Embedded Script SOP — live edge polylines
# ---------------------------------------------------------------------------
_EDGES_SOP_CODE = '''"""edges_sop — rebuilds polylines for every graph edge each cook."""
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

def _get_graph():
    store = op('/project1/graph_store')
    if store is None:
        return None
    try:
        return mod(store).graph
    except Exception:
        return None

def onCook(scriptOp):
    scriptOp.clear()
    graph = _get_graph()
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
# Camera control DAT — module-level state
# ---------------------------------------------------------------------------
_CAM_CTL_CODE = '''"""cam_ctl_script — orbit/dolly/idle-spin state for cam1."""
azimuth   = 25.0
elevation = 12.0
distance  = 12.0

_dragging   = False
_last_mx    = 0.0
_last_my    = 0.0
_idle_frames = 0          # frames since last drag input

ORBIT_SENS   = 220.0      # deg per normalised screen-unit
DOLLY_SENS   = 1.2        # units per scroll tick
MIN_DIST     = 1.5
MAX_DIST     = 28.0
IDLE_FRAMES_TO_SPIN = 120   # ~2 s at 60 fps
IDLE_SPIN_DEG_PER_FRAME = 0.18
'''


_CAM_EXEC_CODE = '''"""cam_exec — onFrameStart drives the orbit camera."""
import math

def onFrameStart(frame):
    mouse = op('cam_mouse')
    ctl_op = op('cam_ctl_script')
    cam    = op('cam1')
    if mouse is None or ctl_op is None or cam is None:
        return
    ctl = mod(ctl_op)

    chan_names = [c.name for c in mouse.chans()]
    def _chan(name, default=0.0):
        return float(mouse[name]) if name in chan_names else default

    mx = _chan('tx')
    my = _chan('ty')
    lb = _chan('lselect')
    mw = _chan('mw')

    drag_now = lb > 0.5

    # Orbit on left-drag
    if drag_now:
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
        ctl._idle_frames = 0
    else:
        ctl._dragging = False
        ctl._idle_frames += 1

    # Dolly on scroll/pinch
    if abs(mw) > 1e-4:
        ctl.distance -= mw * ctl.DOLLY_SENS
        ctl.distance = max(ctl.MIN_DIST, min(ctl.MAX_DIST, ctl.distance))
        ctl._idle_frames = 0

    # Idle auto-spin
    if ctl._idle_frames > ctl.IDLE_FRAMES_TO_SPIN:
        ctl.azimuth += ctl.IDLE_SPIN_DEG_PER_FRAME

    # Spherical -> camera position. Rotation handled by the lookat target.
    az = math.radians(ctl.azimuth)
    el = math.radians(ctl.elevation)
    d  = ctl.distance
    cam.par.tx = d * math.cos(el) * math.sin(az)
    cam.par.ty = d * math.sin(el)
    cam.par.tz = d * math.cos(el) * math.cos(az)
'''
