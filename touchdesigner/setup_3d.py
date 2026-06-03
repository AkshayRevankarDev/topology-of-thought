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
    _set_par(edge_mat, 'colorr', 0.30)
    _set_par(edge_mat, 'colorg', 0.85)
    _set_par(edge_mat, 'colorb', 1.00)
    _set_par(edge_mat, 'alpha', 0.95)

    grid_mat = _make(BASE, constantMAT, 'grid_mat', x=1000, y=700)
    _set_par(grid_mat, 'colorr', 0.12)
    _set_par(grid_mat, 'colorg', 0.55)
    _set_par(grid_mat, 'colorb', 0.95)
    _set_par(grid_mat, 'alpha', 0.55)

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
    # Per-instance colour: this TD build uses a single CHOP reference
    # (instancecolorop) + a mode menu, not per-channel pars. The geo reads
    # the first 3 (or 4) channels matching r/g/b/a from that CHOP.
    _set_par(nodes_geo, 'instancecolorop', node_pos.path)
    _set_par(nodes_geo, 'instancecolormode', 'replace')
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

    # Grid SOP only outputs filled surfaces (poly/mesh/nurbs/bezier), so we build
    # the wireframe directly via a Script SOP — clean rows + cols of polylines.
    grid_sop_dat = _make(BASE, textDAT, 'grid_sop_script', x=600, y=100)
    grid_sop_dat.text = _GRID_SOP_CODE
    grid_sop = grid_geo.create(scriptSOP, 'grid_sop')
    _set_par(grid_sop, 'callbacks', grid_sop_dat.path, 'dat', 'callbackdat')
    grid_sop.render  = True
    grid_sop.display = True
    # Drop the grid further below so the globe floats clearly above it.
    _set_par(grid_geo, 'ty', -5.5)
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
    _set_par(cam, 'tz', 18.0)
    _set_par(cam, 'fov',  50.0, 'fovx', 'angle')
    # Point at the origin regardless of where we place tx/ty/tz.
    _set_par(cam, 'lookat', cam_target.path)
    # Tight near plane so the user can dolly *inside* the globe (around-me view)
    # without geometry getting clipped at the camera.
    _set_par(cam, 'near', 0.05, 'nearclip')

    light = _make(BASE, lightCOMP, 'light1', x=1200, y=400)
    _set_par(light, 'tx', 6.0)
    _set_par(light, 'ty', 6.0)
    _set_par(light, 'tz', 8.0)

    # ------------------------------------------------------------------ #
    # 7. Orbit/dolly/idle-spin camera controller                          #
    # ------------------------------------------------------------------ #
    mouse_chop = _make(BASE, mouseinCHOP, 'cam_mouse', x=1000, y=600)
    # Mouse In CHOP exposes button/wheel channels ONLY when these par fields
    # have a name in them. Default is empty = no channel.
    _set_par(mouse_chop, 'lbuttonname', 'lselect')
    _set_par(mouse_chop, 'rbuttonname', 'rselect')
    _set_par(mouse_chop, 'mbuttonname', 'mselect')
    _set_par(mouse_chop, 'wheel',       'mw')

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
    # Stronger bloom now that the spheres are bright and dense.
    _set_par(glow_blur, 'size', 14.0, 'blursize')

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

    # ------------------------------------------------------------------ #
    # 8b. Webcam preview Script TOP (with landmark dots)                  #
    # ------------------------------------------------------------------ #
    cam_preview_dat = _make(BASE, textDAT, 'cam_preview_script', x=600, y=0)
    cam_preview_dat.text = _CAM_PREVIEW_CODE
    scriptTOP = td.scriptTOP
    noiseTOP  = td.noiseTOP
    # Trigger so the Script TOP re-cooks every frame (same trick as graph_render).
    preview_trigger = _make(BASE, noiseTOP, 'cam_preview_trigger', x=800, y=-50)
    _set_par(preview_trigger, 'resolutionw', 16)
    _set_par(preview_trigger, 'resolutionh', 16)
    cam_preview_top = _make(BASE, scriptTOP, 'cam_preview', x=900, y=0)
    _set_par(cam_preview_top, 'callbacks', cam_preview_dat.path, 'dat', 'callbackdat')
    _set_par(cam_preview_top, 'resolutionw', 640)
    _set_par(cam_preview_top, 'resolutionh', 480)
    preview_trigger.outputConnectors[0].connect(cam_preview_top.inputConnectors[0])

    # ------------------------------------------------------------------ #
    # 9. Re-point gesture_exec at our 3D camera handler                  #
    # ------------------------------------------------------------------ #
    gesture_exec = BASE.op('gesture_exec')
    if gesture_exec is not None:
        gesture_exec.text = _GESTURE_EXEC_CODE
        print('[setup_3d] gesture_exec rewired to drive 3D camera')
    else:
        print('[setup_3d] NOTE: gesture_exec not found; run td_auto_setup first '
              'if you want hand-gesture camera control.')

    print('[setup_3d] JARVIS HUD ready. View /project1/out3d')
    print('[setup_3d] Drag in cam_mouse to orbit, scroll/pinch to dolly.')
    print('[setup_3d] Hand gestures: single pinch+drag=orbit, two-hand pinch=zoom.')
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
# Embedded Script SOP — wireframe perspective grid
# ---------------------------------------------------------------------------
_GRID_SOP_CODE = '''"""grid_sop — builds an XY-plane wireframe grid as polylines.

Output is N+1 horizontal lines and N+1 vertical lines covering a SIZE x SIZE
patch centred on the origin. The Geometry COMP rotates this onto the XZ plane
so it reads as a floor; here we just emit it flat in XY.
"""
SIZE  = 30.0    # total extent of the grid (TD units)
LINES = 31      # number of lines in each direction (=> LINES-1 cells)

def onCook(scriptOp):
    scriptOp.clear()
    half = SIZE / 2.0
    step = SIZE / max(LINES - 1, 1)
    # Horizontal lines (vary y, x sweeps).
    for i in range(LINES):
        y = -half + i * step
        p1 = scriptOp.appendPoint()
        p2 = scriptOp.appendPoint()
        p1.x, p1.y, p1.z = -half, y, 0.0
        p2.x, p2.y, p2.z =  half, y, 0.0
        poly = scriptOp.appendPoly(2, closed=False, addPoints=False)
        poly[0].point = p1
        poly[1].point = p2
    # Vertical lines (vary x, y sweeps).
    for i in range(LINES):
        x = -half + i * step
        p1 = scriptOp.appendPoint()
        p2 = scriptOp.appendPoint()
        p1.x, p1.y, p1.z = x, -half, 0.0
        p2.x, p2.y, p2.z = x,  half, 0.0
        poly = scriptOp.appendPoly(2, closed=False, addPoints=False)
        poly[0].point = p1
        poly[1].point = p2
'''


# ---------------------------------------------------------------------------
# Webcam preview Script TOP — shows live feed + landmark overlay
# ---------------------------------------------------------------------------
_CAM_PREVIEW_CODE = '''"""cam_preview — displays the latest webcam frame + hand landmarks."""
import sys as _sys
_root = r\'''' + str(PROJECT_ROOT) + '''\'
_venv = r\'''' + str(PROJECT_ROOT / '.venv_td/lib/python3.11/site-packages') + '''\'
for _p in (_root, _venv):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import numpy as np

# Skeleton connectivity for MediaPipe hand landmarks (21 points).
_BONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def cook(scriptOp):
    try:
        from touchdesigner import hand_tracking as ht
        import cv2
    except Exception:
        # Module not ready — emit a black frame so the TOP still exists.
        scriptOp.copyNumpyArray(np.zeros((scriptOp.height, scriptOp.width, 4), dtype=np.float32))
        return

    rgb = ht._latest_frame
    if rgb is None:
        scriptOp.copyNumpyArray(np.zeros((scriptOp.height, scriptOp.width, 4), dtype=np.float32))
        return

    img = rgb.copy()
    h, w, _ = img.shape

    # Overlay landmarks for each detected hand.
    for hand in ht._latest_hands or []:
        lms = hand.get('landmarks', [])
        if not lms:
            continue
        pts = [(int(x * w), int(y * h)) for (x, y, _z) in lms]
        for a, b in _BONES:
            if a < len(pts) and b < len(pts):
                cv2.line(img, pts[a], pts[b], (60, 230, 255), 2)
        for (px, py) in pts:
            cv2.circle(img, (px, py), 4, (255, 255, 255), -1)
        # Highlight the pinch midpoint (thumb tip + index tip).
        if len(pts) >= 9:
            mx = (pts[4][0] + pts[8][0]) // 2
            my = (pts[4][1] + pts[8][1]) // 2
            cv2.circle(img, (mx, my), 10, (90, 255, 180), 2)

    # Mirror horizontally so user sees a mirror view (Zoom-style).
    img = np.fliplr(img)

    # Pack into RGBA float32 for TD.
    rgba = np.dstack([img, np.full((h, w, 1), 255, dtype=np.uint8)])
    out = (rgba.astype(np.float32) / 255.0)
    # TD wants row 0 at bottom; numpy has row 0 at top.
    out = np.flipud(out)
    scriptOp.copyNumpyArray(np.ascontiguousarray(out))
'''


# ---------------------------------------------------------------------------
# Gesture Execute DAT — feeds MediaPipe hand events into 3D camera handler
# ---------------------------------------------------------------------------
_GESTURE_EXEC_CODE = '''"""gesture_exec — 3D camera control via webcam hand pinch (rewired)."""
import sys as _sys
_root = r\'''' + str(PROJECT_ROOT) + '''\'
_venv = r\'''' + str(PROJECT_ROOT / '.venv_td/lib/python3.11/site-packages') + '''\'
for _p in (_root, _venv):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

_engine = None
_import_ok = None

_engine     = None
_import_ok  = None

def _try_import():
    global _import_ok, _engine
    try:
        from touchdesigner import hand_tracking as ht
        from touchdesigner.gesture_engine import GestureEngine
        # Source-of-truth = the module's own _cap. If it's None (fresh import,
        # or hand_tracking was reloaded), restart the camera. Looser pinch
        # threshold so casual pinches register.
        if ht._cap is None:
            ht.startup()
            print('[gesture_exec] webcam (re)opened.')
        if _engine is None:
            _engine = GestureEngine(close_threshold=0.08, open_threshold=0.12)
            print('[gesture_exec] GestureEngine ready (close=0.08).')
        _import_ok = True
    except Exception as e:
        _import_ok = False
        global _err_print
        try:
            _err_print
        except NameError:
            _err_print = 0
        import time
        now = time.time()
        if now - _err_print > 3.0:
            print(f'[gesture_exec] WARN: hand pipeline init failed: {e}')
            _err_print = now
    return _import_ok

def onStart():
    _try_import()

def onFrameStart(frame):
    if not _try_import():
        return
    # Drive the MediaPipe inference inline so we don't depend on the
    # hand_tracker Script CHOP cooking every frame.
    from touchdesigner import hand_tracking as ht
    rgb = ht.read_frame()
    if rgb is not None:
        ht.process_frame(rgb)
    hands  = list(ht._latest_hands)
    events = _engine.update(hands)
    if not events and not hands:
        return
    try:
        from touchdesigner import interactions
        interactions.handle_gesture_events(events, hands)
    except Exception as e:
        global _last_err
        try:
            _last_err
        except NameError:
            _last_err = 0
        import time
        now = time.time()
        if now - _last_err > 2.0:
            print(f'[gesture_exec] handler error: {e}')
            _last_err = now
'''


# ---------------------------------------------------------------------------
# Camera control DAT — module-level state
# ---------------------------------------------------------------------------
_CAM_CTL_CODE = '''"""cam_ctl_script — orbit/dolly/idle-spin state for cam1."""
azimuth   = 25.0
elevation = 5.0
distance  = 18.0

_dragging   = False
_last_mx    = 0.0
_last_my    = 0.0
_drag_dist  = 0.0         # accumulated drag distance — used to distinguish click vs drag
_lb_prev    = 0.0         # previous frame's left-button state (rising-edge)
_idle_frames = 0          # frames since last drag input

ORBIT_SENS   = 220.0      # deg per normalised screen-unit
DOLLY_SENS   = 1.4        # units per scroll tick
MIN_DIST     = 0.4        # can dolly inside the globe (radius ~4) for around-me view
MAX_DIST     = 35.0
IDLE_FRAMES_TO_SPIN = 120   # ~2 s at 60 fps
IDLE_SPIN_DEG_PER_FRAME = 0.15
CLICK_DRAG_THRESHOLD = 0.012   # normalised drag distance below this counts as click
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
    lb_prev  = getattr(ctl, '_lb_prev', 0.0)

    # ---- Press: remember where the click started ----
    if drag_now and lb_prev <= 0.5:
        ctl._dragging   = True
        ctl._last_mx    = mx
        ctl._last_my    = my
        ctl._drag_dist  = 0.0

    # ---- Hold: orbit accumulates ----
    if drag_now and ctl._dragging:
        dx = mx - ctl._last_mx
        dy = my - ctl._last_my
        ctl._drag_dist += abs(dx) + abs(dy)
        ctl.azimuth   += dx * ctl.ORBIT_SENS
        ctl.elevation += dy * ctl.ORBIT_SENS
        ctl.elevation = max(-85.0, min(85.0, ctl.elevation))
        ctl._last_mx  = mx
        ctl._last_my  = my
        ctl._idle_frames = 0

    # ---- Release: tiny drag total = treat as a click and pick a node ----
    if (not drag_now) and lb_prev > 0.5 and ctl._dragging:
        if ctl._drag_dist < ctl.CLICK_DRAG_THRESHOLD:
            # Mouse In CHOP captures clicks GLOBALLY (textport, network, etc.),
            # so we only honour clicks whose normalised mouse coords are inside
            # [0,1] — i.e., the user is inside an active panel viewer.
            in_panel = 0.0 <= ctl._last_mx <= 1.0 and 0.0 <= ctl._last_my <= 1.0
            if in_panel:
                try:
                    from touchdesigner import interactions
                    hit = interactions.pick_nearest_in_screen(ctl._last_mx, ctl._last_my)
                    if hit:
                        try:
                            n = mod(op('/project1/graph_store')).graph.get_node(hit)
                            label = n.label if n else hit[:8]
                        except Exception:
                            label = hit[:8]
                        print(f'[cam_exec] PICKED: "{label}"  (id={hit[:8]})')
                    # On a miss we DO NOT clear selection — call
                    # interactions.clear_selection() explicitly from Textport.
                except Exception as e:
                    print(f'[cam_exec] pick failed: {e}')
        ctl._dragging = False

    if not drag_now:
        ctl._idle_frames += 1

    ctl._lb_prev = lb

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
