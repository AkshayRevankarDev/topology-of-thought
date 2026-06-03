"""
render_script_live.py — Callbacks DAT for graph_render Script TOP.

Sphere mode (Iron Man HUD):
  - Nodes live on a Fibonacci-uniform 3D sphere; physics perturbs them along
    the surface so connected concepts cluster.
  - Camera auto-rotates slowly around the globe when idle.
  - Pinch interaction:
      • Quick tap on a node (<0.5s release)   → cam zooms toward that node.
      • Hold pinch on a node (>1.5s)          → fire Ollama decomposition in
        a background thread; new sub-nodes snap onto the sphere near parent.
      • Mid-air pinch + drag                  → rotate the globe.
  - Two-handed pinch-spread (if a 2nd hand is present) → uniform cam zoom.

This file is loaded into TouchDesigner via:
    op('/project1/render_script').text = open(r'<path>/render_script_live.py').read()
"""
import sys as _sys
import time as _time
import threading as _threading

_root = r'/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought'
_venv = r'/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought/.venv_td/lib/python3.11/site-packages'
for _p in (_root, _venv):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

# Clear cached renderer so the next cook re-imports from disk during dev.
_sys.modules.pop('touchdesigner.graph_renderer', None)

# ---------------------------------------------------------------------------
# Camera + interaction state (module-level so it persists across cook calls)
# ---------------------------------------------------------------------------
_cam_yaw         = 0.0
_cam_pitch       = 0.15
_cam_zoom        = 1.0
_cam_target      = (960.0, 540.0, 0.0)   # sphere centre, default

# Smoothly-animated targets (eased toward each frame).
_target_zoom     = 1.0
_target_target   = (960.0, 540.0, 0.0)
_target_yaw      = 0.0

# Auto-rotation when user is not interacting.
_AUTO_YAW_SPEED  = 0.0035    # radians per frame (~12°/sec @60fps)

# Pinch state machine
_grabbed_id      = None
_pinch_start_t   = 0.0
_pinch_start_xy  = None
_was_pinching    = False
_decompose_fired = False
_decompose_busy  = False     # background thread guard

# Decomposition results queued from the worker thread (mutated under _lock).
_decompose_lock  = _threading.Lock()
_decompose_pending: list = []   # [(parent_id, child_specs)]


HOLD_DECOMPOSE_S = 1.5
TAP_ZOOM_S       = 0.5
PHYS_W, PHYS_H   = 1920.0, 1080.0
SPHERE_R         = 480.0
SPHERE_CENTER    = (960.0, 540.0, 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_hand_pos():
    """Return (norm_x, norm_y) in [0,1] from the hand_tracker CHOP, or None."""
    try:
        chop = op('/project1/hand_tracker')
        if chop is None or chop.numChans == 0:
            return None
        def _get(names):
            for n in names:
                try:
                    v = chop[n]
                    if v is not None:
                        return float(v[0])
                except Exception:
                    pass
            return None
        x = _get(['index_tip_x', 'index_x', 'wrist_x', 'tx', 'x', 'chan1'])
        y = _get(['index_tip_y', 'index_y', 'wrist_y', 'ty', 'y', 'chan2'])
        if x is not None and y is not None:
            return (x, y)
    except Exception:
        pass
    return None


def _is_pinching():
    try:
        mode_dat = op('/project1/current_mode')
        return (mode_dat is not None and 'pinch' in mode_dat.text.lower())
    except Exception:
        return False


def _project_node_to_screen(node, w, h):
    """Project a 3D node to screen coords using current camera state."""
    import math
    tx, ty, tz = _cam_target
    cy_, sy_ = math.cos(_cam_yaw),   math.sin(_cam_yaw)
    cp_, sp_ = math.cos(_cam_pitch), math.sin(_cam_pitch)
    x = node.x - tx; y = node.y - ty; z = node.z - tz
    xr = cy_ * x + sy_ * z
    zr = -sy_ * x + cy_ * z
    yr = cp_ * y - sp_ * zr
    zr = sp_ * y + cp_ * zr
    cam_dist = 1400.0
    focal    = 1100.0 * _cam_zoom
    ze = zr + cam_dist
    if ze < 1.0:
        ze = 1.0
    px = w / 2.0 + xr * focal / ze
    py = h / 2.0 + yr * focal / ze
    return px, py, ze


def _pick_node_screen(graph, sx, sy, w, h, radius_px=120.0):
    """Return the nearest node (by screen distance) within radius_px, or None."""
    best, best_d = None, radius_px
    for nd in graph.nodes:
        px, py, ze = _project_node_to_screen(nd, w, h)
        # Skip nodes behind the camera (ze tiny → huge projected coords).
        if ze < 10.0:
            continue
        d = ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
        if d < best_d:
            best, best_d = nd, d
    return best


def _ease(current, target, k=0.12):
    return current + (target - current) * k


def _snap_to_sphere(node):
    """Project a node's (x,y,z) back onto the sphere surface around centre."""
    cx, cy, cz = SPHERE_CENTER
    dx, dy, dz = node.x - cx, node.y - cy, node.z - cz
    r = (dx*dx + dy*dy + dz*dz) ** 0.5
    if r < 1e-6:
        # Push outward along +x if degenerate.
        node.x, node.y, node.z = cx + SPHERE_R, cy, cz
        return
    s = SPHERE_R / r
    node.x = cx + dx * s
    node.y = cy + dy * s
    node.z = cz + dz * s


def _decompose_worker(graph, node_id):  # graph passed directly, not via op storage
    """Background thread: call Ollama; on success, queue children for merge."""
    global _decompose_busy
    try:
        from features.zoom_node import expand_node
        children = expand_node(graph, node_id)
        if children:
            with _decompose_lock:
                _decompose_pending.append([c.id for c in children])
            print(f'[decompose] added {len(children)} children to {node_id[:8]}')
    except Exception as exc:
        print(f'[decompose] ERROR: {exc}')
    finally:
        _decompose_busy = False


def _merge_pending_children(graph):
    """Snap any newly-added decomposition children onto the sphere surface."""
    with _decompose_lock:
        if not _decompose_pending:
            return
        batches = _decompose_pending[:]
        _decompose_pending.clear()
    for child_ids in batches:
        for cid in child_ids:
            nd = graph.get_node(cid)
            if nd is None:
                continue
            # expand_node placed children at radial 2D positions around parent;
            # nudge them outward in z slightly so snap_to_sphere distributes
            # them on the surface near the parent.
            if nd.z == 0.0:
                nd.z = 1.0
            _snap_to_sphere(nd)


# ---------------------------------------------------------------------------
# Main cook callback
# ---------------------------------------------------------------------------
def cook(scriptOp):
    """Called by TouchDesigner each frame to render the knowledge graph."""
    global _cam_yaw, _cam_pitch, _cam_zoom, _cam_target
    global _target_yaw, _target_zoom, _target_target
    global _grabbed_id, _pinch_start_t, _pinch_start_xy
    global _was_pinching, _decompose_fired, _decompose_busy

    # Graph lives in the graph_store Text DAT module — set by physics_exec,
    # never stored in op.store() so TD never tries to pickle it on .toe save.
    try:
        g = mod(op('/project1/graph_store')).graph
    except Exception:
        g = None
    if g is None:
        import numpy as np
        scriptOp.copyNumpyArray(
            np.zeros((scriptOp.height, scriptOp.width, 4), dtype=np.float32)
        )
        return

    try:
        import numpy as np
        from touchdesigner.graph_renderer import render_to_rgba

        out_w = max(scriptOp.width,  1280)
        out_h = max(scriptOp.height,  720)

        # Merge any pending decomposition results onto the sphere.
        _merge_pending_children(g)

        # -----------------------------------------------------------------
        # Pinch state machine
        # -----------------------------------------------------------------
        is_pinching = _is_pinching()
        hand        = _get_hand_pos()
        now         = _time.perf_counter()

        # Hand position in screen pixels (for picking + drag delta).
        hand_px = None
        if hand is not None:
            hand_px = (hand[0] * out_w, hand[1] * out_h)

        # --- pinch START ---
        if is_pinching and not _was_pinching:
            _pinch_start_t  = now
            _pinch_start_xy = hand_px
            _decompose_fired = False
            if hand_px is not None:
                picked = _pick_node_screen(g, hand_px[0], hand_px[1], out_w, out_h)
                if picked:
                    _grabbed_id = picked.id
                    picked.selected = True
                else:
                    _grabbed_id = None

        # --- pinch HELD ---
        if is_pinching:
            held_for = now - _pinch_start_t

            # Long-hold on a node → decompose (fire once).
            if (_grabbed_id is not None
                    and not _decompose_fired
                    and not _decompose_busy
                    and held_for >= HOLD_DECOMPOSE_S):
                _decompose_fired = True
                _decompose_busy  = True
                target = g.get_node(_grabbed_id)
                if target is not None and not target.expanded:
                    th = _threading.Thread(
                        target=_decompose_worker,
                        args=(g, _grabbed_id),
                        daemon=True,
                    )
                    th.start()
                    print(f'[decompose] firing for "{target.label}"')

            # Mid-air pinch (no node grabbed) → rotate globe by hand drag.
            if _grabbed_id is None and hand_px is not None and _pinch_start_xy is not None:
                dx = hand_px[0] - _pinch_start_xy[0]
                dy = hand_px[1] - _pinch_start_xy[1]
                _target_yaw   = _cam_yaw   + dx * 0.005
                # Pitch clamped so the globe doesn't flip upside down.
                _target_pitch = float(np.clip(_cam_pitch - dy * 0.005, -1.2, 1.2))
                _cam_pitch    = _ease(_cam_pitch, _target_pitch, 0.18)
                _pinch_start_xy = hand_px   # incremental drag

        # --- pinch RELEASE ---
        if _was_pinching and not is_pinching:
            held_for = now - _pinch_start_t
            # Quick tap on a node → zoom toward it.
            if _grabbed_id is not None and held_for < TAP_ZOOM_S:
                target = g.get_node(_grabbed_id)
                if target is not None:
                    _target_target = (target.x, target.y, target.z)
                    _target_zoom   = min(_target_zoom * 1.6, 4.0)
            # Tap on empty space → zoom back out a step.
            elif _grabbed_id is None and held_for < TAP_ZOOM_S:
                _target_zoom   = max(_target_zoom / 1.6, 1.0)
                if _target_zoom <= 1.01:
                    _target_target = SPHERE_CENTER
            # Clear selection.
            if _grabbed_id is not None:
                nd = g.get_node(_grabbed_id)
                if nd is not None:
                    nd.selected = False
            _grabbed_id = None

        _was_pinching = is_pinching

        # -----------------------------------------------------------------
        # Camera animation (idle auto-rotate + ease toward targets)
        # -----------------------------------------------------------------
        if not is_pinching:
            _target_yaw += _AUTO_YAW_SPEED
        _cam_yaw  = _ease(_cam_yaw,  _target_yaw,  0.12)
        _cam_zoom = _ease(_cam_zoom, _target_zoom, 0.10)
        cx = _ease(_cam_target[0], _target_target[0], 0.10)
        cy = _ease(_cam_target[1], _target_target[1], 0.10)
        cz = _ease(_cam_target[2], _target_target[2], 0.10)
        _cam_target = (cx, cy, cz)

        # -----------------------------------------------------------------
        # Render
        # -----------------------------------------------------------------
        rgba = render_to_rgba(
            g,
            width=out_w, height=out_h,
            ar=True,
            sphere_mode=True,
            cam_yaw=_cam_yaw,
            cam_pitch=_cam_pitch,
            cam_zoom=_cam_zoom,
            cam_target=_cam_target,
        )
        scriptOp.copyNumpyArray(np.ascontiguousarray(rgba, dtype=np.float32))

    except Exception as e:
        print(f'[render] ERROR: {e}')
        import traceback
        traceback.print_exc()
