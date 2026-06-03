"""
interactions.py — JARVIS-style interaction layer over the 3D HUD.

This module is imported from TD callbacks (cam_exec, gesture wiring) and from
the Textport for ad-hoc queries.  It owns the *selection* notion: a Node is
"selected" when its ``selected`` flag is True on the live GraphState held by
``/project1/graph_store``.  positions_chop.py renders selected nodes brighter
and bigger, so any function in here that flips that flag is automatically
reflected in the visualization on the next cook.

Three entry points
==================
    select_node(node_id)               manual single-node select
    select_nodes(ids, exclusive=True)  multi-select, used by query results
    pick_nearest_in_screen(mx, my)     world->screen projection, used by
                                       mouse-click and webcam-pinch
    ask(query)                         RAG chat + highlight returned nodes
    clear_selection()                  reset

World-space mapping must stay in lock-step with positions_chop.py — see
the constants at the top of both files.
"""
from __future__ import annotations

import sys as _sys

_root = r'/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought'
_venv = r'/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought/.venv_td/lib/python3.11/site-packages'
for _p in (_root, _venv):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

# World-space mapping (must match positions_chop.py / edges_chop.py).
PHYS_CENTER  = (960.0, 540.0, 0.0)
PHYS_RADIUS  = 480.0
SCENE_RADIUS = 4.0


# ---------------------------------------------------------------------------
# TD-global accessors. `op` / `mod` are auto-injected into DAT scripts but NOT
# into modules imported via `from touchdesigner import interactions`, so we
# resolve them lazily from the `td` module.
# ---------------------------------------------------------------------------
def _td_op(path):
    import td as _td
    return _td.op(path)


def _td_mod(arg):
    import td as _td
    return _td.mod(arg)


# ---------------------------------------------------------------------------
# Graph access
# ---------------------------------------------------------------------------
def _get_graph():
    store = _td_op('/project1/graph_store')
    if store is None:
        return None
    try:
        return _td_mod(store).graph
    except Exception:
        return None


def _world_pos(node):
    s = SCENE_RADIUS / PHYS_RADIUS
    return (
        (node.x - PHYS_CENTER[0]) * s,
        (node.y - PHYS_CENTER[1]) * s,
        (node.z - PHYS_CENTER[2]) * s,
    )


# ---------------------------------------------------------------------------
# Selection primitives
# ---------------------------------------------------------------------------
def clear_selection(graph=None) -> None:
    """Mark every node un-selected."""
    g = graph or _get_graph()
    if g is None:
        return
    for n in g.nodes:
        n.selected = False


def select_node(node_id: str, exclusive: bool = True, graph=None) -> None:
    """Select one node by id. If exclusive, clears the rest first."""
    g = graph or _get_graph()
    if g is None:
        return
    if exclusive:
        clear_selection(g)
    n = g.get_node(node_id)
    if n is not None:
        n.selected = True


def select_nodes(node_ids, exclusive: bool = True, graph=None) -> int:
    """Bulk select. Returns how many nodes actually existed in the graph."""
    g = graph or _get_graph()
    if g is None:
        return 0
    if exclusive:
        clear_selection(g)
    hit = 0
    for nid in node_ids:
        n = g.get_node(nid)
        if n is not None:
            n.selected = True
            hit += 1
    return hit


# ---------------------------------------------------------------------------
# Screen-space picking (mouse click, hand pinch)
# ---------------------------------------------------------------------------
def pick_nearest_in_screen(
    mx_norm: float,
    my_norm: float,
    cam_path: str = '/project1/cam1',
    render_path: str = '/project1/render3d',
    pixel_threshold: float = 90.0,
) -> str | None:
    """Project every node to screen space and return the nearest within
    ``pixel_threshold`` of (mx_norm, my_norm).

    Args:
        mx_norm: Mouse X in 0..1 (left to right).
        my_norm: Mouse Y in 0..1 (bottom to top — Mouse In CHOP default).
        pixel_threshold: Max screen-pixel distance to accept a hit. None disables.

    Returns the id of the picked node (and marks it selected), or None.
    """
    g = _get_graph()
    cam = _td_op(cam_path)
    rt  = _td_op(render_path)
    if g is None or cam is None or rt is None:
        return None

    rw = float(rt.width)
    rh = float(rt.height)
    target_x = mx_norm * rw
    # In TD's Mouse In CHOP, ty=0 is BOTTOM. Render pixel y=0 is TOP.
    target_y = (1.0 - my_norm) * rh

    best_id = None
    best_d2 = float('inf')

    for n in g.nodes:
        wx, wy, wz = _world_pos(n)
        sx, sy, sz = _world_to_pixel(cam, wx, wy, wz, rw, rh)
        if sz is None or sz <= 0.0:
            continue  # behind the camera
        dx = sx - target_x
        dy = sy - target_y
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_id = n.id

    if best_id is None:
        return None
    if pixel_threshold is not None and best_d2 > pixel_threshold * pixel_threshold:
        return None
    select_node(best_id, exclusive=True)
    return best_id


def _world_to_pixel(cam, wx, wy, wz, rw, rh):
    """Project a world point to (px, py, depth). depth>0 means in front of cam.

    Uses TD's cameraCOMP.worldToScreen if available, else falls back to a
    manual transform via the camera's worldTransform and projection matrix.
    """
    # Preferred: TD's built-in projection.
    try:
        import td as _td
        wts = getattr(cam, 'worldToScreen', None)
        if callable(wts):
            v = wts(_td.tdu.Vector(wx, wy, wz))
            # Returns NDC-ish coords: x,y in ~[-1, 1], z is depth.
            sx = (float(v[0]) * 0.5 + 0.5) * rw
            sy = (1.0 - (float(v[1]) * 0.5 + 0.5)) * rh
            sz = float(v[2]) if len(v) > 2 else 1.0
            return sx, sy, sz
    except Exception:
        pass

    # Fallback: use cam.worldTransform and projection matrix.
    try:
        import td as _td
        world = _td.tdu.Vector(wx, wy, wz)
        wt = cam.worldTransform
        view = wt.copy()
        view.invert()
        cam_pt = view * world
        if cam_pt.z >= 0:
            # In TD, camera looks down -Z; positive Z is behind.
            return 0.0, 0.0, -cam_pt.z
        # Crude perspective using FOV.
        fov = float(cam.par.fov) if hasattr(cam.par, 'fov') else 45.0
        import math
        f = 1.0 / math.tan(math.radians(fov) * 0.5)
        aspect = rw / rh
        sx_ndc = (cam_pt.x / -cam_pt.z) * (f / aspect)
        sy_ndc = (cam_pt.y / -cam_pt.z) * f
        sx = (sx_ndc * 0.5 + 0.5) * rw
        sy = (1.0 - (sy_ndc * 0.5 + 0.5)) * rh
        return sx, sy, -cam_pt.z
    except Exception:
        return 0.0, 0.0, None


# ---------------------------------------------------------------------------
# Hand-gesture camera controller
# ---------------------------------------------------------------------------
# State is held on the cam_ctl_script Text DAT module so it persists across
# frames AND so both mouse + hand paths share the same camera target.
#
# Two-hand pinch ("Iron Man pinch-spread"):
#     Both hands in PINCHING state. Distance between the two pinch midpoints
#     drives camera distance (dolly). Spread apart → zoom in.
#
# Single-hand pinch + drag:
#     One hand in PINCHING state. Hand X/Y delta drives orbit (azimuth/elev).
#
# Pinch-tap on a node (short pinch, no movement):
#     On pinch_end with tiny travel, pick the node at the pinch screen position.
# ---------------------------------------------------------------------------
HAND_ORBIT_SENS  = 240.0     # deg per normalised hand-screen-unit
HAND_ZOOM_GAIN   = 18.0      # zoom factor for Δ pinch-pair distance
PINCH_TAP_MOVE   = 0.04      # below this normalised travel = a tap
PINCH_TAP_TIME   = 0.5       # seconds — anything longer than this isn't a tap


def _ctl():
    """Return the cam_ctl_script module (same one the mouse exec writes to)."""
    s = _td_op('/project1/cam_ctl_script')
    return _td_mod(s) if s is not None else None


def _ensure_hand_state(ctl):
    """Lazily initialise our hand-state attributes on the camera control module."""
    if not hasattr(ctl, '_h_two_dist'):
        ctl._h_two_dist  = None         # last frame's inter-pinch distance
        ctl._h_pinch0_xy = None         # last frame midpoint of hand 0
        ctl._h_pinch1_xy = None
        ctl._h_pinch0_start    = None   # (x, y, t) at pinch_start for hand 0
        ctl._h_pinch1_start    = None
        ctl._h_pinch0_active   = False
        ctl._h_pinch1_active   = False


def handle_gesture_events(events, latest_hands):
    """Drive the orbit camera from a list of GestureEvent + current hands.

    Args:
        events: List[GestureEvent] produced by GestureEngine.update().
        latest_hands: The current frame's hands list — used to find live
            midpoints for two-hand zoom even when no event was emitted.
    """
    ctl = _ctl()
    if ctl is None:
        return
    _ensure_hand_state(ctl)

    # --- Process discrete events (pinch_start / pinch_end → mode switches) ---
    import time as _time
    for ev in events:
        if ev.name == 'pinch_start':
            if ev.hand_index == 0:
                ctl._h_pinch0_active = True
                ctl._h_pinch0_start  = (ev.position[0], ev.position[1], _time.time())
                ctl._h_pinch0_xy     = ev.position
            elif ev.hand_index == 1:
                ctl._h_pinch1_active = True
                ctl._h_pinch1_start  = (ev.position[0], ev.position[1], _time.time())
                ctl._h_pinch1_xy     = ev.position

        elif ev.name == 'pinch_end':
            if ev.hand_index == 0 and ctl._h_pinch0_start is not None:
                _maybe_pinch_tap(ctl._h_pinch0_start, ev.position)
                ctl._h_pinch0_active = False
                ctl._h_pinch0_start  = None
                ctl._h_pinch0_xy     = None
            elif ev.hand_index == 1 and ctl._h_pinch1_start is not None:
                _maybe_pinch_tap(ctl._h_pinch1_start, ev.position)
                ctl._h_pinch1_active = False
                ctl._h_pinch1_start  = None
                ctl._h_pinch1_xy     = None

    # --- Continuous update — pull current midpoints while pinching ---
    # Refresh midpoints from latest_hands if pinches are still active.
    if ctl._h_pinch0_active and len(latest_hands) >= 1:
        ctl._h_pinch0_xy = _pinch_midpoint(latest_hands[0])
    if ctl._h_pinch1_active and len(latest_hands) >= 2:
        ctl._h_pinch1_xy = _pinch_midpoint(latest_hands[1])

    two_hand = ctl._h_pinch0_active and ctl._h_pinch1_active and \
               ctl._h_pinch0_xy is not None and ctl._h_pinch1_xy is not None

    if two_hand:
        # --- Iron Man pinch-spread zoom ---
        dx = ctl._h_pinch1_xy[0] - ctl._h_pinch0_xy[0]
        dy = ctl._h_pinch1_xy[1] - ctl._h_pinch0_xy[1]
        d  = (dx * dx + dy * dy) ** 0.5
        if ctl._h_two_dist is not None:
            delta = d - ctl._h_two_dist
            ctl.distance -= delta * HAND_ZOOM_GAIN
            ctl.distance = max(ctl.MIN_DIST, min(ctl.MAX_DIST, ctl.distance))
            ctl._idle_frames = 0
        ctl._h_two_dist = d
        # While zooming we DON'T also orbit — feels more controllable.
        return

    # Not in two-hand zoom: reset that pair tracker.
    ctl._h_two_dist = None

    # --- Single-hand pinch-drag → orbit ---
    if ctl._h_pinch0_active and ctl._h_pinch0_xy is not None and ctl._h_pinch0_start is not None:
        cur_x, cur_y = ctl._h_pinch0_xy
        # Use frame-to-frame delta so motion feels velocity-like, not absolute.
        # Stash previous position in _h_pinch0_prev for delta calc.
        prev = getattr(ctl, '_h_pinch0_prev', None)
        if prev is not None:
            dxh = cur_x - prev[0]
            dyh = cur_y - prev[1]
            ctl.azimuth   += dxh * HAND_ORBIT_SENS
            ctl.elevation -= dyh * HAND_ORBIT_SENS    # y inverted for natural feel
            ctl.elevation = max(-85.0, min(85.0, ctl.elevation))
            ctl._idle_frames = 0
        ctl._h_pinch0_prev = (cur_x, cur_y)
    else:
        # No active pinch — clear delta tracker so next pinch_start starts fresh.
        if hasattr(ctl, '_h_pinch0_prev'):
            ctl._h_pinch0_prev = None


def _pinch_midpoint(hand):
    """Return (x,y) midpoint of thumb tip + index tip — same as gesture_engine."""
    try:
        lm = hand['landmarks']
        tx, ty, _ = lm[4]   # THUMB_TIP
        ix, iy, _ = lm[8]   # INDEX_TIP
        return ((tx + ix) / 2.0, (ty + iy) / 2.0)
    except (KeyError, IndexError):
        return None


def _maybe_pinch_tap(start, end_pos):
    """If pinch covered tiny distance in short time → treat as a click pick."""
    import time as _time
    sx, sy, st = start
    ex, ey = end_pos
    travel = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
    dt = _time.time() - st
    if travel < PINCH_TAP_MOVE and dt < PINCH_TAP_TIME:
        # Pinch position is in normalised screen [0, 1]; same coord space as
        # the mouse pick path.
        try:
            hit = pick_nearest_in_screen(ex, ey)
            if hit:
                g = _get_graph()
                n = g.get_node(hit) if g else None
                label = n.label if n else hit[:8]
                print(f'[hand] PINCHED: "{label}"  (id={hit[:8]})')
        except Exception as e:
            print(f'[hand] pinch-pick failed: {e}')


# ---------------------------------------------------------------------------
# RAG: text query → light the constellation
# ---------------------------------------------------------------------------
def ask(query: str, k_chunks: int = 6, k_nodes: int = 8):
    """Run the RAG chat against the indexed corpus, then highlight the
    matching graph nodes in the HUD.

    Usage from the Textport:
        from touchdesigner import interactions
        interactions.ask('What is multi-head attention?')
    """
    g = _get_graph()
    if g is None:
        print('[interactions] graph not loaded — has td_auto_setup finished?')
        return None
    try:
        from core.retrieval import Indexer
        from core.chat import RagChat
    except ImportError as exc:
        print(f'[interactions] RAG deps missing: {exc}')
        print('  Install with:  pip install chromadb')
        return None

    indexer = Indexer.default(session='default')
    if indexer.count() == 0:
        print('[interactions] vector store is empty.')
        print('  Index PDFs first via:  python -m core.chat.cli --session default')
        return None

    chat = RagChat(indexer=indexer, graph=g)
    ans = chat.ask(query, k_chunks=k_chunks, k_nodes=k_nodes)

    n_lit = select_nodes(ans.node_ids_lit, exclusive=True)

    print('─' * 60)
    print(ans.answer)
    print('─' * 60)
    if ans.citations:
        print('Sources:')
        for c in ans.citations:
            where = c.source_paper or c.source_id
            pages = f' p.{",".join(str(p) for p in c.page_refs)}' if c.page_refs else ''
            print(f'  [{c.marker}] {where}{pages}  (score {c.score:.2f})')
    print(f'Highlighted {n_lit} nodes in the HUD.')
    return ans
