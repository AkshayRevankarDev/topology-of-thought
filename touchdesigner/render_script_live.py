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
_sys.modules.pop('touchdesigner.graph_renderer', None)

# ---------------------------------------------------------------------------
# Pinch-drag state  (module-level so it persists across cook calls)
# ---------------------------------------------------------------------------
_grabbed_id   = None   # id of the node currently being dragged
_grab_off_x   = 0.0   # offset from hand to node centre (physics coords)
_grab_off_y   = 0.0
_was_pinching = False


def _get_hand_pos():
    """Return (norm_x, norm_y) in [0,1] from the hand_tracker CHOP, or None."""
    try:
        chop = op('/project1/hand_tracker')
        if chop is None or chop.numChans == 0:
            return None
        # Try several common channel-name conventions
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


def cook(scriptOp):
    """Called by TouchDesigner each frame to render the knowledge graph."""
    global _grabbed_id, _grab_off_x, _grab_off_y, _was_pinching

    g = op('/project1').fetch('graph', None)
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

        # ---------------------------------------------------------------
        # Pinch-drag interaction
        # ---------------------------------------------------------------
        PHYS_W, PHYS_H = 1920.0, 1080.0
        try:
            mode_dat    = op('/project1/current_mode')
            is_pinching = (mode_dat is not None and
                           'pinch' in mode_dat.text.lower())
            hand        = _get_hand_pos()

            if hand is not None:
                hx = hand[0] * PHYS_W   # hand in physics space
                hy = hand[1] * PHYS_H

                if is_pinching and not _was_pinching:
                    # Pinch just started — grab nearest node within 180 px
                    best, best_d = None, 180.0
                    for nd in g.nodes:
                        d = ((nd.x - hx)**2 + (nd.y - hy)**2) ** 0.5
                        if d < best_d:
                            best, best_d = nd, d
                    if best:
                        _grabbed_id = best.id
                        _grab_off_x = best.x - hx
                        _grab_off_y = best.y - hy

                if is_pinching and _grabbed_id:
                    # Move the grabbed node
                    for nd in g.nodes:
                        if nd.id == _grabbed_id:
                            nd.x  = float(np.clip(hx + _grab_off_x, 0, PHYS_W))
                            nd.y  = float(np.clip(hy + _grab_off_y, 0, PHYS_H))
                            nd.vx = 0.0
                            nd.vy = 0.0
                            nd.selected = True
                            break

                if not is_pinching and _was_pinching and _grabbed_id:
                    # Release — clear selection flag
                    for nd in g.nodes:
                        if nd.id == _grabbed_id:
                            nd.selected = False
                            break
                    _grabbed_id = None

            _was_pinching = is_pinching
        except Exception:
            pass   # hand tracking optional

        # ---------------------------------------------------------------
        # Render
        # ---------------------------------------------------------------
        rgba = render_to_rgba(g, width=out_w, height=out_h, ar=True)
        scriptOp.copyNumpyArray(np.ascontiguousarray(rgba, dtype=np.float32))

    except Exception as e:
        print(f'[render] ERROR: {e}')
        import traceback
        traceback.print_exc()
