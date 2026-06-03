"""
td_auto_setup.py — Build the full Topology of Thought TD network from scratch.

HOW TO RUN — paste this ONE line into the TD Textport (Alt+T):

    exec(open('/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought/touchdesigner/td_auto_setup.py').read())

What it does:
  1. Creates every required operator inside /project1
  2. Populates nodes/edges Table DATs from the saved JSON session
  3. Writes all Python scripts into their Text DATs
  4. Wires Script CHOP → hand_track_script DAT
     Wires Script TOP  → render_script DAT
  5. Connects operators (webcam → over → out, render → over)
  6. Saves the project as touchdesigner/topology_of_thought.toe

TD version: 2025.32460
Python: 3.11 (TD's built-in), venv packages visible via sys.path insert
"""

import sys
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _find_project_root() -> Path:
    """Locate the topology_of_thought project root reliably.

    When this script is exec()'d from the TD Textport, __file__ is not set
    and project.folder / cwd both point at TD's app bundle.  We try several
    strategies in priority order, returning the first that contains both
    'td_setup.py' and a 'touchdesigner/' subdirectory.
    """
    def _is_root(p: Path) -> bool:
        return (p / 'td_setup.py').exists() and (p / 'touchdesigner').is_dir()

    # 1. Normal import — __file__ is always right.
    try:
        candidate = Path(__file__).resolve().parent.parent
        if _is_root(candidate):
            return candidate
    except NameError:
        pass

    # 2. TD Textport exec — project.folder is the .toe's containing directory.
    #    Works once a .toe saved inside the project folder is open.
    try:
        candidate = Path(project.folder)  # type: ignore[name-defined]  # TD global
        for d in [candidate] + list(candidate.parents):
            if _is_root(d):
                return d
    except Exception:
        pass

    # 3. Hardcoded known location (this machine's Desktop/Motion path).
    hardcoded = Path('/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought')
    if _is_root(hardcoded):
        return hardcoded

    # 4. Walk upward from cwd.
    for d in [Path.cwd()] + list(Path.cwd().parents):
        if _is_root(d):
            return d

    # 5. Search common Desktop/Documents locations for any matching project.
    for base in (Path.home() / 'Desktop', Path.home() / 'Documents'):
        for child in base.glob('**/topology_of_thought'):
            if _is_root(child):
                return child

    raise RuntimeError(
        "Cannot locate topology_of_thought project root.\n"
        "Make sure you are running from the correct folder or that the .toe\n"
        "is saved inside the project directory."
    )

PROJECT_ROOT = _find_project_root()
SESSION_PATH = PROJECT_ROOT / 'data' / 'sessions' / 'attention_is_all_you_need.json'
TOE_SAVE_PATH = PROJECT_ROOT / 'touchdesigner' / 'topology_of_thought.toe'


def _discover_venv_site() -> Path:
    """Locate a TD-compatible Python 3.11 venv site-packages dir.

    Preference order:
      1. ``.venv_td/lib/python3.11/site-packages``  (built by ``setup_td.sh``)
      2. ``.venv/lib/python3.11/site-packages``
      3. Any ``.venv*/lib/python3.1[0-2]/site-packages`` that exists.
    """
    candidates = [
        PROJECT_ROOT / '.venv_td' / 'lib' / 'python3.11' / 'site-packages',
        PROJECT_ROOT / '.venv' / 'lib' / 'python3.11' / 'site-packages',
    ]
    for c in candidates:
        if c.exists():
            return c
    for venv_dir in sorted(PROJECT_ROOT.glob('.venv*')):
        for py_minor in ('3.12', '3.11', '3.10'):
            site = venv_dir / 'lib' / f'python{py_minor}' / 'site-packages'
            if site.exists():
                return site
    return PROJECT_ROOT / '.venv_td' / 'lib' / 'python3.11' / 'site-packages'


VENV_SITE = _discover_venv_site()
if not VENV_SITE.exists():
    print(f'[td_auto_setup] WARNING: TD-compatible venv not found at {VENV_SITE}')
    print('[td_auto_setup] Run ./setup_td.sh from the project root to create one.')

for _p in (str(PROJECT_ROOT), str(VENV_SITE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Base container — build everything inside /project1
# ---------------------------------------------------------------------------
BASE = op('/project1')

# Clear any Python objects stored in /project1 from previous setup runs.
# Old 'graph' and 'physics_engine' entries contain threading.Lock objects
# which cause TD to fail with a pickle error on .toe save.
for _key in list(BASE.storage.keys()):
    try:
        BASE.unstore(_key)
    except Exception:
        pass
print('[td_auto_setup] Cleared operator storage (prevents pickle errors on save)')

def _make(op_type, name, x=0, y=0):
    """Destroy any existing operator with *name* then create a fresh one.

    Args:
        op_type: TD operator type global (e.g. tableDAT, scriptCHOP).
        name: Operator name string.
        x: Node X position in the network editor.
        y: Node Y position in the network editor.

    Returns:
        The newly created TD operator.
    """
    existing = BASE.op(name)
    if existing is not None:
        existing.destroy()
    n = BASE.create(op_type, name)
    n.nodeX = x
    n.nodeY = y
    return n


def _set_par(op_obj, par_name, value, *fallback_names):
    """Set a TD parameter by name, silently trying fallbacks if the first fails.

    Prints a diagnostic listing all available parameter names when every
    attempt fails so the user knows what name to use.

    Args:
        op_obj: The TD operator whose parameter to set.
        par_name: Primary parameter name to try.
        value: Value to assign.
        *fallback_names: Additional names to try in order.
    """
    for name in (par_name,) + fallback_names:
        try:
            setattr(op_obj.par, name, value)
            return
        except Exception:
            continue
    # All names failed — print available pars so user can find the right name.
    available = [p.name for p in op_obj.pars()]
    print(f'[td_auto_setup] WARNING: could not set {op_obj.name}.par.{par_name} = {value!r}')
    print(f'  Available parameters: {available[:30]}')
    print(f'  Set it manually in TD: op("{op_obj.name}").par.<name> = {value!r}')

# ---------------------------------------------------------------------------
# 1. Load session data from JSON
# ---------------------------------------------------------------------------
if not SESSION_PATH.exists():
    raise FileNotFoundError(
        f"Session not found: {SESSION_PATH}\n"
        "Run `python run_pipeline.py` from the project terminal first."
    )

with open(SESSION_PATH, 'r', encoding='utf-8') as _f:
    _session = json.load(_f)

_nodes = _session.get('nodes', [])
_edges = _session.get('edges', [])
print(f'[td_auto_setup] Loaded {len(_nodes)} nodes, {len(_edges)} edges')

# ---------------------------------------------------------------------------
# 2. Table DATs — nodes and edges
# ---------------------------------------------------------------------------
nodes_tbl = _make(tableDAT, 'nodes_table', x=-600, y=300)
nodes_tbl.clear()
nodes_tbl.appendRow(['id', 'label', 'confidence', 'x', 'y', 'z', 'source_paper', 'page_refs'])
for _n in _nodes:
    nodes_tbl.appendRow([
        str(_n.get('id', ''))[:12],
        str(_n.get('label', '')),
        f"{float(_n.get('confidence', 1.0)):.3f}",
        f"{float(_n.get('x', 0.0)):.1f}",
        f"{float(_n.get('y', 0.0)):.1f}",
        f"{float(_n.get('z', 0.0)):.1f}",
        str(_n.get('source_paper', '')),
        str(_n.get('page_refs', [])),
    ])

edges_tbl = _make(tableDAT, 'edges_table', x=-600, y=100)
edges_tbl.clear()
edges_tbl.appendRow(['id', 'source_id', 'target_id', 'relation', 'weight', 'confidence'])
for _e in _edges:
    edges_tbl.appendRow([
        str(_e.get('id', ''))[:12],
        str(_e.get('source_id', ''))[:12],
        str(_e.get('target_id', ''))[:12],
        str(_e.get('relation', 'related_to')),
        f"{float(_e.get('weight', 1.0)):.3f}",
        f"{float(_e.get('confidence', 1.0)):.3f}",
    ])

print(f'[td_auto_setup] Table DATs populated')

# ---------------------------------------------------------------------------
# 3a. Graph store Text DAT — shared module for passing graph between DATs.
#     Physics exec sets graph_store.graph; render script reads it.
#     Plain Text DAT module-level vars are never pickled by TD on .toe save.
# ---------------------------------------------------------------------------
graph_store_dat = _make(textDAT, 'graph_store', x=-600, y=200)
graph_store_dat.text = 'graph = None  # set by physics_exec.onStart()'
print('[td_auto_setup] graph_store Text DAT created')

# ---------------------------------------------------------------------------
# 3b. State Text DAT  (current interaction mode)
# ---------------------------------------------------------------------------
mode_dat = _make(textDAT, 'current_mode', x=-600, y=-100)
mode_dat.text = 'idle'

# ---------------------------------------------------------------------------
# 4. Hand-tracking script DAT + Script CHOP
# ---------------------------------------------------------------------------
_hand_script_code = f'''# Hand tracking cook function — referenced by hand_tracker Script CHOP
import sys as _sys
_root = r'{PROJECT_ROOT}'
_venv = r'{VENV_SITE}'
for _p in (_root, _venv):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

def start(CHOP):
    """Called when the Script CHOP cooks for the first time."""
    from touchdesigner.hand_tracking import startup
    startup(camera_index=0)

def cook(CHOP):
    """Called every frame to update hand landmark channels."""
    from touchdesigner.hand_tracking import cook as _cook
    _cook(CHOP)

def stop(CHOP):
    """Called when TD exits or the CHOP is deactivated."""
    from touchdesigner.hand_tracking import shutdown
    shutdown()
'''

hand_script_dat = _make(textDAT, 'hand_track_script', x=-400, y=400)
hand_script_dat.text = _hand_script_code

hand_chop = _make(scriptCHOP, 'hand_tracker', x=-200, y=400)
# 'cooktype' does not exist in TD 2025 — Script CHOP always cooks when active.
# Link Script CHOP to its script DAT (try all known parameter name variants).
_set_par(hand_chop, 'callbacks', hand_script_dat, 'dat', 'scriptdat', 'Dat', 'scriptDAT')

print('[td_auto_setup] hand_tracker Script CHOP created')

# ---------------------------------------------------------------------------
# 5. Webcam Video Device In TOP
# ---------------------------------------------------------------------------
cam_top = _make(videodeviceinTOP, 'webcam_in', x=-600, y=-300)
_set_par(cam_top, 'device',  0,   'Device')
_set_par(cam_top, 'resx',    640, 'resolutionw', 'width')
_set_par(cam_top, 'resy',    480, 'resolutionh', 'height')

print('[td_auto_setup] webcam_in TOP created')

# ---------------------------------------------------------------------------
# 6. Graph renderer script DAT + noise_trigger TOP + Script TOP
# ---------------------------------------------------------------------------
# Load the callbacks content from the live file so this setup always stays
# in sync with edits made outside TD.
_render_script_live = PROJECT_ROOT / 'touchdesigner' / 'render_script_live.py'
if _render_script_live.exists():
    _render_script_code = _render_script_live.read_text(encoding='utf-8')
    print(f'[td_auto_setup] Loaded render_script from {_render_script_live.name}')
else:
    # Fallback inline version if the file is missing
    _render_script_code = f'''import sys as _sys
_root = r'{PROJECT_ROOT}'
_venv = r'{VENV_SITE}'
for _p in (_root, _venv):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_sys.modules.pop('touchdesigner.graph_renderer', None)

def cook(scriptOp):
    g = op('/project1').fetch('graph', None)
    if g is None:
        import numpy as np
        scriptOp.copyNumpyArray(
            np.zeros((scriptOp.height, scriptOp.width, 4), dtype=np.float32))
        return
    try:
        import numpy as np
        from touchdesigner.graph_renderer import render_to_rgba
        out_w = max(scriptOp.width, 1280)
        out_h = max(scriptOp.height, 720)
        rgba = render_to_rgba(g, width=out_w, height=out_h)
        scriptOp.copyNumpyArray(np.ascontiguousarray(rgba, dtype=np.float32))
    except Exception as e:
        print(f'[render] ERROR: {{e}}')
        import traceback; traceback.print_exc()
'''
    print('[td_auto_setup] WARNING: render_script_live.py not found, using inline fallback')

render_script_dat = _make(textDAT, 'render_script', x=-200, y=0)
render_script_dat.text = _render_script_code

# noise_trigger: a time-varying Noise TOP whose output drives graph_render
# to cook every frame, without creating a dependency loop.
noise_trigger = _make(noiseTOP, 'noise_trigger', x=-50, y=0)
_set_par(noise_trigger, 'resolutionw', 1280, 'resx', 'width')
_set_par(noise_trigger, 'resolutionh', 720,  'resy', 'height')

render_top = _make(scriptTOP, 'graph_render', x=100, y=0)
_set_par(render_top, 'callbacks', render_script_dat, 'dat', 'scriptdat', 'Dat')
# Resolution is determined by the numpy array we pass to copyNumpyArray() —
# no resolution parameter needed on the Script TOP itself.

# Wire noise_trigger into graph_render input 0 so every noise frame
# causes graph_render to re-cook and update its output.
try:
    render_top.inputConnectors[0].connect(noise_trigger)
    print('[td_auto_setup] noise_trigger → graph_render connected')
except Exception as _e:
    print(f'[td_auto_setup] WARNING noise_trigger connect: {_e}')

print('[td_auto_setup] graph_render Script TOP created')

# ---------------------------------------------------------------------------
# 7. Physics + session-loader Execute DAT (onStart)
# ---------------------------------------------------------------------------
_physics_exec_code = f'''# Physics engine — loads graph on startup, runs sphere simulation in background.
#
# IMPORTANT: we intentionally do NOT use op('/project1').store() for the graph
# or engine.  TD tries to pickle all stored objects when saving the .toe, and
# threading.Lock (inside PhysicsEngine) and large numpy arrays (embeddings in
# GraphState) both fail pickle.  Instead we expose them as module-level vars
# so other DATs can reach them via TD's mod[] accessor:
#
#     g = mod['/project1/physics_exec'].graph
#
import sys as _sys
_root = r'{PROJECT_ROOT}'
_venv = r'{VENV_SITE}'
for _p in (_root, _venv):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

graph  = None   # GraphState — read by render_script via mod[this].graph
_engine = None  # PhysicsEngine — kept here, never pickled

def onStart():
    global graph, _engine
    import json
    from pathlib import Path
    from core.graph_state import GraphState
    from touchdesigner.physics import PhysicsEngine

    session_path = Path(r'{SESSION_PATH}')
    with open(session_path, 'r') as f:
        data = json.load(f)

    graph = GraphState.from_dict(data)

    _engine = PhysicsEngine(
        graph,
        canvas_width=1920,
        canvas_height=1080,
        repulsion=4500,
        spring_k=0.045,
        spring_l=160,
        damping=0.85,
        gravity=0.0,
        ticks_per_second=60,
        sphere_mode=True,
        sphere_radius=480.0,
        sphere_center=(960.0, 540.0, 0.0),
    )
    _engine.start()

    # Share graph via the graph_store Text DAT module.
    # Plain module-level variables in Text DATs are never pickled on .toe save.
    mod(op('/project1/graph_store')).graph = graph

    print(f'[physics_exec] Sphere graph ready: {{len(graph.nodes)}} nodes, {{len(graph.edges)}} edges')
    print('[physics_exec] Globe auto-rotates; pinch=grab/rotate, hold=decompose, tap=zoom.')

def onExit():
    global _engine
    if _engine is not None:
        _engine.stop()
        _engine = None
        print('[physics_exec] Physics engine stopped.')

def onFrameStart(frame):
    pass
'''

physics_exec = _make(executeDAT, 'physics_exec', x=-400, y=-300)
physics_exec.text = _physics_exec_code
_set_par(physics_exec, 'active',     True)
_set_par(physics_exec, 'start',      True,  'onstart',  'Start')
_set_par(physics_exec, 'exit',       True,  'onexit',   'Exit')
_set_par(physics_exec, 'framestart', False, 'onframestart')

print('[td_auto_setup] physics_exec Execute DAT created')

# ---------------------------------------------------------------------------
# 8. Gesture engine Execute DAT (onFrameStart)
# ---------------------------------------------------------------------------
_gesture_exec_code = f'''# Gesture engine — runs every frame to dispatch pinch/swipe events
import sys as _sys
_root = r'{PROJECT_ROOT}'
_venv = r'{VENV_SITE}'
for _p in (_root, _venv):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

_engine = None        # module-level singleton
_import_ok = None     # None = not yet tried, True = ok, False = failed once

def _try_import():
    """Import gesture/hand modules once; return True on success, False on failure.
    Prints the error exactly once so it does not spam the Textport every frame.
    """
    global _import_ok, _engine
    if _import_ok is not None:
        return _import_ok
    try:
        from touchdesigner.gesture_engine import GestureEngine
        _engine = GestureEngine()
        _import_ok = True
        print('[gesture_exec] GestureEngine initialised.')
    except Exception as _e:
        _import_ok = False
        print(f'[gesture_exec] WARNING: could not import gesture/hand modules: {{_e}}')
        print('[gesture_exec] Hand gestures disabled until TD Python Module Path is set.')
        print('[gesture_exec] Edit → Preferences → DATs → Python 64-bit Module Path:')
        print(f'[gesture_exec]   {{_venv}}')
        print('[gesture_exec] Then restart TD.')
    return _import_ok

def onStart():
    _try_import()

def onFrameStart(frame):
    if not _try_import():
        return  # already printed error once; silently skip every subsequent frame

    # Safe import — modules are already loaded after _try_import() succeeded
    from touchdesigner.hand_tracking import _latest_hands
    events = _engine.update(list(_latest_hands))

    graph    = op('/project1').fetch('graph', None)
    mode_dat = op('/project1/current_mode')

    for ev in events:
        if ev.name == 'pinch_start':
            if mode_dat is not None:
                mode_dat.text = 'pinching'

        elif ev.name == 'pinch_end':
            if mode_dat is not None:
                mode_dat.text = 'idle'

        elif ev.name.startswith('swipe_'):
            if mode_dat is not None:
                mode_dat.text = ev.name   # e.g. 'swipe_left'

        elif ev.name == 'open_palm':
            if mode_dat is not None:
                mode_dat.text = 'open_palm'
'''

gesture_exec = _make(executeDAT, 'gesture_exec', x=-200, y=-300)
gesture_exec.text = _gesture_exec_code
_set_par(gesture_exec, 'active',     True)
_set_par(gesture_exec, 'start',      True,  'onstart',  'Start')
_set_par(gesture_exec, 'framestart', True,  'onframestart', 'Framestart')

print('[td_auto_setup] gesture_exec Execute DAT created')

# ---------------------------------------------------------------------------
# 9. Over TOP (graph render composited over webcam)
# ---------------------------------------------------------------------------
over_top = _make(overTOP, 'composite', x=200, y=0)

# ---------------------------------------------------------------------------
# 10. Null TOP (clean monitoring tap before output)
# ---------------------------------------------------------------------------
monitor_top = _make(nullTOP, 'monitor', x=350, y=0)

# ---------------------------------------------------------------------------
# 11. Out TOP (final output)
# ---------------------------------------------------------------------------
out_top = _make(outTOP, 'output', x=500, y=0)

# ---------------------------------------------------------------------------
# 12. Window COMP — opens the globe viewer automatically on .toe load
# ---------------------------------------------------------------------------
win_comp = _make(windowCOMP, 'viewer_window', x=700, y=0)
# TD 2025 Window COMP parameter names (confirmed from live par list):
# winop    — source TOP operator path
# drawwindow — open/show the window (True = visible)
# winoffsetx/y — screen position
# winw/winh — dimensions
# borders   — show OS window chrome
_set_par(win_comp, 'winop',      '/project1/monitor', 'top', 'TOP')
_set_par(win_comp, 'drawwindow', True,  'open', 'openonstart', 'Drawwindow')
_set_par(win_comp, 'borders',    False, 'Borders', 'border')
_set_par(win_comp, 'winw',       1280,  'Winw', 'width')
_set_par(win_comp, 'winh',       720,   'Winh', 'height')
_set_par(win_comp, 'winoffsetx', 0,     'winstartx', 'startx')
_set_par(win_comp, 'winoffsety', 0,     'winstarty', 'starty')

print('[td_auto_setup] viewer_window Window COMP created (opens on start)')
print('[td_auto_setup] All operators created')

# ---------------------------------------------------------------------------
# 12. Wire connections
# ---------------------------------------------------------------------------
# graph_render → composite[0] (background layer)
try:
    over_top.inputConnectors[0].connect(render_top)
except Exception as _e:
    print(f'[td_auto_setup] WARNING composite[0] connect: {_e}')

# webcam_in → composite[1] (foreground layer, shows hand skeleton overlay)
try:
    over_top.inputConnectors[1].connect(cam_top)
except Exception as _e:
    print(f'[td_auto_setup] WARNING composite[1] connect: {_e}')

# composite → monitor → output
try:
    monitor_top.inputConnectors[0].connect(over_top)
    out_top.inputConnectors[0].connect(monitor_top)
except Exception as _e:
    print(f'[td_auto_setup] WARNING output chain connect: {_e}')

print('[td_auto_setup] Operators wired')

# ---------------------------------------------------------------------------
# 13. Layout nodes cleanly in the network editor
# ---------------------------------------------------------------------------
_positions = {
    'nodes_table':      (-600, 300),
    'edges_table':      (-600, 100),
    'current_mode':     (-600, -100),
    'webcam_in':        (-600, -300),
    'hand_track_script':(-400,  400),
    'hand_tracker':     (-200,  400),
    'render_script':    (-200,  200),
    'noise_trigger':    (  50,  200),
    'graph_render':     ( 200,  200),
    'physics_exec':     (-400, -200),
    'gesture_exec':     (-200, -200),
    'composite':        ( 200,  200),
    'monitor':          ( 350,  200),
    'output':           ( 500,  200),
    'viewer_window':    ( 700,  200),
}
for _name, (_x, _y) in _positions.items():
    _op = BASE.op(_name)
    if _op is not None:
        _op.nodeX = _x * 2   # TD uses larger pixel units
        _op.nodeY = _y * 2

# ---------------------------------------------------------------------------
# 14. Save the project as a .toe file
# ---------------------------------------------------------------------------
_toe_path = str(TOE_SAVE_PATH)
try:
    project.save(_toe_path)
    print(f'[td_auto_setup] Project saved: {_toe_path}')
except Exception as _e:
    print(f'[td_auto_setup] WARNING: project.save() failed: {_e}')
    print('  → File → Save As manually to save the .toe')

print('')
print('=' * 60)
print('  Topology of Thought — Iron Man Globe Mode ready')
print(f'  {len(_nodes)} nodes  |  {len(_edges)} edges  |  Fibonacci sphere')
print('  Globe auto-rotates on Y axis when idle (~12°/sec).')
print('  Pinch + drag in empty space  → spin the globe')
print('  Quick pinch on a node        → zoom toward it')
print('  Long pinch on a node (1.5s)  → Ollama decomposition')
print('  Physics starts automatically on next TD launch.')
print('  Viewer window opens automatically on .toe load.')
print('  Manual open: right-click viewer_window → Open Window')
print('  Fullscreen inside viewer: press F')
print('  Open OPEN_IN_TD.command to reopen this project.')
print('=' * 60)

# Trigger physics onStart immediately so graph is loaded without a TD restart.
try:
    mod(op('/project1/physics_exec')).onStart()
    print('[td_auto_setup] Physics started — graph loaded into graph_store.')
except Exception as _e:
    print(f'[td_auto_setup] Note: physics auto-start failed: {_e}')
    print('[td_auto_setup] Run: mod(op(\'/project1/physics_exec\')).onStart()')

# Open the window immediately in this session (don't wait for restart).
# TD 2025 Window COMP: try par.winopen pulse first, fall back to openViewer().
try:
    _win = BASE.op('viewer_window')
    if _win is not None:
        _opened = False
        for _attr in ('winopen', 'open', 'Open', 'Winopen'):
            try:
                getattr(_win.par, _attr).pulse()
                _opened = True
                break
            except Exception:
                pass
        if not _opened:
            try:
                _win.openViewer()
                _opened = True
            except Exception:
                pass
        if _opened:
            print('[td_auto_setup] Viewer window opened.')
        else:
            raise RuntimeError('no working open method found')
except Exception as _e:
    print(f'[td_auto_setup] Note: auto-open failed ({_e})')
    print('[td_auto_setup] In the network: right-click viewer_window → Open Window')
