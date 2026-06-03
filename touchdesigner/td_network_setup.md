# TouchDesigner Network Setup — Topology of Thought

Complete wiring guide for the TD `.toe` file.  
All operator names are exact. Follow the order below top-to-bottom.

---

## Prerequisites

- TouchDesigner 2023.11+ (free non-commercial licence is fine)
- Python venv at `~/Desktop/Motion/topology_of_thought/.venv`
- `ollama serve` running in a terminal
- `data/sessions/attention_is_all_you_need.json` exists (run `python run_pipeline.py` first)

---

## Step 0 — Point TD at the venv Python

`Edit → Preferences → DATs → Python 64-bit Module Path`  
Add the path:
```
/Users/<you>/Desktop/Motion/topology_of_thought/.venv/lib/python3.14/site-packages
```
(Replace `3.14` with your actual Python minor version — check with `python --version`.)

Restart TouchDesigner after saving preferences.

---

## Step 1 — Project root Text DAT (run once on startup)

| Field | Value |
|---|---|
| Name | `project_init` |
| Operator | Text DAT |

Paste this Python into the DAT, then right-click → **Run Script**:

```python
import sys, pathlib
root = pathlib.Path('/Users/<you>/Desktop/Motion/topology_of_thought')
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Load the saved graph session into shared storage.
from features.session_memory import load_session
graph = load_session('attention_is_all_you_need')

# Store on a well-known base COMP so all scripts can find it.
parent.Graph.store('graph', graph)
print(f'Graph loaded: {len(graph.nodes)} nodes, {len(graph.edges)} edges')
```

> Tip: Convert this DAT to an **Execute DAT** (set `onStart` to `True`) so it
> runs automatically every time the `.toe` opens.

---

## Step 2 — Graph state storage Base COMP

| Field | Value |
|---|---|
| Name | `Graph` |
| Operator | Base COMP |
| Position | Top-left of the network |

This COMP acts as a namespace for `.store()` / `.fetch()` calls across all
scripts.  No parameters need changing.

---

## Step 3 — Webcam input

| Field | Value |
|---|---|
| Name | `webcam_in` |
| Operator | Video Device In TOP |
| Device | Select your webcam from the dropdown |
| Resolution | `640 × 480` |
| FPS | `30` |

---

## Step 4 — Hand tracking Script CHOP

| Field | Value |
|---|---|
| Name | `hand_tracker` |
| Operator | Script CHOP |
| Cook Type | `Every Frame` |

**Parameters tab:**
- Cook every frame: ✓

**Script tab — paste `touchdesigner/hand_tracking.py` in full**, then wire up the three callbacks:

```python
# start() callback
import sys, pathlib
root = pathlib.Path('/Users/<you>/Desktop/Motion/topology_of_thought')
sys.path.insert(0, str(root))
from touchdesigner.hand_tracking import startup
startup(camera_index=0)

# cook(scriptOp) callback
from touchdesigner.hand_tracking import cook
cook(scriptOp)

# stop() callback
from touchdesigner.hand_tracking import shutdown
shutdown()
```

**Output channels** (auto-created by `cook()`):  
`hand0_lm0_x … hand1_lm20_z` — 126 channels total (2 hands × 21 landmarks × 3 axes)

---

## Step 5 — Gesture engine Execute DAT

| Field | Value |
|---|---|
| Name | `gesture_engine_dat` |
| Operator | Execute DAT |
| `frameStart` | ✓ (run every frame) |

```python
# frameStart(frame) callback
import sys, pathlib
sys.path.insert(0, '/Users/<you>/Desktop/Motion/topology_of_thought')

from touchdesigner.gesture_engine import GestureEngine
from touchdesigner.hand_tracking import _latest_hands   # live results list

# Lazy-init engine on first frame
if not hasattr(parent(), '_gesture_engine'):
    parent()._gesture_engine = GestureEngine()

engine = parent()._gesture_engine
events = engine.update(list(_latest_hands))

for ev in events:
    print(f'[gesture] {ev.name}  hand={ev.hand_index}  pos={ev.position}')
    # TODO: route events to graph interaction handlers here.
    # Example: if ev.name == 'pinch_start': select_nearest_node(ev.position)
```

---

## Step 6 — Physics Execute DAT

| Field | Value |
|---|---|
| Name | `physics_dat` |
| Operator | Execute DAT |
| `onStart` | ✓ |
| `frameStart` | ✓ |

```python
# onStart() — create and start the engine
import sys
sys.path.insert(0, '/Users/<you>/Desktop/Motion/topology_of_thought')
from touchdesigner.physics import PhysicsEngine

graph = parent.Graph.fetch('graph')
engine = PhysicsEngine(
    graph,
    canvas_width=1920,
    canvas_height=1080,
    repulsion=4500,
    spring_k=0.045,
    spring_l=160,
    damping=0.83,
    gravity=0.009,
    ticks_per_second=60,
)
engine.scatter()
engine.start()
parent.Graph.store('physics_engine', engine)
print('Physics engine started.')

# onExit() — stop cleanly
engine = parent.Graph.fetch('physics_engine')
if engine:
    engine.stop()
```

---

## Step 7 — Graph renderer Script TOP

| Field | Value |
|---|---|
| Name | `graph_render` |
| Operator | Script TOP |
| Resolution | `1920 × 1080` (or your output resolution) |
| Pixel Format | `RGBA 32-bit float` |
| Cook Type | `Every Frame` |

```python
# cook(scriptOp) callback
import sys
sys.path.insert(0, '/Users/<you>/Desktop/Motion/topology_of_thought')

graph = parent.Graph.fetch('graph')
if graph is None:
    return

scriptOp.storage['graph'] = graph  # renderer reads this

from touchdesigner.graph_renderer import cook
cook(scriptOp)
```

---

## Step 8 — Nodes Table DAT (optional — for inspection / debugging)

| Field | Value |
|---|---|
| Name | `nodes_table` |
| Operator | Table DAT |

Add a **Script DAT** next to it (name: `populate_tables`, Execute → `frameStart`):

```python
# Dump graph nodes/edges into Table DATs for inspection
graph = parent.Graph.fetch('graph')
if graph is None:
    return

# Nodes table
nt = op('nodes_table')
nt.clear()
nt.appendRow(['id', 'label', 'confidence', 'x', 'y', 'source_paper', 'page_refs'])
for n in graph.nodes:
    nt.appendRow([
        n.id[:8], n.label, f'{n.confidence:.2f}',
        f'{n.x:.1f}', f'{n.y:.1f}',
        n.source_paper, str(n.page_refs),
    ])

# Edges table
et = op('edges_table')
et.clear()
et.appendRow(['id', 'source', 'target', 'relation', 'weight'])
for e in graph.edges:
    src = graph.get_node(e.source_id)
    tgt = graph.get_node(e.target_id)
    et.appendRow([
        e.id[:8],
        src.label[:20] if src else e.source_id[:8],
        tgt.label[:20] if tgt else e.target_id[:8],
        e.relation,
        f'{e.weight:.2f}',
    ])
```

Create a second Table DAT named `edges_table` alongside.

---

## Step 9 — Output / display

| Name | Operator | Input | Purpose |
|---|---|---|---|
| `render_out` | Out TOP | ← `graph_render` | Send to display |
| `webcam_preview` | Null TOP | ← `webcam_in` | Monitor camera feed |

Connect `graph_render` → `render_out` → your display or Window COMP.

---

## Full operator graph (left → right)

```
[project_init DAT]
        │ (onStart: loads graph into Graph COMP)
        ▼
   [Graph COMP]  ◀──────────────────────────────────────────────────────┐
        │                                                                │
        ├──▶ [physics_dat] (Execute DAT, frameStart)                    │
        │      starts PhysicsEngine in background thread                │
        │      mutates node.x / node.y every 60 Hz                     │
        │                                                                │
        ├──▶ [graph_render] (Script TOP, Every Frame)                   │
        │      reads graph → renders BGR → writes RGBA float32          │
        │      └──▶ [render_out] (Out TOP) → display                   │
        │                                                                │
[webcam_in] (Video Device In TOP)                                        │
        │                                                                │
        └──▶ [hand_tracker] (Script CHOP, Every Frame)                  │
                   126 channels: hand{h}_lm{i}_{x|y|z}                 │
                        │                                                │
                        └──▶ [gesture_engine_dat] (Execute DAT)         │
                                 reads _latest_hands                    │
                                 emits GestureEvents                    │
                                 routes to graph mutations ────────────▶┘
```

---

## Loading a different session at runtime

In the TD Textport (Alt+T), run:

```python
from features.session_memory import load_session
graph = load_session('your_session_name')   # no .json extension
parent.Graph.store('graph', graph)
print(graph)
```

---

## Saving the current graph state from TD

```python
from features.session_memory import save_session
graph = parent.Graph.fetch('graph')
save_session(graph, name='my_session')
# Saves to data/sessions/my_session.json
```

---

## 3D / "Iron Man" globe view (MVP)

After `td_auto_setup.py` has run and `graph_store.graph` is populated, build
the 3D scene from the Textport:

```python
from touchdesigner import setup_3d
setup_3d.build_3d_scene()
```

This creates, under `/project1`:

- `node_positions` / `edge_positions` — Script CHOPs that surface the live
  3D positions written by the sphere-mode `PhysicsEngine`.
- `nodes_geo` — Geometry COMP, small Sphere SOP instanced per node.
- `edges_geo` — Geometry COMP, Script SOP that builds one polyline per edge.
- `cam1`, `light1` — orbit camera + key light.
- `cam_mouse` (Mouse In CHOP) + `cam_exec` (Execute DAT, onFrameStart) —
  drag-to-orbit, scroll/pinch-to-dolly.
- `render3d` (Render TOP, 1280×720), composited over `webcam_in` via
  `composite3d` (Over TOP) for AR passthrough, then `out3d`.

**Controls (MVP):**

| Gesture | Action |
|---|---|
| Left-drag in `cam_mouse`'s panel | Orbit (azimuth / elevation) |
| Scroll / two-finger pinch | Dolly (zoom in/out) |
| Decrease distance below 3.0 | Camera ends up *inside* the globe → "around me" |

Webcam-hand-tracking gestures (pinch/grab) layer on later via the existing
`gesture_engine.py` + `hand_tracking.py` modules.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Script CHOP shows `ImportError` | Check venv path in Preferences; restart TD |
| `graph` is None in renderer | Ensure `project_init` ran and `Graph` COMP exists |
| Physics not running | Check `physics_dat` Execute DAT `onStart` is ticked |
| Webcam blank | Grant Terminal/TD camera permission in System Settings |
| MediaPipe `solutions` error | Confirm mediapipe ≥ 0.10.30 and Tasks API imports |
| Hand landmarker model missing | It auto-downloads on first `startup()` call (~8 MB) |
