# Topology of Thought

Turn academic PDFs into an interactive, force-directed knowledge graph you can
explore with your **mouse** or your **hand** (MediaPipe Tasks HandLandmarker).
Runs entirely locally — Ollama for the LLM, sentence-transformers for
embeddings, OpenCV for the renderer. No cloud APIs.

![preview](data/exports/preview_with_hud.png)

## What it does

1. **Ingest** a PDF → text chunks (PyMuPDF).
2. **Extract** concepts + relationships per chunk via a local `llama3.2` (Ollama).
3. **Deduplicate** semantically similar nodes with `all-MiniLM-L6-v2`.
4. **Persist** the graph as JSON (`data/sessions/*.json`).
5. **Render** an interactive force-directed graph with pinch-to-grab gestures
   and a webcam overlay — no TouchDesigner required (TouchDesigner project files
   included for advanced use).

## Features

| Module | What it does |
| --- | --- |
| `features/anki_export.py` | Export concepts as an Anki `.apkg` deck. |
| `features/confidence_coloring.py` | Per-node confidence → RGB gradient. |
| `features/cross_paper.py` | Compare two PDFs, draw bridge edges. |
| `features/query_mode.py` | Voice query: Whisper → embedding retrieval → grounded Ollama answer. |
| `features/session_memory.py` | Atomic JSON save/load + auto-save thread. |
| `features/zoom_node.py` | Pinch-hold to ask Ollama for a sub-graph decomposition. |

## Quick start (standalone — no TouchDesigner)

```bash
# 1. Setup
./setup.sh                        # creates .venv, installs deps, pulls llama3.2

# 2. Verify everything works (target: 7/7 green)
.venv/bin/python health_check.py

# 3. Run the unit test suite
.venv/bin/python pipeline_test.py

# 4. Build a fresh graph from your own PDF
#    (drop it into data/pdfs/ first, then edit run_pipeline.py)
.venv/bin/python run_pipeline.py

# 5. Open the interactive preview (mouse + webcam gestures)
.venv/bin/python touchdesigner/standalone_preview.py
```

A sample session built from *Attention Is All You Need* (Vaswani et al., 2017)
ships at `data/sessions/attention_is_all_you_need.json` so the preview works
without re-ingesting.

## TouchDesigner path (optional)

The same code can be driven from inside TouchDesigner as a Script TOP / Script
CHOP network — useful if you want to composite the graph onto a live video
feed, hook it up to MIDI controllers, or send the rendered frames over
Spout/NDI.

```bash
# 1. Create a TD-compatible Python 3.11 venv (TD ships with Python 3.11,
#    so its ABI doesn't match the standalone .venv if that is 3.12+).
./setup_td.sh                     # creates .venv_td/, prints the site-pkgs path
```

In TouchDesigner:

1. **Edit → Preferences → DATs → Python 64-bit Module Path** — paste the
   `.venv_td/lib/python3.11/site-packages` path printed by `setup_td.sh`, then
   restart TouchDesigner.
2. Open a blank project, press **Alt+T** to bring up the Textport, and run:
   ```python
   exec(open('/absolute/path/to/topology_of_thought/td_setup.py').read())
   ```
3. The script builds the full network inside `/project1` — webcam input,
   Script CHOP for hand tracking, Script TOP for the renderer, Execute DATs
   for the physics engine and gesture FSM, Over/Out TOPs for compositing — and
   saves itself as `touchdesigner/topology_of_thought.toe`.
4. From then on, double-click `touchdesigner/OPEN_IN_TD.command` to reopen it.

Full operator-by-operator wiring guide: [`touchdesigner/td_network_setup.md`](touchdesigner/td_network_setup.md).

## Controls

**Mouse**: click & drag any node.
**Hand (webcam, MediaPipe Tasks API)**:

| Gesture | Action |
| --- | --- |
| Pinch (thumb + index) | Grab the hovered node |
| Pinch + move | Drag |
| Open palm | Release everything |
| Swipe down | Re-scatter the graph |

The top-left HUD shows live gesture telemetry — hand count, pinch distance,
threshold bar, last event, currently held node.

**Keyboard**: `Q` quit, `S` screenshot, `R` scatter, `Space` pause physics,
`G` toggle webcam thumbnail.

## Requirements

- macOS / Linux (Windows untested).
- Python 3.11+.
- [Ollama](https://ollama.com/) running locally with `llama3.2` pulled.
- A webcam if you want gesture control (mouse mode works without).

On macOS, grant **camera permission** to the terminal app (or to the launcher
that runs `python`) under *System Settings → Privacy & Security → Camera*. The
preview falls back to mouse-only if access is denied.

## Project layout

```
core/                  PDF ingest, concept extraction, dedup, graph state
features/              Anki, confidence colors, cross-paper, query, sessions, zoom
touchdesigner/         Physics, hand tracking, gesture FSM, renderer, preview
data/
  pdfs/                drop PDFs here (gitignored)
  sessions/            persisted graphs (JSON)
  exports/             screenshots, .apkg decks (gitignored)
health_check.py        7-point dependency / runtime check
pipeline_test.py       Unit tests across the full pipeline
run_pipeline.py        End-to-end: PDF → session JSON
setup.sh               One-shot environment bootstrap
```
