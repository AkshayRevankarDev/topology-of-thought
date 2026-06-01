"""
td_setup.py — Entry-point to build the Topology of Thought TouchDesigner network.

Paste ONE of the lines below into the TD Textport (Alt+T) to bootstrap the
project.  The actual setup logic lives in ``touchdesigner/td_auto_setup.py``.

Option A — discover paths automatically (recommended):
    exec(open(project.folder + '/td_setup.py').read())

Option B — absolute path (if launched without a saved .toe):
    exec(open('/path/to/topology_of_thought/td_setup.py').read())

Prerequisites:
    1. Run ./setup_td.sh from a normal terminal to create a Python 3.11
       venv at .venv_td/  (TouchDesigner cannot load Python-3.14 extensions).
    2. Set TD's Python 64-bit Module Path to:
           <project>/.venv_td/lib/python3.11/site-packages
       (Edit → Preferences → DATs → Python 64-bit Module Path), then restart TD.
    3. Have ``ollama serve`` running so the rest of the pipeline works.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Locate td_auto_setup.py — must work both as a plain script AND when
# exec(open(...).read())'d from inside the TD Textport (where __file__ is
# NOT defined and project.folder points at the *open .toe*, not this repo).
# ---------------------------------------------------------------------------

def _find_auto_setup() -> Path:
    candidates = []

    # 1. Ideal: __file__ is defined (running as a real module / script).
    if '__file__' in globals():
        candidates.append(Path(__file__).resolve().parent / 'touchdesigner' / 'td_auto_setup.py')

    # 2. project.folder — the directory of the currently-open .toe file.
    #    Only useful when that .toe lives inside the project tree.
    try:
        candidates.append(Path(project.folder) / 'touchdesigner' / 'td_auto_setup.py')  # type: ignore[name-defined]
    except Exception:
        pass

    # 3. Walk *up* from project.folder looking for the repo root
    #    (identified by td_setup.py + touchdesigner/ co-existing).
    try:
        _pf = Path(project.folder)  # type: ignore[name-defined]
        for _d in [_pf] + list(_pf.parents):
            _test = _d / 'touchdesigner' / 'td_auto_setup.py'
            if _test.exists() and (_d / 'td_setup.py').exists():
                candidates.append(_test)
                break
    except Exception:
        pass

    # 4. Walk up from cwd (works when run from a terminal inside the repo).
    for _d in [Path.cwd()] + list(Path.cwd().parents):
        _test = _d / 'touchdesigner' / 'td_auto_setup.py'
        if _test.exists() and (_d / 'td_setup.py').exists():
            candidates.append(_test)
            break

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "Could not locate td_auto_setup.py.\n"
        "Tried:\n" + "\n".join(f"  {c}" for c in candidates) + "\n\n"
        "Make sure you run this from inside the topology_of_thought repo,\n"
        "or open a .toe that is saved inside the repo directory."
    )


_AUTO_SETUP = _find_auto_setup()
# Pass __file__ explicitly so td_auto_setup.py can compute PROJECT_ROOT correctly.
exec(compile(_AUTO_SETUP.read_text(), str(_AUTO_SETUP), 'exec'),
     {**globals(), '__file__': str(_AUTO_SETUP)})
