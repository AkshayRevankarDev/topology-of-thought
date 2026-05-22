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

# Locate the touchdesigner/td_auto_setup.py script next to this file.
_HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
_AUTO_SETUP = _HERE / 'touchdesigner' / 'td_auto_setup.py'

if not _AUTO_SETUP.exists():
    # When run inside TD's Textport, __file__ may not be defined; fall back
    # to the saved project folder.
    try:
        _AUTO_SETUP = Path(project.folder) / 'touchdesigner' / 'td_auto_setup.py'  # type: ignore[name-defined]
    except Exception:
        pass

if not _AUTO_SETUP.exists():
    raise FileNotFoundError(
        f"Could not locate td_auto_setup.py.  Expected at: {_AUTO_SETUP}"
    )

exec(compile(_AUTO_SETUP.read_text(), str(_AUTO_SETUP), 'exec'))
