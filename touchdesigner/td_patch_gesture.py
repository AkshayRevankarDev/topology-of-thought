"""
td_patch_gesture.py — silence the mediapipe spam in gesture_exec.

Paste this ONE line into the TD Textport (Alt+T):

    exec(open('/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought/touchdesigner/td_patch_gesture.py').read())
"""

_ROOT  = '/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought'
_VENV  = _ROOT + '/.venv/lib/python3.11/site-packages'

_NEW_CODE = '''\
import sys as _sys
_root = '/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought'
_venv = '/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought/.venv/lib/python3.11/site-packages'
for _p in (_root, _venv):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

_engine = None
_import_ok = None  # None=not tried, True=ok, False=failed once

def _try_import():
    global _import_ok, _engine
    if _import_ok is not None:
        return _import_ok
    try:
        from touchdesigner.gesture_engine import GestureEngine
        _engine = GestureEngine()
        _import_ok = True
        print('[gesture_exec] GestureEngine ready.')
    except Exception as _e:
        _import_ok = False
        print('[gesture_exec] WARNING: gesture/hand import failed (printed once):')
        print(f'  {_e}')
        print('[gesture_exec] To enable hand tracking, set TD Python Module Path:')
        print('[gesture_exec]   Edit -> Preferences -> DATs -> Python 64-bit Module Path')
        print(f'[gesture_exec]   {_venv}')
        print('[gesture_exec]   Then restart TouchDesigner.')
    return _import_ok

def onStart():
    _try_import()

def onFrameStart(frame):
    if not _try_import():
        return  # already printed error once; silent every subsequent frame
    from touchdesigner.hand_tracking import _latest_hands
    events = _engine.update(list(_latest_hands))
    mode_dat = op('/project1/current_mode')
    for ev in events:
        if ev.name == 'pinch_start':
            if mode_dat:
                mode_dat.text = 'pinching'
        elif ev.name == 'pinch_end':
            if mode_dat:
                mode_dat.text = 'idle'
        elif ev.name.startswith('swipe_'):
            if mode_dat:
                mode_dat.text = ev.name
        elif ev.name == 'open_palm':
            if mode_dat:
                mode_dat.text = 'open_palm'
'''

_dat = op('/project1/gesture_exec')
if _dat is None:
    print('[td_patch_gesture] ERROR: /project1/gesture_exec not found.')
    print('  Run td_auto_setup.py first.')
else:
    _dat.text = _NEW_CODE
    print('[td_patch_gesture] gesture_exec patched — spam stopped.')
    print('[td_patch_gesture] The error will print ONCE on next frame cook, then go silent.')
