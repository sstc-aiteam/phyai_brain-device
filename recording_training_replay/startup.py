"""Small progress snapshot readable while recording owns the device lock."""
import threading

_LOCK = threading.Lock()
_STATE = {'starting': False, 'camera_ready': False}


def update(**values):
    with _LOCK:
        _STATE.update(values)


def snapshot():
    with _LOCK:
        return dict(_STATE)
