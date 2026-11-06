"""Run queued Python files inside the open Unreal Editor (current session only).

    py "E:/GitHub/Graphics/Projects/Portfolio/SFRoom2/UE/Import/command_watcher.py"

Every half second, each *.py in QUEUE_DIR is executed in order and replaced by
<name>.done (or <name>.error with the traceback). Nothing persists after the editor closes.
"""
import os
import tempfile
import time
import traceback

import unreal

QUEUE_DIR = os.path.join(tempfile.gettempdir(), "sfroom2_ue_queue")
os.makedirs(QUEUE_DIR, exist_ok=True)

_last = [0.0]


def _tick(delta):
    now = time.time()
    if now - _last[0] < 0.5:
        return
    _last[0] = now
    for name in sorted(f for f in os.listdir(QUEUE_DIR) if f.endswith(".py")):
        path = os.path.join(QUEUE_DIR, name)
        with open(path, encoding="utf-8") as f:
            code = f.read()
        os.remove(path)
        scope = {"__name__": "__queued__", "__file__": path}
        try:
            exec(compile(code, path, "exec"), scope)
            out, ext = repr(scope.get("result")), ".done"
        except Exception:
            out, ext = traceback.format_exc(), ".error"
        with open(path[:-3] + ext, "w", encoding="utf-8") as f:
            f.write(out)


if getattr(unreal, "_sfroom2_watcher", None):
    unreal.unregister_slate_post_tick_callback(unreal._sfroom2_watcher)
unreal._sfroom2_watcher = unreal.register_slate_post_tick_callback(_tick)
unreal.log("SFRoom2 command watcher running: " + QUEUE_DIR)
