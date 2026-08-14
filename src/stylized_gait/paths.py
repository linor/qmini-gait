"""Where things live.  Everything else asks this module rather than guessing
from ``__file__``, so the package works from an editable install, a wheel, or a
plain clone."""

import os

# .../src/stylized_gait/paths.py -> repo root
ROOT = os.path.abspath(os.environ.get(
    "QMINI_GAIT_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

DATA = os.path.join(ROOT, "data")
CLIPS = os.path.join(DATA, "clips")
BLENDER_EDIT = os.path.join(DATA, "blender_edit")
WEB = os.path.join(ROOT, "web")
DIST = os.path.join(ROOT, "dist")
ISAAC = os.path.join(ROOT, "isaac")

ROBOT_USD = os.path.join(DATA, "qmini_urdf-2legs.usda")
MESH_JSON = os.path.join(DATA, "qmini_mesh.json")
PARAMS_JSON = os.path.join(DATA, "gait_params.json")
LIBRARY_JSON = os.path.join(DATA, "gait_library.json")
OVERRIDES_JSON = os.path.join(DATA, "gait_overrides.json")


def ensure(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs[0] if len(dirs) == 1 else dirs
