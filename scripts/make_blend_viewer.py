"""
make_blend_viewer.py -- bake the nine clips + the robot mesh into
gait_blend.html, the interactive keyboard-driven blending viewer.
"""
import os, json, glob
import make_viewer as MV
from stylized_gait import paths

HERE = paths.DATA
TEMPLATE = "blend_template.html"

def main():
    clips = [MV.bake(p) for p in sorted(glob.glob(os.path.join(paths.CLIPS, "*.usda")))]
    mesh = R("qmini_mesh.json")
    html = (R(os.path.basename(TEMPLATE))
            .replace("__CLIPS__", json.dumps(clips, separators=(",", ":")))
            .replace("__MESH__", mesh)
            .replace("__INPUT__", R("input_gamepad.js")))
    out = os.path.join(paths.ensure(paths.DIST), "gait_blend.html")
    open(out, "w").write(html)
    print(f"wrote {out}  ({len(html)/1024:.0f} KB, {len(clips)} clips)")

if __name__ == "__main__":
    main()
