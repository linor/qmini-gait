"""make_editor.py -- bake the gait editor into one standalone HTML."""
import os

from stylized_gait import paths


def _asset(name):
    """web sources live in web/, generated data in data/"""
    for d in (paths.WEB, paths.DATA, paths.DIST):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"{name} not found in web/ or data/")

HERE = paths.DATA
def R(n):
    for d in (paths.WEB, paths.DATA):
        q = os.path.join(d, n)
        if os.path.isfile(q):
            return open(q).read()
    raise FileNotFoundError(n)

def main():
    html = (R("editor_template.html")
            .replace("__PARAMS__", R("gait_params.json"))
            .replace("__MESH__",   R("qmini_mesh.json"))
            .replace("__CORE__",   R("param_core.js"))
            .replace("__CURVES__", R("gait_curves.js"))
            .replace("__STEP__",   R("param_step.js"))
            .replace("__INPUT__",  R("input_gamepad.js"))
            .replace("__EDITOR__", R("gait_editor.js")))
    out = os.path.join(paths.ensure(paths.DIST), "gait_editor.html")
    open(out, "w").write(html)
    print(f"wrote {out}  ({len(html)/1024:.0f} KB)")

if __name__ == "__main__":
    main()
