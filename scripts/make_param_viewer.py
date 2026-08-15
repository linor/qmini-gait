"""make_param_viewer.py -- bake the parameter-blending viewer into one HTML."""
import os, json
from stylized_gait import paths


def R(name):
    return open(_asset(name)).read()


def _asset(name):
    """web sources live in web/, generated data in data/"""
    for d in (paths.WEB, paths.DATA, paths.DIST):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"{name} not found in web/ or data/")

HERE = paths.DATA

def main():
    html = open(_asset("param_template.html")).read()
    html = (html
        .replace("__PARAMS__", R("gait_params.json"))
        .replace("__MESH__",   R("qmini_mesh.json"))
        .replace("__CORE__",   R("param_core.js"))
        .replace("__STEP__",   R("param_step.js"))
        .replace("__INPUT__",  R("input_gamepad.js")))
    out = os.path.join(paths.ensure(paths.DIST), "gait_param_blend.html")
    open(out, "w").write(html)
    print(f"wrote {out}  ({len(html)/1024:.0f} KB)")

if __name__ == "__main__":
    main()
