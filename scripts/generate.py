import os, sys, numpy as np
from stylized_gait.planner import solve_sample
from stylized_gait.usd_writer import write_usda
from stylized_gait import samples as SM
from stylized_gait import overrides as GO
from stylized_gait import paths

OUT = paths.ensure(paths.CLIPS)
os.makedirs(OUT, exist_ok=True)

# Sources, in order of preference:
#   python3 generate.py --library gait_library.json   <- Blender round-trip
#   python3 generate.py [gait_overrides.json]         <- HTML editor overrides
#   python3 generate.py                               <- authored samples.py
if "--library" in sys.argv:
    from stylized_gait import library as GL
    _lib = sys.argv[sys.argv.index("--library") + 1]
    g = GL.load(_lib)
    print(f"using {_lib} ({len(g)} samples)")
else:
    _ov = sys.argv[1] if len(sys.argv) > 1 else paths.OVERRIDES_JSON
    g = GO.apply(SM.build(), _ov if os.path.isfile(_ov) else None)
for i, name in enumerate(SM.ORDER, 1):
    s = g[name]
    r = solve_sample(s, fps=60, passes=3)
    p = os.path.join(OUT, f"{i:02d}_{name}.usda")
    write_usda(r, p, fps=60)
    print(f"  {os.path.basename(p):28s} {r['n']+1:3d} frames  T={s.T:.2f}s  v={tuple(np.round(s.vel,3))}")
