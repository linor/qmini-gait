"""The browser must rebuild the curves the way Python does.

gait_curves.js is a hand port of GaitSample.__init__.  Two independent
implementations of the same maths drift, and when they drift the editor shows
the animator one shape while the generator bakes another.

This check has already failed once for real, and the way it failed is the
reason it is written the way it is: `_authored_args` only captured explicitly
passed keyword arguments, so `swing_y_lead` and `foot_yaw_amp` reached the
browser as `undefined`.  Every curve built from them became NaN -- and the NaN
went unnoticed because the comparison was `if (diff > worst)`, which is false
for NaN.  Hence: count non-finite results explicitly and fail on them.
"""
import json
import os
import shutil
import subprocess

import pytest

from stylized_gait import paths

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

DRIVER = r"""
const fs = require('fs');
eval(fs.readFileSync(process.argv[2], 'utf8'));
const P = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));

function evalSpline(sp, x) {
  const t = sp.t, v = sp.v, m = sp.m, n = t.length;
  if (sp.periodic) { const T = t[n-1]-t[0]; x = t[0] + ((x - t[0]) % T + T) % T; }
  x = Math.min(Math.max(x, t[0]), t[n-1]);
  let i = 0; while (i < n-2 && x > t[i+1]) i++;
  const h = t[i+1]-t[i], s = (x-t[i])/h, s2 = s*s, s3 = s2*s;
  return (2*s3-3*s2+1)*v[i] + (s3-2*s2+s)*h*m[i]
       + (-2*s3+3*s2)*v[i+1] + (s3-s2)*h*m[i+1];
}

let worst = 0, nonFinite = 0, checked = 0, worstAt = "";
for (const name of Object.keys(P.samples)) {
  const S = P.samples[name];
  const built = buildCurves(S.args).curves;
  for (const cname of Object.keys(S.curves)) {
    const ref = S.curves[cname];
    const sides = ref.L ? ["L","R"] : [null];
    for (const side of sides) {
      const r = side ? ref[side] : ref;
      const b = side ? built[cname][side] : built[cname];
      if (!b) continue;
      const a = r.t[0], z = r.t[r.t.length-1];
      for (let k = 0; k <= 64; k++) {
        const x = a + (z-a)*k/64;
        const p = evalSpline(r, x), q = evalSpline(b, x);
        checked++;
        if (!isFinite(p) || !isFinite(q)) { nonFinite++; continue; }
        const d = Math.abs(p-q);
        if (d > worst) { worst = d; worstAt = name+"/"+cname+(side||""); }
      }
    }
  }
}
console.log(JSON.stringify({worst, nonFinite, checked, worstAt}));
"""


def test_js_rebuilds_the_same_curves(tmp_path):
    assert os.path.isfile(paths.PARAMS_JSON), "run scripts/export_params.py first"
    drv = tmp_path / "driver.js"
    drv.write_text(DRIVER)
    out = subprocess.run(
        [NODE, str(drv), os.path.join(paths.WEB, "gait_curves.js"), paths.PARAMS_JSON],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout)

    assert r["checked"] > 1000, f"only {r['checked']} samples compared"
    # the NaN that hid last time would show up here and nowhere else
    assert r["nonFinite"] == 0, \
        f"{r['nonFinite']} of {r['checked']} comparisons were NaN or Inf"
    # 1e-6 rad is ~6e-5 deg.  The floor is not the maths: gait_params.json
    # stores control points rounded to 8 decimals, while buildCurves works
    # from the full-precision attribute, which alone accounts for ~5e-9.
    # Anything approaching 1e-6 is a genuine port divergence.
    assert r["worst"] < 1e-6, \
        f"browser and Python disagree by {r['worst']:.3e} at {r['worstAt']}"
