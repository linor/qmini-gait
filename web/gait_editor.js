/* =====================================================================
   gait_editor.js -- hand tuning on top of the authored samples.

   Everything edited here stays inside Pi: the Sec. 4.3 attributes, the Tab. 1
   time intervals, and the control points of the Tab. 2 function curves.  The
   planner and the IK re-run on every change, so the contact constraints are
   re-imposed rather than assumed, and the constraint read-outs tell you
   immediately if an edit has made the gait infeasible.

   Edits are kept sparse -- only what you actually changed -- and exported as
   gait_overrides.json, which the Python generator layers on top of samples.py.
   ===================================================================== */

const EDIT = {
  sel: 0,                 // index into SAMPLES
  mode: "edit",           // "edit" | "blend"
  ov: {},                 // sample name -> {attrs:{}, curves:{}}
  curve: "foot_z.L",
  drag: null,
};

/* attribute sliders: key, label, min, max, step, unit, scale to display */
const DEG = 180 / Math.PI;
const ATTRS = [
  ["T",             "cycle period",   0.55, 1.60, 0.005, "s",   1],
  ["f_ltd",         "left TD",        0.10, 0.48, 0.005, "of T", 1],
  ["f_rlo",         "right LO",       0.30, 0.70, 0.005, "of T", 1],
  ["f_rtd",         "right TD",       0.55, 0.98, 0.005, "of T", 1],
  ["stance_width",  "stance width",   0.20, 0.40, 0.001, "cm",  100],
  ["stance_x_bias", "stance x bias", -0.06, 0.06, 0.001, "cm",  100],
  ["stance_splay",  "stance splay",  -0.26, 0.26, 0.002, "deg", DEG],
  ["swing_height",  "swing height",   0.00, 0.09, 0.001, "cm",  100],
  ["swing_x_lead",  "swing x lead",  -0.05, 0.05, 0.001, "cm",  100],
  ["swing_y_lead",  "swing y lead",  -0.05, 0.05, 0.001, "cm",  100],
  ["foot_yaw_amp",  "swing foot yaw",-0.35, 0.35, 0.002, "deg", DEG],
  ["psi_toe_lo",    "toe off",        0.00, 0.70, 0.002, "deg", DEG],
  ["psi_heel_td",   "heel strike",   -0.70, 0.00, 0.002, "deg", DEG],
  ["dt_psi_lo",     "toe ramp",       0.00, 0.30, 0.005, "s",   1],
  ["dt_psi_td",     "heel ramp",      0.00, 0.30, 0.005, "s",   1],
  ["pelvis_z",      "pelvis height",  0.34, 0.43, 0.001, "cm",  100],
  ["pelvis_bounce", "pelvis bounce",  0.00, 0.03, 0.0005,"cm",  100],
  ["pelvis_roll",   "pelvis lean",    0.00, 0.14, 0.002, "deg", DEG],
  ["pelvis_pitch",  "pelvis pitch",  -0.14, 0.14, 0.002, "deg", DEG],
];
const CURVE_LIST = [
  ["foot_z.L", "left foot z"],   ["foot_z.R", "right foot z"],
  ["foot_x.L", "left foot x"],   ["foot_x.R", "right foot x"],
  ["foot_y.L", "left foot y"],   ["foot_y.R", "right foot y"],
  ["foot_yaw.L", "left foot yaw"], ["foot_yaw.R", "right foot yaw"],
  ["foot_gamma.L", "left foot roll"], ["foot_gamma.R", "right foot roll"],
  ["psi_toe.L", "left toe pitch"], ["psi_toe.R", "right toe pitch"],
  ["psi_heel.L", "left heel pitch"], ["psi_heel.R", "right heel pitch"],
  ["pelvis_z", "pelvis z"], ["pelvis_roll", "pelvis roll"],
  ["pelvis_pitch", "pelvis pitch"], ["pelvis_yaw", "pelvis yaw"],
];
/* display scale + which endpoints are pinned, per curve family */
const CURVE_META = {
  foot_z:      { scale: 100, unit: "cm",  lock: "both" },
  foot_x:      { scale: 100, unit: "cm",  lock: "both" },
  foot_y:      { scale: 100, unit: "cm",  lock: "both" },
  foot_yaw:    { scale: DEG, unit: "deg", lock: "both" },
  foot_gamma:  { scale: DEG, unit: "deg", lock: "both" },
  psi_toe:     { scale: DEG, unit: "deg", lock: "end",  sign: +1 },
  psi_heel:    { scale: DEG, unit: "deg", lock: "start", sign: -1 },
  pelvis_z:    { scale: 100, unit: "cm",  lock: "periodic" },
  pelvis_roll: { scale: DEG, unit: "deg", lock: "periodic" },
  pelvis_pitch:{ scale: DEG, unit: "deg", lock: "periodic" },
  pelvis_yaw:  { scale: DEG, unit: "deg", lock: "periodic" },
};

/* ------------------------------------------------------------ sample state */
SAMPLES.forEach(s => {
  s.baseArgs = Object.assign({}, s.args);
  s.args = Object.assign({}, s.args);
});
const curSample = () => SAMPLES[EDIT.sel];
const ovFor = n => (EDIT.ov[n] = EDIT.ov[n] || { attrs: {}, curves: {} });

/* rebuild a sample in place from its args + explicit curve edits */
function rebuild(s) {
  const A = s.args;
  const built = buildCurves(A);
  s.T = A.T;
  [s.t_llo, s.t_ltd, s.t_rlo, s.t_rtd] = built.events;
  s.events = built.events.slice(0, 4).concat([A.T]);
  s.om = monoSpline(PHI, s.events);
  s.stance_width = A.stance_width;
  s.stance_x_bias = A.stance_x_bias;
  s.stance_splay = A.stance_splay;
  s.dt_psi_lo = A.dt_psi_lo;
  s.dt_psi_td = A.dt_psi_td;
  s.psi_toe_lo = A.psi_toe_lo;
  s.psi_heel_td = A.psi_heel_td;
  s.pelvis_z_nom = A.pelvis_z;

  s.cp = {};                         // control points actually in force
  s.c = {};
  for (const nm in built.curves) {
    const cv = built.curves[nm];
    if (cv.L) {
      s.c[nm] = { L: hermite(cv.L), R: hermite(cv.R) };
      s.cp[nm + ".L"] = cv.L; s.cp[nm + ".R"] = cv.R;
    } else {
      s.c[nm] = hermite(cv); s.cp[nm] = cv;
    }
  }
  const ov = EDIT.ov[s.name];
  if (ov) for (const key in ov.curves) {
    const [nm, side] = key.split(".");
    const iv = curveInterval(A, nm, side || "L");
    const sp = fromNorm(ov.curves[key], iv[0], iv[1], nm.startsWith("pelvis"));
    if (side) s.c[nm][side] = hermite(sp); else s.c[nm] = hermite(sp);
    s.cp[key] = sp;
  }
}
SAMPLES.forEach(rebuild);

/* ---- control points of the currently shown curve, in normalised coords ---- */
function curveCP(s, key) {
  const [nm, side] = key.split(".");
  const iv = curveInterval(s.args, nm, side || "L");
  return toNorm(s.cp[key], iv[0], iv[1]);
}
function setCurveCP(s, key, cp) {
  ovFor(s.name).curves[key] = { x: cp.x.slice(), v: cp.v.slice(), m: cp.m.slice() };
  rebuild(s);
}
function resetCurve(s, key) {
  const ov = EDIT.ov[s.name];
  if (ov) delete ov.curves[key];
  rebuild(s);
}
function setAttr(s, k, v) {
  s.args[k] = v;
  if (Math.abs(v - s.baseArgs[k]) < 1e-12) delete ovFor(s.name).attrs[k];
  else ovFor(s.name).attrs[k] = v;
  rebuild(s);
}
function resetSample(s) {
  delete EDIT.ov[s.name];
  s.args = Object.assign({}, s.baseArgs);
  rebuild(s);
}
const isEdited = s => {
  const o = EDIT.ov[s.name];
  return !!o && (Object.keys(o.attrs).length || Object.keys(o.curves).length);
};

/* ---------------------------------------------------- validity of the edits */
function validate(s) {
  const A = s.args, msgs = [];
  const f = [0, A.f_ltd, A.f_rlo, A.f_rtd, 1];
  for (let i = 0; i < 4; i++)
    if (f[i] >= f[i + 1]) msgs.push("contact events out of order");
  if (A.psi_heel_td > 1e-9) msgs.push("heel strike must be <= 0");
  if (A.psi_toe_lo < -1e-9) msgs.push("toe off must be >= 0");
  return msgs;
}

/* --------------------------------------------------------- import / export */
function buildOverrides() {
  const out = { version: 1, samples: {} };
  for (const s of SAMPLES) {
    const o = EDIT.ov[s.name];
    if (!o) continue;
    const attrs = {}, curves = {};
    for (const k in o.attrs) attrs[k] = +o.attrs[k].toFixed(9);
    for (const k in o.curves) {
      const c = o.curves[k];
      curves[k] = { x: c.x.map(v => +v.toFixed(6)),
                    v: c.v.map(v => +v.toFixed(8)),
                    m: c.m.map(v => +v.toFixed(8)) };
    }
    if (Object.keys(attrs).length || Object.keys(curves).length)
      out.samples[s.name] = { attrs, curves };
  }
  return out;
}
function exportOverrides() {
  const blob = new Blob([JSON.stringify(buildOverrides(), null, 2)],
                        { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "gait_overrides.json";
  a.click();
  URL.revokeObjectURL(a.href);
}
function importOverrides(text) {
  const d = JSON.parse(text);
  if (d.version !== 1) throw new Error("unsupported override version " + d.version);
  EDIT.ov = {};
  for (const s of SAMPLES) s.args = Object.assign({}, s.baseArgs);
  for (const name in d.samples || {}) {
    const s = BY[name];
    if (!s) throw new Error("override for unknown sample '" + name + "'");
    const e = d.samples[name];
    const o = ovFor(name);
    for (const k in e.attrs || {}) { o.attrs[k] = e.attrs[k]; s.args[k] = e.attrs[k]; }
    for (const k in e.curves || {}) o.curves[k] = e.curves[k];
  }
  SAMPLES.forEach(rebuild);
}
