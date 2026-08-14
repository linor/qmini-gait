/* =====================================================================
   gait_curves.js -- the Tab. 2 function curves, rebuilt from the Sec. 4.3
   attributes.

   This is a direct port of GaitSample.__init__ in gait_core.py.  The editor
   needs it so that dragging an attribute slider reshapes every derived curve
   the same way the Python generator would, rather than the browser and the
   generator drifting apart.  build_curves() is checked against the exported
   gait_params.json at startup.
   ===================================================================== */

/* cyclic / natural Catmull-Rom tangents, matching HermiteSpline._auto_tangents */
function autoTangents(t, v, periodic) {
  const n = t.length, m = new Array(n).fill(0);
  if (n === 1) return m;
  if (periodic) {
    const T = t[n - 1] - t[0], k = n - 1;      // last knot duplicates the first
    for (let i = 0; i < k; i++) {
      const im = (i - 1 + k) % k;
      const dtNext = t[i + 1] - t[i];
      const dtPrev = im < i ? t[i] - t[im] : t[i] + T - t[im];
      const dvNext = v[i + 1] - v[i];
      const dvPrev = v[i] - v[im];
      m[i] = 0.5 * (dvNext / dtNext + dvPrev / dtPrev);
    }
    m[n - 1] = m[0];
  } else {
    m[0] = (v[1] - v[0]) / (t[1] - t[0]);
    m[n - 1] = (v[n - 1] - v[n - 2]) / (t[n - 1] - t[n - 2]);
    for (let i = 1; i < n - 1; i++)
      m[i] = 0.5 * ((v[i + 1] - v[i]) / (t[i + 1] - t[i])
                  + (v[i] - v[i - 1]) / (t[i] - t[i - 1]));
  }
  return m;
}
const SP = (t, v, m, periodic) => ({
  t, v, m: m || autoTangents(t, v, !!periodic), periodic: !!periodic });

/* --- the individual curve shapes (gait_core.py, GaitSample.__init__) --- */
function swingCurve(a, b, peak, at, tan0, tan1) {
  at = at === undefined ? 0.5 : at;
  return SP([a, a + (b - a) * at, b], [0, peak, 0], [tan0 || 0, 0, tan1 || 0]);
}
function swingZ(a, b, h) {
  const d = b - a;
  return SP([a, a + 0.28 * d, a + 0.45 * d, a + 0.75 * d, b],
            [0, 0.86 * h, h, 0.62 * h, 0],
            [3.4 * h / d, 1.2 * h / d, 0, -1.6 * h / d, -1.35 * h / d]);
}
const toeCurve  = (a, b, v0) => SP([a, a + 0.35 * (b - a), b], [v0, 0, 0], [0, 0, 0]);
const heelCurve = (a, b, v1) => SP([a, a + 0.60 * (b - a), b], [0, 0, v1], [0, 0, 0]);

/* --- everything a sample's curves are derived from --- */
function buildCurves(Ain) {
  /* every attribute must be present: a missing one would silently become
     undefined and turn the whole curve into NaN */
  const A = Object.assign({
    swing_x_lead: 0, swing_y_lead: 0, foot_yaw_amp: 0,
    psi_toe_lo: 0, psi_heel_td: 0, pelvis_bounce: 0, pelvis_roll: 0,
    pelvis_pitch: 0, swing_height: 0,
  }, Ain);
  for (const k in A) if (typeof A[k] === "number" && !isFinite(A[k]))
    throw new Error("buildCurves: attribute '" + k + "' is " + A[k]);
  const T = A.T;
  const tLLO = 0, tLTD = A.f_ltd * T, tRLO = A.f_rlo * T, tRTD = A.f_rtd * T;
  const midLS = 0.5 * (tLLO + tLTD), midRS = 0.5 * (tRLO + tRTD);
  const P = [tLLO, midLS, tRLO, midRS, T];
  const c = {};
  c.foot_z = { L: swingZ(tLLO, tLTD, A.swing_height),
               R: swingZ(tRLO, tRTD, A.swing_height) };
  c.foot_x = { L: swingCurve(tLLO, tLTD, A.swing_x_lead, 0.55),
               R: swingCurve(tRLO, tRTD, A.swing_x_lead, 0.55) };
  c.foot_y = { L: swingCurve(tLLO, tLTD,  A.swing_y_lead, 0.5),
               R: swingCurve(tRLO, tRTD, -A.swing_y_lead, 0.5) };
  c.foot_yaw = { L: swingCurve(tLLO, tLTD,  A.foot_yaw_amp, 0.5),
                 R: swingCurve(tRLO, tRTD, -A.foot_yaw_amp, 0.5) };
  c.foot_gamma = { L: swingCurve(tLLO, tLTD, 0, 0.5),
                   R: swingCurve(tRLO, tRTD, 0, 0.5) };
  c.psi_toe  = { L: toeCurve(tLLO, tLTD, A.psi_toe_lo),
                 R: toeCurve(tRLO, tRTD, A.psi_toe_lo) };
  c.psi_heel = { L: heelCurve(tLLO, tLTD, A.psi_heel_td),
                 R: heelCurve(tRLO, tRTD, A.psi_heel_td) };
  const b = A.pelvis_bounce, r = A.pelvis_roll, p = A.pelvis_pitch;
  c.pelvis_z     = SP(P, [-b,  b, -b,  b, -b], null, true);
  c.pelvis_roll  = SP(P, [0,   r,  0,  -r,  0], null, true);
  c.pelvis_pitch = SP(P, [p,   p,  p,   p,  p], null, true);
  c.pelvis_yaw   = SP(P, [0,   0,  0,   0,  0], null, true);
  return { curves: c, events: [tLLO, tLTD, tRLO, tRTD, T] };
}

/* --- normalised <-> absolute control points ------------------------------
   Knots are stored in the override file at x in 0..1 across the curve's own
   interval, with tangents per unit x.  A later change to T or to a contact
   event then rescales an edited curve instead of stranding it.            */
function curveInterval(A, name, side) {
  const T = A.T;
  if (name.startsWith("pelvis")) return [0, T];
  return side === "L" ? [0, A.f_ltd * T] : [A.f_rlo * T, A.f_rtd * T];
}
function toNorm(sp, a, b) {
  const d = b - a;
  return { x: sp.t.map(t => (t - a) / d), v: sp.v.slice(), m: sp.m.map(m => m * d) };
}
function fromNorm(cp, a, b, periodic) {
  const d = b - a;
  return { t: cp.x.map(x => a + x * d), v: cp.v.slice(),
           m: cp.m.map(m => m / d), periodic: !!periodic };
}
