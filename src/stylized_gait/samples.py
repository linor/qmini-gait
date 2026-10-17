"""
samples.py -- the 9 key gait samples of Tab. 3, authored for the QMini biped
in a neutral / mechanically clean style.

    Forward Fast   ( xmax,  0,     0    )
    Forward Slow   ( xslow, 0,     0    )
    Step In Place  ( 0,     0,     0    )
    Reverse Slow   (-xslow, 0,     0    )
    Reverse Fast   (-xmax,  0,     0    )
    Left  Strafe   ( 0,     ymax,  0    )
    Right Strafe   ( 0,    -ymax,  0    )
    Left  Turn     ( 0,     0,     tmax )
    Right Turn     ( 0,     0,    -tmax )
"""

import numpy as np
from .core import GaitSample, NOMINAL_HALF_W, NOMINAL_HEIGHT

D = np.deg2rad

# --- key velocities ---------------------------------------------------------
X_MAX   = 0.35     # m/s
X_SLOW  = 0.15     # m/s
Y_MAX   = 0.16     # m/s   HARD LIMIT: lateral motion comes only from the
                   #       +/-15 deg hip roll (this robot has no ankle roll).
                   #       0.20 m/s was requested but saturates the roll joint.
TH_MAX  = 0.60     # rad/s

# --- shared timing: 35 % swing / 15 % double support, symmetric -------------
F_LTD, F_RLO, F_RTD = 0.35, 0.50, 0.85

W0 = 2 * NOMINAL_HALF_W          # 0.3170 m rest stance width
Z0 = NOMINAL_HEIGHT              # 0.4241 m base_link height at rest


# Applied to every sample by build() (2026-10-17, for the RL gait library):
# PELVIS_SWAY = lateral pelvis translation toward the support foot, metres
# (core.GaitSample.c_pelvis_y); PELVIS_ROLL_SCALE scales the authored pelvis
# lean. Module-level rather than build() arguments so overrides.apply(), which
# re-runs build() to recover the authored arguments, sees the same values.
# The defaults reproduce the authored gaits exactly.
PELVIS_SWAY = 0.0
PELVIS_ROLL_SCALE = 1.0


def build(y_max=Y_MAX, th_max=TH_MAX):
    S = []

    def GS(*a, **k):
        # resolves GaitSample at call time, so overrides._authored_args' spy
        # still sees every sample
        k["pelvis_roll"] = k["pelvis_roll"] * PELVIS_ROLL_SCALE
        k.setdefault("pelvis_sway", PELVIS_SWAY)
        return GaitSample(*a, **k)

    S.append(GS(
        "forward_fast", (X_MAX, 0.0, 0.0), T=0.80,
        f_ltd=0.36, f_rlo=0.50, f_rtd=0.86,
        stance_width=W0 - 0.010, stance_x_bias=0.000, stance_splay=D(0.0),
        swing_height=0.048, psi_toe_lo=D(16.0), psi_heel_td=D(-10.0),
        dt_psi_lo=0.10, dt_psi_td=0.09,
        pelvis_z=Z0 - 0.015, pelvis_bounce=0.009,
        pelvis_roll=D(2.5), pelvis_pitch=D(2.0),
        swing_x_lead=0.014, swing_y_lead=0.0, foot_yaw_amp=D(0.0),
        notes="Tab. 3 Forward Fast. Longest stride, most pronounced toe-off."))

    S.append(GS(
        "forward_slow", (X_SLOW, 0.0, 0.0), T=1.00,
        f_ltd=F_LTD, f_rlo=F_RLO, f_rtd=F_RTD,
        stance_width=W0, stance_x_bias=0.000, stance_splay=D(0.0),
        swing_height=0.036, psi_toe_lo=D(9.0), psi_heel_td=D(-5.0),
        dt_psi_lo=0.12, dt_psi_td=0.11,
        pelvis_z=Z0 - 0.012, pelvis_bounce=0.006,
        pelvis_roll=D(2.0), pelvis_pitch=D(1.0),
        swing_x_lead=0.008,
        notes="Tab. 3 Forward Slow."))

    S.append(GS(
        "step_in_place", (0.0, 0.0, 0.0), T=1.10,
        f_ltd=0.34, f_rlo=0.50, f_rtd=0.84,
        stance_width=W0, stance_x_bias=0.000, stance_splay=D(0.0),
        swing_height=0.030, psi_toe_lo=D(0.0), psi_heel_td=D(0.0),
        dt_psi_lo=0.10, dt_psi_td=0.10,
        pelvis_z=Z0 - 0.010, pelvis_bounce=0.005,
        pelvis_roll=D(2.0), pelvis_pitch=D(0.0),
        notes="Tab. 3 Step In Place. Flat-foot contacts, no heel/toe."))

    S.append(GS(
        "reverse_slow", (-X_SLOW, 0.0, 0.0), T=1.05,
        f_ltd=F_LTD, f_rlo=F_RLO, f_rtd=F_RTD,
        stance_width=W0, stance_x_bias=-0.010, stance_splay=D(0.0),
        swing_height=0.036, psi_toe_lo=D(0.0), psi_heel_td=D(0.0),
        dt_psi_lo=0.12, dt_psi_td=0.12,
        pelvis_z=Z0 - 0.012, pelvis_bounce=0.005,
        pelvis_roll=D(2.0), pelvis_pitch=D(-1.0),
        swing_x_lead=-0.006,
        notes="Tab. 3 Reverse Slow. Flat foot: no toe-off when walking backward."))

    S.append(GS(
        "reverse_fast", (-X_MAX, 0.0, 0.0), T=0.85,
        f_ltd=0.36, f_rlo=0.50, f_rtd=0.86,
        stance_width=W0, stance_x_bias=-0.020, stance_splay=D(0.0),
        swing_height=0.044, psi_toe_lo=D(0.0), psi_heel_td=D(-4.0),
        dt_psi_lo=0.11, dt_psi_td=0.10,
        pelvis_z=Z0 - 0.012, pelvis_bounce=0.007,
        pelvis_roll=D(2.5), pelvis_pitch=D(-2.5),
        swing_x_lead=-0.012,
        notes="Tab. 3 Reverse Fast."))

    for side, sgn in (("left", +1.0), ("right", -1.0)):
        S.append(GS(
            f"{side}_strafe", (0.0, sgn * y_max, 0.0), T=0.78,
            f_ltd=0.34, f_rlo=0.50, f_rtd=0.84,
            stance_width=W0 + 0.006, stance_x_bias=0.0, stance_splay=D(0.0),
            swing_height=0.040, psi_toe_lo=D(0.0), psi_heel_td=D(0.0),
            dt_psi_lo=0.11, dt_psi_td=0.11,
            pelvis_z=Z0 - 0.012, pelvis_bounce=0.005,
            pelvis_roll=D(1.5), pelvis_pitch=D(0.0),
            # NOT sgn * 0.008: GaitSample already mirrors swing_y_lead per side
            # (+ for L, - for R), so a signed value made right_strafe the
            # opposite of a mirror image of left_strafe (3.3 deg roll mismatch;
            # found 2026-10-14 via the half-cycle mirror check)
            swing_y_lead=0.008,
            notes=f"Tab. 3 {side.capitalize()} Strafe. Lateral motion comes only "
                  f"from the +/-15 deg hip roll (no ankle roll on this robot)."))

    for side, sgn in (("left", +1.0), ("right", -1.0)):
        S.append(GS(
            f"{side}_turn", (0.0, 0.0, sgn * th_max), T=1.00,
            f_ltd=F_LTD, f_rlo=F_RLO, f_rtd=F_RTD,
            # unsigned for the same reason as swing_y_lead in the strafe samples:
            # stance_splay and foot_yaw_amp are mirrored per side by GaitSample,
            # and sgn * made the right turn toe-in where the left turn is
            # toe-out (14 deg hip-yaw mirror mismatch; 2026-10-14)
            stance_width=W0, stance_x_bias=0.0, stance_splay=D(3.0),
            swing_height=0.038, psi_toe_lo=D(5.0), psi_heel_td=D(-3.0),
            dt_psi_lo=0.12, dt_psi_td=0.11,
            pelvis_z=Z0 - 0.012, pelvis_bounce=0.005,
            pelvis_roll=D(2.0), pelvis_pitch=D(0.5),
            foot_yaw_amp=D(4.0),
            notes=f"Tab. 3 {side.capitalize()} Turn. Stance splay biased into "
                  f"the turn; pelvis yaw follows the path frame."))

    return {s.name: s for s in S}


ORDER = ["forward_fast", "forward_slow", "step_in_place",
         "reverse_slow", "reverse_fast",
         "left_strafe", "right_strafe", "left_turn", "right_turn"]
