"""
stylized_gait -- procedural walking gaits for the QMini biped, following

    Hopkins, Wiedebach, Cesare, Bishop, Knoop, Bacher,
    "Interactive Design of Stylized Walking Gaits for Robotic Characters",
    ACM TOG 2024 (Disney Research / WDI R&D).

Pipeline
    samples.py            authored Pi, the nine Tab. 3 key velocities
    overrides.py          hand tuning layered on top (from the HTML editor)
    blender_io.py         Pi <-> USD, for tuning in Blender
    library.py            the compiled, blend-ready intermediate
    blend.py              Sec. 5.2 parameter blending
    planner.py            Sec. 4.2 planner + 5-DoF leg IK
    usd_writer.py         phase-tagged UsdSkel clips
    verify.py             independent re-parse and constraint checks
"""

__version__ = "0.1.0"

from . import paths  # noqa: F401
