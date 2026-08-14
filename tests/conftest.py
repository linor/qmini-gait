"""Make the package importable straight from a clone, no install required."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("QMINI_GAIT_ROOT", ROOT)

import pytest


@pytest.fixture(scope="session")
def gaits():
    from stylized_gait import samples as SM
    return SM.build()


@pytest.fixture(scope="session")
def solved(gaits):
    """every sample run through the planner once -- the expensive fixture"""
    from stylized_gait import planner as GP
    return {k: GP.solve_sample(v, fps=60, passes=3) for k, v in gaits.items()}
