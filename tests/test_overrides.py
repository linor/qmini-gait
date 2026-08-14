"""The hand-tuning layer.

An override file is sparse: it names a sample, an attribute or a curve, and a
new value.  Two failure modes matter.  A silently ignored override wastes the
animator's time; a silently accepted invalid one produces a clip that
violates the contact constraint.  Both are checked.
"""
import json

import numpy as np
import pytest

from stylized_gait import overrides as GO, samples as SM


def test_no_override_file_is_a_no_op():
    a = SM.build()
    b = GO.apply(SM.build(), None, verbose=False)
    for k in a:
        assert abs(a[k].T - b[k].T) < 1e-12
        assert np.allclose(a[k].vel, b[k].vel)


def test_attribute_override_reaches_the_curves():
    """changing swing_height must reshape foot_z, not just sit in a dict."""
    ov = {"forward_slow": {"attrs": {"swing_height": 0.10}}}
    g = GO.apply(SM.build(), ov, verbose=False)
    s = g["forward_slow"]
    peak = max(s.c_foot_z["L"](t) for t in np.linspace(0, s.t_ltd, 200))
    assert abs(peak - 0.10) < 1e-3, f"swing apex is {peak:.4f}, expected 0.10"


def test_unknown_sample_is_rejected():
    with pytest.raises((KeyError, ValueError)):
        GO.apply(SM.build(), {"99_does_not_exist": {"attrs": {}}}, verbose=False)


def test_unknown_attribute_is_rejected():
    with pytest.raises((KeyError, ValueError)):
        GO.apply(SM.build(), {"forward_slow": {"attrs": {"nonsense": 1.0}}},
                 verbose=False)


def test_every_attribute_has_a_default():
    """_authored_args fills defaults from the signature.  When it did not,
    two attributes reached the browser as `undefined` and every curve built
    from them became NaN -- and the NaN hid, because the round-trip check
    compared with `>` which is false for NaN."""
    args = GO._authored_args()
    for name, a in args.items():
        for k in GO.ATTR_KEYS:
            assert k in a, f"{name} is missing a value for {k}"
            assert np.isfinite(a[k]), f"{name}.{k} is {a[k]}"


def test_template_dump_roundtrips(tmp_path):
    """the template written for the editor must itself be valid input to
    apply(), or the editor's Save button produces a file the generator
    rejects."""
    p = str(tmp_path / "template.json")
    GO.dump_template(SM.build(), p)
    # both entry points must accept it: the path, and the parsed dict
    GO.apply(SM.build(), p, verbose=False)
    with open(p) as f:
        GO.apply(SM.build(), json.load(f), verbose=False)
