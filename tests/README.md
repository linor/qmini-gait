# Tests

    make test          # or: python3 -m pytest -q

Nothing here needs an install; `conftest.py` puts `src/` and `scripts/` on the
path. Some tests read generated artefacts, so if `data/` is empty run
`make clips params isaac` first.

| file | guards |
|---|---|
| `test_clips.py` | the nine clips satisfy every constraint, re-parsed from the `.usda` independently of the generator |
| `test_blend.py` | a parameter blend is *itself* a valid clip; pose blending measurably is not |
| `test_overrides.py` | hand tuning reaches the curves, and invalid overrides are rejected rather than ignored |
| `test_blender_roundtrip.py` | export → edit → import preserves the plan and carries the edit |
| `test_web_parity.py` | `gait_curves.js` rebuilds the same curves as Python (needs `node`) |
| `test_isaac.py` | the `.npz` sign convention, periodicity and sampler consistency (needs `torch` for the last one) |

**Skipped by default:** `test_backend_agreement` and `test_stage_validates` need
`pxr` (`pip install usd-core`). They have never run — see the note in
`CLAUDE.md`. They are the first thing to try on a machine that has it.

Each assertion corresponds to a bug that actually occurred. The comments say
which. Please don't loosen a tolerance without reading the comment above it.
