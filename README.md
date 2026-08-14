# stylized_gait — procedural walking gaits for QMini

An implementation of Hopkins et al., *"Interactive Design of Stylized Walking
Gaits for Robotic Characters"* (ACM TOG 2024), targeting the QMini bipedal
droid, with a Blender round-trip for hand tuning and an Isaac Lab hand-off.

## Quick start

```bash
pip install -e ".[usd,plot,dev]"
make test
```

`make` on its own lists every target.

```bash
make clips      # regenerate data/clips/*.usda from samples.py
make check      # re-parse them and verify every invariant
make web        # rebuild the standalone HTML tools into dist/
make isaac      # data/qmini_step_in_place.npz
```

Everything in `data/` is generated and reproducible; nothing is precious.

## Try it in a browser

Open any of these directly — single files, no server, no build step:

| file | what it does |
|---|---|
| `dist/gait_viewer.html` | play back one clip on the real robot geometry |
| `dist/gait_blend.html` | keyboard-driven **pose** blending (kept as the counter-example) |
| `dist/gait_param_blend.html` | keyboard + gamepad **parameter** blending — the correct one |
| `dist/gait_editor.html` | tune attributes and curves, export a sparse override JSON |

Run `make web` after changing anything in `web/`.

## Hand tuning

Two routes, and they meet in the same place.

**In the browser.** Edit in `dist/gait_editor.html`, save the override JSON to
`data/gait_overrides.json`, then `make clips`. Overrides are sparse — anything
you do not mention keeps its authored value.

**In Blender.**

```bash
make blender-export      # data/blender_edit/*.usda
# open, tune the rig under /Gait/Rig, export back over the same file
make blender-import      # compiles to data/gait_library.json
make clips-library       # regenerate the clips from it
```

Only `/Gait/Rig` is editable; `/Gait/Reference` is a skinned robot for looking
at. What travels in the USD is deliberately minimal: the Tab. 2 curves as
transform animation, plus the nine scalars the planner cannot recover from
them. Anything that merely *constructs* a curve is left out, so an edited curve
can never disagree with a stale scalar.

`gait_library.json` is the hand-off point. Once it exists, blending, clip
generation and the Isaac export all read it and never touch USD again.

> **Blender is untested.** The round-trip was verified by rewriting the USD in
> code. That proves the reader and writer agree; it does not prove Blender
> preserves the rig. See `CLAUDE.md`.

## Isaac Lab

`isaac/` holds `motion_player.py` and `qmini_leg_env.py`. Copy them into your
Isaac Lab tasks directory alongside `data/qmini_step_in_place.npz`.

The `.npz` is in the robot's own joint order with PhysX sign conventions already
applied — see the `convention` field inside it, and the sign discussion in
`CLAUDE.md` before touching anything about joint directions.

## Layout

```
src/stylized_gait/   the library
scripts/             CLI entry points (run these via make)
web/                 JS + HTML templates → dist/
isaac/               Isaac Lab integration
data/                generated artefacts + the source robot USD
tests/               see tests/README.md
docs/GAIT_NOTES.md   long-form design notes
CLAUDE.md            context for Claude Code — read this first
```

## Requirements

Python ≥3.9 and numpy. Optional: `usd-core` (Blender round-trip, reads `.usdc`),
`matplotlib` (docs figures), `torch` (Isaac player), `node` (browser parity
test), `pytest`.

Without `usd-core` the round-trip falls back to a hand-written ASCII reader that
handles `.usda` only — which is the path that has actually been exercised.
