.PHONY: help install clips params web library isaac test check clean
PY ?= python3

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

install:            ## editable install with every extra
	$(PY) -m pip install -e ".[usd,plot,dev]"

clips:              ## regenerate data/clips/*.usda from samples.py (+ overrides)
	$(PY) scripts/generate.py

clips-library:      ## regenerate clips from the Blender round-trip library
	$(PY) scripts/generate.py --library data/gait_library.json

params:             ## data/gait_params.json for the browser tools
	$(PY) scripts/export_params.py

isaac:              ## data/qmini_step_in_place.npz for Isaac Lab
	$(PY) scripts/export_isaac_motion.py

blender-export:     ## data/blender_edit/*.usda for tuning in Blender
	$(PY) -m stylized_gait.blender_io export

blender-import:     ## compile edited USD back into data/gait_library.json
	$(PY) -m stylized_gait.blender_io import

web:                ## rebuild the four standalone HTML tools into dist/
	$(PY) scripts/make_viewer.py
	$(PY) scripts/make_blend_viewer.py
	$(PY) scripts/make_param_viewer.py
	$(PY) scripts/make_editor.py

check:              ## re-parse the clips and check every invariant
	$(PY) -m stylized_gait.verify data/clips

test:               ## the full suite
	$(PY) -m pytest -q

clean:
	rm -rf dist/*.html __pycache__ **/__pycache__ .pytest_cache
