.PHONY: install test demo bench report real
install:  ; pip install -e ".[dev]"
test:     ; pytest -q
demo:     ; python -m driftguard demo
bench:    ; python -m driftguard bench && python -m driftguard report
report:   ; python -m driftguard report
real:     ; python scripts/run_real_study.py --download
