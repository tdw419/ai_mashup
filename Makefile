# ---- knobs ----
PY?=python3
VENV?=.venv
PIP?=$(VENV)/bin/pip
PYBIN?=$(VENV)/bin/python
PKG_DEPS?=lancedb>=0.10 "pydantic<3" requests pyyaml pytest pre-commit

# prompt/output (overridable): make run PROMPT="Implement CLICK_AT opcode in PixelRunner" OUT=docs/decisions/click_at.md
PROMPT?=Implement CLICK_AT opcode in PixelRunner
OUT?=docs/decisions/click_at.md
GLOB?=history/**/*.*

# ---- bootstrap ----
.PHONY: venv
venv:
	@test -d $(VENV) || ($(PY) -m venv $(VENV) && echo "✓ venv created")
	@$(PIP) -q install --upgrade pip
	@$(PIP) -q install $(PKG_DEPS)
	@echo "✓ deps installed"

# ---- indexing (RAG history) ----
.PHONY: index
index: venv
	@$(PYBIN) scripts/index_history.py --glob "$(GLOB)"

# ---- run one builder pass ----
.PHONY: run
run: venv
	@mkdir -p $(dir $(OUT))
	@$(PYBIN) tools/run_builder.py --prompt "$(PROMPT)" --out "$(OUT)"

# ---- tests & gate ----
.PHONY: test
test: venv
	@$(PYBIN) -m pytest -q

.PHONY: test-lancedb
test-lancedb: venv
	@$(PYBIN) scripts/test_lancedb.py

.PHONY: gate
gate: venv
	@# Audit all decision docs; fail if any do not pass the builder gate
	@files=$$(ls docs/decisions/*.md 2>/dev/null || true); \
	if [ -z "$$files" ]; then \
		echo "No decision docs found in docs/decisions/. Skipping gate."; \
	else \
		for f in $$files; do \
			echo "Gate: $$f"; \
			$(PYBIN) tools/gate_check.py --file "$$f"; \
		done; \
	fi
	@echo "✓ gate passed"

# ---- full CI bundle ----
.PHONY: ci
ci: test gate

# ---- pre-commit setup ----
.PHONY: hooks
hooks: venv
	@$(VENV)/bin/pre-commit install
	@echo "✓ pre-commit installed"

# ---- convenience clean ----
.PHONY: clean
clean:
	@rm -rf .pytest_cache .mypy_cache **/__pycache__
