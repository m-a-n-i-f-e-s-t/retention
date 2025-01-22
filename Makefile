# Get version from pyproject.toml
VERSION := $(shell python scripts/get_version.py)
PACKAGE_NAME := power-attention
# Find Python 3.11+
PYTHON := $(shell for py in python3.12 python3.11 python3 python; do \
    if command -v $$py >/dev/null && $$py --version 2>&1 | grep -q "Python 3.1[1-9]"; then \
        echo $$py; \
        break; \
    fi \
done)

ifeq ($(PYTHON),)
    $(error Python 3.11 or higher is required. Please install Python 3.11+)
endif

.PHONY: dev venv deps test benchmark clean check-version check-test-version release release-test help

# Allow overriding venv path through environment variable, default to .venv
VENV_DIR ?= $(if $(POWER_ATTENTION_VENV_PATH),$(POWER_ATTENTION_VENV_PATH),.venv)
PIP := $(VENV_DIR)/bin/pip
PYTEST := $(VENV_DIR)/bin/pytest

define get_deps
$(VENV_DIR)/bin/python -c 'import tomllib; print("\n".join(tomllib.load(open("pyproject.toml", "rb"))["dependency-groups"]["$(1)"]))'
endef

$(VENV_DIR)/.deps_venv: # Ensure venv is created and make venv is idempotent
	@echo "Creating virtual environment using $(PYTHON) ($(shell $(PYTHON) --version 2>&1))"
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	tourch $@
$(VENV_DIR)/.deps_test: $(VENV_DIR)/.deps_venv
	@$(call get_deps,test) | $(PIP) install -r /dev/stdin && touch $@

venv: $(VENV_DIR)/.deps_venv
deps-test: venv $(VENV_DIR)/.deps_test

# Development commands
dev:
	CC=gcc CXX=g++ $(PIP) install -e .[dev]
build:
	$(PIP) install .
test: deps-test
	$(PYTEST) perf/tests


# Clean and check
clean:
	rm -rf dist/ build/ *.egg-info/ *.so wheelhouse/ $(VENV_DIR)/.deps_*

# Version checking
check-version:
	@echo "Local version: $(VERSION)"
	@python scripts/version_check.py "$(VERSION)" "$(PACKAGE_NAME)"

check-test-version:
	@echo "Local version: $(VERSION)"
	@python scripts/version_check.py "$(VERSION)" "$(PACKAGE_NAME)" --test

# Release commands
release: clean check-version
	@echo "Building wheels with cibuildwheel..."
	python -m cibuildwheel --output-dir dist
	python -m twine check dist/*
	@echo "Uploading to PyPI..."
	python -m twine upload dist/*
	@echo "Release $(VERSION) completed!"

release-test: clean check-test-version
	@echo "Building wheels with cibuildwheel..."
	python -m cibuildwheel --output-dir dist
	python -m twine check dist/*
	@echo "Uploading to TestPyPI..."
	python -m twine upload --repository testpypi dist/*
	@echo "Test release $(VERSION) completed!"

# Visualization
plot-regressions:
	@echo "Generating regression visualization..."
	$(PYTHON) perf/plot_regressions.py

# Help
help:
	@echo "Available commands:"
	@echo "  make venv          - Create virtual environment"
	@echo "  make dev           - Build and install in editable mode"
	@echo "  make build         - Build and install"
	@echo "  make test          - Run tests"
	@echo "  make benchmark     - Run benchmarks"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make release       - Release to PyPI (includes version check)"
	@echo "  make release-test  - Release to TestPyPI"
	@echo "  make check-version - Check version against PyPI"
	@echo "  make check-test-version - Check version against TestPyPI"
	@echo "  make plot-regressions  - Generate interactive regression visualization"
	@echo "Current version: $(VERSION)" 
