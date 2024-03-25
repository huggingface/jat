.PHONY: quality style test

# Define directories variable
DIRS = data examples jat scripts tests

# Check that source code meets quality standards
quality:
	black --check $(DIRS) setup.py
	ruff $(DIRS) setup.py

# Format source code automatically
style:
	black $(DIRS) setup.py
	ruff $(DIRS) setup.py --fix

# Run tests for the library
test:
	python -m pytest -v tests/
