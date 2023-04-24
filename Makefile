.PHONY: quality style test

# Define directories variable
DIRS = data examples gia scripts tests

# Check that source code meets quality standards
quality:
	black --check $(DIRS)
	ruff check 119 $(DIRS)

# Format source code automatically
style:
	black $(DIRS)
	ruff $(DIRS)

# Run tests for the library
test:
	python -m pytest tests/
