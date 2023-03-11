.PHONY: quality style test

# Check that source code meets quality standards
quality:
	black --check --line-length 119 --target-version py38 tests gia data
	isort --check-only --profile black tests gia data
	flake8 --max-line-length 119 tests gia --ignore=E203

# Format source code automatically
style:
	black --line-length 119 --target-version py38 tests gia data
	isort --profile black tests gia data

# Run tests for the library
test:
	python -m pytest tests/

