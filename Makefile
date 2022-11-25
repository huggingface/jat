.PHONY: quality style test unity-test

# Check that source code meets quality standards
quality:
	black --check --line-length 119 --target-version py38 tests agi examples 
	isort --check-only tests src examples 
	flake8 tests agi examples

# Format source code automatically
style:
	black --line-length 119 --target-version py38 tests agi examples
	isort tests src examples integrations/Unity/tests

# Run tests for the library
test:
	python -m pytest -n auto --dist=loadfile -s -v  ./tests/

