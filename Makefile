.PHONY: quality style test unity-test

# Check that source code meets quality standards
quality:
	black --check --line-length 119 --target-version py38 tests gia 
	isort --check-only tests gia 
	flake8 tests gia

# Format source code automatically
style:
	black --line-length 119 --target-version py38 tests gia
	isort tests gia  

# Run tests for the library
test:
	python -m pytest tests/

