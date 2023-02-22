.PHONY: quality style test

# Check that source code meets quality standards
quality:
	black --check --line-length 119 --target-version py38 tests gia 
	isort --check-only --profile black tests gia 
	flake8 --max-line-length 119 tests gia

# Format source code automatically
style:
	black --line-length 119 --target-version py38 tests gia
	isort --profile black tests gia 

# Run tests for the library
test:
	python -m pytest tests/

