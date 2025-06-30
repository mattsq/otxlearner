.PHONY: format lint typecheck

format:
	pre-commit run --files $(FILES)

lint:
	ruff src tests

typecheck:
	mypy --strict src

