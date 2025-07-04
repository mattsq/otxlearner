.PHONY: format lint typecheck bench

format:
	pre-commit run --files $(FILES)

lint:
	ruff src tests

typecheck:
        mypy --strict src

bench:
        python bench/harness.py $(ARGS)

