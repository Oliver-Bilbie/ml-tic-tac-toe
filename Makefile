bootstrap: install-deps install-dev-deps
	@pre-commit install

test: unit-test int-test

install-deps:
	@echo "[INFO] Installing dependencies"
	@python -m pipenv lock --pre
	@python -m pipenv install

install-dev-deps:
	@echo "[INFO] Installing dev dependencies"
	@python -m pipenv lock --pre
	@python -m pipenv install --dev

format-src:
	@echo "[INFO] Formatting source code using black"
	@python -m pipenv run black src

lint:
	@echo "[INFO] Linting source code using pylint"
	@python -m pipenv run pylint --fail-under 8 src/service/*

bandit:
	@echo "[INFO] Linting source code using bandit to look for common security issues in python source"
	@python -m pipenv run  bandit -r src/service --configfile bandit.yaml

type-check:
	@echo "[INFO] Checking static typing of source code using mypy"
	@python -m mypy src/service --ignore-missing-imports

unit-test:
	@echo "[INFO] Running unit tests"
	@export COVERAGE_FILE=./target/coverage/.cov_unit && python -m pipenv run coverage run --source=src/service -m pytest -s --junitxml=target/unit-test/unit-result.xml src/test/unit/*.py
	@export COVERAGE_FILE=./target/coverage/.cov_unit && python -m pipenv run coverage html --directory=target/unit-test/coverage --fail-under=100
	@export COVERAGE_FILE=./target/coverage/.cov_unit && python -m pipenv run coverage report
	@export COVERAGE_FILE=./target/coverage/.cov_unit && python -m pipenv run coverage xml -o target/unit-test/coverage/coverage.xml

int-test:
	@echo "[INFO] Running component integration tests"
	@export COVERAGE_FILE=./target/coverage/.cov_int && python -m pipenv run coverage run --source=src/service -m pytest -s --junitxml=target/integration-test/integration-result.xml src/test/integration/*.py
	@export COVERAGE_FILE=./target/coverage/.cov_int && python -m pipenv run coverage html --directory=target/integration-test/coverage --fail-under=80
	@export COVERAGE_FILE=./target/coverage/.cov_int && python -m pipenv run coverage report
	@export COVERAGE_FILE=./target/coverage/.cov_int && python -m pipenv run coverage xml -o target/integration-test/coverage/coverage.xml
