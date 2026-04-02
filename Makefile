.PHONY: install dev test lint format clean docker-build docker-up export-onnx

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=perception_stack --cov-report=html

lint:
	ruff check perception_stack/ tests/
	mypy perception_stack/

format:
	ruff format perception_stack/ tests/

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf __pycache__ .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

export-onnx:
	python -m perception_stack.export.onnx_export --config configs/default.yaml
