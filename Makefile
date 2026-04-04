.PHONY: install train test lint serve gradio audit docker-build hf-remote hf-push

install:
	pip install -r requirements.txt -r requirements-dev.txt

train:
	python -m src.training.train --config config/config.yaml

test:
	python -m pytest tests/ -v --tb=short --cov=src --cov-fail-under=70

lint:
	python -m black src/ tests/
	python -m flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	python -m isort src/ tests/ --profile black
	python -m mypy src/
	python -m bandit -r src/ -ll -ii

serve:
	python -m uvicorn src.api.app:app --reload --port 8000

gradio:
	python -m src.api.gradio_demo

audit:
	python -m pip_audit -r requirements.txt

docker-build:
	docker build -t b3-lstm-forecaster .

# HF Space deployment
# Usage:
#   make hf-remote HF_USER=Priyrajsinh HF_SPACE=my-space-name
#   make hf-push   HF_SPACE_DIR=hf_space
hf-remote:
	git remote add hf-space https://huggingface.co/spaces/$(HF_USER)/$(HF_SPACE)
	@echo "Remote 'hf-space' added → push with: make hf-push HF_SPACE_DIR=hf_space"

hf-push:
	git subtree push --prefix=$(HF_SPACE_DIR) hf-space main
