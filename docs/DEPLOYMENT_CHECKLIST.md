# Production Deployment Checklist

## Pre-Deployment

### Environment Setup
- [ ] Python 3.10.x installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed: `pip install -e .`
- [ ] `.env` configured (copy `.env.example` if present)

### Configuration Validation
- [ ] `MODEL_SERVER` points to Ollama with `/v1` suffix (e.g., `http://localhost:11434/v1`)
- [ ] `MODEL_NAME` set to supported model (e.g., `qwen3:8b`)
- [ ] `API_KEY` set to `EMPTY` for Ollama
- [ ] Ollama running: `ollama serve`
- [ ] Model pulled: `ollama pull qwen3:8b`

### Testing & Quality
- [ ] Unit tests pass: `pytest production/tests/ -v`
- [ ] Coverage ≥ 90%: `pytest --cov=qwen_pipeline`
- [ ] Lint: `ruff check production/qwen_pipeline/`
- [ ] Types: `mypy production/qwen_pipeline/ --config-file production-pyproject.toml`
- [ ] Security: `bandit -r production/qwen_pipeline/ -q`

## Deployment

### Dry Run
- [ ] Launch CLI: `python -m qwen_pipeline.cli`
- [ ] Type `exit` to quit

### Running
1. In one terminal: `ollama serve`
2. In another terminal: `python -m qwen_pipeline.cli`

## Monitoring & Safety
- [ ] Optional Sentry DSN configured: `SENTRY_DSN`
- [ ] No `exec()` or `eval()` in production modules
- [ ] Expression length limits enforced in calculator
- [ ] Logs monitored for errors

## Rollback
- [ ] `git log --oneline` to select previous stable commit/tag
- [ ] `git checkout <ref>` then reinstall with `pip install -e .`
