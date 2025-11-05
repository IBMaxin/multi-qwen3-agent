.PHONY: help install format lint type-check security test check-standards all clean

help:
	@echo "Qwen-Agent Standards Enforcement Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install          Install package in dev mode"
	@echo "  format           Auto-format code (black + ruff)"
	@echo "  lint             Check code style (ruff)"
	@echo "  type-check       Run mypy type checking"
	@echo "  security         Security scan (bandit - blocks exec/eval)"
	@echo "  test             Run tests with coverage"
	@echo "  check-standards  Full standards check (REQUIRED before commit)"
	@echo "  all              Run everything (format + check-standards)"
	@echo "  clean            Remove cache files"

install:
	@echo "📦 Installing qwen_pipeline..."
	cd production && pip install -e .[dev]

hooks-install:
	@echo "🔗 Installing Git hooks via pre-commit (commit + push)..."
	. .venv310/Scripts/activate && pre-commit install --install-hooks
	. .venv310/Scripts/activate && pre-commit install --hook-type pre-push --install-hooks
	@echo "✅ Hooks installed. Commits and pushes will be checked."

format:
	@echo "🎨 Auto-formatting code to Qwen standards..."
	black --line-length 100 production/qwen_pipeline/ examples/
	ruff check --fix production/qwen_pipeline/ examples/
	@echo "✅ Formatting complete!"

lint:
	@echo "🔍 Checking code style..."
	ruff check production/qwen_pipeline/ examples/
	@echo "✅ Lint check passed!"

type-check:
	@echo "🔬 Running type checks..."
	cd production && mypy qwen_pipeline/ --config-file ../production-pyproject.toml
	@echo "✅ Type check passed!"

security:
	@echo "🔐 Running security scan (blocking exec/eval)..."
	bandit -r production/qwen_pipeline/ -c production-pyproject.toml
	@echo "✅ Security check passed!"

test:
	@echo "🧪 Running tests..."
	cd production && pytest tests/ -v --cov=qwen_pipeline --cov-report=term-missing
	@echo "✅ Tests passed!"

check-qwen-patterns:
	@echo "🎯 Verifying Qwen-Agent patterns..."
	@# Check for exec/eval
	@if grep -r "exec(" production/qwen_pipeline/ --include="*.py" 2>/dev/null; then \
		echo "❌ FAILED: exec() detected - BANNED!"; \
		exit 1; \
	fi
	@if grep -r "eval(" production/qwen_pipeline/ --include="*.py" 2>/dev/null | grep -v "asteval"; then \
		echo "❌ FAILED: eval() detected - Use asteval!"; \
		exit 1; \
	fi
	@# Check for proper tool registration
	@if grep -r "class.*Tool" production/qwen_pipeline/ --include="*.py" | grep -v "BaseTool" | grep -v "register_tool" 2>/dev/null; then \
		echo "⚠️  Warning: Tool class without BaseTool inheritance?"; \
	fi
	@echo "✅ Qwen patterns verified!"

check-examples:
	@echo "📝 Checking examples follow official pattern..."
	@for file in examples/*.py; do \
		if ! grep -q "def init_agent_service" "$$file"; then \
			echo "❌ $$file missing init_agent_service()"; \
			exit 1; \
		fi; \
	done
	@echo "✅ Examples follow official pattern!"

check-standards: lint type-check security check-qwen-patterns check-examples
	@echo ""
	@echo "═══════════════════════════════════════════════════"
	@echo "✅ ALL QWEN-AGENT STANDARDS CHECKS PASSED!"
	@echo "═══════════════════════════════════════════════════"
	@echo ""
	@echo "Your code follows official Qwen-Agent conventions."
	@echo "Safe to commit! 🚀"
	@echo ""

all: format check-standards test
	@echo ""
	@echo "═══════════════════════════════════════════════════"
	@echo "✅ COMPLETE! Everything formatted, checked, and tested."
	@echo "═══════════════════════════════════════════════════"

pre-commit: check-standards
	@echo "🔒 Running pre-commit checks..."
	pre-commit run --all-files || (echo "❌ Pre-commit checks failed! Run 'make format' to fix." && exit 1)
	@echo "✅ Ready to commit!"

clean:
	@echo "🧹 Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage 2>/dev/null || true
	@echo "✅ Clean complete!"

.PHONY: check-qwen-patterns check-examples pre-commit
