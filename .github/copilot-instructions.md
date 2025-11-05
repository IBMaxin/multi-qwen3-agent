3. Use `@register_tool` for tools - don't invent new patterns
## QwenAgent-Best – Copilot instructions

These are the essentials an AI coding agent needs to work productively in this repo. Keep it concise, stay within official Qwen-Agent patterns, and prefer editing existing modules over inventing new ones.

### Big picture
- Two tracks:
    - `production/`: Modular, test-backed multi‑agent system with HITL (Planner → Coder → Reviewer). See `agent.py:create_agents`, `pipeline.py:run_pipeline`, `tools.py:SafeCalculatorTool`, `config.py:get_llm_config`.
    - `examples/`: Self‑contained demo with richer tools and GUI. See `examples/qwen3-agentV2-complete.py` and `examples/run_qwen_gui.py`.
- LLM runs via Ollama’s OpenAI‑compatible endpoint; config is env‑driven and centralized in `production/config.py`.

### Run and verify (local, Windows PowerShell)
```pwsh
# Ollama
ollama serve
ollama pull qwen3:8b

# Env
cp .env.example .env  # MODEL_SERVER=http://localhost:11434/v1, MODEL_NAME=qwen3:8b, API_KEY=EMPTY

# Production pipeline (HITL)
# IMPORTANT: run as a module from repo root to preserve package-relative imports
python -m production.cli

# Tests and coverage (production only)
cd production; pytest -v; coverage run -m pytest; coverage report
```

### Core patterns you must follow
- Agents: use official classes only. `Assistant` and `ReActChat` assembled into a `GroupChat` (see `production/agent.py`).
- Tools: subclass `BaseTool` and decorate with `@register_tool`. Parse input with `json5.loads(params: str)` and return `json.dumps(...)` string. Example: `SafeCalculatorTool` in `production/tools.py`.
- Safety: never use `exec()`/`eval()`. Use `asteval.Interpreter()` for math. Coder agent is instructed “No file/system access.”
- HITL: all critical outputs flow through `human_approval(step_name, content)` in `production/pipeline.py` (options: yes/no/edit) and are logged.
- Logging: use `structlog.get_logger()`; no `print()` in production modules (CLI may print for UX/HITL prompt).

### Configuration and integration
- `get_llm_config()` builds the dict consumed by Qwen‑Agent: `{model, model_server, api_key, generate_cfg}` with sensible defaults.
- Optional monitoring: set `SENTRY_DSN` to enable Sentry in `production/config.py`.
- Typical toolset: `['code_interpreter', SafeCalculatorTool()]` passed into `create_agents()`; add new tools by expanding this list.

### Tests and quality gates
- Primary tests live in `production/tests/` (e.g., `test_pipeline.py`, `test_tools.py`). Aim for full coverage when changing public behavior.
- Standards (see `QWEN_STANDARDS.md`):
    - Keep to official Qwen‑Agent APIs and message formats.
    - Type hints required; 100‑char lines; import order per checklist; no `exec`/`eval`.
    - Run: `make check-standards` or individually `ruff`, `mypy`, `bandit`.

### Useful references in this repo
- Agent wiring: `production/agent.py:create_agents` (Planner/Coder/Reviewer → `GroupChat`).
- HITL flow: `production/pipeline.py:human_approval`, `run_pipeline`.
- Safe tool example: `production/tools.py:SafeCalculatorTool` (JSON5 in, JSON string out).
- Rich demo tools and GUI: `examples/qwen3-agentV2-complete.py` (DuckDuckGo search, image gen, calculator, filesystem), `examples/run_qwen_gui.py`.

### Pitfalls to avoid
- Wrong endpoint: Ollama requires `/v1` suffix.
- Wrong tool I/O types: always parse JSON5 string and return JSON string.
- Blocking on HITL in non‑interactive contexts—stub or mock `human_approval` in tests.
- Missing model: run `ollama pull qwen3:8b` before executing.

Golden rule: “If it’s not in the official Qwen‑Agent examples, don’t do it.” See `QWEN_STANDARDS.md` for specifics.
