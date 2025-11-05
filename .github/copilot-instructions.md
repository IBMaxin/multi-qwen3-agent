## QwenAgent-Best – AI Agent Instructions

Consolidated Qwen-Agent implementation with production-grade multi-agent system and example tools. **Python 3.10 ONLY** (newer versions not supported). Follow official Qwen-Agent patterns exclusively—no custom inventions.

### Architecture Overview

**Two parallel tracks:**
- `production/qwen_pipeline/`: Modular HITL system with Planner → Coder → Reviewer agents in a `GroupChat`
  - Entry: `agent.py:create_agents()` wires agents; `pipeline.py:run_pipeline()` orchestrates flow
  - Config: `config.py:get_llm_config()` reads env vars (with dotenv support) and returns Qwen-Agent dict
  - Safety: `tools.py:SafeCalculatorTool` uses `asteval.Interpreter()` (no `exec`/`eval`)
  - HITL: `pipeline.py:human_approval()` gates critical steps (yes/no/edit) with `structlog` logging
  - CLI: `cli.py:main()` provides interactive terminal interface
- `examples/`: Self-contained demos with richer tools (DuckDuckGo search, image gen, filesystem, Gradio GUI)
  - Canonical example: `examples/qwen3-agentV2-complete.py` (498 lines, full toolset)
  - GUI demo: `examples/run_qwen_gui.py` (uses `qwen_agent.gui.WebUI`)

**LLM integration:** Ollama's OpenAI-compatible endpoint (`http://localhost:11434/v1`). Config centralized in `production/qwen_pipeline/config.py` with env-driven defaults from `.env` (uses `python-dotenv` for auto-loading).

### Critical Developer Workflows

```pwsh
# Setup (Windows PowerShell, Python 3.10 venv required)
ollama serve                           # Start Ollama
ollama pull qwen3:8b                   # Pull model (or qwen3:4b)
cp .env.example .env                   # Configure: MODEL_SERVER, MODEL_NAME, API_KEY=EMPTY
cd production && pip install -e .      # Install qwen_pipeline package
python -m qwen_pipeline.cli            # Run production HITL pipeline

# Quality gates (run from repo root)
make check-standards                   # Full check: ruff, mypy, bandit (blocks exec/eval)
cd production && pytest -v             # Run tests
coverage run -m pytest && coverage report  # Coverage (aim for 100%)

# Individual checks
ruff check . --fix                     # Lint + auto-fix
black --line-length 100 production/ examples/  # Format
mypy production/qwen_pipeline/ --config-file production-pyproject.toml  # Type check
bandit -r production/qwen_pipeline/    # Security scan
```

### Qwen-Agent Patterns (Strictly Enforce)

**Agent creation:** Use official classes only. See `production/qwen_pipeline/agent.py:create_agents()`:
```python
from qwen_agent.agents import Assistant, ReActChat, GroupChat
llm_cfg = get_llm_config()
planner = Assistant(llm=llm_cfg, system_message="...", name="planner")
coder = ReActChat(llm=llm_cfg, function_list=tools, system_message="...", name="coder")
manager = GroupChat(agents=[planner, coder, reviewer], llm=llm_cfg)
```

**Tool registration:** Subclass `BaseTool`, decorate with `@register_tool`. See `production/qwen_pipeline/tools.py:SafeCalculatorTool`:
```python
from qwen_agent.tools.base import BaseTool, register_tool
import json5, json

@register_tool("safe_calculator")
class SafeCalculatorTool(BaseTool):
    description = "Safely calculate math like sqrt(16) or sin(3.14)."
    parameters = [{"name": "expression", "type": "string", "required": True}]

    def call(self, params: str, **kwargs) -> str:
        params_dict = json5.loads(params)  # Parse JSON5 string input
        result = self.aeval(expression)    # asteval.Interpreter() instance
        return json.dumps({"result": result})  # Return JSON string
```

**Message format:** Official schema only: `[{"role": "user", "content": "Hello"}]`

**Import order:** (1) stdlib (2) third-party non-Qwen (3) qwen_agent (4) local. See `QWEN_STANDARDS.md`.

### Production-Specific Conventions

**HITL:** `production/qwen_pipeline/pipeline.py:human_approval(step_name, content)` prompts `yes/no/edit`, logs with `structlog`, raises `ValueError` on rejection. Tests must mock: `@patch("qwen_pipeline.pipeline.human_approval", return_value="mock")`

**Logging:** Use `structlog.get_logger()`; no `print()` in production modules (CLI may print for UX).

**Safety:** `exec()`/`eval()` BANNED (pre-commit blocks). Use `asteval.Interpreter()` for math.

**Testing example from `production/tests/test_pipeline.py`:**
```python
@patch("qwen_pipeline.pipeline.human_approval", return_value="mock output")
@patch("qwen_pipeline.pipeline.create_agents")
def test_run_pipeline(mock_agents, mock_approval):
    result = run_pipeline("query")
    assert "mock" in result
```

### Configuration & Integration

**LLM config:** `production/qwen_pipeline/config.py:get_llm_config()` reads:
- `MODEL_SERVER` (default: `http://localhost:11434/v1` — `/v1` suffix required!)
- `MODEL_NAME` (default: `qwen3:8b`)
- `API_KEY` (default: `EMPTY` for Ollama)
- Returns dict with `generate_cfg: {top_p: 0.8, temperature: 0.7, max_input_tokens: 6000}`

**Tools:** Pass list to `create_agents()`: `['code_interpreter', SafeCalculatorTool()]`

**Monitoring:** Set `SENTRY_DSN` in `.env` to enable Sentry in `production/qwen_pipeline/config.py:get_llm_config()`.

### Common Pitfalls

- **Missing `/v1` suffix:** Ollama endpoint must be `http://localhost:11434/v1`
- **Wrong tool I/O:** Always `json5.loads(params: str)` input, `json.dumps(...)` output
- **Module imports:** Run as module: `python -m qwen_pipeline.cli` (not `python production/cli.py`)
- **Blocking HITL:** Mock `human_approval` in tests
- **Missing model:** Run `ollama pull qwen3:8b` before executing
- **Python version:** Only 3.10.x supported

### Key Files Reference

- Agent wiring: `production/qwen_pipeline/agent.py:create_agents` (Planner/Coder/Reviewer → `GroupChat`)
- HITL flow: `production/qwen_pipeline/pipeline.py:human_approval`, `run_pipeline`
- Safe tool example: `production/qwen_pipeline/tools.py:SafeCalculatorTool` (JSON5 in, JSON string out, asteval)
- Config builder: `production/qwen_pipeline/config.py:get_llm_config`
- Rich example: `examples/qwen3-agentV2-complete.py` (DuckDuckGo, image gen, calculator, GUI)
- Standards checklist: `QWEN_STANDARDS.md` (official patterns, pre-commit rules)
- Test patterns: `production/tests/test_pipeline.py` (HITL mocking)

**Golden rule:** "If it's not in the official Qwen-Agent examples, don't do it." See `QWEN_STANDARDS.md` for specifics.
