
# QwenAgent-Best – AI Agent Coding Instructions

**Python 3.10.x only.** Follow official Qwen-Agent patterns—no custom inventions. All code must match the latest Qwen-Agent and this repo’s standards.

## Architecture & Key Components

- **production/qwen_pipeline/**: Modular multi-agent system (Planner → Coder → Reviewer in `GroupChat`). Entry: `agent.py:create_agents()`, orchestrated by `pipeline.py:run_pipeline()`. HITL approval via `pipeline.py:human_approval()` (yes/no/edit, logs with structlog).
- **examples/**: Self-contained demos with richer tools (web search, image gen, file ops, GUI). See `examples/qwen3-agentV2-complete.py` for canonical usage.
- **Custom tools**: Add via `tools_custom.py` (e.g., `LocalVectorSearch` for local RAG). Register with `@register_tool` and subclass `BaseTool`.
- **LLM config**: Centralized in `config.py:get_llm_config()`; reads `.env` (MODEL_SERVER, MODEL_NAME, API_KEY, SENTRY_DSN).

## Critical Developer Workflows

```pwsh
# Setup (PowerShell, Python 3.10 venv required)
ollama serve
ollama pull qwen3:4b-instruct-2507-q4_K_M
cp .env.example .env  # Edit for your Ollama config
cd production; pip install -e .
python -m qwen_pipeline.cli  # Run production pipeline

# Quality gates
make check-standards  # ruff, mypy, bandit
cd production; pytest -v
coverage run -m pytest && coverage report
```

## Project-Specific Patterns & Conventions

- **Agent creation:**
  ```python
  from qwen_agent.agents import Assistant, ReActChat, GroupChat
  llm_cfg = get_llm_config()
  planner = Assistant(llm=llm_cfg, system_message="...", name="planner")
  coder = ReActChat(llm=llm_cfg, function_list=tools, system_message="...", name="coder")
  reviewer = Assistant(llm=llm_cfg, system_message="...", name="reviewer")
  manager = GroupChat(agents=[planner, coder, reviewer], llm=llm_cfg)
  ```
- **Tool registration:**
  ```python
  from qwen_agent.tools.base import BaseTool, register_tool
  @register_tool("safe_calculator")
  class SafeCalculatorTool(BaseTool):
      description = "Safely calculate math like sqrt(16) or sin(3.14)."
      parameters = [{"name": "expression", "type": "string", "required": True}]
      def call(self, params: str, **kwargs) -> str:
          params_dict = json5.loads(params)
          result = self.aeval(expression)
          return json.dumps({"result": result})
  ```
- **Message format:** Always use `[{'role': 'user', 'content': 'Hello'}]` (never tuples or custom schemas).
- **Import order:** stdlib → third-party → qwen_agent → local. See `QWEN_STANDARDS.md`.
- **No exec/eval:** Use `asteval.Interpreter()` for math. `exec()`/`eval()` are blocked by pre-commit.
- **Logging:** Use `structlog.get_logger()` (no `print()` in production modules).
- **Testing:** Mock HITL in tests: `@patch('qwen_pipeline.pipeline.human_approval', return_value='mock')`.
- **RAG:** For local vector search, use `LocalVectorSearch` in `tools_custom.py` (see RAG patterns in `docs/patterns/RAG_PATTERNS.md`).

## Integration & Configuration

- **LLM config:** `production/qwen_pipeline/config.py:get_llm_config()` (MODEL_SERVER must end with `/v1`).
- **Tool list:** Pass to `create_agents()` (e.g., `['code_interpreter', SafeCalculatorTool()]`).
- **Monitoring:** Set `SENTRY_DSN` in `.env` to enable Sentry.

## Common Pitfalls

- Ollama endpoint must be `http://localhost:11434/v1` (with `/v1`)
- Always use `json5.loads(params: str)` input, `json.dumps(...)` output for tools
- Run as module: `python -m qwen_pipeline.cli` (not `python production/cli.py`)
- Only Python 3.10.x is supported
- Reference only official Qwen-Agent patterns—see `QWEN_STANDARDS.md`

## Key References

- Agent wiring: `production/qwen_pipeline/agent.py:create_agents`
- HITL: `production/qwen_pipeline/pipeline.py:human_approval`, `run_pipeline`
- Custom tools: `production/qwen_pipeline/tools_custom.py`, `tools.py`, `tools_github.py`
- Example: `examples/qwen3-agentV2-complete.py`
- Patterns: `docs/patterns/CONVERSATION_PATTERNS.md`, `RAG_PATTERNS.md`
- Standards: `QWEN_STANDARDS.md`
- Tests: `production/tests/test_pipeline.py`

**Golden rule:** If it’s not in the official Qwen-Agent examples, don’t do it. See `QWEN_STANDARDS.md` for specifics.
