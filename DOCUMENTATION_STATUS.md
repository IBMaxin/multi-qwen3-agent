# Documentation Status

Last updated: 2025-11-06

## Summary

- Total tests: 190
- Passed: 190
- Overall coverage: 93%

## Completed Tasks

- test_tools_custom.py: 100% coverage, 16 tests
- test_tools_github.py: 97% coverage, 27 tests
- test_agent.py: 90% coverage, 39 tests
- All tests passing (190/190)
- docs/patterns/RAG_PATTERNS.md: Added official RAG patterns (7 sections)
- docs/patterns/CONVERSATION_PATTERNS.md: Added conversation patterns (8 sections)
- examples/local_vector_rag_example.py: Added init_agent_service(), demo_script(), app_tui(), fixed multi-turn pattern

## Pending/Follow-ups

- README improvements: more quick-starts for Windows PowerShell
- Optional: add a small CLI example for RAG TUI in examples/

## How to Run

- Production HITL pipeline:
```pwsh
cd production
python -m qwen_pipeline.cli
```

- Full test suite with coverage:
```pwsh
cd production
pytest tests/ --cov=qwen_pipeline --cov-report=term
```

- Local Vector RAG TUI:
```pwsh
python examples/local_vector_rag_example.py
```
