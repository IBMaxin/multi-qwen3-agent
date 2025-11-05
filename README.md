# QwenAgent-Best - Consolidated Qwen Agent Implementation

> Requires Python 3.10 only (3.10.x). Newer Python versions are not supported.

This directory contains the **best and most well-developed** Qwen Agent code from Bobby's collection.

## 📁 Directory Structure

- **production/** - Production-ready multi-agent system (from qwen-alpha1)
- **examples/** - Complete working examples and GUI tools
- **docs/** - All documentation and setup guides
- **tests/** - Test suites (from production code)

## 🚀 Quick Start

### Production Multi-Agent System (Python 3.10 required)
```bash
cd production/
# Configure environment
cp ../.env.example .env
# Edit .env with your Ollama settings
# Run the production pipeline
python cli.py
```

### Single-File Example
```bash
cd examples/
# Run the complete example
python qwen3-agentV2-complete.py
```

## �� What's Included

### From qwen-alpha1 (Production Grade):
- Multi-agent system (Planner, Coder, Reviewer)
- Human-in-the-Loop (HITL) approval
- Safe code execution with asteval
- Structured logging
- Complete test suite
- Modern Python packaging

### From qwen3-agentV2.py (Complete Example):
- 498 lines of well-documented code
- Comprehensive toolset (web search, calculator, file ops)
- Ready-to-run Ollama integration
- Professional error handling

## 🎯 Recommended Usage

1. **Learning**: Start with xamples/qwen3-agentV2-complete.py
2. **Development**: Use production/ for serious projects
3. **GUI**: Try xamples/run_qwen_gui.py for visual interface

---
*Consolidated from 15+ scattered Qwen projects on November 4, 2025*
