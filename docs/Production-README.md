# Qwen Pipeline Package

Production-grade safe coding assistant using Qwen-Agent framework.

## 🚀 Features

- **Multi-Agent System**: Planner, Coder, and Reviewer agents working in harmony
- **HITL (Human-in-the-Loop)**: Human approval for critical pipeline steps
- **Safe Code Execution**: Uses `asteval` for secure math operations
- **Production Logging**: Structured logging with `structlog`
- **Error Monitoring**: Optional Sentry integration
- **Local LLMs**: Works with Ollama for privacy and offline use
- **100% Test Coverage**: Comprehensive pytest suite
- **CI/CD Ready**: GitHub Actions workflow included
- **Docker Support**: Production-ready containerization

> Note: This project supports Python 3.10 only (3.10.x). Use a 3.10 virtual environment.

## 📦 Installation

### From PyPI (when published)
```bash
pip install qwen_pipeline
```

### From Source
```bash
git clone <repository-url>
cd qwen-alpha1
pip install -e .[dev]
```

## 🔧 Configuration

### Prerequisites
Install Ollama and pull a Qwen model:
```bash
# Install Ollama from https://ollama.ai/
# Pull a Qwen model
ollama pull qwen3:8b
```

### Environment Setup
Copy the example environment file and configure:
```bash
cp .env.example .env
```

Edit `.env` with your settings:
- `MODEL_SERVER`: Ollama server endpoint (default: http://localhost:11434/v1)
- `MODEL_NAME`: Ollama model name (default: qwen3:8b)
- `API_KEY`: API key (default: EMPTY for Ollama)
- `SENTRY_DSN`: Optional Sentry DSN for error monitoring

## 🎯 Usage

### Command Line Interface
```bash
qwen-pipeline
```

### Python API
```python
from qwen_pipeline.pipeline import run_pipeline

result = run_pipeline("Calculate sqrt(144)")
print(result)
```

## 🧪 Development

### Setup Development Environment (Python 3.10)
```bash
# Create and activate a 3.10 venv (example: .venv310)
python3.10 -m venv .venv310
.venv310\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -e .[dev]
```

### Run Tests
```bash
coverage run -m pytest
coverage report
```

### Code Quality
```bash
# Format code
black qwen_pipeline/

# Lint code
ruff check qwen_pipeline/

# Fix auto-fixable issues
ruff check --fix qwen_pipeline/

# Type check
mypy qwen_pipeline/

# Or use Make commands
make format
make lint
make type-check
```

### Build Package
```bash
python -m build
```

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t qwen-pipeline .
```

### Run Container
```bash
docker run -it --env-file .env qwen-pipeline
```

## 🏗️ Architecture

```
qwen_pipeline/
├── config.py      # Environment configuration
├── agent.py       # Multi-agent setup
├── tools.py       # Safe calculator tool
├── pipeline.py    # HITL pipeline
└── cli.py         # CLI interface
```

## 🛡️ Security

- No `exec()` or `eval()` - uses `asteval` for safe evaluation
- HITL approval required for critical operations
- No file system or network access by default
- Environment-based configuration
- Optional Sentry monitoring for production

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run `flake8` and `pytest`
5. Submit a pull request

## 📚 Documentation

For more details on Qwen-Agent, visit:
- [Qwen-Agent GitHub](https://github.com/QwenLM/Qwen-Agent)
- [Qwen-Agent Documentation](https://qwen-agent.readthedocs.io/)

## 🐛 Troubleshooting

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Model Connection Issues
Verify `MODEL_SERVER` is accessible:
```bash
curl $MODEL_SERVER/health
```

### Test Failures
Run with verbose output:
```bash
pytest -vv
```
