# Quick Setup - Qwen Standards Enforcement

## 🚀 Initial Setup (One-time)

```pwsh
# 1. Install pre-commit hooks
pip install pre-commit
pre-commit install

# 2. Install development dependencies
cd production
pip install -e .[dev]
cd ..

# 3. Verify everything works
make check-standards
```

## 📝 Daily Workflow

### Before Starting Work:
```pwsh
# Pull latest and verify environment
git pull
make install
```

### While Coding:
Your editor will auto-format on save (if using VSCode settings).

### Before Committing:
```pwsh
# Auto-fix formatting issues
make format

# Run all checks
make check-standards

# If adding tests
make test
```

### Git Commit:
```pwsh
git add .
git commit -m "Your message"
# Pre-commit hooks run automatically!
# If they fail, fix issues and try again
```

## 🛡️ What Gets Enforced

### Automatic (Pre-commit hooks):
- ✅ Black formatting (100 char lines)
- ✅ Import sorting
- ✅ Ruff linting
- ✅ No `exec()` or `eval()` (Bandit)
- ✅ No debug statements
- ✅ Type checking
- ✅ Trailing whitespace removal

### Manual Check (Before commit):
```pwsh
make check-standards
```
This runs:
1. **Lint** - Code style
2. **Type check** - Mypy strict mode
3. **Security** - Bandit scan
4. **Pattern check** - Verify Qwen conventions
5. **Example check** - Ensure examples follow official structure

## 🔧 Quick Fixes

### If pre-commit fails:
```pwsh
make format       # Auto-fix formatting
make check-standards  # See what's wrong
```

### If bandit complains:
```python
# ❌ This will be blocked:
eval(user_input)

# ✅ Do this instead:
from asteval import Interpreter
aeval = Interpreter()
result = aeval(user_input)
```

### If pattern check fails:
Check `QWEN_STANDARDS.md` for examples.

## 📚 References

- **Coding Standards**: `QWEN_STANDARDS.md`
- **Official Examples**: https://github.com/QwenLM/Qwen-Agent/tree/main/examples
- **Pre-commit Config**: `.pre-commit-config.yaml`
- **CI/CD**: `.github/workflows/enforce-standards.yml`

## 🎯 Remember

> "Stay vanilla. Stay official. Stay compatible."

If you're doing something that's not in the official Qwen-Agent repo, you're probably doing it wrong.
