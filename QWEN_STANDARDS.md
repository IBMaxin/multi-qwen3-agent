# Qwen-Agent Coding Standards Checklist

## Before committing ANY code, verify:

### ✅ **Agent Creation Pattern**
```python
# ✅ CORRECT - Official Qwen pattern
from qwen_agent.agents import Assistant, ReActChat

def init_agent_service():
    llm_cfg = {
        'model': 'qwen3:8b',
        'model_server': 'http://localhost:11434/v1',
        'api_key': 'EMPTY',
    }
    bot = Assistant(llm=llm_cfg, function_list=tools)
    return bot

# ❌ WRONG - Custom agent class
class MyCustomAgent(Agent):  # Don't do this!
    def __init__(self): ...
```

### ✅ **Tool Registration Pattern**
```python
# ✅ CORRECT - Official pattern
from qwen_agent.tools.base import BaseTool, register_tool

@register_tool('my_tool')
class MyTool(BaseTool):
    description = "..."
    parameters = [...]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        return json.dumps(result)

# ❌ WRONG - No registration
class MyTool:  # Missing BaseTool!
    def execute(self, data): ...  # Wrong method name!
```

### ✅ **Import Order** (Official Qwen Style)
```python
# 1. Standard library
import os
import json
from typing import List, Dict, Optional

# 2. Third-party (non-Qwen)
import json5
from asteval import Interpreter

# 3. Qwen-Agent imports
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.gui import WebUI

# 4. Local imports (if package)
from .config import get_llm_config
```

### ✅ **Security Rules** (ENFORCED)
```python
# ✅ ALLOWED
from asteval import Interpreter
aeval = Interpreter()
result = aeval(expression)

# ❌ BANNED - Will fail pre-commit
eval(code)  # NEVER
exec(code)  # NEVER
__import__('os').system(cmd)  # NEVER
```

### ✅ **Example Script Structure** (Official Pattern)
```python
# EVERY example must follow this structure:

def init_agent_service():
    """Initialize and return agent"""
    pass

def test(query='...'):
    """Quick test function"""
    bot = init_agent_service()
    messages = [{'role': 'user', 'content': query}]
    for response in bot.run(messages):
        print('bot response:', response)

def app_tui():
    """Terminal UI - interactive loop"""
    bot = init_agent_service()
    messages = []
    while True:
        query = input('user question: ')
        messages.append({'role': 'user', 'content': query})
        response = []
        for response in bot.run(messages):
            print('bot response:', response)
        messages.extend(response)

def app_gui():
    """Gradio Web UI"""
    bot = init_agent_service()
    WebUI(bot).run()

if __name__ == '__main__':
    # test()
    # app_tui()
    app_gui()
```

### ✅ **Configuration Pattern**
```python
# ✅ CORRECT - Dict-based config
llm_cfg = {
    'model': 'qwen3:8b',
    'model_server': 'http://localhost:11434/v1',
    'api_key': 'EMPTY',
    'generate_cfg': {
        'top_p': 0.8,
        'temperature': 0.7,
    }
}

# ❌ WRONG - Custom config classes
class QwenConfig:  # Don't do this
    def __init__(self): ...
```

### ✅ **Message Format**
```python
# ✅ CORRECT - Official format
messages = [
    {'role': 'user', 'content': 'Hello'},
    {'role': 'assistant', 'content': 'Hi there!'}
]

# Or with Message class
from qwen_agent.llm.schema import Message
messages = [Message('user', 'Hello')]

# ❌ WRONG - Custom format
messages = [("user", "Hello")]  # Wrong!
```

### ✅ **Tool Parameters**
```python
# ✅ CORRECT - Use json/json5 for parsing
import json5

def call(self, params: str, **kwargs) -> str:
    params_dict = json5.loads(params)
    return json.dumps(result)

# ❌ WRONG - Direct dict access
def call(self, params: dict) -> dict:  # Wrong types!
    return result  # Wrong return type!
```

---

## 🚨 **Pre-Commit Will Block These:**

1. ❌ Using `exec()` or `eval()`
2. ❌ Import statements not properly ordered
3. ❌ Line length > 100 characters
4. ❌ Trailing whitespace
5. ❌ Debug statements (`breakpoint()`, `pdb.set_trace()`)
6. ❌ Type hints missing (mypy strict mode)
7. ❌ Unsafe deserialization (pickle)

---

## 📚 **Reference Official Examples:**

When in doubt, check:
- https://github.com/QwenLM/Qwen-Agent/tree/main/examples/assistant_qwen3.py
- https://github.com/QwenLM/Qwen-Agent/tree/main/examples/assistant_add_custom_tool.py
- https://github.com/QwenLM/Qwen-Agent/tree/main/qwen_agent/agents/react_chat.py

---

## 🔧 **Quick Commands:**

```pwsh
# Before committing:
pre-commit run --all-files  # Check everything
ruff check .                # Lint
mypy qwen_pipeline/         # Type check
bandit -r qwen_pipeline/    # Security scan
pytest                      # Tests

# Auto-fix formatting:
black .
ruff check --fix .
```

---

## 💡 **When You Want to Add Something New:**

### Ask yourself:
1. ✅ Does the official Qwen-Agent repo do this already?
2. ✅ Can I use an existing Qwen agent (Assistant, ReActChat)?
3. ✅ Can I use an existing Qwen tool?
4. ⚠️ Am I inventing a new pattern that doesn't exist in official repo?

### If #4 is YES:
- **STOP** - You're probably doing it wrong
- Check official repo again
- Ask: "How would the Qwen team solve this?"
- Use their pattern, not yours

---

## 🎯 **Golden Rule:**

> "If it's not in the official Qwen-Agent examples, don't do it."

Stay vanilla. Stay official. Stay compatible.
