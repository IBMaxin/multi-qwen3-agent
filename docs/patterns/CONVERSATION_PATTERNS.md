# Conversation Patterns for Qwen-Agent

Official Qwen-Agent patterns for conversation management. **Follow these patterns exclusively—no custom inventions.**

## Overview

Qwen-Agent supports multiple conversation modes:
- **Single-turn**: Stateless request-response
- **Multi-turn**: Stateful conversation with history
- **Streaming**: Real-time token-by-token responses
- **File attachments**: Image/document input for multimodal agents

## Pattern 1: Single-Turn Conversation

**Use case:** Stateless queries without conversation history.

```python
from qwen_agent.agents import ReActChat
from qwen_pipeline.config import get_llm_config

llm_cfg = get_llm_config()
agent = ReActChat(
    llm=llm_cfg,
    function_list=["code_interpreter"],
    system_message="You are a helpful coding assistant."
)

# Single query
messages = [{"role": "user", "content": "Calculate 2 + 2"}]
responses = []
for response in agent.run(messages=messages):
    responses.append(response)

# Extract final answer
final_response = responses[-1]
print(final_response[0]["content"])
```

**Message format:**
```python
messages = [
    {"role": "user", "content": "Your query here"}
]
```

## Pattern 2: Multi-Turn Conversation

**Use case:** Conversational AI with context preservation.

```python
from qwen_agent.agents import ReActChat
from qwen_pipeline.config import get_llm_config

llm_cfg = get_llm_config()
agent = ReActChat(
    llm=llm_cfg,
    function_list=["code_interpreter"],
    system_message="You are a helpful assistant."
)

# Initialize conversation history
messages = []

# Turn 1
messages.append({"role": "user", "content": "What is 5 * 6?"})
responses = []
for response in agent.run(messages=messages):
    responses.append(response)
messages.extend(responses)  # CRITICAL: extend, not reassign

# Turn 2 (agent remembers previous context)
messages.append({"role": "user", "content": "Multiply that by 2"})
responses = []
for response in agent.run(messages=messages):
    responses.append(response)
messages.extend(responses)

# Turn 3
messages.append({"role": "user", "content": "And add 10"})
responses = []
for response in agent.run(messages=messages):
    responses.append(response)
messages.extend(responses)

print("Full conversation:", messages)
```

**Critical pattern:** Always use `messages.extend(responses)`, never `messages = responses`.

## Pattern 3: Streaming Responses

**Use case:** Real-time response generation for better UX.

```python
from qwen_agent.agents import ReActChat
from qwen_pipeline.config import get_llm_config

llm_cfg = get_llm_config()
agent = ReActChat(
    llm=llm_cfg,
    function_list=[],
    system_message="You are a helpful assistant."
)

messages = [{"role": "user", "content": "Write a poem about coding"}]

# Stream responses token-by-token
for chunk in agent.run(messages=messages):
    # Each chunk is a list of message dicts
    if chunk and isinstance(chunk, list):
        for msg in chunk:
            if msg.get("role") == "assistant":
                print(msg.get("content", ""), end="", flush=True)
print()  # Newline after streaming completes
```

**Production streaming pattern:**
```python
from qwen_pipeline.pipeline import run_pipeline_streaming

# Streaming with HITL pipeline
for chunk in run_pipeline_streaming("Your query", timeout=30.0):
    print(chunk, end="", flush=True)
print()
```

## Pattern 4: GroupChat Multi-Turn Conversation

**Use case:** Multi-agent conversation with planner, coder, and reviewer.

```python
from qwen_pipeline.agent import create_agents
from qwen_pipeline.pipeline import run_pipeline

# Create 3-agent pipeline
manager = create_agents(["code_interpreter", "safe_calculator"])

# Multi-turn with GroupChat
messages = []

# Turn 1: Initial query
query1 = "Calculate the square root of 144"
result1 = run_pipeline(query1, manager=manager)
messages.append({"role": "user", "content": query1})
messages.append({"role": "assistant", "content": result1})

# Turn 2: Follow-up (planner sees previous context)
query2 = "Now multiply that by 3"
result2 = run_pipeline(query2, manager=manager)
messages.append({"role": "user", "content": query2})
messages.append({"role": "assistant", "content": result2})

print("Conversation history:", messages)
```

**Note:** `run_pipeline` is stateless per call. Manage history manually by appending results to `messages`.

## Pattern 5: File Attachments (Multimodal)

**Use case:** Image analysis, document processing with VL agents.

```python
from qwen_agent.agents import ReActChat
from qwen_pipeline.config import get_llm_config

llm_cfg = get_llm_config()
agent = ReActChat(
    llm=llm_cfg,
    function_list=["image_zoom_in_qwen3vl"],  # VL tool
    system_message="You are a vision-language assistant."
)

# Message with image attachment
messages = [
    {
        "role": "user",
        "content": [
            {"text": "What's in this image?"},
            {"image": "https://example.com/image.jpg"}  # URL or local path
        ]
    }
]

responses = []
for response in agent.run(messages=messages):
    responses.append(response)
print(responses[-1])
```

**Local file attachment:**
```python
messages = [
    {
        "role": "user",
        "content": [
            {"text": "Analyze this diagram"},
            {"file": "file:///path/to/local/image.png"}  # file:// URI
        ]
    }
]
```

## Pattern 6: Conversation with HITL (Human-in-the-Loop)

**Use case:** Production pipeline with approval gates.

```python
from qwen_pipeline.pipeline import run_pipeline

# HITL pipeline prompts for approval at critical steps
query = "Generate a Python script to delete files in /tmp"
result = run_pipeline(query)  # Will prompt: "Approve plan? (yes/no/edit)"

# User can:
# - Type "yes" to proceed
# - Type "no" to reject (raises ValueError)
# - Type "edit <new_plan>" to modify
```

**Disable HITL for testing:**
```python
from unittest.mock import patch

with patch("qwen_pipeline.pipeline.human_approval", return_value="approved"):
    result = run_pipeline(query)  # Bypasses human approval
```

## Pattern 7: Conversation State Management

**Use case:** Save/load conversation history for persistence.

```python
import json
from pathlib import Path

# Save conversation
def save_conversation(messages: list, filepath: str):
    Path(filepath).write_text(json.dumps(messages, indent=2, ensure_ascii=False))

# Load conversation
def load_conversation(filepath: str) -> list:
    return json.loads(Path(filepath).read_text())

# Usage
messages = [...]  # Conversation history
save_conversation(messages, "conversation_2025-11-06.json")

# Later: Resume conversation
messages = load_conversation("conversation_2025-11-06.json")
messages.append({"role": "user", "content": "Continue from where we left off"})
```

## Pattern 8: Error Handling in Conversations

**Use case:** Graceful degradation when agent fails.

```python
from qwen_agent.agents import ReActChat
from qwen_pipeline.config import get_llm_config

llm_cfg = get_llm_config()
agent = ReActChat(llm=llm_cfg, function_list=["code_interpreter"])

messages = []

def safe_agent_run(query: str) -> str:
    """Run agent with error handling."""
    messages.append({"role": "user", "content": query})
    try:
        responses = []
        for response in agent.run(messages=messages):
            responses.append(response)
        messages.extend(responses)
        return responses[-1][0]["content"]
    except Exception as e:
        error_msg = f"Agent error: {str(e)}"
        messages.append({"role": "assistant", "content": error_msg})
        return error_msg

# Conversation continues even if one turn fails
result1 = safe_agent_run("Calculate 10 / 2")
result2 = safe_agent_run("Invalid @#$% query!")  # Handles error
result3 = safe_agent_run("What was the first calculation?")  # Context preserved
```

## Common Pitfalls

### ❌ Wrong: Losing conversation history
```python
messages = [{"role": "user", "content": "Query 1"}]
messages = agent.run(messages=messages)  # OVERWRITES!
messages.append({"role": "user", "content": "Query 2"})  # Context lost
```

### ✅ Correct: Preserving history
```python
messages = [{"role": "user", "content": "Query 1"}]
responses = list(agent.run(messages=messages))
messages.extend(responses)  # APPENDS
messages.append({"role": "user", "content": "Query 2"})  # Context preserved
```

### ❌ Wrong: Blocking on streaming
```python
# Waits for full response before printing anything
responses = list(agent.run(messages=messages))
for r in responses:
    print(r)
```

### ✅ Correct: True streaming
```python
# Prints tokens as they arrive
for chunk in agent.run(messages=messages):
    if chunk:
        print(chunk[0].get("content", ""), end="", flush=True)
```

### ❌ Wrong: Stateful GroupChat assumption
```python
manager = create_agents([])
run_pipeline("Query 1", manager=manager)
run_pipeline("What was my previous query?", manager=manager)  # Fails: no memory
```

### ✅ Correct: Manual state management
```python
manager = create_agents([])
messages = []

result1 = run_pipeline("Query 1", manager=manager)
messages.append({"role": "user", "content": "Query 1"})
messages.append({"role": "assistant", "content": result1})

# Pass context manually if needed, or use ReActChat for multi-turn
```

## Official Examples Reference

**Qwen-Agent repository:**
- `examples/llm_assistant.py`: Single-turn Assistant
- `examples/react_data_analysis.py`: Multi-turn with tools
- `examples/group_chat.py`: Multi-agent conversation
- `examples/visual_storytelling.py`: Multimodal with images

**Local examples:**
- `examples/qwen3-agentV2-complete.py`: Full conversation demo
- `examples/run_qwen_gui.py`: Gradio chat interface
- `production/qwen_pipeline/cli.py`: Terminal-based conversation loop

## Testing Conversation Patterns

```python
from unittest.mock import patch, MagicMock

def test_multi_turn_conversation():
    """Test conversation history preservation."""
    agent = MagicMock()
    agent.run.return_value = iter([[{"role": "assistant", "content": "Response"}]])

    messages = []
    messages.append({"role": "user", "content": "Query 1"})
    responses = list(agent.run(messages=messages))
    messages.extend(responses)

    messages.append({"role": "user", "content": "Query 2"})
    responses = list(agent.run(messages=messages))
    messages.extend(responses)

    # Verify history preserved
    assert len(messages) == 4  # 2 user + 2 assistant
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
```

**See:** `production/tests/test_pipeline.py` for HITL conversation tests.

---

**Last updated:** 2025-11-06
**Qwen-Agent version:** Compatible with official patterns as of Nov 2025
