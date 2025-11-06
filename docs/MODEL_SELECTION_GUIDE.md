# Model Selection Guide for Multi-Agent Systems

Performance analysis and recommendations for assigning Qwen models to different agent roles in your multi-agent pipeline.

## Available Models Analysis

Based on your `ollama list` output:

| Model | Size | Parameters | Context | Specialization | Speed |
|-------|------|------------|---------|----------------|-------|
| `qwen3:8b` | 5.2 GB | 8B | 32K | General-purpose | Medium |
| `qwen2.5-coder:7b-instruct` | 4.7 GB | 7B | 128K | Code generation | Medium |
| `qwen3:4b-thinking-2507` | 2.5 GB | 4B | 32K | Reasoning/planning | Fast |
| `qwen3:4b-instruct-2507` | 2.5 GB | 4B | 32K | Instruction following | Fast |
| `qwen3:4b` | 2.5 GB | 4B | 32K | General (smaller) | Fast |
| `qwen3-vl:latest` | 6.1 GB | 8B | 32K | Vision-language | Medium |
| `phi4-mini:3.8b` | 2.5 GB | 3.8B | 16K | Compact reasoning | Fast |
| `qwen3-embedding:4b` | 2.5 GB | 4B | 32K | Embeddings (balanced) | Fast |
| `qwen3-embedding:0.6b` | 639 MB | 0.6B | 32K | Embeddings (fast) | Very Fast |

**Note:** The model names used in this guide are examples. You should replace them with the actual model names available in your environment, which can be configured in your `.env` file.

## Agent Role Analysis

### 1. **Planner Agent** (High-Level Strategy)

**Requirements:**
- Strong reasoning and planning capabilities
- Long-context understanding
- Ability to break down complex tasks
- Doesn't need code generation

**Recommended Models (Priority Order):**

1. **`qwen3:4b-thinking-2507` ⭐ BEST**
   - ✅ Specifically designed for reasoning/thinking
   - ✅ Fast response (2.5 GB)
   - ✅ 32K context (sufficient for planning)
   - ✅ Lower resource usage = more capacity for other agents
   - ⚠️ May lack some domain knowledge vs 8B

2. **`qwen3:8b`** (Fallback for complex planning)
   - ✅ More knowledge and reasoning depth
   - ✅ Better at complex multi-step planning
   - ❌ Slower, uses more resources
   - Use when: Planning highly complex workflows

3. **`phi4-mini:3.8b`** (Ultra-fast alternative)
   - ✅ Excellent reasoning for size
   - ✅ Fastest inference
   - ❌ Smaller context (16K)
   - Use when: Speed is critical, simple plans

**Configuration Example:**
```python
# Load model name from environment
model_name = os.getenv('MODEL_NAME_PLANNER', 'qwen3:4b-thinking-2507')

planner = Assistant(
    llm={
        'model': model_name,
        'model_server': 'http://localhost:11434/v1',
        'api_key': 'EMPTY',
        'generate_cfg': {
            'temperature': 0.7,  # Creative planning
            'top_p': 0.9,
            'max_tokens': 2000,  # Plans don't need to be long
        }
    },
    name='planner',
    system_message='You are a strategic planner. Break down tasks into clear steps.',
)
```

---

### 2. **Coder Agent** (Code Generation & Execution)

**Requirements:**
- Deep code understanding
- Multi-language support
- Long context (for reading large codebases)
- Precise syntax generation

**Recommended Models (Priority Order):**

1. **`qwen2.5-coder:7b-instruct` ⭐ BEST**
   - ✅ Specialized for code generation
   - ✅ 128K context (can read entire files)
   - ✅ Supports 92+ programming languages
   - ✅ Trained on code-specific datasets
   - ✅ Better at debugging and code reasoning

2. **`qwen3:8b`** (General fallback)
   - ✅ Good code capabilities
   - ✅ Better at explaining code to humans
   - ❌ Shorter context (32K)
   - Use when: Need code + natural language explanation

3. **`qwen3:4b-instruct-2507`** (Fast prototyping)
   - ✅ Faster for simple code tasks
   - ✅ Good instruction following
   - ❌ Less specialized code knowledge
   - Use when: Simple scripts, config files

**Configuration Example:**
```python
# Load model name from environment
model_name = os.getenv('MODEL_NAME_CODER', 'qwen2.5-coder:7b-instruct')

coder = ReActChat(
    llm={
        'model': model_name,
        'model_server': 'http://localhost:11434/v1',
        'api_key': 'EMPTY',
        'generate_cfg': {
            'temperature': 0.2,  # Low for precise code
            'top_p': 0.95,
            'max_tokens': 4096,  # Longer code blocks
            'stop': ['```\n\n', '</code>'],  # Stop at code end
        }
    },
    name='coder',
    function_list=['code_interpreter'],
    system_message='You are an expert programmer. Write clean, efficient, well-documented code.',
)
```

---

### 3. **Reviewer Agent** (Code Review & Quality Check)

**Requirements:**
- Critical thinking
- Pattern recognition (bugs, security issues)
- Code quality assessment
- Doesn't need generation speed

**Recommended Models (Priority Order):**

1. **`qwen3:8b` ⭐ BEST**
   - ✅ More comprehensive knowledge base
   - ✅ Better at identifying subtle issues
   - ✅ Can reference best practices
   - ✅ Good at explaining issues clearly
   - ⚠️ Slower, but that's OK for review

2. **`qwen2.5-coder:7b-instruct`** (Code-focused)
   - ✅ Deep code understanding
   - ✅ 128K context (review full files)
   - ✅ Trained on code patterns
   - ❌ May be less strict than 8B general model
   - Use when: Large files need review

3. **`qwen3:4b-thinking-2507`** (Fast checks)
   - ✅ Fast iteration
   - ✅ Good reasoning for obvious issues
   - ❌ May miss subtle bugs
   - Use when: Quick sanity checks

**Configuration Example:**
```python
# Load model name from environment
model_name = os.getenv('MODEL_NAME_REVIEWER', 'qwen3:8b')

reviewer = Assistant(
    llm={
        'model': model_name,
        'model_server': 'http://localhost:11434/v1',
        'api_key': 'EMPTY',
        'generate_cfg': {
            'temperature': 0.3,  # Balanced: strict but not rigid
            'top_p': 0.9,
            'max_tokens': 2048,
        }
    },
    name='reviewer',
    system_message='You are a senior code reviewer. Check for bugs, security issues, and best practices.',
)
```

---

### 4. **Executor Agent** (Tool Execution & Orchestration)

**Requirements:**
- Fast response time
- Good instruction following
- Minimal reasoning needed
- High throughput

**Recommended Models (Priority Order):**

1. **`qwen3:4b-instruct-2507` ⭐ BEST**
   - ✅ Optimized for instruction following
   - ✅ Fast execution (2.5 GB)
   - ✅ Lower latency for tool calls
   - ✅ Efficient resource usage

2. **`phi4-mini:3.8b`** (Ultra-fast)
   - ✅ Fastest option
   - ✅ Good function calling
   - ❌ Smaller context
   - Use when: Simple tool execution

3. **`qwen3:4b`** (Balanced)
   - ✅ Fast and capable
   - ✅ General-purpose
   - ❌ Less instruction-tuned than 4b-instruct
   - Use when: Need flexibility

**Configuration Example:**
```python
# Load model name from environment
model_name = os.getenv('MODEL_NAME_EXECUTOR', 'qwen3:4b-instruct-2507')

executor = FnCallAgent(
    llm={
        'model': model_name,
        'model_server': 'http://localhost:11434/v1',
        'api_key': 'EMPTY',
        'generate_cfg': {
            'temperature': 0.1,  # Very low - precise execution
            'top_p': 0.95,
            'max_tokens': 1024,  # Short responses
        }
    },
    name='executor',
    function_list=['code_interpreter', 'web_search', 'local_vector_search'],
    system_message='You are a tool executor. Follow instructions precisely and use tools correctly.',
)
```

---

### 5. **RAG/Memory Agent** (Document Retrieval)

**Requirements:**
- Fast retrieval
- Query understanding
- Embedding generation (separate model)

**Recommended Models (Priority Order):**

1. **`qwen3:4b` ⭐ BEST** (for query processing)
   - ✅ Fast query understanding
   - ✅ Efficient for simple retrieval tasks
   - ✅ Works well with RAG pipeline

2. **`qwen3-embedding:4b` ⭐ BEST** (for embeddings - balanced)
   - ✅ Dedicated embedding model
   - ✅ Better quality than 0.6b
   - ✅ Still fast (2.5 GB)
   - ✅ Good balance of speed and accuracy

3. **`qwen3-embedding:0.6b`** (for embeddings - ultra-fast)
   - ✅ Smallest embedding model
   - ✅ Very fast (639 MB)
   - ✅ Use when speed is critical
   - ⚠️ Lower quality than 4b version

**Configuration Example:**
```python
# Memory agent (query understanding)
model_name = os.getenv('MODEL_NAME_RAG', 'qwen3:4b')
memory = Memory(
    llm={
        'model': model_name,
        'model_server': 'http://localhost:11434/v1',
        'api_key': 'EMPTY',
    },
    rag_cfg={
        'max_ref_token': 20000,
        'parser_page_size': 500,
        'rag_searchers': ['local_vector_search', 'keyword_search']
    }
)

# Embedding config (in LocalVectorSearch)
# Recommended: Use 4b for better quality, 0.6b for speed
embedding_cfg = {
    'embedding_model': 'qwen3-embedding:4b',  # or 'qwen3-embedding:0.6b' for faster
    'base_url': 'http://localhost:11434',
}
```

---

### 6. **Vision Agent** (Multimodal Tasks)

**Requirements:**
- Vision-language understanding
- Image analysis
- OCR and diagram interpretation

**Recommended Models:**

1. **`qwen3-vl:latest` ⭐ ONLY OPTION**
   - ✅ Only vision model in your collection
   - ✅ Strong vision-language capabilities
   - ✅ Good at charts, diagrams, screenshots
   - ⚠️ Large (6.1 GB), use only when needed

**Configuration Example:**
```python
# Load model name from environment
model_name = os.getenv('MODEL_NAME_VISION', 'qwen3-vl:latest')

vision_agent = FnCallAgent(
    llm={
        'model_type': 'qwenvl_oai',  # Special type for VL models
        'model': model_name,
        'model_server': 'http://localhost:11434/v1',
        'api_key': 'EMPTY',
    },
    name='vision_analyzer',
    function_list=['image_zoom_in_tool'],
    system_message='You analyze images and diagrams. Describe what you see in detail.',
)
```

---

## Performance Comparison Matrix

### Speed vs Quality Trade-offs

| Agent Role | Recommended Model | Speed | Quality | Resource | Parallel Capacity |
|-----------|-------------------|-------|---------|----------|-------------------|
| **Planner** | qwen3:4b-thinking | ⚡⚡⚡ | ⭐⭐⭐⭐ | Low | High (4 instances) |
| **Coder** | qwen2.5-coder:7b | ⚡⚡ | ⭐⭐⭐⭐⭐ | Medium | Medium (2 instances) |
| **Reviewer** | qwen3:8b | ⚡⚡ | ⭐⭐⭐⭐⭐ | Medium | Medium (2 instances) |
| **Executor** | qwen3:4b-instruct | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Low | High (6 instances) |
| **RAG** | qwen3:4b | ⚡⚡⚡ | ⭐⭐⭐ | Low | High (4 instances) |
| **Embeddings** | qwen3-embedding:4b | ⚡⚡⚡ | ⭐⭐⭐⭐ | Low | High (4 instances) |
| **Vision** | qwen3-vl | ⚡⚡ | ⭐⭐⭐⭐⭐ | High | Low (1 instance) |

---

## Recommended Multi-Agent Architecture

### **Configuration A: Balanced Performance** (16GB RAM)

```python
from qwen_agent.agents import Assistant, ReActChat, GroupChat
import os

# Planner: Fast reasoning
planner = Assistant(
    llm={'model': os.getenv('MODEL_NAME_PLANNER', 'qwen3:4b-thinking-2507'), ...},
    name='planner'
)

# Coder: Specialized code generation
coder = ReActChat(
    llm={'model': os.getenv('MODEL_NAME_CODER', 'qwen2.5-coder:7b-instruct'), ...},
    function_list=['code_interpreter'],
    name='coder'
)

# Reviewer: Comprehensive review
reviewer = Assistant(
    llm={'model': os.getenv('MODEL_NAME_REVIEWER', 'qwen3:8b'), ...},
    name='reviewer'
)

# Orchestrator
manager = GroupChat(
    agents=[planner, coder, reviewer],
    llm={'model': os.getenv('MODEL_NAME_MANAGER', 'qwen3:4b-instruct-2507'), ...},  # Fast coordination
)
```

**Memory Usage:** ~5.2 GB (planner) + 4.7 GB (coder) + 5.2 GB (8b) = **~15 GB**
**Throughput:** ~2-3 requests/min (depending on task complexity)

---

### **Configuration B: Maximum Speed** (12GB RAM)

```python
import os

# Planner: Ultra-fast
planner = Assistant(
    llm={'model': os.getenv('MODEL_NAME_PLANNER_FAST', 'phi4-mini:3.8b'), ...},
    name='planner'
)

# Coder: Balanced speed/quality
coder = ReActChat(
    llm={'model': os.getenv('MODEL_NAME_CODER_FAST', 'qwen3:4b-instruct-2507'), ...},
    function_list=['code_interpreter'],
    name='coder'
)

# Reviewer: Fast checks
reviewer = Assistant(
    llm={'model': os.getenv('MODEL_NAME_REVIEWER_FAST', 'qwen3:4b-thinking-2507'), ...},
    name='reviewer'
)

# Orchestrator
manager = GroupChat(
    agents=[planner, coder, reviewer],
    llm={'model': os.getenv('MODEL_NAME_MANAGER_FAST', 'qwen3:4b'), ...},
)
```

**Memory Usage:** ~2.5 GB × 4 agents = **~10 GB**
**Throughput:** ~5-8 requests/min
**Trade-off:** Lower quality, especially in complex code generation

---

### **Configuration C: Maximum Quality** (24GB+ RAM)

```python
import os

# Planner: Best reasoning
planner = Assistant(
    llm={'model': os.getenv('MODEL_NAME_PLANNER_QUALITY', 'qwen3:8b'), ...},
    name='planner'
)

# Coder: Specialized
coder = ReActChat(
    llm={'model': os.getenv('MODEL_NAME_CODER_QUALITY', 'qwen2.5-coder:7b-instruct'), ...},
    function_list=['code_interpreter'],
    name='coder'
)

# Reviewer: Comprehensive
reviewer = Assistant(
    llm={'model': os.getenv('MODEL_NAME_REVIEWER_QUALITY', 'qwen3:8b'), ...},
    name='reviewer'
)

# Orchestrator: Smart coordination
manager = GroupChat(
    agents=[planner, coder, reviewer],
    llm={'model': os.getenv('MODEL_NAME_MANAGER_QUALITY', 'qwen3:8b'), ...},
)
```

**Memory Usage:** ~5.2 GB × 3 + 4.7 GB = **~20 GB**
**Throughput:** ~1-2 requests/min
**Trade-off:** Slower but highest quality output

---

## Specialized Agent Configurations

### **Document Analysis Pipeline**

```python
import os

# RAG Agent: Fast retrieval
rag_agent = Assistant(
    llm={'model': os.getenv('MODEL_NAME_RAG', 'qwen3:4b'), ...},
    rag_cfg={
        'rag_searchers': ['local_vector_search', 'keyword_search']
    }
)

# Analyzer: Deep understanding
analyzer = Assistant(
    llm={'model': os.getenv('MODEL_NAME_ANALYZER', 'qwen3:8b'), ...},
    name='analyzer'
)

# Summarizer: Fast output
summarizer = Assistant(
    llm={'model': os.getenv('MODEL_NAME_SUMMARIZER', 'qwen3:4b-instruct-2507'), ...},
    name='summarizer'
)
```

**Use Case:** Process large documents, extract insights
**Memory:** ~10 GB
**Strength:** Balance speed (retrieval) with quality (analysis)

---

### **Code Review Pipeline**

```python
import os

# Code Generator
coder = ReActChat(
    llm={'model': os.getenv('MODEL_NAME_CODER', 'qwen2.5-coder:7b-instruct'), ...},
    function_list=['code_interpreter']
)

# Primary Reviewer
reviewer_primary = Assistant(
    llm={'model': os.getenv('MODEL_NAME_REVIEWER', 'qwen3:8b'), ...},
    name='reviewer_primary'
)

# Security Reviewer (fast checks)
reviewer_security = Assistant(
    llm={'model': os.getenv('MODEL_NAME_SECURITY_REVIEWER', 'qwen3:4b-thinking-2507'), ...},
    name='security_checker'
)

# Final Approver
approver = Assistant(
    llm={'model': os.getenv('MODEL_NAME_APPROVER', 'qwen3:8b'), ...},
    name='approver'
)
```

**Use Case:** Generate and thoroughly review code
**Memory:** ~18 GB
**Strength:** Multi-stage review with different focus areas

---

## Performance Benchmarks (Estimated)

### Single Request Latency

| Model | First Token (ms) | Tokens/sec | 500 Token Response (s) |
|-------|------------------|------------|------------------------|
| qwen3:8b | 800-1200 | 15-25 | 20-33 |
| qwen2.5-coder:7b | 700-1000 | 18-28 | 18-28 |
| qwen3:4b-thinking | 400-600 | 25-40 | 12-20 |
| qwen3:4b-instruct | 350-550 | 30-45 | 11-17 |
| qwen3:4b | 350-550 | 30-45 | 11-17 |
| phi4-mini:3.8b | 300-500 | 35-50 | 10-14 |
| qwen3-embedding:4b | 400-600 | N/A | N/A (embeddings only) |
| qwen3-embedding:0.6b | 200-400 | N/A | N/A (embeddings only) |

*Note: Benchmarks vary based on hardware (CPU/GPU), prompt length, and system load*

---

## Temperature & Sampling Recommendations

### By Agent Role

| Agent | Temperature | Top-P | Max Tokens | Reasoning |
|-------|-------------|-------|------------|-----------|
| Planner | 0.7-0.8 | 0.9 | 2000 | Creative problem-solving |
| Coder | 0.1-0.2 | 0.95 | 4096 | Precise syntax, less variance |
| Reviewer | 0.3-0.4 | 0.9 | 2048 | Balanced: strict but flexible |
| Executor | 0.1 | 0.95 | 1024 | Deterministic tool calling |
| RAG | 0.5 | 0.9 | 1024 | Balanced retrieval |
| Vision | 0.3 | 0.9 | 2048 | Accurate descriptions |

---

## Resource Management Strategies

### **Strategy 1: Sequential Execution** (Low RAM)

Run one agent at a time, load/unload models:

```python
def run_pipeline_sequential(task):
    # Load planner
    planner_response = planner.run(task)

    # Unload planner, load coder
    del planner
    coder_response = coder.run(planner_response)

    # Unload coder, load reviewer
    del coder
    reviewer_response = reviewer.run(coder_response)

    return reviewer_response
```

**Pros:** Minimum RAM (only ~5 GB at once)
**Cons:** Slow (model loading overhead)

---

### **Strategy 2: Parallel with Model Pool** (Medium RAM)

Pre-load models, reuse across requests:

```python
# Keep 2-3 models hot
model_pool = {
    'fast': qwen3_4b_instruct,    # Always loaded
    'smart': qwen3_8b,             # Always loaded
    'coder': qwen2_5_coder,        # Load on demand
}

def run_pipeline_pooled(task, priority='balanced'):
    if priority == 'speed':
        return run_with_models(model_pool['fast'])
    elif priority == 'quality':
        return run_with_models(model_pool['smart'])
```

**Pros:** Fast switching, better throughput
**Cons:** ~12-15 GB RAM needed

---

### **Strategy 3: Distributed Agents** (Cloud/Multi-Machine)

Run different agents on different machines:

```
Machine 1 (16GB): Planner (qwen3:4b-thinking) + Executor (qwen3:4b-instruct)
Machine 2 (16GB): Coder (qwen2.5-coder:7b) + Reviewer (qwen3:8b)
Machine 3 (8GB):  RAG (qwen3:4b) + Embeddings (qwen3-embedding:0.6b)
```

**Pros:** Highest throughput, parallel processing
**Cons:** Complex orchestration, network latency

---

## Testing & Optimization

### Benchmark Script

```python
import time
import os
from dotenv import load_dotenv
from qwen_agent.agents import Assistant

load_dotenv()

def benchmark_model(model_name, prompt, num_runs=5):
    """Benchmark model performance."""
    llm_cfg = {'model': model_name, 'model_server': 'http://localhost:11434/v1'}
    agent = Assistant(llm=llm_cfg)

    latencies = []
    for i in range(num_runs):
        start = time.time()
        response = list(agent.run([{'role': 'user', 'content': prompt}]))
        latency = time.time() - start
        latencies.append(latency)
        print(f"Run {i+1}: {latency:.2f}s")

    avg_latency = sum(latencies) / len(latencies)
    print(f"\n{model_name} Average: {avg_latency:.2f}s")
    return avg_latency

# Test different models
# Models are now loaded from environment variables for flexibility
models_to_test = [
    os.getenv('MODEL_NAME_QUALITY', 'qwen3:8b'),
    os.getenv('MODEL_NAME_THINKING', 'qwen3:4b-thinking-2507'),
    os.getenv('MODEL_NAME_INSTRUCT', 'qwen3:4b-instruct-2507'),
    os.getenv('MODEL_NAME_FAST', 'phi4-mini:3.8b')
]

prompt = "Write a Python function to calculate fibonacci numbers."

for model in models_to_test:
    if model: # Ensure the environment variable was set
        benchmark_model(model, prompt)
```

---

## Recommendations Summary

### **Your Optimal Setup (Based on Available Models):**

1. **Planner**: `qwen3:4b-thinking-2507` - Fast reasoning, good planning
2. **Coder**: `qwen2.5-coder:7b-instruct` - Specialized code generation
3. **Reviewer**: `qwen3:8b` - Comprehensive review capability
4. **Executor**: `qwen3:4b-instruct-2507` - Fast tool execution
5. **RAG**: `qwen3:4b` (query) + `qwen3-embedding:4b` (embeddings - recommended)
6. **Embeddings (alternative)**: `qwen3-embedding:0.6b` - Use for ultra-fast retrieval
7. **Vision**: `qwen3-vl:latest` - When image analysis needed

**Total Memory:** ~17 GB (if all loaded simultaneously) + 2.5 GB (qwen3-embedding:4b)
**Recommendation:** Use model pooling with 2-3 hot models
**Embedding Choice:** Use `qwen3-embedding:4b` for production (better quality), `0.6b` for development/speed

---

## Next Steps

1. **Benchmark Your Hardware**: Run the benchmark script to get actual numbers
2. **Profile Memory**: Monitor RAM usage during multi-agent runs
3. **A/B Test Configurations**: Compare quality vs speed for your use cases
4. **Optimize Context Windows**: Adjust `max_tokens` based on actual needs
5. **Monitor Bottlenecks**: Identify slowest agent in pipeline

Would you like me to:
1. Create the benchmark script?
2. Set up model pooling configuration?
3. Implement dynamic model selection based on task complexity?
