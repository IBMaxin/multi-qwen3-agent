import importlib.util as _util
import json
import os
from functools import lru_cache
from typing import Any

import structlog
from qwen_agent.agents import Assistant, GroupChat, ReActChat

from .config import get_llm_config  # Load .env early so qwen_agent sees env at import time

# Import SafeCalculatorTool to ensure @register_tool side-effect registers it with Qwen-Agent.
# This allows using "safe_calculator" by name in function_list.
from .tools import SafeCalculatorTool as _SafeCalculatorTool  # noqa: F401
from .tools_github import GitHubSearchTool as _GitHubSearchTool  # noqa: F401

try:
    # qwen_agent.agent exposes TOOL_REGISTRY for available builtin tools
    from qwen_agent.agent import TOOL_REGISTRY as QWEN_TOOL_REGISTRY

    _REGISTRY_AVAILABLE = True
except Exception:  # pragma: no cover - fallback if API changes
    QWEN_TOOL_REGISTRY = {}
    _REGISTRY_AVAILABLE = False


structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
logger = structlog.get_logger()


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _cached_registry_names() -> set[str]:
    if isinstance(QWEN_TOOL_REGISTRY, dict):
        try:
            return set(QWEN_TOOL_REGISTRY.keys())
        except Exception:
            return set()
    return set()


def _normalize_tool_entry(t: Any) -> str:
    # Normalize tool entry to a stable string for caching/signature
    if isinstance(t, str):
        return t
    if isinstance(t, dict):
        try:
            return json.dumps(t, sort_keys=True)
        except Exception:
            return str(t)
    name = getattr(t, "name", None)
    if isinstance(name, str):
        return name
    return str(type(t))


def _tools_signature(tools: list[Any]) -> tuple[str, ...]:
    return tuple(_normalize_tool_entry(t) for t in tools)


@lru_cache(maxsize=8)
def _create_agents_cached(sig: tuple[str, ...], flags: tuple[bool, bool, bool]) -> GroupChat:
    # Reconstruct tools list from signature for ReActChat (strings are fine for qwen-agent tools)
    llm_cfg = get_llm_config()
    tools: list[Any] = list(sig)

    enable_all, enable_vl, enable_mcp = flags

    if enable_all:
        official_tools: list[str] = [
            "code_interpreter",
            "web_search",
            "web_extractor",
            "image_gen",
            "image_search",
            "doc_parser",
            "simple_doc_parser",
            "extract_doc_vocabulary",
            "retrieval",
            "storage",
            "amap_weather",
        ]
        if enable_vl:
            official_tools.append("image_zoom_in_qwen3vl")

        existing = set(sig)
        registry = _cached_registry_names()
        for t in official_tools:
            if t in existing:
                continue
            # If the official tool registry is available, strictly honor it.
            # Empty registry => skip all unlisted tools.
            if _REGISTRY_AVAILABLE and t not in registry:
                logger.warning("Skipping unregistered tool", tool=t)
                continue
            if t == "amap_weather":
                try:
                    has_openpyxl = _util.find_spec("openpyxl") is not None
                except Exception:
                    has_openpyxl = False
                if not (has_openpyxl and os.getenv("AMAP_API_KEY")):
                    logger.warning("Skipping amap_weather: missing openpyxl or AMAP_API_KEY")
                    continue
            tools.append(t)

        if enable_mcp:
            mcp_config = {
                "mcpServers": {
                    "time": {"command": "uvx", "args": ["mcp-server-time", "--local-timezone=UTC"]},
                    "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
                }
            }
            tools.insert(0, mcp_config)

    planner = Assistant(
        llm=llm_cfg,
        system_message=(
            "You are a Planner. Break the task into steps. "
            "Be clear and simple. Avoid unsafe actions. "
            "If the user simply greets (e.g., 'hi', 'hello'), reply with a brief "
            "friendly greeting and stop."
        ),
        name="planner",
    )
    coder = ReActChat(
        llm=llm_cfg,
        function_list=tools,
        system_message=(
            "You are a Coder. Write and run safe code. "
            "Use tools. Explain steps. No file/system access."
        ),
        name="coder",
    )
    reviewer = Assistant(
        llm=llm_cfg,
        system_message="You are a Reviewer. Check for errors. Suggest fixes. Ensure safety.",
        name="reviewer",
    )
    group_agents: list[Assistant | ReActChat] = [planner, coder, reviewer]
    return GroupChat(agents=group_agents, llm=llm_cfg)


def create_agents(tools: list[Any]) -> GroupChat:
    """Create and return the group chat agent (GroupChat).

    Args:
        tools: List of tools for the agents.

    Returns:
        GroupChat instance with configured agents.
    """
    logger.info("Creating agents.")
    sig = _tools_signature(tools)
    flags = (
        _env_bool("ENABLE_ALL_OFFICIAL_TOOLS", False),
        _env_bool("ENABLE_VL_TOOLS", False),
        _env_bool("ENABLE_MCP", False),
    )
    logger.info("Agents created.")
    return _create_agents_cached(sig, flags)


def _has_any_env_keys(names: list[str]) -> bool:
    """Return True if any of the given environment variables are set to a non-empty value."""
    for n in names:
        v = os.getenv(n)
        if v and v.strip():
            return True
    return False


def create_agents_all_tools_no_keys(
    *, enable_vl: bool = False, enable_mcp: bool = False
) -> GroupChat:
    """Create a GroupChat agent with all available official tools, excluding those requiring
    API keys that are not configured.

    This uses the official Qwen-Agent registry (when available) to include only registered tools,
    and excludes tools known to commonly require API keys (e.g., image search/generation) unless
    corresponding keys are present in the environment. Always includes our registered
    `safe_calculator` tool.

    Args:
        enable_vl: Include vision-related tool(s) if available in registry.
        enable_mcp: Include example MCP servers configuration.

    Returns:
        GroupChat instance configured with filtered toolset.
    """
    registry = _cached_registry_names()

    # Start with baseline tools we always want
    tools: list[str | dict[str, Any]] = [
        "code_interpreter",
        "safe_calculator",  # our registered tool
    ]

    # Candidate official tools to include when available
    official_tools: list[str] = [
        # web_search requires a provider API key; we'll add it conditionally below
        # "web_search",
        "web_extractor",
        "doc_parser",
        "simple_doc_parser",
        "extract_doc_vocabulary",
        "retrieval",
        "storage",
        "github_search",  # Add our custom GitHub search tool
        # Conditional ones handled below:
        # "image_gen",
        # "image_search",
        # "amap_weather",
    ]

    if enable_vl:
        official_tools.append("image_zoom_in_qwen3vl")

    # Conditional tool gating based on environment
    # - amap_weather requires openpyxl + AMAP_API_KEY
    # - image_gen typically needs a model provider key (e.g., DASHSCOPE_API_KEY or OPENAI_API_KEY)
    # - image_search often needs a search API key (SERPAPI_API_KEY or BING_API_KEY)
    # - web_search requires SERPER_API_KEY (Serper.dev) per Qwen-Agent web_search tool
    try:
        has_openpyxl = _util.find_spec("openpyxl") is not None
    except Exception:
        has_openpyxl = False

    if has_openpyxl and os.getenv("AMAP_API_KEY"):
        official_tools.append("amap_weather")

    if _has_any_env_keys(["DASHSCOPE_API_KEY", "OPENAI_API_KEY"]):
        official_tools.append("image_gen")

    if _has_any_env_keys(["SERPAPI_API_KEY", "BING_API_KEY"]):
        official_tools.append("image_search")

    # web_search (Qwen-Agent) uses Serper.dev; require SERPER_API_KEY explicitly
    if _has_any_env_keys(["SERPER_API_KEY"]):
        official_tools.append("web_search")

    # Add only tools that are registered and not already present
    existing = set(_tools_signature(tools))
    for t in official_tools:
        if _REGISTRY_AVAILABLE and t not in registry:
            logger.warning("Skipping unregistered tool", tool=t)
            continue
        if t in existing:
            continue
        tools.append(t)
        existing.add(t)

    # Optional: MCP servers
    if enable_mcp:
        mcp_config: dict[str, Any] = {
            "mcpServers": {
                "time": {"command": "uvx", "args": ["mcp-server-time", "--local-timezone=UTC"]},
                "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
            }
        }
        tools.insert(0, mcp_config)

    # Build directly with the composed tool list; pass flags with enable_all=False to avoid
    # further automatic augmentation inside the cache builder.
    sig = _tools_signature([*tools])
    flags = (
        False,  # enable_all
        bool(enable_vl),
        bool(enable_mcp),
    )
    return _create_agents_cached(sig, flags)


def create_fast_agent(tools: list[Any]) -> ReActChat:
    """Create a single ReAct agent (no GroupChat) for faster responses.

    Uses the default ReAct system message which includes proper tool usage formatting.
    This follows the official Qwen-Agent pattern where ReActChat handles its own prompt.

    Args:
        tools: List of tools for the agent (strings, dict tool configs, or tool instances).

    Returns:
        ReActChat instance with default ReAct prompt formatting.
    """
    llm_cfg = get_llm_config()
    # Lower temperature to make tool-calling more deterministic and reliable.
    # A high temperature encourages "creative" text, which can lead the model
    # to generate conversational replies instead of tool calls.
    llm_cfg["generate_cfg"]["temperature"] = 0.1
    return ReActChat(
        llm=llm_cfg,
        function_list=tools,
        # No system_message override - use DEFAULT_SYSTEM_MESSAGE with built-in ReAct format
        name="assistant",
    )


def create_fast_agent_all_tools_no_keys(
    *, enable_vl: bool = False, enable_mcp: bool = False
) -> ReActChat:
    """Create a single ReAct agent with all available official tools, excluding those
    requiring API keys that are not configured.

    Mirrors create_agents_all_tools_no_keys but returns a ReActChat instead of GroupChat.
    Always includes our registered `safe_calculator` tool and `code_interpreter`.
    """
    registry = _cached_registry_names()

    tools: list[str | dict[str, Any]] = [
        "code_interpreter",
        "safe_calculator",
    ]

    official_tools: list[str] = [
        # web_search will be added conditionally when keys are present
        # "web_search",
        "web_extractor",
        "doc_parser",
        "simple_doc_parser",
        "extract_doc_vocabulary",
        "retrieval",
        "storage",
        "github_search",  # Add our custom GitHub search tool
    ]

    if enable_vl:
        official_tools.append("image_zoom_in_qwen3vl")

    try:
        has_openpyxl = _util.find_spec("openpyxl") is not None
    except Exception:
        has_openpyxl = False

    if has_openpyxl and os.getenv("AMAP_API_KEY"):
        official_tools.append("amap_weather")
    if _has_any_env_keys(["DASHSCOPE_API_KEY", "OPENAI_API_KEY"]):
        official_tools.append("image_gen")
    if _has_any_env_keys(["SERPAPI_API_KEY", "BING_API_KEY"]):
        official_tools.append("image_search")
    # web_search (Qwen-Agent) uses Serper.dev; require SERPER_API_KEY explicitly
    if _has_any_env_keys(["SERPER_API_KEY"]):
        official_tools.append("web_search")

    existing = set(_tools_signature(tools))
    for t in official_tools:
        if _REGISTRY_AVAILABLE and t not in registry:
            logger.warning("Skipping unregistered tool", tool=t)
            continue
        if t in existing:
            continue
        tools.append(t)
        existing.add(t)

    if enable_mcp:
        mcp_config: dict[str, Any] = {
            "mcpServers": {
                "time": {"command": "uvx", "args": ["mcp-server-time", "--local-timezone=UTC"]},
                "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
            }
        }
        tools.insert(0, mcp_config)

    return create_fast_agent(tools)
