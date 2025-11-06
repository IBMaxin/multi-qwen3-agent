import importlib.util as _util
import json
import os
from functools import lru_cache
from typing import Any

import structlog
from qwen_agent.agents import Assistant, GroupChat, ReActChat

try:
    # qwen_agent.agent exposes TOOL_REGISTRY for available builtin tools
    from qwen_agent.agent import TOOL_REGISTRY as QWEN_TOOL_REGISTRY
    _REGISTRY_AVAILABLE = True
except Exception:  # pragma: no cover - fallback if API changes
    QWEN_TOOL_REGISTRY = {}
    _REGISTRY_AVAILABLE = False

from .config import get_llm_config

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
            "Be clear and simple. Avoid unsafe actions."
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
