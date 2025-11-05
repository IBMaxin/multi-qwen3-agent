import importlib.util as _util
import os
from typing import Any

import structlog
from qwen_agent.agents import Assistant, GroupChat, ReActChat

try:
    # qwen_agent.agent exposes TOOL_REGISTRY for available builtin tools
    from qwen_agent.agent import TOOL_REGISTRY as QWEN_TOOL_REGISTRY
except Exception:  # pragma: no cover - fallback if API changes
    QWEN_TOOL_REGISTRY = {}

from .config import get_llm_config

structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
logger = structlog.get_logger()


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def create_agents(tools: list[Any]) -> GroupChat:
    """Create and return the group chat agent (GroupChat).

    Args:
        tools: List of tools for the agents.

    Returns:
        GroupChat instance with configured agents.
    """
    llm_cfg = get_llm_config()
    logger.info("Creating agents.")

    # Optionally augment provided tools with official Qwen-Agent tools.
    if _env_bool("ENABLE_ALL_OFFICIAL_TOOLS", False):
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
        # Add Visual tools only if explicitly enabled (requires Qwen3-VL models)
        if _env_bool("ENABLE_VL_TOOLS", False):
            official_tools.append("image_zoom_in_qwen3vl")

        # Merge unique items, only if the tool is actually registered in this qwen-agent version
        existing = {(t if isinstance(t, str) else getattr(t, "name", str(type(t)))) for t in tools}
        for t in official_tools:
            if t in existing:
                continue
            if isinstance(QWEN_TOOL_REGISTRY, dict) and t not in QWEN_TOOL_REGISTRY:
                logger.warning("Skipping unregistered tool", tool=t)
                continue
            # Skip tools with heavy optional deps if prerequisites are missing
            if t == "amap_weather":
                # Requires openpyxl for pandas read_excel and AMAP_API_KEY present
                try:
                    has_openpyxl = _util.find_spec("openpyxl") is not None
                except Exception:
                    has_openpyxl = False
                if not (has_openpyxl and os.getenv("AMAP_API_KEY")):
                    logger.warning("Skipping amap_weather: missing openpyxl or AMAP_API_KEY")
                    continue
            tools.append(t)

        # Optionally include MCP servers if enabled
        if _env_bool("ENABLE_MCP", False):
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
    manager = GroupChat(agents=group_agents, llm=llm_cfg)
    logger.info("Agents created.")
    return manager
