from typing import Any

import structlog
from qwen_agent.agents import Assistant, GroupChat, ReActChat

from .config import get_llm_config

structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
logger = structlog.get_logger()


def create_agents(tools: list[Any]) -> GroupChat:
    """Create and return the group chat agent (GroupChat).

    Args:
        tools: List of tools for the agents.

    Returns:
        GroupChat instance with configured agents.
    """
    llm_cfg = get_llm_config()
    logger.info("Creating agents.")
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
