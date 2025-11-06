from typing import Any
from unittest.mock import patch

import pytest

import qwen_pipeline.agent as agent_mod
from qwen_pipeline.agent import create_agents


class FakeAssistant:
    def __init__(self, llm: dict[str, Any], system_message: str, name: str) -> None:
        self.llm = llm
        self.system_message = system_message
        self.name = name


class FakeReActChat:
    def __init__(
        self,
        llm: dict[str, Any],
        function_list: list[Any],
        system_message: str,
        name: str,
    ) -> None:
        self.llm = llm
        self.function_list = function_list
        self.system_message = system_message
        self.name = name


class FakeGroupChat:
    def __init__(self, agents: list[Any], llm: dict[str, Any]) -> None:
        self.agents = agents
        self.llm = llm


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure clean environment for each test
    for key in [
        "ENABLE_ALL_OFFICIAL_TOOLS",
        "ENABLE_VL_TOOLS",
        "ENABLE_MCP",
        "AMAP_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)


@patch.object(agent_mod, "Assistant", FakeAssistant)
@patch.object(agent_mod, "ReActChat", FakeReActChat)
@patch.object(agent_mod, "GroupChat", FakeGroupChat)
def test_enable_all_official_tools_and_vl(monkeypatch: pytest.MonkeyPatch) -> None:
    # Enable official tools and VL tools
    monkeypatch.setenv("ENABLE_ALL_OFFICIAL_TOOLS", "1")
    monkeypatch.setenv("ENABLE_VL_TOOLS", "true")

    # Only some tools are registered, others should be skipped with warning
    monkeypatch.setattr(
        agent_mod,
        "QWEN_TOOL_REGISTRY",
        {
            "code_interpreter": object(),
            "web_search": object(),
            "image_zoom_in_qwen3vl": object(),
        },
        raising=True,
    )

    # Force openpyxl to be missing so amap_weather is skipped
    monkeypatch.setattr(agent_mod._util, "find_spec", lambda name: None, raising=True)

    manager = create_agents(["code_interpreter"])  # start with base tool
    assert isinstance(manager, FakeGroupChat)

    # The coder is the second agent
    coder = manager.agents[1]
    assert isinstance(coder, FakeReActChat)

    # Should include code_interpreter (existing), web_search (added), VL zoom (added)
    function_list = coder.function_list
    assert "code_interpreter" in function_list
    assert "web_search" in function_list
    assert "image_zoom_in_qwen3vl" in function_list

    # Should NOT include amap_weather due to missing deps
    assert "amap_weather" not in function_list


@patch.object(agent_mod, "Assistant", FakeAssistant)
@patch.object(agent_mod, "ReActChat", FakeReActChat)
@patch.object(agent_mod, "GroupChat", FakeGroupChat)
def test_enable_mcp_inserts_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # Enable MCP inside the official tools block
    monkeypatch.setenv("ENABLE_ALL_OFFICIAL_TOOLS", "1")
    monkeypatch.setenv("ENABLE_MCP", "yes")
    # Empty registry so official tools are skipped; only MCP insertion matters
    monkeypatch.setattr(agent_mod, "QWEN_TOOL_REGISTRY", {}, raising=True)

    manager = create_agents([])
    coder = manager.agents[1]

    assert isinstance(coder.function_list[0], dict)
    assert "mcpServers" in coder.function_list[0]


@patch.object(agent_mod, "Assistant", FakeAssistant)
@patch.object(agent_mod, "ReActChat", FakeReActChat)
@patch.object(agent_mod, "GroupChat", FakeGroupChat)
def test_unregistered_tools_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    # Enable official tools, but registry is empty => everything except existing should be skipped
    monkeypatch.setenv("ENABLE_ALL_OFFICIAL_TOOLS", "1")
    monkeypatch.setattr(agent_mod, "QWEN_TOOL_REGISTRY", {}, raising=True)
    # Clear cached registry names from previous tests
    agent_mod._cached_registry_names.cache_clear()

    manager = create_agents(["code_interpreter"])  # provide only base
    coder = manager.agents[1]

    # Only the base tool remains; others skipped as unregistered
    assert coder.function_list == ["code_interpreter"]
