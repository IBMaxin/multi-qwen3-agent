from qwen_pipeline.agent import create_agents


def test_create_agents() -> None:
    tools = ["code_interpreter"]
    manager = create_agents(tools)
    assert manager is not None
    assert len(manager.agents) == 3
