from unittest.mock import patch

from qwen_pipeline.pipeline import run_pipeline_structured


def test_run_pipeline_structured_success():
    class Manager:
        def run(self, messages=None):
            yield {"content": "final"}

    with patch("qwen_pipeline.pipeline.create_agents", return_value=Manager()), \
        patch("qwen_pipeline.pipeline.human_approval", return_value="final"):
        res = run_pipeline_structured("test", timeout_seconds=5)

    assert res.success is True
    assert res.response == "final"
    assert res.query == "test"
    assert res.duration_seconds >= 0.0
    assert res.tools_available >= 1


def test_run_pipeline_structured_agent_error():
    with patch("qwen_pipeline.pipeline.create_agents", side_effect=RuntimeError("boom")):
        res = run_pipeline_structured("oops", timeout_seconds=5)

    assert res.success is False
    assert res.response == ""
    assert "boom" in (res.error or "")
