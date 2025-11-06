from unittest.mock import patch

from qwen_pipeline.pipeline import run_pipeline


def test_run_pipeline_immediate_timeout_returns_error():
    # Arrange a manager that would yield if called
    class SlowManager:
        def run(self, messages=None):
            yield {"content": "This should not be reached"}

    with patch("qwen_pipeline.pipeline.create_agents", return_value=SlowManager()):
        # Act: timeout_seconds=0 triggers immediate timeout before loop
        result = run_pipeline("test", timeout_seconds=0)

    # Assert
    assert result.startswith("Error: Pipeline timeout")
