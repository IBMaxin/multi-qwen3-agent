from unittest.mock import patch

from qwen_pipeline.metrics import get_metrics
from qwen_pipeline.pipeline import run_pipeline


def test_metrics_records_success_and_resets():
    metrics = get_metrics()
    metrics.reset()

    class Manager:
        def run(self, messages=None):
            yield {"content": "ok"}

    with patch("qwen_pipeline.pipeline.create_agents", return_value=Manager()), \
        patch("qwen_pipeline.pipeline.human_approval", return_value="ok"):
        result = run_pipeline("test", timeout_seconds=5)

    assert result == "ok"
    data = metrics.to_dict()
    assert data["total_queries"] == 1
    assert data["failed_queries"] == 0
    assert data["success_rate_percent"] == 100.0

    # Reset and verify cleared
    metrics.reset()
    data_after = metrics.to_dict()
    assert data_after["total_queries"] == 0
