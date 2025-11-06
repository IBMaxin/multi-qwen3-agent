import pytest
from unittest.mock import MagicMock, patch
import qwen_pipeline.web_rag_ingestion as wr


def test_create_ingestion_agent_returns_agent():
    agent = wr.create_ingestion_agent()
    assert hasattr(agent, "run")
    # function_map is a dict of tool names to callables
    assert hasattr(agent, "function_map")
    # Should include web_search and web_extractor tools
    tool_names = list(agent.function_map.keys())
    assert "web_search" in tool_names
    assert "web_extractor" in tool_names
    assert agent.name == "WebRAGIngestionAgent"


def test_extract_url_with_retry_success():
    agent = MagicMock()
    long_content = "x" * 120  # longer than min_content_length
    agent.run.return_value = iter([{"content": long_content}])
    result = wr.extract_url_with_retry(agent, "http://example.com", max_retries=1, delay=0)
    assert result == long_content


def test_extract_url_with_retry_failure():
    agent = MagicMock()
    agent.run.side_effect = Exception("fail")
    result = wr.extract_url_with_retry(agent, "http://fail.com", max_retries=1, delay=0)
    assert result is None
