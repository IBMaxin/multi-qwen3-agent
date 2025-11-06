"""Tests for GitHub search tool.

This test module follows official Qwen-Agent testing patterns and ensures
complete test coverage without requiring actual GitHub API access.

Copyright: Apache License 2.0
"""

from http import HTTPStatus
from unittest.mock import MagicMock, patch

import pytest
import qwen_agent.tools.base as tools_base
import requests

from qwen_pipeline.tools_github import GitHubSearchTool


@pytest.fixture
def github_tool():
    """Create GitHubSearchTool instance for testing."""
    return GitHubSearchTool()


@pytest.fixture
def mock_github_response():
    """Create mock GitHub API response data."""
    return {
        "items": [
            {
                "path": "src/agents/assistant.py",
                "html_url": "https://github.com/QwenLM/Qwen-Agent/blob/main/src/agents/assistant.py",
                "text_matches": [
                    {"fragment": "class Assistant(Agent):\n    def __init__(self):"},
                    {"fragment": "def run(self, messages):"},
                ],
            },
            {
                "path": "examples/assistant_example.py",
                "html_url": "https://github.com/QwenLM/Qwen-Agent/blob/main/examples/assistant_example.py",
                "text_matches": [{"fragment": "bot = Assistant(llm_cfg)"}],
            },
        ]
    }


class TestGitHubSearchToolInit:
    """Test GitHubSearchTool initialization."""

    def test_init_default(self, github_tool):
        """Test initialization with default configuration."""
        assert github_tool.name == "github_search"
        assert "GitHub" in github_tool.description
        assert len(github_tool.parameters) == 2

    def test_init_with_config(self):
        """Test initialization with custom configuration."""
        cfg = {"custom_option": "value"}
        tool = GitHubSearchTool(cfg=cfg)

        assert tool.name == "github_search"
        assert len(tool.parameters) == 2

    def test_parameters_structure(self, github_tool):
        """Test that parameters have correct structure."""
        params = github_tool.parameters

        # Check repo parameter
        repo_param = next(p for p in params if p["name"] == "repo")
        assert repo_param["type"] == "string"
        assert repo_param["required"] is True
        assert "owner/repo" in repo_param["description"]

        # Check query parameter
        query_param = next(p for p in params if p["name"] == "query")
        assert query_param["type"] == "string"
        assert query_param["required"] is True

    def test_tool_registered(self) -> None:
        """Test that tool is properly registered with Qwen-Agent."""
        assert "github_search" in tools_base.TOOL_REGISTRY
        assert tools_base.TOOL_REGISTRY["github_search"] == GitHubSearchTool


class TestGitHubSearchToolCall:
    """Test call method (main entry point)."""

    @patch("qwen_pipeline.tools_github.requests.get")
    def test_call_with_dict_params(self, mock_get, github_tool, mock_github_response):
        """Test call with dictionary parameters."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_github_response
        mock_get.return_value = mock_response

        params = {"repo": "QwenLM/Qwen-Agent", "query": "Assistant"}

        result = github_tool.call(params)

        assert "Found 2 results" in result
        assert "src/agents/assistant.py" in result
        assert "examples/assistant_example.py" in result

    @patch("qwen_pipeline.tools_github.requests.get")
    def test_call_with_string_params(self, mock_get, github_tool, mock_github_response):
        """Test call with JSON string parameters."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_github_response
        mock_get.return_value = mock_response

        params = '{"repo": "QwenLM/Qwen-Agent", "query": "Assistant"}'

        result = github_tool.call(params)

        assert "Found 2 results" in result
        mock_get.assert_called_once()

    @patch("qwen_pipeline.tools_github.requests.get")
    def test_call_missing_repo(self, mock_get, github_tool):
        """Test call with missing repo parameter."""
        params = {"query": "Assistant"}

        result = github_tool.call(params)

        assert "unexpected error" in result or "required" in result
        assert "required" in result.lower()
        mock_get.assert_not_called()

    @patch("qwen_pipeline.tools_github.requests.get")
    def test_call_missing_query(self, mock_get, github_tool):
        """Test call with missing query parameter."""
        params = {"repo": "QwenLM/Qwen-Agent"}

        result = github_tool.call(params)

        assert "unexpected error" in result or "required" in result
        assert "required" in result.lower()
        mock_get.assert_not_called()

    @patch("qwen_pipeline.tools_github.requests.get")
    def test_call_missing_both_params(self, mock_get, github_tool):
        """Test call with both parameters missing."""
        params = {}

        result = github_tool.call(params)

        assert "unexpected error" in result or "required" in result
        assert "required" in result.lower()
        mock_get.assert_not_called()

    @patch("qwen_pipeline.tools_github.requests.get")
    def test_call_handles_kwargs(self, mock_get, github_tool, mock_github_response):
        """Test that **kwargs are accepted (even if unused)."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_github_response
        mock_get.return_value = mock_response

        params = {"repo": "QwenLM/Qwen-Agent", "query": "Assistant"}

        # Should not raise error with extra kwargs
        result = github_tool.call(params, extra_param="ignored", another=123)

        assert "Found 2 results" in result


class TestGitHubSearchToolAPIInteraction:
    """Test _search_github method (API interaction)."""

    @patch("qwen_pipeline.tools_github.requests.get")
    def test_search_github_success(self, mock_get, github_tool, mock_github_response):
        """Test successful GitHub API search."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_github_response
        mock_get.return_value = mock_response

        results = github_tool._search_github("QwenLM/Qwen-Agent", "Assistant")

        # Verify API was called correctly
        mock_get.assert_called_once()
        call_args = mock_get.call_args

        # Check URL
        assert call_args[0][0] == "https://api.github.com/search/code"

        # Check query parameters
        params = call_args[1]["params"]
        assert "Assistant repo:QwenLM/Qwen-Agent" in params["q"]
        assert params["per_page"] == 5

        # Check results
        assert len(results) == 2
        assert results[0]["path"] == "src/agents/assistant.py"

    @patch("qwen_pipeline.tools_github.GITHUB_TOKEN", "fake_token_12345")
    @patch("qwen_pipeline.tools_github.requests.get")
    def test_search_github_with_token(self, mock_get, github_tool):
        """Test API search with authentication token."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"items": []}
        mock_get.return_value = mock_response

        github_tool._search_github("owner/repo", "query")

        # Verify Authorization header was set
        call_args = mock_get.call_args
        headers = call_args[1]["headers"]
        assert "Authorization" in headers
        assert headers["Authorization"] == "token fake_token_12345"

    @patch("qwen_pipeline.tools_github.GITHUB_TOKEN", None)
    @patch("qwen_pipeline.tools_github.requests.get")
    def test_search_github_without_token(self, mock_get, github_tool):
        """Test API search without authentication token."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"items": []}
        mock_get.return_value = mock_response

        github_tool._search_github("owner/repo", "query")

        # Verify Authorization header was NOT set
        call_args = mock_get.call_args
        headers = call_args[1]["headers"]
        assert "Authorization" not in headers

    @patch("qwen_pipeline.tools_github.requests.get")
    def test_search_github_empty_results(self, mock_get, github_tool):
        """Test handling of empty search results."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"items": []}
        mock_get.return_value = mock_response

        results = github_tool._search_github("owner/repo", "nonexistent")

        assert results == []

    @patch("qwen_pipeline.tools_github.requests.get")
    def test_search_github_no_items_key(self, mock_get, github_tool):
        """Test handling of malformed API response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}  # No 'items' key
        mock_get.return_value = mock_response

        results = github_tool._search_github("owner/repo", "query")

        assert results == []


class TestGitHubSearchToolErrorHandling:
    """Test error handling in call method."""

    @patch("qwen_pipeline.tools_github.requests.get")
    def test_call_handles_401_unauthorized(self, mock_get, github_tool):
        """Test handling of 401 Unauthorized error."""
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.UNAUTHORIZED
        mock_get.side_effect = requests.exceptions.HTTPError(response=mock_response)

        params = {"repo": "QwenLM/Qwen-Agent", "query": "test"}
        result = github_tool.call(params)

        assert "401 Unauthorized" in result
        assert "GITHUB_TOKEN" in result

    @patch("qwen_pipeline.tools_github.requests.get")
    def test_call_handles_403_forbidden(self, mock_get, github_tool):
        """Test handling of 403 Forbidden error (rate limit)."""
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.FORBIDDEN
        mock_get.side_effect = requests.exceptions.HTTPError(response=mock_response)

        params = {"repo": "QwenLM/Qwen-Agent", "query": "test"}
        result = github_tool.call(params)

        assert "GitHub API error" in result

    @patch("qwen_pipeline.tools_github.requests.get")
    def test_call_handles_404_not_found(self, mock_get, github_tool):
        """Test handling of 404 Not Found error."""
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.NOT_FOUND
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response

        params = {"repo": "nonexistent/repo", "query": "test"}
        result = github_tool.call(params)

        assert "GitHub API error" in result

    @patch("qwen_pipeline.tools_github.requests.get")
    def test_call_handles_connection_error(self, mock_get, github_tool):
        """Test handling of network connection errors."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Network unreachable")

        params = {"repo": "QwenLM/Qwen-Agent", "query": "test"}
        result = github_tool.call(params)

        assert "unexpected error" in result.lower()

    @patch("qwen_pipeline.tools_github.requests.get")
    def test_call_handles_timeout(self, mock_get, github_tool):
        """Test handling of request timeout."""
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        params = {"repo": "QwenLM/Qwen-Agent", "query": "test"}
        result = github_tool.call(params)

        assert "unexpected error" in result.lower()


class TestGitHubSearchToolFormatResults:
    """Test _format_results method (output formatting)."""

    def test_format_results_empty(self, github_tool):
        """Test formatting of empty results."""
        result = github_tool._format_results([])

        assert result == "No results found."

    def test_format_results_single(self, github_tool):
        """Test formatting of single result."""
        items = [
            {
                "path": "src/agent.py",
                "html_url": "https://github.com/owner/repo/blob/main/src/agent.py",
                "text_matches": [{"fragment": "def run():"}],
            }
        ]

        result = github_tool._format_results(items)

        assert "Found 1 results" in result
        assert "src/agent.py" in result
        assert "https://github.com/owner/repo" in result
        assert "def run():" in result

    def test_format_results_multiple(self, github_tool, mock_github_response):
        """Test formatting of multiple results."""
        result = github_tool._format_results(mock_github_response["items"])

        assert "Found 2 results" in result
        assert "[1]" in result
        assert "[2]" in result
        assert "src/agents/assistant.py" in result
        assert "examples/assistant_example.py" in result

    def test_format_results_no_snippets(self, github_tool):
        """Test formatting when text_matches are absent."""
        items = [
            {
                "path": "README.md",
                "html_url": "https://github.com/owner/repo/blob/main/README.md",
            }
        ]

        result = github_tool._format_results(items)

        assert "Found 1 results" in result
        assert "README.md" in result
        # Should not crash without text_matches

    def test_format_results_empty_snippets(self, github_tool):
        """Test formatting with empty text_matches array."""
        items = [
            {
                "path": "src/test.py",
                "html_url": "https://github.com/owner/repo/blob/main/src/test.py",
                "text_matches": [],
            }
        ]

        result = github_tool._format_results(items)

        assert "Found 1 results" in result
        assert "src/test.py" in result

    def test_format_results_multiple_snippets(self, github_tool):
        """Test formatting with multiple snippets per file."""
        items = [
            {
                "path": "src/main.py",
                "html_url": "https://github.com/owner/repo/blob/main/src/main.py",
                "text_matches": [
                    {"fragment": "  class MyClass:"},
                    {"fragment": "  def method1():"},
                    {"fragment": "  def method2():"},
                ],
            }
        ]

        result = github_tool._format_results(items)

        assert "class MyClass:" in result
        assert "def method1():" in result
        assert "def method2():" in result


class TestGitHubSearchToolIntegration:
    """Integration-style tests (still mocked but more comprehensive)."""

    @patch("qwen_pipeline.tools_github.requests.get")
    def test_full_search_workflow(self, mock_get, github_tool):
        """Test complete search workflow from call to formatted output."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "items": [
                {
                    "path": "qwen_agent/agents/assistant.py",
                    "html_url": "https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/agents/assistant.py",
                    "text_matches": [
                        {"fragment": "class Assistant(FnCallAgent):"},
                        {"fragment": "    def _run(self, messages):"},
                    ],
                }
            ]
        }
        mock_get.return_value = mock_response

        params = {"repo": "QwenLM/Qwen-Agent", "query": "Assistant agent"}

        result = github_tool.call(params)

        # Verify complete workflow
        assert "Found 1 results" in result
        assert "qwen_agent/agents/assistant.py" in result
        assert "class Assistant(FnCallAgent):" in result
        assert "def _run(self, messages):" in result
        assert "https://github.com/QwenLM/Qwen-Agent" in result
