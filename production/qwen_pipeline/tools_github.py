"""
Custom tool for searching a GitHub repository.
"""

import os
from http import HTTPStatus
from typing import Any

import requests
from qwen_agent.tools.base import BaseTool, register_tool

# GitHub API endpoint for code search
GITHUB_API_URL = "https://api.github.com/search/code"
# It's good practice to use a token for higher rate limits,
# though not strictly required for public repos.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


@register_tool("github_search")
class GitHubSearchTool(BaseTool):
    """
    Tool for searching code within a specific GitHub repository.
    """

    name = "github_search"
    description = (
        "Searches for code within a specific GitHub repository. "
        "Useful for finding functions, classes, or examples in a codebase."
    )

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        self.parameters = [
            {
                "name": "repo",
                "type": "string",
                "description": (
                    "The repository to search in, formatted as 'owner/repo' "
                    "(e.g., 'QwenLM/Qwen-Agent')."
                ),
                "required": True,
            },
            {
                "name": "query",
                "type": "string",
                "description": "The code or text to search for.",
                "required": True,
            },
        ]

    def call(self, params: str | dict, **kwargs: Any) -> str:  # noqa: ARG002
        """
        Executes the GitHub repository search.

        Args:
            params: A string or dictionary containing 'repo' and 'query'.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A formatted string of search results or an error message.
        """
        try:
            params_dict = self._verify_json_format_args(params)
            repo = params_dict.get("repo")
            query = params_dict.get("query")

            if not repo or not query:
                return "Error: Both 'repo' and 'query' parameters are required."

            # Type narrowing: ensure we have strings
            if not isinstance(repo, str) or not isinstance(query, str):
                return "Error: 'repo' and 'query' must be strings."

            search_results = self._search_github(repo, query)
            return self._format_results(search_results)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == HTTPStatus.UNAUTHORIZED:
                return (
                    "Error: GitHub API request failed with a 401 Unauthorized error. "
                    "Please ensure the `GITHUB_TOKEN` environment variable is set with a valid "
                    "GitHub Personal Access Token."
                )
            return f"A GitHub API error occurred: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    def _search_github(self, repo: str, query: str) -> list[dict[str, Any]]:
        """
        Performs the search via the GitHub API.
        """
        headers = {
            "Accept": "application/vnd.github.v3.text-match+json",
        }
        if GITHUB_TOKEN:
            headers["Authorization"] = f"token {GITHUB_TOKEN}"

        # Construct the search query string required by GitHub API
        q = f"{query} repo:{repo}"

        api_params: dict[str, str | int] = {"q": q, "per_page": 5}  # Limit to top 5 results

        response = requests.get(GITHUB_API_URL, headers=headers, params=api_params)
        response.raise_for_status()  # Raise an exception for bad status codes

        json_response: dict[str, Any] = response.json()
        items: list[dict[str, Any]] = json_response.get("items", [])
        return items

    def _format_results(self, search_results: list[dict[str, Any]]) -> str:
        """
        Formats the search results into a readable string for the LLM.
        """
        if not search_results:
            return "No results found."

        formatted_string = f"Found {len(search_results)} results:\n\n"
        for i, item in enumerate(search_results, 1):
            path = item.get("path")
            html_url = item.get("html_url")

            # Extract text matches for a snippet
            snippets = []
            if "text_matches" in item:
                for match in item["text_matches"]:
                    snippets.append(match.get("fragment", ""))

            snippet_str = "\n".join(f"  ... {s.strip()}" for s in snippets)

            formatted_string += f"[{i}] File: `{path}`\n"
            formatted_string += f"    URL: {html_url}\n"
            if snippets:
                formatted_string += f"    Snippets:\n{snippet_str}\n\n"

        return formatted_string
