"""MCP server exposing rlmgw repository exploration tools to Claude Code."""

import json
import logging

import httpx
from mcp.server.fastmcp import FastMCP

from .config import load_config_from_env
from .context_pack import ContextPackBuilder
from .repo_env import RepoContextTools

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "rlmgw",
    instructions="Repository exploration tools powered by RLM Gateway. "
    "Provides intelligent context selection for large codebases.",
)

# Lazy-initialized singletons
_repo_tools: RepoContextTools | None = None
_config = None


def _get_config():
    global _config
    if _config is None:
        _config = load_config_from_env()
    return _config


def _get_repo_tools() -> RepoContextTools:
    global _repo_tools
    if _repo_tools is None:
        config = _get_config()
        _repo_tools = RepoContextTools(config.repo_root)
    return _repo_tools


def _discover_upstream_model(base_url: str) -> str | None:
    """Discover the model name from a vLLM /v1/models endpoint."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url}/models")
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            if models:
                model_id = models[0]["id"]
                logger.info(f"Discovered upstream model: {model_id}")
                return model_id
    except Exception as e:
        logger.warning(f"Could not discover model from vLLM: {e}")
    return None


@mcp.tool()
def repo_select_context(query: str, max_chars: int | None = None) -> str:
    """Use RLM to intelligently select the most relevant files for a query.

    This tool uses a local LLM (via vLLM) with recursive language model
    exploration to find and return the minimal set of high-signal files
    relevant to your query. Best for complex questions about large codebases.

    Falls back to keyword-based selection if the upstream LLM is unavailable.

    Args:
        query: Natural language description of what you're looking for
            (e.g., "How does authentication work?" or "Find the database models").
        max_chars: Maximum total characters of file content to return.
            Defaults to RLMGW_MAX_CONTEXT_PACK_CHARS (12000).

    Returns:
        JSON with relevant_files, file_contents, and repo_fingerprint.
    """
    config = _get_config()
    tools = _get_repo_tools()

    if max_chars is not None:
        config.max_context_pack_chars = max_chars

    # Dynamically discover the model from vLLM instead of hardcoding
    discovered_model = _discover_upstream_model(config.upstream_base_url)
    if discovered_model:
        config.upstream_model = discovered_model

    # Try RLM-based selection first
    if config.use_rlm_context_selection:
        try:
            from .context_pack_rlm import RLMContextPackBuilder

            builder = RLMContextPackBuilder(tools.collector, config)
            pack = builder.build_from_query(query)
            return json.dumps(
                {
                    "relevant_files": pack.relevant_files,
                    "file_contents": pack.file_contents,
                    "repo_fingerprint": pack.repo_fingerprint,
                    "selection_mode": "rlm",
                    "upstream_model": config.upstream_model,
                }
            )
        except Exception as e:
            logger.warning(f"RLM selection failed, falling back to simple: {e}")

    # Fallback to simple keyword-based selection
    builder = ContextPackBuilder(tools.collector, config.max_context_pack_chars)
    pack = builder.build_from_query(query)
    return json.dumps(
        {
            "relevant_files": pack.relevant_files,
            "file_contents": pack.file_contents,
            "repo_fingerprint": pack.repo_fingerprint,
            "selection_mode": "simple",
            "upstream_model": None,
        }
    )


@mcp.tool()
def repo_tree() -> str:
    """Get the directory tree structure of the repository.

    Returns a hierarchical view of the codebase, useful for understanding
    project layout before diving into specific files.

    Returns:
        JSON nested object representing the directory hierarchy.
        Excludes .git, node_modules, .venv, __pycache__, and other common noise.
    """
    tools = _get_repo_tools()
    tree = tools.get_tree()
    return json.dumps(tree)


@mcp.tool()
def repo_fingerprint() -> str:
    """Get the repository fingerprint (git HEAD hash or content hash).

    Useful for cache invalidation — if the fingerprint changes, the
    repo contents have changed.

    Returns:
        Repository fingerprint string.
    """
    tools = _get_repo_tools()
    return tools.get_fingerprint()


@mcp.tool()
def vllm_status() -> str:
    """Check if the upstream vLLM server is available and what model it serves.

    Returns:
        JSON with status, model name, and base URL.
    """
    config = _get_config()
    base_url = config.upstream_base_url

    model = _discover_upstream_model(base_url)
    if model:
        return json.dumps({"status": "available", "model": model, "base_url": base_url})
    return json.dumps({"status": "unavailable", "model": None, "base_url": base_url})


if __name__ == "__main__":
    mcp.run(transport="stdio")
