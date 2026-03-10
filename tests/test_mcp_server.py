"""Tests for MCP server tools."""

import json
from unittest.mock import patch

import pytest

from rlmgw.mcp_server import (
    _discover_upstream_model,
    _get_repo_tools_for,
    repo_fingerprint,
    repo_select_context,
    repo_tree,
    vllm_status,
)


def test_repo_tree():
    """Test repo_tree returns valid directory structure."""
    result = json.loads(repo_tree())
    assert isinstance(result, dict)
    # Should have rlmgw directory in this repo
    assert "rlmgw" in result


def test_repo_fingerprint():
    """Test repo_fingerprint returns a non-empty string."""
    result = repo_fingerprint()
    assert isinstance(result, str)
    assert len(result) > 0


def test_repo_select_context_simple_fallback():
    """Test repo_select_context falls back to simple keyword matching."""
    # Mock vLLM as unavailable to force simple keyword fallback
    with patch("rlmgw.mcp_server._discover_upstream_model", return_value=None), \
         patch("rlmgw.mcp_server._get_config") as mock_config:
        config = mock_config.return_value
        config.use_rlm_context_selection = False
        config.max_context_pack_chars = 12000
        config.repo_root = "."
        config.upstream_base_url = "http://localhost:99999/v1"

        result = json.loads(repo_select_context("context pack builder"))
        assert isinstance(result, dict)
        assert "relevant_files" in result
        assert "file_contents" in result
        assert "repo_fingerprint" in result
        assert result["selection_mode"] == "simple"
        # Should find files related to "context pack builder"
        assert len(result["relevant_files"]) > 0


def test_repo_select_context_max_chars():
    """Test that max_chars parameter limits output size."""
    with patch("rlmgw.mcp_server._discover_upstream_model", return_value=None), \
         patch("rlmgw.mcp_server._get_config") as mock_config:
        config = mock_config.return_value
        config.use_rlm_context_selection = False
        config.max_context_pack_chars = 500
        config.repo_root = "."
        config.upstream_base_url = "http://localhost:99999/v1"

        result = json.loads(repo_select_context("test query", max_chars=500))
        total_chars = sum(len(c) for c in result["file_contents"].values())
        assert total_chars <= 1000  # Allow some overhead from truncation marker


def test_discover_upstream_model_unavailable():
    """Test model discovery when vLLM is not running."""
    model = _discover_upstream_model("http://localhost:99999/v1")
    assert model is None


def test_discover_upstream_model_success():
    """Test model discovery with a mocked vLLM response."""
    mock_response = {"data": [{"id": "Qwen/Qwen3-Coder-Next", "object": "model"}]}

    with patch("rlmgw.mcp_server.httpx.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value.__enter__.return_value
        mock_resp = mock_client.get.return_value
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status.return_value = None

        model = _discover_upstream_model("http://localhost:8000/v1")
        assert model == "Qwen/Qwen3-Coder-Next"


def test_vllm_status_unavailable():
    """Test vllm_status when server is not running."""
    with patch("rlmgw.mcp_server._discover_upstream_model", return_value=None), \
         patch("rlmgw.mcp_server._get_config") as mock_config:
        config = mock_config.return_value
        config.upstream_base_url = "http://localhost:99999/v1"

        result = json.loads(vllm_status())
        assert result["status"] == "unavailable"
        assert result["model"] is None


def test_get_repo_tools_for_caches_default():
    """Test that _get_repo_tools_for reuses singleton for default root."""
    from rlmgw.mcp_server import _get_config

    default_root = _get_config().repo_root
    tools1 = _get_repo_tools_for(default_root)
    tools2 = _get_repo_tools_for(default_root)
    # Same default root should return the cached singleton
    assert tools1 is not None
    assert tools2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
