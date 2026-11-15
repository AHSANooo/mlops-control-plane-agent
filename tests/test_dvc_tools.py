"""Tests for DVC data management tools."""

from __future__ import annotations

from agent.schemas import ToolResult
from agent.tools import dvc_tools


def test_list_tracked_datasets_returns_tool_result() -> None:
    res = dvc_tools.list_tracked_datasets()
    assert isinstance(res, ToolResult)
    assert res.tool_name == "dvc_list_tracked_datasets"
    assert res.success is True
    assert "tracked_datasets" in res.data


def test_get_dataset_version_info_valid() -> None:
    res = dvc_tools.get_dataset_version_info("HEAD")
    assert isinstance(res, ToolResult)
    assert res.success is True
    assert res.data["git_reference"] == "HEAD"
    assert "dvc_data_hash" in res.data
    assert "git_commit" in res.data


def test_get_dataset_version_info_invalid_ref() -> None:
    res = dvc_tools.get_dataset_version_info("non-existent-tag-xyz-999")
    assert isinstance(res, ToolResult)
    assert res.success is False
    assert "not found" in res.error.lower()


def test_diff_dataset_versions() -> None:
    res = dvc_tools.diff_dataset_versions("v1.0", "v2.0")
    assert isinstance(res, ToolResult)
    assert res.success is True
    assert "hash_changed" in res.data


def test_get_reproduction_commands() -> None:
    res = dvc_tools.get_reproduction_commands("v1.0")
    assert isinstance(res, ToolResult)
    assert res.success is True
    assert "commands" in res.data
    assert any("git checkout v1.0" in cmd for cmd in res.data["commands"])
    assert any("dvc checkout" in cmd for cmd in res.data["commands"])
