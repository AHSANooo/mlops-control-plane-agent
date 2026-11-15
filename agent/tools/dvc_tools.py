"""DVC Data Management Agent Tool Module."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from agent.config import settings
from agent.schemas import ToolResult
from agent.telemetry import telemetry


def _run_git_or_dvc(cmd: list[str], cwd: str | None = None) -> tuple[int, str, str]:
    """Execute git or dvc command safely with shell=False."""
    work_dir = cwd or settings.dvc_repo_dir
    proc = subprocess.run(
        cmd,
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def list_tracked_datasets() -> ToolResult:
    """Identify and list all datasets tracked by DVC in the repository."""
    start_time = time.perf_counter()
    tool_name = "dvc_list_tracked_datasets"
    try:
        repo_dir = Path(settings.dvc_repo_dir)
        dvc_files = list(repo_dir.glob("**/*.dvc"))
        datasets: list[dict[str, Any]] = []

        for df_path in dvc_files:
            try:
                with open(df_path, encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    outs = content.get("outs", [])
                    for out in outs:
                        datasets.append(
                            {
                                "dvc_file": str(df_path.relative_to(repo_dir)),
                                "tracked_path": out.get("path"),
                                "md5_hash": out.get("md5"),
                                "size_bytes": out.get("size"),
                            }
                        )
            except Exception as read_err:
                telemetry.log_event("TOOL_WARN", tool_name=tool_name, extra={"warn": str(read_err)})

        duration_ms = (time.perf_counter() - start_time) * 1000
        result = ToolResult(
            success=True,
            tool_name=tool_name,
            data={"tracked_datasets": datasets, "total_datasets": len(datasets)},
            error=None,
            duration_ms=duration_ms,
        )
        telemetry.log_event("TOOL_EXEC", tool_name=tool_name, status="SUCCESS", duration_ms=duration_ms)
        return result
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        telemetry.log_event(
            "TOOL_EXEC", tool_name=tool_name, status="ERROR", duration_ms=duration_ms, extra={"err": str(e)}
        )
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Failed to list DVC tracked datasets: {e!s}",
            duration_ms=duration_ms,
        )


def get_dataset_version_info(version_or_tag: str = "HEAD") -> ToolResult:
    """Retrieve the dataset version, Git commit hash, and DVC md5 hash for a given git reference."""
    start_time = time.perf_counter()
    tool_name = "dvc_get_dataset_version_info"
    try:
        # Get commit hash
        rc_commit, stdout_commit, stderr_commit = _run_git_or_dvc(["git", "rev-parse", "--short", version_or_tag])
        if rc_commit != 0:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                success=False,
                tool_name=tool_name,
                data=None,
                error=f"Git reference '{version_or_tag}' not found: {stderr_commit}",
                duration_ms=duration_ms,
            )
        commit_hash = stdout_commit

        # Get tag
        _, stdout_tag, _ = _run_git_or_dvc(["git", "describe", "--tags", "--exact-match", version_or_tag])
        tag = stdout_tag if stdout_tag else "untagged"

        # Show file content from git at that revision
        dvc_path = "data/dataset.csv.dvc"
        rc_show, stdout_show, _ = _run_git_or_dvc(["git", "show", f"{version_or_tag}:{dvc_path}"])
        dvc_hash = "unknown"
        size_bytes = 0
        if rc_show == 0 and stdout_show:
            dvc_meta = yaml.safe_load(stdout_show)
            outs = dvc_meta.get("outs", [{}])
            if outs:
                dvc_hash = outs[0].get("md5", "unknown")
                size_bytes = outs[0].get("size", 0)

        info = {
            "git_reference": version_or_tag,
            "git_commit": commit_hash,
            "git_tag": tag,
            "dvc_file": dvc_path,
            "dvc_data_hash": dvc_hash,
            "size_bytes": size_bytes,
        }
        duration_ms = (time.perf_counter() - start_time) * 1000
        telemetry.log_event("TOOL_EXEC", tool_name=tool_name, status="SUCCESS", duration_ms=duration_ms)
        return ToolResult(
            success=True,
            tool_name=tool_name,
            data=info,
            error=None,
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Failed to get dataset version info: {e!s}",
            duration_ms=duration_ms,
        )


def diff_dataset_versions(target_a: str = "v1.0", target_b: str = "v2.0") -> ToolResult:
    """Compare tracked dataset hashes and metadata between two git tags or commits."""
    start_time = time.perf_counter()
    tool_name = "dvc_diff_dataset_versions"
    try:
        # Run git diff on the .dvc file
        dvc_file = "data/dataset.csv.dvc"
        rc, stdout, stderr = _run_git_or_dvc(["git", "diff", target_a, target_b, "--", dvc_file])
        if rc != 0:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                success=False,
                tool_name=tool_name,
                data=None,
                error=f"Failed to diff versions {target_a} and {target_b}: {stderr}",
                duration_ms=duration_ms,
            )

        info_a = get_dataset_version_info(target_a)
        info_b = get_dataset_version_info(target_b)

        diff_summary = {
            "version_a": info_a.data,
            "version_b": info_b.data,
            "raw_git_diff": stdout,
            "hash_changed": bool(
                info_a.data and info_b.data and info_a.data.get("dvc_data_hash") != info_b.data.get("dvc_data_hash")
            ),
        }
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=True,
            tool_name=tool_name,
            data=diff_summary,
            error=None,
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Failed to compare dataset versions: {e!s}",
            duration_ms=duration_ms,
        )


def check_dataset_status() -> ToolResult:
    """Detect whether local tracked data has changed compared to committed DVC pointer."""
    start_time = time.perf_counter()
    tool_name = "dvc_check_dataset_status"
    try:
        rc, stdout, stderr = _run_git_or_dvc(["dvc", "status"])
        is_clean = "Data and pipelines are up to date." in stdout or (rc == 0 and not stdout)

        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=True,
            tool_name=tool_name,
            data={
                "up_to_date": is_clean,
                "status_output": stdout or stderr,
            },
            error=None,
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Failed to check DVC status: {e!s}",
            duration_ms=duration_ms,
        )


def get_reproduction_commands(version_or_tag: str = "v1.0") -> ToolResult:
    """Explain and return exact, safe CLI commands required to reproduce/checkout a specific dataset version."""
    start_time = time.perf_counter()
    tool_name = "dvc_get_reproduction_commands"
    try:
        commands = [
            f"git checkout {version_or_tag}",
            "dvc pull",
            "dvc checkout",
        ]
        explanation = (
            f"To reproduce dataset version '{version_or_tag}', check out the matching Git commit or tag, "
            f"then pull the underlying dataset blob from the configured DVC remote and check it out into the workspace."
        )
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=True,
            tool_name=tool_name,
            data={
                "target_version": version_or_tag,
                "commands": commands,
                "explanation": explanation,
            },
            error=None,
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Failed to generate reproduction commands: {e!s}",
            duration_ms=duration_ms,
        )
