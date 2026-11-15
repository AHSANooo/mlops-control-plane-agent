"""Kubernetes Operations and Diagnostics Agent Tool Module."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import Any, cast

from kubernetes.client.exceptions import ApiException

from agent.config import settings
from agent.schemas import ToolResult
from agent.telemetry import telemetry
from kubernetes import client, config


def _init_k8s_client() -> bool:
    """Load in-cluster or local kubeconfig context safely."""
    try:
        config.load_incluster_config()
        return True
    except Exception:
        pass

    try:
        if settings.kubeconfig_path:
            config.load_kube_config(config_file=os.path.expanduser(settings.kubeconfig_path))
        else:
            config.load_kube_config()
        return True
    except Exception as e:
        telemetry.log_event("K8S_INIT_WARN", tool_name="k8s_init", extra={"warn": str(e)})
        return False


def _check_namespace_authorized(namespace: str) -> str | None:
    """Verify namespace is within allowed boundary."""
    if namespace not in settings.k8s_namespace_allowlist:
        return (
            f"SAFETY REJECTION: Access to namespace '{namespace}' is forbidden. "
            f"Authorized namespaces: {settings.k8s_namespace_allowlist}"
        )
    return None


def _safe_str(val: Any, default: str = "unknown") -> str:
    """Safely convert value to string or return default if None."""
    if val is None:
        return default
    return str(val)


def list_deployments_and_pods(namespace: str = "mlops-prod") -> ToolResult:
    """List application Pods and Deployments including replica counts and phase statuses."""
    start_time = time.perf_counter()
    tool_name = "k8s_list_deployments_and_pods"

    auth_err = _check_namespace_authorized(namespace)
    if auth_err:
        duration_ms = (time.perf_counter() - start_time) * 1000
        telemetry.log_event("SAFETY_REJECTION", tool_name=tool_name, status="SECURITY_REJECTION")
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=auth_err,
            duration_ms=duration_ms,
        )

    if not _init_k8s_client():
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error="Unable to connect to Kubernetes cluster: kubeconfig could not be loaded",
            duration_ms=duration_ms,
        )

    try:
        apps_v1 = client.AppsV1Api()
        core_v1 = client.CoreV1Api()

        # Query deployments
        deps = apps_v1.list_namespaced_deployment(namespace=namespace)
        deployment_list: list[dict[str, Any]] = []
        for d in cast(list[Any], getattr(deps, "items", [])):
            meta = getattr(d, "metadata", None)
            d_spec = getattr(d, "spec", None)
            d_status = getattr(d, "status", None)
            tmpl = getattr(d_spec, "template", None) if d_spec else None
            tmpl_spec = getattr(tmpl, "spec", None) if tmpl else None
            containers = getattr(tmpl_spec, "containers", []) if tmpl_spec else []
            image_name = containers[0].image if containers else "unknown"

            deployment_list.append(
                {
                    "name": _safe_str(getattr(meta, "name", None)),
                    "replicas_desired": getattr(d_spec, "replicas", 0) if d_spec else 0,
                    "replicas_ready": getattr(d_status, "ready_replicas", 0) or 0 if d_status else 0,
                    "replicas_available": getattr(d_status, "available_replicas", 0) or 0 if d_status else 0,
                    "image": _safe_str(image_name),
                }
            )

        # Query pods
        pods = core_v1.list_namespaced_pod(namespace=namespace)
        pod_list: list[dict[str, Any]] = []
        for p in cast(list[Any], getattr(pods, "items", [])):
            p_meta = getattr(p, "metadata", None)
            p_status = getattr(p, "status", None)
            raw_c_statuses = getattr(p_status, "container_statuses", None) if p_status else None
            container_statuses: list[dict[str, Any]] = []
            restart_count = 0
            is_ready = False
            state_reason = "Running"

            if raw_c_statuses:
                for cs in cast(list[Any], raw_c_statuses):
                    rc = getattr(cs, "restart_count", 0) or 0
                    restart_count += rc
                    if getattr(cs, "ready", False):
                        is_ready = True
                    cs_state = getattr(cs, "state", None)
                    cs_waiting = getattr(cs_state, "waiting", None) if cs_state else None
                    cs_term = getattr(cs_state, "terminated", None) if cs_state else None
                    if cs_waiting:
                        state_reason = _safe_str(getattr(cs_waiting, "reason", None), "Waiting")
                    elif cs_term:
                        state_reason = _safe_str(getattr(cs_term, "reason", None), "Terminated")

                    container_statuses.append(
                        {
                            "name": _safe_str(getattr(cs, "name", None)),
                            "ready": bool(getattr(cs, "ready", False)),
                            "restart_count": rc,
                            "image": _safe_str(getattr(cs, "image", None)),
                            "state": state_reason,
                        }
                    )

            pod_list.append(
                {
                    "name": _safe_str(getattr(p_meta, "name", None)),
                    "phase": _safe_str(getattr(p_status, "phase", None), "Unknown"),
                    "is_ready": is_ready,
                    "restarts": restart_count,
                    "state_reason": state_reason,
                    "pod_ip": _safe_str(getattr(p_status, "pod_ip", None), "None"),
                    "containers": container_statuses,
                }
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=True,
            tool_name=tool_name,
            data={
                "namespace": namespace,
                "deployments": deployment_list,
                "pods": pod_list,
                "total_pods": len(pod_list),
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
            error=f"Kubernetes API error: {e!s}",
            duration_ms=duration_ms,
        )


def get_pod_logs(
    pod_name: str,
    namespace: str = "mlops-prod",
    tail_lines: int = 50,
) -> ToolResult:
    """Retrieve standard output/error logs from a specific container Pod."""
    start_time = time.perf_counter()
    tool_name = "k8s_get_pod_logs"

    auth_err = _check_namespace_authorized(namespace)
    if auth_err:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=auth_err,
            duration_ms=duration_ms,
        )

    if not _init_k8s_client():
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error="Kubernetes client unavailable",
            duration_ms=duration_ms,
        )

    try:
        core_v1 = client.CoreV1Api()
        raw_logs = core_v1.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            tail_lines=tail_lines,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=True,
            tool_name=tool_name,
            data={
                "pod_name": pod_name,
                "namespace": namespace,
                "tail_lines": tail_lines,
                "logs": _safe_str(raw_logs),
            },
            error=None,
            duration_ms=duration_ms,
        )
    except ApiException as api_err:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Kubernetes API returned {api_err.status}: {api_err.reason}",
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Failed to read pod logs: {e!s}",
            duration_ms=duration_ms,
        )


def get_unhealthy_pods_and_events(namespace: str = "mlops-prod") -> ToolResult:
    """Identify unhealthy Pods (CrashLoopBackOff, ErrImagePull, OOMKilled) and warning cluster events."""
    start_time = time.perf_counter()
    tool_name = "k8s_get_unhealthy_pods_and_events"

    auth_err = _check_namespace_authorized(namespace)
    if auth_err:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=auth_err,
            duration_ms=duration_ms,
        )

    if not _init_k8s_client():
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error="Kubernetes client unavailable",
            duration_ms=duration_ms,
        )

    try:
        core_v1 = client.CoreV1Api()
        pods = core_v1.list_namespaced_pod(namespace=namespace)
        unhealthy: list[dict[str, Any]] = []

        for p in cast(list[Any], getattr(pods, "items", [])):
            p_meta = getattr(p, "metadata", None)
            p_status = getattr(p, "status", None)
            phase = _safe_str(getattr(p_status, "phase", None), "Unknown")
            is_unhealthy = False
            reasons: list[str] = []

            if phase not in ("Running", "Succeeded"):
                is_unhealthy = True
                reasons.append(f"Phase is {phase}")

            raw_c_statuses = getattr(p_status, "container_statuses", None) if p_status else None
            if raw_c_statuses:
                for cs in cast(list[Any], raw_c_statuses):
                    c_name = _safe_str(getattr(cs, "name", None))
                    c_ready = bool(getattr(cs, "ready", False))
                    c_restarts = getattr(cs, "restart_count", 0) or 0
                    if not c_ready and phase == "Running":
                        is_unhealthy = True
                        reasons.append(f"Container '{c_name}' not ready")
                    if c_restarts > 0:
                        reasons.append(f"Restart count: {c_restarts}")
                    cs_state = getattr(cs, "state", None)
                    cs_waiting = getattr(cs_state, "waiting", None) if cs_state else None
                    if cs_waiting:
                        is_unhealthy = True
                        w_reason = _safe_str(getattr(cs_waiting, "reason", None), "Waiting")
                        w_msg = _safe_str(getattr(cs_waiting, "message", None), "")
                        reasons.append(f"Waiting reason: {w_reason} ({w_msg})")

            if is_unhealthy:
                unhealthy.append(
                    {
                        "pod_name": _safe_str(getattr(p_meta, "name", None)),
                        "phase": phase,
                        "reasons": reasons,
                    }
                )

        # Get warning events
        events = core_v1.list_namespaced_event(namespace=namespace)
        warning_events: list[dict[str, Any]] = []
        for e in cast(list[Any], getattr(events, "items", [])):
            e_type = _safe_str(getattr(e, "type", None), "")
            e_reason = _safe_str(getattr(e, "reason", None), "")
            if e_type == "Warning" or "Failed" in e_reason:
                inv_obj = getattr(e, "involved_object", None)
                obj_name = _safe_str(getattr(inv_obj, "name", None)) if inv_obj else "unknown"
                ts = getattr(e, "last_timestamp", None) or getattr(e, "event_time", None)
                warning_events.append(
                    {
                        "type": e_type,
                        "reason": e_reason,
                        "message": _safe_str(getattr(e, "message", None)),
                        "object": obj_name,
                        "last_timestamp": str(ts),
                    }
                )

        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=True,
            tool_name=tool_name,
            data={
                "namespace": namespace,
                "unhealthy_pods_count": len(unhealthy),
                "unhealthy_pods": unhealthy,
                "warning_events_count": len(warning_events),
                "warning_events": warning_events,
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
            error=f"Failed to check unhealthy pods: {e!s}",
            duration_ms=duration_ms,
        )


def inspect_services(namespace: str = "mlops-prod") -> ToolResult:
    """Inspect Kubernetes Services, exposed ports, and ClusterIP assignments."""
    start_time = time.perf_counter()
    tool_name = "k8s_inspect_services"

    auth_err = _check_namespace_authorized(namespace)
    if auth_err:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=auth_err,
            duration_ms=duration_ms,
        )

    if not _init_k8s_client():
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error="Kubernetes client unavailable",
            duration_ms=duration_ms,
        )

    try:
        core_v1 = client.CoreV1Api()
        svcs = core_v1.list_namespaced_service(namespace=namespace)
        service_list: list[dict[str, Any]] = []
        for s in cast(list[Any], getattr(svcs, "items", [])):
            meta = getattr(s, "metadata", None)
            spec = getattr(s, "spec", None)
            raw_ports = getattr(spec, "ports", []) if spec else []
            ports = [
                {
                    "port": getattr(p, "port", 0),
                    "target_port": str(getattr(p, "target_port", 0)),
                    "protocol": _safe_str(getattr(p, "protocol", "TCP")),
                }
                for p in (raw_ports or [])
            ]
            service_list.append(
                {
                    "name": _safe_str(getattr(meta, "name", None)),
                    "type": _safe_str(getattr(spec, "type", None)),
                    "cluster_ip": _safe_str(getattr(spec, "cluster_ip", None)),
                    "ports": ports,
                }
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=True,
            tool_name=tool_name,
            data={"namespace": namespace, "services": service_list},
            error=None,
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Failed to inspect services: {e!s}",
            duration_ms=duration_ms,
        )


def get_deployed_app_version(namespace: str = "mlops-prod") -> ToolResult:
    """Report the currently deployed application and model version from ConfigMap and Deployment spec."""
    start_time = time.perf_counter()
    tool_name = "k8s_get_deployed_app_version"

    auth_err = _check_namespace_authorized(namespace)
    if auth_err:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=auth_err,
            duration_ms=duration_ms,
        )

    if not _init_k8s_client():
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error="Kubernetes client unavailable",
            duration_ms=duration_ms,
        )

    try:
        core_v1 = client.CoreV1Api()
        apps_v1 = client.AppsV1Api()

        # Read configmap
        cm_data: dict[str, str] = {}
        try:
            cm = core_v1.read_namespaced_config_map(name="model-inference-config", namespace=namespace)
            cm_data = getattr(cm, "data", None) or {}
        except Exception:
            pass

        # Read deployment image
        image_deployed = "unknown"
        try:
            dep = apps_v1.read_namespaced_deployment(name="model-inference", namespace=namespace)
            d_spec = getattr(dep, "spec", None)
            tmpl = getattr(d_spec, "template", None) if d_spec else None
            t_spec = getattr(tmpl, "spec", None) if tmpl else None
            containers = getattr(t_spec, "containers", []) if t_spec else []
            if containers:
                image_deployed = _safe_str(getattr(containers[0], "image", None))
        except Exception:
            pass

        info = {
            "namespace": namespace,
            "deployment_name": "model-inference",
            "deployed_image": image_deployed,
            "model_version": cm_data.get("MODEL_VERSION", "unknown"),
            "mlflow_run_id": cm_data.get("MLFLOW_RUN_ID", "unknown"),
            "service_name": cm_data.get("SERVICE_NAME", "unknown"),
        }

        duration_ms = (time.perf_counter() - start_time) * 1000
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
            error=f"Failed to get deployed app version: {e!s}",
            duration_ms=duration_ms,
        )


def restart_deployment(
    deployment_name: str = "model-inference",
    namespace: str = "mlops-prod",
    user_confirmed: bool = False,
) -> ToolResult:
    """Perform a rolling restart on a deployment. Strictly requires namespace='mlops-prod' and user_confirmed=True."""
    start_time = time.perf_counter()
    tool_name = "k8s_restart_deployment"

    # Guardrail Rule 4: Verify namespace == "mlops-prod"
    if namespace != "mlops-prod":
        duration_ms = (time.perf_counter() - start_time) * 1000
        err_msg = (
            f"SAFETY REJECTION: Mutating deployment in namespace '{namespace}' is forbidden. "
            "All mutating operations are strictly confined to 'mlops-prod'."
        )
        telemetry.log_event("SAFETY_REJECTION", tool_name=tool_name, status="SECURITY_REJECTION")
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=err_msg,
            duration_ms=duration_ms,
        )

    # Guardrail Rule 4: Verify user_confirmed is True
    if not user_confirmed:
        duration_ms = (time.perf_counter() - start_time) * 1000
        err_msg = (
            "SAFETY REJECTION: Cluster state mutation rejected. Restarting deployments modifies live "
            "cluster workloads and requires explicit user confirmation (user_confirmed=True)."
        )
        telemetry.log_event("SAFETY_REJECTION", tool_name=tool_name, status="SECURITY_REJECTION")
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=err_msg,
            duration_ms=duration_ms,
        )

    if not _init_k8s_client():
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error="Kubernetes client unavailable",
            duration_ms=duration_ms,
        )

    try:
        apps_v1 = client.AppsV1Api()
        # Patch pod template annotation to trigger rolling restart
        restart_time_str = datetime.now(UTC).isoformat()
        body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": restart_time_str,
                        }
                    }
                }
            }
        }
        apps_v1.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=body,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000
        telemetry.log_event("MUTATION_EXEC", tool_name=tool_name, status="SUCCESS", duration_ms=duration_ms)
        return ToolResult(
            success=True,
            tool_name=tool_name,
            data={
                "deployment_name": deployment_name,
                "namespace": namespace,
                "restarted_at": restart_time_str,
                "message": f"Deployment '{deployment_name}' rollout successfully triggered in namespace '{namespace}'.",
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
            error=f"Failed to restart deployment: {e!s}",
            duration_ms=duration_ms,
        )
