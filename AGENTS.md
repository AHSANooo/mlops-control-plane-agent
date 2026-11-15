# MLOps AI Agent Engineering Guardrails & Architecture Standards

## Core Engineering Principles
1. **Strict Type Safety & Schemas**:
   - All Python functions must declare strict type annotations.
   - All tool inputs and outputs must use Pydantic V2 schemas (`BaseModel`).
2. **Subprocess Security**:
   - NEVER use `shell=True` in `subprocess.run`, `subprocess.Popen`, or any system execution calls. Always pass arguments as explicit lists.
3. **Structured Tool Returns & Error Resilience**:
   - Every tool must return a standardized `ToolResult(success: bool, tool_name: str, data: Any, error: Optional[str], duration_ms: float)`.
   - Tools must catch exceptions internally and return failure in `ToolResult`, NEVER raising raw unhandled exceptions directly to the LLM orchestrator.
4. **Safety & Guardrails for State Mutations**:
   - All Kubernetes state-mutating operations (e.g. restart, delete, scale, roll back) must strictly verify:
     - `namespace == "mlops-prod"`
     - `user_confirmed is True`
   - Any mutation request outside `mlops-prod` or where `user_confirmed is not True` must immediately fail with an explicit safety rejection error.
   - Apache Airflow DAG trigger operations must also require `user_confirmed is True`.
5. **Grounded Multi-Tool Reasoning**:
   - The LLM orchestrator must never hallucinate, invent, or speculate metrics, hashes, pod names, or logs when a tool fails or is unreachable.
   - For status or retraining inquiries, the agent must categorize evidence into:
     - **Known Facts** (retrieved directly from tools)
     - **Uncertainties** (missing, unavailable, or ambiguous data)
     - **Recommended Actions** (recommended next steps, never automatically executing mutations)
   - Highest F1 score alone does not imply production readiness; DVC data lineage and Kubernetes deployment state must be verified.
6. **Telemetry & Privacy**:
   - All tool invocations and orchestrator calls must log structured JSON telemetry recording `request_id`, `tool_name`, `status`, and `duration_ms`.
   - Secrets, API keys, passwords, and tokens must be masked/redacted using regex filters before writing logs.
