import httpx

r_auth = httpx.post(
    "http://localhost:8080/auth/token",
    json={"username": "admin", "password": "3zHWhuXKTm7ADCqr"},
)
token = r_auth.json()["access_token"]
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Inspect POST /api/v2/dags/churn_model_training_pipeline/dagRuns schema
r_spec = httpx.get("http://localhost:8080/openapi.json")
post_schema = r_spec.json()["paths"]["/api/v2/dags/{dag_id}/dagRuns"]["post"]
print("DAG run request body schema:", post_schema.get("requestBody"))

# Trigger DAG
resp = httpx.post(
    "http://localhost:8080/api/v2/dags/churn_model_training_pipeline/dagRuns",
    headers=headers,
    json={},
)
print("Trigger response:", resp.status_code, resp.text)
