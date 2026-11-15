import httpx

r = httpx.post("http://localhost:8080/auth/token", json={"username": "admin", "password": "3zHWhuXKTm7ADCqr"})
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

dags_resp = httpx.get("http://localhost:8080/api/v2/dags", headers=headers)
print("DAGs API response:", dags_resp.status_code, dags_resp.json())
