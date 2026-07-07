"""Ingest test documents into the system."""
import requests
import os

API_URL = "http://localhost:8001"

# Ingest the sample contract
test_file = "data/sample_contract.txt"

if not os.path.exists(test_file):
    print(f"❌ {test_file} not found! Create it first.")
    exit(1)

print(f"📤 Ingesting {test_file}...")

with open(test_file, "rb") as f:
    response = requests.post(
        f"{API_URL}/ingest",
        files={"file": (os.path.basename(test_file), f, "text/plain")}
    )

result = response.json()
print(f"✅ Response: {result}")

# Verify by checking health
health = requests.get(f"{API_URL}/health")
print(f"✅ API Status: {health.json()}")