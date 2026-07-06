"""Test the compliance audit system."""
import requests
from reportlab.pdfgen import canvas
import os

API_URL = "http://localhost:8001"

def create_test_contract():
    """Create test contract with known violations."""
    filename = "test_contract_violations.pdf"
    c = canvas.Canvas(filename)
    c.setFont("Helvetica", 12)
    y = 750
    
    c.drawString(50, y, "TEST CONTRACT")
    y -= 20
    c.drawString(50, y, "=" * 40)
    
    # GDPR Violation
    y -= 30
    c.drawString(50, y, "Clause 1: We share your data with advertisers without consent.")
    
    # SAFE
    y -= 20
    c.drawString(50, y, "Clause 2: Payment shall be made within 30 days of invoice.")
    
    # SOX Violation
    y -= 20
    c.drawString(50, y, "Clause 3: Financial records at management discretion, no audits.")
    
    c.save()
    return filename


print("Auditing contract...")
filename = create_test_contract()

with open(filename, "rb") as f:
    response = requests.post(
        f"{API_URL}/audit",
        files={"file": (filename, f, "application/pdf")}
    )

result = response.json()

print(f"\nAudit Report for {result['filename']}")
print(f"Total clauses: {result['total_clauses']}")
print(f"Violations found: {result['violations_found']}")
print(f"Safe clauses: {result['safe_clauses']}")
print(f"\nRisk Breakdown:")
print(f"  HIGH: {result['risk_breakdown']['high']}")
print(f"  MEDIUM: {result['risk_breakdown']['medium']}")

if result['violations']:
    print(f"\nViolations:")
    for i, v in enumerate(result['violations'], 1):
        print(f"  {i}. {v['violation']} ({v['severity']})")

os.remove(filename)
print("\nDone!")