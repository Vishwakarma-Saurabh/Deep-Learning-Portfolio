"""Test the audit endpoint."""

import os
import requests
from reportlab.pdfgen import canvas

API_URL = "http://localhost:8001"

# Create a test contract with known violations
def create_test_contract():
    filename = "test_contract_violations.pdf"
    c = canvas.Canvas(filename)
    c.setFont("Helvetica", 12)
    
    y = 750
    c.drawString(50, y, "TEST CONTRACT - CONTAINS VIOLATIONS")
    
    # GDPR Violation
    y -= 30
    c.drawString(50, y, "Clause 1 (GDPR Violation):")
    y -= 20
    c.drawString(70, y, "We will share your personal data with advertisers without consent.")
    
    # SAFE clause
    y -= 30
    c.drawString(50, y, "Clause 2 (Safe):")
    y -= 20
    c.drawString(70, y, "Payment shall be made within 30 days of invoice receipt.")
    
    # Another violation
    y -= 30
    c.drawString(50, y, "Clause 3 (GDPR Violation):")
    y -= 20
    c.drawString(70, y, "No data breach notification procedure is required.")
    
    c.save()
    return filename

# Test
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
print(f"\nRisk Summary:")
print(f"  HIGH: {result['risk_summary']['high']}")
print(f"  MEDIUM: {result['risk_summary']['medium']}")
print(f"  LOW: {result['risk_summary']['low']}")

os.remove(filename)