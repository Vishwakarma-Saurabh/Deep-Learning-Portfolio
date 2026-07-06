"""
End-to-end test for the RAG system.
Tests with a proper PDF file.
"""

import requests
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

API_URL = "http://localhost:8001"

def create_test_pdf():
    """Create a sample PDF for testing."""
    filename = "sample_contract.pdf"
    
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica", 12)
    
    # Page 1 content
    c.drawString(50, 750, "CONTRACT AGREEMENT")
    c.drawString(50, 730, "=" * 50)
    
    c.drawString(50, 700, "Section 1: Liability")
    y = 680
    text = "The liability cap for any damages arising from this agreement "
    text += "is set at $500,000 (Five Hundred Thousand Dollars). "
    text += "This cap applies to all claims including but not limited to "
    text += "breach of contract, negligence, and statutory violations."
    c.drawString(50, y, text)
    
    c.drawString(50, 640, "Section 2: Termination")
    y = 620
    text = "Either party may terminate this agreement with 30 days written notice. "
    text += "Termination shall not affect any accrued rights or obligations. "
    text += "Upon termination, all confidential information must be returned within 15 days."
    c.drawString(50, y, text)
    
    c.drawString(50, 580, "Section 3: Data Protection")
    y = 560
    text = "Both parties shall comply with GDPR requirements for data handling. "
    text += "Personal data must be encrypted at rest and in transit. "
    text += "Data breaches must be reported within 72 hours of discovery. "
    text += "A Data Protection Officer (DPO) must be appointed by each party."
    c.drawString(50, y, text)
    
    c.drawString(50, 520, "Section 4: Payment Terms")
    y = 500
    text = "Payment shall be made within 30 days of invoice receipt. "
    text += "Late payments incur a 5% penalty per month. "
    text += "All amounts are in USD unless otherwise specified."
    c.drawString(50, y, text)
    
    c.save()
    print(f"✓ Created test PDF: {filename}")
    return filename


def test_health():
    """Test if API is running."""
    response = requests.get(f"{API_URL}/health")
    print(f"Health Check: {response.json()}")
    return response.status_code == 200


def test_ingest():
    """Test document ingestion with PDF."""
    filename = create_test_pdf()
    
    # Upload PDF
    with open(filename, "rb") as f:
        response = requests.post(
            f"{API_URL}/ingest",
            files={"file": (filename, f, "application/pdf")}
        )
    
    result = response.json()
    print(f"\nIngestion Response: {result}")
    
    if response.status_code == 200:
        print(f"✓ Ingested {result.get('chunks_created', 0)} chunks from {filename}")
    else:
        print(f"✗ Ingestion failed: {result}")
    
    # Cleanup PDF
    os.remove(filename)
    
    return response.status_code == 200


def test_query():
    """Test querying the system."""
    questions = [
        "What is the liability cap amount?",
        "How many days notice is needed for termination?",
        "What are the GDPR data protection requirements?",
        "What is the penalty for late payment?"
    ]
    
    print("\n" + "=" * 60)
    print("QUERY TESTING")
    print("=" * 60)
    
    for i, question in enumerate(questions, 1):
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question, "top_k": 3}
        )
        
        result = response.json()
        print(f"\n📝 Question {i}: {question}")
        print(f"💡 Answer: {result['answer']}")
        print(f"📌 Sources: {result['sources']}")
        if result.get('tokens_used'):
            print(f"🔢 Tokens: {result['tokens_used']}")
        print("-" * 50)


if __name__ == "__main__":
    print("\n=== RAG System Test ===\n")
    
    # Check health
    if not test_health():
        print("✗ API is not running. Start with: python serving/api.py")
        exit(1)
    print("✓ API is running\n")
    
    # Ingest document
    if not test_ingest():
        print("✗ Ingestion failed - check API logs")
        exit(1)
    
    # Query
    test_query()
    
    print("\n✓ All tests completed successfully!")