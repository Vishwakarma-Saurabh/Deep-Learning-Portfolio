"""
End-to-end test for the RAG system.
Run: python test_rag.py
"""

import requests
import os

API_URL = "http://localhost:8001"

def test_health():
    """Test if API is running."""
    response = requests.get(f"{API_URL}/health")
    print(f"Health Check: {response.json()}")
    return response.status_code == 200

def test_ingest():
    """Test document ingestion."""
    # First, create a sample PDF for testing
    sample_text = """CONTRACT AGREEMENT
    
    Section 1: Liability
    The liability cap for any damages arising from this agreement is set at $500,000.
    
    Section 2: Termination
    Either party may terminate this agreement with 30 days written notice.
    
    Section 3: Data Protection
    Both parties shall comply with GDPR requirements for data handling and processing."""
    
    # Save as text file (we'll test with PDF later)
    with open("sample_contract.txt", "w") as f:
        f.write(sample_text)
    
    # Upload to API
    with open("sample_contract.txt", "rb") as f:
        response = requests.post(
            f"{API_URL}/ingest",
            files={"file": ("sample_contract.txt", f, "text/plain")}
        )
    
    print(f"Ingestion: {response.json()}")
    return response.status_code == 200

def test_query():
    """Test querying the system."""
    questions = [
        "What is the liability cap?",
        "How can the contract be terminated?",
        "What are the GDPR requirements?",
        "Who is the CEO of Microsoft?"  # Should not be in documents
    ]
    
    for question in questions:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question}
        )
        
        result = response.json()
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Tokens: {result['tokens_used']}")
        print("-" * 50)

if __name__ == "__main__":
    print("=== Testing RAG System ===\n")
    
    if test_health():
        print("✓ API is running\n")
    else:
        print("✗ API is not running. Start with: python serving/api.py")
        exit(1)
    
    if test_ingest():
        print("✓ Document ingested\n")
    else:
        print("✗ Ingestion failed")
        exit(1)
    
    test_query()
    print("\n=== All tests complete ===")
    
    # Cleanup
    os.remove("sample_contract.txt")