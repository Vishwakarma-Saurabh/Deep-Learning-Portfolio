"""Test hybrid search (dense + sparse combination)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

API_URL = "http://localhost:8001"


def test_hybrid_search():
    """Test that hybrid search returns results from both methods."""
    print("=" * 60)
    print("TESTING HYBRID SEARCH")
    print("=" * 60)
    
    # Test queries that benefit from hybrid search
    test_queries = [
        "What is the liability cap amount in Section 1?",
        "Tell me about data sharing with third parties",
        "What are the payment terms and deadlines?",
        "How to terminate according to the contract?",
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        
        # Test hybrid endpoint
        r = requests.post(f"{API_URL}/query/hybrid", json={"question": query})
        
        if r.status_code == 200:
            result = r.json()
            print(f"   Provider: {result.get('provider', 'N/A')}")
            print(f"   Answer: {result['answer'][:100]}...")
            
            # Show source types
            if 'results' in result:
                types = [res.get('type', '?') for res in result['results']]
                print(f"   Sources: {types}")
        else:
            print(f"   ❌ Error: {r.status_code}")
    
    # Compare with regular query
    print(f"\n{'='*60}")
    print("COMPARISON: Regular vs Hybrid")
    print("=" * 60)
    
    query = "Section 4.2 liability cap exact amount"
    
    # Regular query
    r1 = requests.post(f"{API_URL}/query", json={"question": query})
    t1 = r1.json().get('tokens_used', {}).get('total', 0)
    
    # Hybrid query
    r2 = requests.post(f"{API_URL}/query/hybrid", json={"question": query})
    t2 = r2.json().get('tokens_used', 0)
    
    print(f"Regular: {r1.json()['answer'][:80]}...")
    print(f"Hybrid:  {r2.json()['answer'][:80]}...")
    print(f"Regular tokens: {t1}, Hybrid tokens: {t2}")


if __name__ == "__main__":
    test_hybrid_search()
    print("\n✅ Hybrid search test complete!")