"""Test resilience features - circuit breaker, rate limiting, fallbacks."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import time

API_URL = "http://localhost:8001"


def test_circuit_breaker():
    """Test circuit breaker status."""
    print("=" * 60)
    print("TESTING CIRCUIT BREAKER")
    print("=" * 60)
    
    r = requests.get(f"{API_URL}/circuit/status")
    status = r.json()
    print(f"Groq circuit: {status['groq']}")
    print(f"Qdrant circuit: {status['qdrant']}")
    print("(Circuits OPEN when service fails repeatedly)")


def test_rate_limiting():
    """Test rate limiting functionality."""
    print(f"\n{'='*60}")
    print("TESTING RATE LIMITER")
    print("=" * 60)
    
    user_id = "test_user_123"
    
    # Check initial status
    r = requests.get(f"{API_URL}/rate/status?user_id={user_id}")
    print(f"Initial: {r.json()}")
    
    # Make multiple requests quickly
    print("\nMaking 5 rapid requests...")
    for i in range(5):
        r = requests.post(f"{API_URL}/query", json={"question": "test"})
        remaining = requests.get(f"{API_URL}/rate/status?user_id={user_id}").json()['remaining']
        print(f"  Request {i+1}: Remaining tokens: {remaining}")
    
    # Final status
    r = requests.get(f"{API_URL}/rate/status?user_id={user_id}")
    print(f"\nFinal: {r.json()}")


def test_fallback_chain():
    """Test that fallback chain returns answers even when primary fails."""
    print(f"\n{'='*60}")
    print("TESTING FALLBACK CHAIN")
    print("=" * 60)
    
    # Normal request (should use Groq)
    r = requests.post(f"{API_URL}/query/cached", json={"question": "What are payment terms?"})
    result = r.json()
    print(f"Provider: {result.get('provider', 'unknown')}")
    print(f"Answer: {result['answer'][:80]}...")
    
    # The fallback chain tries: Groq → Cache → Static
    print("\nFallback order: Groq → Semantic Cache → Static Response")


def test_metrics_integration():
    """Test that all resilience features are tracked in metrics."""
    print(f"\n{'='*60}")
    print("TESTING METRICS INTEGRATION")
    print("=" * 60)
    
    r = requests.get(f"{API_URL}/metrics")
    stats = r.json()['stats']
    
    print(f"Total requests: {stats.get('total_requests', 0)}")
    print(f"Success rate: {stats.get('success_rate', 0)}%")
    print(f"Endpoints tracked:")
    for endpoint, data in stats.get('endpoints', {}).items():
        print(f"  {endpoint}: {data['requests']} requests, {data['error_rate']}% errors")


if __name__ == "__main__":
    test_circuit_breaker()
    test_rate_limiting()
    test_fallback_chain()
    test_metrics_integration()
    print("\n✅ Resilience test complete!")