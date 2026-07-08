"""Test semantic caching - speed and cost savings."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import time

API_URL = "http://localhost:8001"


def test_semantic_cache():
    """Test that cache returns instant responses for similar questions."""
    print("=" * 60)
    print("TESTING SEMANTIC CACHE")
    print("=" * 60)
    
    base_question = "What is the liability cap for damages?"
    similar_questions = [
        base_question,                           # Exact match
        "Tell me about the liability cap amount", # Similar
        "What's the damage liability limit?",     # Similar
        "Explain the cap on liabilities",         # Similar
        "How do I terminate the contract?",       # Different
    ]
    
    for i, question in enumerate(similar_questions):
        print(f"\n📝 Q{i+1}: {question}")
        
        start = time.time()
        r = requests.post(f"{API_URL}/query/cached", json={"question": question})
        latency = time.time() - start
        
        result = r.json()
        cache_hit = result.get('cache_hit', False)
        
        if cache_hit:
            print(f"   🚀 CACHE HIT! ({latency*1000:.0f}ms)")
            print(f"   Similar to: {result.get('similar_to', 'N/A')[:60]}...")
        else:
            print(f"   💾 Cache miss ({latency*1000:.0f}ms)")
            print(f"   Answer: {result['answer'][:80]}...")
    
    # Show cache statistics
    print(f"\n{'='*60}")
    print("CACHE STATISTICS")
    print("=" * 60)
    
    r = requests.get(f"{API_URL}/cache/stats")
    stats = r.json()
    
    print(f"Cache size: {stats.get('cache_size', 0)} entries")
    print(f"Hits: {stats.get('hits', 0)}")
    print(f"Misses: {stats.get('misses', 0)}")
    print(f"Hit rate: {stats.get('hit_rate', '0%')}")
    print(f"Estimated tokens saved: {stats.get('estimated_tokens_saved', 0)}")


if __name__ == "__main__":
    test_semantic_cache()
    print("\n✅ Cache test complete!")