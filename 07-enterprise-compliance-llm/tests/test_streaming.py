"""Test streaming responses - real-time token generation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
import time

API_URL = "http://localhost:8001"


def test_streaming():
    """Test that streaming returns tokens one by one."""
    print("=" * 60)
    print("TESTING STREAMING RESPONSE")
    print("=" * 60)
    
    question = "What are the payment terms?"
    
    print(f"\n📝 Question: {question}")
    print("📤 Streaming tokens:")
    
    # Stream the response
    start = time.time()
    response = requests.post(
        f"{API_URL}/query/stream",
        json={"question": question},
        stream=True
    )
    
    token_count = 0
    full_text = ""
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                    
                    if 'token' in data:
                        token_count += 1
                        full_text = data['full']
                        # Print token without newline
                        print(data['token'], end='', flush=True)
                    
                    if 'done' in data:
                        print()  # New line after completion
                        print(f"\n✅ Stream complete!")
                        print(f"   Tokens: {token_count}")
                        print(f"   Time: {time.time() - start:.2f}s")
                    
                    if 'error' in data:
                        print(f"\n❌ Error: {data['error']}")
                
                except json.JSONDecodeError:
                    pass
    
    # Compare with non-streaming
    print(f"\n{'='*60}")
    print("COMPARISON: Streaming vs Regular")
    print("=" * 60)
    
    start2 = time.time()
    r2 = requests.post(f"{API_URL}/query", json={"question": question})
    latency2 = time.time() - start2
    
    print(f"Streaming tokens: {token_count}")
    print(f"Regular tokens: {r2.json().get('tokens_used', {}).get('total', 'N/A')}")
    print(f"Regular latency: {latency2:.2f}s")


if __name__ == "__main__":
    test_streaming()
    print("\n✅ Streaming test complete!")