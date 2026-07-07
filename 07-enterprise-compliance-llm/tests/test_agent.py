"""
Test the AI agent with various complex requests.
"""
import requests

API_URL = "http://localhost:8001"

test_requests = [
    "Search for data sharing clauses and tell me if there are any GDPR issues",
    "Find all contract clauses and check for compliance violations",
    "Generate a compliance summary report and email it to saurabhg2722006@gmail.com", 
]

print("🤖 Testing AI Agent\n")

for request in test_requests:
    print(f"\n{'='*60}")
    print(f"REQUEST: {request}")
    print('='*60)
    
    response = requests.post(
        f"{API_URL}/agent",
        json={"request": request, "max_steps": 5}
    )
    
    result = response.json()
    
    if result['success']:
        print(f"\n✅ Agent completed in {result['steps_taken']} steps")
        print(f"📝 Final Answer: {result['final_answer']}")
    else:
        print(f"\n❌ Agent failed: {result['final_answer']}")
    
    # Show execution trace
    for step in result['execution_trace']:
        if step['type'] == 'action':
            print(f"  Step {step['step']}: Used {step['action']}")