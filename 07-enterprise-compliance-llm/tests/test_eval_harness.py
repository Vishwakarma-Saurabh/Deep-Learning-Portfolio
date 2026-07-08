"""
Test the evaluation harness to ensure it scores answers correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llmops.eval_harness import EvalHarness


def test_eval_harness():
    """Test that the evaluation harness works correctly."""
    
    harness = EvalHarness()
    
    print("Testing Evaluation Harness\n")
    
    # Test 1: Perfect answer should pass
    print("Test 1: Perfect answer")
    result = harness.evaluate_answer(
        "What is the liability cap?",
        "The liability cap is set at $500,000 for any damages arising from this agreement.",
        harness.test_cases[0]  # tc_001 expects liability + $500,000
    )
    assert result["passed"] == True, "Perfect answer should pass!"
    print(f"  ✅ Passed: {result['passed']}")
    
    # Test 2: Missing required content should fail
    print("\nTest 2: Incomplete answer")
    result = harness.evaluate_answer(
        "What is the liability cap?",
        "I don't have enough information to answer this question.",
        harness.test_cases[0]
    )
    # This should fail because it contains "I don't know" (must_not_contain)
    assert result["passed"] == False, "Answer with forbidden text should fail!"
    print(f"  ✅ Correctly failed: {result['passed']}")
    
    # Test 3: Short answer should fail min_tokens check
    print("\nTest 3: Too short answer")
    result = harness.evaluate_answer(
        "What are the GDPR requirements?",
        "GDPR required.",
        harness.test_cases[3]  # tc_004 expects min 15 tokens
    )
    print(f"  Passed: {result['passed']}")
    print(f"  Checks: {result['checks']}")
    
    # Test 4: Run full evaluation with mock function
    print("\nTest 4: Full evaluation run")
    
    def mock_answer_function(question):
        """Simulate an LLM that answers correctly with sufficient detail."""
        answers = {
            "What is the liability cap?": "The liability cap is set at $500,000 for any damages arising from this agreement including breach of contract and negligence claims.",
            "How can this contract be terminated?": "Either party may terminate this agreement with 30 days written notice. Termination does not affect any accrued rights or obligations.",
            "Is data shared with third parties?": "Yes, the contract states that data may be shared with third-party advertising partners without obtaining prior user consent.",
            "What are the GDPR requirements?": "GDPR requires data encryption at rest and in transit, breach notification within 72 hours, and explicit user consent for data processing activities.",
            "Are there any compliance violations?": "Yes, there are multiple GDPR violations including data sharing without consent and lack of breach notification procedures.",
            "What payment terms are specified?": "Payment shall be made within 30 days of invoice receipt with late payments incurring a 5 percent penalty per month.",
            "Is there a data breach notification requirement?": "The contract explicitly states that no data breach notification to users or authorities is required, which violates GDPR Article 33 requirements.",
            "How is user data encrypted?": "User data shall be encrypted using AES-256 encryption protocol both at rest in database storage and in transit over public networks.",
            "What are the audit requirements?": "Annual external audits shall be conducted by an independent certified public accounting firm with complete access to all financial records and transactions.",
            "Are financial controls mandatory?": "The contract states that implementation of internal controls over financial reporting is currently optional and determined by individual department heads."
        }
        return answers.get(question, "I don't have enough information to answer this question accurately.")
    
    results = harness.run_evaluation(mock_answer_function)
    
    print(f"\nFinal Results:")
    print(f"  Accuracy: {results['accuracy']}")
    print(f"  Passed: {results['passed']}/{results['total_tests']}")
    
    # Show failed tests
    failed = [r for r in results['results'] if not r['passed']]
    if failed:
        print(f"\n  Failed Tests:")
        for f in failed:
            print(f"    - {f['question'][:50]}...")
            print(f"      Failed checks: {[k for k, v in f['checks'].items() if not v]}")
    
    print("\n✅ All harness tests complete!")


if __name__ == "__main__":
    test_eval_harness()