"""
Evaluation Harness - Automated testing of LLM outputs.
Runs test cases and scores accuracy, relevance, and safety.
"""

import json
import time
from typing import Dict
from pathlib import Path


class EvalHarness:
    """Automated evaluation system for LLM outputs."""
    
    def __init__(self, test_cases_path: str = None):
        if test_cases_path is None:
            test_cases_path = Path(__file__).parent / "test_cases.json"
        
        with open(test_cases_path) as f:
            self.test_cases = json.load(f)
        
        self.results = []
    
    def evaluate_answer(self, question: str, answer: str, test_case: Dict) -> Dict:
        """Evaluate a single answer against test criteria."""
        checks = {}
        
        # Check 1: Must contain required strings
        if "must_contain" in test_case:
            checks["must_contain"] = all(
                phrase.lower() in answer.lower() 
                for phrase in test_case["must_contain"]
            )
        
        # Check 2: Must NOT contain forbidden strings
        if "must_not_contain" in test_case:
            checks["must_not_contain"] = not any(
                phrase.lower() in answer.lower() 
                for phrase in test_case["must_not_contain"]
            )
        
        # Check 3: Token count within bounds
        if "min_tokens" in test_case:
            word_count = len(answer.split())
            checks["min_tokens"] = word_count >= test_case["min_tokens"]
        
        if "max_tokens" in test_case:
            word_count = len(answer.split())
            checks["max_tokens"] = word_count <= test_case["max_tokens"]
        
        # Check 4: Contains expected answer (fuzzy match)
        if "expected_answer" in test_case:
            checks["expected_answer"] = (
                test_case["expected_answer"].lower() in answer.lower()
            )
        
        # Overall pass/fail
        passed = all(checks.values()) if checks else True
        
        return {
            "question": question,
            "answer": answer[:200],
            "passed": passed,
            "checks": checks,
            "test_case_id": test_case.get("id", "unknown")
        }
    
    def run_evaluation(self, answer_function) -> Dict:

        print(f"Running {len(self.test_cases)} test cases...\n")
        
        passed = 0
        total_latency = 0
        
        for i, test_case in enumerate(self.test_cases):
            print(f"Test {i+1}/{len(self.test_cases)}: {test_case['question'][:60]}...")
            
            # Time the response
            start = time.time()
            answer = answer_function(test_case["question"])
            latency = time.time() - start
            
            # Evaluate
            result = self.evaluate_answer(test_case["question"], answer, test_case)
            result["latency"] = latency
            self.results.append(result)
            
            total_latency += latency
            
            if result["passed"]:
                passed += 1
                print(f"  ✅ PASS ({latency:.2f}s)")
            else:
                print(f"  ❌ FAIL ({latency:.2f}s)")
                failed_checks = [k for k, v in result["checks"].items() if not v]
                print(f"     Failed checks: {failed_checks}")
        
        accuracy = (passed / len(self.test_cases)) * 100
        avg_latency = total_latency / len(self.test_cases)
        
        summary = {
            "total_tests": len(self.test_cases),
            "passed": passed,
            "failed": len(self.test_cases) - passed,
            "accuracy": f"{accuracy:.1f}%",
            "avg_latency": f"{avg_latency:.2f}s",
            "results": self.results
        }
        
        print(f"\n{'='*50}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*50}")
        print(f"Accuracy: {summary['accuracy']}")
        print(f"Average Latency: {summary['avg_latency']}")
        print(f"Passed: {passed}/{len(self.test_cases)}")
        
        return summary


if __name__ == "__main__":
    # Test with mock answer function
    def mock_answer(question):
        if "liability" in question.lower():
            return "The liability cap is $500,000 for any damages."
        elif "termination" in question.lower():
            return "Either party may terminate with 30 days written notice."
        return "I don't have enough information to answer this question."
    
    harness = EvalHarness()
    results = harness.run_evaluation(mock_answer)
    print(f"\nFinal Score: {results['accuracy']}")