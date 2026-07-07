"""
Audit Tool - Agent can check documents for compliance violations.
Wraps the fine-tuned compliance model from Milestone 2.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compliance import check_compliance
from agents.tools.search_tool import search_documents


def audit_document_clauses(clauses: list) -> dict:

    print(f"📋 Agent auditing {len(clauses)} clauses...")
    
    violations = []
    safe_count = 0
    
    for i, clause in enumerate(clauses):
        result = check_compliance(clause)
        
        if result["violation"] != "SAFE":
            violations.append({
                "clause_number": i + 1,
                "clause_preview": result["clause_preview"],
                "violation": result["violation"],
                "severity": result["severity"],
                "explanation": result["explanation"]
            })
        else:
            safe_count += 1
    
    high_risk = len([v for v in violations if v["severity"] == "HIGH"])
    
    return {
        "success": True,
        "total_clauses": len(clauses),
        "violations_found": len(violations),
        "safe_clauses": safe_count,
        "high_risk_count": high_risk,
        "violations": violations,
        "summary": f"Found {len(violations)} violations ({high_risk} HIGH risk) out of {len(clauses)} clauses."
    }


# Alternative: Audit by searching for relevant clauses first
def audit_by_search(search_query: str) -> dict:
    
    # First search
    search_result = search_documents(search_query)
    
    if not search_result["success"]:
        return {
            "success": False,
            "message": "Could not find relevant clauses to audit"
        }
    
    # Extract clause texts from search results
    clauses = [r["content_preview"] for r in search_result["results"]]
    
    # Then audit
    return audit_document_clauses(clauses)


# Tool description for the LLM
TOOL_DESCRIPTION = """
Tool: audit_document_clauses
Description: Check contract clauses for GDPR and SOX compliance violations.
Input: clauses (list of strings) - The clause texts to audit
Output: Violation report with severity levels and explanations
Use when: User asks to check compliance, find violations, or audit contracts

Tool: audit_by_search
Description: Search for specific types of clauses, then audit them.
Input: search_query (string) - What type of clauses to find and audit
Output: Audit results for matching clauses
Use when: User asks to audit specific topics like "data sharing" or "payment terms"
"""


# Test independently
if __name__ == "__main__":
    test_clauses = [
        "We share user data with advertisers without consent",
        "Payment due within 30 days of invoice",
        "Financial records at management discretion"
    ]
    
    result = audit_document_clauses(test_clauses)
    print(f"\nAudit Summary: {result['summary']}")
    for v in result['violations']:
        print(f"  - {v['violation']} ({v['severity']})")