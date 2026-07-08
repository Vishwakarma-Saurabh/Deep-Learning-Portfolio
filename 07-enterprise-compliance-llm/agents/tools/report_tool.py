"""
Report Tool - Agent can generate professional summaries and reports.
Uses Groq LLM to create well-formatted reports.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("LLM_API_KEY"))


def generate_report(findings: dict, report_type: str = "summary") -> dict:
    """
    Generate a professional report from audit findings.
    Handles any input format gracefully.
    """
    print(f"📊 Agent generating {report_type} report...")
    
    # Extract useful info regardless of format
    violations_found = findings.get("violations_found", findings.get("total_clauses", "N/A"))
    high_risk = findings.get("high_risk_count", "N/A")
    
    # Build simple summary for prompt
    findings_str = json.dumps(findings, indent=2) if isinstance(findings, dict) else str(findings)
    
    if report_type == "executive":
        prompt = f"""Create a 3-sentence executive summary of these compliance findings:

{findings_str}

Format:
1. Overall status
2. Critical issues
3. Recommended action"""
    
    elif report_type == "detailed":
        prompt = f"""Create a detailed compliance report:

{findings_str}

Include sections:
- Executive Summary
- Findings Detail
- Risk Assessment
- Recommendations"""
    
    else:  # summary
        prompt = f"""Summarize these audit findings in 2-3 bullet points:

{findings_str}"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a compliance reporter. Be concise and factual."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        report = response.choices[0].message.content
        
        return {
            "success": True,
            "report_type": report_type,
            "report": report,
            "tokens_used": response.usage.total_tokens
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Report generation failed: {str(e)}",
            "report": f"Error generating report: {str(e)}"
        }

# Tool description for the LLM
TOOL_DESCRIPTION = """
Tool: generate_report
Description: Create professional compliance reports from audit findings.
Input: 
  - findings (dict): Audit results from audit_document_clauses
  - report_type (string): "summary", "detailed", or "executive"
Output: Formatted report text
Use when: User asks for a report, summary, or documentation of findings
"""


# Test independently
if __name__ == "__main__":
    test_findings = {
        "total_clauses": 10,
        "violations_found": 3,
        "high_risk_count": 2,
        "violations": [
            {"violation": "GDPR_Violation", "severity": "HIGH", "explanation": "No consent for data sharing"},
            {"violation": "SOX_Violation", "severity": "HIGH", "explanation": "Missing audit requirements"}
        ]
    }
    
    result = generate_report(test_findings, "executive")
    print(result['report'])