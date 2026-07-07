"""
Report Tool - Agent can generate professional summaries and reports.
Uses Groq LLM to create well-formatted reports.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("LLM_API_KEY"))


def generate_report(findings: dict, report_type: str = "summary") -> dict:

    print(f"📊 Agent generating {report_type} report...")
    
    # Build prompt based on report type
    if report_type == "executive":
        prompt = f"""Create an executive summary for leadership. Be concise and focus on business impact.

Audit Findings:
- Total clauses reviewed: {findings.get('total_clauses', 'N/A')}
- Violations found: {findings.get('violations_found', 'N/A')}
- High risk issues: {findings.get('high_risk_count', 'N/A')}

Details:
{findings.get('violations', [])}

Format as:
1. Overall Risk Assessment (1-2 sentences)
2. Critical Issues (bullet points)
3. Recommended Actions (numbered list)"""
    
    elif report_type == "detailed":
        prompt = f"""Create a detailed compliance report for the legal team.

Findings: {findings}

Include:
1. Executive Summary
2. Methodology
3. Detailed Findings (each violation with clause reference)
4. Risk Matrix
5. Remediation Recommendations
6. Timeline for Fixes"""
    
    else: 
        prompt = f"""Create a brief summary of these audit findings.

Findings: {findings}

Format as:
- Total issues found
- Risk level breakdown
- Top 3 most critical issues
- Next steps"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a legal compliance reporter. Create professional, clear reports."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
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
            "message": f"Report generation failed: {str(e)}"
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