"""
Agent Orchestrator - The brain that plans and executes multi-step workflows.
Uses ReAct pattern: Reasoning + Acting.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv


from agents.tools.search_tool import search_documents
from agents.tools.audit_tool import audit_document_clauses, audit_by_search
from agents.tools.report_tool import generate_report
from agents.tools.email_tool import send_email

load_dotenv()
client = Groq(api_key=os.getenv("LLM_API_KEY"))

SYSTEM_PROMPT = """You are a legal compliance AI agent with access to tools.

AVAILABLE TOOLS:
1. search_documents("query") - Search through legal documents
2. audit_document_clauses(["clause1", "clause2"]) - Check clauses for violations  
3. audit_by_search("topic") - Find and audit clauses about a topic
4. generate_report({"findings": {...}, "report_type": "summary"}) - Create reports
5. send_email({"to": "...", "subject": "...", "body": "..."}) - Send emails

CRITICAL RULES:
- When a search returns no results, tell the user. Do NOT try to email about it.
- Use EXACTLY this format for tool calls:
  Thought: [your reasoning]
  Action: tool_name
  Action Input: [simple string for search/audit_by_search, valid JSON for email/report]
- For send_email, Action Input MUST be valid JSON: {"to": "...", "subject": "...", "body": "..."}
- If you have enough information or hit an error, give FINAL ANSWER immediately.

EXAMPLE - When search fails:
Thought: Search returned no results. I should tell the user.
FINAL ANSWER: No documents were found. Please ingest documents first using the /ingest endpoint.

EXAMPLE - When everything works:
Thought: I found violations. Let me generate a report.
Action: generate_report
Action Input: {"findings": {"violations_found": 2}, "report_type": "summary"}
"""


def execute_agent(user_request: str, max_steps: int = 10) -> dict:
    """Execute multi-step agent workflow."""
    print(f"\n{'='*60}")
    print(f"🤖 AGENT: {user_request}")
    print(f"{'='*60}\n")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request}
    ]
    
    execution_trace = []
    
    for step in range(max_steps):
        print(f"--- Step {step + 1} ---")
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )
        
        llm_output = response.choices[0].message.content
        print(f"LLM: {llm_output[:300]}...")
        
        # Check for final answer
        if "FINAL ANSWER:" in llm_output:
            final_answer = llm_output.split("FINAL ANSWER:")[-1].strip()
            execution_trace.append({"step": step + 1, "type": "final", "output": final_answer})
            return {
                "success": True,
                "final_answer": final_answer,
                "steps_taken": step + 1,
                "execution_trace": execution_trace
            }
        
        if step >= 3:
            last_actions = [t['action'] for t in execution_trace[-3:] if t['type'] == 'action']
            if len(set(last_actions)) == 1 and len(last_actions) == 3:
                return {
                    "success": False,
                    "final_answer": f"The {last_actions[0]} tool failed repeatedly. Please check the tool configuration.",
                    "steps_taken": step + 1,
                    "execution_trace": execution_trace
                }

        # Parse action
        action_match = re.search(r'Action:\s*(\w+)', llm_output)
        
        if not action_match:
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": "Use format: Thought/Action/Action Input or FINAL ANSWER"})
            continue
        
        action = action_match.group(1).strip()
        
        # Extract action input
        action_input = _extract_action_input(llm_output, action)
        
        print(f"Action: {action}")
        
        # Execute tool
        try:
            observation = _execute_tool(action, action_input)
        except Exception as e:
            observation = {"success": False, "message": str(e)}
        
        obs_str = json.dumps(observation) if isinstance(observation, dict) else str(observation)
        print(f"Result: {obs_str[:200]}...\n")
        
        execution_trace.append({
            "step": step + 1,
            "type": "action",
            "action": action,
            "observation": obs_str[:500]
        })
        
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({"role": "user", "content": f"Observation: {obs_str}"})
    
    return {
        "success": False,
        "final_answer": "Could not complete within step limit.",
        "steps_taken": max_steps,
        "execution_trace": execution_trace
    }


def _extract_action_input(llm_output: str, action: str) -> str:
    """Extract action input, handling both JSON and plain text."""
    
    if action in ["send_email", "generate_report"]:
        # Find the first complete JSON object
        json_match = re.search(r'\{[^{}]*\}', llm_output)
        if json_match:
            return json_match.group(0).strip()
    
    if action in ["search_documents", "audit_by_search"]:
        text_match = re.search(r'Action Input:\s*["\'](.+?)["\']', llm_output)
        if text_match:
            return text_match.group(1).strip()
    
    if action == "audit_document_clauses":
        array_match = re.search(r'\[(.*?)\]', llm_output, re.DOTALL)
        if array_match:
            return "[" + array_match.group(1) + "]"
    
    parts = llm_output.split("Action Input:")
    if len(parts) > 1:
        return parts[-1].strip()[:200]  # Limit length
    
    return ""


def _execute_tool(action: str, action_input: str):
    """Execute tool with proper parameter parsing."""
    
    # Try to parse as JSON
    try:
        params = json.loads(action_input)
    except:
        params = action_input
    
    if action == "search_documents":
        query = params if isinstance(params, str) else params.get("query", str(params))
        return search_documents(query)
    
    elif action == "audit_document_clauses":
        if isinstance(params, list):
            return audit_document_clauses(params)
        elif isinstance(params, dict) and "clauses" in params:
            return audit_document_clauses(params["clauses"])
        return audit_document_clauses([str(params)])
    
    elif action == "audit_by_search":
        query = params if isinstance(params, str) else params.get("search_query", str(params))
        return audit_by_search(query)
    
    elif action == "generate_report":
        if isinstance(params, dict):
            if "findings" in params:
                findings = params["findings"]
                report_type = params.get("report_type", "summary")
                return generate_report(findings, report_type)
            else:
                return generate_report(params, "summary")
        elif isinstance(params, str):
            try:
                parsed = json.loads(params)
                findings = parsed.get("findings", parsed)
                report_type = parsed.get("report_type", "summary")
                return generate_report(findings, report_type)
            except:
                return generate_report({"raw_input": params}, "summary")
        elif isinstance(params, list):
            return generate_report({"items": params}, "summary")
        else:
            return {"success": False, "message": f"Cannot parse generate_report input: {type(params)}"}
      
    elif action == "send_email":
        if isinstance(params, dict):
            return send_email(
                params.get("to", ""),
                params.get("subject", ""),
                params.get("body", ""),
                params.get("attachment")
            )
        return {"success": False, "message": "Invalid JSON for send_email. Use: {\"to\":\"...\", \"subject\":\"...\", \"body\":\"...\"}"}
    
    return {"success": False, "message": f"Unknown tool: {action}"}


if __name__ == "__main__":
    result = execute_agent(
        "Search for data sharing clauses, audit them, and email the report to saurabhg2722006@gmail.com"
    )
    print(f"\nFinal: {result['final_answer']}")