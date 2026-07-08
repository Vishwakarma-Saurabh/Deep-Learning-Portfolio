"""
Audit Logger - Records all requests and responses for compliance.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict


LOG_DIR = Path("audit_logs")
LOG_DIR.mkdir(exist_ok=True)


def log_request(request_data: Dict) -> str:

    log_id = f"req_{int(time.time() * 1000)}"
    
    log_entry = {
        "log_id": log_id,
        "timestamp": datetime.now().isoformat(),
        "type": "request",
        **request_data
    }
    
    # Write to daily log file
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"requests_{date_str}.jsonl"
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    return log_id


def log_response(log_id: str, response_data: Dict):
    """Log API response linked to a request."""
    log_entry = {
        "log_id": log_id,
        "timestamp": datetime.now().isoformat(),
        "type": "response",
        **response_data
    }
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"responses_{date_str}.jsonl"
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def get_audit_trail(date: str = None) -> list:
    """Retrieve audit trail for a specific date."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    log_file = LOG_DIR / f"requests_{date}.jsonl"
    
    if not log_file.exists():
        return []
    
    logs = []
    with open(log_file) as f:
        for line in f:
            logs.append(json.loads(line.strip()))
    
    return logs


if __name__ == "__main__":
    # Test logging
    log_id = log_request({
        "endpoint": "/query",
        "user_input": "What is the liability cap?",
        "client_ip": "127.0.0.1"
    })
    print(f"Logged request: {log_id}")
    
    log_response(log_id, {
        "answer": "The liability cap is $500,000",
        "tokens_used": 150,
        "latency_ms": 1200
    })
    print("Logged response")