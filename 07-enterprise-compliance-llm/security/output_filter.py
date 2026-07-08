"""
Output Security Filter - Redacts sensitive information from responses.
"""

import re
from typing import Dict


# Patterns for PII detection
PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
    "credit_card": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
}


def filter_output(response_text: str) -> Dict:

    redactions = []
    filtered_text = response_text
    
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, filtered_text)
        if matches:
            for match in matches:
                filtered_text = filtered_text.replace(match, f"[REDACTED_{pii_type.upper()}]")
                redactions.append({"type": pii_type, "original": match})
    
    return {
        "text": filtered_text,
        "redacted": len(redactions) > 0,
        "redaction_count": len(redactions),
        "redactions": redactions
    }


def contains_harmful_content(response_text: str) -> bool:
    """Check if response contains potentially harmful content."""
    harmful_patterns = [
        r'\b(password|secret|token|api.key)\s*[:=]\s*\S+',
        r'exec\s*\(|eval\s*\(|system\s*\(',
    ]
    
    for pattern in harmful_patterns:
        if re.search(pattern, response_text, re.IGNORECASE):
            return True
    
    return False


if __name__ == "__main__":
    # Test PII redaction
    response = "Contact john.doe@company.com or call 555-123-4567 for details."
    result = filter_output(response)
    print(f"Filtered: {result['text']}")
    print(f"Redactions: {result['redaction_count']}")
    
    # Test harmful content
    malicious = "Here is the password: admin123"
    print(f"Contains harmful: {contains_harmful_content(malicious)}")