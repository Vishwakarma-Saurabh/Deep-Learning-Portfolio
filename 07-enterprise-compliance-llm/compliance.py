"""
Compliance checker using fine-tuned LoRA model from Google Colab.
Detects GDPR and SOX violations in contract clauses.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "unsloth/Llama-3.2-1B"
LORA_PATH = "compliance_lora"

_model = None
_tokenizer = None


def load_model():
    """Load model once and reuse."""
    global _model, _tokenizer
    
    if _model is None:
        print("Loading compliance model...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        
        _model = PeftModel.from_pretrained(base_model, LORA_PATH)
        print("Model loaded!")
    
    return _model, _tokenizer


def check_compliance(clause_text: str) -> dict:
    """Check clause for compliance violations."""
    model, tokenizer = load_model()
    
    prompt = (
        f"### Instruction:\n"
        f"Classify this contract clause for compliance violations.\n\n"
        f"### Input:\n{clause_text}\n\n"
        f"### Response:\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=80, temperature=0.1)
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = prediction.split("### Response:")[-1].strip()
    
    # Parse severity
    severity = "LOW"
    if "HIGH" in prediction.upper():
        severity = "HIGH"
    elif "MEDIUM" in prediction.upper():
        severity = "MEDIUM"
    
    # Parse violation type
    violation = "NEEDS_REVIEW"
    if "GDPR" in prediction.upper():
        violation = "GDPR_Violation"
    elif "SOX" in prediction.upper():
        violation = "SOX_Violation"
    elif "SAFE" in prediction.upper():
        violation = "SAFE"
    
    return {
        "violation": violation,
        "severity": severity,
        "explanation": prediction,
        "clause_preview": clause_text[:100] + "..."
    }


if __name__ == "__main__":
    print("Testing compliance model:\n")
    
    tests = [
        "We share user data with advertisers without consent",
        "Payment due within 30 days of invoice",
        "Financial records at management discretion",
    ]
    
    for t in tests:
        r = check_compliance(t)
        print(f"Clause: {t}")
        print(f"Result: {r['violation']} ({r['severity']})")
        print(f"Detail: {r['explanation'][:80]}...")
        print("-" * 50)