"""
Compliance checker using fine-tuned LoRA model.
Model loads once and stays cached in memory.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "unsloth/Llama-3.2-1B"
LORA_PATH = "compliance_lora"

# Load model ONCE at module level (not per request)
_model = None
_tokenizer = None

def load_model():
    """Load model once - subsequent calls return cached model."""
    global _model, _tokenizer
    
    if _model is not None:
        return _model, _tokenizer
    
    print("Loading compliance model (one-time)...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        torch_dtype=torch.float32
    )
    
    _model = PeftModel.from_pretrained(base_model, LORA_PATH)
    print("Model loaded and cached!")
    
    return _model, _tokenizer


def check_compliance(clause_text: str) -> dict:
    """Check clause for compliance violations."""
    model, tokenizer = load_model()  # Uses cached model after first call
    
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
    
    severity = "LOW"
    if "HIGH" in prediction.upper():
        severity = "HIGH"
    elif "MEDIUM" in prediction.upper():
        severity = "MEDIUM"
    
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