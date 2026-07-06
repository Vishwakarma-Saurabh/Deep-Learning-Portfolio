"""
Compliance checking tool using fine-tuned model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Lazy load model (only load when needed)
_model = None
_tokenizer = None

MODEL_NAME = "unsloth/Llama-3.2-1B"
LORA_PATH = "fine_tuning/compliance_lora"


def load_model():
    """Load model once and cache it."""
    global _model, _tokenizer
    
    if _model is None:
        print("Loading compliance model...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        _model = PeftModel.from_pretrained(_model, LORA_PATH)
        print("Model loaded!")
    
    return _model, _tokenizer


def check_compliance(clause_text: str) -> dict:

    model, tokenizer = load_model()
    
    prompt = f"""### Instruction:
Classify this contract clause for compliance violations:

### Input:
{clause_text}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    # Parse the response
    result = {
        "clause": clause_text[:100] + "...",
        "raw_output": response,
        "violation": "Unknown",
        "severity": "LOW",
        "explanation": response
    }
    
    # Try to extract structured info
    if "GDPR" in response:
        result["violation"] = "GDPR_Violation"
    elif "SOX" in response:
        result["violation"] = "SOX_Violation"
    elif "SAFE" in response:
        result["violation"] = "SAFE"
    elif "NEEDS_REVIEW" in response:
        result["violation"] = "NEEDS_REVIEW"
    
    if "HIGH" in response:
        result["severity"] = "HIGH"
    elif "MEDIUM" in response:
        result["severity"] = "MEDIUM"
    
    return result
