"""
Evaluate the fine-tuned compliance classifier.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "unsloth/Llama-3.2-1B"
LORA_PATH = "fine_tuning/compliance_lora"
DATASET_PATH = "fine_tuning/dataset/compliance_data.json"


def load_model():
    """Load base model with LoRA adapter."""
    print("Loading fine-tuned model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        torch_dtype=torch.float32
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, LORA_PATH)
    
    return model, tokenizer


def classify_clause(model, tokenizer, clause_text: str) -> str:
    """Classify a single clause."""
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
            temperature=0.1,  # Low for consistency
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response


def evaluate():
    """Run evaluation on test set."""
    model, tokenizer = load_model()
    
    # Load test data (last 20% of dataset)
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    
    split_idx = int(len(data) * 0.8)
    test_data = data[split_idx:]
    
    print(f"\nEvaluating on {len(test_data)} test examples...\n")
    
    correct = 0
    total = 0
    
    for item in test_data:
        prediction = classify_clause(model, tokenizer, item["text"])
        
        # Check if prediction matches label
        is_correct = item["label"] in prediction
        
        if is_correct:
            correct += 1
        
        total += 1
        
        print(f"Clause: {item['text'][:80]}...")
        print(f"Expected: {item['label']}")
        print(f"Predicted: {prediction[:80]}...")
        print(f"Correct: {'✓' if is_correct else '✗'}")
        print("-" * 50)
    
    accuracy = correct / total * 100
    print(f"\n{'='*50}")
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print('='*50)


if __name__ == "__main__":
    evaluate()