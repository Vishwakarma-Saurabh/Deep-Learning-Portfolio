"""
Fine-tune Llama-3.2-1B on compliance dataset using QLoRA.
Python 3.14 compatible version.
"""

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configuration
MODEL_NAME = "unsloth/Llama-3.2-1B"
OUTPUT_DIR = "fine_tuning/compliance_lora"
DATASET_PATH = "fine_tuning/dataset/compliance_data.json"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1


def load_and_format_dataset():
    """Load dataset and format for training - avoids Datasets library issues."""
    print("Loading dataset...")
    
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    
    # Format as instruction prompts
    formatted_data = []
    for item in data:
        if item["label"] == "SAFE":
            response = f"SAFE - {item['explanation']}"
        else:
            response = f"{item['label']} (Severity: {item['severity']}) - {item['explanation']}"
        
        prompt = f"""### Instruction:
Classify this contract clause for compliance violations:

### Input:
{item['text']}

### Response:
{response}"""
        
        formatted_data.append({"text": prompt})
    
    # Split manually (no Datasets library)
    split_idx = int(len(formatted_data) * 0.8)
    
    train_data = formatted_data[:split_idx]
    test_data = formatted_data[split_idx:]
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    return train_data, test_data


def train():
    """Main training function."""
    print("=" * 50)
    print("COMPLIANCE CLASSIFIER - LORA FINE-TUNING")
    print("=" * 50)
    
    # Load data (plain lists, no Datasets library)
    train_data, test_data = load_and_format_dataset()
    
    print("\nLoading tokenizer and model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="cpu",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenize data manually
    print("\nTokenizing data...")
    
    def tokenize_texts(texts):
        """Tokenize a list of texts."""
        return tokenizer(
            [t["text"] for t in texts],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
    
    # Create simple dataset class
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        
        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.encodings["input_ids"][idx].clone()
            }
        
        def __len__(self):
            return len(self.encodings["input_ids"])
    
    train_encodings = tokenize_texts(train_data)
    test_encodings = tokenize_texts(test_data)
    
    train_dataset = SimpleDataset(train_encodings)
    test_dataset = SimpleDataset(test_encodings)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator
    )
    
    # Train
    print("\nStarting training on CPU...")
    print("(15-25 minutes for 200 examples)")
    trainer.train()
    
    # Save
    print(f"\nSaving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    train()