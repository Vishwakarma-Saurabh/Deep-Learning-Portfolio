"""
Generate synthetic compliance training dataset using Groq LLM.
Creates labeled contract clauses for fine-tuning.
"""

import json
import os
import time
from collections import Counter
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("LLM_API_KEY"))

# Categories we want to detect
CATEGORIES = [
    "GDPR_Violation",      # Data protection issues
    "SOX_Violation",       # Financial compliance issues  
    "SAFE",                # Compliant clause
    "NEEDS_REVIEW"         # Ambiguous, requires human review
]

# Templates for generating diverse clauses
CLAUSE_TYPES = [
    "Data Processing & Privacy",
    "Payment Terms & Financial",
    "Termination & Renewal",
    "Liability & Indemnification",
    "Confidentiality",
    "Third-Party Sharing",
    "Security Requirements",
    "Audit Rights"
]


def generate_batch(category: str, clause_type: str, count: int = 5) -> list:
   
    prompt = f"""You are a legal expert creating training data for a compliance AI.

Generate {count} realistic contract clauses about "{clause_type}" that would be classified as "{category}".

Rules:
- GDPR_Violation: Missing consent, no data protection, illegal data sharing, no breach notification
- SOX_Violation: Financial misrepresentation, missing audit trails, improper record keeping
- SAFE: Properly written, includes required protections, follows best practices
- NEEDS_REVIEW: Has minor issues, ambiguous language, partially compliant

Return ONLY a JSON array. Each object should have:
- "text": The clause text (50-150 words)
- "label": "{category}"
- "explanation": Why this classification (20-50 words)
- "severity": "HIGH"/"MEDIUM"/"LOW" (LOW for SAFE)

Example:
[
  {{
    "text": "Company shall transfer all user data to third-party partners without prior notification to users.",
    "label": "GDPR_Violation",
    "explanation": "No user consent or notification for data transfers, violating GDPR Article 13 transparency requirements.",
    "severity": "HIGH"
  }}
]

Generate {count} examples for category "{category}" about "{clause_type}":
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a legal expert. Return only valid JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,  # Higher for diversity
            max_tokens=2000
        )
        
        # Parse the JSON response
        text = response.choices[0].message.content
        
        # Find JSON array in response
        start = text.find('[')
        end = text.rfind(']') + 1
        
        if start >= 0 and end > start:
            json_str = text[start:end]
            examples = json.loads(json_str)
            return examples
        else:
            print(f"Failed to parse JSON from: {text[:200]}...")
            return []
            
    except Exception as e:
        print(f"Error generating batch: {e}")
        return []


def main():
    """Generate complete dataset."""
    all_examples = []
    examples_per_category = 50  # 50 examples × 4 categories = 200 total
    
    print("Generating compliance training dataset...")
    print(f"Target: {examples_per_category} examples per category")
    print(f"Total: {examples_per_category * len(CATEGORIES)} examples\n")
    
    for category in CATEGORIES:
        print(f"\n{'='*50}")
        print(f"Generating {category} examples...")
        print('='*50)
        
        category_examples = []
        batches_needed = examples_per_category // 5  # 5 per batch
        
        for i in range(batches_needed):
            # Rotate through clause types for diversity
            clause_type = CLAUSE_TYPES[i % len(CLAUSE_TYPES)]
            
            print(f"  Batch {i+1}/{batches_needed}: {clause_type}")
            batch = generate_batch(category, clause_type, count=5)
            
            if batch:
                category_examples.extend(batch)
                print(f"    ✓ Generated {len(batch)} examples")
            
            # Rate limiting - Groq free tier allows 30 req/min
            time.sleep(2)
        
        all_examples.extend(category_examples[:examples_per_category])
        print(f"  Total {category}: {len(category_examples[:examples_per_category])} examples")
    
    # Save dataset
    os.makedirs("fine_tuning/dataset", exist_ok=True)
    output_path = "fine_tuning/dataset/compliance_data.json"
    
    with open(output_path, 'w') as f:
        json.dump(all_examples, f, indent=2)
    
    # Print statistics
    print(f"\n{'='*50}")
    print("DATASET GENERATION COMPLETE")
    print('='*50)
    print(f"Total examples: {len(all_examples)}")
    
    # Count per category
    labels = Counter(ex["label"] for ex in all_examples)
    for label, count in labels.items():
        print(f"  {label}: {count}")
    
    print(f"\nSaved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()