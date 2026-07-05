"""
Synthesizer - Generates answers using retrieved context.
Concept: RAG synthesis - combining retrieval with generation.
"""

import os
from typing import Dict
from groq import Groq
from dotenv import load_dotenv
from rag.prompts import RAG_SYSTEM_PROMPT, get_rag_prompt

load_dotenv()

# Initialize LLM client
LLM_CLIENT = Groq(api_key=os.getenv("LLM_API_KEY"))
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")


def synthesize_answer(question: str, context: str) -> Dict:

    # Build the prompt
    user_prompt = get_rag_prompt(context, question)
    
    # Call LLM
    response = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,  # Lower = more factual, less creative
        max_tokens=500
    )
    
    return {
        "answer": response.choices[0].message.content,
        "tokens_used": {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens
        }
    }
