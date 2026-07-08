"""
Synthesizer - Generates answers using retrieved context.
Uses Groq LLM with semantic caching and fallback support.
"""

import os
from typing import Dict
from groq import Groq
from dotenv import load_dotenv
from rag.prompts import RAG_SYSTEM_PROMPT, get_rag_prompt

load_dotenv()

# Initialize Groq client
LLM_CLIENT = Groq(api_key=os.getenv("LLM_API_KEY"))
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")


def synthesize_answer(question: str, context: str) -> Dict:

    # Check if context is empty
    if not context or context.strip() == "":
        return {
            "answer": "No relevant documents found. Please ingest documents first using the /ingest endpoint, or try rephrasing your question.",
            "tokens_used": {
                "prompt": 0,
                "completion": 0,
                "total": 0
            }
        }
    
    # Build the prompt
    user_prompt = get_rag_prompt(context, question)
    
    try:
        # Call Groq LLM
        response = LLM_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower = more factual, less creative
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        
        # If answer says "not present", verify context actually has content
        if "not present" in answer.lower() and len(context) > 100:
            # Context exists but model couldn't find answer - try with stronger prompt
            retry_prompt = f"""The following context IS available. Please extract the relevant information.

Context:
{context}

Question: {question}

Find the answer in the context above. If the exact words aren't there, summarize what the context says about this topic:"""
            
            retry_response = LLM_CLIENT.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer based on the provided context. Extract relevant information even if not worded exactly as asked."},
                    {"role": "user", "content": retry_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = retry_response.choices[0].message.content
            total_tokens = response.usage.total_tokens + retry_response.usage.total_tokens
        
        else:
            total_tokens = response.usage.total_tokens
        
        return {
            "answer": answer,
            "tokens_used": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": total_tokens
            }
        }
    
    except Exception as e:
        return {
            "answer": f"I encountered an error while generating a response: {str(e)}. Please try again.",
            "tokens_used": {
                "prompt": 0,
                "completion": 0,
                "total": 0
            },
            "error": str(e)
        }


# Test independently
if __name__ == "__main__":
    test_context = "[Source: contract.pdf, Chunk 2]\nLiability is capped at $500,000 for any damages."
    test_question = "What is the liability cap?"
    
    result = synthesize_answer(test_question, test_context)
    print(f"Answer: {result['answer']}")
    print(f"Tokens: {result['tokens_used']}")