"""
Fallback Chain - Try multiple LLM providers in order.
If Groq fails, try Ollama. If Ollama fails, use cache.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from typing import Dict, Optional
from groq import Groq
from dotenv import load_dotenv
from caching.semantic_cache import semantic_cache
from resilience.circuit_breaker import groq_breaker

load_dotenv()


class LLMFallbackChain:
    """Chain of LLM providers with automatic fallback."""
    
    def __init__(self):
        self.providers = [
            self._try_groq,
            self._try_cache,
            self._try_static_response
        ]
    
    def generate(self, system_prompt: str, user_prompt: str, question: str) -> Dict:
        """Try each provider in order until one works."""
        errors = []
        
        for provider in self.providers:
            try:
                result = provider(system_prompt, user_prompt, question)
                if result and result.get("answer"):
                    result["errors_bypassed"] = errors
                    return result
            except Exception as e:
                errors.append(f"{provider.__name__}: {str(e)}")
                continue
        
        return {
            "answer": "I apologize, but I'm unable to process your request right now. All services are unavailable. Please try again later.",
            "all_failed": True,
            "errors": errors
        }
    
    def _try_groq(self, system_prompt: str, user_prompt: str, question: str) -> Optional[Dict]:
        """Try Groq API with circuit breaker."""
        def call_groq():
            client = Groq(api_key=os.getenv("LLM_API_KEY"))
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return {
                "answer": response.choices[0].message.content,
                "provider": "groq",
                "tokens": response.usage.total_tokens
            }
        
        return groq_breaker.call(call_groq)
    
    def _try_cache(self, system_prompt: str, user_prompt: str, question: str) -> Optional[Dict]:
        """Try semantic cache."""
        cached = semantic_cache.get(question)
        if cached:
            cached["provider"] = "cache"
            return cached
        return None
    
    def _try_static_response(self, system_prompt: str, user_prompt: str, question: str) -> Dict:
        """Static fallback when all else fails."""
        return {
            "answer": "I'm currently experiencing connectivity issues. Your document data is safe. Please try again in a moment.",
            "provider": "static_fallback",
            "tokens": 0
        }


# Global fallback chain
fallback_chain = LLMFallbackChain()


if __name__ == "__main__":
    result = fallback_chain.generate(
        "You are a helpful assistant.",
        "What is AI?",
        "What is AI?"
    )
    print(f"Provider: {result.get('provider', 'unknown')}")
    print(f"Answer: {result['answer'][:100]}...")