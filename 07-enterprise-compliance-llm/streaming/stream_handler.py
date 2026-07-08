"""
Streaming Handler - Sends LLM responses token by token.
Provides ChatGPT-like real-time text generation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import asyncio
from typing import AsyncGenerator
from groq import Groq, AsyncGroq
from dotenv import load_dotenv

load_dotenv()


class StreamHandler:
    """Handles streaming LLM responses."""
    
    def __init__(self):
        self.client = AsyncGroq(api_key=os.getenv("LLM_API_KEY"))
    
    async def stream_response(self, system_prompt: str, user_prompt: str) -> AsyncGenerator[str, None]:

        try:
            stream = await self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                stream=True
            )
            
            full_response = ""
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    
                    # SSE format
                    data = json.dumps({"token": token, "full": full_response})
                    yield f"data: {data}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'done': True, 'full': full_response})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    async def stream_to_list(self, system_prompt: str, user_prompt: str) -> dict:
        """Stream and collect full response."""
        tokens = []
        full_text = ""
        
        async for chunk in self.stream_response(system_prompt, user_prompt):
            data = json.loads(chunk.replace("data: ", ""))
            if "token" in data:
                tokens.append(data["token"])
                full_text = data["full"]
            elif "done" in data:
                break
        
        return {
            "answer": full_text,
            "tokens": len(tokens),
            "streamed": True
        }


# Global handler
stream_handler = StreamHandler()


# Synchronous wrapper for easy use
def generate_streaming(system_prompt: str, user_prompt: str) -> dict:
    """Synchronous wrapper for streaming."""
    import asyncio
    return asyncio.run(stream_handler.stream_to_list(system_prompt, user_prompt))


if __name__ == "__main__":
    result = generate_streaming(
        "You are a helpful assistant.",
        "Explain what a circuit breaker is in one sentence."
    )
    print(f"Answer: {result['answer']}")
    print(f"Tokens: {result['tokens']}")