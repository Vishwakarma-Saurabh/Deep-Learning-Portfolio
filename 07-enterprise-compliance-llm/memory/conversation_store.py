"""
Conversation Memory - Maintains context across multiple questions.
System remembers previous questions and answers.
"""

from typing import List, Dict
from collections import defaultdict
import time


class ConversationStore:
    """Stores conversation history per user/session."""
    
    def __init__(self, max_history: int = 10, ttl_minutes: int = 30):
        self.max_history = max_history
        self.ttl = ttl_minutes * 60
        self.conversations: Dict[str, List[Dict]] = defaultdict(list)
    
    def add_exchange(self, session_id: str, question: str, answer: str):
        """Add a Q&A pair to conversation history."""
        self._cleanup(session_id)
        
        self.conversations[session_id].append({
            "question": question,
            "answer": answer[:200],  # Truncate for context
            "timestamp": time.time()
        })
        
        # Keep only recent history
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]
    
    def get_context(self, session_id: str) -> str:
        """Get conversation context for prompt injection."""
        self._cleanup(session_id)
        
        history = self.conversations[session_id]
        if not history:
            return ""
        
        context = "Previous conversation:\n"
        for exchange in history[-5:]:  # Last 5 exchanges
            context += f"User: {exchange['question']}\n"
            context += f"Assistant: {exchange['answer'][:100]}...\n"
        
        return context
    
    def _cleanup(self, session_id: str):
        """Remove expired conversations."""
        now = time.time()
        self.conversations[session_id] = [
            ex for ex in self.conversations[session_id]
            if now - ex["timestamp"] < self.ttl
        ]
    
    def get_stats(self) -> Dict:
        """Get conversation statistics."""
        total_sessions = len(self.conversations)
        total_exchanges = sum(len(v) for v in self.conversations.values())
        return {
            "active_sessions": total_sessions,
            "total_exchanges": total_exchanges
        }


# Global store
conversation_store = ConversationStore()


if __name__ == "__main__":
    store = ConversationStore()
    
    store.add_exchange("user1", "What is the liability cap?", "It's $500,000.")
    store.add_exchange("user1", "How do I terminate?", "30 days written notice.")
    
    context = store.get_context("user1")
    print(context)
    print(store.get_stats())