"""
Chat Page - Document Q&A with conversation memory.
"""

import streamlit as st
from components.file_uploader import render_file_uploader
from components.chat import render_chat_history, render_user_message, render_assistant_message
from utils.api_client import chat_with_memory, ask_question
from utils.session import add_message, get_chat_history, get_session_id


def render():
    """Render the chat page."""
    
    st.title("💬 Document Q&A")
    st.caption("Ask questions about your uploaded legal documents")
    
    # Layout: Chat (left) + Upload (right sidebar)
    chat_col, upload_col = st.columns([3, 1])
    
    with upload_col:
        render_file_uploader()
        
        # Search options
        st.divider()
        st.subheader("⚙️ Options")
        use_cache = st.checkbox("Use cache", value=True, help="Enable semantic caching for faster responses")
        use_hybrid = st.checkbox("Hybrid search", value=False, help="Combine keyword and semantic search")
    
    with chat_col:
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Render existing messages
            messages = get_chat_history()
            if not messages:
                st.info("👋 Hello! Upload a document and start asking questions.")
                
                # Example questions
                with st.expander("💡 Example questions"):
                    st.markdown("""
                    - What is the liability cap amount?
                    - How can I terminate this contract?
                    - Are there any data sharing violations?
                    - What are the payment terms?
                    - Explain the GDPR requirements in this document
                    """)
            
            render_chat_history(messages)
        
        # Chat input at the bottom
        st.divider()
        
        if prompt := st.chat_input("Ask about your documents...", key="chat_input"):
            try:
                # Display user message
                render_user_message(prompt)
                add_message("user", prompt)
                
                # Get AI response
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Analyzing documents..."):
                        
                        if use_hybrid:
                            result = ask_question(prompt, use_cache=use_cache)
                        else:
                            result = chat_with_memory(prompt, get_session_id())
                        
                        answer = result.get("answer", "Sorry, I couldn't process that.")
                        st.markdown(answer)
                        
                        # Show metadata
                        metadata = {
                            "sources": result.get("sources", []),
                            "cache_hit": result.get("cache_hit", False),
                            "provider": result.get("provider", "groq"),
                            "tokens": result.get("tokens_used", {}).get("total", 0)
                        }
                        
                        # Show sources
                        if metadata["sources"]:
                            with st.expander("📌 Sources"):
                                for src in metadata["sources"]:
                                    st.caption(f"• {src}")
                        
                        # Show cache indicator
                        if metadata["cache_hit"]:
                            st.caption("⚡ Instant response from cache (0 tokens used)")
                        elif metadata["tokens"]:
                            st.caption(f"🔢 {metadata['tokens']} tokens used")
                        
                        # Store in session
                        add_message("assistant", answer, metadata)
            
            except Exception as e:
                st.error(f"❌ Something went wrong: {str(e)[:150]}")
                st.info("Please try again or contact support if the problem continues.")
                # Optional: Log to console for debugging
                print(f"Chat error: {e}")
            
            st.rerun()


if __name__ == "__main__":
    render()