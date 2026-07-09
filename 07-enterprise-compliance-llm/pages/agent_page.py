""" Agent Page - Multi-step autonomous workflow interface. """

import streamlit as st
from utils.api_client import run_agent
from utils.session import add_agent_step, clear_agent_steps, get_agent_steps


def render():
    """Render the agent page."""
    st.title("🤖 AI Agent")
    st.caption("Autonomous multi-step compliance workflows")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Task Description")
        st.info("""
        **Describe what you want the agent to do.**

        The agent can:
        - 🔍 Search through documents
        - 📋 Audit clauses for violations
        - 📊 Generate compliance reports
        - ✉️ Email reports to stakeholders
        """)

        with st.expander("💡 Example tasks"):
            st.markdown("""
            Try these examples:

            1. **Simple audit:**  
               Find all GDPR violations in data sharing clauses

            2. **Report generation:**  
               Audit all payment terms and generate an executive summary

            3. **Full workflow:**  
               Check all vendor contracts for compliance issues, generate a detailed report, and email it to legal@company.com

            4. **Multi-document search:**  
               Search for termination clauses across all documents and check if they're compliant
            """)

        agent_request = st.text_area(
            "What should the agent do?",
            height=150,
            placeholder="Describe the task in plain English...\n\nExample: Find all GDPR violations in data sharing clauses and email a report to legal@company.com",
            key="agent_request"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            max_steps = st.number_input("Max steps", min_value=1, max_value=20, value=8, 
                                      help="Maximum number of tool calls the agent can make")
        with col_b:
            show_details = st.checkbox("Show details", value=True, 
                                     help="Show input/output for each step")

        if st.button("🚀 Execute Agent", type="primary", use_container_width=True, disabled=not agent_request):
            if not agent_request:
                st.warning("Please describe what you want the agent to do")
            else:
                clear_agent_steps()
                progress_container = st.container()
                
                with progress_container:
                    with st.status("🤖 Agent working...", expanded=True) as status:
                        result = run_agent(agent_request, max_steps=max_steps)
                        
                        if result.get("execution_trace"):
                            for step in result["execution_trace"]:
                                if step["type"] == "action":
                                    add_agent_step(step)
                                    action_icons = {
                                        "search_documents": "🔍",
                                        "audit_document_clauses": "📋",
                                        "audit_by_search": "🔍📋",
                                        "generate_report": "📊",
                                        "send_email": "✉️"
                                    }
                                    icon = action_icons.get(step["action"], "🔧")
                                    st.write(f"{icon} **Step {step['step']}:** {step['action']}")
                                    
                                    if show_details:
                                        input_preview = str(step.get("input", ""))[:200]
                                        obs_preview = str(step.get("observation", ""))[:300]
                                        with st.expander(f"Details for Step {step['step']}"):
                                            st.caption("**Input:**")
                                            st.code(input_preview, language="json")
                                            st.caption("**Observation:**")
                                            st.code(obs_preview, language="json")
                                
                                elif step["type"] == "final":
                                    add_agent_step(step)
                                    st.write("📝 **Final Answer:**")
                                    st.info(step.get("output", "")[:500])
                        
                        if result.get("success"):
                            status.update(label=f"✅ Agent completed in {result.get('steps_taken', 0)} steps!", state="complete")
                        else:
                            status.update(label=f"⚠️ Agent stopped after {result.get('steps_taken', 0)} steps", state="error")
                
                st.session_state.last_agent_result = result
                st.rerun()

    with col2:
        st.subheader("📊 Results")

        if "last_agent_result" in st.session_state:
            result = st.session_state.last_agent_result

            if result.get("success"):
                st.success(f"✅ Task completed in {result.get('steps_taken', 0)} steps")
            else:
                st.warning(f"⚠️ Task incomplete after {result.get('steps_taken', 0)} steps")

            st.subheader("📝 Final Answer")
            final_answer = result.get("final_answer", "No answer produced")
            st.markdown(final_answer)

            agent_steps = get_agent_steps()
            if agent_steps:
                with st.expander("🔍 Execution Trace", expanded=False):
                    for i, step in enumerate(agent_steps):
                        if step["type"] == "action":
                            st.markdown(f"**Step {step['step']}:** {step['action']}")
                        else:
                            st.markdown(f"**Final:** {step.get('output', '')[:200]}")
                
                with st.expander("📋 Raw Trace (JSON)", expanded=False):
                    st.json(agent_steps)

            st.divider()
            st.subheader("📤 Actions")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("📧 Email Results", use_container_width=True):
                    email = st.text_input("Email address", placeholder="legal@company.com", key="result_email")
                    if email and st.button("Send", key="send_email_btn"):
                        email_result = run_agent(f"Email the following results to {email}: {final_answer[:500]}")
                        if email_result.get("success"):
                            st.success(f"✅ Sent to {email}")
                        else:
                            st.error("Failed to send email")

            with col_b:
                if st.button("🔄 Run Again", use_container_width=True):
                    st.session_state.last_agent_result = None
                    clear_agent_steps()
                    st.rerun()
        else:
            st.info("👈 Describe a task and click 'Execute Agent' to see results")

    with st.expander("🤔 How agents work"):
        st.markdown("""
        **The AI Agent** uses a ReAct (Reasoning + Acting) pattern:

        **Thought → Action → Observation → Thought → ...**

        1. **Thought**: The agent plans what to do next
        2. **Action**: It selects and uses a tool
        3. **Observation**: It sees the result
        4. **Repeat**: Until the task is complete

        **Available tools:**
        - 🔍 `search_documents`: Find relevant clauses
        - 📋 `audit_document_clauses`: Check for violations
        - 📊 `generate_report`: Create professional reports
        - ✉️ `send_email`: Deliver reports to stakeholders
        """)

    with st.expander("💡 Pro tips"):
        st.markdown("""
        - Be specific about what you want
        - Include email addresses if you want reports sent
        - Use "find", "check", "audit", "generate", "email" keywords
        - The agent works best with clear, actionable tasks
        - Start with simple tasks and build up to complex workflows
        """)


if __name__ == "__main__":
    render()