"""
AI Chat Page — Conversational interface with the LearnFlow AI agent.
Full chat UI with memory-backed conversations, tool invocations,
and real-time streaming-style responses.
"""
import streamlit as st
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def render():
    st.markdown('<div class="section-header">💬 AI Chat</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: #94A3B8;">'
        'Chat with LearnFlow AI about your learning journey. '
        'Ask about mastery, weak topics, study plans, or anything academic.'
        '</p>',
        unsafe_allow_html=True,
    )

    student_id = st.session_state.get("selected_student", 10000)
    agent = st.session_state.get("agent")

    if not agent:
        st.warning("Agent not initialized. Please run the pipeline first.")
        return

    # ── Chat Header ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="width: 40px; height: 40px; border-radius: 50%;
                background: linear-gradient(135deg, #6366F1, #06B6D4);
                display: flex; align-items: center; justify-content: center;
                font-size: 1.2rem;">🤖</div>
            <div>
                <div style="color: #F8FAFC; font-weight: 700;">LearnFlow AI</div>
                <div style="color: #22C55E; font-size: 0.75rem;">● Online — Student {student_id}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        memory_label = "4-Layer Memory Active" if agent.memory else "No Memory"
        st.markdown(f"""
        <div style="text-align: right; padding-top: 8px;">
            <span style="background: rgba(34, 197, 94, 0.15); color: #22C55E;
                padding: 4px 12px; border-radius: 20px; font-size: 0.75rem;
                font-weight: 600; border: 1px solid rgba(34, 197, 94, 0.3);">
                🧠 {memory_label}
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        if st.button("🗑️ Clear Chat", key="clear_chat", use_container_width=True):
            st.session_state.chat_history = []
            if agent and agent.memory:
                agent.memory.conversation.clear(student_id)
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Initialize Chat History ─────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Add welcome message if history is empty
    if not st.session_state.chat_history:
        welcome_msg = (
            f"👋 Hi! I'm **LearnFlow AI**, your personal learning assistant.\n\n"
            f"I'm here to help **Student {student_id}** with their learning journey. "
            f"Here are some things you can ask me:\n\n"
            f"- 📊 *\"What's my current mastery level?\"*\n"
            f"- 🎯 *\"What are my weakest topics?\"*\n"
            f"- 📚 *\"Recommend resources for my weak areas\"*\n"
            f"- 📅 *\"Create a study plan for this week\"*\n"
            f"- 📈 *\"How am I doing? Show my progress\"*\n\n"
            f"I have full memory of our conversations and your learning history. Let's get started! 🚀"
        )
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": welcome_msg,
            "timestamp": datetime.now().isoformat(),
        })

    # ── Chat Message Display ────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]
            timestamp = msg.get("timestamp", "")
            time_str = ""
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%H:%M")
                except (ValueError, TypeError):
                    time_str = ""

            if role == "user":
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin: 12px 0;">
                    <div class="chat-user">
                        <div style="margin-bottom: 4px;">{content}</div>
                        <div style="text-align: right; font-size: 0.7rem; opacity: 0.7;">{time_str}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin: 12px 0;">
                    <div style="display: flex; gap: 10px; max-width: 85%;">
                        <div style="width: 32px; height: 32px; border-radius: 50%;
                            background: linear-gradient(135deg, #6366F1, #06B6D4);
                            display: flex; align-items: center; justify-content: center;
                            font-size: 0.9rem; flex-shrink: 0; margin-top: 4px;">🤖</div>
                        <div class="chat-assistant">
                            <div style="margin-bottom: 4px;">{content}</div>
                            <div style="font-size: 0.7rem; color: #64748B;">{time_str}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Quick Action Buttons ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    quick_cols = st.columns(5)
    quick_actions = [
        ("📊 Mastery", "Show my current mastery levels across all concepts"),
        ("🎯 Weak Topics", "What are my weakest topics right now?"),
        ("📚 Recommend", "Recommend the best study resources for my weak areas"),
        ("📅 Study Plan", "Create a personalized 7-day study plan for me"),
        ("📈 Progress", "How am I doing? Show my progress report"),
    ]

    for i, (label, prompt) in enumerate(quick_actions):
        with quick_cols[i]:
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                _send_message(agent, student_id, prompt)
                st.rerun()

    # ── Chat Input ──────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask LearnFlow AI anything about your learning...",
        key="chat_input",
    )

    if user_input:
        _send_message(agent, student_id, user_input)
        st.rerun()

    # ── Memory Sidebar Panel ───────────────────────────────────────────
    with st.expander("🧠 Agent Memory Inspector", expanded=False):
        if agent and agent.memory:
            tab_conv, tab_profile, tab_recs, tab_history = st.tabs([
                "💬 Conversation", "👤 Profile", "📚 Recommendations", "📜 History"
            ])

            with tab_conv:
                conv_history = agent.memory.conversation.get_history(student_id, last_n=10)
                if conv_history:
                    for msg in conv_history:
                        role_icon = "👤" if msg["role"] == "user" else "🤖"
                        st.markdown(f"""
                        <div style="padding: 6px 12px; border-left: 3px solid
                            {'#6366F1' if msg['role'] == 'user' else '#06B6D4'};
                            background: rgba(30, 41, 59, 0.5); border-radius: 4px;
                            margin: 4px 0; font-size: 0.85rem;">
                            <span>{role_icon}</span>
                            <span style="color: #94A3B8; font-size: 0.75rem; margin-left: 4px;">
                                {msg.get('timestamp', '')[:16]}
                            </span>
                            <div style="color: #F8FAFC; margin-top: 4px;">
                                {msg['content'][:150]}{'...' if len(msg['content']) > 150 else ''}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No conversation memory yet.")

            with tab_profile:
                summary = agent.memory.profiles.get_summary(student_id)
                st.markdown(f"""
                <div class="glass-container">
                    <div style="color: #F8FAFC; white-space: pre-line; font-size: 0.9rem;">
                        {summary}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with tab_recs:
                rec_summary = agent.memory.recommendations.get_summary(student_id)
                st.markdown(f"""
                <div class="glass-container">
                    <div style="color: #F8FAFC; white-space: pre-line; font-size: 0.9rem;">
                        {rec_summary}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                past = agent.memory.recommendations.get_past_recommendations(student_id, last_n=5)
                for rec in reversed(past):
                    st.markdown(f"""
                    <div style="padding: 6px 12px; border-left: 3px solid #8B5CF6;
                        background: rgba(30, 41, 59, 0.5); border-radius: 4px; margin: 4px 0;">
                        <span style="color: #94A3B8; font-size: 0.75rem;">
                            {rec.get('timestamp', '')[:16]}
                        </span>
                        <span style="color: #F8FAFC; margin-left: 8px; font-weight: 600;">
                            {rec.get('concept', '')}
                        </span>
                        <span style="color: #94A3B8; margin-left: 8px;">
                            — {len(rec.get('resources', []))} resources
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

            with tab_history:
                weekly = agent.memory.history.get_weekly_summary(student_id)
                st.markdown(f"""
                <div class="glass-container">
                    <pre style="color: #F8FAFC; font-family: 'Inter', sans-serif;
                        white-space: pre-line; font-size: 0.9rem;">{weekly}</pre>
                </div>
                """, unsafe_allow_html=True)

                events = agent.memory.history.get_events(student_id, last_n=10)
                for event in reversed(events):
                    etype = event.get("event_type", "unknown")
                    etime = event.get("timestamp", "")[:16]
                    eicon = {
                        "mastery_update": "📊", "gap_detected": "🎯",
                        "assessment": "📝", "resource_completed": "✅",
                        "study_session": "📖",
                    }.get(etype, "📎")
                    st.markdown(f"""
                    <div style="padding: 6px 12px; border-left: 3px solid #6366F1;
                        background: rgba(30, 41, 59, 0.5); border-radius: 4px; margin: 4px 0;">
                        <span>{eicon}</span>
                        <span style="color: #94A3B8; font-size: 0.75rem; margin-left: 4px;">
                            {etime}
                        </span>
                        <span style="color: #F8FAFC; margin-left: 8px;">
                            {etype.replace('_', ' ').title()}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)


def _send_message(agent, student_id: int, message: str):
    """Send a message and get the agent's response."""
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().isoformat(),
    })

    # Get agent response
    try:
        response = agent.chat(student_id, message)
    except Exception as e:
        response = f"I encountered an error processing your request: {str(e)}. Please try again."

    # Add assistant response to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat(),
    })
