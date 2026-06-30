"""
Progress Analytics Page — Track improvement over time.
"""
import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import format_module_name


def render():
    st.markdown('<div class="section-header">📈 Progress Analytics</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8;">Track mastery improvement, learning events, and weekly progress reports.</p>',
                unsafe_allow_html=True)

    student_id = st.session_state.get("selected_student", 10000)
    agent = st.session_state.get("agent")

    if not agent:
        st.warning("Agent not initialized.")
        return

    col1, col2 = st.columns([3, 1])
    with col2:
        st.button("🔄 Refresh Progress", key="refresh_progress", use_container_width=True)

    try:
        progress = agent.get_progress(student_id)
    except Exception as e:
        st.error(f"Error: {e}")
        return

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    overall_imp = progress.get("overall_improvement_pct", 0)
    with col1:
        delta_color = "delta-positive" if overall_imp >= 0 else "delta-negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Overall Change</div>
            <div class="metric-value">{overall_imp:+.1f}%</div>
            <div class="metric-delta {delta_color}">
                {'📈 Improving' if overall_imp > 0 else '📉 Declining' if overall_imp < 0 else '➡️ Stable'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Improving Concepts</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #22C55E, #06B6D4);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {len(progress.get('improving_concepts', []))}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Declining Concepts</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #EF4444, #F97316);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {len(progress.get('declining_concepts', []))}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stable Concepts</div>
            <div class="metric-value">{len(progress.get('stable_concepts', []))}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Progress report text
    report_text = progress.get("report_text", "No report available.")
    
    from config import MODULE_NAMES
    for raw_code in MODULE_NAMES.keys():
        for suffix in ["_TMA", "_CMA", "_Exam", ""]:
            target = f"{raw_code}{suffix}"
            if target in report_text:
                report_text = report_text.replace(target, format_module_name(target))

    st.markdown(f"""
    <div class="glass-container">
        <div style="color: #F8FAFC; white-space: pre-line; line-height: 1.8;">
            {report_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Improvement by concept
    improvements = progress.get("improvement_by_concept", {})
    if improvements:
        st.markdown('<div class="section-header">📊 Concept-Level Changes</div>', unsafe_allow_html=True)

        # Sort by change
        sorted_imp = sorted(improvements.items(), key=lambda x: x[1].get("change", 0), reverse=True)

        for concept, data in sorted_imp[:10]:
            change = data.get("change", 0)
            current = data.get("current", 0)
            previous = data.get("previous", 0)

            if change > 0.05:
                indicator = "✅"
                color = "#22C55E"
            elif change < -0.05:
                indicator = "⚠️"
                color = "#EF4444"
            else:
                indicator = "➡️"
                color = "#94A3B8"

            st.markdown(f"""
            <div class="resource-card" style="display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 1.2rem;">{indicator}</span>
                    <span style="color: #F8FAFC; font-weight: 600;">{format_module_name(concept)}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 20px;">
                    <span style="color: #94A3B8;">{previous:.0%}</span>
                    <span style="color: #94A3B8;">→</span>
                    <span style="color: #F8FAFC; font-weight: 600;">{current:.0%}</span>
                    <span style="color: {color}; font-weight: 700; min-width: 55px; text-align: right;">
                        {change:+.0%}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Learning history from memory
    if agent.memory:
        st.markdown('<div class="section-header">📜 Learning History</div>', unsafe_allow_html=True)

        weekly_summary = agent.memory.history.get_weekly_summary(student_id)
        st.markdown(f"""
        <div class="glass-container">
            <pre style="color: #F8FAFC; font-family: 'Inter', sans-serif; white-space: pre-line;">{weekly_summary}</pre>
        </div>
        """, unsafe_allow_html=True)

        # Recent events
        events = agent.memory.history.get_events(student_id, last_n=10)
        if events:
            with st.expander("📋 Recent Learning Events", expanded=False):
                for event in reversed(events):
                    etype = event.get("event_type", "unknown")
                    etime = event.get("timestamp", "")[:16]
                    eicon = {
                        "mastery_update": "📊", "gap_detected": "🎯",
                        "assessment": "📝", "resource_completed": "✅",
                    }.get(etype, "📎")
                    st.markdown(f"""
                    <div style="padding: 6px 12px; border-left: 3px solid #6366F1;
                        background: rgba(30, 41, 59, 0.5); border-radius: 4px; margin: 4px 0;">
                        <span>{eicon}</span>
                        <span style="color: #94A3B8; font-size: 0.8rem; margin-left: 4px;">{etime}</span>
                        <span style="color: #F8FAFC; margin-left: 8px;">{etype.replace('_', ' ').title()}</span>
                    </div>
                    """, unsafe_allow_html=True)
