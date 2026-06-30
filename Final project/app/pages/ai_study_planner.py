"""
AI Study Planner Page — Personalized 7-day study plan with interactive timeline.
"""
import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def render():
    st.markdown('<div class="section-header">📅 AI Study Planner</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8;">Rule-based + LLM powered personalized 7-day study schedule.</p>',
                unsafe_allow_html=True)

    student_id = st.session_state.get("selected_student", 10000)
    agent = st.session_state.get("agent")

    if not agent:
        st.warning("Agent not initialized.")
        return

    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        daily_hours = st.slider("Daily study hours", 0.5, 6.0, 2.0, 0.5, key="plan_hours")
    with col2:
        generate = st.button("📝 Generate Plan", key="gen_plan", use_container_width=True)

    # Generate plan
    plan = None
    if generate or "study_plan" not in st.session_state:
        with st.spinner("Creating your personalized study plan..."):
            try:
                plan = agent.get_study_plan(student_id, daily_hours)
                st.session_state["study_plan"] = plan
            except Exception as e:
                st.error(f"Error: {e}")
                return
    else:
        plan = st.session_state.get("study_plan")

    if not plan:
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Study Time</div>
            <div class="metric-value">{plan.get('total_study_minutes', 0)} min</div>
            <div class="metric-delta">{plan.get('total_study_minutes', 0) / 60:.1f} hours</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Topics Covered</div>
            <div class="metric-value">{len(plan.get('topics_covered', []))}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Daily Target</div>
            <div class="metric-value">{daily_hours}h</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual timeline
    from src.utils.visualization import study_plan_timeline
    fig = study_plan_timeline(plan)
    st.plotly_chart(fig, use_container_width=True)

    # Day-by-day cards
    st.markdown('<div class="section-header">📋 Daily Breakdown</div>', unsafe_allow_html=True)

    days = plan.get("days", [])
    for day in days:
        is_review = day.get("review", False)
        border_color = "#8B5CF6" if is_review else "#6366F1"
        day_icon = "📖" if is_review else f"📅"

        activities_html = ""
        for activity in day.get("activities", []):
            act_type = activity.get("type", "study")
            act_icon = {
                "video": "🎬", "reading": "📄", "practice": "✏️",
                "review": "📖", "assessment": "📝", "break": "🧘",
                "reflection": "💭",
            }.get(act_type, "📎")

            resource = activity.get("resource")
            resource_note = ""
            if resource:
                title = resource.get("title", "")[:50]
                url = resource.get("url")
                if url:
                    resource_note = f'<br><a href="{url}" target="_blank" style="color: #8B5CF6; font-size: 0.8rem; text-decoration: none; font-weight: 600; display: inline-block; margin-top: 4px;">↳ {title} ↗</a>'
                else:
                    resource_note = f'<br><span style="color: #64748B; font-size: 0.75rem; display: inline-block; margin-top: 4px;">↳ {title}</span>'

            activities_html += f"""<div style="display: flex; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
<span style="font-size: 1.1rem; width: 30px;">{act_icon}</span>
<div style="flex: 1;">
<span style="color: #F8FAFC; font-size: 0.9rem;">{activity.get('activity', '')}</span>
{resource_note}
</div>
<span style="color: #06B6D4; font-weight: 600; font-size: 0.85rem; min-width: 60px; text-align: right;">{activity.get('time', '')}</span>
</div>"""

        from config import format_module_name
        focus_topic_raw = day.get('focus_topic', '')
        focus_topic = format_module_name(focus_topic_raw)

        st.markdown(f"""
        <div class="day-card" style="border-left-color: {border_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <span style="font-size: 1.3rem;">{day_icon}</span>
                    <span style="color: #F8FAFC; font-weight: 700; font-size: 1.1rem; margin-left: 8px;">
                        {day.get('day_label', '')}
                    </span>
                    <span style="color: #94A3B8; font-size: 0.85rem; margin-left: 8px;">
                        — {focus_topic}
                    </span>
                </div>
                <span style="color: #06B6D4; font-weight: 600;">
                    {day.get('total_minutes', 0)} min
                </span>
            </div>
            {activities_html}
        </div>
        """, unsafe_allow_html=True)

    # Motivation
    motivation = plan.get("motivation", "")
    if motivation:
        st.markdown(f"""
        <div class="glass-container" style="text-align: center; margin-top: 20px;
            border: 1px solid rgba(99, 102, 241, 0.3);">
            <p style="color: #F8FAFC; font-size: 1.1rem; font-weight: 600; margin: 0;">
                {motivation}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("✅ Simulate Completing This Plan (Boost Progress)", key="sim_complete", use_container_width=True):
        with st.spinner("Simulating a week of studying..."):
            agent.simulate_study_plan_completion(student_id, plan)
            st.success("Study plan completed! Check the Progress Analytics page to see your improvement.")
