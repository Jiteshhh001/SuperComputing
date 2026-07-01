"""
Weak Topic Analysis Page — Gap detection with severity visualization.
"""
import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def render():
    st.markdown('<div class="section-header">🎯 Weak Topic Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8;">XGBoost predicts, for each concept the student takes, the risk of scoring below 50% on the next assessment.</p>',
                unsafe_allow_html=True)

    student_id = st.session_state.get("selected_student", 10000)
    agent = st.session_state.get("agent")

    if not agent:
        st.warning("Agent not initialized.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider("Concepts to show", 1, 15, 6, key="gap_topk")
    with col2:
        run = st.button("🔄 Refresh", key="run_gaps", use_container_width=True)

    # Auto-run or on button
    try:
        gaps = agent.get_gaps(student_id)
    except Exception as e:
        st.error(f"Error: {e}")
        return

    weak_topics = gaps.get("weak_topics", [])
    ranked = gaps.get("all_concepts") or weak_topics

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">At-Risk Concepts</div>
            <div class="metric-value">{gaps.get('total_gaps', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Critical</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #EF4444, #DC2626);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {gaps.get('critical_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">High Priority</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #F59E0B, #D97706);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {gaps.get('high_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gap severity chart
    if not weak_topics and ranked:
        st.success("🎉 No concept is flagged as at-risk. Concepts are still ranked below so the "
                   "student knows where to focus next.")
    if ranked:
        from src.utils.visualization import gap_severity_chart
        fig = gap_severity_chart(ranked[:top_k], f"Next-assessment risk by concept — Student {student_id}")
        st.plotly_chart(fig, use_container_width=True)

    # Detailed gap cards
    st.markdown('<div class="section-header">📋 Concept Risk Ranking</div>', unsafe_allow_html=True)

    from config import format_module_name
    
    for i, gap in enumerate(ranked[:top_k]):
        severity = gap.get("severity", "Low")
        severity_class = f"severity-{severity.lower()}"
        confidence = gap.get("confidence", 0)
        raw_concept = gap.get("concept", "Unknown")
        concept = format_module_name(raw_concept)

        icon = {"Critical": "🔴", "High": "🟠", "Moderate": "🟡", "Low": "🔵"}.get(severity, "⚪")

        st.markdown(f"""
        <div class="resource-card" style="border-left: 4px solid
            {'#EF4444' if severity == 'Critical' else '#F59E0B' if severity == 'High' else '#EAB308' if severity == 'Moderate' else '#06B6D4'};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.2rem;">{icon}</span>
                    <span style="color: #F8FAFC; font-weight: 700; font-size: 1.05rem; margin-left: 8px;">
                        #{i+1} — {concept}
                    </span>
                </div>
                <span class="resource-type-badge {severity_class}" style="border-radius: 20px; padding: 4px 16px;">
                    {severity}{" · at risk" if gap.get("is_weak", True) else ""}
                </span>
            </div>
            <div style="margin-top: 12px; display: flex; gap: 24px;">
                <div>
                    <span style="color: #94A3B8; font-size: 0.8rem;">Risk of &lt;50% next</span>
                    <div style="color: #F8FAFC; font-weight: 600;">{confidence:.0%}</div>
                </div>
                <div>
                    <span style="color: #94A3B8; font-size: 0.8rem;">Priority</span>
                    <div style="color: #F8FAFC; font-weight: 600;">#{gap.get('priority', i+1)}</div>
                </div>
                <div>
                    <span style="color: #94A3B8; font-size: 0.8rem;">Average score so far</span>
                    <div style="color: #F8FAFC; font-weight: 600;">{(gap.get("avg_score") or 0):.0%}</div>
                </div>
                <div style="flex: 1;">
                    <span style="color: #94A3B8; font-size: 0.8rem;">Risk</span>
                    <div style="width: 100%; height: 8px; background: #334155; border-radius: 4px; margin-top: 6px;">
                        <div style="width: {int(confidence*100)}%; height: 100%;
                            background: {'#EF4444' if severity == 'Critical' else '#F59E0B'}; border-radius: 4px;"></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if not ranked:
        st.info("No assessment data for this student yet.")
