"""
Knowledge Tracing Page — DKT model visualization and mastery analysis.
"""
import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import format_module_name


def render():
    st.markdown('<div class="section-header">🧬 Knowledge Tracing</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8;">Deep Knowledge Tracing (LSTM) — Track mastery per concept over time.</p>',
                unsafe_allow_html=True)

    student_id = st.session_state.get("selected_student", 10000)
    agent = st.session_state.get("agent")

    if not agent:
        st.warning("Agent not initialized. Please run the pipeline first.")
        return

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Run Knowledge Tracing", key="run_kt", use_container_width=True):
            with st.spinner("Running DKT model..."):
                try:
                    result = agent.get_mastery(student_id)
                    st.session_state["kt_result"] = result
                except Exception as e:
                    st.error(f"Error: {e}")

    result = st.session_state.get("kt_result")
    if not result:
        # Auto-run
        try:
            result = agent.get_mastery(student_id)
            st.session_state["kt_result"] = result
        except Exception:
            st.info("Click 'Run Knowledge Tracing' to analyze this student.")
            return

    # Overall mastery card
    avg = result.get("average_mastery", 0)
    level = result.get("overall_level", "Unknown")
    strong = result.get("strong_concepts", [])
    weak = result.get("weak_concepts", [])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Overall Mastery</div>
            <div class="metric-value">{avg:.0%}</div>
            <div class="metric-delta">{level}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Concepts Tracked</div>
            <div class="metric-value">{result.get('num_concepts', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Strong</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #22C55E, #06B6D4);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{len(strong)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Weak</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #EF4444, #F97316);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{len(weak)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Mastery radar chart
    mastery_data = result.get("mastery_by_concept", {})
    if mastery_data:
        from src.utils.visualization import mastery_radar_chart
        fig = mastery_radar_chart(mastery_data, f"Concept Mastery — Student {student_id}")
        st.plotly_chart(fig, use_container_width=True)

    # Detailed mastery table
    st.markdown('<div class="section-header">📋 Detailed Mastery Scores</div>', unsafe_allow_html=True)

    from config import MASTERY_LEVELS
    for concept, info in mastery_data.items():
        score = info["score"] if isinstance(info, dict) else info
        level_num = info.get("level_num", 0) if isinstance(info, dict) else 0
        level_info = MASTERY_LEVELS.get(level_num, MASTERY_LEVELS[0])

        pct = int(score * 100)
        st.markdown(f"""
        <div class="resource-card" style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <span style="color: #F8FAFC; font-weight: 600;">{format_module_name(concept)}</span>
                <span class="resource-type-badge" style="margin-left: 8px;
                    background: rgba({int(level_info['color'][1:3], 16)}, {int(level_info['color'][3:5], 16)}, {int(level_info['color'][5:7], 16)}, 0.2);
                    color: {level_info['color']};">
                    {level_info['label']}
                </span>
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 150px; height: 8px; background: var(--bg-tertiary, #334155); border-radius: 4px;">
                    <div style="width: {pct}%; height: 100%; background: {level_info['color']};
                        border-radius: 4px; transition: width 0.5s ease;"></div>
                </div>
                <span style="color: {level_info['color']}; font-weight: 700; min-width: 45px;">{score:.0%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
