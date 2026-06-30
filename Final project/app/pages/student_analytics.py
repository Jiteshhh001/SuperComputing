"""
Student Analytics Page — Comprehensive student profile and performance view.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def render():
    st.markdown('<div class="section-header">👤 Student Analytics</div>', unsafe_allow_html=True)

    student_id = st.session_state.get("selected_student", 10000)
    agent = st.session_state.get("agent")

    if not agent:
        st.warning("Agent not initialized. Please run the pipeline first.")
        return

    # Profile section
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div class="glass-container" style="text-align: center;">
            <div style="width: 80px; height: 80px; border-radius: 50%;
                background: linear-gradient(135deg, #6366F1, #06B6D4);
                display: flex; align-items: center; justify-content: center;
                margin: 0 auto 16px auto; font-size: 2rem;">👤</div>
            <h3 style="color: #F8FAFC; margin: 0;">Student {student_id}</h3>
            <p style="color: #94A3B8; font-size: 0.85rem;">ID: {student_id}</p>
        </div>
        """, unsafe_allow_html=True)

        # Memory info
        if agent and agent.memory:
            profile = agent.memory.profiles.get_profile(student_id)
            summary = agent.memory.profiles.get_summary(student_id)
            st.markdown(f"""
            <div class="glass-container">
                <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">
                    Memory Status
                </div>
                <div style="color: #F8FAFC; font-size: 0.85rem; white-space: pre-line;">
                    {summary}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        # Run analysis if not cached
        if st.button("🔄 Refresh Analysis", key="refresh_analytics"):
            with st.spinner("Running analysis..."):
                try:
                    result = agent.analyze_student(student_id)
                    st.session_state.analysis_cache[student_id] = result
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        cached = st.session_state.analysis_cache.get(student_id)

        if cached:
            mastery = cached.get("mastery", {})

            # Key metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                avg = mastery.get("average_mastery", 0)
                st.metric("Avg Mastery", f"{avg:.0%}", f"{mastery.get('overall_level', 'N/A')}")
            with m2:
                st.metric("Strong Topics", len(mastery.get("strong_concepts", [])))
            with m3:
                st.metric("Weak Topics", len(mastery.get("weak_concepts", [])))
            with m4:
                gaps = cached.get("gaps", {})
                st.metric("Critical Gaps", gaps.get("critical_count", 0))

            # Mastery radar
            from src.utils.visualization import mastery_radar_chart
            mastery_data = mastery.get("mastery_by_concept", {})
            if mastery_data:
                fig = mastery_radar_chart(mastery_data, f"Mastery Profile — Student {student_id}")
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Click 'Refresh Analysis' to load student data.")

    # Student profiles table
    st.markdown('<div class="section-header">📋 All Students Overview</div>', unsafe_allow_html=True)

    try:
        from config import format_module_name
        profiles_path = PROJECT_ROOT / "data" / "processed" / "student_profiles.csv"
        if profiles_path.exists():
            df = pd.read_csv(profiles_path)
            
            if "code_module" in df.columns:
                df["code_module"] = df["code_module"].apply(format_module_name)
                
            display_cols = [c for c in ["id_student", "code_module", "gender", "age_band",
                                         "highest_education", "final_result", "avg_score",
                                         "total_clicks", "engagement_level", "risk_level"]
                            if c in df.columns]
            st.dataframe(
                df[display_cols].head(50),
                use_container_width=True,
                height=400,
            )
        else:
            st.info("No processed data available. Run the pipeline first.")
    except Exception as e:
        st.warning(f"Could not load student profiles: {e}")
