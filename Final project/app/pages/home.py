"""
Home Page — Dashboard overview with key metrics and quick actions.
"""
import streamlit as st


def render():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 40px 0 20px 0;">
        <h1 style="font-size: 3rem; font-weight: 800;
            background: linear-gradient(135deg, #6366F1, #06B6D4);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🧠 LearnFlow AI
        </h1>
        <p style="color: #94A3B8; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
            AI-powered personalized learning journeys for every student.
            Powered by Deep Knowledge Tracing, XGBoost, and LLM agents.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    agent = st.session_state.get("agent")
    student_id = st.session_state.get("selected_student", 10000)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Active Students</div>
            <div class="metric-value">2,000+</div>
            <div class="metric-delta delta-positive">↑ from OULAD dataset</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Concepts Tracked</div>
            <div class="metric-value">24</div>
            <div class="metric-delta delta-positive">Across 4 modules</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">DKT Model AUC</div>
            <div class="metric-value">>0.75</div>
            <div class="metric-delta delta-positive">↑ LSTM-based</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Resources Available</div>
            <div class="metric-value">20+</div>
            <div class="metric-delta delta-positive">Videos, PDFs, Practice</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick Actions
    st.markdown('<div class="section-header">⚡ Quick Actions</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="glass-container" style="text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">🎯</div>
            <h3 style="color: #F8FAFC; font-size: 1.1rem;">Analyze Student</h3>
            <p style="color: #94A3B8; font-size: 0.85rem;">
                Run full pipeline: mastery tracking → gap detection → recommendations → study plan
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Run Full Analysis", key="home_analyze", use_container_width=True):
            if agent:
                with st.spinner("Running full analysis pipeline..."):
                    try:
                        result = agent.analyze_student(student_id)
                        st.session_state.analysis_cache[student_id] = result
                        st.success(f"✅ Analysis complete for Student {student_id}")
                        st.markdown(result.get("summary", ""))
                    except Exception as e:
                        st.error(f"Error: {e}")

    with col2:
        st.markdown("""
        <div class="glass-container" style="text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">📊</div>
            <h3 style="color: #F8FAFC; font-size: 1.1rem;">View Mastery</h3>
            <p style="color: #94A3B8; font-size: 0.85rem;">
                Check current knowledge levels across all concepts with radar visualization
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Check Mastery", key="home_mastery", use_container_width=True):
            st.session_state.current_page = "Knowledge Tracing"
            st.rerun()

    with col3:
        st.markdown("""
        <div class="glass-container" style="text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">💬</div>
            <h3 style="color: #F8FAFC; font-size: 1.1rem;">Chat with AI</h3>
            <p style="color: #94A3B8; font-size: 0.85rem;">
                Ask questions about progress, get study tips, or explore recommendations
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Chat", key="home_chat", use_container_width=True):
            st.session_state.current_page = "AI Chat"
            st.rerun()

    # Architecture overview
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🏗️ System Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-container">
        <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; text-align: center;">
            <div style="padding: 16px; background: rgba(99, 102, 241, 0.1); border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.2);">
                <div style="font-size: 1.5rem;">📥</div>
                <div style="color: #F8FAFC; font-weight: 600; font-size: 0.8rem; margin-top: 8px;">Step 1</div>
                <div style="color: #94A3B8; font-size: 0.7rem;">Data & EDA</div>
            </div>
            <div style="padding: 16px; background: rgba(139, 92, 246, 0.1); border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.2);">
                <div style="font-size: 1.5rem;">⚙️</div>
                <div style="color: #F8FAFC; font-weight: 600; font-size: 0.8rem; margin-top: 8px;">Step 2</div>
                <div style="color: #94A3B8; font-size: 0.7rem;">Features</div>
            </div>
            <div style="padding: 16px; background: rgba(6, 182, 212, 0.1); border-radius: 12px; border: 1px solid rgba(6, 182, 212, 0.2);">
                <div style="font-size: 1.5rem;">🧬</div>
                <div style="color: #F8FAFC; font-weight: 600; font-size: 0.8rem; margin-top: 8px;">Step 3</div>
                <div style="color: #94A3B8; font-size: 0.7rem;">DKT (LSTM)</div>
            </div>
            <div style="padding: 16px; background: rgba(245, 158, 11, 0.1); border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.2);">
                <div style="font-size: 1.5rem;">🎯</div>
                <div style="color: #F8FAFC; font-weight: 600; font-size: 0.8rem; margin-top: 8px;">Step 4</div>
                <div style="color: #94A3B8; font-size: 0.7rem;">Gap + Recommender</div>
            </div>
            <div style="padding: 16px; background: rgba(34, 197, 94, 0.1); border-radius: 12px; border: 1px solid rgba(34, 197, 94, 0.2);">
                <div style="font-size: 1.5rem;">🤖</div>
                <div style="color: #F8FAFC; font-weight: 600; font-size: 0.8rem; margin-top: 8px;">Step 5</div>
                <div style="color: #94A3B8; font-size: 0.7rem;">AI Agent</div>
            </div>
            <div style="padding: 16px; background: rgba(239, 68, 68, 0.1); border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.2);">
                <div style="font-size: 1.5rem;">🖥️</div>
                <div style="color: #F8FAFC; font-weight: 600; font-size: 0.8rem; margin-top: 8px;">Step 6</div>
                <div style="color: #94A3B8; font-size: 0.7rem;">Dashboard</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ML Models used
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🧪 ML/DL Models</div>', unsafe_allow_html=True)

    models_data = [
        ("Deep Knowledge Tracing", "LSTM", "Sequential mastery tracking per concept", "AUC-ROC > 0.75"),
        ("Gap Detector", "XGBoost", "Weak topic identification with severity ranking", "F1 > 0.70"),
        ("Collaborative Filtering", "SVD", "Student similarity-based resource matching", "NDCG@5"),
        ("Content-Based Filter", "Sentence-BERT", "Semantic matching of resources to gaps", "Precision@3"),
        ("LLM Ranker", "OpenRouter", "Contextual re-ranking of recommendations", "Qualitative"),
    ]

    for name, tech, desc, metric in models_data:
        st.markdown(f"""
        <div class="resource-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #F8FAFC; font-weight: 700;">{name}</span>
                    <span class="resource-type-badge badge-video" style="margin-left: 8px;">{tech}</span>
                </div>
                <span style="color: #06B6D4; font-weight: 600; font-size: 0.85rem;">{metric}</span>
            </div>
            <p style="color: #94A3B8; font-size: 0.85rem; margin: 8px 0 0 0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
