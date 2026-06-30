"""
Recommendation Center Page — Hybrid resource recommendations.
"""
import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import format_module_name

def render():
    st.markdown('<div class="section-header">📚 Recommendation Center</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color: #94A3B8;">
        Three-stage hybrid recommendations:
        <span style="color: #6366F1; font-weight: 600;">Collaborative Filtering</span> +
        <span style="color: #8B5CF6; font-weight: 600;">Content-Based</span> +
        <span style="color: #06B6D4; font-weight: 600;">LLM Ranking</span>
    </p>
    """, unsafe_allow_html=True)

    student_id = st.session_state.get("selected_student", 10000)
    agent = st.session_state.get("agent")

    if not agent:
        st.warning("Agent not initialized.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider("Number of recommendations", 3, 10, 5, key="rec_topk")
    with col2:
        st.button("🔄 Get Recommendations", key="run_recs", use_container_width=True)

    try:
        recs = agent.get_recommendations(student_id)
    except Exception as e:
        st.error(f"Error: {e}")
        return

    recommendations = recs.get("recommendations", [])
    type_counts = recs.get("resource_types", {})

    # Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Resources</div>
            <div class="metric-value">{recs.get('total_resources', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🎬 Videos</div>
            <div class="metric-value">{type_counts.get('videos', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📄 PDFs</div>
            <div class="metric-value">{type_counts.get('pdfs', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">✏️ Practice</div>
            <div class="metric-value">{type_counts.get('practice', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter tabs
    tab_all, tab_video, tab_pdf, tab_practice = st.tabs(["All", "🎬 Videos", "📄 PDFs", "✏️ Practice"])

    def render_resource_cards(resources):
        if not resources:
            st.info("No resources found for this filter.")
            return

        for r in resources:
            rtype = r.get("type", "resource").lower()
            badge_class = f"badge-{rtype}" if rtype in ["video", "pdf", "practice"] else "badge-video"
            icon = {"video": "🎬", "pdf": "📄", "practice": "✏️"}.get(rtype, "📎")
            difficulty = r.get("difficulty", "unknown")
            duration = r.get("duration_min", "?")
            source = r.get("source", "hybrid")

            diff_color = {"beginner": "#22C55E", "intermediate": "#F59E0B", "advanced": "#EF4444"}.get(difficulty, "#94A3B8")

            st.markdown(f"""
            <div class="resource-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div style="flex: 1;">
                        <span style="font-size: 1.3rem;">{icon}</span>
                        <span style="color: #F8FAFC; font-weight: 700; font-size: 1.05rem; margin-left: 8px;">
                            {r.get('title', 'Resource')}
                        </span>
                        <div style="margin-top: 8px;">
                            <span class="resource-type-badge {badge_class}">{rtype.upper()}</span>
                            <span style="color: {diff_color}; font-size: 0.8rem; margin-left: 12px;
                                font-weight: 600;">{difficulty.title()}</span>
                            <span style="color: #94A3B8; font-size: 0.8rem; margin-left: 12px;">
                                ⏱️ {duration} min</span>
                        </div>
                        <p style="color: #94A3B8; font-size: 0.85rem; margin: 8px 0 0 0;">
                            {r.get('description', '')}
                        </p>
                        <div style="margin-top: 8px;">
                            <span style="color: #64748B; font-size: 0.75rem;">
                                Topic: {format_module_name(r.get('topic', r.get('concept', 'General')))} |
                                Source: {source.replace('_', ' ').title()}
                            </span>
                        </div>
                    </div>
                    <div style="margin-left: 16px; margin-top: 8px;">
                        <a href="{r.get('url', '#')}" target="_blank" style="display: inline-block; padding: 8px 16px; background: linear-gradient(135deg, #6366F1, #8B5CF6); color: #F8FAFC; text-decoration: none; border-radius: 6px; font-weight: 600; font-size: 0.85rem; transition: transform 0.2s, box-shadow 0.2s; box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.4);">
                            Study Now
                        </a>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_all:
        render_resource_cards(recommendations[:top_k])

    with tab_video:
        videos = [r for r in recommendations if r.get("type") == "video"]
        render_resource_cards(videos[:top_k])

    with tab_pdf:
        pdfs = [r for r in recommendations if r.get("type") == "pdf"]
        render_resource_cards(pdfs[:top_k])

    with tab_practice:
        practice = [r for r in recommendations if r.get("type") == "practice"]
        render_resource_cards(practice[:top_k])

    # Past recommendations from memory
    if agent.memory:
        past_recs = agent.memory.recommendations.get_past_recommendations(student_id, last_n=5)
        if past_recs:
            with st.expander("📜 Past Recommendations", expanded=False):
                for rec in reversed(past_recs):
                    st.markdown(f"""
                    <div style="padding: 8px 16px; background: var(--bg-secondary, #1E293B);
                        border-radius: 8px; margin: 4px 0; border-left: 3px solid #6366F1;">
                        <span style="color: #94A3B8; font-size: 0.75rem;">{rec.get('timestamp', '')[:16]}</span>
                        <span style="color: #F8FAFC; margin-left: 8px;">{format_module_name(rec.get('concept', ''))}</span>
                        <span style="color: #94A3B8; margin-left: 8px;">
                            — {len(rec.get('resources', []))} resources
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
