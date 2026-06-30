"""
═══════════════════════════════════════════════════════════════════════════════
  LearnFlow AI — Streamlit Dashboard
  Main entry point. Multi-page app with sidebar navigation.
  Run: streamlit run app/streamlit_app.py
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from config import STREAMLIT_PAGE_TITLE, STREAMLIT_PAGE_ICON, STREAMLIT_LAYOUT

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout=STREAMLIT_LAYOUT,
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Fonts ─── */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Root Variables ─── */
    :root {
        --primary: #8B5CF6;
        --primary-light: #A78BFA;
        --secondary: #3B82F6;
        --accent: #F43F5E;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        
        /* Deep rich dark backgrounds */
        --bg-base: #0B0E14;
        --bg-surface: rgba(22, 27, 34, 0.6);
        --bg-glass: rgba(30, 41, 59, 0.4);
        --bg-glass-hover: rgba(45, 55, 72, 0.5);
        
        /* Typography */
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        
        /* Borders & Effects */
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-glass-hover: rgba(255, 255, 255, 0.15);
        --shadow-glass: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        --glow: 0 0 20px rgba(139, 92, 246, 0.3);
    }

    /* ── Global ─── */
    .stApp {
        background-color: var(--bg-base);
        background-image: 
            radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(244, 63, 94, 0.1) 0px, transparent 50%);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
        background-attachment: fixed;
    }
    
    h1, h2, h3, h4, h5, h6, .section-header {
        font-family: 'Outfit', sans-serif !important;
    }

    /* ── Sidebar ─── */
    section[data-testid="stSidebar"] {
        background: rgba(11, 14, 20, 0.7);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-right: 1px solid var(--border-glass);
    }

    section[data-testid="stSidebar"] .stRadio label {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: var(--text-secondary) !important;
        padding: 10px 16px;
        border-radius: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 6px;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255, 255, 255, 0.05);
        color: var(--text-primary) !important;
        transform: translateX(4px);
    }

    section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white !important;
        box-shadow: var(--glow);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* ── Cards ─── */
    .metric-card {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: var(--shadow-glass);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 50%;
        height: 100%;
        background: linear-gradient(to right, transparent, rgba(255,255,255,0.03), transparent);
        transform: skewX(-20deg);
        transition: 0.7s;
    }

    .metric-card:hover::before {
        left: 200%;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        border-color: var(--border-glass-hover);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5), var(--glow);
        background: var(--bg-glass-hover);
    }

    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
        letter-spacing: -0.5px;
    }

    .metric-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }

    .metric-delta {
        font-size: 0.95rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 20px;
        background: rgba(255,255,255,0.05);
    }

    .delta-positive { color: var(--success); }
    .delta-negative { color: var(--danger); }

    /* ── Section Headers ─── */
    .section-header {
        font-size: 1.7rem;
        font-weight: 700;
        color: white;
        margin: 32px 0 20px 0;
        padding-bottom: 12px;
        position: relative;
        display: inline-block;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60%;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), transparent);
        border-radius: 2px;
    }

    /* ── Glass Containers ─── */
    .glass-container {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: 20px;
        padding: 28px;
        margin: 16px 0;
        box-shadow: var(--shadow-glass);
    }

    /* ── Resource Cards ─── */
    .resource-card {
        background: rgba(22, 27, 34, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }

    .resource-card:hover {
        background: rgba(30, 41, 59, 0.6);
        border-color: rgba(139, 92, 246, 0.4);
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    .resource-type-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .badge-video { background: rgba(244, 63, 94, 0.15); color: #FDA4AF; border: 1px solid rgba(244, 63, 94, 0.3); }
    .badge-pdf { background: rgba(16, 185, 129, 0.15); color: #6EE7B7; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-practice { background: rgba(59, 130, 246, 0.15); color: #93C5FD; border: 1px solid rgba(59, 130, 246, 0.3); }

    /* ── Chat Messages ─── */
    .chat-user {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border-radius: 18px 18px 4px 18px;
        padding: 16px 22px;
        margin: 12px 0;
        color: white;
        max-width: 85%;
        margin-left: auto;
        box-shadow: var(--glow);
    }

    .chat-assistant {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-glass);
        border-radius: 18px 18px 18px 4px;
        padding: 16px 22px;
        margin: 12px 0;
        color: var(--text-primary);
        max-width: 85%;
        box-shadow: var(--shadow-glass);
    }

    /* ── Study Plan Day Card ─── */
    .day-card {
        background: var(--bg-glass);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        border-left: 4px solid var(--primary);
        transition: all 0.3s ease;
    }

    .day-card:hover {
        background: var(--bg-glass-hover);
        border-left-color: var(--secondary);
        transform: translateX(6px);
        box-shadow: var(--shadow-glass);
    }

    /* ── Buttons ─── */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
    }

    /* ── Progress Bar ─── */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
        border-radius: 10px;
    }

    /* ── Tabs ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--bg-secondary);
        border-radius: 10px;
        border: 1px solid var(--border);
        color: var(--text-secondary);
        font-weight: 500;
        padding: 8px 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        border: none !important;
    }

    /* ── Selectbox ─── */
    .stSelectbox > div > div {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }

    /* ── Hide Streamlit branding (keep sidebar toggle visible) ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Hide the deploy button only */
    .stAppDeployButton {display: none !important;}
    /* Make header background transparent but keep toggle functional */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* ── Logo ─── */
    .logo-container {
        text-align: center;
        padding: 20px 0;
    }

    .logo-text {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .logo-subtitle {
        font-size: 0.85rem;
        color: var(--text-muted);
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* ── Expander ─── */
    .streamlit-expanderHeader {
        background: var(--bg-secondary) !important;
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
    }

    /* ── Animations ─── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .pulse {
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    .shimmer {
        background: linear-gradient(90deg,
            var(--bg-secondary) 25%,
            var(--bg-tertiary) 50%,
            var(--bg-secondary) 75%
        );
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ── JavaScript: Force sidebar open + ensure toggle works ─────────────
import streamlit.components.v1 as components
components.html("""
<script>
(function() {
    var doc = window.parent.document;
    function expandSidebar() {
        var sidebar = doc.querySelector('section[data-testid="stSidebar"]');
        if (sidebar && sidebar.getAttribute('aria-expanded') === 'false') {
            var btn = doc.querySelector('[data-testid="collapsedControl"] button');
            if (btn) { btn.click(); return true; }
        }
        if (sidebar) {
            var style = window.parent.getComputedStyle(sidebar);
            if (style.width === '0px' || (style.transform && style.transform.includes('-'))) {
                var btn = doc.querySelector('[data-testid="collapsedControl"] button');
                if (btn) { btn.click(); return true; }
            }
        }
        return false;
    }
    if (!expandSidebar()) {
        var attempts = 0;
        var interval = setInterval(function() {
            if (expandSidebar() || attempts > 15) clearInterval(interval);
            attempts++;
        }, 300);
    }
})();
</script>
""", height=0)


# ── Initialize Session State ────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"
if "selected_student" not in st.session_state:
    st.session_state.selected_student = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}


def get_agent():
    """Lazy-load the learning agent."""
    if st.session_state.agent is None:
        try:
            from src.agent.agent import LearningAgent
            st.session_state.agent = LearningAgent()
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            return None
    return st.session_state.agent


# ── Sidebar Navigation ──────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="logo-container">
        <div class="logo-text">🧠 LearnFlow AI</div>
        <div class="logo-subtitle">Personalized Learning Agent</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation
    pages = {
        "🏠 Home": "Home",
        "👤 Student Analytics": "Student Analytics",
        "📊 Dataset Explorer": "Dataset Explorer",
        "🔬 EDA": "EDA",
        "🧬 Knowledge Tracing": "Knowledge Tracing",
        "🎯 Weak Topic Analysis": "Weak Topic Analysis",
        "📚 Recommendation Center": "Recommendation Center",
        "📅 AI Study Planner": "AI Study Planner",
        "📈 Progress Analytics": "Progress Analytics",
        "💬 AI Chat": "AI Chat",
        "⚙️ Settings": "Settings",
    }

    # Determine index from current_page for sync with quick-action buttons
    page_values = list(pages.values())
    current = st.session_state.get("current_page", "Home")
    default_idx = page_values.index(current) if current in page_values else 0

    selected = st.radio(
        "Navigation",
        list(pages.keys()),
        index=default_idx,
        label_visibility="collapsed",
    )
    st.session_state.current_page = pages[selected]

    st.markdown("---")

    # Student selector
    st.markdown("**🎓 Active Student**")
    agent = get_agent()
    if agent:
        student_ids = agent.get_student_ids()
        if student_ids:
            selected_student = st.selectbox(
                "Select Student",
                student_ids[:50],
                key="student_selector",
                label_visibility="collapsed",
                format_func=lambda x: f"Student #{x}"
            )
            st.session_state.selected_student = selected_student
        else:
            st.info("Run pipeline first to load student data")
            st.session_state.selected_student = 10000  # Default

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #94A3B8; font-size: 0.75rem;'>"
        "LearnFlow AI v1.0<br>Powered by DKT + XGBoost + LLM"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Page Router ──────────────────────────────────────────────────────────
page = st.session_state.current_page

if page == "Home":
    from app.pages.home import render
    render()
elif page == "Student Analytics":
    from app.pages.student_analytics import render
    render()
elif page == "Dataset Explorer":
    from app.pages.dataset_explorer import render
    render()
elif page == "EDA":
    from app.pages.eda import render
    render()
elif page == "Knowledge Tracing":
    from app.pages.knowledge_tracing import render
    render()
elif page == "Weak Topic Analysis":
    from app.pages.weak_topic_analysis import render
    render()
elif page == "Recommendation Center":
    from app.pages.recommendation_center import render
    render()
elif page == "AI Study Planner":
    from app.pages.ai_study_planner import render
    render()
elif page == "Progress Analytics":
    from app.pages.progress_analytics import render
    render()
elif page == "AI Chat":
    from app.pages.ai_chat import render
    render()
elif page == "Settings":
    from app.pages.settings import render
    render()
