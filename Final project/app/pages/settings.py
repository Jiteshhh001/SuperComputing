"""
Settings Page — Configuration panel for the LearnFlow AI system.
Manages API keys, model hyperparameters, UI preferences, memory, and system info.
"""
import streamlit as st
import json
import os
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PROJECT_ROOT as CFG_ROOT,
    OPENROUTER_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    DKT_CONFIG, XGBOOST_CONFIG, RECOMMENDER_CONFIG,
    MASTERY_THRESHOLD, AT_RISK_THRESHOLD,
    MASTERY_LEVELS, MODELS_DIR, PROCESSED_DIR, RAW_DIR,
)


def render():
    st.markdown('<div class="section-header">⚙️ Settings</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: #94A3B8;">'
        'Configure the LearnFlow AI system — API keys, model parameters, memory management, and more.'
        '</p>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "🔑 API & LLM",
        "🧬 Model Parameters",
        "🧠 Memory Management",
        "📊 System Info",
        "🎨 Appearance",
    ])

    # ═════════════════════════════════════════════════════════════════════
    #  Tab 1: API & LLM Configuration
    # ═════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("### 🔑 OpenRouter API Configuration")

        # Current status
        api_status = "✅ Connected" if OPENROUTER_API_KEY else "❌ Not configured"
        api_color = "#22C55E" if OPENROUTER_API_KEY else "#EF4444"
        masked_key = ""
        if OPENROUTER_API_KEY:
            masked_key = OPENROUTER_API_KEY[:8] + "•" * 24 + OPENROUTER_API_KEY[-4:]

        st.markdown(f"""
        <div class="glass-container">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #F8FAFC; font-weight: 700;">API Status</span>
                    <span style="color: {api_color}; margin-left: 12px; font-weight: 600;">{api_status}</span>
                </div>
                <span style="color: #64748B; font-size: 0.8rem; font-family: monospace;">
                    {masked_key if masked_key else 'No key set'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # API key input
        with st.expander("🔧 Update API Key", expanded=not bool(OPENROUTER_API_KEY)):
            new_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                placeholder="sk-or-v1-...",
                help="Get a free key at https://openrouter.ai/keys",
                key="api_key_input",
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save API Key", key="save_api_key", use_container_width=True):
                    if new_key and new_key.strip():
                        env_path = CFG_ROOT / ".env"
                        _update_env_file(env_path, "OPENROUTER_API_KEY", new_key.strip())
                        st.success("✅ API key saved to `.env`. Restart the app to apply.")
                    else:
                        st.warning("Please enter a valid API key.")
            with col2:
                if st.button("🔍 Test Connection", key="test_api", use_container_width=True):
                    _test_api_connection(new_key or OPENROUTER_API_KEY)

        # LLM Model selector
        st.markdown("### 🤖 LLM Model")

        free_models = [
            "meta-llama/llama-4-scout:free",
            "meta-llama/llama-4-maverick:free",
            "google/gemini-2.5-flash-preview:free",
            "deepseek/deepseek-chat-v3-0324:free",
            "qwen/qwen3-235b-a22b:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
        ]

        current_model = LLM_MODEL
        if current_model not in free_models:
            free_models.insert(0, current_model)

        selected_model = st.selectbox(
            "Select LLM Model",
            free_models,
            index=free_models.index(current_model) if current_model in free_models else 0,
            key="model_select",
        )

        col1, col2 = st.columns(2)
        with col1:
            new_temp = st.slider(
                "Temperature", 0.0, 1.0, LLM_TEMPERATURE, 0.05,
                help="Lower = more focused, higher = more creative",
                key="temperature_slider",
            )
        with col2:
            new_max_tokens = st.number_input(
                "Max Tokens", 256, 4096, LLM_MAX_TOKENS, 128,
                help="Maximum response length",
                key="max_tokens_input",
            )

        if st.button("💾 Save LLM Settings", key="save_llm", use_container_width=True):
            env_path = CFG_ROOT / ".env"
            _update_env_file(env_path, "LLM_MODEL", selected_model)
            st.success(f"✅ Model set to `{selected_model}`. Restart the app to apply.")

        # Model info card
        st.markdown(f"""
        <div class="resource-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #F8FAFC; font-weight: 700;">Active Model</span>
                    <span class="resource-type-badge badge-video" style="margin-left: 8px;">OpenRouter</span>
                </div>
                <span style="color: #06B6D4; font-weight: 600; font-size: 0.85rem;">Free Tier</span>
            </div>
            <div style="margin-top: 8px; color: #94A3B8; font-size: 0.85rem;">
                Model: <code style="color: #A78BFA;">{current_model}</code> |
                Temp: {LLM_TEMPERATURE} | Max Tokens: {LLM_MAX_TOKENS}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════
    #  Tab 2: Model Parameters
    # ═════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("### 🧬 Model Hyperparameters")
        st.markdown(
            '<p style="color: #94A3B8; font-size: 0.85rem;">'
            'These parameters are set in <code>config.py</code>. '
            'Changes here are informational — modify <code>config.py</code> and re-train to apply.'
            '</p>',
            unsafe_allow_html=True,
        )

        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["🧬 DKT (LSTM)", "🎯 XGBoost", "📚 Recommender"])

        with sub_tab1:
            st.markdown("#### Deep Knowledge Tracing — LSTM Configuration")
            dkt_df_data = {
                "Parameter": list(DKT_CONFIG.keys()),
                "Value": [str(v) for v in DKT_CONFIG.values()],
            }
            st.dataframe(dkt_df_data, use_container_width=True, hide_index=True)

            st.markdown(f"""
            <div class="glass-container">
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
                    <div style="text-align: center;">
                        <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">Architecture</div>
                        <div style="color: #F8FAFC; font-weight: 700; font-size: 1.1rem;">
                            {DKT_CONFIG['num_layers']}-Layer LSTM
                        </div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">Hidden Size</div>
                        <div style="color: #F8FAFC; font-weight: 700; font-size: 1.1rem;">
                            {DKT_CONFIG['hidden_size']}
                        </div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #94A3B8; font-size: 0.75rem; text-transform: uppercase;">Embedding</div>
                        <div style="color: #F8FAFC; font-weight: 700; font-size: 1.1rem;">
                            {DKT_CONFIG['embedding_dim']}-dim
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with sub_tab2:
            st.markdown("#### XGBoost Gap Detector Configuration")
            xgb_df_data = {
                "Parameter": list(XGBOOST_CONFIG.keys()),
                "Value": [str(v) for v in XGBOOST_CONFIG.values()],
            }
            st.dataframe(xgb_df_data, use_container_width=True, hide_index=True)

        with sub_tab3:
            st.markdown("#### Hybrid Recommender Configuration")
            rec_df_data = {
                "Parameter": list(RECOMMENDER_CONFIG.keys()),
                "Value": [str(v) for v in RECOMMENDER_CONFIG.values()],
            }
            st.dataframe(rec_df_data, use_container_width=True, hide_index=True)

            # Hybrid weights visualization
            cf_w = RECOMMENDER_CONFIG["cf_weight"]
            cbf_w = RECOMMENDER_CONFIG["cbf_weight"]
            llm_w = RECOMMENDER_CONFIG["llm_weight"]

            st.markdown(f"""
            <div class="glass-container">
                <div style="color: #F8FAFC; font-weight: 700; margin-bottom: 12px;">Hybrid Weights</div>
                <div style="margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: #6366F1; font-weight: 600;">Collaborative Filtering</span>
                        <span style="color: #F8FAFC;">{cf_w:.0%}</span>
                    </div>
                    <div style="width: 100%; height: 8px; background: #334155; border-radius: 4px;">
                        <div style="width: {cf_w*100}%; height: 100%; background: #6366F1; border-radius: 4px;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: #8B5CF6; font-weight: 600;">Content-Based Filtering</span>
                        <span style="color: #F8FAFC;">{cbf_w:.0%}</span>
                    </div>
                    <div style="width: 100%; height: 8px; background: #334155; border-radius: 4px;">
                        <div style="width: {cbf_w*100}%; height: 100%; background: #8B5CF6; border-radius: 4px;"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: #06B6D4; font-weight: 600;">LLM Ranking</span>
                        <span style="color: #F8FAFC;">{llm_w:.0%}</span>
                    </div>
                    <div style="width: 100%; height: 8px; background: #334155; border-radius: 4px;">
                        <div style="width: {llm_w*100}%; height: 100%; background: #06B6D4; border-radius: 4px;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Thresholds
        st.markdown("### 📏 Decision Thresholds")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mastery Threshold</div>
                <div class="metric-value">{MASTERY_THRESHOLD}</div>
                <div style="color: #94A3B8; font-size: 0.8rem;">Below this → weak concept</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">At-Risk Threshold</div>
                <div class="metric-value">{AT_RISK_THRESHOLD}</div>
                <div style="color: #94A3B8; font-size: 0.8rem;">Below this → at-risk student</div>
            </div>
            """, unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════
    #  Tab 3: Memory Management
    # ═════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("### 🧠 Agent Memory Management")

        agent = st.session_state.get("agent")
        student_id = st.session_state.get("selected_student", 10000)

        if not agent:
            st.warning("Agent not initialized. Run the pipeline first.")
        else:
            # Memory layer status
            memory_dir = CFG_ROOT / "data" / "memory"
            profiles_path = memory_dir / "student_profiles.json"
            recs_path = memory_dir / "recommendations.json"
            history_path = memory_dir / "learning_history.json"

            layers = [
                ("💬 Conversation Memory", "In-memory (session)", True, "Active"),
                ("👤 Student Profiles", str(profiles_path.name), profiles_path.exists(),
                 f"{profiles_path.stat().st_size / 1024:.1f} KB" if profiles_path.exists() else "Empty"),
                ("📚 Recommendations", str(recs_path.name), recs_path.exists(),
                 f"{recs_path.stat().st_size / 1024:.1f} KB" if recs_path.exists() else "Empty"),
                ("📜 Learning History", str(history_path.name), history_path.exists(),
                 f"{history_path.stat().st_size / 1024:.1f} KB" if history_path.exists() else "Empty"),
            ]

            for name, storage, exists, size in layers:
                status_color = "#22C55E" if exists else "#94A3B8"
                st.markdown(f"""
                <div class="resource-card" style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: #F8FAFC; font-weight: 700;">{name}</span>
                        <span style="color: #94A3B8; font-size: 0.8rem; margin-left: 12px;">
                            {storage}
                        </span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="color: #94A3B8; font-size: 0.85rem;">{size}</span>
                        <span style="color: {status_color}; font-size: 0.9rem;">●</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Actions
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🗑️ Clear Chat Memory", key="clear_conv_mem", use_container_width=True):
                    agent.memory.conversation.clear(student_id)
                    st.session_state.chat_history = []
                    st.success(f"✅ Conversation memory cleared for Student {student_id}")
            with col2:
                if st.button("🗑️ Clear All Memory", key="clear_all_mem", use_container_width=True):
                    agent.memory.clear_student(student_id)
                    st.session_state.chat_history = []
                    st.session_state.analysis_cache.pop(student_id, None)
                    st.success(f"✅ All memory cleared for Student {student_id}")
            with col3:
                if st.button("📋 Export Memory", key="export_mem", use_container_width=True):
                    context = agent.memory.get_full_context(student_id)
                    st.text_area("Exported Memory Context", context, height=300, key="exported_mem")

    # ═════════════════════════════════════════════════════════════════════
    #  Tab 4: System Info
    # ═════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("### 📊 System Information")

        # Disk usage
        st.markdown("#### 📁 Storage")
        dirs_to_check = [
            ("Raw Data", RAW_DIR),
            ("Processed Data", PROCESSED_DIR),
            ("Trained Models", MODELS_DIR),
            ("Memory Store", CFG_ROOT / "data" / "memory"),
        ]

        for label, dir_path in dirs_to_check:
            if dir_path.exists():
                total_size = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())
                file_count = sum(1 for f in dir_path.rglob("*") if f.is_file())
                size_str = f"{total_size / 1024 / 1024:.2f} MB" if total_size > 1024 * 1024 else f"{total_size / 1024:.1f} KB"
            else:
                size_str = "Not found"
                file_count = 0

            st.markdown(f"""
            <div class="resource-card" style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #F8FAFC; font-weight: 600;">{label}</span>
                    <span style="color: #64748B; font-size: 0.8rem; margin-left: 12px;">
                        {dir_path.relative_to(CFG_ROOT)}
                    </span>
                </div>
                <div style="display: flex; gap: 20px;">
                    <span style="color: #94A3B8; font-size: 0.85rem;">{file_count} files</span>
                    <span style="color: #06B6D4; font-weight: 600;">{size_str}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Model files
        st.markdown("#### 🧬 Trained Model Files")
        if MODELS_DIR.exists():
            model_files = list(MODELS_DIR.glob("*"))
            if model_files:
                for mf in model_files:
                    if mf.is_file():
                        size = mf.stat().st_size
                        size_str = f"{size / 1024 / 1024:.2f} MB" if size > 1024 * 1024 else f"{size / 1024:.1f} KB"
                        modified = datetime.fromtimestamp(mf.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                        ext = mf.suffix
                        icon = {".pt": "🧬", ".pkl": "📦", ".json": "📄"}.get(ext, "📎")

                        st.markdown(f"""
                        <div style="padding: 8px 16px; border-left: 3px solid #6366F1;
                            background: rgba(30, 41, 59, 0.5); border-radius: 4px; margin: 4px 0;
                            display: flex; justify-content: space-between; align-items: center;">
                            <span>
                                {icon}
                                <span style="color: #F8FAFC; margin-left: 8px;">{mf.name}</span>
                            </span>
                            <span style="color: #94A3B8; font-size: 0.8rem;">
                                {size_str} — {modified}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No trained models found. Run `python run_pipeline.py` to train models.")
        else:
            st.info("Models directory not found.")

        # Python environment
        st.markdown("#### 🐍 Runtime")
        import platform
        try:
            import torch
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            cuda_info = f"CUDA {torch.version.cuda}" if cuda_available else "CPU only"
        except ImportError:
            torch_version = "Not installed"
            cuda_info = "N/A"

        try:
            import pandas
            pd_version = pandas.__version__
        except ImportError:
            pd_version = "Not installed"

        env_items = [
            ("Python", platform.python_version()),
            ("Platform", f"{platform.system()} {platform.release()}"),
            ("PyTorch", torch_version),
            ("Compute", cuda_info),
            ("Pandas", pd_version),
        ]
        for label, value in env_items:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 6px 0;
                border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                <span style="color: #94A3B8;">{label}</span>
                <span style="color: #F8FAFC; font-weight: 600;">{value}</span>
            </div>
            """, unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════
    #  Tab 5: Appearance
    # ═════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("### 🎨 Appearance & Preferences")

        st.markdown("""
        <div class="glass-container">
            <div style="color: #F8FAFC; font-weight: 700; margin-bottom: 12px;">🎨 Current Theme</div>
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px;">
                <div style="text-align: center;">
                    <div style="width: 40px; height: 40px; border-radius: 8px;
                        background: #6366F1; margin: 0 auto 6px auto;"></div>
                    <span style="color: #94A3B8; font-size: 0.7rem;">Primary</span>
                </div>
                <div style="text-align: center;">
                    <div style="width: 40px; height: 40px; border-radius: 8px;
                        background: #8B5CF6; margin: 0 auto 6px auto;"></div>
                    <span style="color: #94A3B8; font-size: 0.7rem;">Secondary</span>
                </div>
                <div style="text-align: center;">
                    <div style="width: 40px; height: 40px; border-radius: 8px;
                        background: #06B6D4; margin: 0 auto 6px auto;"></div>
                    <span style="color: #94A3B8; font-size: 0.7rem;">Accent</span>
                </div>
                <div style="text-align: center;">
                    <div style="width: 40px; height: 40px; border-radius: 8px;
                        background: #22C55E; margin: 0 auto 6px auto;"></div>
                    <span style="color: #94A3B8; font-size: 0.7rem;">Success</span>
                </div>
                <div style="text-align: center;">
                    <div style="width: 40px; height: 40px; border-radius: 8px;
                        background: #EF4444; margin: 0 auto 6px auto;"></div>
                    <span style="color: #94A3B8; font-size: 0.7rem;">Danger</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Mastery level legend
        st.markdown("#### 📊 Mastery Level Legend")
        for level, info in MASTERY_LEVELS.items():
            low, high = info["range"]
            color = info["color"]
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 6px 0; padding: 10px 16px;
                background: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1);
                border-radius: 8px; border-left: 4px solid {color};">
                <span style="color: {color}; font-weight: 700; width: 70px;">Level {level}</span>
                <span style="color: #F8FAFC; font-weight: 600; width: 120px;">{info['label']}</span>
                <span style="color: #94A3B8;">Range: {low:.1f} – {high:.1f}</span>
                <div style="margin-left: auto; width: 24px; height: 24px; border-radius: 4px;
                    background: {color};"></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-container" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 8px;">🧠</div>
            <div style="color: #F8FAFC; font-weight: 700; font-size: 1.1rem;">LearnFlow AI v1.0</div>
            <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
                Powered by DKT (LSTM) + XGBoost + Hybrid Recommender + OpenRouter LLM
            </div>
            <div style="color: #64748B; font-size: 0.75rem; margin-top: 8px;">
                Built with ❤️ for personalized education
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────────

def _update_env_file(env_path: Path, key: str, value: str):
    """Update or insert a key-value pair in the .env file."""
    lines = []
    found = False

    if env_path.exists():
        with open(env_path, "r") as f:
            lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"{key}={value}\n")

    with open(env_path, "w") as f:
        f.writelines(new_lines)


def _test_api_connection(api_key: str):
    """Test the OpenRouter API connection."""
    if not api_key:
        st.error("❌ No API key provided.")
        return

    try:
        import httpx
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            },
            timeout=15,
        )
        if response.status_code == 200:
            st.success("✅ API connection successful! Model is responding.")
        else:
            st.error(f"❌ API returned status {response.status_code}: {response.text[:200]}")
    except Exception as e:
        st.error(f"❌ Connection failed: {str(e)}")
