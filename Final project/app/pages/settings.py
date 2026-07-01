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
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    DKT_CONFIG, XGBOOST_CONFIG,
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
        agent = st.session_state.get("agent")
        llm = agent.llm if agent else None
        configured = bool(llm and llm.available)

        if configured:
            st.success("✅ AI model enabled — chat answers and recommendation ranking use OpenRouter.")
        else:
            st.warning("⚠️ No OpenRouter key configured — the app runs in offline mode "
                       "(data-driven chat answers and heuristic resource ranking).")

        with st.expander("🔧 Use your own OpenRouter key for this session", expanded=not configured):
            st.caption("The key is kept only in this browser session's memory — it is never written "
                       "to disk or shared with other visitors. Get a free key at https://openrouter.ai/keys")
            new_key = st.text_input("OpenRouter API Key", type="password",
                                    placeholder="sk-or-v1-...", key="api_key_input")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Use this key", key="save_api_key", use_container_width=True):
                    if llm and new_key.strip():
                        llm.set_api_key(new_key)
                        st.success("Key applied for this session.")
                    else:
                        st.warning("Enter a key first.")
            with col2:
                if st.button("🔍 Test connection", key="test_api", use_container_width=True):
                    _test_api_connection(llm)

        st.markdown(
            "To enable the AI model for everyone, set `OPENROUTER_API_KEY` in a local `.env` file, "
            "or in **Streamlit Community Cloud → App settings → Secrets** as "
            "`OPENROUTER_API_KEY = \"sk-or-v1-...\"`.")

        st.markdown("### 🤖 Model fallback chain")
        st.caption("Free models are often rate-limited or retired, so each request tries these in order "
                   "and remembers the last one that answered.")
        models = llm.models if llm else [LLM_MODEL]
        last = llm.last_model if llm else None
        st.dataframe(
            {"Order": list(range(1, len(models) + 1)), "Model": models,
             "Status": ["✅ last used" if m == last else "" for m in models]},
            use_container_width=True, hide_index=True,
        )
        st.caption(f"Temperature {LLM_TEMPERATURE} · max tokens {LLM_MAX_TOKENS}"
                   + (f" · last error: {llm.last_error}" if llm and llm.last_error else ""))

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
            st.markdown("#### Recommender")
            from src.models.recommender import CATALOG
            st.markdown(
                f"- **Candidates:** {len(CATALOG)} curated, link-checked resources grouped by OULAD domain "
                "(Social Sciences / STEM) and assessment type (TMA / CMA / Exam)\n"
                "- **Ranking:** the LLM picks and explains the best matches for the student's at-risk concepts; "
                "without an API key a severity-aware heuristic ranks them\n"
                "- **Safety:** the LLM can only choose from the catalog, so every link is real")
            st.dataframe(
                {"Resource": [r["title"] for r in CATALOG], "Domain": [r["domain"] for r in CATALOG],
                 "Type": [r["type"] for r in CATALOG], "Level": [r["difficulty"] for r in CATALOG]},
                use_container_width=True, hide_index=True)

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
                    st.session_state.setdefault("chat_histories", {}).pop(student_id, None)
                    st.success(f"✅ Conversation memory cleared for Student {student_id}")
            with col2:
                if st.button("🗑️ Clear All Memory", key="clear_all_mem", use_container_width=True):
                    agent.memory.clear_student(student_id)
                    st.session_state.setdefault("chat_histories", {}).pop(student_id, None)
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
                Powered by DKT (LSTM) + XGBoost + LLM Recommender (OpenRouter) + offline fallback catalog
            </div>
            <div style="color: #64748B; font-size: 0.75rem; margin-top: 8px;">
                Built with ❤️ for personalized education
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────────

def _test_api_connection(llm):
    """Send a tiny request through the model fallback chain."""
    if not llm or not llm.available:
        st.error("❌ No API key configured.")
        return
    with st.spinner("Contacting OpenRouter…"):
        reply = llm.complete([{"role": "user", "content": "Reply with the single word OK."}],
                             max_tokens=200, time_budget=40)
    if reply:
        st.success(f"✅ Connected — `{llm.last_model}` replied: {reply[:60]}")
    else:
        st.error(f"❌ No model answered ({llm.last_error}).")
