"""
EDA Page — Interactive exploratory data analysis with 8 analysis tabs.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import RAW_DIR, PROCESSED_DIR


def render():
    st.markdown('<div class="section-header">🔬 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8;">Comprehensive 8-step EDA covering all aspects of the educational datasets.</p>',
                unsafe_allow_html=True)

    tabs = st.tabs([
        "1️⃣ Load & Audit", "2️⃣ Missing Values", "3️⃣ Distributions",
        "4️⃣ Time-Series", "5️⃣ Categorical", "6️⃣ Correlation",
        "7️⃣ Class Imbalance", "8️⃣ Target Engineering"
    ])

    # Load data
    oulad_dir = RAW_DIR / "oulad"

    @st.cache_data
    def load_oulad_table(name):
        path = oulad_dir / f"{name}.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    student_info = load_oulad_table("studentInfo")
    student_assessment = load_oulad_table("studentAssessment")
    student_vle = load_oulad_table("studentVle")
    assessments = load_oulad_table("assessments")

    # ── Tab 1: Load & Audit ──
    with tabs[0]:
        st.markdown("### 📋 Dataset Audit Summary")

        tables = {
            "studentInfo": student_info, "studentAssessment": student_assessment,
            "studentVle": student_vle, "assessments": assessments,
            "courses": load_oulad_table("courses"), "vle": load_oulad_table("vle"),
            "studentRegistration": load_oulad_table("studentRegistration"),
        }

        audit_rows = []
        for name, df in tables.items():
            if not df.empty:
                audit_rows.append({
                    "Table": name, "Rows": f"{len(df):,}", "Columns": len(df.columns),
                    "Missing %": f"{df.isnull().mean().mean():.2%}",
                    "Duplicates": df.duplicated().sum(),
                    "Memory (MB)": f"{df.memory_usage(deep=True).sum()/1024/1024:.2f}",
                })

        if audit_rows:
            st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

        # Data types
        if not student_info.empty:
            with st.expander("📊 studentInfo — Data Types"):
                dtypes = pd.DataFrame({
                    "Column": student_info.columns,
                    "Dtype": student_info.dtypes.astype(str).values,
                    "Non-Null": student_info.notna().sum().values,
                    "Unique": student_info.nunique().values,
                })
                st.dataframe(dtypes, use_container_width=True, hide_index=True)

    # ── Tab 2: Missing Values ──
    with tabs[1]:
        st.markdown("### 🔍 Missing Value Analysis")

        if not student_assessment.empty:
            from src.utils.visualization import missing_value_heatmap

            selected = st.selectbox("Select table", list(tables.keys()), key="missing_table")
            df = tables.get(selected, pd.DataFrame())
            if not df.empty:
                missing_summary = pd.DataFrame({
                    "Column": df.columns,
                    "Missing": df.isnull().sum().values,
                    "Missing %": (df.isnull().mean() * 100).round(2).values,
                }).sort_values("Missing %", ascending=False)
                missing_summary = missing_summary[missing_summary["Missing"] > 0]

                if len(missing_summary) > 0:
                    st.dataframe(missing_summary, use_container_width=True, hide_index=True)
                    fig = missing_value_heatmap(df, f"Missing Values — {selected}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success(f"✅ No missing values in {selected}!")

    # ── Tab 3: Distributions ──
    with tabs[2]:
        st.markdown("### 📊 Score Distributions")

        if not student_assessment.empty and "score" in student_assessment.columns:
            from src.utils.visualization import score_distribution

            fig = score_distribution(student_assessment["score"], "Assessment Score Distribution")
            st.plotly_chart(fig, use_container_width=True)

            # Per-module distributions
            if not assessments.empty:
                merged = student_assessment.merge(assessments, on="id_assessment", how="left")
                if "code_module" in merged.columns:
                    from config import format_module_name
                    merged["code_module"] = merged["code_module"].apply(format_module_name)
                    
                    import plotly.express as px
                    fig2 = px.box(merged, x="code_module", y="score", color="code_module",
                                   title="Score Distribution by Subject")
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#F8FAFC"), height=400,
                    )
                    fig2.update_xaxes(gridcolor="rgba(148,163,184,0.1)")
                    fig2.update_yaxes(gridcolor="rgba(148,163,184,0.1)")
                    st.plotly_chart(fig2, use_container_width=True)

            # Normality check
            from scipy import stats
            stat, p_value = stats.normaltest(student_assessment["score"].dropna())
            st.markdown(f"""
            <div class="glass-container">
                <b style="color: #F8FAFC;">Normality Test (D'Agostino-Pearson):</b><br>
                <span style="color: #94A3B8;">Statistic: {stat:.4f} | p-value: {p_value:.4e}</span><br>
                <span style="color: {'#22C55E' if p_value > 0.05 else '#EF4444'};">
                    {'✅ Scores appear normally distributed' if p_value > 0.05 else '⚠️ Scores are NOT normally distributed (skewed)'}
                </span>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 4: Time-Series ──
    with tabs[3]:
        st.markdown("### 📈 Student Engagement Over Time")

        if not student_vle.empty and "date" in student_vle.columns:
            from src.utils.visualization import engagement_timeline
            fig = engagement_timeline(student_vle, "VLE Engagement Over Time (All Students)")
            st.plotly_chart(fig, use_container_width=True)

            # Active vs dropout
            if not student_info.empty and "final_result" in student_info.columns:
                active_ids = student_info[student_info["final_result"].isin(["Pass", "Distinction"])]["id_student"]
                dropout_ids = student_info[student_info["final_result"] == "Withdrawn"]["id_student"]

                active_vle = student_vle[student_vle["id_student"].isin(active_ids)]
                dropout_vle = student_vle[student_vle["id_student"].isin(dropout_ids)]

                import plotly.graph_objects as go
                fig2 = go.Figure()

                if not active_vle.empty:
                    active_daily = active_vle.groupby("date")["sum_click"].mean().reset_index()
                    fig2.add_trace(go.Scatter(x=active_daily["date"], y=active_daily["sum_click"],
                                              name="Active Students", line=dict(color="#22C55E", width=2)))

                if not dropout_vle.empty:
                    dropout_daily = dropout_vle.groupby("date")["sum_click"].mean().reset_index()
                    fig2.add_trace(go.Scatter(x=dropout_daily["date"], y=dropout_daily["sum_click"],
                                              name="Withdrawn Students", line=dict(color="#EF4444", width=2)))

                fig2.update_layout(title="Active vs Withdrawn: Avg Daily Clicks", height=400,
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#F8FAFC"))
                fig2.update_xaxes(gridcolor="rgba(148,163,184,0.1)", title="Day")
                fig2.update_yaxes(gridcolor="rgba(148,163,184,0.1)", title="Avg Clicks")
                st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 5: Categorical ──
    with tabs[4]:
        st.markdown("### 📊 Categorical Analysis")

        if not student_info.empty:
            import plotly.express as px

            cat_col = st.selectbox("Group by", ["gender", "region", "highest_education",
                                                  "age_band", "disability"], key="cat_col")

            if cat_col in student_info.columns and "final_result" in student_info.columns:
                fig = px.histogram(student_info, x=cat_col, color="final_result",
                                    barmode="group", title=f"Performance by {cat_col}")
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F8FAFC"), height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 6: Correlation ──
    with tabs[5]:
        st.markdown("### 🔗 Correlation Analysis")

        profiles_path = PROCESSED_DIR / "student_profiles.csv"
        if profiles_path.exists():
            profiles = pd.read_csv(profiles_path)
            numeric = profiles.select_dtypes(include=[np.number])
            if len(numeric.columns) > 2:
                from src.utils.visualization import correlation_heatmap
                cols = [c for c in ["avg_score", "total_clicks", "total_vle_days",
                                     "num_submissions", "studied_credits", "num_of_prev_attempts"]
                        if c in numeric.columns]
                if cols:
                    fig = correlation_heatmap(profiles, cols, "Feature Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run pipeline to generate processed profiles for correlation analysis.")

    # ── Tab 7: Class Imbalance ──
    with tabs[6]:
        st.markdown("### ⚖️ Class Imbalance Analysis")

        if not student_info.empty and "final_result" in student_info.columns:
            from src.utils.visualization import class_imbalance_chart
            fig = class_imbalance_chart(student_info["final_result"], "Student Outcome Distribution")
            st.plotly_chart(fig, use_container_width=True)

            # Numeric summary
            counts = student_info["final_result"].value_counts()
            total = len(student_info)
            for result, count in counts.items():
                pct = count / total * 100
                color = "#22C55E" if result in ["Pass", "Distinction"] else "#EF4444"
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin: 4px 0;">
                    <div style="width: {pct}%; background: {color}; height: 24px; border-radius: 4px;
                        display: flex; align-items: center; padding-left: 8px; min-width: 60px;">
                        <span style="color: white; font-size: 0.8rem; font-weight: 600;">{result}: {count:,} ({pct:.1f}%)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 8: Target Engineering ──
    with tabs[7]:
        st.markdown("### 🎯 Target Engineering — Mastery Levels")
        st.markdown("""
        <div class="glass-container">
            <p style="color: #F8FAFC;">
                Mastery levels are defined on a <b>0–4 scale</b> for DKT training.
                Each concept is assigned a mastery level based on the student's normalized score:
            </p>
        </div>
        """, unsafe_allow_html=True)

        from config import MASTERY_LEVELS
        for level, info in MASTERY_LEVELS.items():
            low, high = info["range"]
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 4px 0; padding: 8px 16px;
                background: rgba({int(info['color'][1:3], 16)}, {int(info['color'][3:5], 16)}, {int(info['color'][5:7], 16)}, 0.15);
                border-radius: 8px; border-left: 4px solid {info['color']};">
                <span style="color: {info['color']}; font-weight: 700; width: 60px;">Level {level}</span>
                <span style="color: #F8FAFC; font-weight: 600; width: 120px;">{info['label']}</span>
                <span style="color: #94A3B8;">Score range: {low:.1f} – {high:.1f}</span>
            </div>
            """, unsafe_allow_html=True)
