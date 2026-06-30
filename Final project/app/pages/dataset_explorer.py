"""
Dataset Explorer Page — Browse and inspect raw datasets interactively.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import RAW_DIR


def render():
    st.markdown('<div class="section-header">📊 Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8;">Browse and inspect the OULAD and UCI educational datasets.</p>',
                unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎓 OULAD Dataset", "📚 UCI Student Performance"])

    with tab1:
        oulad_dir = RAW_DIR / "oulad"
        csv_files = sorted(oulad_dir.glob("*.csv")) if oulad_dir.exists() else []

        if csv_files:
            selected_table = st.selectbox(
                "Select Table",
                [f.stem for f in csv_files],
                key="oulad_table_select",
            )

            df = pd.read_csv(oulad_dir / f"{selected_table}.csv")

            # Table info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                missing_pct = df.isnull().mean().mean() * 100
                st.metric("Missing %", f"{missing_pct:.1f}%")
            with col4:
                mem = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("Memory", f"{mem:.1f} MB")

            # Schema
            with st.expander("📋 Column Schema", expanded=False):
                schema = pd.DataFrame({
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str).values,
                    "Non-Null": df.notna().sum().values,
                    "Unique": df.nunique().values,
                    "Sample": [str(df[c].iloc[0])[:50] if len(df) > 0 else "N/A" for c in df.columns],
                })
                st.dataframe(schema, use_container_width=True, hide_index=True)

            # Data preview
            st.markdown("**Data Preview**")
            n_rows = st.slider("Rows to display", 5, 100, 20, key="oulad_rows")
            st.dataframe(df.head(n_rows), use_container_width=True, height=400)

            # Quick stats for numeric columns
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                with st.expander("📊 Descriptive Statistics"):
                    st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
        else:
            st.info("OULAD dataset not found. Run the pipeline to download datasets.")

    with tab2:
        uci_dir = RAW_DIR / "uci"
        uci_files = sorted(uci_dir.glob("*.csv")) if uci_dir.exists() else []

        if uci_files:
            selected_file = st.selectbox(
                "Select Dataset",
                [f.name for f in uci_files if f.suffix == ".csv"],
                key="uci_file_select",
            )

            df = pd.read_csv(uci_dir / selected_file, sep=";")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                missing_pct = df.isnull().mean().mean() * 100
                st.metric("Missing %", f"{missing_pct:.1f}%")
            with col4:
                mem = df.memory_usage(deep=True).sum() / 1024
                st.metric("Memory", f"{mem:.0f} KB")

            st.dataframe(df.head(20), use_container_width=True, height=400)

            with st.expander("📊 Descriptive Statistics"):
                st.dataframe(df.describe().round(2), use_container_width=True)
        else:
            st.info("UCI dataset not found. Run the pipeline to download datasets.")
