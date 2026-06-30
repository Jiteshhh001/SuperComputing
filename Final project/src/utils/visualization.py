"""
═══════════════════════════════════════════════════════════════════════════════
  Visualization Utilities — Plotly charts for dashboard and EDA
  All functions return Plotly figures for Streamlit rendering.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Color Palette ────────────────────────────────────────────────────────
COLORS = {
    "primary": "#6366F1",    # Indigo
    "secondary": "#8B5CF6",  # Violet
    "success": "#22C55E",    # Green
    "warning": "#F59E0B",    # Amber
    "danger": "#EF4444",     # Red
    "info": "#06B6D4",       # Cyan
    "bg": "#0F172A",         # Slate 900
    "surface": "#1E293B",    # Slate 800
    "text": "#F8FAFC",       # Slate 50
    "muted": "#94A3B8",      # Slate 400
}

MASTERY_COLORS = ["#EF4444", "#F97316", "#EAB308", "#22C55E", "#06B6D4"]
GRADIENT = ["#6366F1", "#8B5CF6", "#A78BFA", "#C4B5FD", "#DDD6FE"]


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent dark theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            bgcolor="rgba(30,41,59,0.8)",
            bordercolor="rgba(148,163,184,0.2)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.1)", zerolinecolor="rgba(148,163,184,0.1)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.1)", zerolinecolor="rgba(148,163,184,0.1)")
    return fig


def mastery_radar_chart(mastery_data: dict, title: str = "Concept Mastery") -> go.Figure:
    """Radar chart showing mastery levels across concepts."""
    original_concepts = list(mastery_data.keys())[:12]
    scores = [mastery_data[c]["score"] if isinstance(mastery_data[c], dict)
              else mastery_data[c] for c in original_concepts]
    
    from config import format_module_name
    formatted_concepts = [format_module_name(c) for c in original_concepts]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=formatted_concepts + [formatted_concepts[0]],
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.2)",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=8, color=COLORS["primary"]),
        name="Current Mastery",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(148,163,184,0.15)"),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(text=title, font=dict(size=16)),
        showlegend=False,
        height=400,
    )
    return apply_dark_theme(fig)


def mastery_heatmap(interaction_matrix: pd.DataFrame, title: str = "Student-Concept Mastery") -> go.Figure:
    """Heatmap of student mastery across concepts."""
    # Sample for performance
    sample = interaction_matrix.head(50) if len(interaction_matrix) > 50 else interaction_matrix

    fig = go.Figure(data=go.Heatmap(
        z=sample.values,
        x=sample.columns.tolist(),
        y=[str(i) for i in sample.index.tolist()],
        colorscale=[
            [0, "#EF4444"], [0.25, "#F97316"],
            [0.5, "#EAB308"], [0.75, "#22C55E"], [1, "#06B6D4"]
        ],
        colorbar=dict(title="Mastery", tickformat=".0%"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Concepts",
        yaxis_title="Students",
        height=500,
    )
    return apply_dark_theme(fig)


def score_distribution(scores: pd.Series, title: str = "Score Distribution") -> go.Figure:
    """Histogram with KDE overlay for score distributions."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=30,
        marker_color=COLORS["primary"],
        opacity=0.7,
        name="Scores",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Score",
        yaxis_title="Count",
        height=350,
        bargap=0.05,
    )
    return apply_dark_theme(fig)


def engagement_timeline(vle_data: pd.DataFrame, title: str = "Student Engagement Over Time") -> go.Figure:
    """Line chart of VLE engagement over time."""
    if "date" in vle_data.columns and "sum_click" in vle_data.columns:
        daily = vle_data.groupby("date")["sum_click"].sum().reset_index()
        daily.columns = ["Day", "Total Clicks"]
        daily = daily.sort_values("Day")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["Day"],
            y=daily["Total Clicks"],
            mode="lines",
            line=dict(color=COLORS["primary"], width=2),
            fill="tonexty",
            fillcolor="rgba(99, 102, 241, 0.1)",
        ))
        fig.update_layout(title=title, xaxis_title="Day", yaxis_title="Total Clicks", height=350)
    else:
        fig = go.Figure()
        fig.update_layout(title=title, height=350)

    return apply_dark_theme(fig)


def correlation_heatmap(df: pd.DataFrame, columns: list = None, title: str = "Correlation Matrix") -> go.Figure:
    """Correlation heatmap for numeric columns."""
    if columns:
        df = df[columns]
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        colorbar=dict(title="Correlation"),
    ))
    fig.update_layout(title=title, height=500, width=600)
    return apply_dark_theme(fig)


def class_imbalance_chart(labels: pd.Series, title: str = "Class Distribution") -> go.Figure:
    """Donut chart showing class imbalance."""
    counts = labels.value_counts()

    fig = go.Figure(data=[go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.55,
        marker=dict(colors=[COLORS["success"], COLORS["danger"], COLORS["warning"], COLORS["info"]]),
        textinfo="label+percent",
        textfont=dict(size=12),
    )])
    fig.update_layout(title=title, height=350, showlegend=True)
    return apply_dark_theme(fig)


def gap_severity_chart(gaps: list[dict], title: str = "Weak Topic Analysis") -> go.Figure:
    """Horizontal bar chart showing gap severity."""
    if not gaps:
        fig = go.Figure()
        fig.update_layout(title=title, height=300)
        return apply_dark_theme(fig)

    from config import format_module_name
    concepts = [format_module_name(g["concept"]) for g in gaps]
    confidences = [g["confidence"] for g in gaps]
    colors = []
    for g in gaps:
        sev = g.get("severity", "Low")
        if sev == "Critical":
            colors.append(COLORS["danger"])
        elif sev == "High":
            colors.append(COLORS["warning"])
        elif sev == "Moderate":
            colors.append("#EAB308")
        else:
            colors.append(COLORS["info"])

    fig = go.Figure(go.Bar(
        x=confidences,
        y=concepts,
        orientation="h",
        marker_color=colors,
        text=[f"{c:.0%}" for c in confidences],
        textposition="auto",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Weakness Confidence",
        xaxis=dict(range=[0, 1]),
        height=max(250, len(gaps) * 50),
    )
    return apply_dark_theme(fig)


def progress_chart(history: dict, title: str = "Training History") -> go.Figure:
    """Dual-axis chart for training loss and AUC-ROC."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("train_loss", []),
        name="Train Loss", line=dict(color=COLORS["danger"], width=2),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("val_loss", []),
        name="Val Loss", line=dict(color=COLORS["warning"], width=2, dash="dot"),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("val_auc", []),
        name="Val AUC", line=dict(color=COLORS["success"], width=2),
    ), secondary_y=True)

    fig.update_layout(title=title, height=400, xaxis_title="Epoch")
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_yaxes(title_text="AUC-ROC", secondary_y=True)
    return apply_dark_theme(fig)


def study_plan_timeline(plan: dict) -> go.Figure:
    """Visual timeline of the 7-day study plan."""
    days = plan.get("days", [])
    if not days:
        return apply_dark_theme(go.Figure())

    fig = go.Figure()

    for day in days:
        color = COLORS["info"] if not day.get("review") else COLORS["secondary"]
        # focus_topic is already human-readable (formatted by tools.py)
        focus_topic = day.get('focus_topic', '')

        fig.add_trace(go.Bar(
            x=[day["total_minutes"]],
            y=[day["day_label"]],
            orientation="h",
            marker_color=color,
            text=[f"{focus_topic} ({day['total_minutes']} min)"],
            textposition="inside",
            name=day["day_label"],
            showlegend=False,
        ))

    fig.update_layout(
        title="📅 7-Day Study Plan",
        xaxis_title="Study Minutes",
        height=350,
        barmode="stack",
    )
    return apply_dark_theme(fig)


def missing_value_heatmap(df: pd.DataFrame, title: str = "Missing Values") -> go.Figure:
    """Heatmap showing missing value patterns."""
    missing = df.isnull().astype(int)

    # Sample rows for performance
    if len(missing) > 100:
        missing = missing.sample(100, random_state=42)

    fig = go.Figure(data=go.Heatmap(
        z=missing.values,
        x=missing.columns.tolist(),
        y=list(range(len(missing))),
        colorscale=[[0, "#1E293B"], [1, COLORS["danger"]]],
        showscale=False,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Columns",
        yaxis_title="Rows",
        height=400,
    )
    return apply_dark_theme(fig)
