<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

# 🧠 LearnFlow AI — Personalized Student Learning Agent

An end-to-end ML/DL pipeline that analyzes student performance data, identifies weak knowledge areas using **Deep Knowledge Tracing (LSTM)**, detects learning gaps with **XGBoost**, recommends resources with an **LLM-based recommender** (OpenRouter) backed by a curated offline catalog, and delivers personalized 7-day study plans through an **AI agent** — all surfaced in an elegant **11-page Streamlit dashboard**.

---

## 🏗️ Architecture

```
Data Pipeline          ML/DL Models              AI Agent              Dashboard
┌──────────────┐    ┌──────────────────┐    ┌───────────────┐    ┌──────────────┐
│ OULAD + UCI  │───▶│ DKT (LSTM)      │───▶│ 5 Agent Tools │───▶│ 11-Page      │
│ Datasets     │    │ XGBoost Gap Det. │    │ 4-Layer       │    │ Streamlit    │
│ Feature Eng. │    │ LLM Recommender  │    │ Memory System │    │ + FastAPI    │
└──────────────┘    └──────────────────┘    └───────────────┘    └──────────────┘
```

---

## ✨ Features

### ML/DL Models
- **Deep Knowledge Tracing (DKT)** — 2-layer LSTM tracking mastery per concept over time
- **Gap Detector** — XGBoost classifier identifying weak topics with severity ranking
- **Recommender** — OpenRouter LLM (free tier) produces explained resource suggestions per weak concept; if no API key is set or the LLM is unavailable, it falls back to a curated offline catalog of real learning resources, so the dashboard always works

### AI Agent (4-Layer Memory)
- **Conversation Memory** — Rolling window chat history per student
- **Student Profile Memory** — Persistent profiles with strengths, weaknesses, goals
- **Recommendation Memory** — Tracks past suggestions, prevents repeats, records feedback
- **Learning History** — Event log (assessments, mastery updates, milestones)

### Dashboard (11 Pages)
| Page | Description |
|------|-------------|
| 🏠 Home | Overview, quick actions, architecture diagram |
| 👤 Student Analytics | Profile cards, mastery radar, student table |
| 📊 Dataset Explorer | Browse OULAD + UCI datasets with schema inspection |
| 🔬 EDA | 8-tab interactive analysis (distributions, correlations, etc.) |
| 🧬 Knowledge Tracing | DKT mastery visualization with per-concept progress bars |
| 🎯 Weak Topic Analysis | Gap severity charts, ranked weakness cards |
| 📚 Recommendation Center | Resource cards with type filtering |
| 📅 AI Study Planner | 7-day plan generator with timeline visualization |
| 📈 Progress Analytics | Improvement tracking, concept-level change reports |
| 💬 AI Chat | Conversational UI with quick actions and memory inspector |
| ⚙️ Settings | API keys, model parameters, memory management, system info |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **pip** (Python package manager)
- **OpenRouter API Key** — Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys)

### 1. Clone & Setup
```bash
git clone https://github.com/Jiteshhh001/SuperComputing.git
cd "SuperComputing/Final project"
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
# Copy the template and add your OpenRouter key
copy .env.template .env       # Windows
# cp .env.template .env       # Linux/macOS

# Edit .env and set your key:
# OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 3. Run the Pipeline
```bash
python run_pipeline.py
```
This will:
- Download OULAD + UCI datasets (or generate synthetic fallbacks)
- Clean and engineer features
- Train DKT (LSTM) and the XGBoost Gap Detector, and set up the recommender
- Initialize the AI agent and run a test analysis

### 4. Launch the Dashboard
```bash
streamlit run app/streamlit_app.py
```

### 5. (Optional) Run the API
```bash
uvicorn api.main:app --reload
```

---

## 📁 Project Structure

```
Final project/
├── config.py                       # Central configuration (hyperparams, paths, thresholds)
├── requirements.txt                # Python dependencies
├── run_pipeline.py                 # End-to-end pipeline runner
├── .env.template                   # Environment variable template
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT License
│
├── data/
│   ├── download_datasets.py        # OULAD + UCI auto-downloader with fallbacks
│   ├── raw/                        # OULAD + UCI datasets (included for the demo)
│   ├── processed/                  # Feature-engineered data (included for the demo)
│   ├── memory/                     # Persistent agent memory (git-ignored)
│   └── resources/                  # Static learning resources
│
├── src/
│   ├── data_processing/
│   │   ├── loader.py               # Unified data loading (OULAD 7-table merge, UCI)
│   │   ├── cleaner.py              # Missing value handling, deduplication, audit trail
│   │   └── feature_engineer.py     # Interaction matrices, DKT sequences, gap features
│   │
│   ├── models/
│   │   ├── dkt_model.py            # LSTM-based Deep Knowledge Tracing architecture
│   │   ├── dkt_trainer.py          # Training loop with early stopping, AUC metrics
│   │   ├── gap_detector.py         # XGBoost weak-topic classifier
│   │   └── recommender.py          # LLM recommender with offline fallback catalog
│   │
│   ├── agent/
│   │   ├── memory.py               # 4-layer persistent memory system
│   │   ├── tools.py                # 5 agent tools (tracer, detector, recommender, planner, reporter)
│   │   ├── agent.py                # Orchestrator with OpenRouter LLM integration
│   │   └── prompts.py              # System prompts and templates
│   │
│   └── utils/
│       ├── metrics.py              # AUC-ROC, F1, NDCG@K, Precision@K
│       └── visualization.py        # Plotly charts with dark theme
│
├── models/                         # Saved model artifacts (included for the demo)
│   ├── dkt_best.pt                 # Trained DKT LSTM weights
│   └── gap_detector.pkl            # Trained XGBoost gap detector
│
├── api/
│   └── main.py                     # FastAPI REST backend
│
└── app/
    ├── streamlit_app.py            # Main dashboard entry point with theme
    └── pages/                      # 11 dashboard pages
        ├── home.py
        ├── student_analytics.py
        ├── dataset_explorer.py
        ├── eda.py
        ├── knowledge_tracing.py
        ├── weak_topic_analysis.py
        ├── recommendation_center.py
        ├── ai_study_planner.py
        ├── progress_analytics.py
        ├── ai_chat.py
        └── settings.py
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/student/{id}/analyze` | Full analysis pipeline |
| `GET` | `/api/student/{id}/mastery` | Current mastery scores |
| `GET` | `/api/student/{id}/gaps` | Weak topics |
| `GET` | `/api/student/{id}/recommendations` | Resource recommendations |
| `POST` | `/api/student/{id}/plan` | 7-day study plan |
| `GET` | `/api/student/{id}/progress` | Progress report |
| `POST` | `/api/chat` | Chat with the AI agent |
| `GET` | `/api/students` | List available students |

---

## 📊 Evaluation Metrics

| Component | Metric | Target |
|-----------|--------|--------|
| DKT (LSTM) | AUC-ROC | >0.75 |
| Gap Detector (XGBoost) | F1-Score | >0.70 |
| Recommender | NDCG@5 | >0.60 |
| Recommender | Precision@3 | >0.50 |

Saved DKT checkpoint (best epoch): validation AUC-ROC **0.82**, accuracy 0.76 over 12 concepts.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch (LSTM) |
| **Machine Learning** | XGBoost, scikit-learn |
| **LLM Integration** | OpenRouter API (free tier — Llama, Gemini, DeepSeek) |
| **AI Agent** | Custom tool-calling agent (5 tools, 4-layer memory) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit + Plotly |
| **Data** | OULAD (32K students) + UCI Student Performance (649 students) |

---

## ☁️ Deployment

**Streamlit Community Cloud (free):** sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub → *Create app* → repository `Jiteshhh001/SuperComputing`, branch `main`, main file path `Final project/app/streamlit_app.py` → *Deploy*. Optional: in *Advanced settings → Secrets* add `OPENROUTER_API_KEY = "your-key"` for LLM-generated recommendations and chat (without it the app uses the offline catalog).

**Docker:** `docker build -t learnflow-ai .` then `docker run -p 8501:8501 learnflow-ai` (open http://localhost:8501).

---

## 🙏 Acknowledgements

- [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset)
- [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
- [OpenRouter](https://openrouter.ai/) for free-tier LLM access
- [Streamlit](https://streamlit.io/) for the dashboard framework
