<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-Agent-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

# 🧠 LearnFlow AI — Personalized Student Learning Agent

An end-to-end ML/DL pipeline that analyzes student performance data, identifies weak knowledge areas using **Deep Knowledge Tracing (LSTM)**, detects learning gaps with **XGBoost**, recommends resources via a **hybrid engine** (Collaborative Filtering + Content-Based Filtering + LLM Ranking), and delivers personalized 7-day study plans through an **AI agent** — all surfaced in an elegant **11-page Streamlit dashboard**.

---

## 🏗️ Architecture

```
Data Pipeline          ML/DL Models              AI Agent              Dashboard
┌──────────────┐    ┌──────────────────┐    ┌───────────────┐    ┌──────────────┐
│ OULAD + UCI  │───▶│ DKT (LSTM)       │───▶│ 5 Agent Tools │───▶│ 11-Page      │
│ Datasets     │    │ XGBoost Gap Det. │    │ 4-Layer       │    │ Streamlit    │
│ Feature Eng. │    │ Hybrid Recommend.│    │ Memory System │    │ + FastAPI    │
└──────────────┘    └──────────────────┘    └───────────────┘    └──────────────┘
```

---

## ✨ Features

### ML/DL Models
- **Deep Knowledge Tracing (DKT)** — 2-layer LSTM tracking mastery per concept over time
- **Gap Detector** — XGBoost classifier identifying weak topics with severity ranking
- **Hybrid Recommender** — 3-stage pipeline:
  - Collaborative Filtering (SVD via `scikit-surprise`)
  - Content-Based Filtering (Sentence-BERT embeddings)
  - LLM Ranking (OpenRouter API — free tier)

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
| 📚 Recommendation Center | Hybrid resource cards with type filtering |
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
git clone https://github.com/<your-username>/personalized-learning-agent.git
cd personalized-learning-agent
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
- Train DKT (LSTM), XGBoost Gap Detector, and Hybrid Recommender
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
personalized-learning-agent/
├── config.py                       # Central configuration (hyperparams, paths, thresholds)
├── requirements.txt                # Pinned dependencies
├── run_pipeline.py                 # End-to-end pipeline runner
├── .env.template                   # Environment variable template
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT License
│
├── data/
│   ├── download_datasets.py        # OULAD + UCI auto-downloader with fallbacks
│   ├── raw/                        # Downloaded datasets (git-ignored)
│   ├── processed/                  # Feature-engineered data (git-ignored)
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
│   │   └── recommender.py          # 3-stage hybrid recommendation engine
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
├── models/                         # Saved model artifacts (git-ignored)
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

| Component | Metric | Benchmark |
|-----------|--------|-----------|
| DKT (LSTM) | AUC-ROC | >0.75 |
| Gap Detector (XGBoost) | F1-Score | >0.70 |
| Recommender (Hybrid) | NDCG@5 | >0.60 |
| Recommender (Hybrid) | Precision@3 | >0.50 |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch (LSTM) |
| **Machine Learning** | XGBoost, scikit-learn, scikit-surprise |
| **NLP & Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) |
| **LLM Integration** | OpenRouter API (free tier — Llama, Gemini, DeepSeek) |
| **AI Agent** | LangChain |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit + Plotly |
| **Data** | OULAD (32K students) + UCI Student Performance (649 students) |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset)
- [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
- [OpenRouter](https://openrouter.ai/) for free-tier LLM access
- [Streamlit](https://streamlit.io/) for the dashboard framework
- [LangChain](https://www.langchain.com/) for AI agent orchestration
