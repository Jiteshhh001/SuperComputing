<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

# 🧠 LearnFlow AI — Personalized Student Learning Agent

🔗 **Live demo:** https://supercomputinginternship.streamlit.app/ *(the app may take a few seconds to wake up)*

An AI agent that designs a personalised learning journey for each student. It is trained on the real **Open University Learning Analytics Dataset (OULAD, 32,593 students)**:

- **Deep Knowledge Tracing (LSTM)** estimates the student's mastery of every concept.
- An **XGBoost risk model** predicts which concepts the student is likely to score below 50% in on the next assessment.
- An **LLM ranks resources** from a curated, link-checked catalog for the at-risk concepts.
- The agent builds a **7-day study plan** and answers questions in an **AI chat** grounded in the student's own data.

Everything is shown in an 11-page Streamlit dashboard, with a FastAPI backend.

---

## 🏗️ Architecture

```
Data Pipeline            ML/DL Models                 AI Agent                  Dashboard
┌────────────────┐    ┌───────────────────────┐    ┌────────────────────┐    ┌──────────────┐
│ OULAD (real,   │───▶│ DKT (2-layer LSTM)    │───▶│ 5 tools            │───▶│ 11-page      │
│ 32,593 stud.)  │    │ XGBoost next-         │    │ 4-layer memory     │    │ Streamlit    │
│ + UCI          │    │ assessment risk model │    │ LLM chat (Open-    │    │ + FastAPI    │
│ Feature eng.   │    │ Curated catalog +     │    │ Router, model      │    │              │
│                │    │ LLM ranking           │    │ fallback chain)    │    │              │
└────────────────┘    └───────────────────────┘    └────────────────────┘    └──────────────┘
```

---

## 📊 Results (real OULAD)

| Component | Setup | Result |
|-----------|-------|--------|
| **DKT (LSTM)** | 19,956 student sequences (≥ 3 assessments), 14 concepts (module × assessment type), 70/15/15 split by student | **Test AUC-ROC 0.829**, accuracy 0.923 (target > 0.75) |
| **Gap / risk detector (XGBoost)** | 130,018 history → next-assessment rows, 5-fold CV grouped by student, threshold tuned on out-of-fold F1 (0.23) | **CV AUC-ROC 0.871**, F1 0.48 (precision 0.44, recall 0.54), base rate 8.5% |

The numbers are stored in [`models/model_metrics.json`](models/model_metrics.json) and shown on the dashboard.

**Gap detector design.** The model only uses information available *before* the assessment it predicts:
- the score history for the concept
- lateness
- VLE clicks up to that point
- module, assessment type and demographics

This avoids target leakage. Probabilities are not re-weighted, so the risk shown to students is a calibrated estimate.

---

## ✨ Features

### Models and tools
- **Knowledge Tracer:** DKT LSTM mastery per concept, shown only for the modules the student is actually taking.
- **Gap Detector:** XGBoost risk of scoring below 50% on the next assessment of each concept, with a severity ranking.
- **Resource Recommender:**
  - draws on 23 curated resources grouped by OULAD domain (Social Sciences / STEM) and assessment type (TMA / CMA / Exam)
  - an LLM picks and explains the best matches
  - without a key, a severity-aware heuristic ranks them
  - the LLM never invents links
- **Study Planner:** a 7-day plan that rotates between the highest-risk concepts, with assessment-specific activities.
- **Progress Reporter:** mastery change over time, read from the learning-history memory.

### AI chat (4-layer memory)
- Answers are **grounded in the tools' outputs**: profile, mastery, risk, resources, plan and progress.
- An OpenRouter LLM writes the answers. It tries a **chain of free models** and remembers the one that works, because free models are often rate-limited.
- **Offline mode:** without an API key, the chat still answers from the data using templates, including study tips.
- Memory has four layers: conversation, student profile, past recommendations and learning history.

### Dashboard (11 pages)
| Page | Description |
|------|-------------|
| 🏠 Home | Live metrics, one-click full analysis, architecture |
| 👤 Student Analytics | Profile cards and student table |
| 📊 Dataset Explorer | Browse the OULAD sample and UCI tables |
| 🔬 EDA | 8-tab interactive analysis |
| 🧬 Knowledge Tracing | DKT mastery radar and per-concept bars |
| 🎯 Weak Topic Analysis | Risk ranking of every concept |
| 📚 Recommendation Center | Ranked resources with reasons |
| 📅 AI Study Planner | 7-day plan with a timeline |
| 📈 Progress Analytics | Concept-level change reports |
| 💬 AI Chat | Grounded chat with quick actions and a memory inspector |
| ⚙️ Settings | Session-only API key, model chain, model parameters, memory |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- *(Optional)* a free **OpenRouter API key** from [openrouter.ai/keys](https://openrouter.ai/keys), which enables the AI chat and LLM ranking

### 1. Clone & Setup
```bash
git clone https://github.com/Jiteshhh001/SuperComputing.git
cd "SuperComputing/Final project"
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### 2. Configure the API key (optional)
```bash
copy .env.template .env       # Windows
# cp .env.template .env       # Linux/macOS
# then set OPENROUTER_API_KEY=sk-or-v1-...
```

### 3. Launch the dashboard
The repository already contains the trained models and a 2,000-student sample of the real OULAD, so the app runs straight away:
```bash
streamlit run app/streamlit_app.py
```

### 4. (Optional) Retrain from the full dataset
```bash
python run_pipeline.py            # downloads the real OULAD, cleans it, trains DKT + XGBoost
python data/build_demo_subset.py  # shrinks the data back to the 2,000-student demo sample
```

### 5. (Optional) Run the API
```bash
uvicorn api.main:app --reload
```

---

## 📁 Project Structure

```
Final project/
├── config.py                       # Central configuration (paths, hyperparameters, LLM model chain)
├── requirements.txt                # Python dependencies (CPU PyTorch on Linux hosts)
├── run_pipeline.py                 # End-to-end pipeline: data → features → models → metrics
├── Dockerfile                      # Container for any Docker host
├── .env.template                   # Environment variable template
│
├── data/
│   ├── download_datasets.py        # OULAD (UCI mirror) + UCI Student Performance downloader
│   ├── build_demo_subset.py        # Stratified 2,000-student sample for the hosted app
│   ├── raw/                        # OULAD sample (studentVle as .csv.gz) + UCI tables
│   ├── processed/                  # DKT sequences, risk features, profiles (demo sample)
│   └── memory/                     # Agent memory written at runtime (git-ignored)
│
├── src/
│   ├── data_processing/            # loader.py, cleaner.py, feature_engineer.py
│   ├── models/
│   │   ├── dkt_model.py            # LSTM Deep Knowledge Tracing
│   │   ├── dkt_trainer.py          # Training loop, early stopping, AUC metrics
│   │   ├── gap_detector.py         # XGBoost next-assessment risk model
│   │   └── recommender.py          # Curated catalog + LLM ranking
│   ├── agent/
│   │   ├── agent.py                # Orchestrator: tools → grounded context → LLM / offline answer
│   │   ├── llm.py                  # OpenRouter client with model fallback chain
│   │   ├── tools.py                # 5 tools (tracer, detector, recommender, planner, reporter)
│   │   ├── memory.py               # 4-layer persistent memory
│   │   └── prompts.py              # System prompts
│   └── utils/                      # metrics.py, visualization.py
│
├── models/                         # dkt_best.pt, gap_detector.pkl, model_metrics.json
├── api/main.py                     # FastAPI REST backend
└── app/
    ├── streamlit_app.py            # Dashboard entry point
    └── pages/                      # 11 dashboard pages
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/student/{id}/analyze` | Full analysis pipeline |
| `GET` | `/api/student/{id}/mastery` | Current mastery scores |
| `GET` | `/api/student/{id}/gaps` | At-risk topics |
| `GET` | `/api/student/{id}/recommendations` | Resource recommendations |
| `POST` | `/api/student/{id}/plan` | 7-day study plan |
| `GET` | `/api/student/{id}/progress` | Progress report |
| `POST` | `/api/chat` | Chat with the AI agent |
| `GET` | `/api/students` | List available students |

---

## ☁️ Deployment

**Streamlit Community Cloud (free):**
1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. Choose *Create app*, then repository `Jiteshhh001/SuperComputing`, branch `main`, main file path `Final project/app/streamlit_app.py`.
3. Click *Deploy*.

To turn on the AI chat for everyone, open *App settings → Secrets* and add:
```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
```
Without it, the app runs in offline mode.

**Docker:** `docker build -t learnflow-ai .` then `docker run -p 8501:8501 learnflow-ai` (open http://localhost:8501).

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch (LSTM) |
| **Machine Learning** | XGBoost, scikit-learn |
| **LLM Integration** | OpenRouter API (free-tier models with fallback) |
| **AI Agent** | Custom tool-calling agent (5 tools, 4-layer memory) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit + Plotly |
| **Data** | OULAD (32,593 students; 2,000-student demo sample) + UCI Student Performance (1,044 students) |

---

## 🙏 Acknowledgements

- Kuzilek, J., Hlosta, M. & Zdrahal, Z. (2017). *Open University Learning Analytics dataset.* Scientific Data 4, 170171 ([UCI mirror](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset), CC BY 4.0)
- [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
- Piech, C. et al. (2015). *Deep Knowledge Tracing.* NeurIPS
- [OpenRouter](https://openrouter.ai/) for free-tier LLM access
- [Streamlit](https://streamlit.io/) for the dashboard framework
