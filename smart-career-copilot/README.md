<div align="center">

# ✦ Smart Career Copilot

### AI-Powered Multi-Agent Career Development Platform

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-6366f1?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

*A production-grade multi-agent AI system for resume optimization, deep research, autonomous coding, and interview preparation.*

</div>

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────┐
                    │      React Frontend (Vite)   │
                    │   Premium Dark UI + WebSocket │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     FastAPI Backend          │
                    │  REST API + WebSocket Stream  │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    LangGraph Orchestrator    │
                    │  Reason → Act → Observe →    │
                    │  Reflect → Retry             │
                    └──────────────┬──────────────┘
                                   │
              ┌────────┬───────────┼───────────┬────────┐
              ▼        ▼           ▼           ▼        │
         ┌─────────┐┌─────────┐┌─────────┐┌─────────┐  │
         │ Resume  ││Research ││ Coding  ││Interview│  │
         │ Agent   ││ Agent   ││ Agent   ││ Agent   │  │
         └────┬────┘└────┬────┘└────┬────┘└────┬────┘  │
              │          │         │          │         │
         ┌────▼──────────▼─────────▼──────────▼────┐   │
         │        Tool Registry + RAG Pipeline      │   │
         │   ChromaDB │ OpenAI Embeddings │ SQLite  │   │
         └──────────────────────────────────────────┘   │
```

## ✨ Features

### 📄 Smart Resume Tailor Agent
- Upload & parse resume PDFs
- ATS compatibility scoring (keyword, format, section analysis)
- Skill extraction (technical + soft skills)
- Job description comparison & skill gap analysis
- AI-powered bullet point rewriting
- Cover letter generation
- Download improved resume

### 🔍 Deep Research Agent
- Multi-source web search (Tavily API)
- Academic paper search (arXiv)
- Webpage content extraction
- Synthesized research reports with citations
- Chart data generation
- Conversation memory for follow-up queries
- Export to Markdown

### 💻 Autonomous Coding Agent
- **Plan → Execute → Review → Improve** workflow
- Automatic file generation
- Sandboxed Python code execution
- Test running & validation
- Code review with reflection loops (max 3 iterations)
- README generation
- Context maintenance across iterations

### 🎯 Interview Coach Agent
- Technical interview questions (algorithms, system design)
- Behavioral questions (STAR method)
- Resume-based personalized questions
- Real-time answer evaluation & scoring
- Detailed feedback per question
- Final scorecard with category breakdown
- Personalized improvement plan

### 🧠 Memory & RAG
- **Conversation Memory**: Per-session chat history with SQLite persistence
- **Session Management**: Create, list, resume, delete sessions
- **Long-Term Memory**: ChromaDB-backed semantic memory
- **RAG Pipeline**: Document ingestion → chunking → embedding → retrieval

### 🎨 Premium UI
- Dark mode with glassmorphism effects
- Animated cards and micro-interactions
- Chat bubbles with markdown rendering
- Typing indicator with animated dots
- Source citation cards
- Drag-and-drop file upload
- Responsive design

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Node.js 20+
- OpenAI API Key
- Tavily API Key (for research agent)

### 1. Clone & Setup Backend

```bash
cd smart-career-copilot/backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env with your API keys
```

### 2. Setup Frontend

```bash
cd smart-career-copilot/frontend

# Install dependencies
npm install
```

### 3. Run the Application

```bash
# Terminal 1 — Backend
cd backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

### 4. Docker (Alternative)

```bash
# Copy and configure .env
copy backend/.env.example backend/.env

# Build and run
docker-compose up --build
```

Open **http://localhost:3000** in your browser.

---

## 📁 Project Structure

```
smart-career-copilot/
├── backend/
│   ├── app/
│   │   ├── agents/          # LangGraph agent implementations
│   │   │   ├── orchestrator.py   # Main orchestration pipeline
│   │   │   ├── router.py         # Intent classification
│   │   │   ├── resume_agent.py   # Resume analysis
│   │   │   ├── research_agent.py # Deep research
│   │   │   ├── coding_agent.py   # Autonomous coding
│   │   │   └── interview_agent.py# Interview coaching
│   │   ├── api/routes/      # FastAPI route handlers
│   │   ├── tools/           # LangChain tool implementations
│   │   ├── rag/             # RAG pipeline (ChromaDB + embeddings)
│   │   ├── memory/          # Conversation, session, long-term memory
│   │   ├── models/          # Pydantic schemas & database models
│   │   ├── utils/           # Logging & helpers
│   │   ├── config.py        # Pydantic settings
│   │   └── main.py          # FastAPI entry point
│   ├── tests/               # Unit tests
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/      # Reusable React components
│   │   │   ├── layout/      # Sidebar, Header, MainLayout
│   │   │   ├── chat/        # ChatWindow, ChatBubble, ChatInput
│   │   │   └── common/      # FileUpload, AnimatedCard
│   │   ├── pages/           # Dashboard, agent pages
│   │   ├── hooks/           # useChat, useFileUpload, useWebSocket
│   │   ├── services/        # API client (Axios)
│   │   ├── store/           # Zustand state management
│   │   └── styles/          # Design system CSS
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🔧 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key for LLM + embeddings | ✅ |
| `TAVILY_API_KEY` | Tavily API key for web search | ✅ |
| `ORCHESTRATOR_MODEL` | LLM for orchestrator (default: `gpt-4o`) | ❌ |
| `WORKER_MODEL` | LLM for workers (default: `gpt-4o-mini`) | ❌ |
| `EMBEDDING_MODEL` | Embedding model (default: `text-embedding-3-small`) | ❌ |
| `DEBUG` | Enable debug mode (default: `true`) | ❌ |

---

## 🧪 Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 19, Vite, Zustand, React Router, Axios |
| **Backend** | FastAPI, Python 3.12, Pydantic v2, SQLAlchemy |
| **AI/ML** | LangChain, LangGraph, OpenAI GPT-4o/4o-mini |
| **RAG** | ChromaDB, OpenAI Embeddings, Recursive Text Splitter |
| **Search** | Tavily API, arXiv API, BeautifulSoup |
| **Database** | SQLite (sessions/messages), ChromaDB (vectors/memory) |
| **DevOps** | Docker, Docker Compose, Nginx |
| **Design** | Custom CSS, Glassmorphism, Inter font |

---

## 📝 API Documentation

Once the backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

<div align="center">

**Built using LangGraph, React, and FastAPI**

</div>
