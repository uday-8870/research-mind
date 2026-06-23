# ResearchMind

An autonomous multi-agent AI system that researches any topic end-to-end — searches the web, reads sources, writes a report, and scores itself.

Built with LangGraph, FastAPI, ChromaDB, and React.

---

## What it does

Takes a question → breaks it into sub-queries → searches the web in parallel → reads and chunks documents → synthesizes a cited report → runs a self-evaluation pipeline that scores faithfulness, coverage, and hallucination rate.

---

## Stack

- **Agents** — LangGraph orchestrator with 5 specialized agents
- **LLM** — OpenAI GPT-4o or Groq (free tier supported)
- **Search** — Tavily API
- **Memory** — ChromaDB vector store with local embeddings
- **API** — FastAPI with async endpoints
- **Frontend** — React + Vite

---

## Quickstart

```bash
git clone https://github.com/uday-8870/research-mind.git
cd research-mind

python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

cp .env.example .env         # Add your API keys

uvicorn backend.api.main:app --reload --port 8000
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`

---

## API keys needed

| Key | Where |
|-----|-------|
| `OPENAI_API_KEY` or `GROQ_API_KEY` | platform.openai.com or console.groq.com (free) |
| `TAVILY_API_KEY` | app.tavily.com (free tier) |

---

## License

MIT
