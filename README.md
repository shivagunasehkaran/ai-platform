# AI Platform with agentic system

A comprehensive AI platform demonstrating Chat, RAG, Agent, and Code Assistant capabilities using Azure-hosted GPT-4o.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AI Platform                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Chat CLI   │  │   RAG API    │  │    Agent     │  │    Code      │ │
│  │              │  │   /qa        │  │   Planner    │  │   Assistant  │ │
│  │  Streaming   │  │  Citations   │  │  Tool Calls  │  │  Self-Heal   │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                 │                 │                 │          │
│         └────────────┬────┴────────┬────────┴────────┬────────┘          │
│                      │             │                 │                   │
│                      ▼             ▼                 ▼                   │
│              ┌─────────────────────────────────────────────┐             │
│              │              Shared Layer                   │             │
│              │  ┌─────────┐ ┌──────────┐ ┌──────────────┐ │             │
│              │  │   LLM   │ │ Telemetry│ │    Memory    │ │             │
│              │  │ Client  │ │  Metrics │ │ (Last 10)    │ │             │
│              │  └────┬────┘ └────┬─────┘ └──────────────┘ │             │
│              └───────┼───────────┼───────────────────────-┘             │
│                      │           │                                       │
│         ┌────────────┘           └────────────┐                         │
│         ▼                                     ▼                         │
│  ┌─────────────┐                      ┌─────────────┐                   │
│  │ Azure GPT-4o│                      │   FAISS     │                   │
│  │   (LLM)     │                      │ Vector DB   │                   │
│  └─────────────┘                      └─────────────┘                   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Streamlit Dashboard                          │    │
│  │         Latency | Cost | Retrieval Accuracy | Agent Stats       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- **Python**: 3.11+
- **Docker**: Docker Desktop (for containerized deployment)
- **Rust** (optional): For Rust code generation in Code Assistant
- **Node.js** (optional): For JavaScript code generation

## Quick Start

### 1. Clone and Setup

```bash
cd ai-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your credentials
```

**Required environment variables:**

| Variable          | Description                   |
| ----------------- | ----------------------------- |
| `OPENAI_BASE_URL` | Azure OpenAI endpoint URL     |
| `OPENAI_API_KEY`  | API key for authentication    |
| `MODEL_NAME`      | Model name (default: `Gpt4o`) |

### 3. Ingest Corpus (for RAG)

```bash
# Add .txt files to corpus/ directory (≥50MB)
# Then run ingestion
python -m rag.ingest
```

## Running Each Task

### Task 3.1: Chat CLI (Streaming + Telemetry)

```bash
python -m chat.chat
```

**Features:**

- Token-level streaming
- Last 10 messages in memory
- Per-turn metrics: `[stats] prompt=X completion=Y cost=$Z latency=Wms`

### Task 3.2: RAG Question Answering

```bash
# Start API server
uvicorn rag.api:app --host 0.0.0.0 --port 8000

# Or test CLI
python -m rag.retrieve "Who is Mr. Darcy?"

# Run evaluation (24 test questions)
python -m rag.evaluate
```

**API Endpoints:**

- `GET /health` — Health check
- `POST /qa` — Question answering with citations

### Task 3.3: Agent Planner

```bash
python -m agent.planner "Plan a 2-day trip to Auckland for NZ$500"
```

**Features:**

- 4 tools: flights, hotels, weather, attractions
- Scratch-pad reasoning in logs
- Budget/date constraint handling

### Task 3.4: Code Assistant

```bash
# Python
python -m code_assistant.repair_loop "Write a function to check if a number is prime"

# Rust
python -m code_assistant.repair_loop "Write fibonacci in Rust"

# JavaScript
python -m code_assistant.repair_loop "Write factorial in JavaScript"
```

**Features:**

- Multi-language support (Python, Rust, JavaScript)
- Self-healing: captures errors, retries up to 3 times

## Running Tests

```bash
# Run all tests
pytest -q

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_chat.py -v
```

## Docker Deployment

### Start All Services

```bash
# Build and start
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Services

| Service     | URL                   | Description                 |
| ----------- | --------------------- | --------------------------- |
| `rag-api`   | http://localhost:8000 | FastAPI + FAISS Vector DB   |
| `dashboard` | http://localhost:8501 | Streamlit Metrics Dashboard |

### Dashboard Features

- **Chat Metrics**: Latency, cost over time, token distribution
- **RAG Metrics**: Recall@5, MRR@5 gauges, latency histogram
- **Agent Metrics**: Success/failure breakdown, tool calls per task

## Project Structure

```
ai-platform/
├── chat/chat.py              # Task 3.1: Streaming chat
├── rag/                      # Task 3.2: RAG pipeline
│   ├── ingest.py             # Corpus ingestion
│   ├── retrieve.py           # Query retrieval
│   ├── api.py                # FastAPI endpoints
│   ├── evaluate.py           # Evaluation script
│   └── qa_pairs.json         # Test questions (24)
├── agent/                    # Task 3.3: Planning agent
│   ├── planner.py            # Agent loop
│   ├── tools.py              # Mock tools
│   └── schemas.py            # Pydantic models
├── code_assistant/           # Task 3.4: Code assistant
│   ├── repair_loop.py        # Self-healing loop
│   └── runner.py             # Test execution
├── shared/                   # Shared utilities
│   ├── config.py             # Configuration
│   ├── llm.py                # LLM client
│   ├── telemetry.py          # Metrics tracking
│   └── memory.py             # Conversation memory
├── dashboard/app.py          # Streamlit dashboard
├── tests/                    # Unit tests
├── corpus/                   # Text files for RAG
├── docker-compose.yml        # Container orchestration
├── Dockerfile                # Container build
├── requirements.txt          # Dependencies
└── report.md                 # Design decisions
```

## Metrics & Evaluation

### RAG Performance

| Metric           | Value | Target |
| ---------------- | ----- | ------ |
| Recall@5         | ~90%  | ≥70%   |
| MRR@5            | ~0.85 | ≥0.50  |
| Median Retrieval | 10ms  | ≤300ms |

### Cost Tracking

All operations log to `metrics.json`:

- Chat: tokens, cost, latency per message
- RAG: retrieval latency, recall, MRR
- Agent: success rate, tool calls, cost

## Output screenshots
<img width="2052" height="1355" alt="Screenshot 2026-02-12 at 11 50 27 PM" src="https://github.com/user-attachments/assets/aa90aa72-6829-4eed-8b28-686d09ddc349" />
<img width="2050" height="1379" alt="Screenshot 2026-02-12 at 11 50 55 PM" src="https://github.com/user-attachments/assets/86c0d37d-a6cf-46af-8414-b1fe610027a1" />
<img width="1695" height="1068" alt="Screenshot 2026-02-12 at 11 52 07 PM" src="https://github.com/user-attachments/assets/6448b795-9caa-40a9-a9fa-486e94bd536e" />
<img width="1232" height="653" alt="Screenshot 2026-02-12 at 11 53 41 PM" src="https://github.com/user-attachments/assets/40307d67-0ebc-4a60-8f16-a5eece14e79c" />
<img width="1375" height="1000" alt="Screenshot 2026-02-12 at 11 54 57 PM" src="https://github.com/user-attachments/assets/ddc1142f-e019-48de-84f9-aede0dfd6372" />
<img width="1369" height="999" alt="Screenshot 2026-02-12 at 11 55 43 PM" src="https://github.com/user-attachments/assets/d81ed4f0-a26f-4124-89f5-942bc66ea730" />
<img width="1367" height="1003" alt="Screenshot 2026-02-12 at 11 56 38 PM" src="https://github.com/user-attachments/assets/4326b6fe-81f9-47a5-9d7f-fc67be7e64d0" />
<img width="1371" height="1004" alt="Screenshot 2026-02-12 at 11 57 54 PM" src="https://github.com/user-attachments/assets/1f5205c8-0838-40d6-92ce-55708ac455a2" />
