# HybridQA Multi-Agent System

<div align="center">
  <img src="https://img.shields.io/badge/LangGraph-0.0.40-FF6F00?logo=langchain&logoColor=white" alt="LangGraph">
  <img src="https://img.shields.io/badge/FastAPI-0.111.0-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-3.8-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Async-Await-5E35B1?logo=typescript&logoColor=white" alt="Async">
</div>

## Overview
Implements multi-hop question answering over tabular data and Wikipedia passages using LangGraph agents.

```diff
+ 70k Q-A pairs processing
+ 13k tables with linked Wikipedia passages
+ Multi-modal retrieval (structured + unstructured)
```

## Workflow  
```mermaid
graph LR
    A[📥 User Query] --> B{{ReAct Planner Agent}}
    B -->|"table_task"| C[🛠️ Table Tool Agent]
    C --> B
    C -->|"extracted_data"| D[🔬 Analysis Agent]
    D -->|"tool_calling"| E[🌐 Wikipedia RAG]
    E --> D
    D -->|"final_answer"| F[📤 API Response]
```

### Docker Setup
```bash
cd docker

# Start with compose (auto-builds)
docker compose build

docker compose run  -p 8000:8000 --remove-orphans  hybridqa-agent
```

### API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Submit question with table JSON |
| `GET`  | `/health` | Service status check (**TBD**)|

**Example Request**:
```bash
curl -X POST http://localhost:8000/query \
  -F "table_file=@./input_samples/input.json"
```

## Key Features

| Feature | Tech Stack |
|---------|------------|
| 🧠 Intelligent Planning | GPT-4o + ReAct |
| 📊 Table Processing | Pandas Agent + LangGraph | 
| 🔍 Hybrid Retrieval | ChromaDB + BM25 | 
| ⚡ Async API | FastAPI + Uvicorn + Asyncio | 
| 🐳 Containerized | Docker Compose | 
