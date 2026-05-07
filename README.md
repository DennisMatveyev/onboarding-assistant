# Onboarding Assistant

A RAG-based chat application that helps newcomers get familiar with company processes by answering questions from onboarding documents over a WebSocket interface.

## Architecture

```
ws_frontend.html  →  FastAPI WebSocket  →  LangGraph  →  LangChain RAG chain
                                                              ↓
                                                        RedisVL (HNSW index)
                                                        SemanticCache (LLM cache)
```

- **FastAPI** — WebSocket server, one persistent connection per user
- **LangGraph** — routes messages between a RAG node and a birthday lookup node, with per-user conversation memory
- **LangChain** — retrieval chain (`create_retrieval_chain`) with a custom `RedisRetriever`
- **RedisVL** — vector index (`AsyncSearchIndex`) for document search + `SemanticCache` for LLM response caching
- **OpenAI** — configurable LLM model, embedding model

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Redis Stack (includes vector search module)
- OpenAI API key

## Setup

**1. Start Redis Stack**

```bash
docker run -d --name redis_onboard_assist -p 6379:6379 redis:8.4
```

**2. Install dependencies**

```bash
uv sync
```

**3. Configure environment**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=...
HF_TOKEN=...   # optional, to enable higher rate limits and faster downloads
REDIS_URL=redis://localhost:6379  # optional, this is the default
```

**4. Add wiki documents**

Place your documents in the `wiki_docs/` directory. Supported formats: `.txt`, `.pdf`, `.docx`.

```
wiki_docs/
├── handbook.pdf
├── processes.docx
└── birthdays.txt
```

**5. Run the app**

```bash
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

On first startup, the app will:
1. Create the Redis vector index
2. Embed and index all documents from `wiki_docs/`

On subsequent startups, indexing is skipped unless the documents have changed (detected via file modification times).

## Using the chat

Open `ws_frontend.html` in a browser:

```bash
open ws_frontend.html
```

Enter a User ID, click **Connect**, then ask questions about your wiki documents.

## Document updates

Simply add, edit, or remove files in `wiki_docs/` and restart the app. The change is detected automatically — the index is wiped and rebuilt, and the LLM cache is cleared.

## Configuration

All settings are in `configs.py`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `gpt-3.5-turbo` | OpenAI chat model |
| `TEMPERATURE` | `0` | LLM temperature |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model (3072 dims) |
| `DOCS_PATH` | `./wiki_docs` | Path to wiki documents |
| `SYSTEM_PROMPT` | *(see configs.py)* | Instruction prompt sent to the LLM on every request |
