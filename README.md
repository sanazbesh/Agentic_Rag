# Agentic_Rag

Base scaffold for an agentic RAG system.

## Project structure

```text
src/agentic_rag/
  config/
  evaluation/
  indexing/
  ingestion/
  orchestration/
  prompts/
  retrieval/
  tools/
  types.py
app.py
ui/
```

Each module currently includes abstract base interfaces to help you plug in concrete implementations incrementally.

## Streamlit legal RAG test UI

A local-first inspection dashboard is available for testing the legal RAG pipeline:

```bash
streamlit run app.py
```

The UI supports:
- strict final result rendering (`answer_text`, `grounded`, `sufficient_context`, `citations`, `warnings`)
- mock backend mode for immediate local testing
- a clean adapter boundary for wiring your real `run_legal_rag_turn(...)` runner
- expandable debug payload inspection panels

## Persistence foundation (Postgres)

For upcoming persistent ingestion pipeline work, Postgres connection settings are environment-driven:

- `DATABASE_URL` (required to initialize engine/session factory)
- `AGENTIC_RAG_DB_ECHO` (optional; `true/false`, default `false`)


## Docker Compose local stack

For a local production-like setup (app + Postgres + Qdrant), run:

```bash
docker compose up --build
```

Services and local defaults:
- `app` (Streamlit UI on `http://localhost:8501`)
- `postgres` (`postgres:16`)
- `qdrant` (`qdrant/qdrant`)

Configured environment variables in Compose:
- `DATABASE_URL`
- `QDRANT_URL`
- `QDRANT_COLLECTION_NAME`
- `DOCUMENT_STORAGE_PATH`

Persistent named volumes:
- `postgres_data`
- `qdrant_data`
- `documents_data`
