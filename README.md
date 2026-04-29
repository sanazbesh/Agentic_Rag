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

## Backup and restore (local persistent RAG state)

Ticket 10.2 adds local scripts to back up and restore the persisted RAG knowledge-base state used by Docker Compose.

What is included in backups:
- Postgres database dump (`agentic_rag`).
- Qdrant collection snapshots when available (with raw Qdrant storage export fallback).
- Stored document files from `DOCUMENT_STORAGE_PATH` (`/app/data/documents` in Compose).

Create a timestamped backup:

```bash
./scripts/backup_rag_state.sh
```

Optional backup root folder (default is `./backups`):

```bash
./scripts/backup_rag_state.sh /path/to/backups
```

The script creates `backups/YYYYMMDD_HHMMSS/` and fails clearly if required Docker services are not running.

Restore from a specific backup folder:

```bash
./scripts/restore_rag_state.sh backups/YYYYMMDD_HHMMSS
```

Restore behavior:
- Restores Postgres schema/data from the backup dump.
- Restores Qdrant from snapshots when present, otherwise restores raw Qdrant storage export.
- Restores stored document files.

Safety checks:
- Fails if Docker services (`postgres`, `qdrant`, `app`) are not running.
- Fails if the provided restore backup path does not exist.
- Backup script avoids overwriting existing timestamped backup folders.
