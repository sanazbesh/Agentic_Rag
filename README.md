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


## Legal RAG architecture and persistent data pipeline

### Overview
This repository implements a hybrid, local-first legal RAG system using explicit orchestration graphs rather than a free-form agent loop. Retrieval and answer generation are deterministic stage flows with typed state, with optional local LLM calls used inside bounded graph nodes.

Persistent ingestion is a core part of the design: document state is stored and versioned across runs, instead of relying on session-only indexing. This keeps retrieval reproducible, enables re-indexing and deletion workflows, and supports traceable evaluation.

### High-level architecture (layered)
- **UI layer (Streamlit multi-page app):** inspection and operations surfaces (query inspection, dashboards, trace/debug views, triage/review workflows).
- **Backend adapter:** strict UI/backend boundary with mock vs real backend wiring, plus enforced final answer schema for stable rendering contracts.
- **Orchestration graphs:**
  - retrieval-stage graph with typed retrieval state
  - answer-stage graph that extends retrieval state
  - explicit path: query understanding → retrieval → answerability gate → synthesis
- **Ingestion/data pipeline:** offline ingestion orchestration for registration, parsing, chunking, persistence, indexing, validation, lifecycle status changes, and job tracking.
- **Storage layer:** Postgres (structured ingestion/retrieval state), Qdrant (dense vectors), and local file storage (raw source files).
- **Observability/evaluation layer:** structured stage spans, metrics, offline eval pipelines, CI gating, and failure triage workflows.

### Ingestion and persistent data pipeline
The persistent ingestion flow is coordinated by `IngestionOrchestrator` and related services:

1. Upload document.
2. Store raw file in local document storage (`LocalDocumentStore`).
3. Register document identity and create/reuse a document version in Postgres (`DocumentRegistry`, hash-based).
4. Parse the file (format-specific ingestors).
5. Create parent/child chunks.
6. Persist chunks in Postgres.
7. Embed child chunks.
8. Upsert child vectors into Qdrant.
9. Run ingestion validation checks.
10. Mark the version `READY` and promote it as current only after validation succeeds.

This ingestion path is separate from the online query path: ingestion writes/updates persistent state; query execution reads from that state.

### Storage responsibilities
- **Postgres:** source-of-truth structured state (`documents`, `document_versions`, `chunks`, `ingestion_jobs`, plus ingestion metadata and lifecycle statuses).
- **Qdrant:** dense vector index for child chunks and vector payload metadata used during retrieval.
- **Local file storage:** persisted raw uploaded files used for ingestion, re-indexing, and recovery operations.

### Runtime query flow
At runtime, query orchestration follows a graph-based retrieval/answer pipeline:

1. Query input.
2. Query understanding/routing (with optional rewrite/entity extraction).
3. Hybrid retrieval (dense + sparse/BM25) and reranking.
4. Qdrant returns child hits (point payload/IDs).
5. Resolve child chunks from Postgres (Qdrant → Postgres resolver).
6. Parent expansion for broader legal context.
7. Context compression when context size thresholds are exceeded.
8. Answerability/sufficiency gate.
9. Grounded answer generation with citations.

### Reliability and production-oriented design
The system includes production-oriented reliability features:
- idempotent ingestion via content hashing
- document versioning with explicit lifecycle states
- ingestion job tracking and retry support
- re-indexing existing document versions
- safe deletion workflows
- ingestion validation before a version can be marked `READY`
- trace metadata and stage-level spans for reproducibility/debugging
- strict separation of offline ingestion from online query execution

### Local development
Use Docker Compose for local development with persistent services:
- app
- Postgres
- Qdrant

The Compose stack uses persistent volumes so database state, vectors, and stored documents survive container restarts.

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
