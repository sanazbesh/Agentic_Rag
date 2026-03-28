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
