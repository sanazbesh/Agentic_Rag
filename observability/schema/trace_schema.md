# Legal RAG Trace Schema (Solo Portfolio Project)

## 1) Purpose

This document defines a single trace schema contract for the legal RAG pipeline.

The schema exists to make one request observable end to end, from query understanding to final synthesis, with a stable structure that supports:
- debugging failed or degraded runs,
- offline evaluation review,
- regression analysis across releases, and
- disciplined release decisions.

Stable trace structure is required so traces can be compared across runs without brittle ad hoc parsing.

---

## 2) Scope

This schema covers the full legal RAG request lifecycle for one user query.

It defines the tracing **contract** (field names, required spans, required/optional fields, status conventions, and stability rules).

It does **not** define instrumentation code, storage, dashboards, alerting, or vendor tooling.

---

## 3) Trace Design Principles

1. **One request -> one trace.**
2. **One major pipeline stage -> one span.**
3. **Stable field names over ad hoc debug blobs.**
4. **Readable by a human and machine-usable for later analysis.**
5. **Useful for both offline eval review and future production observability.**
6. **Small and maintainable by one builder.**
7. **Contract-first:** missing data uses stable null/empty conventions instead of dropping fields.

---

## 4) Trace-Level Fields (Top-Level Contract)

All trace records must include the following top-level fields.

| Field | Type | Required | Notes |
|---|---|---:|---|
| `trace_id` | string | Yes | Unique ID for this trace. |
| `request_id` | string | Yes | Stable per-run request identifier. |
| `timestamp_utc` | string (ISO-8601) | Yes | Trace creation timestamp. |
| `query` | string | Yes | Original user query. |
| `selected_document_ids` | list[string] | Yes | Selected document scope; empty list if none. |
| `active_family` | string \| null | Yes | Active legal question family when available; `null` if unknown. |
| `overall_status` | enum | Yes | One of: `success`, `partial`, `failed`. |
| `total_latency_ms` | integer \| null | Yes | End-to-end latency when captured; `null` if unavailable. |
| `schema_version` | string | Yes | Version of this trace contract (for example `trace.v1`). |
| `pipeline_version` | string \| null | Yes | Pipeline/release identifier if available. |
| `spans` | list[span] | Yes | Ordered list containing required stage spans. |

---

## 5) Standard Span Structure (Applies to Every Span)

Every span must follow this structure.

| Field | Type | Required | Notes |
|---|---|---:|---|
| `span_name` | string | Yes | Human-readable stage span name. |
| `stage` | enum | Yes | Controlled stage key (see required spans below). |
| `start_time_utc` | string (ISO-8601) \| null | Yes | Start timestamp if captured; `null` if unavailable. |
| `end_time_utc` | string (ISO-8601) \| null | Yes | End timestamp if captured; `null` if unavailable. |
| `duration_ms` | integer \| null | Yes | Duration if computed; `null` if unavailable. |
| `status` | enum | Yes | One of: `success`, `partial`, `failed`, `skipped`. |
| `inputs_summary` | object | Yes | Stable summary of key inputs for the stage. |
| `outputs_summary` | object | Yes | Stable summary of key outputs for the stage. |
| `warnings` | list[object] | Yes | Stage warnings (machine-readable entries), empty list if none. |
| `error` | object \| null | Yes | Error payload on failure; otherwise `null`. |

### Warning object shape

Each warning entry should use:
- `code` (string, required),
- `message` (string, required),
- `severity` (enum: `low`, `medium`, `high`, required).

### Error object shape

`error` should use:
- `code` (string, required when error is not null),
- `message` (string, required when error is not null).

---

## 6) Required Spans (Core Schema)

The following seven spans are required and form the core contract for every request trace:

1. `query_understanding`
2. `decomposition`
3. `retrieval`
4. `rerank`
5. `parent_expansion`
6. `answerability`
7. `final_synthesis`

> Optional future spans may be added later (for example query rewrite or context compression), but these seven are the required baseline contract.

### A) Query Understanding Span

- **Purpose:** capture deterministic classification/routing decisions before retrieval.
- **Starts:** when query classification begins.
- **Ends:** when structured query understanding output is finalized.

**Required fields (`outputs_summary`)**
- `normalized_query` (string)
- `question_type` (string)
- `answerability_expectation` (string)
- `legal_question_family` (string | null)
- `is_followup` (boolean)
- `is_document_scoped` (boolean)
- `may_need_decomposition` (boolean)

**Optional fields**
- `resolved_document_hints` (list[string])
- `resolved_topic_hints` (list[string])
- `resolved_clause_hints` (list[string])
- `routing_notes` (list[string])
- `ambiguity_notes` (list[string])

**Common warnings/errors**
- warnings: `context_reference_ambiguous`, `query_understanding_fallback`
- errors: `query_understanding_failed`

**Stability expectation**
- Keep classification field names and meanings fixed; do not rename without schema version bump.

### B) Decomposition Span

- **Purpose:** record whether decomposition was required and its validated plan outcome.
- **Starts:** after query understanding completes.
- **Ends:** once decomposition decision + plan validation complete.

**Required fields (`outputs_summary`)**
- `needs_decomposition` (boolean)
- `decomposition_gate_reasons` (list[string])
- `subquery_count` (integer)
- `validation_outcome` (enum: `valid`, `invalid`, `not_applicable`)
- `validation_errors` (list[string])

**Optional fields**
- `strategy` (string | null)
- `subquery_ids` (list[string])

**Common warnings/errors**
- warnings: `decomposition_validation_failed`, `decomposition_plan_simplified`
- errors: `decomposition_failed`

**Stability expectation**
- `needs_decomposition` must always come from runtime gate decision, not inferred from hints.

### C) Retrieval Span

- **Purpose:** capture child-chunk evidence retrieval behavior.
- **Starts:** when effective retrieval query + filters are finalized.
- **Ends:** when child candidates are returned.

**Required fields (`outputs_summary`)**
- `effective_query` (string)
- `selected_document_scope` (list[string])
- `retrieval_mode` (string)
- `retrieved_child_count` (integer)
- `top_child_chunk_ids` (list[string])

**Optional fields**
- `filters` (object | null)
- `family_retrieval_notes` (list[string])
- `subquery_retrieval_counts` (object)

**Common warnings/errors**
- warnings: `retrieval_empty`, `retrieval_scope_too_narrow`
- errors: `retrieval_failed`

**Stability expectation**
- top child IDs should preserve rank order and use stable identifiers.

### D) Rerank Span

- **Purpose:** record reranking from retrieved children to prioritized evidence.
- **Starts:** when rerank receives candidate list.
- **Ends:** when reranked ordering is finalized.

**Required fields (`outputs_summary`)**
- `input_candidate_count` (integer)
- `output_candidate_count` (integer)
- `top_reranked_child_ids` (list[string])
- `ranking_source` (string)

**Optional fields**
- `rerank_model` (string | null)
- `degraded_mode` (boolean)

**Common warnings/errors**
- warnings: `rerank_skipped`, `rerank_degraded`
- errors: `rerank_failed`

**Stability expectation**
- count semantics remain fixed (`input` = pre-rerank count, `output` = post-rerank count).

### E) Parent Expansion Span

- **Purpose:** capture child->parent expansion and traceability.
- **Starts:** when reranked child IDs are selected for expansion.
- **Ends:** when parent chunks are fetched and ordered.

**Required fields (`outputs_summary`)**
- `collected_parent_ids` (list[string])
- `fetched_parent_count` (integer)
- `child_parent_traceability` (object)
- `ordering_notes` (list[string])

**Optional fields**
- `missing_parent_ids` (list[string])
- `compressed_context_count` (integer | null)

**Common warnings/errors**
- warnings: `parent_chunk_missing`, `parent_expansion_partial`
- errors: `parent_expansion_failed`

**Stability expectation**
- child-parent mapping format must remain consistent for auditability.

### F) Answerability Span

- **Purpose:** record support/sufficiency decision before final answer generation.
- **Starts:** when answerability evaluator receives resolved query + retrieval context.
- **Ends:** when should-answer decision is finalized.

**Required fields (`outputs_summary`)**
- `has_relevant_context` (boolean)
- `sufficient_context` (boolean)
- `partially_supported` (boolean)
- `support_level` (string)
- `insufficiency_reason` (string | null)
- `should_answer` (boolean)

**Optional fields**
- `matched_parent_chunk_ids` (list[string])
- `matched_headings` (list[string])
- `family_answerability_notes` (list[string])

**Common warnings/errors**
- warnings: `definition_not_supported`, `subquery_coverage_below_threshold`
- errors: `answerability_assessment_failed`

**Stability expectation**
- support decision fields are contract-critical and must keep stable meaning across releases.

### G) Final Synthesis Span

- **Purpose:** capture final response assembly and output status.
- **Starts:** when synthesis route is selected.
- **Ends:** when final answer payload is produced.

**Required fields (`outputs_summary`)**
- `final_answer_status` (enum: `answered`, `partial_answer`, `insufficient_context`, `failed_safe`)
- `grounded` (boolean)
- `citation_count` (integer)
- `warning_count` (integer)
- `synthesis_path` (string)
- `final_output_status` (enum: `success`, `partial`, `failed`)

**Optional fields**
- `response_route` (string)
- `final_answer_length_chars` (integer)

**Common warnings/errors**
- warnings: `insufficient_context`, `fallback_after_generation`
- errors: `answer_generation_failed`

**Stability expectation**
- final status fields must map cleanly to release/eval interpretation and not drift in meaning.

---

## 7) Stability Rules (Required)

In this project, **stable across runs** means:

1. **Required field names do not change casually.**
   - Rename/remove/add required fields only with explicit schema version update.

2. **Required spans are always present in deterministic order** for the same pipeline path.
   - If a stage is bypassed, span still exists with `status="skipped"` and stable empty/null outputs.

3. **Enums come from controlled sets.**
   - No free-form status strings.

4. **Optional missing data uses stable null/empty conventions.**
   - list -> `[]`, object -> `{}` when structurally expected, scalar -> `null`.

5. **Field meaning remains stable even if internals evolve.**
   - Example: `retrieved_child_count` must always mean the number of child candidates returned by retrieval.

6. **Warning and error codes are stable identifiers.**
   - Message text may improve, but `code` should remain machine-comparable.

---

## 8) Required vs Optional Fields

### Required for every trace
- all top-level fields in Section 4,
- `spans` containing all seven required stages,
- each span containing all standard fields in Section 5.

### Optional fields
- stage-specific optional fields in Section 6,
- extra diagnostic fields that do not redefine required semantics.

Rule: optional fields can be added safely, but required contract fields must remain backward-stable within a schema version.

---

## 9) Status and Warning Conventions

### Status values

Use only these status values:
- `success`
- `partial`
- `failed`
- `skipped` (span-level only)

Trace-level `overall_status` should be derived from span outcomes:
- `success`: all required spans `success` (or expected `skipped` where appropriate),
- `partial`: at least one required span `partial`, none `failed`,
- `failed`: any required span `failed`.

### Warning conventions

- warnings must be stage-specific whenever possible,
- warnings should be machine-readable (`code`) and human-readable (`message`),
- duplicate warning codes should be deduplicated within a span.

---

## 10) Example Trace Skeleton (Concise)

```json
{
  "trace_id": "tr_20260419_000123",
  "request_id": "req_9b7b",
  "timestamp_utc": "2026-04-19T14:21:33Z",
  "query": "Compare governing law and dispute resolution clauses.",
  "selected_document_ids": ["doc-msa-017"],
  "active_family": "issue_spotting",
  "overall_status": "success",
  "total_latency_ms": 1842,
  "schema_version": "trace.v1",
  "pipeline_version": "legal_rag.v0.19",
  "spans": [
    {
      "span_name": "Query Understanding",
      "stage": "query_understanding",
      "start_time_utc": "2026-04-19T14:21:33Z",
      "end_time_utc": "2026-04-19T14:21:33Z",
      "duration_ms": 42,
      "status": "success",
      "inputs_summary": {"query": "Compare governing law and dispute resolution clauses."},
      "outputs_summary": {
        "normalized_query": "compare governing law and dispute resolution clauses",
        "question_type": "comparison_query",
        "answerability_expectation": "comparison",
        "legal_question_family": "issue_spotting",
        "is_followup": false,
        "is_document_scoped": true,
        "may_need_decomposition": true
      },
      "warnings": [],
      "error": null
    },
    {"span_name": "Decomposition", "stage": "decomposition", "status": "success", "inputs_summary": {}, "outputs_summary": {"needs_decomposition": true, "decomposition_gate_reasons": ["comparison_query"], "subquery_count": 2, "validation_outcome": "valid", "validation_errors": []}, "warnings": [], "error": null, "start_time_utc": null, "end_time_utc": null, "duration_ms": null},
    {"span_name": "Retrieval", "stage": "retrieval", "status": "success", "inputs_summary": {}, "outputs_summary": {"effective_query": "compare governing law and dispute resolution clauses", "selected_document_scope": ["doc-msa-017"], "retrieval_mode": "hybrid", "retrieved_child_count": 12, "top_child_chunk_ids": ["c183", "c311"]}, "warnings": [], "error": null, "start_time_utc": null, "end_time_utc": null, "duration_ms": null},
    {"span_name": "Rerank", "stage": "rerank", "status": "success", "inputs_summary": {}, "outputs_summary": {"input_candidate_count": 12, "output_candidate_count": 8, "top_reranked_child_ids": ["c183", "c311"], "ranking_source": "cross_encoder"}, "warnings": [], "error": null, "start_time_utc": null, "end_time_utc": null, "duration_ms": null},
    {"span_name": "Parent Expansion", "stage": "parent_expansion", "status": "success", "inputs_summary": {}, "outputs_summary": {"collected_parent_ids": ["p45", "p51"], "fetched_parent_count": 2, "child_parent_traceability": {"c183": "p45", "c311": "p51"}, "ordering_notes": []}, "warnings": [], "error": null, "start_time_utc": null, "end_time_utc": null, "duration_ms": null},
    {"span_name": "Answerability", "stage": "answerability", "status": "success", "inputs_summary": {}, "outputs_summary": {"has_relevant_context": true, "sufficient_context": true, "partially_supported": false, "support_level": "strong", "insufficiency_reason": null, "should_answer": true}, "warnings": [], "error": null, "start_time_utc": null, "end_time_utc": null, "duration_ms": null},
    {"span_name": "Final Synthesis", "stage": "final_synthesis", "status": "success", "inputs_summary": {}, "outputs_summary": {"final_answer_status": "answered", "grounded": true, "citation_count": 2, "warning_count": 0, "synthesis_path": "generate_answer", "final_output_status": "success"}, "warnings": [], "error": null, "start_time_utc": null, "end_time_utc": null, "duration_ms": null}
  ]
}
```

---

## 11) How This Schema Is Used

For this solo project, the same builder designs the schema, inspects traces, and uses them in evaluation and release review.

The schema supports:
- **debugging failed runs:** exact stage + warning/error code localization,
- **offline eval review:** stable extraction of retrieval, answerability, and synthesis outcomes,
- **regression analysis:** before/after comparisons by stage and family,
- **family-level error analysis:** track where specific legal families break (retrieval vs answerability vs synthesis),
- **future instrumentation tickets:** implementation can be added later without redefining semantics.

---

## 12) Non-Goals

This document does **not**:
- implement tracing code,
- define dashboards,
- define trace storage backend,
- define alerting/monitoring policy,
- require vendor-specific instrumentation.

It only defines the stable tracing contract for the current legal RAG pipeline.
