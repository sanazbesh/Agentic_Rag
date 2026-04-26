# Expert Demo Report: Evaluation + Observability Stack

## 1) Executive summary

This project implements a **contract-driven quality system** for a legal RAG pipeline, not a collection of ad hoc logs. The stack couples deterministic checks, model-based grading, structured tracing, and release gates so each candidate can be measured before and during rollout. The goal is to make release decisions evidence-based and reversible.  

What this layer supports in practice:
- **Pre-release go/no-go** based on contract metrics and family-level regressions.
- **Layer-localized debugging** when quality drops (routing/retrieval/rerank/answerability/synthesis).
- **Live risk control** through sampled traffic + shadow grading + rollback thresholds.

The system is intentionally solo-project sized, but governance is still production-minded: explicit thresholds, stage gates, and rollback criteria are all documented and enforced by tooling.  

---

## 2) System architecture for eval + observability

### 2.1 Architecture map (offline + runtime)

```text
                         ┌──────────────────────────────────────┐
                         │ Offline eval datasets (JSONL)        │
                         │ evals/datasets/*.jsonl               │
                         └──────────────────────────────────────┘
                                            │
                                            ▼
                         ┌──────────────────────────────────────┐
                         │ Offline eval runner                  │
                         │ evals/runners/run_offline_eval.py    │
                         └──────────────────────────────────────┘
                                            │
      ┌─────────────────────────────────────┼─────────────────────────────────────┐
      ▼                                     ▼                                     ▼
┌───────────────┐                    ┌───────────────┐                    ┌────────────────┐
│Deterministic  │                    │Model-based    │                    │Run artifacts   │
│graders        │                    │LLM judges     │                    │evals/runs/*.json│
│contract/citation│                  │grounded/safe  │                    └────────────────┘
│answerability/etc│                  │failure/etc    │
└───────────────┘                    └───────────────┘
                                            │
                                            ▼
                               ┌──────────────────────────┐
                               │Reporting + comparison    │
                               │build_report / compare_runs│
                               └──────────────────────────┘
                                            │
                                            ▼
                               ┌──────────────────────────┐
                               │Release gates + checklist │
                               │merge/staging/ramp policy │
                               └──────────────────────────┘

Runtime path (same quality model, different timing)

User request → structured trace spans → request metrics record
            → optional production traffic sampling
            → optional online shadow grading linked to trace_id
            → dashboards/triage + rollback decisions
```

### 2.2 Why this is a layered system (not logging utilities)

- **Contract-first policy layer:** explicit thresholds and hard blockers are defined in `docs/quality_contract.md` and `docs/release_gates.md`.
- **Measurement layer:** deterministic graders + optional LLM judges are executed in offline and shadow contexts.
- **Trace contract layer:** the trace schema defines required top-level fields and required spans with stable stage semantics.
- **Runtime observability layer:** per-request structured metrics, traffic sampling, and shadow grading artifacts include version identifiers and trace linkage.
- **Decision layer:** production ramp plan + rollback criteria convert metrics into explicit release actions.

---

## 3) Quality model and score layers

The quality model intentionally prevents “single blended score” blindness.

### 3.1 Core dimensions tracked

1. **Family-level quality** (pass rates per legal family).  
2. **Citation correctness** (support-match correctness + missing-citation control).  
3. **False confident behavior** (hard-blocked increases).  
4. **Safe-failure behavior** (insufficient evidence must defer safely).  
5. **Latency/cost** (warning or rollback bands by stage).  
6. **Output contract checks** (malformed output is release-relevant).

### 3.2 How measured

- Deterministic evaluators: `contract_checks`, `citation_checks`, `answerability_checks`, `retrieval_checks`, `family_routing`.
- Model-based judges (when configured): groundedness + safe failure.
- Aggregations and trends: family pass rates, false-confident rate, citation correctness trend, safe-failure trend.

This ensures the stack can separately answer:
- “Did quality regress?”
- “In which family?”
- “At what pipeline layer?”
- “Is this severe enough to block/revert?”

---

## 4) Sample trace walkthrough (layer-localized)

### 4.1 Trace contract and stage boundaries

The trace schema and tracing helpers require a stable lifecycle across stages: query understanding, decomposition, retrieval, rerank, parent expansion, answerability, final synthesis. Each span has `status`, `inputs_summary`, `outputs_summary`, warnings, and error fields.

### 4.2 Concrete request walkthrough

Using the structured tracing test scenario (`"Who is the employer?"`), the trace captures:

1. **Query understanding**  
   - Family routing note resolves to `party_role_verification`.
2. **Retrieval**  
   - Child candidates returned with IDs.
3. **Rerank**  
   - Ranked child IDs recorded, linked to retrieval span (`parent_span_id`).
4. **Parent expansion**  
   - Parent chunk mapping retained for child→parent traceability.
5. **Answerability**  
   - `sufficient_context`, `support_level`, `insufficiency_reason` are explicit.
6. **Answer generation / final synthesis**  
   - Final status, grounding flag, sufficient-context flag, and citation count are preserved.

### 4.3 Failure localization example

In the “no results” scenario from the same tracing tests:
- retrieval returns zero child chunks,
- rerank becomes `skipped`,
- answer generation becomes `skipped`,
- answerability records insufficient context.

That means the failure is localizable upstream (retrieval/coverage), not vaguely “bad model output.”

---

## 5) Family-level scorecards

Family slicing is first-class across reporting paths.

### 5.1 Scorecard structure

Family-level pass rates are computed per run and emitted as rows with:
- `family`
- `case_count`
- `pass_rate`
- `fail_count`
- `fail_rate`

### 5.2 Why this matters

Release policy explicitly protects Tier 1 families (`party_role_verification`, `chronology_date_event`, `employment_lifecycle`) with tighter tolerances than aggregate metrics. A globally "good" release can still be blocked by a Tier 1 drop.

---

## 6) Regression catching example

A concrete baseline-vs-candidate regression case is already encoded in the compare-runs tests:

- Baseline (2026-04-21): chronology case passes, no false confident/citation failure.
- Candidate (2026-04-22): chronology case fails, `false_positive=True`, citation check fails, plus slower latency and higher cost.

Comparison logic marks this as:
- **risk regression** (false confident and citation correctness directions),
- **family regression** in chronology,
- **performance regression** (latency/cost direction).

This is exactly what pre-release gating needs: one diff exposing quality + risk + efficiency deltas before traffic exposure.

---

## 7) Live vs offline monitoring

## 7.1 Offline: regression confidence before release
- Dataset-backed runs produce standardized JSON artifacts.
- Build/compare reports quantify pass rates, family deltas, and failure breakdowns.

## 7.2 Live: sampled runtime quality signals
- Production traffic sampling persists JSONL records with:
  - request/trace IDs,
  - family,
  - sampling reasons,
  - debug payload fragment,
  - version identifiers.
- Sampling reasons include random sampling, high-risk family targeting, low-confidence signals, and high-cost triggers.

## 7.3 Bridge: shadow grading linked to traces
- Sampled records can be shadow-graded asynchronously.
- Shadow results persist deterministic/model checks and link back via `trace_id`.
- A trace-link index keeps latest shadow eval IDs per trace.

This closes the loop: offline quality model -> runtime samples -> graded runtime evidence -> triage/rollback decisions.

---

## 8) Release gating flow

```text
Stage 0 Offline candidate approval
  ├─ Run offline eval (baseline + candidate)
  ├─ Compare runs (quality/risk/perf + family deltas)
  └─ Block if hard threshold violated

Stage 1 Staging validation
  ├─ Smoke-query coverage (incl. Tier 1 families)
  ├─ Trace contract presence + debug fields
  └─ Block on malformed outputs or severe regressions

Stage 2 Limited live sample pass
  ├─ Minimum sampled review volume
  ├─ False-confident/citation/family stability checks
  └─ Rollback if rollback-band triggers fire

Stage 3 Broader rollout
  ├─ Expand exposure windows gradually
  ├─ Keep sampled trace review and metric checks
  └─ GO / GO_WITH_WARNINGS / NO_GO decision
```

All stages are tied to explicit threshold tables and rollback triggers, not discretionary judgment.

---

## 9) Why this is production quality (explicit acceptance check)

### 9.1 “This is not ad hoc logging”

- There is a documented trace schema contract with required spans/fields.
- Metrics and eval outputs have stable data shapes and deterministic aggregators.
- Release artifacts (`build_report`, `compare_runs`) consume those stable structures.

### 9.2 “This is a production quality system”

- Hard-blocker thresholds, warning bands, and rollback criteria are numerically defined.
- The ramp process requires staged evidence and explicit release-owner decisions.
- Runtime sampling + shadow grading provide post-deploy quality controls.

### 9.3 “Failures can be localized to a specific layer”

- Failure taxonomy enforces earliest-dominant root-cause labeling.
- Trace drilldown computes stage statuses and derives likely failure layer.
- Span-level outputs (retrieval counts, rerank outputs, answerability sufficiency) make upstream/downstream attribution concrete.

---

## Appendix A — Real repo artifacts used in this report

- Governance + gates: `docs/quality_contract.md`, `docs/release_gates.md`, `docs/rollback_criteria.md`, `docs/production_ramp_plan.md`, `docs/release_checklist.md`.
- Trace contract + runtime tracing: `observability/schema/trace_schema.md`, `src/agentic_rag/orchestration/tracing.py`.
- Structured metrics/version attribution: `src/agentic_rag/orchestration/metrics.py`, `src/agentic_rag/versioning.py`.
- Offline reporting/comparison: `evals/reports/build_report.py`, `evals/reports/compare_runs.py`, `evals/reports/quality_dashboard_data.py`.
- Trace debugging: `evals/reports/trace_dashboard_data.py`, `ui/trace_dashboard.py`.
- Live linkage: `src/agentic_rag/orchestration/traffic_sampling.py`, `src/agentic_rag/orchestration/online_shadow_grading.py`.
- Concrete examples referenced: `tests/test_structured_tracing_ticket20.py`, `tests/test_compare_runs_ticket29.py`, `tests/test_quality_dashboard_data.py`, `tests/test_production_traffic_sampling_ticket26.py`, `tests/test_online_shadow_grading_ticket27.py`.
