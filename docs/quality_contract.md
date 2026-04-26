# Legal RAG Quality Contract (Solo Portfolio Project)

## 1) Purpose

This document defines the quality contract for this legal RAG system: what “good” looks like, how it is measured, and what must pass before release.

This contract governs three operating contexts:
- offline evaluation (dataset-based checks before release),
- staging/release checks (go/no-go gates), and
- online production monitoring (live behavior and drift detection).

This is a **portfolio project run by one builder**. The governance model is intentionally lightweight but strict on legal-safety outcomes.

---

## 2) Scope

This contract covers the full answer path from user query to final response:
- query understanding and routing,
- retrieval (child and parent chunk recall/relevance),
- answerability/sufficiency decision,
- answer generation,
- citations attached to claims,
- safe-failure behavior when evidence is weak,
- runtime performance (latency), and
- request-level cost stability.

Out of scope: implementation details of any specific runner, dashboard, tracer, or evaluator tooling.

---

## 3) Core Quality Principles

1. **Grounded over fluent**  
   A less polished answer backed by cited evidence is better than a polished unsupported answer.

2. **Safe failure over unsupported confidence**  
   If evidence is insufficient, the system should explicitly say so and guide next steps.

3. **Responsive evidence over generic relevance**  
   Retrieval quality is judged by support for the exact legal question, not broad topical overlap.

4. **Traceable decisions**  
   The path from question → retrieval → answerability decision → final answer must be auditable.

5. **Family-specific correctness**  
   Quality must hold across legal question families; global averages alone are not acceptable.

---

## 4) Legal Question Family Framing

This legal RAG is evaluated by **question family**, not only by aggregate score. Release decisions require both:
- global metrics at/above contract thresholds, and
- no major family-specific collapse.

Primary families used for quality slicing:
- chronology / timeline reconstruction,
- policy or contract interpretation,
- compliance obligation extraction,
- correspondence and dispute context,
- financial entitlement / calculation support,
- mitigation and remedial-action guidance,
- issue spotting and risk flagging.

A release candidate that meets global averages but fails materially in one family is considered incomplete.

---

## 5) Core Quality Dimensions and Thresholds

> Roles are intentionally separated for clarity, but in this portfolio project all roles are performed by the same person.

| Dimension | Definition | Why it matters | Measurement approach | Owner role | Threshold | Classification |
|---|---|---|---|---|---|---|
| **Groundedness** | Share of answerable responses whose material claims are supported by retrieved evidence. | Prevents hallucinated legal statements. | **Grounded answer rate** on offline eval set and monitored samples. | Quality owner | **>= 92%** grounded answer rate | **Hard blocker** |
| **Citation correctness** | Citations point to the correct source span for supported claims and are present when required. | Legal answers must be verifiable. | (1) **Citation correctness >= 98%** on cited claims. (2) **Missing citation rate on grounded answers = 0%**. | Quality owner | As stated | **Hard blocker** |
| **Answerability correctness** | Correctly decides whether available evidence is sufficient to answer. | Prevents over-answering when context is weak. | (1) **Answerability decision accuracy >= 93%**. (2) **False sufficient rate <= 3%**. | Quality owner | As stated | **Hard blocker** |
| **Safe failure quality** | Quality of refusal/deferral when evidence is insufficient (clear limit statement + next best step). | Reduces unsafe confidence and improves user trust. | **Safe failure quality score >= 4.3/5** via rubric grading + manual spot review. | Project owner (as quality owner) | >= 4.3/5 | **Hard blocker** |
| **Retrieval relevance** | Ability to retrieve necessary legal evidence at child and parent levels. | Upstream cap on answer quality. | (1) **Gold child recall@K >= 90%**. (2) **Gold parent recall >= 95%**. | Retrieval owner | As stated | **Hard blocker** |
| **Latency** | End-to-end response time for completed requests. | Production usability. | Track distribution on representative workload: **p95 <= 8.0s**, **p99 <= 12.0s**. | Project owner | p95 <= 8.0s, p99 <= 12.0s | **Warning-only** |
| **Cost** | Average cost per request stability relative to baseline release. | Keeps system sustainable and predictable. | **Average cost/request drift <= +15%** vs pinned baseline under comparable traffic mix. | Project owner | <= +15% drift | **Warning-only** |

### Additional explicit safety metric

| Metric | Definition | Threshold | Classification |
|---|---|---|---|
| **False confident answer rate** | Fraction of responses that present unsupported claims with confident framing. | **<= 2%** | **Hard blocker** |

---

## 6) Measurement Layers

Quality is assessed through layered evidence:

1. **Deterministic checks**  
   Structural and rule-based checks (citation presence, schema conformance, required fields, latency/cost aggregation).

2. **Retrieval checks**  
   Recall/relevance checks against labeled gold child/parent evidence.

3. **Model-based grading**  
   Rubric-based grading for groundedness, answerability, and safe-failure quality.

4. **Human review (manual, by project owner)**  
   Targeted spot review of failures, family outliers, and release-boundary cases.

If layers disagree, deterministic and retrieval evidence take precedence; manual owner review resolves final gate decisions.

---

## 7) Evaluation Modes

### Offline evaluation (pre-release)
- Run contract metrics on a versioned evaluation set with family-level slices.
- Produce a release report with thresholds, deltas vs previous version, and failure examples.
- This mode is required for go/no-go decisions.

### Online evaluation / production monitoring
- Monitor live metrics for groundedness proxies, latency, cost, and sampled quality checks.
- Use alert thresholds aligned with this contract.
- Online monitoring informs rollback/review decisions and future offline dataset updates.

> Note: This contract defines required behavior and expected monitoring outputs. Specific tooling may evolve over time.

---

## 8) Release Gates

### Hard blockers (automatic no-release)
A release is blocked if **any** hard-blocker threshold is missed:
- grounded answer rate
- false confident answer rate
- citation correctness
- missing citation rate on grounded answers
- answerability decision accuracy
- false sufficient rate
- safe failure quality score
- gold child recall@K
- gold parent recall

### Warning-only conditions (release allowed with explicit review note)
- p95 latency miss
- p99 latency miss
- average cost/request drift above threshold

Warning-only issues require documented rationale and a remediation plan in release notes, but do not automatically stop release.

---

## 9) Reporting Requirements

Each evaluated version / release candidate must report:
1. version identifier (code commit/tag),
2. evaluation date,
3. dataset/eval slice identifier,
4. all contract metrics with threshold and pass/fail status,
5. family-level metric breakdown,
6. regression delta vs prior accepted baseline,
7. top failure examples (groundedness, answerability, citations, safe failure),
8. latency and cost summary,
9. final gate decision: **Pass**, **Pass with warnings**, or **Blocked**.

---

## 10) Solo Ownership Model

This project uses three role labels:
- **Project owner**
- **Retrieval owner**
- **Quality owner**

In this portfolio project, all three roles are performed by the same person. The split is used to keep responsibilities explicit, not to imply multiple team members.

---

## 11) Definition of Production-Ready

This legal RAG is production-ready when the current release candidate passes all hard-blocker thresholds, has only acceptable warning-level deviations (if any), and includes a complete evaluation report with family-level results and traceable evidence for quality claims.
