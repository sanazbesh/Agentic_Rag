# Legal RAG Failure Taxonomy (Solo Portfolio Project)

## 1) Purpose

This taxonomy defines how failures are labeled in this legal RAG so analysis is precise, repeatable, and useful for engineering decisions.

It replaces vague labels such as “the RAG failed” with structured triage. For every failed case, the project owner (acting as evaluator and reviewer) records:
- **exactly one primary failure class** (root-cause layer),
- **one or more legal-family tags** (legal-domain area), and
- **optional secondary notes** (symptoms, contributing factors, or edge conditions).

The objective is operational: faster debugging, cleaner regression tracking, and clearer release decisions.

---

## 2) Scope

This taxonomy covers the end-to-end answer path:
- legal family routing / intent framing,
- retrieval,
- reranking,
- answerability decision,
- citation attachment and citation correctness,
- synthesis of final response,
- safe-failure response wording.

Out of scope:
- implementation details of evaluators, dashboards, tracing, or data pipelines,
- broader product/UX concerns not tied to a failed answer case.

---

## 3) Classification Principles

1. **One failed case, one primary failure class.**
2. **Primary class = earliest dominant root-cause layer**, not the most visible downstream symptom.
3. **Legal-family tags are separate from failure class.**
4. **Secondary symptoms may be noted**, but do not replace the single primary class.
5. **Safe failure itself is acceptable behavior** when evidence is insufficient; it is only labeled as failure when the decision or wording is wrong.
6. **Classify for actionability.** The label should indicate the first layer to fix.

---

## 4) Primary Failure Classes

These classes are the required backbone of failure triage.

| Primary class | Definition | When to use | When not to use | Common symptoms | Example scenario | Pipeline layer |
|---|---|---|---|---|---|---|
| **family misclassification** | Query is routed/framed under the wrong legal family, leading to incorrect downstream retrieval or reasoning path. | Evidence shows the case should have been handled under a different family (e.g., party/role vs chronology). | If family routing is correct but retrieval quality is poor; then use retrieval/rerank class. | Irrelevant family-specific context, repeated misses on otherwise straightforward prompts. | A party-role question is interpreted as timeline reconstruction, causing retrieval of date-heavy snippets instead of role clauses. | Query understanding / family routing |
| **retrieval miss** | Required evidence was never retrieved into candidate set. | Gold/expected supporting text is absent from retrieved set. | If correct evidence is present but ranked too low for use; then rerank miss. | `fact_not_found` despite existing support, topical but non-responsive chunks. | A party-role question retrieves generic contract boilerplate and returns `fact_not_found` even though role definitions exist in corpus. | Retrieval |
| **rerank miss** | Correct evidence is retrieved but not surfaced high enough for answer stage. | Correct source appears in retrieved candidates but is deprioritized. | If source never appears at all (retrieval miss). | Strong evidence present deep in list, answer built from weaker top snippets. | Correct indemnity clause appears at rank 18 but top-ranked snippets are adjacent, less relevant clauses. | Reranking |
| **answerability false positive** | System says answerable and gives a substantive answer when evidence is insufficient/unsupported. | Answer asserts legal fact/conclusion without adequate support. | If evidence is sufficient but synthesis misstates details; then synthesis failure. | Confident tone with weak citations, definition answered from token overlap only. | A definition query is answered confidently from title similarity rather than an actual definitional clause. | Answerability decision |
| **answerability false negative** | System says unanswerable (or refuses) even though sufficient evidence is available. | Retrieved/reranked evidence clearly supports a direct answer. | If the refusal is caused by missing evidence; then retrieval or rerank miss. | Unnecessary “insufficient evidence” responses. | Clear notice period language is present, but system still returns `fact_not_found`. | Answerability decision |
| **citation failure** | Citations are missing, incorrect, or do not support the associated claim. | Claim-to-source mapping is wrong, incomplete, or absent where required. | If core claim itself is wrong due to reasoning despite correct citations; then synthesis failure. | Broken citation links, section mismatch, unsupported quoted claim. | Final answer cites a compensation section for a termination-cause claim that is actually supported elsewhere. | Citation assembly / verification |
| **synthesis failure** | Retrieved evidence and answerability decision are adequate, but final response is logically incorrect, incomplete, or contradictory. | Subanswers/evidence are mostly right but combined output is wrong. | If incorrectness is primarily unsupported confidence; then answerability false positive. | Contradictions, omitted controlling condition, wrong final conclusion from correct snippets. | System retrieves correct eligibility rules but combines them into an incorrect entitlement conclusion. | Final synthesis |
| **safe failure wording issue** | Safe-failure decision is directionally correct, but wording is misleading, over-absolute, or unhelpful. | Refusal/deferral should happen, but phrasing creates legal misunderstanding risk. | If refusal itself is incorrect (evidence existed); then answerability false negative. | “No evidence exists” instead of “insufficient in provided documents”, missing next step guidance. | Unsupported query is declined, but response falsely implies the right cannot exist at all. | Safe-failure response generation |

---

## 5) Optional Supporting Classes (Use Sparingly)

These are optional and should only be used when they add clarity without overlapping the required classes.

| Supporting class | Purpose | Boundary rule |
|---|---|---|
| **document scope resolution failure** | Follow-up question fails because active document scope (selected file/segment) was not resolved correctly. | Prefer this only when scope resolution is a distinct system layer; otherwise map to retrieval miss with note `scope_resolution`. |
| **evidence selection failure** | Too many retrieved candidates are available, but evidence packing/selection into synthesis context is wrong. | Do not use when reranking already captures the same issue; rerank miss remains primary in most cases. |
| **prompt/formatting failure** | Output format or prompt contract breaks (e.g., malformed citation structure) independent of legal reasoning. | If malformed output causes unsupported claims/citation mismatch, choose the earlier dominant class and note formatting secondarily. |
| **evaluation contract failure** | Failure is in test harness/label contract mismatch rather than model behavior. | Use only for evaluator-side defects; never for genuine model response errors. |

If uncertain, default to required primary classes and add a short secondary note instead of creating taxonomy sprawl.

---

## 6) Legal-Family Tags

Legal-family tags describe **what legal area the case belongs to**, not why the system failed.

Minimum tag set:
- **party/role**
- **chronology**
- **employment lifecycle**
- **metadata**
- **litigation milestones**
- **mitigation**
- **financial/entitlement**
- **issue spotting**

### Tag definitions

| Family tag | Use for |
|---|---|
| **party/role** | Party identity, obligations by actor, role attribution, authority/responsibility mapping. |
| **chronology** | Timeline ordering, event sequencing, date-dependent reasoning. |
| **employment lifecycle** | Hiring, role changes, performance, leave, termination/resignation, post-employment clauses. |
| **metadata** | Document properties (source, version, date, author, jurisdiction marker, provenance). |
| **litigation milestones** | Claims, defenses, notice, filing, hearing, settlement, procedural posture events. |
| **mitigation** | Remedial actions, cure periods, mitigation duties, corrective obligations. |
| **financial/entitlement** | Compensation, benefits, damages, reimbursement, eligibility criteria, calculation dependencies. |
| **issue spotting** | Multi-issue extraction, risk flagging, potential non-compliance or conflict identification. |

### Critical distinction

- **Primary failure class** = root-cause layer of failure.
- **Legal-family tag** = legal domain of the question.

Example: **primary failure class = retrieval miss**; **family tag = party/role**.

---

## 7) Classification Decision Guide (2-Minute Workflow)

Apply this sequence in order. Stop at the first “yes” that identifies the earliest dominant root cause.

1. **Family routing check**  
   Was the query handled as the wrong legal family?  
   - Yes → **family misclassification**

2. **Retrieval coverage check**  
   Is the required supporting evidence absent from retrieved candidates?  
   - Yes → **retrieval miss**

3. **Rerank check**  
   Is required evidence retrieved but buried/deprioritized such that answer stage misses it?  
   - Yes → **rerank miss**

4. **Answerability check**  
   - Evidence insufficient but system answered confidently → **answerability false positive**  
   - Evidence sufficient but system refused/deflected → **answerability false negative**

5. **Citation check**  
   Are citations missing/incorrect/non-supportive for material claims?  
   - Yes → **citation failure**

6. **Synthesis check**  
   Are evidence and answerability mostly correct, but final reasoning/output is wrong?  
   - Yes → **synthesis failure**

7. **Safe-failure wording check**  
   Was safe-failure decision appropriate, but wording misleading or over-absolute?  
   - Yes → **safe failure wording issue**

If none apply cleanly, pick the closest earliest class and record ambiguity in secondary notes.

---

## 8) Tie-Break Rules (Single Primary Class)

When multiple issues appear in one case, use this explicit rule:

1. **Earliest-dominant-root-cause wins.** Choose the earliest pipeline layer that, if corrected, would most likely prevent the failure.
2. **Do not label downstream consequences as primary** if they were induced by an upstream miss.
3. **Use one primary class only.** Record additional problems as secondary notes.
4. **Decision priority order (earliest to latest):**  
   family misclassification → retrieval miss → rerank miss → answerability FP/FN → citation failure → synthesis failure → safe failure wording issue.

Practical tie-break example:
- Wrong evidence retrieved, wrong citation emitted, bad conclusion produced.  
  Primary = **retrieval miss** (upstream dominant cause). Secondary notes: citation + synthesis symptoms.

---

## 9) Example Failure Mappings

| Case summary | Primary failure class | Legal-family tag(s) | Why |
|---|---|---|---|
| Party-role question returns `fact_not_found`; retrieved text is generic and misses role-definition clauses present in corpus. | **retrieval miss** | **party/role** | Correct evidence never entered candidate set. |
| Follow-up on a selected document fails because active document context was not resolved; answer says insufficient evidence. | **document scope resolution failure** *(or retrieval miss + note, if optional class not used in runbook)* | **metadata**, **chronology** | Scope layer prevented retrieval of relevant segments in selected document. |
| Definition query is answered confidently from title/token overlap, without actual defining clause support. | **answerability false positive** | **issue spotting** (or **employment lifecycle**, based on topic) | System should have withheld answer due to insufficient evidence. |
| Unsupported query is declined, but phrasing implies “this right does not exist” instead of “not found in provided materials.” | **safe failure wording issue** | **financial/entitlement** | Safe-failure direction is right; wording is legally misleading. |
| Correct evidence is present at low rank, top context uses adjacent but irrelevant snippets, producing refusal. | **rerank miss** | **litigation milestones** | Upstream retrieval succeeded; ranking failed to promote decisive evidence. |
| Evidence and answerability are correct, but final answer combines conditions incorrectly and reverses conclusion. | **synthesis failure** | **mitigation** | Final reasoning/aggregation layer is dominant failure point. |

---

## 10) Operational Usage

This taxonomy supports five practical workflows for this solo project:

1. **Offline eval labeling**  
   Each failed eval case receives one primary class + family tags, enabling stable error distributions across versions.

2. **Online failure triage**  
   New production-like failures are labeled consistently, so debugging starts at the right pipeline layer.

3. **Regression dataset creation**  
   Repeated failure patterns become targeted regression cases grouped by class and family tag.

4. **Release reviews**  
   The project owner can assess whether release risk is dominated by upstream retrieval/routing issues or downstream synthesis/citation quality.

5. **Family-level quality reporting**  
   Failure rates can be sliced by legal-family tags to catch localized collapses hidden by global averages.

Role note: in this portfolio project, project owner, evaluator, and reviewer are one person wearing different process hats.

---

## 11) Definition of Done for a Triaged Failure

A failure is considered properly triaged only when all fields are present:
- **one primary failure class**,
- **at least one legal-family tag**,
- **a short root-cause note** (1–3 lines),
- **reproducibility status** (reproducible / intermittent / not reproducible),
- **regression-case recommendation** (yes/no; include brief rationale when yes).

If any item is missing, triage is incomplete.
