# Legal RAG Failure Taxonomy (Solo Portfolio Project)

## 1) Purpose

This taxonomy defines how failures are labeled in this legal RAG so triage is precise, repeatable, and useful for iteration.

It replaces vague labels like “the RAG failed” with a consistent record for each failed case:
- exactly **one primary failure class** (root-cause layer),
- **one or more legal-family tags** (legal-domain area), and
- optional short notes for context.

This project is maintained by one person. The terms **project owner**, **evaluator**, and **reviewer** are used for clarity, but all three roles are performed by the same builder.

---

## 2) Scope

This taxonomy covers the end-to-end answer path:
- query-to-family routing,
- retrieval,
- reranking,
- answerability/sufficiency decision,
- synthesis of the final answer,
- citations attached to claims,
- safe-failure wording when evidence is insufficient.

Out of scope: instrumentation design, evaluator implementation, dashboards, dataset schema, or tracing architecture.

---

## 3) Classification Principles

1. **Exactly one primary failure class per failed case.**
2. The primary class identifies the **earliest dominant root-cause layer**.
3. **Legal-family tags are separate from failure class** and can be multiple.
4. Secondary symptoms may be noted, but do not change the one-class rule.
5. A safe refusal is not a failure by itself; it is a failure only when the decision is wrong or the wording is misleading.

---

## 4) Primary Failure Classes

| Class | Definition | When to use | When not to use | Common symptoms | Pipeline layer | Short example |
|---|---|---|---|---|---|---|
| **family misclassification** | Query is routed to the wrong legal family, causing downstream mismatch. | Family/router chose the wrong domain bucket before retrieval. | Retrieval had correct family context but missed evidence anyway. | Irrelevant family-specific chunks; answer tone/logic for wrong legal task. | Routing / family classification | A party-role question is routed as chronology, so role clauses are never prioritized. |
| **retrieval miss** | Needed evidence was never retrieved in candidate set. | Gold evidence absent from retrieved set despite correct family and query intent. | Gold evidence appears in candidates but lower rank causes miss in context window. | `fact_not_found` despite existing support; high lexical overlap but wrong sections. | Retrieval | The contract defines “Cause,” but no chunk containing that definition is retrieved. |
| **rerank miss** | Needed evidence was retrieved but not ranked high enough to influence answer. | Gold evidence in retrieved candidates but pushed below used top-N. | Gold evidence never retrieved (retrieval miss), or was present and used correctly. | Top results are plausible but not decisive; correct chunk buried. | Reranking / context selection | Correct termination clause is in rank 18, while top 5 contain generic policy text. |
| **answerability false positive** | System says evidence is sufficient and answers, but support is insufficient. | Final output gives a confident answer without enough evidence. | Evidence is sufficient but synthesis/citation is wrong (use synthesis/citation failure). | Confident unsupported claim; extrapolation from title/token overlap. | Answerability decision | A definition is answered confidently from keyword overlap, not from an actual definitional clause. |
| **answerability false negative** | System says evidence is insufficient when sufficient support exists. | Correct supporting evidence is available, but system refuses/defers. | Evidence truly insufficient (no failure), or failure is earlier retrieval/rerank. | Unnecessary `fact_not_found`; refusal despite explicit supporting text in context. | Answerability decision | Retrieved excerpts name the responsible party, but system still returns insufficient evidence. |
| **citation failure** | Citation is missing, incorrect, or does not support the associated claim. | Claim may be right or wrong, but citation linkage/support is broken. | Citation is correct but reasoning is wrong (synthesis failure). | Wrong source span, dangling reference, unsupported cited claim. | Citation attachment / formatting | Answer states severance amount but cites an unrelated notice-period paragraph. |
| **synthesis failure** | Final answer combines evidence incorrectly despite sufficient evidence and answerability decision. | Sub-evidence exists, but logic, aggregation, or final wording is materially wrong. | Problem is mainly unsupported sufficiency decision (answerability FP/FN). | Contradictory summary; wrong party/date/condition despite correct retrieved snippets. | Generation / synthesis | Correct dates are retrieved, but the timeline summary reverses event order. |
| **safe failure wording issue** | Safe failure intent is correct, but wording is misleading, overbroad, or unhelpful. | System should defer, but phrasing implies wrong reason or wrong next step. | Evidence was actually sufficient (then it is answerability false negative). | “No evidence exists” when evidence is merely out of scope; confusing next-step guidance. | Safe-failure response wording | Query is unsupported; response safely declines but incorrectly says the selected document has no legal clauses at all. |

---

## 5) Optional Supporting Classes (Use Sparingly)

These are optional and should only be used when they materially improve triage clarity.

| Class | Use case |
|---|---|
| **document scope resolution failure** | Follow-up references a selected document/context, but scope resolution fails before retrieval. |
| **evidence selection failure** | Correct evidence appears in context, but selected subset for final answer omits decisive span. |
| **output contract failure** | Output violates required response contract/format even if legal reasoning is otherwise correct. |

If uncertain, prefer a primary class from Section 4 and note these as secondary symptoms.

---

## 6) Legal-Family Tags

Legal-family tags label **what legal area** the failure belongs to. They are not root-cause classes.

Required family tags:
- **party/role**
- **chronology**
- **employment lifecycle**
- **metadata**
- **litigation milestones**
- **mitigation**
- **financial/entitlement**
- **issue spotting**

A failed case should include one or more tags.

**Example distinction:**
- primary failure class: `retrieval miss`
- legal-family tag: `party/role`

---

## 7) Classification Decision Guide

Use this decision flow in order:

1. **Family routing check:** Was the query routed to the wrong legal family?  
   - Yes → `family misclassification`
2. **Retrieval coverage check:** Was required evidence absent from retrieved candidates?  
   - Yes → `retrieval miss`
3. **Rerank placement check:** Was required evidence retrieved but ranked too low to be used?  
   - Yes → `rerank miss`
4. **Answerability decision check:**
   - Evidence insufficient but system answered confidently → `answerability false positive`
   - Evidence sufficient but system refused/deferred → `answerability false negative`
5. **Citation check:** Are citations missing/wrong/unsupported for material claims?  
   - Yes → `citation failure`
6. **Synthesis check:** With correct evidence and answerability, is final reasoning/combination wrong?  
   - Yes → `synthesis failure`
7. **Safe-failure wording check:** If safe deferral is appropriate, is wording misleading or materially poor?  
   - Yes → `safe failure wording issue`

If no condition is met, do not force-label as a failure.

---

## 8) Tie-Break Rule (Required)

When multiple issues appear, assign the primary class using the **earliest-dominant-root-cause rule**:

1. Start from the earliest pipeline layer.
2. Identify the first error that plausibly caused downstream symptoms.
3. Assign that as the **one primary failure class**.
4. Record later issues as secondary notes only.

Priority order for tie-breaks:
`family misclassification` → `retrieval miss` → `rerank miss` → `answerability false positive/false negative` → `citation failure` → `synthesis failure` → `safe failure wording issue`.

---

## 9) Example Mappings

| Scenario | Primary failure class | Legal-family tag(s) | Why |
|---|---|---|---|
| Party-role question retrieves unrelated clauses and returns `fact_not_found`. | retrieval miss | party/role | Needed role evidence was never retrieved. |
| Selected-document follow-up fails because prior document scope was not resolved. | document scope resolution failure *(optional)* | metadata, chronology | Failure happened at context/document scope resolution before effective retrieval. |
| Definition query is answered confidently from title/token overlap rather than a true definition clause. | answerability false positive | issue spotting | System treated weak overlap as sufficient evidence. |
| Unsupported query is declined, but wording wrongly implies the repository contains no relevant legal material at all. | safe failure wording issue | mitigation | Deferral intent is correct; explanation is misleading. |
| Correct evidence is retrieved but buried below top-N; answer uses weaker snippets. | rerank miss | financial/entitlement | Retrieval found support, ranking prevented use. |
| Correct snippets and sufficiency decision exist, but final answer merges terms incorrectly. | synthesis failure | employment lifecycle | Error is in final composition, not evidence access. |

---

## 10) How This Taxonomy Is Used (Solo Project)

The same person (project owner/evaluator/reviewer) uses this taxonomy to:

1. **Label offline evaluation failures** consistently.
2. **Triage online failures** quickly with one primary class plus legal-family tags.
3. **Build regression cases** from recurring failure patterns.
4. **Run release reviews** by checking class frequency and severity shifts.
5. **Report family-level quality** (for example, whether `party/role` failures are mostly retrieval vs synthesis).

This keeps iteration practical for a single builder while preserving production-grade rigor.

---

## 11) Minimum Triage Record

A failure is considered properly triaged only when it includes:
- one primary failure class,
- at least one legal-family tag,
- a short root-cause note,
- reproducibility status (reproducible / non-reproducible / intermittent),
- whether it should be promoted to a regression case.

