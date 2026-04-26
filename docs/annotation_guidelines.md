# Legal RAG Annotation Guidelines (Solo Portfolio Project)

## 1) Purpose

This document defines how manual review is performed for legal RAG outputs so labels are **consistent, repeatable, and auditable**.

The goal is disciplined evaluation, not ad hoc judgment. In this solo portfolio project, the roles **project owner**, **evaluator**, and **reviewer** are used for clarity, and are all performed by the same person.

These guidelines are designed so the same reviewer can apply the same rubric across different review sessions and still reach comparable outcomes.

---

## 2) Scope

These guidelines apply to manual review of one evaluated sample at a time.

Each sample may include some or all of the following artifacts:
- user query,
- selected-document context (when present),
- final answer,
- citations,
- debug/context metadata (when available, such as retrieval notes or selected chunks).

Out of scope: grader implementation, dataset creation, dashboards, tracing architecture, or any code changes.

---

## 3) Review Principles

1. **Evaluate the answer against the question actually asked.**
2. **Use retrieved evidence, not intuition or outside knowledge.**
3. **Score correctness and grounding separately.**
4. **Score safe-failure quality separately from correctness.**
5. **When the answer is bad, assign one primary legal family label.**
6. **Prefer rubric-based, reproducible decisions over subjective impressions.**

---

## 4) Review Workflow (Single Sample)

Use this sequence in order for each sample:

1. Read the user query.
2. Identify the expected legal family (question type).
3. Review selected-document/conversation scope if present.
4. Read the final answer fully.
5. Inspect each citation and the cited evidence.
6. Grade **correctness**.
7. Grade **grounding**.
8. Grade **citation quality**.
9. Grade **safe failure** (if the sample is answerability-constrained).
10. If failed, assign one primary **failure family** label.
11. Record annotation fields and short reviewer note.

---

## 5) Correctness Rubric (4-point)

**Definition:** How accurately the answer resolves the asked legal question.

| Grade | Label | When to use | When not to use |
|---|---|---|---|
| **C3** | Fully correct | Answer is complete, materially accurate, and responsive to the exact question. | Do not use if key requested element is missing or materially wrong. |
| **C2** | Partially correct | Core direction is right, but answer is incomplete, imprecise, or misses one required element. | Do not use when central claim is wrong. |
| **C1** | Incorrect | Material legal fact/role/date/outcome is wrong for the asked question. | Do not use when answer is intentionally safe failure due to missing evidence. |
| **C0** | Not answerable / should have failed safely | Evidence does not support answering; a safe deferral was required. | Do not use when sufficient evidence existed and model simply answered poorly. |

### Correctness notes
- Correctness is about **what was answered**, not whether citation formatting was good.
- If answer includes both correct and incorrect claims, score by material impact.

### Correctness examples
- **C3:** “The employer is Northwind Logistics Inc.; the employee is Dana Li, per the agreement recital.”
- **C2:** Correctly identifies employer but omits employee when both were asked.
- **C1:** Swaps claimant/respondent roles.
- **C0:** Question asks for termination date, but no termination event exists in provided evidence.

---

## 6) Grounding Rubric (4-point)

**Definition:** Degree to which answer claims are supported by provided evidence.

| Grade | Label | When to use | When not to use |
|---|---|---|---|
| **G3** | Fully grounded | All material claims are directly supported by provided evidence. | Do not use if any key claim depends on unstated inference. |
| **G2** | Partially grounded | Main claim is supported, but one or more secondary claims are weakly supported or inferred. | Do not use when most of answer is speculative. |
| **G1** | Unsupported / speculative | Material claims are not supported by cited or available evidence. | Do not use when support exists but citation linking is merely weak. |
| **G0** | Not applicable (safe failure) | Model correctly declines due to insufficient evidence; no factual claim to ground. | Do not use if model made factual assertions while refusing. |

### What counts as unsupported inference
Use **G1** when the answer infers facts not explicitly evidenced, such as:
- inferring party identity from a signature witness line,
- inferring event sequence from document order without dates,
- inferring severance entitlement from generic policy text without governing clause linkage.

### Grounding examples
- **G3:** Start date and probation duration each mapped to explicit clauses.
- **G2:** Termination date grounded, but “for cause” rationale inferred from unrelated paragraph.
- **G1:** Claims severance amount not present in any provided text.
- **G0:** “Insufficient evidence to verify the requested event date” with no extra factual claims.

---

## 7) Citation Quality Rubric (4-point)

**Definition:** Whether citations are present, relevant, and actually support associated claims.

| Grade | Label | When to use | When not to use |
|---|---|---|---|
| **Q3** | Present and supportive | Citations are present for material claims and point to evidence that directly supports each claim. | Do not use if citations are generic or only loosely related. |
| **Q2** | Present but weak/partial | Citations exist but only support part of the claim, are too broad, or omit key claim linkage. | Do not use if citations are clearly irrelevant. |
| **Q1** | Missing citations | Material claims lack citations where citations are expected. | Do not use when citations are present but wrong; use Q0/Q2 as appropriate. |
| **Q0** | Irrelevant or unsupported citations | Citations are present but do not support the claim or point to unrelated text. | Do not use when citations are merely incomplete but still directionally relevant. |

### What is a good citation
A good citation should:
- point to the exact clause/span backing the claim,
- match the legal fact asserted (party, date, amount, condition),
- avoid relying on unrelated headings or nearby but non-responsive text.

### Citation examples
- **Q3:** Citation points to clause stating “Employment commences on 2023-05-01.”
- **Q2:** Citation points to compensation section but not to the bonus condition being asserted.
- **Q1:** Answer states termination date with no citation.
- **Q0:** Citation points to confidentiality clause for a severance amount claim.

---

## 8) Safe Failure Rubric (4-point)

**Definition:** Quality of behavior when evidence is insufficient to answer safely.

| Grade | Label | When to use | When not to use |
|---|---|---|---|
| **S3** | Correct safe failure | Correctly declines, states insufficiency precisely, avoids speculation, and gives a useful next-evidence request. | Do not use if wording overstates certainty or is confusing. |
| **S2** | Safe failure but poorly worded | Declines correctly but wording is vague, overly broad, or weakly actionable. | Do not use if it should have answered from available evidence. |
| **S1** | False insufficient | Claims insufficiency even though adequate evidence exists. | Do not use when evidence is actually missing. |
| **S0** | False confident answer | Answers confidently despite insufficient evidence; should have failed safely. | Do not use when evidence was sufficient and answer is merely wrong synthesis. |

### Acceptable safe-failure behavior
A safe failure is acceptable when it:
- explicitly states which requested fact is not verifiable,
- does not invent parties/dates/amounts,
- requests the minimal additional evidence needed.

### Misleading safe-failure wording
Wording is misleading when it says things like:
- “no evidence exists” when evidence may exist outside provided scope,
- “cannot answer legal questions” when only one field is missing,
- broad refusal without stating what is missing.

### Safe-failure examples
- **S3:** “I cannot verify the termination date from the provided excerpts; please provide the termination notice or event log clause.”
- **S2:** “Not enough information.” (correct direction, poor specificity)
- **S1:** Refuses to identify employer despite explicit recital text.
- **S0:** Guesses a termination date from sequence assumptions without dated evidence.

---

## 9) Failure-Family Labeling

When a sample fails, label the **legal family of the question** first. This is not the same as root-cause class.

- Family = *what type of legal question this is*.
- Root cause = *why the system failed* (tracked elsewhere if needed).

Use one primary family label per failed sample:
- **party/role**
- **chronology**
- **employment lifecycle**
- **metadata**
- **litigation milestones**
- **mitigation**
- **financial/entitlement**
- **issue spotting**

If multiple families appear, choose the family that best matches the user’s main requested outcome.

---

## 10) Annotation Fields to Record

Minimum required fields per reviewed sample:

| Field | Description |
|---|---|
| `sample_id` | Unique sample identifier. |
| `family` | One primary legal family label. |
| `correctness_grade` | C3/C2/C1/C0 from Section 5. |
| `grounding_grade` | G3/G2/G1/G0 from Section 6. |
| `citation_quality_grade` | Q3/Q2/Q1/Q0 from Section 7. |
| `safe_failure_grade` | S3/S2/S1/S0 from Section 8. |
| `overall_decision` | Pass/Fail (or Accept/Reject) using local eval policy. |
| `reviewer_note` | Short rationale, especially for borderline cases. |

---

## 11) Examples by Tier 1 Family

The examples below are synthetic but realistic and aligned to legal RAG review tasks.

### 11.1 Party / Role / Verification

**Good answer example**
- **Query:** “Who are the parties to the employment agreement?”
- **Evidence:** Recital: “This Employment Agreement is between Aurora Care Ltd. (Employer) and Maya Chen (Employee).”
- **Answer:** “The parties are Aurora Care Ltd. (Employer) and Maya Chen (Employee).”
- **Expected grades:** C3, G3, Q3, S3 (N/A context if not a refusal)

**Bad answer example**
- **Query:** “Who is the employer?”
- **Evidence:** Recital identifies Aurora Care Ltd. as employer.
- **Answer:** “Maya Chen is the employer.”
- **Why bad:** Role inversion.
- **Expected grades:** C1, G1, Q0 or Q1, S0 (if answered confidently without support)

**Safe-failure example**
- **Query:** “Is Blue Harbor LLC a party to this agreement?”
- **Evidence:** No party clause shown; only an email footer mentions Blue Harbor LLC.
- **Answer:** “I can’t verify Blue Harbor LLC as a contracting party from the provided text. Please provide the party-definition or recital section.”
- **Expected grades:** C0, G0, Q3 (if citation points to missing-scope evidence), S3

### 11.2 Chronology / Date / Event

**Good answer example**
- **Query:** “What happened first: suspension notice or termination letter?”
- **Evidence:** Suspension notice dated 2024-02-10; termination letter dated 2024-02-18.
- **Answer:** “The suspension notice came first (2024-02-10), followed by the termination letter (2024-02-18).”
- **Expected grades:** C3, G3, Q3, S3 (N/A context if not a refusal)

**Bad answer example**
- **Query:** “When was termination effective?”
- **Evidence:** Clause states effective termination date is 2024-03-01.
- **Answer:** “Termination was effective on 2024-02-20.”
- **Why bad:** Wrong date despite explicit evidence.
- **Expected grades:** C1, G1, Q0/Q1, S0

**Safe-failure example**
- **Query:** “What occurred between March 3 and March 5?”
- **Evidence:** Documents include obligations but no dated events in that range.
- **Answer:** “The provided excerpts do not contain dated events between March 3 and March 5, so I can’t confirm a sequence for that window.”
- **Expected grades:** C0, G0, Q3 (if cites reviewed range), S3

### 11.3 Employment Lifecycle

**Good answer example**
- **Query:** “What are the employee’s base salary and severance terms?”
- **Evidence:** Compensation clause: "$92,000 annual salary." Termination clause: “Severance equals 8 weeks base pay if terminated without cause after probation.”
- **Answer:** “Base salary is $92,000 annually. Severance is 8 weeks of base pay if termination is without cause after probation.”
- **Expected grades:** C3, G3, Q3, S3 (N/A context if not a refusal)

**Bad answer example**
- **Query:** “When did probation end?”
- **Evidence:** Probation clause: “90 days from start date of 2023-06-01.”
- **Answer:** “Probation ended after 6 months.”
- **Why bad:** Invented term inconsistent with clause.
- **Expected grades:** C1, G1, Q0/Q1, S0

**Safe-failure example**
- **Query:** “What severance amount is owed today?”
- **Evidence:** Contract has severance formula but no termination date or current salary basis in provided scope.
- **Answer:** “I can cite the severance formula, but I can’t compute an owed amount from the provided material because current salary basis and triggering event details are missing.”
- **Expected grades:** C0, G0, Q3, S3

---

## 12) Consistency Rules (Solo Project)

To keep repeated review passes consistent over time:

1. Use the rubric definitions literally.
2. Do not silently rescore historical samples under a new interpretation.
3. If rubric interpretation changes, document date and reason in review notes.
4. When unsure, add a short note instead of inventing a new scoring rule.
5. Before a large pass, re-check a small personal calibration set of prior samples.

This is the solo-project equivalent of inter-rater consistency: one reviewer, stable standards.

---

## 13) Decision Rules for Difficult Cases

Use these tie-break rules for ambiguous samples:

1. **Correct answer but weak citation**
   - Score correctness independently (often C3/C2).
   - Lower citation quality (Q2/Q1/Q0 based on severity).

2. **Good citation but partial answer**
   - Keep grounding/citation high if evidence support is strong.
   - Lower correctness for incompleteness (usually C2).

3. **Safe failure under ambiguous evidence**
   - If evidence is genuinely inconclusive, prefer safe failure (S3/S2) over speculative assertion.
   - If evidence is sufficient, refusal is S1 (false insufficient).

4. **Partially correct answer with unsupported extra claim**
   - Correctness usually C2 if core answer is right.
   - Grounding drops to G2 or G1 depending on materiality of unsupported addition.
   - Citation quality drops if extra claim lacks support.

5. **Family-label ambiguity**
   - Choose family by the user’s primary requested outcome, not by where the model failed internally.

