# Legal RAG Evaluation Families (Solo Portfolio Project)

## 1) Purpose

This document defines legal question **families** as the core unit of quality measurement for this legal RAG.

Evaluating only a single overall average is not enough. A model can look strong in aggregate while failing badly on a specific legal task (for example, timeline reconstruction or party verification). In legal workflows, those family-specific failures matter more than a smooth global score.

The goal here is to measure **family-specific correctness**, not generic answer fluency. For each family, evaluation must confirm:
- the answer is responsive to the question type,
- the evidence type is appropriate for that family,
- known false positives are rejected, and
- safe-failure behavior is correct when evidence is missing.

This project is maintained by one person. The role labels **project owner**, **evaluator**, and **reviewer** are used for clarity, and in this project they are all performed by the same builder.

---

## 2) Scope

This document defines legal evaluation families for the legal RAG system.

It is used to support:
- offline eval design and execution,
- regression tracking by family,
- observability review by family,
- release decisions that include family-level pass/fail judgment.

Out of scope for this document: dataset implementation, grader implementation, tracing design, and dashboard implementation.

---

## 3) Family-Based Evaluation Principles

1. **Families are measured separately.**
   Each legal question family has its own quality signal.

2. **Evidence expectations are family-specific.**
   The same evidence standard does not apply equally across all families.

3. **False-positive risk is family-specific.**
   Common traps differ by family and must be explicitly listed.

4. **Safe-failure behavior is family-specific.**
   Refusal/defer behavior must match the evidence requirements of that family.

5. **Cross-family performance does not transfer automatically.**
   Strong results in one family do not imply strong results in another.

---

## 4) Family Schema

Each family specification in this document must include the following required fields:

| Field | Required | What it must contain |
|---|---|---|
| Representative questions | Yes | Typical user questions that belong in the family. |
| Required evidence | Yes | Evidence types that are acceptable to answer those questions. |
| False positives | Yes | Evidence patterns that may look relevant but must not count. |
| Safe failure rule | Yes | Expected system behavior when responsive evidence is missing or insufficient. |

Optional fields when useful:
- common query variants,
- notes on ambiguity,
- evaluation emphasis.

---

## 5) Tier 1 Families (First Production Priority)

Tier 1 families are the first families that must be fully specified and used for implementation-facing evaluation work.

### 5.1 Party / Role / Verification (Tier 1)

**Representative questions**
- Who is the employer?
- Who is the employee?
- Who are the parties?
- Who are the two parties involved?
- Is this agreement between X and Y?
- Is <Person/Entity> actually listed as a party or only referenced elsewhere?

**Required evidence**
- Agreement introduction/recital language that explicitly identifies parties.
- “Between X and Y” or equivalent bilateral party structure.
- Explicit role labels such as “Employer,” “Employee,” “Company,” “Contractor,” where role assignment is clear.
- Signature block or party block only when it reliably identifies legal parties (not just recipients/copy lines).

**False positives (must not count)**
- Unrelated clauses that mention “employer” or “employee” in generic terms.
- Substantive clauses (confidentiality, IP, termination, etc.) that reference role words without party-identification context.
- Title-only references (document title, section heading, email subject) without explicit party evidence.
- Mentions of third parties (insurers, counsel, affiliates, witnesses) that are not contracting parties.

**Safe failure rule**
- The system must not guess parties or roles.
- If responsive party-identification evidence is missing or ambiguous, return a constrained “insufficient evidence to verify parties/roles” result and request a clause or section with explicit party definition.

---

### 5.2 Chronology / Date / Event (Tier 1)

**Representative questions**
- When did employment start?
- When was the termination notice given?
- What happened first?
- What happened after X?
- What happened between date A and date B?
- What is the sequence of notice, response, and termination events?

**Required evidence**
- Dated event lines tying a concrete event to a date/time.
- Effective-date language for agreements, amendments, or status changes.
- Termination dates and notice dates.
- Letter/email dates where the event being asked about is captured.
- Filing, service, or delivery dates where legally relevant.
- Event descriptions explicitly linked to those dates (not date-only headers).

**False positives (must not count)**
- Undated clauses describing legal obligations but no event timing.
- General legal text with no date-event linkage.
- Section headings with dates but no event content.
- Dates from unrelated documents/events that do not answer the asked timeline.

**Safe failure rule**
- The system must not invent timeline order or missing dates.
- If dated responsive evidence is insufficient, return an “insufficient dated evidence for chronology” response and only report verified events/dates that are explicitly supported.

---

### 5.3 Employment Lifecycle (Tier 1)

**Representative questions**
- When did the employment relationship begin?
- What were the compensation terms?
- When did probation end?
- What benefits applied?
- When did termination take effect?
- What severance was offered?
- When was the ROE issued?

**Required evidence**
- Effective-date language for commencement/start.
- Compensation clauses (salary, hourly rate, bonus terms, pay cadence).
- Benefits clauses (eligibility, plan references, start conditions).
- Probation language (duration, end condition, extension terms).
- Termination provisions and effective termination statements.
- Severance language (amount, formula, conditions, timing).
- ROE references with issuance timing/status when present.

**False positives (must not count)**
- Generic duties/responsibilities language that does not answer lifecycle facts.
- Unrelated policy sections not tied to the asked lifecycle question.
- Broad references to “employment” without the specific fact requested (start date, compensation term, probation end, severance, ROE).
- Benefit or compensation mentions from non-governing drafts/attachments when controlling terms are elsewhere.

**Safe failure rule**
- The system must not invent lifecycle facts.
- If responsive lifecycle evidence is missing, the system must explicitly mark the requested field as not verifiable from current evidence and avoid filling gaps with assumptions.

---

## 6) Additional Future Families (Not Tier 1)

These are future-phase families and are intentionally secondary until Tier 1 coverage is operational.

- Matter / document metadata
- Correspondence / litigation milestones
- Mitigation
- Financial / entitlement
- Policy / issue spotting

These may be expanded later, but Tier 1 remains the release-critical baseline for initial family-based evaluation.

---

## 7) How Families Are Used in This Solo Project

The project owner/evaluator/reviewer (same person) uses this family framework to drive practical quality work:

1. **Evaluation dataset design**
   Build question sets grouped by family so each family has explicit coverage.

2. **Regression grouping**
   Track failures and pass rates by family, not only globally.

3. **Failure analysis**
   Diagnose whether issues are concentrated in a specific family and evidence pattern.

4. **Release review**
   Require Tier 1 family-level review before accepting a release candidate.

5. **Observability dashboards later**
   Use these family definitions as stable categories for later telemetry/dashboard slices, without requiring dashboard implementation in this document.

---

## 8) Tier 1 Completion Criteria

A Tier 1 family is considered “specified enough” for implementation and evaluation use when all criteria below are met:

1. Representative questions clearly distinguish what belongs in the family.
2. Required evidence is concrete enough to guide labeling and grading.
3. False positives are explicit enough to prevent weak matches from passing.
4. Safe failure rule is testable and prohibits unsupported guessing.
5. The family can be used as a standalone grouping for dataset items, regression results, and release review notes.

If any criterion is missing, the family is draft-only and should not be treated as fully operational.
