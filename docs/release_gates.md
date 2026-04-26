# Legal RAG Release Gates (Solo Portfolio Project)

## 1) Purpose

This document defines the release-gate policy for this legal RAG system: when a change is allowed to merge, when a build is allowed to pass staging, and when a version is allowed to ramp in production.

No release step proceeds without measurable quality checks. Decisions are based on recorded evidence, not intuition.

These gates cover **quality, latency, and cost**. Correct code alone is not enough for release if legal-answer quality regresses or runtime behavior becomes unsafe.

---

## 2) Scope

This policy applies to the full legal RAG answer path:
- retrieval and evidence selection,
- answerability/sufficiency decisions,
- answer generation and citations,
- safe-failure behavior,
- response latency, and
- request-level cost.

This document covers three gate levels:
- **merge gates** (pre-merge quality checks),
- **staging gates** (release-candidate qualification), and
- **production ramp gates** (controlled rollout decisions).

---

## 3) Release-Gate Principles

1. **Quality gates are required before release.**
2. **Legal-family quality matters more than global average alone.**
3. **Safe-failure regressions are release-relevant.**
4. **Release decisions are evidence-based and recorded.**
5. **Solo-project releases still use formal pass/fail rules.**

> Role labels used in this document are **project owner**, **evaluator**, and **reviewer**. In this solo project, all three roles are performed by the same person.

---

## 4) Gate Levels

| Gate level | Primary goal | Required evidence | Blocks progress when | Warning-only examples |
|---|---|---|---|---|
| **Merge gate** | Prevent obvious regressions from entering mainline. | Targeted offline eval for touched families + mandatory smoke checks. | Any hard-blocker metric regresses beyond merge tolerance. | Minor non-Tier-1 drift, small cost drift, minor latency drift below blocker limits. |
| **Staging gate** | Validate full release candidate before any production traffic. | Full offline regression suite, family slices, contract-shape checks, latency/cost report. | Any hard-blocker at staging thresholds; contract-shape break; material safe-failure regression. | Small non-material phrasing drift; modest cost increase under warning threshold. |
| **Production ramp gate** | Control real-traffic risk with monitored rollout. | Staged traffic ramp metrics, sampled trace review, post-release quality checks. | Triggered stop-ramp or rollback condition. | Temporary warning-level latency/cost drift with documented plan and close monitoring. |

---

## 5) Merge Gate Rules

A change may merge only if all rules below pass:

1. **Relevant offline eval must pass.**
   - Every touched family must be evaluated.
   - If Tier 1 behavior is touched, Tier 1 slices are mandatory.

2. **Smoke checks are mandatory for every change** (including small refactors and prompt tweaks):
   - output shape/schema consistency,
   - citation field presence when answerable,
   - no crash/time-out on smoke set.

3. **Required merge metrics (vs current accepted baseline):**
   - **False confident answer rate delta: <= +0.00 percentage points** (no increase allowed).
   - **Citation correctness delta: >= 0.00 percentage points** (no drop allowed).
   - **Tier 1 family score delta (each Tier 1 family): >= -0.50 percentage points**.
   - **p95 latency delta: <= +8%**.

A merge is blocked if any hard-blocker threshold is exceeded.

---

## 6) Staging Gate Rules

Staging is stricter than merge gating. A build passes staging only when all of the following are true:

1. **Full offline regression pass** on the versioned eval set.
2. **Tier 1 families remain above threshold** and do not exceed allowed regression tolerance.
3. **No contract-shape regressions** (response schema/fields/required keys unchanged unless intentionally versioned).
4. **No material safe-failure regression** (insufficient-evidence behavior remains correct and clear).
5. **Latency and cost remain within acceptable staging bounds.**

Minimum staging evidence:
- at least **300 total offline eval prompts**,
- at least **60 prompts per Tier 1 family**,
- family-level pass/fail summary and deltas vs baseline.

---

## 7) Production Ramp Gate Rules

Production rollout uses a controlled, solo-friendly ramp:

1. **Ramp phases**
   - Phase 1: 5% traffic for >= 60 minutes
   - Phase 2: 25% traffic for >= 120 minutes
   - Phase 3: 50% traffic for >= 180 minutes
   - Phase 4: 100% only if prior phases pass

2. **Monitored checks during ramp**
   - quality metrics on sampled outputs,
   - latency and cost drift,
   - safe-failure behavior on sampled insufficient-evidence cases.

3. **Sampled trace review requirement**
   - project owner/evaluator/reviewer (same person) reviews at least **30 sampled traces per phase**, including Tier 1-heavy cases.

4. **Ramp progression rule**
   - do not advance to next phase if any hard-blocker condition is triggered.

---

## 8) Hard Blockers vs Warning-Only Conditions

### Hard blockers (must stop merge/staging/ramp)

1. **False confident answers increase** above allowed delta for the gate level.
2. **Citation correctness drops** below allowed delta for the gate level.
3. **Any Tier 1 family score drops** beyond allowed tolerance.
4. **p95 latency increase** exceeds blocker threshold.
5. **Contract-shape regression** in staging/production candidate.
6. **Material safe-failure regression** (incorrectly answers when evidence is insufficient).
7. **Regression test pass rate** below required threshold.

### Warning-only conditions (release may proceed with explicit recorded acceptance)

1. Modest cost increase below blocker threshold.
2. Minor latency drift below blocker threshold.
3. Small non-Tier-1 family drift below blocker threshold.
4. Small safe-failure wording changes that do not alter correctness.
5. p99 latency drift in warning band while p95 remains within blocker limits.

Warnings must be explicitly accepted in the release record with mitigation notes and target follow-up date.

---

## 9) Thresholds

All thresholds are measured against the latest accepted production baseline using comparable workload mix.

| Metric | Merge gate | Staging gate | Production ramp (stop/rollback) |
|---|---:|---:|---:|
| **False confident answer rate delta** | **Block if > +0.00 pp** | **Block if > +0.00 pp** | **Stop/rollback if > +0.20 pp** |
| **Citation correctness delta** | **Block if < 0.00 pp** | **Block if < 0.00 pp** | **Stop/rollback if < -0.30 pp** |
| **Tier 1 family score delta (each family)** | **Block if < -0.50 pp** | **Block if < -0.30 pp** | **Stop/rollback if < -0.70 pp** |
| **Non-Tier-1 family score delta** | Warn at < -1.00 pp; block at < -2.00 pp | Warn at < -0.80 pp; block at < -1.50 pp | Stop at < -2.00 pp |
| **p95 latency delta** | Warn at > +5%; block at > +8% | Warn at > +4%; block at > +7% | Stop/rollback at > +10% |
| **p99 latency delta** | Warn at > +8%; block at > +12% | Warn at > +7%; block at > +10% | Stop/rollback at > +15% |
| **Cost per request drift** | Warn at > +8%; block at > +15% | Warn at > +6%; block at > +12% | Stop/rollback at > +18% sustained for 2 consecutive windows |
| **Regression tests pass rate** | **100% required** on required suite | **100% required** on full suite | Stop ramp if < 100% on ramp checks |
| **Minimum eval coverage** | >= 120 prompts total; >= 25 per touched family | >= 300 prompts total; >= 60 per Tier 1 family | >= 30 sampled traces per ramp phase |

(pp = percentage points)

---

## 10) Tier 1 Emphasis

Tier 1 families have the strictest protection and the tightest tolerances:
- **party / role / verification**,
- **chronology / date / event**,
- **employment lifecycle**.

A regression in any Tier 1 family is treated as more release-sensitive than equivalent drift in non-Tier-1 families. A strong global average does not override Tier 1 failure.

---

## 11) Rollback / Stop-Ramp Rules

During production ramp, immediately stop progression and roll back to the last accepted version if any of the following occurs:

1. false confident answer rate delta exceeds **+0.20 pp**,
2. citation correctness delta drops below **-0.30 pp**,
3. any Tier 1 family score delta drops below **-0.70 pp**,
4. p95 latency exceeds **+10%** for two consecutive monitoring windows,
5. contract-shape regression or material safe-failure regression is observed.

If a warning-only condition persists across two phases, pause ramp and require a reviewer decision note before continuing.

---

## 12) Release Record Requirements

Each release candidate must include a concise release record with:

1. version identifier (commit/tag),
2. changed area(s),
3. touched legal families,
4. eval summary (metrics, deltas, coverage, and key failure examples),
5. gate outcome at each level (merge/staging/ramp),
6. any accepted warnings and rationale.

Release decisions are only valid when this record is complete and auditable.
