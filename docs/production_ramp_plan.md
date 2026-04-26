# Production Ramp Plan (Ticket 30)

## Purpose

Define a staged, solo-project rollout process that converts offline quality evidence into disciplined production promotion decisions.

## Release owner model

- **Release owner:** single decision maker for this project.
- The same person performs evaluator/reviewer roles, but decisions are still recorded per stage.

## Stage overview

1. **Stage 0 — Offline candidate approval**
2. **Stage 1 — Staging validation**
3. **Stage 2 — Limited live sample pass**
4. **Stage 3 — Broader rollout**

Progression rule: do not advance stages if any rollback threshold is hit.

---

## Stage 0 — Offline candidate approval

### Entry criteria

- Candidate code is frozen for release decision.
- Baseline release artifact selected.
- Offline eval datasets and tooling are available.

### Checks performed

- Run full offline eval for candidate and baseline.
- Run baseline-vs-candidate comparison (`compare_runs`).
- Verify required metrics:
  - false confident rate,
  - citation correctness,
  - Tier 1 family deltas,
  - p95 latency,
  - cost/request (if captured).

### Exit criteria

- All pass thresholds meet `docs/rollback_criteria.md` pass bands.
- No Tier 1 material regression.
- Any warning metric has documented mitigation + owner target date.

### Who decides

- Release owner.

### Rollback action if failed

- Do not promote to staging-release candidate; treat as `NO_GO` and fix candidate.

---

## Stage 1 — Staging validation

### Entry criteria

- Stage 0 completed and marked pass.
- Deployable candidate available in staging/local-staging environment.

### Checks performed

- Run representative smoke query set (legal-family coverage, including Tier 1).
- Confirm trace spans render with required fields.
- Confirm debug payload fields are populated (`family`, answerability, citations, warnings).
- Verify response contract shape is valid (no malformed outputs).
- Review fallback/failure pattern frequency against baseline.

### Exit criteria

- 100% smoke-query contract compliance.
- 0 malformed outputs in staging sample.
- No unexpected severe failure pattern.
- Metrics stay within pass or accepted warning bands.

### Who decides

- Release owner.

### Rollback action if failed

- Stop progression; revert staging to prior stable release config and file remediation tasks.

---

## Stage 2 — Limited live sample pass

### Entry criteria

- Stage 1 pass recorded.
- Sampling/review mechanism available (real sampled traffic or curated live-style set).

### Checks performed

- Evaluate a limited sample window:
  - **minimum 30 samples** total,
  - **minimum 10 Tier 1-heavy samples**,
  - include insufficient-evidence cases for safe-failure checks.
- Manual review for groundedness and citation behavior.
- Calculate false confident proxy, citation correctness proxy, family stability signals.
- Check malformed output and severe failure-family repetition.

### Exit criteria

- False confident proxy <= 2.0%.
- Citation correctness >= 98%.
- No Tier 1 family delta below rollback band.
- No repeated severe failure family pattern.
- No malformed output above pass threshold (target 0).

### Who decides

- Release owner.

### Rollback action if failed

- Roll back to last-known-good release and open incident note per rollback criteria.

---

## Stage 3 — Broader rollout

### Entry criteria

- Stage 2 pass recorded.
- Rollback path validated and ready.

### Checks performed

- Gradual traffic expansion (solo-friendly):
  - 25% exposure for one monitoring window,
  - 50% exposure for one monitoring window,
  - 100% only after both windows pass.
- At each window, compare key metrics to baseline and rollback table.
- Continue sampled manual review of at least 10 traces/window.

### Exit criteria

- No rollback triggers hit in any window.
- Warning conditions, if any, are accepted with a remediation deadline.
- Release record marked `GO` or `GO_WITH_WARNINGS`.

### Who decides

- Release owner.

### Rollback action if failed

- Immediate rollback to last-known-good and freeze further ramp until remediation is validated.

---

## Promotion rules summary

- **Promote** only when current stage exit criteria are fully met.
- **Pause** when warning-band metrics appear; investigate before continuing.
- **Rollback** immediately when rollback-band thresholds or immediate rollback triggers are hit.

## Evidence required in release record

For each stage, log:

- entry timestamp,
- checks run and sample sizes,
- metric values vs thresholds,
- decision (`PASS`, `PAUSE`, `ROLLBACK`),
- release owner note.

