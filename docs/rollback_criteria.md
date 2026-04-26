# Rollback Criteria (Ticket 30)

## Purpose

Define explicit, measurable conditions for pausing a rollout or rolling back to the last accepted release. This document is threshold-first and metric-first.

## Severity model

- **Pass:** within expected range, rollout can continue.
- **Warning:** degrade detected; pause phase progression until reviewed.
- **Rollback:** unacceptable risk/regression; immediately revert to last-known-good release.

## Baseline reference

All thresholds are measured against the **latest accepted production baseline** using comparable workload mix, dataset slice, and configuration.

## Metric thresholds (pass / warning / rollback)

| Metric | Baseline reference | Pass threshold | Warning threshold | Rollback threshold | Rationale |
|---|---|---|---|---|---|
| False confident answer rate | Baseline run + live sample proxy | Candidate <= baseline + 0.00 pp (offline/staging); live sample <= 2.0% absolute | > pass and <= 2.2% live sample | > 2.2% live sample **or** > baseline + 0.20 pp in rollout window | Unsupported confident legal answers are highest safety risk. |
| Citation correctness | Baseline cited-claim correctness | >= 98% and no drop vs baseline in offline/staging | >= 97.7% and < 98% | < 97.7% **or** delta < -0.30 pp vs baseline | Verifiability must remain near-perfect for legal claims. |
| Tier 1 family score (each family) | Baseline per-family pass rate (`party_role_verification`, `chronology_date_event`, `employment_lifecycle`) | Delta >= -0.30 pp (offline/staging) | Delta < -0.30 pp and >= -0.70 pp | Delta < -0.70 pp for any Tier 1 family | Prevent hidden regressions masked by global average. |
| p95 latency | Baseline p95 latency from eval + live window | <= baseline + 7% (staging), <= baseline + 10% (ramp) | > pass and <= baseline + 10% (staging), <= baseline + 12% (ramp) | > baseline + 12% for 2 consecutive windows | Quality is primary, but unusable latency still blocks release value. |
| Cost per request (if available) | Baseline avg cost/request | <= baseline + 12% | > baseline + 12% and <= +18% | > baseline + 18% for 2 consecutive windows | Solo project budgets matter; sustained spikes require rollback. |
| Malformed output / contract break rate | Baseline malformed rate (target 0) | 0% malformed in staging and live sample | >0% and <=0.5% in one window with immediate fix in progress | >0.5% in any window **or** repeated malformed output across 2 windows | Contract breaks can invalidate downstream grading and UX. |
| Severe repeated failure family in live sample | Baseline failure taxonomy mix | No repeated severe family pattern | Same severe family appears in >=2 samples in one window | Same severe family appears in >=3 samples in one window or recurs in 2 consecutive windows | Repetition signals systematic, not random, failure. |
| Safe-failure quality (if measured) | Baseline safe-failure score | >= 4.3/5 and no material wording-to-behavior regression | >= 4.1/5 and < 4.3/5 | < 4.1/5 or unsafe over-answering pattern appears | Must fail safely when evidence is weak. |

Notes:
- `pp` = percentage points.
- If a metric is temporarily unavailable, rollout is **paused** (not promoted) until measurement is restored.

## Immediate rollback triggers (no warning period)

Rollback immediately to last-known-good if any of these occur:

1. Malformed output/contract break affects user-facing responses and is repeatable.
2. Any critical legal-safety incident: clearly unsupported confident legal claim in reviewed live sample with high severity.
3. Missing/invalid citation structure appears in repeated sampled outputs where citations are required.
4. Data integrity issue makes quality metrics untrustworthy for current release window.

## Warning-only conditions (pause and investigate)

- Single-window warning-band latency or cost drift without quality regression.
- Non-Tier-1 family drift that does not cross Tier 1 or global rollback bands.
- Isolated severe sample that does not repeat and has a plausible one-off cause.

If warning conditions persist across two consecutive windows, escalate to rollback decision.

## Rollback execution standard

When rollback is triggered:

1. Freeze promotion immediately (no further ramp progression).
2. Revert to last accepted release version.
3. Verify restore health with smoke queries + trace contract checks.
4. Log incident note with: trigger metric, measured value, baseline value, rollback timestamp, next remediation action.

## Post-rollback review notes (required)

Record at minimum:

- Trigger(s) and exact measured values.
- Affected family/families and user-visible impact.
- Whether regression was detectable offline, staging-only, or live-only.
- Candidate fix plan and additional guardrail to prevent recurrence.
- Re-entry criteria for retrying release.

