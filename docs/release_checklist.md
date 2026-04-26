# Release Checklist (Ticket 30)

## Purpose

Operational go/no-go checklist for each release candidate. This checklist is the execution artifact; detailed trigger definitions live in `docs/rollback_criteria.md`, and staged promotion rules live in `docs/production_ramp_plan.md`.

## Release inputs required (before checklist starts)

- Candidate commit SHA/tag and pinned baseline release SHA/tag.
- Offline eval outputs for candidate and baseline (`evals/runs/*.json`).
- `compare_runs` output comparing baseline vs candidate.
- Staging smoke-query results and trace captures.
- Live-sample review set (sampled traffic or curated live-style prompts) and summary metrics.
- Active rollback table version (`docs/rollback_criteria.md`) confirmed unchanged or intentionally updated.

## Pre-release checks

- [ ] Scope check: this release does not bypass quality contract requirements in `docs/quality_contract.md`.
- [ ] Baseline/candidate artifacts are from comparable dataset and config.
- [ ] Tier 1 families are explicitly included in evaluation slices.
- [ ] Release owner identified (solo owner).

## Offline pass checklist (must pass before staging)

- [ ] Run offline eval on baseline and candidate with comparable settings.
- [ ] Run release comparison (`evals/reports/compare_runs.py`) baseline vs candidate.
- [ ] False confident rate check passes (no increase beyond pass threshold).
- [ ] Citation correctness check passes (no drop beyond pass threshold).
- [ ] Tier 1 family scores pass (no material regression per family).
- [ ] p95 latency remains within allowed pass band.
- [ ] Any warning-band metric has mitigation note and explicit owner follow-up date.

## Staging pass checklist (must pass before live sample)

- [ ] Representative smoke query set passes end-to-end.
- [ ] Traces are emitted and render correctly for required spans.
- [ ] Key debug fields are present (`family`, answerability fields, citations, warnings).
- [ ] No malformed output / contract-shape breaks found in staging samples.
- [ ] Failure/fallback pattern rates do not exceed warning threshold.
- [ ] Staging latency/cost are within pass or accepted warning bands.

## Live sample pass checklist (must pass before broad rollout)

- [ ] Limited live sample executed (or curated live-style sample if real traffic is unavailable).
- [ ] Sampled outputs remain grounded on manual review.
- [ ] False confident proxy remains below pass threshold.
- [ ] Citation behavior remains in pass band.
- [ ] Family routing + answerability behavior remains stable (no new Tier 1 drift).
- [ ] No severe new failure family appears in repeated samples.

## Rollback readiness confirmation (required before promotion)

- [ ] Rollback triggers for this release are explicitly documented and numeric/rule-based.
- [ ] Last-known-good version is identified and restorable.
- [ ] Rollback command/steps are rehearsed and time-to-restore is acceptable.
- [ ] On-call/decision role is clear (release owner).

## Final go / no-go signoff

Promote only if all are true:

- [ ] Offline pass complete.
- [ ] Staging pass complete.
- [ ] Live sample pass complete.
- [ ] Rollback criteria confirmed and actionable.
- [ ] Final decision recorded by release owner as one of: `GO`, `GO_WITH_WARNINGS`, `NO_GO`.

