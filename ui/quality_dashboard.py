from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from evals.reports.quality_dashboard_data import (
    compare_runs,
    compute_family_pass_rates,
    compute_run_trends,
    compute_top_failing_datasets,
    discover_run_files,
    filter_cases_by_family,
    load_eval_runs,
    select_recent_runs,
)


def render_quality_dashboard() -> None:
    st.title("Legal RAG Quality Dashboard")
    st.caption("Local-first offline eval quality dashboard for legal-family reliability and release regressions.")

    default_dir = st.session_state.get("quality_dashboard_run_dir", "evals/runs")
    run_dir = st.text_input("Eval run directory", value=default_dir)
    st.session_state["quality_dashboard_run_dir"] = run_dir

    run_files = discover_run_files(run_dir)
    if not run_files:
        st.info("No run JSON files found. Point this dashboard at a directory containing offline eval outputs.")
        return

    runs = load_eval_runs(run_files)
    if not runs:
        st.warning("Run files were found, but no valid run payloads could be loaded.")
        return

    max_runs = len(runs)
    last_n = st.slider("Time filter: last N runs", min_value=1, max_value=max_runs, value=min(10, max_runs))
    visible_runs = select_recent_runs(runs, last_n=last_n)

    all_families = sorted({str(case.get("family") or "unknown") for run in visible_runs for case in run.cases})
    selected_families = st.multiselect("Family filter", options=all_families, default=[])

    filtered_cases = [
        case
        for run in visible_runs
        for case in filter_cases_by_family(run.cases, selected_families)
    ]

    _render_overview(filtered_cases, visible_runs, selected_families)
    _render_family_quality(filtered_cases)
    _render_trends(visible_runs, selected_families)
    _render_top_failing_datasets(filtered_cases)
    _render_release_comparison(visible_runs, selected_families)


def _render_overview(cases: list[dict], runs: list, families: list[str]) -> None:
    st.subheader("Overview")
    total_cases = len(cases)
    total_passed = sum(1 for case in cases if _case_passed(case))
    pass_rate = (total_passed / total_cases) if total_cases else 0.0

    trends = compute_run_trends(runs, families=families)
    latest = trends[-1] if trends else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total evaluated cases", total_cases)
    c2.metric("Overall pass rate", _pct(pass_rate))
    c3.metric("False confident rate", _pct_or_na(latest.false_confident_rate if latest else None))
    c4.metric("Citation correctness", _pct_or_na(latest.citation_correctness_rate if latest else None))
    c5.metric("Safe failure", _pct_or_na(latest.safe_failure_rate if latest else None))


def _render_family_quality(cases: list[dict]) -> None:
    st.subheader("Family quality")
    rows = compute_family_pass_rates(cases)
    if not rows:
        st.info("No cases available after filters.")
        return
    table = pd.DataFrame(rows)
    if not table.empty:
        table["pass_rate"] = table["pass_rate"].map(_pct)
        table["fail_rate"] = table["fail_rate"].map(_pct)
    st.dataframe(table, use_container_width=True)


def _render_trends(runs: list, families: list[str]) -> None:
    st.subheader("Trend view")
    trends = compute_run_trends(runs, families=families)
    if not trends:
        st.info("No trend data available.")
        return

    df = pd.DataFrame(
        [
            {
                "run": point.run_id,
                "false_confident_rate": point.false_confident_rate,
                "citation_correctness_rate": point.citation_correctness_rate,
                "safe_failure_rate": point.safe_failure_rate,
            }
            for point in trends
        ]
    )
    st.line_chart(df.set_index("run")[["false_confident_rate", "citation_correctness_rate", "safe_failure_rate"]])

    unavailable: list[str] = []
    if df["false_confident_rate"].isna().all():
        unavailable.append("false confident")
    if df["citation_correctness_rate"].isna().all():
        unavailable.append("citation correctness")
    if df["safe_failure_rate"].isna().all():
        unavailable.append("safe failure")
    if unavailable:
        st.caption(f"Unavailable in current run artifacts: {', '.join(unavailable)}")


def _render_top_failing_datasets(cases: list[dict]) -> None:
    st.subheader("Top failing datasets")
    rows = compute_top_failing_datasets(cases)
    if not rows:
        st.info("No dataset-level failures available.")
        return
    table = pd.DataFrame(rows)
    if not table.empty:
        table["fail_rate"] = table["fail_rate"].map(_pct)
    st.dataframe(table, use_container_width=True)


def _render_release_comparison(runs: list, families: list[str]) -> None:
    st.subheader("Release comparison")
    if len(runs) < 2:
        st.info("Need at least two runs to compare releases.")
        return

    run_options = [run.run_id for run in runs]
    default_baseline_index = max(0, len(run_options) - 2)
    default_candidate_index = len(run_options) - 1

    col_a, col_b = st.columns(2)
    baseline_id = col_a.selectbox("Baseline run", run_options, index=default_baseline_index)
    candidate_id = col_b.selectbox("Candidate run", run_options, index=default_candidate_index)

    baseline = next(run for run in runs if run.run_id == baseline_id)
    candidate = next(run for run in runs if run.run_id == candidate_id)
    comparison = compare_runs(baseline, candidate, families=families)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall pass delta", _signed_pct(comparison["overall_delta"]))
    m2.metric("False confident delta", _signed_pct_or_na(comparison["false_confident_delta"]))
    m3.metric("Citation correctness delta", _signed_pct_or_na(comparison["citation_correctness_delta"]))
    m4.metric("Safe failure delta", _signed_pct_or_na(comparison["safe_failure_delta"]))

    st.caption("Family-level deltas")
    family_df = pd.DataFrame(comparison["family_deltas"])
    if not family_df.empty:
        family_df["baseline_pass_rate"] = family_df["baseline_pass_rate"].map(_pct)
        family_df["candidate_pass_rate"] = family_df["candidate_pass_rate"].map(_pct)
        family_df["delta"] = family_df["delta"].map(_signed_pct)
    st.dataframe(family_df, use_container_width=True)

    st.caption("Top regressions introduced")
    regressions_df = pd.DataFrame(comparison["top_regressions"])
    if regressions_df.empty:
        st.write("No new dataset-level regressions detected.")
    else:
        regressions_df["baseline_fail_rate"] = regressions_df["baseline_fail_rate"].map(_pct)
        regressions_df["candidate_fail_rate"] = regressions_df["candidate_fail_rate"].map(_pct)
        st.dataframe(regressions_df, use_container_width=True)


def _case_passed(case: dict) -> bool:
    if str(case.get("runner_status") or "") != "ok":
        return False
    for group_name in ("deterministic_eval_results", "llm_judge_results"):
        group = case.get(group_name)
        if not isinstance(group, dict):
            continue
        for result in group.values():
            if not isinstance(result, dict):
                continue
            if result.get("status") in {"skipped"}:
                continue
            if result.get("status") == "error":
                return False
            for key in ("passed", "is_correct"):
                if isinstance(result.get(key), bool) and result[key] is False:
                    return False
    return True


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _pct_or_na(value: float | None) -> str:
    if value is None:
        return "N/A"
    return _pct(value)


def _signed_pct(value: float) -> str:
    return f"{value * 100:+.1f}%"


def _signed_pct_or_na(value: float | None) -> str:
    if value is None:
        return "N/A"
    return _signed_pct(value)
