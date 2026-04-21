from __future__ import annotations

from typing import Any

import streamlit as st

from evals.reports.trace_dashboard_data import (
    build_trace_drilldown,
    discover_trace_run_files,
    load_trace_runs,
)


def render_trace_dashboard() -> None:
    st.title("Legal RAG Trace Debugging Dashboard")
    st.caption("Local-first trace drilldown to quickly identify failure layers for one bad answer.")

    default_dir = st.session_state.get("trace_dashboard_run_dir", "evals/runs")
    run_dir = st.text_input("Trace run directory", value=default_dir)
    st.session_state["trace_dashboard_run_dir"] = run_dir

    run_files = discover_trace_run_files(run_dir)
    if not run_files:
        st.info("No run JSON files found. Point this dashboard at offline eval outputs containing debug_payload traces.")
        return

    runs = load_trace_runs(run_files)
    if not runs:
        st.warning("Run files were found, but no valid cases could be loaded.")
        return

    selected_run = st.selectbox("Run", options=runs, format_func=lambda run: f"{run.run_id} ({run.source_path.name})")

    warning_only = st.checkbox("Show warning/failure traces only", value=False)
    families = sorted({str(case.get("family") or "unknown") for case in selected_run.cases})
    selected_family = st.selectbox("Family filter", options=["all"] + families)

    visible_cases: list[dict[str, Any]] = []
    for case in selected_run.cases:
        if selected_family != "all" and str(case.get("family") or "unknown") != selected_family:
            continue
        drilldown = build_trace_drilldown(case)
        if warning_only and drilldown.get("failure_layer") is None:
            continue
        visible_cases.append(case)

    if not visible_cases:
        st.info("No traces match current filters.")
        return

    selected_case = st.selectbox(
        "Case",
        options=visible_cases,
        format_func=lambda case: f"{case.get('case_id', case.get('id', 'unknown'))} | {case.get('family', 'unknown')}",
    )

    drilldown = build_trace_drilldown(selected_case)
    _render_failure_summary(drilldown)
    _render_stage_statuses(drilldown)

    _render_key_value("Classification", drilldown["classification"])
    _render_key_value("Rewrite", drilldown["rewrite"])
    _render_key_value("Decomposition Plan", drilldown["decomposition"])
    _render_key_value("Retrieval Candidates", drilldown["retrieval"])
    _render_key_value("Rerank Outputs", drilldown["rerank"])
    _render_key_value("Answerability Result", drilldown["answerability"])
    _render_key_value("Final Answer", drilldown["final_answer"])

    st.subheader("Citations")
    st.caption(f"Citation count: {drilldown['citation_count']}")
    if drilldown["citations"]:
        st.json(drilldown["citations"], expanded=False)
    else:
        st.info("No citations available.")

    st.subheader("Warnings")
    if drilldown["warnings"]:
        st.json(drilldown["warnings"], expanded=False)
    else:
        st.info("No warnings recorded.")

    with st.expander("Raw payload escape hatch", expanded=False):
        st.json(drilldown["raw"], expanded=False)


def _render_failure_summary(drilldown: dict[str, Any]) -> None:
    st.subheader("Likely failure layer")
    failure = drilldown.get("failure_layer")
    if failure:
        st.error(f"{failure['stage']} — {failure['status']} ({failure.get('reason') or 'no reason'})")
    else:
        st.success("No clear failure layer detected from structural trace signals.")


def _render_stage_statuses(drilldown: dict[str, Any]) -> None:
    st.subheader("Stage status timeline")
    for row in drilldown.get("stage_statuses", []):
        label = f"{row['stage']}: {row['status']}"
        reason = row.get("reason")
        if row["status"] in {"failed"}:
            st.error(f"{label} — {reason or 'failed'}")
        elif row["status"] in {"warning", "suspicious", "not_available"}:
            st.warning(f"{label} — {reason or 'needs review'}")
        else:
            st.success(label)


def _render_key_value(title: str, payload: dict[str, Any]) -> None:
    st.subheader(title)
    if payload:
        st.json(payload, expanded=False)
    else:
        st.info("Not available")
