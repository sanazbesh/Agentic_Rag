from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from evals.reports.human_review_queue import (
    DATASET_FEEDBACK_ACTIONS,
    REVIEW_ACTIONS,
    REVIEW_CATEGORIES,
    REVIEW_STATUSES,
    REVIEW_STORE_PATH,
    apply_review_decision,
    load_review_queue,
    load_review_records,
)


def render_review_queue_dashboard() -> None:
    st.title("Human Review Queue")
    st.caption("Local-first review queue for sensitive legal failures. Single-reviewer workflow with explicit decision actions.")

    run_dir = st.text_input("Eval run directory", value=st.session_state.get("review_run_dir", "evals/runs"))
    st.session_state["review_run_dir"] = run_dir

    review_store = st.text_input("Review record store", value=str(st.session_state.get("review_store_path", REVIEW_STORE_PATH)))
    st.session_state["review_store_path"] = review_store

    queue = load_review_queue(run_dir=run_dir, review_store_path=review_store)
    if not queue:
        st.info("No sensitive review candidates found in current local artifacts.")
        return

    _render_summary(queue)

    category_filter = st.selectbox("Category filter", options=["all"] + list(REVIEW_CATEGORIES), index=0)
    status_filter = st.selectbox("Status filter", options=["all"] + list(REVIEW_STATUSES), index=0)

    rows = [
        item
        for item in queue
        if (category_filter == "all" or item.category == category_filter)
        and (status_filter == "all" or item.review_status == status_filter)
    ]
    if not rows:
        st.info("No review items match current filters.")
        return

    selected = st.selectbox(
        "Review item",
        options=rows,
        format_func=lambda item: f"[{item.review_status}] {item.category} | {item.family} | {item.source_id}",
    )

    _render_review_item_detail(selected)
    existing = load_review_records(review_store).get(selected.review_item_id)
    _render_review_action_form(selected, existing=existing, review_store_path=review_store)


def _render_summary(queue: list) -> None:
    pending = sum(1 for item in queue if item.review_status == "pending_review")
    escalated = sum(1 for item in queue if item.review_status == "escalated")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sensitive review items", len(queue))
    c2.metric("Pending review", pending)
    c3.metric("Escalated", escalated)


def _render_review_item_detail(item: object) -> None:
    st.subheader("Review context")
    st.write(f"**Review item ID:** `{item.review_item_id}`")
    st.write(f"**Source:** `{item.source_type}` / `{item.source_id}`")
    st.write(f"**Category:** `{item.category}`")
    st.write(f"**Family:** `{item.family}`")
    st.write(f"**Query:** {item.query or '(missing query)'}")
    st.write(f"**Trigger reason(s):** {', '.join(item.trigger_reasons) if item.trigger_reasons else 'n/a'}")
    st.write(f"**Current status:** `{item.review_status}`")

    st.subheader("Answer and citations")
    st.write(item.final_answer or "(missing final answer)")
    if item.citations:
        st.json(item.citations, expanded=False)
    else:
        st.warning("No citations present.")

    st.subheader("Warnings")
    if item.warnings:
        st.json(item.warnings, expanded=False)
    else:
        st.info("No warnings recorded.")

    st.subheader("Trace linkage")
    if item.trace_id:
        st.write(f"**Trace ID:** `{item.trace_id}`")
        if st.button("Open in Trace Debug dashboard", key=f"review_trace_{item.review_item_id}"):
            st.session_state["trace_dashboard_run_dir"] = run_dir = st.session_state.get("review_run_dir", "evals/runs")
            st.info(f"Trace Debug run directory set to `{run_dir}`. Open Trace Debug from sidebar.")
    else:
        st.caption("Trace link unavailable for this review item.")


def _render_review_action_form(item: object, *, existing: dict | None, review_store_path: str) -> None:
    st.subheader("Review decision")
    current = dict(existing or {})

    with st.form(key=f"review_form_{item.review_item_id}"):
        action = st.selectbox(
            "Review action",
            options=list(REVIEW_ACTIONS),
            index=_index_or_default(REVIEW_ACTIONS, str(current.get("review_action") or "approve")),
        )
        family = st.text_input("Family", value=str(current.get("family") or item.family))
        category = st.selectbox(
            "Category (for relabel)",
            options=list(REVIEW_CATEGORIES),
            index=_index_or_default(REVIEW_CATEGORIES, str(current.get("relabeled_category") or item.category)),
        )
        failure_taxonomy = st.text_input("Failure taxonomy", value=str(current.get("failure_taxonomy") or ""))
        severity = st.text_input("Severity", value=str(current.get("severity") or ""))
        note = st.text_area("Reviewer note", value=str(current.get("reviewer_note") or ""), height=100)

        dataset_feedback_action = st.selectbox(
            "Dataset feedback",
            options=list(DATASET_FEEDBACK_ACTIONS),
            index=_index_or_default(
                DATASET_FEEDBACK_ACTIONS,
                str(current.get("dataset_feedback_action") or "none"),
            ),
        )
        regression_case_id = st.text_input("Regression case ID", value=str(current.get("regression_case_id") or ""))
        regression_dataset_file = st.text_input(
            "Regression dataset file",
            value=str(current.get("regression_dataset_file") or ""),
        )
        fixed_version = st.text_input("Fixed version", value=str(current.get("fixed_version") or ""))
        create_draft = st.checkbox("Create regression case draft", value=False)

        submitted = st.form_submit_button("Save review decision")
        if submitted:
            record = apply_review_decision(
                item=item,
                action=action,
                reviewer_note=note,
                review_store_path=review_store_path,
                family=family,
                category=category,
                failure_taxonomy=failure_taxonomy or None,
                severity=severity or None,
                dataset_feedback_action=dataset_feedback_action,
                regression_case_id=regression_case_id or None,
                regression_dataset_file=regression_dataset_file or None,
                fixed_version=fixed_version or None,
                create_regression_draft=create_draft,
            )
            st.success("Review decision saved.")
            st.json(asdict(record), expanded=False)


def _index_or_default(options: tuple[str, ...] | list[str], value: str) -> int:
    items = list(options)
    try:
        return items.index(value)
    except ValueError:
        return 0
