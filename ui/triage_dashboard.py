from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

import streamlit as st

from evals.reports.triage_workflow import (
    FAMILY_OPTIONS,
    FAILURE_TAXONOMY_OPTIONS,
    SEVERITY_OPTIONS,
    TRIAGE_STORE_PATH,
    TriageQueueItem,
    TriageRecord,
    append_regression_case_draft,
    build_regression_case_draft,
    discover_eval_run_files,
    discover_regression_dataset_files,
    find_existing_regression_case,
    load_triage_queue,
    load_triage_records,
    save_triage_record,
)


def render_triage_dashboard() -> None:
    st.title("Failure Triage Workflow")
    st.caption("Local-first triage queue for turning bad answers into classified, reproducible regression candidates.")

    run_dir = st.text_input("Eval run directory", value=st.session_state.get("triage_run_dir", "evals/runs"))
    st.session_state["triage_run_dir"] = run_dir

    triage_store = st.text_input("Triage record store", value=str(st.session_state.get("triage_store_path", TRIAGE_STORE_PATH)))
    st.session_state["triage_store_path"] = triage_store

    run_files = discover_eval_run_files(run_dir)
    if not run_files:
        st.info("No run outputs found. Point this page to offline eval JSON outputs with failed cases.")
        return

    queue = load_triage_queue(run_files=run_files, triage_store_path=triage_store)
    if not queue:
        st.success("No failure candidates found in current run outputs.")
        return

    _render_queue_summary(queue)

    status_filter = st.selectbox("Queue filter", options=["all", "untriaged", "triaged", "fixed", "regressed"], index=1)
    family_filter = st.selectbox("Family filter", options=["all"] + list(FAMILY_OPTIONS), index=0)
    severity_filter = st.selectbox("Severity filter", options=["all"] + list(SEVERITY_OPTIONS), index=0)

    triage_records = load_triage_records(triage_store)
    filtered_queue = _apply_filters(
        queue=queue,
        triage_records=triage_records,
        status_filter=status_filter,
        family_filter=family_filter,
        severity_filter=severity_filter,
    )
    if not filtered_queue:
        st.info("No queue items match current filters.")
        return

    selected = st.selectbox(
        "Failure candidate",
        options=filtered_queue,
        format_func=lambda item: f"[{item.status}] {item.case_id} | {item.family} | {item.failure_summary}",
    )
    _render_issue_detail(selected)
    _render_triage_form(selected, triage_records.get(selected.issue_id), triage_store)


def _render_queue_summary(queue: list[TriageQueueItem]) -> None:
    untriaged_count = sum(1 for item in queue if item.status == "untriaged")
    triaged_count = len(queue) - untriaged_count
    c1, c2, c3 = st.columns(3)
    c1.metric("Failure candidates", len(queue))
    c2.metric("Untriaged", untriaged_count)
    c3.metric("Triaged/fixed", triaged_count)


def _apply_filters(
    *,
    queue: list[TriageQueueItem],
    triage_records: dict[str, dict],
    status_filter: str,
    family_filter: str,
    severity_filter: str,
) -> list[TriageQueueItem]:
    rows: list[TriageQueueItem] = []
    for item in queue:
        if family_filter != "all" and item.family != family_filter:
            continue
        record = triage_records.get(item.issue_id, {})
        status = str(record.get("status") or item.status)
        severity = str(record.get("severity") or "")

        if status_filter != "all":
            if status_filter == "triaged" and status == "untriaged":
                continue
            elif status_filter != "triaged" and status != status_filter:
                continue
        if severity_filter != "all" and severity != severity_filter:
            continue
        rows.append(item)
    return rows


def _render_issue_detail(item: TriageQueueItem) -> None:
    st.subheader("Failure context")
    st.write(f"**Issue ID:** `{item.issue_id}`")
    st.write(f"**Case ID:** `{item.case_id}`")
    st.write(f"**Family:** `{item.family}`")
    st.write(f"**Query:** {item.query or '(missing query)'}")
    st.write(f"**Failure summary:** `{item.failure_summary}`")
    st.write(f"**Run file:** `{item.run_file}`")

    if item.trace_id:
        st.write(f"**Trace ID:** `{item.trace_id}`")
        if st.button("Open in Trace Debug dashboard", key=f"trace_link_{item.issue_id}"):
            st.session_state["trace_dashboard_run_dir"] = str(item.run_file.rsplit("/", 1)[0])
            st.info("Switched trace dashboard run directory. Open 'Trace Debug' page from sidebar.")
    else:
        st.caption("Trace link unavailable for this item (no trace_id found).")

    if item.dataset_file:
        st.write(f"**Dataset source:** `{item.dataset_file}`")


def _render_triage_form(item: TriageQueueItem, existing: dict | None, triage_store: str) -> None:
    st.subheader("Issue labeling flow")
    current = dict(existing or {})
    existing_reg_id, existing_reg_dataset = find_existing_regression_case(item, discover_regression_dataset_files())

    with st.form(key=f"triage_form_{item.issue_id}"):
        family = st.selectbox(
            "Family",
            options=list(FAMILY_OPTIONS),
            index=_index_or_default(FAMILY_OPTIONS, str(current.get("family") or item.family)),
        )
        severity = st.selectbox(
            "Severity",
            options=list(SEVERITY_OPTIONS),
            index=_index_or_default(SEVERITY_OPTIONS, str(current.get("severity") or "P2")),
        )
        failure_taxonomy = st.selectbox(
            "Failure taxonomy",
            options=list(FAILURE_TAXONOMY_OPTIONS),
            index=_index_or_default(
                FAILURE_TAXONOMY_OPTIONS,
                str(current.get("failure_taxonomy") or "retrieval_miss"),
            ),
        )
        reproduced = st.checkbox("Reproduced", value=bool(current.get("reproduced", False)))
        fixed_version = st.text_input("Fixed version", value=str(current.get("fixed_version") or ""))
        status = st.selectbox(
            "Status",
            options=["triaged", "fixed", "regressed"],
            index=_index_or_default(("triaged", "fixed", "regressed"), str(current.get("status") or "triaged")),
        )
        regression_case_id = st.text_input(
            "Regression case ID",
            value=str(current.get("regression_case_id") or existing_reg_id or ""),
        )
        regression_dataset_file = st.text_input(
            "Regression dataset file",
            value=str(current.get("regression_dataset_file") or existing_reg_dataset or ""),
        )
        notes = st.text_area("Notes", value=str(current.get("notes") or ""), height=110)

        save = st.form_submit_button("Save triage record")
        if save:
            record = TriageRecord(
                issue_id=item.issue_id,
                run_id=item.run_id,
                run_file=item.run_file,
                case_id=item.case_id,
                trace_id=item.trace_id,
                dataset_file=item.dataset_file,
                query=item.query,
                family=family,
                severity=severity,
                failure_taxonomy=failure_taxonomy,
                reproduced=reproduced,
                fixed_version=fixed_version or None,
                regression_case_id=regression_case_id or None,
                regression_dataset_file=regression_dataset_file or None,
                notes=notes.strip(),
                status=status,
                updated_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            save_triage_record(record, path=triage_store)
            st.success("Triage record saved.")
            st.json(asdict(record), expanded=False)

    st.subheader("Regression linkage")
    if existing_reg_id:
        st.success(f"Existing regression case detected: {existing_reg_id} ({existing_reg_dataset})")
    else:
        st.warning("No matching regression case detected yet.")

    if st.button("Create regression case draft", key=f"regression_draft_{item.issue_id}"):
        base_record = TriageRecord(
            issue_id=item.issue_id,
            run_id=item.run_id,
            run_file=item.run_file,
            case_id=item.case_id,
            trace_id=item.trace_id,
            dataset_file=item.dataset_file,
            query=item.query,
            family=str(current.get("family") or item.family),
            severity=str(current.get("severity") or "P2"),
            failure_taxonomy=str(current.get("failure_taxonomy") or "retrieval_miss"),
            reproduced=bool(current.get("reproduced", False)),
            fixed_version=_none_if_blank(current.get("fixed_version")),
            regression_case_id=_none_if_blank(current.get("regression_case_id")),
            regression_dataset_file=_none_if_blank(current.get("regression_dataset_file")),
            notes=str(current.get("notes") or ""),
            status=str(current.get("status") or "triaged"),
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        draft = build_regression_case_draft(item, base_record)
        append_regression_case_draft(draft)
        st.success("Regression draft appended to data/triage/regression_case_drafts.jsonl")
        st.json(draft, expanded=False)


def _index_or_default(options: tuple[str, ...] | list[str], value: str) -> int:
    items = list(options)
    try:
        return items.index(value)
    except ValueError:
        return 0


def _none_if_blank(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
