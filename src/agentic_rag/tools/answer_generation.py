"""Grounded legal answer generation from retrieved parent-chunk context.

This module implements a thin `generate_answer(context, query)` tool that:
- synthesizes answers strictly from supplied context (no retrieval, no memory)
- preserves legal qualifiers/limitations through extractive evidence use
- returns typed, traceable citations for downstream auditability
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agentic_rag.tools.evidence_units import EvidenceUnit, build_evidence_units

@dataclass(slots=True, frozen=True)
class AnswerCitation:
    """Traceable citation to a single parent chunk supporting an answer claim."""

    parent_chunk_id: str
    document_id: str | None
    source_name: str | None
    heading: str | None
    supporting_excerpt: str | None


@dataclass(slots=True, frozen=True)
class GenerateAnswerResult:
    """Structured grounded-answer output.

    The answer is document-grounded, explicitly marks insufficiency, and keeps
    citations separate from prose for reliable downstream traceability.
    """

    answer_text: str
    grounded: bool
    sufficient_context: bool
    citations: list[AnswerCitation]
    warnings: list[str]


@dataclass(slots=True)
class _ChronologyEvent:
    parent_chunk_id: str
    document_id: str | None
    source_name: str | None
    heading: str | None
    event_date: datetime
    event_date_text: str
    event_text: str
    supporting_excerpt: str


@dataclass(slots=True)
class LegalAnswerSynthesizer:
    """Deterministic extractive legal answer synthesizer.

    Grounded answering policy:
    - only uses provided parent/chunk context
    - forms claims from extracted sentences, not external knowledge
    - marks insufficiency when context cannot fully answer the query
    """

    min_query_overlap_for_relevance: int = 1
    max_support_points: int = 4
    qualifier_terms: tuple[str, ...] = (
        "unless",
        "except",
        "subject to",
        "provided that",
        "notwithstanding",
        "only if",
    )

    def generate(self, context: Sequence[object], query: str) -> GenerateAnswerResult:
        """Generate a grounded legal answer from retrieved context only."""

        try:
            normalized_query = (query or "").strip()
            normalized_context = [_unit_to_context_row(unit) for unit in build_evidence_units(context)]

            if not normalized_context:
                return GenerateAnswerResult(
                    answer_text=(
                        "Direct answer: No relevant information was retrieved for this question.\n\n"
                        "Supporting points:\n"
                        "- No parent chunks were provided to support an answer.\n\n"
                        "Caveats / limitations:\n"
                        "- The question cannot be answered from the available context."
                    ),
                    grounded=False,
                    sufficient_context=False,
                    citations=[],
                    warnings=["insufficient_context: no retrieved parent chunks"],
                )

            matter_metadata_response = self._generate_matter_metadata_answer(normalized_context, normalized_query)
            if matter_metadata_response is not None:
                return matter_metadata_response
            party_role_response = self._generate_party_role_answer(normalized_context, normalized_query)
            if party_role_response is not None:
                return party_role_response
            correspondence_response = self._generate_correspondence_litigation_answer(normalized_context, normalized_query)
            if correspondence_response is not None:
                return correspondence_response
            lifecycle_response = self._generate_employment_lifecycle_answer(normalized_context, normalized_query)
            if lifecycle_response is not None:
                return lifecycle_response
            chronology_response = self._generate_chronology_answer(normalized_context, normalized_query)
            if chronology_response is not None:
                return chronology_response

            ranked = self._rank_relevant_chunks(normalized_context, normalized_query)
            if not ranked:
                return self._insufficient_response(
                    reason="insufficient_context: retrieved chunks do not address the query"
                )

            selected = ranked[: self.max_support_points]
            citations: list[AnswerCitation] = []
            supporting_lines: list[str] = []

            for item in selected:
                excerpt = self._best_excerpt(item["text"], normalized_query)
                if not excerpt:
                    continue
                citations.append(
                    AnswerCitation(
                        parent_chunk_id=item["parent_chunk_id"],
                        document_id=item.get("document_id"),
                        source_name=item.get("source_name"),
                        heading=item.get("heading"),
                        supporting_excerpt=excerpt,
                    )
                )
                heading = item.get("heading") or "(no heading)"
                supporting_lines.append(f"- {heading}: {excerpt}")

            if not citations:
                return self._insufficient_response(
                    reason="insufficient_context: no supported claims could be extracted"
                )

            direct = self._direct_answer(citations)
            caveats: list[str] = []

            sufficient_context = self._is_fully_answerable(ranked, normalized_query)
            if not sufficient_context:
                caveats.append("- The retrieved context appears partial for the full question.")

            qualifier_warning = self._check_qualifier_coverage(selected, citations)
            grounded = bool(citations) and qualifier_warning is None
            warnings: list[str] = []
            if qualifier_warning is not None:
                warnings.append(qualifier_warning)
                caveats.append("- Material legal qualifiers may be incompletely represented in the extracted support.")

            answer_text = (
                f"Direct answer: {direct}\n\n"
                "Supporting points:\n"
                + "\n".join(supporting_lines)
                + "\n\nCaveats / limitations:\n"
                + ("\n".join(caveats) if caveats else "- None identified within the retrieved context.")
            )

            if not citations:
                grounded = False

            return GenerateAnswerResult(
                answer_text=answer_text,
                grounded=grounded,
                sufficient_context=sufficient_context,
                citations=citations,
                warnings=warnings,
            )
        except Exception as exc:  # pragma: no cover - exercised in explicit fallback test
            return self._failure_response(str(exc))

    def _generate_party_role_answer(
        self,
        context: Sequence[dict[str, Any]],
        query: str,
    ) -> GenerateAnswerResult | None:
        lowered_query = query.lower()
        if not self._is_party_role_question(lowered_query):
            return None

        role_assignment = self._resolve_party_roles_from_intro(context)
        if role_assignment is None:
            return GenerateAnswerResult(
                answer_text=(
                    "Direct answer: The retrieved context includes party-related language, but roles cannot be assigned "
                    "reliably from the available agreement-introduction evidence.\n\n"
                    "Supporting points:\n"
                    "- Party-role assignment could not be resolved with sufficient confidence.\n\n"
                    "Caveats / limitations:\n"
                    "- A reliable employer/employee/party mapping requires clearer introductory role labels."
                ),
                grounded=False,
                sufficient_context=False,
                citations=[],
                warnings=["party_role_assignment_unresolved"],
            )

        citation = AnswerCitation(
            parent_chunk_id=role_assignment.source_parent_chunk_id,
            document_id=role_assignment.document_id,
            source_name=role_assignment.source_name,
            heading=role_assignment.heading,
            supporting_excerpt=role_assignment.supporting_excerpt,
        )
        source_line = f"- {role_assignment.heading or '(no heading)'}: {role_assignment.supporting_excerpt}"

        if "who is the employer" in lowered_query:
            if not role_assignment.employer:
                return self._insufficient_party_role_with_citation(citation, source_line)
            return self._party_role_success(
                direct=f"The employer is {role_assignment.employer}.",
                citation=citation,
                source_line=source_line,
            )

        if "who is the employee" in lowered_query:
            if not role_assignment.employee:
                return self._insufficient_party_role_with_citation(citation, source_line)
            return self._party_role_success(
                direct=f"The employee is {role_assignment.employee}.",
                citation=citation,
                source_line=source_line,
            )

        if "who are the parties" in lowered_query:
            if len(role_assignment.parties) < 2:
                return self._insufficient_party_role_with_citation(citation, source_line)
            return self._party_role_success(
                direct=f"The parties are {role_assignment.parties[0]} and {role_assignment.parties[1]}.",
                citation=citation,
                source_line=source_line,
            )

        if "which company is this agreement for" in lowered_query:
            company = role_assignment.employer or self._pick_company_party(role_assignment.parties)
            if not company:
                return self._insufficient_party_role_with_citation(citation, source_line)
            return self._party_role_success(
                direct=f"The agreement appears to be for {company}.",
                citation=citation,
                source_line=source_line,
            )

        between_match = re.search(r"\bis\s+this\s+agreement\s+between\s+(.+?)\s+and\s+(.+?)\??$", lowered_query)
        if between_match:
            requested_a = self._normalize_party_text(between_match.group(1))
            requested_b = self._normalize_party_text(between_match.group(2))
            extracted = {self._normalize_party_text(party) for party in role_assignment.parties}
            both_match = requested_a in extracted and requested_b in extracted
            direct = (
                "Yes, the agreement-introduction evidence identifies those two parties."
                if both_match
                else "No, the agreement-introduction evidence does not identify that exact pair of parties."
            )
            return self._party_role_success(direct=direct, citation=citation, source_line=source_line)

        return None

    def _generate_matter_metadata_answer(
        self,
        context: Sequence[dict[str, Any]],
        query: str,
    ) -> GenerateAnswerResult | None:
        lowered_query = query.lower()
        target = self._metadata_target(lowered_query)
        if target is None:
            return None

        match = self._resolve_matter_metadata_value(context, target)
        if match is None:
            return self._insufficient_response("insufficient_context: metadata-responsive evidence not found")

        citation = AnswerCitation(
            parent_chunk_id=match["parent_chunk_id"],
            document_id=match.get("document_id"),
            source_name=match.get("source_name"),
            heading=match.get("heading"),
            supporting_excerpt=match["excerpt"],
        )
        source_line = f"- {match.get('heading') or '(no heading)'}: {match['excerpt']}"
        direct = self._metadata_direct_answer(target, match["value"])

        return GenerateAnswerResult(
            answer_text=(
                f"Direct answer: {direct}\n\n"
                "Supporting points:\n"
                f"{source_line}\n\n"
                "Caveats / limitations:\n"
                "- Response is limited to metadata-responsive caption/header/matter-information evidence in retrieved context."
            ),
            grounded=True,
            sufficient_context=True,
            citations=[citation],
            warnings=[],
        )

    def _party_role_success(
        self,
        *,
        direct: str,
        citation: AnswerCitation,
        source_line: str,
    ) -> GenerateAnswerResult:
        return GenerateAnswerResult(
            answer_text=(
                f"Direct answer: {direct}\n\n"
                "Supporting points:\n"
                f"{source_line}\n\n"
                "Caveats / limitations:\n"
                "- Role assignment is based on the retrieved agreement-introduction party language."
            ),
            grounded=True,
            sufficient_context=True,
            citations=[citation],
            warnings=[],
        )

    def _insufficient_party_role_with_citation(
        self,
        citation: AnswerCitation,
        source_line: str,
    ) -> GenerateAnswerResult:
        return GenerateAnswerResult(
            answer_text=(
                "Direct answer: The retrieved party evidence is not sufficient to assign the requested role safely.\n\n"
                "Supporting points:\n"
                f"{source_line}\n\n"
                "Caveats / limitations:\n"
                "- The text identifies parties but does not reliably label the requested role."
            ),
            grounded=False,
            sufficient_context=False,
            citations=[citation],
            warnings=["party_role_assignment_unresolved"],
        )

    def _is_party_role_question(self, lowered_query: str) -> bool:
        patterns = (
            r"\bwho\s+is\s+the\s+employer\b",
            r"\bwho\s+is\s+the\s+employee\b",
            r"\bwho\s+are\s+the\s+parties\b",
            r"\bwhich\s+company\s+is\s+this\s+agreement\s+for\b",
            r"\bis\s+this\s+agreement\s+between\b",
        )
        return any(re.search(pattern, lowered_query) for pattern in patterns)

    def _metadata_target(self, lowered_query: str) -> str | None:
        patterns: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("file_number", (r"\bfile\s+number\b", r"\bdocket\s+number\b", r"\bcourt\s+file\b")),
            ("jurisdiction", (r"\bjurisdiction\b",)),
            ("court", (r"\bwhat\s+court\b", r"\bwhich\s+court\b", r"\bcourt\s+is\s+involved\b")),
            ("client", (r"\bwho\s+is\s+the\s+client\b", r"\bclient\b")),
            ("case_or_matter_name", (r"\bcase\s+name\b", r"\bmatter\s+name\b")),
            ("matter_about", (r"\bwhat\s+is\s+this\s+matter\s+about\b", r"\bwhat\s+is\s+this\s+document\s+about\b")),
        )
        for target, target_patterns in patterns:
            if any(re.search(pattern, lowered_query) for pattern in target_patterns):
                return target
        return None

    def _resolve_matter_metadata_value(
        self,
        context: Sequence[dict[str, Any]],
        target: str,
    ) -> dict[str, Any] | None:
        for item in context:
            haystack = " ".join(
                (
                    str(item.get("heading") or ""),
                    str(item.get("text") or ""),
                    " ".join(f"{k}: {v}" for k, v in dict(item.get("metadata") or {}).items()),
                )
            )
            value = self._extract_metadata_value(haystack, target, dict(item.get("metadata") or {}))
            if not value:
                continue
            excerpt = self._best_excerpt(str(item.get("text") or haystack), target.replace("_", " "))
            return {
                "value": value,
                "excerpt": excerpt or value,
                "parent_chunk_id": item.get("parent_chunk_id", ""),
                "document_id": item.get("document_id"),
                "source_name": item.get("source_name"),
                "heading": item.get("heading"),
            }
        return None

    def _extract_metadata_value(self, haystack: str, target: str, metadata: dict[str, Any]) -> str | None:
        lower = haystack.lower()
        if target == "file_number":
            for key in ("file_number", "court_file_number", "docket_number"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            match = re.search(r"\b(?:file|court file|docket|case)\s*(?:no\.?|number)\s*[:#-]?\s*([a-z0-9\-\/]+)", lower)
            return match.group(1).strip().upper() if match else None
        if target == "jurisdiction":
            value = metadata.get("jurisdiction")
            if isinstance(value, str) and value.strip():
                return value.strip()
            match = re.search(r"\bjurisdiction\s*[:\-]\s*([a-z][a-z\s]+)", lower)
            if match:
                return match.group(1).strip().title()
            match = re.search(r"\bgoverned by the laws of\s+([a-z][a-z\s]+)", lower)
            return match.group(1).strip().title() if match else None
        if target == "court":
            value = metadata.get("court")
            if isinstance(value, str) and value.strip():
                return value.strip()
            match = re.search(r"\b((?:supreme|district|chancery|high|appeal)\s+court(?:\s+of\s+[a-z\s]+)?)\b", lower)
            return match.group(1).strip().title() if match else None
        if target == "client":
            for key in ("client", "client_name"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            match = re.search(r"\bclient\s*[:\-]\s*([a-z0-9.,&()' -]+)", haystack, flags=re.IGNORECASE)
            return match.group(1).strip() if match else None
        if target == "case_or_matter_name":
            for key in ("case_name", "matter_name", "title"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            match = re.search(r"\b(?:case|matter)\s+name\s*[:\-]\s*([^\n.;]+)", haystack, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
            match = re.search(r"\b([A-Z][A-Za-z0-9&.,' -]+\s+v\.\s+[A-Z][A-Za-z0-9&.,' -]+)\b", haystack)
            return match.group(1).strip() if match else None
        if target == "matter_about":
            for key in ("subject", "matter"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            match = re.search(r"\b(?:subject|re|regarding|matter)\s*[:\-]\s*([^\n.;]+)", haystack, flags=re.IGNORECASE)
            return match.group(1).strip() if match else None
        return None

    def _metadata_direct_answer(self, target: str, value: str) -> str:
        if target == "file_number":
            return f"The file number is {value}."
        if target == "jurisdiction":
            return f"The applicable jurisdiction is {value}."
        if target == "court":
            return f"The court involved is {value}."
        if target == "client":
            return f"The client is {value}."
        if target == "case_or_matter_name":
            return f"The case/matter name is {value}."
        return f"The matter/document is about {value}."

    def _generate_correspondence_litigation_answer(
        self,
        context: Sequence[dict[str, Any]],
        query: str,
    ) -> GenerateAnswerResult | None:
        lowered_query = query.lower()
        target = self._correspondence_litigation_target(lowered_query)
        if target is None:
            return None

        if target == "procedural_history":
            milestones = self._extract_procedural_milestones(context)
            if not milestones:
                return self._insufficient_response("insufficient_context: procedural milestone evidence not found")
            lines: list[str] = []
            citations: list[AnswerCitation] = []
            for milestone in milestones[: self.max_support_points]:
                lines.append(f"- {milestone['label']}")
                citations.append(
                    AnswerCitation(
                        parent_chunk_id=str(milestone["parent_chunk_id"]),
                        document_id=milestone.get("document_id"),
                        source_name=milestone.get("source_name"),
                        heading=milestone.get("heading"),
                        supporting_excerpt=str(milestone["excerpt"]),
                    )
                )
            return GenerateAnswerResult(
                answer_text=(
                    "Direct answer: The retrieved context supports the following procedural milestones.\n\n"
                    "Supporting points:\n"
                    + "\n".join(lines)
                    + "\n\nCaveats / limitations:\n"
                    "- Response is limited to correspondence/procedural milestone evidence in the retrieved context."
                ),
                grounded=True,
                sufficient_context=True,
                citations=citations,
                warnings=[],
            )

        match = self._resolve_correspondence_litigation_value(context, target)
        if match is None:
            return self._insufficient_response("insufficient_context: correspondence/procedural-responsive evidence not found")

        if self._procedural_when_date_required(lowered_query, target) and not self._contains_concrete_date(str(match["value"])):
            return self._insufficient_response("insufficient_context: correspondence/procedural date evidence not found")

        citation = AnswerCitation(
            parent_chunk_id=match["parent_chunk_id"],
            document_id=match.get("document_id"),
            source_name=match.get("source_name"),
            heading=match.get("heading"),
            supporting_excerpt=match["excerpt"],
        )
        source_line = f"- {match.get('heading') or '(no heading)'}: {match['excerpt']}"
        direct = self._correspondence_litigation_direct_answer(target, str(match["value"]))

        return GenerateAnswerResult(
            answer_text=(
                f"Direct answer: {direct}\n\n"
                "Supporting points:\n"
                f"{source_line}\n\n"
                "Caveats / limitations:\n"
                "- Response is limited to correspondence/procedural milestone evidence in the retrieved context."
            ),
            grounded=True,
            sufficient_context=True,
            citations=[citation],
            warnings=[],
        )

    def _correspondence_litigation_target(self, lowered_query: str) -> str | None:
        if any(token in lowered_query for token in ("what letters were sent", "what emails were sent", "communications were sent")):
            return "communications_sent"
        if "deadline" in lowered_query and any(token in lowered_query for token in ("demand", "demanded")):
            return "demand_deadline"
        if "claim" in lowered_query and "filed" in lowered_query:
            return "claim_filed"
        if "defen" in lowered_query and any(token in lowered_query for token in ("due", "filed")):
            return "defence_due_or_filed"
        if any(token in lowered_query for token in ("procedural history", "what happened procedurally", "procedural status")):
            return "procedural_history"
        return None

    def _resolve_correspondence_litigation_value(
        self,
        context: Sequence[dict[str, Any]],
        target: str,
    ) -> dict[str, Any] | None:
        for item in context:
            text = str(item.get("text") or "")
            heading = str(item.get("heading") or "")
            haystack = f"{heading}\n{text}".strip()
            value = self._extract_correspondence_litigation_value(haystack, target)
            if not value:
                continue
            excerpt = self._best_excerpt(text or haystack, value)
            return {
                "value": value,
                "excerpt": excerpt or value,
                "parent_chunk_id": item.get("parent_chunk_id", ""),
                "document_id": item.get("document_id"),
                "source_name": item.get("source_name"),
                "heading": item.get("heading"),
            }
        return None

    def _extract_correspondence_litigation_value(self, haystack: str, target: str) -> str | None:
        lowered = haystack.lower()
        date_pattern = r"(?:\d{4}-\d{2}-\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},\s*\d{4})"
        if target == "communications_sent":
            match = re.search(rf"\b(?:letter|email|correspondence)[^.\n;]{{0,120}}(?:sent|delivered|emailed|issued|dated)[^.\n;]{{0,120}}({date_pattern})", haystack, flags=re.IGNORECASE)
            if match:
                return match.group(0).strip()
            return None
        if target == "demand_deadline":
            match = re.search(rf"\bdemand[^.\n;]{{0,160}}(?:by|no later than|within|deadline)[^.\n;]{{0,120}}({date_pattern}|\d+\s+days?)", haystack, flags=re.IGNORECASE)
            if match:
                return match.group(0).strip()
            return None
        if target == "claim_filed":
            match = re.search(rf"\b(?:statement\s+of\s+claim|claim)[^.\n;]{{0,100}}(?:filed|issued)[^.\n;]{{0,80}}({date_pattern})", haystack, flags=re.IGNORECASE)
            if match:
                return match.group(0).strip()
            return None
        if target == "defence_due_or_filed":
            match = re.search(rf"\b(?:statement\s+of\s+defen(?:c|s)e|defen(?:c|s)e)[^.\n;]{{0,120}}(?:due|filed|served)[^.\n;]{{0,90}}({date_pattern}|\d+\s+days?)", haystack, flags=re.IGNORECASE)
            if match:
                return match.group(0).strip()
            return None
        return None

    def _extract_procedural_milestones(self, context: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        milestones: list[dict[str, Any]] = []
        marker_pattern = re.compile(
            r"\b(?:statement of claim|statement of defen(?:c|s)e|defen(?:c|s)e|pleading|served|service|filed|issued|default notice|settlement discussion|court filing)\b",
            flags=re.IGNORECASE,
        )
        for item in context:
            text = str(item.get("text") or "")
            if not text:
                continue
            match = marker_pattern.search(text)
            if not match:
                continue
            excerpt = self._best_excerpt(text, match.group(0)) or text
            date_match = re.search(
                r"\b(?:\d{4}-\d{2}-\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},\s*\d{4})\b",
                text,
                flags=re.IGNORECASE,
            )
            if date_match:
                label = f"{date_match.group(0)}: {excerpt}"
            else:
                label = excerpt
            milestones.append(
                {
                    "label": label,
                    "excerpt": excerpt,
                    "parent_chunk_id": str(item.get("parent_chunk_id") or ""),
                    "document_id": item.get("document_id"),
                    "source_name": item.get("source_name"),
                    "heading": item.get("heading"),
                }
            )
        return milestones

    def _correspondence_litigation_direct_answer(self, target: str, value: str) -> str:
        if target == "communications_sent":
            return f"The dated communications evidence states: {value}."
        if target == "demand_deadline":
            return f"The demand/deadline evidence states: {value}."
        if target == "claim_filed":
            return f"The claim filing evidence states: {value}."
        return f"The defence due/filed evidence states: {value}."

    def _procedural_when_date_required(self, lowered_query: str, target: str) -> bool:
        if "when" not in lowered_query:
            return False
        return target in {"communications_sent", "claim_filed", "defence_due_or_filed", "demand_deadline"}

    def _generate_chronology_answer(
        self,
        context: Sequence[dict[str, Any]],
        query: str,
    ) -> GenerateAnswerResult | None:
        lowered_query = query.lower()
        if not self._is_chronology_question(lowered_query):
            return None

        events = self._extract_chronology_events(context)
        if not events:
            return self._insufficient_response("insufficient_context: chronology-responsive dated events not found")

        selected: list[_ChronologyEvent] = []
        direct = ""
        missing_reason = "insufficient_context: chronology query not supported by available dated event evidence"

        if "what happened first" in lowered_query or "first event" in lowered_query:
            if len(events) < 2:
                return self._insufficient_response(missing_reason)
            selected = [events[0]]
            direct = f"The first dated event is {selected[0].event_text} on {selected[0].event_date_text}."
        elif "what happened last" in lowered_query or "last event" in lowered_query:
            if len(events) < 2:
                return self._insufficient_response(missing_reason)
            selected = [events[-1]]
            direct = f"The last dated event is {selected[0].event_text} on {selected[0].event_date_text}."
        elif "after" in lowered_query:
            anchor = self._extract_anchor_phrase(lowered_query, "after")
            anchor_event = self._find_anchor_event(events, anchor)
            if anchor_event is None:
                return self._insufficient_response(missing_reason)
            selected = [event for event in events if event.event_date > anchor_event.event_date]
            if not selected:
                return self._insufficient_response(missing_reason)
            direct = f"After {anchor_event.event_text} ({anchor_event.event_date_text}), the next dated events are listed below."
        elif "before" in lowered_query:
            anchor = self._extract_anchor_phrase(lowered_query, "before")
            anchor_event = self._find_anchor_event(events, anchor)
            if anchor_event is None:
                return self._insufficient_response(missing_reason)
            selected = [event for event in events if event.event_date < anchor_event.event_date]
            if not selected:
                return self._insufficient_response(missing_reason)
            direct = f"Before {anchor_event.event_text} ({anchor_event.event_date_text}), the dated events are listed below."
        elif "between" in lowered_query:
            range_dates = self._extract_query_dates(query)
            if len(range_dates) < 2:
                return self._insufficient_response(missing_reason)
            start, end = min(range_dates), max(range_dates)
            selected = [event for event in events if start <= event.event_date <= end]
            if not selected:
                return self._insufficient_response(missing_reason)
            direct = f"Between {start.strftime('%B %d, %Y')} and {end.strftime('%B %d, %Y')}, the supported dated events are:"
        elif "employment start" in lowered_query or "employment began" in lowered_query or "start date" in lowered_query:
            for event in events:
                lowered = event.event_text.lower()
                if "employment" in lowered and any(token in lowered for token in ("start", "commenc", "effective", "begin")):
                    selected = [event]
                    break
            if not selected:
                return self._insufficient_response(missing_reason)
            direct = f"The employment start-related dated event is {selected[0].event_date_text}."
        elif "termination notice" in lowered_query or ("termination" in lowered_query and "notice" in lowered_query):
            for event in events:
                lowered = event.event_text.lower()
                if "termination" in lowered and any(token in lowered for token in ("notice", "letter", "email", "served")):
                    selected = [event]
                    break
            if not selected:
                return self._insufficient_response(missing_reason)
            direct = f"The termination notice-related dated event is {selected[0].event_date_text}."
        elif "all dated events" in lowered_query or "timeline" in lowered_query or "chronology" in lowered_query:
            selected = events
            direct = "The dated chronology evidence in the retrieved context is listed below."
        elif "when did" in lowered_query or "when was" in lowered_query:
            selected = events[:1]
            direct = f"The best-supported dated event answer is {selected[0].event_date_text}."
        else:
            return None

        if not selected:
            return self._insufficient_response(missing_reason)

        citations: list[AnswerCitation] = []
        lines: list[str] = []
        for event in selected[: self.max_support_points]:
            lines.append(f"- {event.event_date_text}: {event.event_text}")
            citations.append(
                AnswerCitation(
                    parent_chunk_id=event.parent_chunk_id,
                    document_id=event.document_id,
                    source_name=event.source_name,
                    heading=event.heading,
                    supporting_excerpt=event.supporting_excerpt,
                )
            )

        return GenerateAnswerResult(
            answer_text=(
                f"Direct answer: {direct}\n\n"
                "Supporting points:\n"
                + "\n".join(lines)
                + "\n\nCaveats / limitations:\n"
                "- Response is limited to chronology-responsive dated events in the retrieved context."
            ),
            grounded=True,
            sufficient_context=True,
            citations=citations,
            warnings=[],
        )

    def _generate_employment_lifecycle_answer(
        self,
        context: Sequence[dict[str, Any]],
        query: str,
    ) -> GenerateAnswerResult | None:
        lowered_query = query.lower()
        target = self._employment_lifecycle_target(lowered_query)
        if target is None:
            return None

        match = self._resolve_employment_lifecycle_value(context, target)
        if match is None:
            return self._insufficient_response("insufficient_context: employment lifecycle-responsive evidence not found")
        if self._is_lifecycle_when_date_required_query(lowered_query, target) and not self._contains_concrete_date(str(match["value"])):
            return self._insufficient_response("insufficient_context: lifecycle date value not found")

        citation = AnswerCitation(
            parent_chunk_id=match["parent_chunk_id"],
            document_id=match.get("document_id"),
            source_name=match.get("source_name"),
            heading=match.get("heading"),
            supporting_excerpt=match["excerpt"],
        )
        source_line = f"- {match.get('heading') or '(no heading)'}: {match['excerpt']}"
        direct = self._employment_lifecycle_direct_answer(target, match["value"])

        return GenerateAnswerResult(
            answer_text=(
                f"Direct answer: {direct}\n\n"
                "Supporting points:\n"
                f"{source_line}\n\n"
                "Caveats / limitations:\n"
                "- Response is limited to employment lifecycle-responsive agreement language in the retrieved context."
            ),
            grounded=True,
            sufficient_context=True,
            citations=[citation],
            warnings=[],
        )

    def _employment_lifecycle_target(self, lowered_query: str) -> str | None:
        if "offer" in lowered_query and "accept" in lowered_query:
            return "offer_acceptance"
        if any(token in lowered_query for token in ("start date", "commencement", "employment begin", "employment start", "employment relationship begin")):
            return "employment_start"
        if "probation" in lowered_query:
            return "probation"
        if any(token in lowered_query for token in ("compensation", "salary", "wage", "remuneration")):
            return "compensation"
        if "benefit" in lowered_query:
            return "benefits"
        if "termination" in lowered_query and any(token in lowered_query for token in ("effective", "take effect", "date", "terminated")):
            return "termination_effective"
        if "severance" in lowered_query:
            return "severance"
        if "roe" in lowered_query or "record of employment" in lowered_query:
            return "roe"
        return None

    def _resolve_employment_lifecycle_value(
        self,
        context: Sequence[dict[str, Any]],
        target: str,
    ) -> dict[str, Any] | None:
        for item in context:
            text = str(item.get("text") or "")
            heading = str(item.get("heading") or "")
            haystack = f"{heading}\n{text}"
            value = self._extract_employment_lifecycle_value(text, target)
            if not value:
                value = self._extract_employment_lifecycle_value(haystack, target)
            if not value:
                continue
            excerpt = self._best_excerpt(text or haystack, value)
            return {
                "value": value,
                "excerpt": excerpt or value,
                "parent_chunk_id": item.get("parent_chunk_id", ""),
                "document_id": item.get("document_id"),
                "source_name": item.get("source_name"),
                "heading": item.get("heading"),
            }
        return None

    def _extract_employment_lifecycle_value(self, haystack: str, target: str) -> str | None:
        lowered = haystack.lower()
        date_pattern = r"(?:\d{4}-\d{2}-\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},\s*\d{4})"
        if target == "offer_acceptance":
            if "offer" in lowered and any(token in lowered for token in ("accept", "accepted", "acceptance")):
                dated_match = re.search(
                    rf"\boffer[^.\n;]{{0,80}}(?:accepted|acceptance)[^.\n;]{{0,80}}(?:on\s+)?({date_pattern})",
                    haystack,
                    flags=re.IGNORECASE,
                )
                if dated_match:
                    return dated_match.group(0).strip()
                match = re.search(r"\boffer[^.\n;]{0,120}(?:accepted|acceptance)[^.\n;]{0,120}", haystack, flags=re.IGNORECASE)
                return match.group(0).strip() if match else "Offer acceptance terms are present."
            return None
        if target == "employment_start":
            match = re.search(rf"\b(?:effective date|commence(?:ment|d)?|start date|employment (?:began|begins|commenced|commences|start(?:ed|s)?)).{{0,40}}({date_pattern})", haystack, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
            if "effective date" in lowered or "commence" in lowered or "start date" in lowered:
                return "Employment start/commencement language is present."
            return None
        if target == "probation":
            match = re.search(r"\bprobation(?:ary)?[^.\n;]*", haystack, flags=re.IGNORECASE)
            return match.group(0).strip() if match else None
        if target == "compensation":
            match = re.search(r"\b(?:compensation|salary|base salary|wage)[^.\n;]*", haystack, flags=re.IGNORECASE)
            return match.group(0).strip() if match else None
        if target == "benefits":
            match = re.search(r"\bbenefits?[^.\n;]*", haystack, flags=re.IGNORECASE)
            return match.group(0).strip() if match else None
        if target == "termination_effective":
            match = re.search(rf"\b(?:termination|terminated)[^.\n;]{{0,60}}(?:effective|on)\s+({date_pattern})", haystack, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
            if "termination" in lowered and "effective" in lowered:
                return "Termination effective language is present."
            return None
        if target == "severance":
            match = re.search(r"\bseverance[^.\n;]*", haystack, flags=re.IGNORECASE)
            return match.group(0).strip() if match else None
        if target == "roe":
            match = re.search(r"\b(?:record of employment|roe)[^.\n;]*", haystack, flags=re.IGNORECASE)
            return match.group(0).strip() if match else None
        return None

    def _employment_lifecycle_direct_answer(self, target: str, value: str) -> str:
        if target == "offer_acceptance":
            return f"The offer/acceptance evidence states: {value}"
        if target == "employment_start":
            return f"The employment start/commencement evidence is: {value}."
        if target == "probation":
            return f"The probation evidence states: {value}."
        if target == "compensation":
            return f"The compensation evidence states: {value}."
        if target == "benefits":
            return f"The benefits evidence states: {value}."
        if target == "termination_effective":
            return f"The termination effective-date evidence is: {value}."
        if target == "severance":
            return f"The severance evidence states: {value}."
        return f"The ROE evidence states: {value}."

    def _is_lifecycle_when_date_required_query(self, lowered_query: str, target: str) -> bool:
        if "when" not in lowered_query:
            return False
        return target in {"offer_acceptance", "employment_start", "probation", "termination_effective"}

    def _contains_concrete_date(self, value: str) -> bool:
        return bool(
            re.search(
                r"\b(?:\d{4}-\d{2}-\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},\s*\d{4})\b",
                value or "",
                flags=re.IGNORECASE,
            )
        )

    def _is_chronology_question(self, lowered_query: str) -> bool:
        patterns = (
            r"\bwhen did\b",
            r"\bwhen was\b",
            r"\btimeline\b",
            r"\bchronology\b",
            r"\bwhat happened first\b",
            r"\bwhat happened last\b",
            r"\bwhat happened after\b",
            r"\bwhat happened before\b",
            r"\bwhat happened between\b",
            r"\ball dated events\b",
        )
        return any(re.search(pattern, lowered_query) for pattern in patterns)

    def _extract_chronology_events(self, context: Sequence[dict[str, Any]]) -> list[_ChronologyEvent]:
        date_pattern = re.compile(
            r"\b(?:\d{4}-\d{2}-\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},\s*\d{4})\b",
            flags=re.IGNORECASE,
        )
        events: list[_ChronologyEvent] = []
        for item in context:
            text = str(item.get("text") or "")
            if not text:
                continue
            for match in date_pattern.finditer(text):
                parsed = self._parse_date(match.group(0))
                if parsed is None:
                    continue
                start = max(0, match.start() - 80)
                end = min(len(text), match.end() + 120)
                excerpt = " ".join(text[start:end].split())
                event_text = excerpt
                events.append(
                    _ChronologyEvent(
                        parent_chunk_id=str(item.get("parent_chunk_id") or ""),
                        document_id=item.get("document_id"),
                        source_name=item.get("source_name"),
                        heading=item.get("heading"),
                        event_date=parsed,
                        event_date_text=match.group(0),
                        event_text=event_text,
                        supporting_excerpt=excerpt,
                    )
                )
        events.sort(key=lambda event: event.event_date)
        return events

    def _parse_date(self, value: str) -> datetime | None:
        raw = (value or "").strip()
        for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
        return None

    def _extract_anchor_phrase(self, query: str, relation: str) -> str:
        match = re.search(rf"\b{relation}\s+(.+?)(?:\?|$)", query)
        if not match:
            return ""
        anchor = re.sub(r"[^a-z0-9\s]+", " ", match.group(1).lower())
        anchor = re.sub(r"\b(the|a|an)\b", " ", anchor)
        return " ".join(anchor.split())

    def _find_anchor_event(self, events: Sequence[_ChronologyEvent], anchor_phrase: str) -> _ChronologyEvent | None:
        if not anchor_phrase:
            return None
        for event in events:
            normalized = re.sub(r"[^a-z0-9\s]+", " ", event.event_text.lower())
            if anchor_phrase in " ".join(normalized.split()):
                return event
        return None

    def _extract_query_dates(self, query: str) -> list[datetime]:
        date_pattern = re.compile(
            r"\b(?:\d{4}-\d{2}-\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},\s*\d{4})\b",
            flags=re.IGNORECASE,
        )
        values: list[datetime] = []
        for match in date_pattern.finditer(query):
            parsed = self._parse_date(match.group(0))
            if parsed is not None:
                values.append(parsed)
        return values

    def _resolve_party_roles_from_intro(self, context: Sequence[dict[str, Any]]) -> "_PartyRoleAssignment | None":
        for item in context:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            assignment = self._extract_intro_assignment(text)
            if assignment is None:
                continue
            assignment.source_parent_chunk_id = str(item.get("parent_chunk_id") or "")
            assignment.document_id = item.get("document_id")
            assignment.source_name = item.get("source_name")
            assignment.heading = item.get("heading")
            assignment.supporting_excerpt = self._best_excerpt(text, "parties employer employee agreement")
            if not assignment.supporting_excerpt:
                assignment.supporting_excerpt = text
            return assignment
        return None

    def _extract_intro_assignment(self, text: str) -> "_PartyRoleAssignment | None":
        lowered = text.lower()
        has_intro_anchor = bool(
            re.search(r"\b(this\s+.+?\s+agreement\s+is\s+made(?:\s+effective)?|by\s+and\s+between|between)\b", lowered)
        )
        if not has_intro_anchor:
            return None

        employer_label = re.search(r"\bemployer\s*[:\-]\s*([^;\n.]+)", text, flags=re.IGNORECASE)
        employee_label = re.search(r"\bemployee\s*[:\-]\s*([^;\n.]+)", text, flags=re.IGNORECASE)
        employer = self._clean_party_name(employer_label.group(1)) if employer_label else None
        employee = self._clean_party_name(employee_label.group(1)) if employee_label else None

        between_match = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[.;\n]|$)", text, flags=re.IGNORECASE)
        parties: list[str] = []
        if between_match:
            first = self._clean_party_name(between_match.group(1))
            second = self._clean_party_name(between_match.group(2))
            if first:
                parties.append(first)
            if second:
                parties.append(second)

            first_role = self._detect_inline_role(between_match.group(1))
            second_role = self._detect_inline_role(between_match.group(2))
            if first_role == "employer":
                employer = employer or first
            elif first_role == "employee":
                employee = employee or first
            if second_role == "employer":
                employer = employer or second
            elif second_role == "employee":
                employee = employee or second

        if len(parties) >= 2 and (employer is None or employee is None):
            inferred_employer, inferred_employee = self._infer_employer_employee(parties[0], parties[1], lowered)
            employer = employer or inferred_employer
            employee = employee or inferred_employee

        if employer and self._is_placeholder_party(employer):
            employer = None
        if employee and self._is_placeholder_party(employee):
            employee = None
        parties = [party for party in parties if not self._is_placeholder_party(party)]

        if not parties and employer and employee:
            parties = [employer, employee]

        if len(parties) < 2:
            return None

        return _PartyRoleAssignment(
            parties=tuple(parties[:2]),
            employer=employer,
            employee=employee,
            source_parent_chunk_id="",
            document_id=None,
            source_name=None,
            heading=None,
            supporting_excerpt="",
        )

    def _detect_inline_role(self, value: str) -> str | None:
        lowered = value.lower()
        if "employer" in lowered or "company" in lowered:
            return "employer"
        if "employee" in lowered:
            return "employee"
        return None

    def _infer_employer_employee(self, first: str, second: str, lowered_text: str) -> tuple[str | None, str | None]:
        if "employment agreement" not in lowered_text:
            return None, None
        first_is_org = self._looks_like_organization(first)
        second_is_org = self._looks_like_organization(second)
        if first_is_org == second_is_org:
            return None, None
        if first_is_org:
            return first, second
        return second, first

    def _pick_company_party(self, parties: Sequence[str]) -> str | None:
        for party in parties:
            if self._looks_like_organization(party):
                return party
        return None

    def _looks_like_organization(self, value: str) -> bool:
        lowered = value.lower()
        org_markers = ("inc", "llc", "ltd", "limited", "corp", "corporation", "company", "co.", "plc")
        return any(marker in lowered for marker in org_markers)

    def _clean_party_name(self, value: str) -> str:
        cleaned = re.sub(r"\((?:the\s+)?[\"“']?(?:employer|employee|company)[\"”']?\)", "", value, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;.")
        return cleaned

    def _is_placeholder_party(self, value: str) -> bool:
        normalized = self._normalize_party_text(value)
        placeholders = {
            "company",
            "employee",
            "employer",
            "party a",
            "party b",
            "first party",
            "second party",
        }
        return normalized in placeholders

    def _normalize_party_text(self, value: str) -> str:
        normalized = re.sub(r"[^a-z0-9\s]", " ", (value or "").lower())
        normalized = re.sub(r"\b(the|this|that)\b", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def _rank_relevant_chunks(self, context: Sequence[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        query_terms = _query_terms(query)
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for idx, item in enumerate(context):
            text = item["text"].lower()
            overlap = sum(1 for term in query_terms if term in text)
            if overlap >= self.min_query_overlap_for_relevance:
                scored.append((overlap, idx, item))
        scored.sort(key=lambda row: (-row[0], row[1], row[2]["parent_chunk_id"]))
        return [row[2] for row in scored]

    def _best_excerpt(self, text: str, query: str) -> str | None:
        sentences = _split_sentences(text)
        if not sentences:
            return None

        query_terms = _query_terms(query)
        ranked = sorted(
            sentences,
            key=lambda s: (
                -sum(1 for term in query_terms if term in s.lower()),
                -int(any(token in s.lower() for token in self.qualifier_terms)),
                len(s),
            ),
        )
        best = ranked[0].strip()
        if not best:
            return None

        qualifier_sentence = next(
            (s.strip() for s in sentences if any(token in s.lower() for token in self.qualifier_terms)),
            None,
        )
        if qualifier_sentence and qualifier_sentence not in best:
            return f"{best} {qualifier_sentence}".strip()
        return best

    def _direct_answer(self, citations: Sequence[AnswerCitation]) -> str:
        first = citations[0].supporting_excerpt or "The retrieved context contains relevant provisions."
        return first

    def _is_fully_answerable(self, ranked: Sequence[dict[str, Any]], query: str) -> bool:
        query_terms = _query_terms(query)
        if not query_terms:
            return bool(ranked)
        normalized_terms = {_normalize_term(term) for term in query_terms}
        covered = {
            term
            for term in normalized_terms
            if any(term and term in _normalize_term(item["text"].lower()) for item in ranked)
        }
        if normalized_terms and covered == normalized_terms:
            return True

        top_text = _normalize_term(ranked[0]["text"].lower()) if ranked else ""
        top_overlap = sum(1 for term in normalized_terms if term and term in top_text)
        return top_overlap >= 2

    def _check_qualifier_coverage(
        self,
        selected: Sequence[dict[str, Any]],
        citations: Sequence[AnswerCitation],
    ) -> str | None:
        selected_text = " ".join(item["text"].lower() for item in selected)
        cited_text = " ".join((citation.supporting_excerpt or "").lower() for citation in citations)
        for qualifier in self.qualifier_terms:
            if qualifier in selected_text and qualifier not in cited_text:
                return "grounding_risk: qualifier_or_limitation_may_be_omitted"
        return None

    def _insufficient_response(self, reason: str) -> GenerateAnswerResult:
        return GenerateAnswerResult(
            answer_text=(
                "Direct answer: The retrieved context does not provide enough information to answer this question fully.\n\n"
                "Supporting points:\n"
                "- Available chunks do not contain sufficient directly responsive content.\n\n"
                "Caveats / limitations:\n"
                "- A reliable answer cannot be produced without additional relevant context."
            ),
            grounded=False,
            sufficient_context=False,
            citations=[],
            warnings=[reason],
        )

    def _failure_response(self, reason: str) -> GenerateAnswerResult:
        return GenerateAnswerResult(
            answer_text=(
                "Direct answer: The answer generation step failed safely.\n\n"
                "Supporting points:\n"
                "- No grounded claims were produced.\n\n"
                "Caveats / limitations:\n"
                "- Please retry with the same context after resolving the failure."
            ),
            grounded=False,
            sufficient_context=False,
            citations=[],
            warnings=[f"failure: {reason}"],
        )


_DEFAULT_ANSWER_SYNTHESIZER = LegalAnswerSynthesizer()


@dataclass(slots=True)
class _PartyRoleAssignment:
    parties: tuple[str, str]
    employer: str | None
    employee: str | None
    source_parent_chunk_id: str
    document_id: str | None
    source_name: str | None
    heading: str | None
    supporting_excerpt: str


def generate_answer(context: Sequence[object], query: str) -> GenerateAnswerResult:
    """Generate a grounded legal answer from parent-chunk context.

    This tool is intentionally narrow: it does not retrieve data, use external
    knowledge, or access conversation memory. It reports insufficiency instead
    of guessing and returns structured citations for every supported claim.
    """

    return _DEFAULT_ANSWER_SYNTHESIZER.generate(context=context, query=query)


def _unit_to_context_row(unit: EvidenceUnit) -> dict[str, Any]:
    return {
        "parent_chunk_id": unit.parent_chunk_id,
        "document_id": _as_optional_str(unit.document_id),
        "source_name": _as_optional_str(unit.source_name),
        "heading": _as_optional_str(unit.heading),
        "text": unit.evidence_text,
        "metadata": dict(unit.metadata),
    }


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _query_terms(query: str) -> list[str]:
    stop = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "what",
        "how",
        "when",
        "where",
        "why",
        "under",
        "for",
        "to",
        "of",
        "in",
        "on",
        "and",
        "or",
    }
    terms = [term for term in re.findall(r"[a-zA-Z0-9]+", (query or "").lower()) if term not in stop]
    return list(dict.fromkeys(terms))


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _normalize_term(value: str) -> str:
    token = re.sub(r"[^a-z0-9\s]", " ", value.lower()).strip()
    token = re.sub(r"\b([a-z]{3,})s\b", r"\1", token)
    return re.sub(r"\s+", " ", token)
