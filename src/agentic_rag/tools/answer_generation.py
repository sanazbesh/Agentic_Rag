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

            party_role_response = self._generate_party_role_answer(normalized_context, normalized_query)
            if party_role_response is not None:
                return party_role_response

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
