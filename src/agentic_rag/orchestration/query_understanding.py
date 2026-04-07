"""Deterministic query-understanding for legal RAG orchestration.

This module classifies a user query *before retrieval* so the orchestrator can
route conservatively, resolve conversational references safely, and enforce
answerability expectations (for example, strict definition requirements).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

try:  # pragma: no cover - optional runtime dependency
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover - fallback for constrained envs
    from agentic_rag._compat_pydantic import BaseModel, ConfigDict, Field


QuestionType = Literal[
    "meta_query",
    "definition_query",
    "document_content_query",
    "document_summary_query",
    "extractive_fact_query",
    "comparison_query",
    "followup_query",
    "ambiguous_query",
    "other_query",
]
AnswerabilityExpectation = Literal[
    "definition_required",
    "clause_lookup",
    "summary",
    "fact_extraction",
    "comparison",
    "meta_response",
    "general_grounded_response",
    "clarification_needed",
]


class QueryUnderstandingResult(BaseModel):
    """Strict, inspectable query-understanding output consumed by orchestrators."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_query: str
    normalized_query: str
    question_type: QuestionType

    is_followup: bool
    is_context_dependent: bool
    use_conversation_context: bool

    is_document_scoped: bool
    refers_to_prior_document_scope: bool
    refers_to_prior_clause_or_topic: bool

    should_rewrite: bool
    should_extract_entities: bool
    should_retrieve: bool
    may_need_decomposition: bool
    answerability_expectation: AnswerabilityExpectation

    resolved_document_hints: list[str] = Field(default_factory=list)
    resolved_topic_hints: list[str] = Field(default_factory=list)
    resolved_clause_hints: list[str] = Field(default_factory=list)

    ambiguity_notes: list[str] = Field(default_factory=list)
    routing_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _extract_doc_label(document: Any) -> str | None:
    if isinstance(document, str):
        return document.strip() or None
    if isinstance(document, Mapping):
        for key in ("name", "title", "source_name", "filename", "file_name", "id", "document_id", "path"):
            value = document.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
    for key in ("name", "title", "source_name", "filename", "file_name", "id", "document_id", "path"):
        value = getattr(document, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _known_documents(selected_documents: Sequence[Any] | None, active_documents: Sequence[Any] | None) -> list[str]:
    labels: list[str] = []
    for item in (selected_documents or []):
        label = _extract_doc_label(item)
        if label and label not in labels:
            labels.append(label)
    for item in (active_documents or []):
        label = _extract_doc_label(item)
        if label and label not in labels:
            labels.append(label)
    return labels


def _canonicalize_phrase(value: str) -> str:
    lowered = value.lower().strip()
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    lowered = " ".join(lowered.split())
    return lowered


def _extract_what_is_subject(normalized_query: str) -> str | None:
    match = re.match(r"^what\s+is(?:\s+the)?\s+(.+?)(?:\?)?$", normalized_query.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    subject = _canonicalize_phrase(match.group(1))
    return subject or None


def _hint_match_confidence(subject: str, hints: Sequence[str]) -> float:
    if not subject:
        return 0.0
    subject_tokens = subject.split()
    best = 0.0
    for hint in hints:
        canonical_hint = _canonicalize_phrase(hint)
        if not canonical_hint:
            continue
        if canonical_hint == subject:
            return 1.0
        if canonical_hint in subject or subject in canonical_hint:
            hint_tokens = canonical_hint.split()
            containment_ratio = min(len(subject_tokens), len(hint_tokens)) / max(len(subject_tokens), len(hint_tokens))
            best = max(best, containment_ratio)
            continue
        hint_tokens = canonical_hint.split()
        if not hint_tokens:
            continue
        overlap = len(set(subject_tokens) & set(hint_tokens))
        if overlap:
            ratio = overlap / max(len(subject_tokens), len(hint_tokens))
            best = max(best, ratio)
    return best


def understand_query(
    query: str,
    conversation_summary: str | None = None,
    recent_messages: Sequence[Mapping[str, Any]] | None = None,
    active_documents: Sequence[Any] | None = None,
    selected_documents: Sequence[Any] | None = None,
) -> QueryUnderstandingResult:
    """Classify query intent and context dependence for safe retrieval routing.

    Conservative context dependence detection prevents unsafe scope guessing:
    explicit self-contained queries stay self-contained, while referential turns
    (e.g., ``what about governing law``) are flagged for controlled context use.
    """

    original = _text(query)
    normalized = " ".join(original.split())
    lowered = normalized.lower()
    context_available = bool(_text(conversation_summary) or list(recent_messages or []))

    meta_markers = ("how many documents", "what files are loaded", "what documents are uploaded", "what docs are loaded")
    comparison_markers = ("compare", "differ", "difference", "vs ", "versus")
    summary_markers = ("summarize", "summary", "tl;dr")
    doc_content_markers = ("what does", "say about", "mention", "in the ", "in this ", "in that ")
    extractive_markers = ("who are", "what is the notice period", "which law governs", "what law governs", "when is", "how long")
    definition_starters = ("what is ", "define ", "meaning of ")
    followup_starters = ("what about", "how about", "does it", "does that", "what does it", "what does that", "does this")
    pronouns = (" it ", " this ", " that ", " those ", " these ")
    topic_markers = (
        "confidentiality",
        "governing law",
        "termination",
        "notice",
        "indemnity",
        "arbitration",
        "liability",
        "assignment",
    )

    padded = f" {lowered} "
    known_documents = _known_documents(selected_documents, active_documents)
    explicit_doc_mentions = [doc for doc in known_documents if doc.lower() in lowered]
    has_generic_doc_reference = any(token in lowered for token in ("document", "agreement", "contract", "nda", "lease", "msa"))
    explicit_document_scope = bool(explicit_doc_mentions) or ("what does" in lowered and has_generic_doc_reference)

    is_followup = lowered.startswith(followup_starters) or any(marker in padded for marker in pronouns)
    refers_to_prior_document_scope = any(marker in padded for marker in (" it ", " this document ", " that document ", " that agreement ", " this agreement "))
    refers_to_prior_clause_or_topic = any(
        marker in lowered for marker in ("that clause", "this clause", "still apply", "what about", "does it mention")
    )

    resolved_topic_hints = [topic for topic in topic_markers if topic in lowered]
    what_is_subject = _extract_what_is_subject(normalized)
    is_canonical_what_is = what_is_subject is not None
    if is_canonical_what_is and what_is_subject:
        subject_tokens = set(what_is_subject.split())
        for topic in topic_markers:
            if topic in resolved_topic_hints:
                continue
            if subject_tokens & set(topic.split()):
                resolved_topic_hints.append(topic)
    resolved_clause_hints = list(resolved_topic_hints)
    if is_canonical_what_is and what_is_subject:
        # Preserve the full "what is X" subject for clause-label matching when
        # X is a multi-word phrase. This prevents weak single-token fallbacks
        # (for example: "termination") from being the only clause hint signal.
        canonical_subject = _canonicalize_phrase(what_is_subject)
        if canonical_subject and len(canonical_subject.split()) >= 2 and canonical_subject not in resolved_clause_hints:
            resolved_clause_hints.insert(0, canonical_subject)
    resolved_document_hints = list(explicit_doc_mentions)
    clause_override_triggered = False
    clause_hint_match = False

    ambiguity_notes: list[str] = []
    warnings: list[str] = []
    routing_notes: list[str] = []

    # Classification precedence: meta > comparison > summary > document content >
    # extractive fact > definition > ambiguous > other.
    if any(marker in lowered for marker in meta_markers):
        question_type: QuestionType = "meta_query"
    elif any(marker in lowered for marker in comparison_markers):
        question_type = "comparison_query"
    elif any(marker in lowered for marker in summary_markers):
        question_type = "document_summary_query"
    elif explicit_document_scope and any(marker in lowered for marker in doc_content_markers):
        question_type = "document_content_query"
    elif any(marker in lowered for marker in extractive_markers):
        question_type = "extractive_fact_query"
    elif lowered.startswith(definition_starters):
        question_type = "definition_query"
    elif is_followup and not context_available:
        question_type = "ambiguous_query"
        ambiguity_notes.append("followup_reference_without_prior_context")
    elif normalized.endswith("?") and len(normalized.split()) <= 4 and is_followup and not context_available:
        question_type = "ambiguous_query"
        ambiguity_notes.append("underspecified_followup_query")
    else:
        question_type = "other_query"

    # Deterministic "what is X?" rule insertion point:
    # apply only after hint extraction and before final answerability assignment.
    if is_canonical_what_is:
        available_documents = bool(active_documents or selected_documents)
        combined_hints = [*resolved_topic_hints, *resolved_clause_hints]
        best_hint_confidence = _hint_match_confidence(what_is_subject or "", combined_hints)
        threshold = 0.8
        is_document_grounded = bool(available_documents and combined_hints and best_hint_confidence >= threshold)
        has_hint_signal = bool(available_documents and combined_hints)
        clause_hint_match = bool(best_hint_confidence >= threshold and combined_hints)
        routing_notes.append("debug:canonical_what_is=true")
        routing_notes.append(f"debug:clause_hint_match={'true' if clause_hint_match else 'false'}")

        if is_document_grounded:
            question_type = "document_content_query"
            clause_override_triggered = True
            routing_notes.append("what_is_document_grounded_clause_lookup_override")
        else:
            question_type = "definition_query"
            if has_hint_signal and best_hint_confidence > 0.0:
                ambiguity_notes.append("ambiguous_definition_vs_clause")
                routing_notes.append("what_is_hint_match_below_threshold")
        routing_notes.append(f"debug:clause_override_triggered={'true' if clause_override_triggered else 'false'}")

    # Resolve document hints conservatively when pronoun-based and safe.
    if not resolved_document_hints and is_followup and len(known_documents) == 1 and context_available:
        resolved_document_hints = [known_documents[0]]
        refers_to_prior_document_scope = True
        routing_notes.append("resolved_document_scope_from_followup_context")
    elif not resolved_document_hints and refers_to_prior_document_scope:
        if len(known_documents) == 1:
            resolved_document_hints = [known_documents[0]]
            routing_notes.append("resolved_document_scope_from_single_available_document")
        elif len(known_documents) > 1:
            ambiguity_notes.append("multiple_active_documents_scope_unclear")
            warnings.append("ambiguous_document_scope")

    is_document_scoped = (
        explicit_document_scope
        or bool(resolved_document_hints)
        or refers_to_prior_document_scope
        or clause_override_triggered
    )
    is_context_dependent = is_followup or refers_to_prior_document_scope or refers_to_prior_clause_or_topic
    use_conversation_context = bool(
        (context_available or (refers_to_prior_document_scope and bool(resolved_document_hints)))
        and is_context_dependent
        and not explicit_document_scope
    )

    if question_type == "meta_query":
        answerability: AnswerabilityExpectation = "meta_response"
        should_retrieve = False
    elif question_type == "definition_query":
        answerability = "definition_required"
        should_retrieve = True
        routing_notes.append("require_defining_language_not_heading_only")
    elif question_type == "document_summary_query":
        answerability = "summary"
        should_retrieve = True
    elif question_type == "document_content_query":
        answerability = "clause_lookup"
        should_retrieve = True
    elif question_type == "extractive_fact_query":
        answerability = "fact_extraction"
        should_retrieve = True
    elif question_type == "comparison_query":
        answerability = "comparison"
        should_retrieve = True
    elif question_type == "ambiguous_query":
        answerability = "clarification_needed"
        should_retrieve = False
    else:
        answerability = "general_grounded_response"
        should_retrieve = True

    should_rewrite = question_type in {"ambiguous_query"} or (is_context_dependent and not explicit_document_scope)
    should_extract_entities = any(
        marker in lowered
        for marker in ("law", "jurisdiction", "court", "statute", "regulation", "clause", "section", "ontario", "california")
    )
    may_need_decomposition = question_type == "comparison_query" or (" and " in padded and "?" in normalized)

    if is_followup:
        routing_notes.append("followup_like_query")
    if is_context_dependent:
        routing_notes.append("context_dependent")
    if use_conversation_context:
        routing_notes.append("use_conversation_context")
    if should_rewrite:
        routing_notes.append("rewrite_recommended")
    if should_extract_entities:
        routing_notes.append("entity_extraction_useful")

    if question_type == "other_query" and refers_to_prior_document_scope and not context_available:
        ambiguity_notes.append("pronoun_reference_without_resolvable_scope")
        answerability = "clarification_needed"
        should_retrieve = False
        question_type = "ambiguous_query"

    return QueryUnderstandingResult(
        original_query=original,
        normalized_query=normalized,
        question_type=question_type,
        is_followup=is_followup,
        is_context_dependent=is_context_dependent,
        use_conversation_context=use_conversation_context,
        is_document_scoped=is_document_scoped,
        refers_to_prior_document_scope=refers_to_prior_document_scope,
        refers_to_prior_clause_or_topic=refers_to_prior_clause_or_topic,
        should_rewrite=should_rewrite,
        should_extract_entities=should_extract_entities,
        should_retrieve=should_retrieve,
        may_need_decomposition=may_need_decomposition,
        resolved_document_hints=resolved_document_hints,
        resolved_topic_hints=resolved_topic_hints,
        resolved_clause_hints=resolved_clause_hints,
        answerability_expectation=answerability,
        ambiguity_notes=ambiguity_notes,
        routing_notes=routing_notes,
        warnings=warnings,
    )
