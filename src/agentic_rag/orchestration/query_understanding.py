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


def _is_generic_document_label_phrase(value: str) -> bool:
    canonical = _canonicalize_phrase(value)
    if not canonical:
        return False
    tokens = canonical.split()
    if not tokens:
        return False
    generic_heads = {"agreement", "contract", "document", "nda", "lease", "msa"}
    if tokens[-1] in generic_heads:
        return True
    return canonical in generic_heads


def _is_document_label_like_hint(hint: str, known_documents: Sequence[str]) -> bool:
    canonical_hint = _canonicalize_phrase(hint)
    if not canonical_hint:
        return False
    for document_label in known_documents:
        canonical_label = _canonicalize_phrase(document_label)
        if not canonical_label:
            continue
        if canonical_hint == canonical_label:
            return True
        if canonical_hint in canonical_label or canonical_label in canonical_hint:
            return True
    return False


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




def _is_party_role_entity_query(normalized_query: str) -> bool:
    lowered = _canonicalize_phrase(normalized_query)
    if not lowered:
        return False

    role_markers = ("employer", "employee", "parties", "party", "company", "entity")
    if any(marker in lowered for marker in role_markers):
        role_intents = ("who is", "who are", "which", "what", "is this agreement", "agreement for")
        if any(intent in lowered for intent in role_intents):
            return True

    if lowered.startswith("is this agreement between") or lowered.startswith("is the agreement between"):
        return True
    if "which company is this agreement for" in lowered:
        return True
    return False


def _is_chronology_date_event_query(normalized_query: str) -> bool:
    lowered = _canonicalize_phrase(normalized_query)
    if not lowered:
        return False

    direct_patterns = (
        r"\bwhen did\b",
        r"\bwhen was\b",
        r"\btimeline\b",
        r"\bchronology\b",
        r"\bwhat happened first\b",
        r"\bwhat happened last\b",
        r"\bfirst event\b",
        r"\blast event\b",
        r"\bwhat happened after\b",
        r"\bwhat happened before\b",
        r"\bbetween\b.+\band\b",
        r"\ball dated events\b",
    )
    if any(re.search(pattern, lowered) for pattern in direct_patterns):
        return True

    temporal_words = (
        "effective date",
        "commencement",
        "start date",
        "termination date",
        "notice date",
        "letter date",
        "email date",
        "filing date",
        "service date",
    )
    has_temporal_word = any(word in lowered for word in temporal_words)
    has_question_frame = any(
        phrase in lowered
        for phrase in (
            "what date",
            "on what date",
            "what happened",
            "when",
            "after",
            "before",
            "between",
        )
    )
    return has_temporal_word and has_question_frame


def _is_matter_metadata_query(normalized_query: str) -> bool:
    lowered = _canonicalize_phrase(normalized_query)
    if not lowered:
        return False

    metadata_patterns = (
        r"\bwhat\s+is\s+the\s+file\s+number\b",
        r"\bwhat\s+is\s+the\s+matter\s+name\b",
        r"\bwhat\s+is\s+the\s+case\s+name\b",
        r"\bwho\s+is\s+the\s+client\b",
        r"\bwhat\s+jurisdiction\s+applies\b",
        r"\bwhich\s+jurisdiction\s+applies\b",
        r"\bwhat\s+court\s+is\s+involved\b",
        r"\bwhich\s+court\s+is\s+involved\b",
        r"\bwhat\s+is\s+this\s+matter\s+about\b",
        r"\bwhat\s+is\s+this\s+document\s+about\b",
    )
    if any(re.search(pattern, lowered) for pattern in metadata_patterns):
        return True

    metadata_markers = (
        "matter name",
        "case name",
        "client",
        "file number",
        "court file number",
        "docket number",
        "jurisdiction",
        "governing forum",
        "court",
        "caption",
        "matter information",
    )
    has_marker = any(marker in lowered for marker in metadata_markers)
    has_question_frame = any(
        marker in lowered
        for marker in (
            "what is",
            "which is",
            "which",
            "who is",
            "who's",
            "what court",
            "what jurisdiction",
            "which court",
            "which jurisdiction",
            "about this matter",
            "about this document",
        )
    )
    return has_marker and has_question_frame


def _is_employment_contract_lifecycle_query(normalized_query: str) -> bool:
    lowered = _canonicalize_phrase(normalized_query)
    if not lowered:
        return False

    patterns = (
        r"\bwhen\s+did\s+(?:employment|the employment relationship)\s+(?:begin|start|commence)\b",
        r"\b(?:employment\s+)?start\s+date\b",
        r"\bcommencement\s+date\b",
        r"\boffer\s+and\s+acceptance\b",
        r"\bwhen\s+was\s+the\s+offer\s+accepted\b",
        r"\bprobation(?:ary)?\b",
        r"\bwhen\s+did\s+probation\s+end\b",
        r"\bcompensation\s+terms\b",
        r"\bsalary\b",
        r"\bbenefits\b",
        r"\btermination\s+effective\s+date\b",
        r"\bwhen\s+did\s+termination\s+take\s+effect\b",
        r"\bseverance\b",
        r"\brecord\s+of\s+employment\b",
        r"\broe\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _is_employment_mitigation_query(normalized_query: str) -> bool:
    lowered = _canonicalize_phrase(normalized_query)
    if not lowered:
        return False

    patterns = (
        r"\bmitigation\b",
        r"\bmitigate\b",
        r"\bmitigation efforts?\b",
        r"\bjob applications?\b",
        r"\bhow many job applications?\b",
        r"\binterviews?\b",
        r"\boffers?\s+(?:received|rejected)\b",
        r"\balternative employment\b",
        r"\bnew employment\b",
        r"\bjob search\b",
        r"\bmitigation evidence\b",
    )
    if any(re.search(pattern, lowered) for pattern in patterns):
        return True

    evidence_markers = (
        "application record",
        "application log",
        "job search log",
        "mitigation journal",
        "offer letter",
        "interview invitation",
        "employment update",
    )
    has_employment_frame = any(token in lowered for token in ("employment", "job", "offer", "interview", "application"))
    return has_employment_frame and any(marker in lowered for marker in evidence_markers)




def _is_correspondence_litigation_milestone_query(normalized_query: str) -> bool:
    lowered = _canonicalize_phrase(normalized_query)
    if not lowered:
        return False

    patterns = (
        r"\bwhat\s+letters?\s+(?:were|was)\s+sent\b",
        r"\bwhat\s+emails?\s+(?:were|was)\s+sent\b",
        r"\bwhat\s+communications?\s+(?:were|was)\s+sent\b",
        r"\bwhen\s+(?:was|were|did)\s+.+\b(?:letter|email|communication|correspondence)\b",
        r"\bwhat\s+deadlines?\s+(?:were|was)\s+demanded\b",
        r"\bwhen\s+was\s+the\s+claim\s+filed\b",
        r"\bwhen\s+was\s+the\s+statement\s+of\s+claim\s+filed\b",
        r"\bwhen\s+was\s+the\s+defen(?:c|s)e\s+(?:due|filed)\b",
        r"\bwhat\s+pleadings?\s+.*\b(?:filed|served)\b",
        r"\bwhat\s+happened\s+procedurally\b",
        r"\bprocedural\s+(?:history|status)\b",
        r"\bcourt\s+filings?\b",
        r"\bdefault\s+notice\b",
        r"\bsettlement\s+discussion",
    )
    if any(re.search(pattern, lowered) for pattern in patterns):
        return True

    marker_groups = (
        ("letter", "email", "communication", "correspondence", "demand"),
        ("claim", "statement of claim", "pleading", "defence", "defense", "reply", "default notice"),
        ("filed", "served", "service", "issued", "delivered"),
    )
    return (
        any(marker in lowered for marker in marker_groups[0])
        and any(marker in lowered for marker in ("when", "what", "deadline", "deadlines", "demanded", "sent"))
    ) or (
        any(marker in lowered for marker in marker_groups[1])
        and any(marker in lowered for marker in marker_groups[2] + ("due", "deadline", "procedurally", "status"))
    )
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
    is_party_role_entity_query = _is_party_role_entity_query(normalized)
    is_chronology_date_event_query = _is_chronology_date_event_query(normalized)
    is_matter_metadata_query = _is_matter_metadata_query(normalized)
    is_employment_contract_lifecycle_query = _is_employment_contract_lifecycle_query(normalized)
    is_employment_mitigation_query = _is_employment_mitigation_query(normalized)
    is_correspondence_litigation_milestone_query = _is_correspondence_litigation_milestone_query(normalized)

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
        "probation",
        "compensation",
        "benefits",
        "severance",
        "roe",
        "record of employment",
        "commencement",
        "effective date",
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
            topic_tokens = set(topic.split())
            overlap = subject_tokens & topic_tokens
            if (
                (len(topic_tokens) == 1 and overlap)
                or (len(topic_tokens) > 1 and len(overlap) >= 2)
                or (len(subject_tokens) == 1 and len(overlap) == 1)
            ):
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
    elif (
        is_party_role_entity_query
        or is_chronology_date_event_query
        or is_matter_metadata_query
        or is_employment_contract_lifecycle_query
        or is_employment_mitigation_query
        or is_correspondence_litigation_milestone_query
        or any(marker in lowered for marker in extractive_markers)
    ):
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
    if is_canonical_what_is and not is_matter_metadata_query:
        available_documents = bool(active_documents or selected_documents)
        combined_hints = [*resolved_topic_hints, *resolved_clause_hints]
        canonical_subject = _canonicalize_phrase(what_is_subject or "")
        independent_clause_hints: list[str] = []
        for hint in resolved_topic_hints:
            canonical_hint = _canonicalize_phrase(hint)
            if not canonical_hint:
                continue
            if _is_document_label_like_hint(canonical_hint, known_documents):
                continue
            if _is_generic_document_label_phrase(canonical_hint):
                continue
            if canonical_hint not in independent_clause_hints:
                independent_clause_hints.append(canonical_hint)
        for hint in resolved_clause_hints:
            canonical_hint = _canonicalize_phrase(hint)
            if not canonical_hint:
                continue
            if canonical_hint == canonical_subject:
                # Canonical "what is X" subject hints are synthetic for override
                # gating and must not self-qualify clause lookup.
                continue
            if _is_document_label_like_hint(canonical_hint, known_documents):
                continue
            if _is_generic_document_label_phrase(canonical_hint):
                continue
            if canonical_hint not in independent_clause_hints:
                independent_clause_hints.append(canonical_hint)

        best_hint_confidence = _hint_match_confidence(what_is_subject or "", independent_clause_hints)
        threshold = 0.8
        subject_token_count = len((what_is_subject or "").split())
        has_topic_marker_support = bool(
            set(independent_clause_hints) & {_canonicalize_phrase(topic) for topic in resolved_topic_hints}
        )
        topic_support_threshold = 0.3
        has_high_confidence_match = best_hint_confidence >= threshold
        has_topic_supported_match = (
            subject_token_count >= 2 and has_topic_marker_support and best_hint_confidence >= topic_support_threshold
        )
        is_document_grounded = bool(available_documents and independent_clause_hints and (has_high_confidence_match or has_topic_supported_match))
        has_hint_signal = bool(available_documents and independent_clause_hints)
        clause_hint_match = bool((has_high_confidence_match or has_topic_supported_match) and independent_clause_hints)
        routing_notes.append("debug:canonical_what_is=true")
        routing_notes.append(f"debug:clause_hint_match={'true' if clause_hint_match else 'false'}")
        if combined_hints and not independent_clause_hints:
            routing_notes.append("what_is_override_blocked_by_non_independent_hint_evidence")

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

    should_rewrite = (
        question_type in {"ambiguous_query"}
        or (is_context_dependent and not explicit_document_scope)
        or is_party_role_entity_query
        or is_matter_metadata_query
        or is_employment_contract_lifecycle_query
        or is_employment_mitigation_query
        or is_correspondence_litigation_milestone_query
    )
    should_extract_entities = is_party_role_entity_query or is_matter_metadata_query or any(
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
    if is_party_role_entity_query:
        routing_notes.append("legal_question_family:party_role_entity")
    if is_chronology_date_event_query:
        routing_notes.append("legal_question_family:chronology_date_event")
    if is_matter_metadata_query:
        routing_notes.append("legal_question_family:matter_document_metadata")
    if is_employment_contract_lifecycle_query:
        routing_notes.append("legal_question_family:employment_contract_lifecycle")
    if is_employment_mitigation_query:
        routing_notes.append("legal_question_family:employment_mitigation")
    if is_correspondence_litigation_milestone_query:
        routing_notes.append("legal_question_family:correspondence_litigation_milestone")

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
