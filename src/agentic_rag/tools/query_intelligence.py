"""Query-intelligence tools for retrieval-focused legal RAG pipelines.

These utilities transform user queries before retrieval without coupling to any
vector store, retriever, or answer-generation logic.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


logger = logging.getLogger(__name__)


class QueryTransformationLLM(Protocol):
    """Minimal prompt-based LLM abstraction for query transformation."""

    def complete(self, prompt: str) -> str:
        """Return raw model text for a prompt."""


@dataclass(slots=True, frozen=True)
class QueryRewriteResult:
    """Structured output for a retrieval-oriented query rewrite.

    Rewriting improves retrieval by clarifying ambiguous phrasing while
    preserving legal meaning (jurisdiction, parties, time scope, and cited
    sections/cases) from the original query and optional conversation context.
    """

    original_query: str
    rewritten_query: str
    used_conversation_context: bool
    rewrite_notes: str = ""


@dataclass(slots=True, frozen=True)
class QueryDecompositionResult:
    """Structured output for decomposition of legal queries into sub-queries.

    Decomposition improves legal retrieval coverage by splitting multi-issue
    questions into focused retrieval targets (e.g., rule vs. exception) while
    preserving deterministic ordering and avoiding invented legal issues.
    """

    original_query: str
    sub_queries: tuple[str, ...]
    used_conversation_context: bool
    decomposition_notes: str = ""


@dataclass(slots=True, frozen=True)
class LegalEntityFilters:
    """Subset filter payload derived strictly from extracted legal entities."""

    jurisdiction: list[str]
    court: list[str]
    document_type: list[str]
    date_from: str | None
    date_to: str | None
    clause_type: list[str]


@dataclass(slots=True, frozen=True)
class LegalEntityExtractionResult:
    """Structured legal query parsing output for retrieval optimization.

    This model is intentionally conservative for legal RAG: it extracts only
    explicit signals from the *current* query so downstream retrieval/ranking
    can filter and score accurately without hallucinating legal details.
    """

    original_query: str
    normalized_query: str | None
    document_types: list[str]
    legal_topics: list[str]
    jurisdictions: list[str]
    courts: list[str]
    laws_or_regulations: list[str]
    legal_citations: list[str]
    clause_types: list[str]
    parties: list[str]
    dates: list[str]
    time_constraints: list[str]
    obligations: list[str]
    remedies: list[str]
    procedural_posture: list[str]
    causes_of_action: list[str]
    factual_entities: list[str]
    keywords: list[str]
    filters: LegalEntityFilters
    ambiguity_notes: list[str]
    warnings: list[str]
    extraction_notes: list[str] | None


@dataclass(slots=True)
class LegalEntityExtractor:
    """Deterministic parser for high-precision legal entity extraction.

    The extractor keeps the retrieval tool thin and debuggable:
    1) deterministic patterns and controlled vocab only,
    2) conservative field population,
    3) filter derivation strictly from extracted entities.
    """

    _DOCUMENT_TYPES: tuple[str, ...] = (
        "contract",
        "lease",
        "non-disclosure agreement",
        "employment agreement",
        "statute",
        "regulation",
        "case",
        "judgment",
        "policy",
    )
    _LEGAL_TOPICS: tuple[str, ...] = (
        "termination",
        "breach",
        "confidentiality",
        "indemnity",
        "negligence",
        "liability",
        "arbitration",
        "governing law",
    )
    _JURISDICTIONS: tuple[str, ...] = (
        "ontario",
        "canada",
        "new york",
        "delaware",
        "uk",
        "eu",
        "federal",
        "california",
    )
    _COURTS: tuple[str, ...] = (
        "supreme court",
        "court of appeal",
        "chancery court",
        "district court",
        "court of chancery",
    )
    _LAWS: tuple[str, ...] = (
        "gdpr",
        "employment standards act",
        "ucc",
        "civil code",
    )
    _CLAUSE_TYPES: tuple[str, ...] = (
        "termination clause",
        "confidentiality clause",
        "indemnity clause",
        "arbitration clause",
        "governing law clause",
    )
    _PARTIES: tuple[str, ...] = (
        "tenant",
        "landlord",
        "employer",
        "employee",
        "plaintiff",
        "defendant",
    )
    _REMEDIES: tuple[str, ...] = ("damages", "injunction", "termination", "indemnity", "specific performance")
    _PROCEDURAL: tuple[str, ...] = ("motion to dismiss", "appeal", "summary judgment")
    _CAUSES: tuple[str, ...] = ("breach of contract", "fraud", "negligence", "unjust enrichment")
    _FACTUAL: tuple[str, ...] = ("notice period", "severance", "customer data", "trade secrets")
    _KEYWORD_CANDIDATES: tuple[str, ...] = (
        "enforceability",
        "elements",
        "exceptions",
        "standard",
        "interpretation",
    )

    _LEGAL_CITATION_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\bSection\s+[\w.\-()]+", flags=re.IGNORECASE),
        re.compile(r"\bRule\s+[\w.\-()]+", flags=re.IGNORECASE),
        re.compile(r"\b\d+\s+U\.S\.C\.\s+§+\s*[\w.\-()]+", flags=re.IGNORECASE),
    )
    _DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\b(19|20)\d{2}\b"),
        re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},\s*(19|20)\d{2}\b", flags=re.IGNORECASE),
    )
    _TIME_CONSTRAINT_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\bwithin\s+\d+\s+(?:day|days|month|months|year|years)\b", flags=re.IGNORECASE),
        re.compile(r"\b(?:after|before|since|until)\s+(?:\d{4}|[A-Za-z]+\s+\d{4})\b", flags=re.IGNORECASE),
        re.compile(r"\brecent\b", flags=re.IGNORECASE),
    )
    _OBLIGATION_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\bmust\b", flags=re.IGNORECASE),
        re.compile(r"\bshall\b", flags=re.IGNORECASE),
        re.compile(r"\bmay not\b", flags=re.IGNORECASE),
        re.compile(r"\brequired to\b", flags=re.IGNORECASE),
    )

    def extract(self, query: str) -> LegalEntityExtractionResult:
        """Parse a legal query into strict structured entities for RAG retrieval.

        Conservative extraction principle:
        - Explicit textual evidence only.
        - No guessed jurisdictions/courts/topics.
        - Filters are strict subset projections from extracted fields.
        """

        original_query = query
        raw = (query or "").strip()
        if not raw:
            return LegalEntityExtractionResult(
                original_query=original_query,
                normalized_query=None,
                document_types=[],
                legal_topics=[],
                jurisdictions=[],
                courts=[],
                laws_or_regulations=[],
                legal_citations=[],
                clause_types=[],
                parties=[],
                dates=[],
                time_constraints=[],
                obligations=[],
                remedies=[],
                procedural_posture=[],
                causes_of_action=[],
                factual_entities=[],
                keywords=[],
                filters=LegalEntityFilters(
                    jurisdiction=[],
                    court=[],
                    document_type=[],
                    date_from=None,
                    date_to=None,
                    clause_type=[],
                ),
                ambiguity_notes=[],
                warnings=["empty_input"],
                extraction_notes=["deterministic_pattern_matching"],
            )

        normalized = re.sub(r"\bNDA\b", "non-disclosure agreement", raw, flags=re.IGNORECASE)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        lowered = normalized.lower()

        document_types = self._find_terms(lowered, self._DOCUMENT_TYPES)
        legal_topics = self._find_terms(lowered, self._LEGAL_TOPICS)
        jurisdictions = self._title_terms(self._find_terms(lowered, self._JURISDICTIONS))
        courts = self._title_terms(self._find_terms(lowered, self._COURTS))
        laws = self._title_terms(self._find_terms(lowered, self._LAWS))
        clause_types = self._find_terms(lowered, self._CLAUSE_TYPES)
        parties = self._find_terms(lowered, self._PARTIES)
        remedies = self._find_terms(lowered, self._REMEDIES)
        procedural = self._find_terms(lowered, self._PROCEDURAL)
        causes = self._find_terms(lowered, self._CAUSES)
        factual = self._find_terms(lowered, self._FACTUAL)

        legal_citations = self._extract_patterns(normalized, self._LEGAL_CITATION_PATTERNS)
        dates = self._extract_patterns(normalized, self._DATE_PATTERNS)
        time_constraints = self._extract_patterns(normalized, self._TIME_CONSTRAINT_PATTERNS)
        obligations = self._extract_patterns(normalized, self._OBLIGATION_PATTERNS)

        ambiguity_notes: list[str] = []
        if "governing law" in lowered and not jurisdictions:
            ambiguity_notes.append("governing_law_without_jurisdiction")
        if re.search(r"\brecent\b", lowered) and not dates:
            ambiguity_notes.append("recent_without_explicit_timeframe")

        filters = LegalEntityFilters(
            jurisdiction=list(jurisdictions),
            court=list(courts),
            document_type=list(document_types),
            date_from=self._derive_date_boundary(time_constraints, "from"),
            date_to=self._derive_date_boundary(time_constraints, "to"),
            clause_type=list(clause_types),
        )

        keywords = self._derive_keywords(
            lowered=lowered,
            structured_values=[
                *document_types,
                *legal_topics,
                *jurisdictions,
                *courts,
                *laws,
                *legal_citations,
                *clause_types,
                *parties,
                *dates,
                *time_constraints,
                *obligations,
                *remedies,
                *procedural,
                *causes,
                *factual,
            ],
        )

        return LegalEntityExtractionResult(
            original_query=original_query,
            normalized_query=normalized if normalized != raw else None,
            document_types=document_types,
            legal_topics=legal_topics,
            jurisdictions=jurisdictions,
            courts=courts,
            laws_or_regulations=laws,
            legal_citations=legal_citations,
            clause_types=clause_types,
            parties=parties,
            dates=dates,
            time_constraints=time_constraints,
            obligations=obligations,
            remedies=remedies,
            procedural_posture=procedural,
            causes_of_action=causes,
            factual_entities=factual,
            keywords=keywords,
            filters=filters,
            ambiguity_notes=ambiguity_notes,
            warnings=[],
            extraction_notes=["deterministic_pattern_matching", "controlled_vocabulary_mapping"],
        )

    def _find_terms(self, lowered_query: str, terms: Sequence[str]) -> list[str]:
        return [term for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered_query)]

    def _title_terms(self, terms: Sequence[str]) -> list[str]:
        normalized: list[str] = []
        for item in terms:
            if item in {"uk", "eu", "gdpr", "ucc"}:
                normalized.append(item.upper())
            else:
                normalized.append(item.title())
        return normalized

    def _extract_patterns(self, query: str, patterns: Sequence[re.Pattern[str]]) -> list[str]:
        seen: list[str] = []
        for pattern in patterns:
            for match in pattern.finditer(query):
                value = match.group(0).strip()
                if value not in seen:
                    seen.append(value)
        return seen

    def _derive_date_boundary(self, constraints: Sequence[str], direction: str) -> str | None:
        for text in constraints:
            lowered = text.lower()
            year_match = re.search(r"\b(19|20)\d{2}\b", lowered)
            if not year_match:
                continue
            year = year_match.group(0)
            if direction == "from" and any(token in lowered for token in ("after", "since")):
                return f"{year}-01-01"
            if direction == "to" and any(token in lowered for token in ("before", "until")):
                return f"{year}-12-31"
        return None

    def _derive_keywords(self, lowered: str, structured_values: Sequence[str]) -> list[str]:
        blocked_terms = {token.strip().lower() for value in structured_values for token in value.split()}
        output: list[str] = []
        for candidate in self._KEYWORD_CANDIDATES:
            if candidate in lowered and candidate not in blocked_terms:
                output.append(candidate)
        return output


@dataclass(slots=True)
class QueryTransformationService:
    """Shared service for deterministic legal query rewriting and decomposition.

    Conversation context is optional. It is only applied when the incoming query
    appears referential/ambiguous and needs disambiguation for retrieval.
    """

    _AMBIGUOUS_REFERENCE_PATTERN = re.compile(
        r"\b(that|this|those|these|previous|prior)\s+(clause|section|case|example|one)\b|\b(it|that one|this one)\b",
        flags=re.IGNORECASE,
    )

    _LEGAL_REFERENCE_PATTERNS = (
        re.compile(r"\bSection\s+[\w.\-()]+", flags=re.IGNORECASE),
        re.compile(r"\bClause\s+[\w.\-()]+", flags=re.IGNORECASE),
        re.compile(r"\bArticle\s+[\w.\-()]+", flags=re.IGNORECASE),
        re.compile(r"\b[A-Z][a-zA-Z]+\s+v\.\s+[A-Z][a-zA-Z]+\b"),
        re.compile(r"\b[A-Z][A-Za-z\s]+\s+Act\b"),
    )
    llm_client: QueryTransformationLLM | None = None

    _PARTY_ROLE_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\bwho\s+is\s+the\s+(employer|employee)\b", flags=re.IGNORECASE),
        re.compile(r"\bwho\s+are\s+the\s+parties\b", flags=re.IGNORECASE),
        re.compile(r"\bwhich\s+company\s+is\s+this\s+agreement\s+for\b", flags=re.IGNORECASE),
        re.compile(r"\bis\s+this\s+agreement\s+between\b", flags=re.IGNORECASE),
    )
    _MATTER_METADATA_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\bwhat\s+is\s+the\s+file\s+number\b", flags=re.IGNORECASE),
        re.compile(r"\b(?:what|which)\s+jurisdiction\s+applies\b", flags=re.IGNORECASE),
        re.compile(r"\b(?:what|which)\s+court\s+is\s+involved\b", flags=re.IGNORECASE),
        re.compile(r"\bwho\s+is\s+the\s+client\b", flags=re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+the\s+(?:case|matter)\s+name\b", flags=re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+this\s+(?:matter|document)\s+about\b", flags=re.IGNORECASE),
    )
    _EMPLOYMENT_LIFECYCLE_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\bwhen\s+did\s+(?:employment|the employment relationship)\s+(?:begin|start|commence)\b", flags=re.IGNORECASE),
        re.compile(r"\b(?:employment\s+)?start\s+date\b", flags=re.IGNORECASE),
        re.compile(r"\bcommencement\s+date\b", flags=re.IGNORECASE),
        re.compile(r"\boffer\s+and\s+acceptance\b", flags=re.IGNORECASE),
        re.compile(r"\bwhen\s+was\s+the\s+offer\s+accepted\b", flags=re.IGNORECASE),
        re.compile(r"\bprobation(?:ary)?\b", flags=re.IGNORECASE),
        re.compile(r"\bcompensation\s+terms\b", flags=re.IGNORECASE),
        re.compile(r"\bsalary\b", flags=re.IGNORECASE),
        re.compile(r"\bbenefits\b", flags=re.IGNORECASE),
        re.compile(r"\btermination\s+effective\s+date\b", flags=re.IGNORECASE),
        re.compile(r"\bwhen\s+did\s+termination\s+take\s+effect\b", flags=re.IGNORECASE),
        re.compile(r"\bseverance\b", flags=re.IGNORECASE),
        re.compile(r"\broe\b", flags=re.IGNORECASE),
        re.compile(r"\brecord\s+of\s+employment\b", flags=re.IGNORECASE),
    )
    _EMPLOYMENT_MITIGATION_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\bmitigat(?:e|ion)\b", flags=re.IGNORECASE),
        re.compile(r"\bmitigation\s+efforts?\b", flags=re.IGNORECASE),
        re.compile(r"\bjob\s+applications?\b", flags=re.IGNORECASE),
        re.compile(r"\bhow\s+many\s+job\s+applications?\b", flags=re.IGNORECASE),
        re.compile(r"\binterviews?\b", flags=re.IGNORECASE),
        re.compile(r"\boffers?\s+(?:received|rejected)\b", flags=re.IGNORECASE),
        re.compile(r"\balternative\s+employment\b", flags=re.IGNORECASE),
        re.compile(r"\bnew\s+employment\b", flags=re.IGNORECASE),
        re.compile(r"\bmitigation\s+evidence\b", flags=re.IGNORECASE),
        re.compile(r"\bjob\s+search\s+log\b", flags=re.IGNORECASE),
    )

    def _is_party_role_entity_query(self, query: str) -> bool:
        normalized = (query or "").strip()
        if not normalized:
            return False
        return any(pattern.search(normalized) for pattern in self._PARTY_ROLE_QUERY_PATTERNS)

    def _expand_party_role_query(self, query: str) -> str:
        normalized = re.sub(r"\s+", " ", (query or "").strip())
        if not normalized:
            return ""

        expansion_terms = [
            "agreement parties",
            "between",
            "by and between",
            "made effective",
            "employment agreement",
            "employer",
            "employee",
            "company",
        ]
        expanded = f"{normalized} {' '.join(expansion_terms)}"
        return re.sub(r"\s+", " ", expanded).strip()

    def _is_matter_metadata_query(self, query: str) -> bool:
        normalized = (query or "").strip()
        if not normalized:
            return False
        return any(pattern.search(normalized) for pattern in self._MATTER_METADATA_QUERY_PATTERNS)

    def _expand_matter_metadata_query(self, query: str) -> str:
        normalized = re.sub(r"\s+", " ", (query or "").strip())
        if not normalized:
            return ""

        expansion_terms = [
            "caption",
            "introductory header",
            "matter information",
            "court heading",
            "file number",
            "court file number",
            "jurisdiction",
            "court",
            "client",
            "case name",
            "matter name",
            "governing forum",
        ]
        expanded = f"{normalized} {' '.join(expansion_terms)}"
        return re.sub(r"\s+", " ", expanded).strip()

    def _is_employment_lifecycle_query(self, query: str) -> bool:
        normalized = (query or "").strip()
        if not normalized:
            return False
        return any(pattern.search(normalized) for pattern in self._EMPLOYMENT_LIFECYCLE_QUERY_PATTERNS)

    def _expand_employment_lifecycle_query(self, query: str) -> str:
        normalized = re.sub(r"\s+", " ", (query or "").strip())
        if not normalized:
            return ""

        expansion_terms = [
            "effective date",
            "commencement",
            "term of employment",
            "offer acceptance",
            "probation period",
            "compensation",
            "salary",
            "benefits",
            "termination",
            "severance",
            "record of employment",
            "ROE",
        ]
        expanded = f"{normalized} {' '.join(expansion_terms)}"
        return re.sub(r"\s+", " ", expanded).strip()

    def _is_employment_mitigation_query(self, query: str) -> bool:
        normalized = (query or "").strip()
        if not normalized:
            return False
        return any(pattern.search(normalized) for pattern in self._EMPLOYMENT_MITIGATION_QUERY_PATTERNS)

    def _expand_employment_mitigation_query(self, query: str) -> str:
        normalized = re.sub(r"\s+", " ", (query or "").strip())
        if not normalized:
            return ""

        expansion_terms = [
            "mitigation efforts",
            "job search log",
            "mitigation journal",
            "application records",
            "resume submission",
            "interview invitation",
            "interview date",
            "offer letter",
            "offer received",
            "new employment start date",
            "employment update email",
        ]
        expanded = f"{normalized} {' '.join(expansion_terms)}"
        return re.sub(r"\s+", " ", expanded).strip()

    def rewrite_query(
        self,
        query: str,
        conversation_summary: str | None = None,
        recent_messages: Sequence[Any] | None = None,
    ) -> QueryRewriteResult:
        """Return one retrieval-optimized query string with optional context use."""

        original_query = query
        normalized_query = (query or "").strip()
        if not normalized_query:
            return QueryRewriteResult(
                original_query=original_query,
                rewritten_query="",
                used_conversation_context=False,
                rewrite_notes="empty_input",
            )

        if self._is_party_role_entity_query(normalized_query):
            return QueryRewriteResult(
                original_query=original_query,
                rewritten_query=self._expand_party_role_query(normalized_query),
                used_conversation_context=False,
                rewrite_notes="party_role_entity_query_expansion",
            )
        if self._is_matter_metadata_query(normalized_query):
            return QueryRewriteResult(
                original_query=original_query,
                rewritten_query=self._expand_matter_metadata_query(normalized_query),
                used_conversation_context=False,
                rewrite_notes="matter_document_metadata_query_expansion",
            )
        if self._is_employment_lifecycle_query(normalized_query):
            return QueryRewriteResult(
                original_query=original_query,
                rewritten_query=self._expand_employment_lifecycle_query(normalized_query),
                used_conversation_context=False,
                rewrite_notes="employment_contract_lifecycle_query_expansion",
            )
        if self._is_employment_mitigation_query(normalized_query):
            return QueryRewriteResult(
                original_query=original_query,
                rewritten_query=self._expand_employment_mitigation_query(normalized_query),
                used_conversation_context=False,
                rewrite_notes="employment_mitigation_query_expansion",
            )

        context_blob = _build_context_blob(conversation_summary, recent_messages)
        needs_context = bool(self._AMBIGUOUS_REFERENCE_PATTERN.search(normalized_query))

        if not needs_context:
            return QueryRewriteResult(
                original_query=original_query,
                rewritten_query=normalized_query,
                used_conversation_context=False,
                rewrite_notes="query_already_clear",
            )

        if self.llm_client is not None and context_blob:
            llm_ok, llm_result = self._llm_rewrite_query(normalized_query, context_blob)
            if llm_ok and llm_result is not None:
                return QueryRewriteResult(
                    original_query=original_query,
                    rewritten_query=llm_result,
                    used_conversation_context=True,
                    rewrite_notes="resolved_reference_with_llm",
                )
            if not llm_ok:
                return QueryRewriteResult(
                    original_query=original_query,
                    rewritten_query=normalized_query,
                    used_conversation_context=False,
                    rewrite_notes="llm_failure_fallback_original_query",
                )

        referent = self._extract_reference_target(context_blob) if context_blob else None
        if not referent:
            return QueryRewriteResult(
                original_query=original_query,
                rewritten_query=normalized_query,
                used_conversation_context=False,
                rewrite_notes="ambiguous_but_no_context_reference_found",
            )

        rewritten = self._replace_ambiguous_reference(normalized_query, referent)
        rewritten = re.sub(r"\s+", " ", rewritten).strip()

        return QueryRewriteResult(
            original_query=original_query,
            rewritten_query=rewritten,
            used_conversation_context=True,
            rewrite_notes="resolved_reference_from_context",
        )

    def decompose_query(
        self,
        query: str,
        conversation_summary: str | None = None,
        recent_messages: Sequence[Any] | None = None,
    ) -> QueryDecompositionResult:
        """Split legal queries into retrieval-oriented sub-queries."""

        rewrite_result = self.rewrite_query(
            query=query,
            conversation_summary=conversation_summary,
            recent_messages=recent_messages,
        )
        rewritten_query = rewrite_result.rewritten_query
        if not rewritten_query:
            return QueryDecompositionResult(
                original_query=query,
                sub_queries=(),
                used_conversation_context=rewrite_result.used_conversation_context,
                decomposition_notes="empty_input",
            )

        if not _is_complex_query(rewritten_query):
            return QueryDecompositionResult(
                original_query=query,
                sub_queries=(rewritten_query,),
                used_conversation_context=rewrite_result.used_conversation_context,
                decomposition_notes="single_query",
            )

        parts: list[str]
        if self.llm_client is not None:
            llm_ok, llm_parts = self._llm_decompose_query(rewritten_query, conversation_summary, recent_messages)
            if not llm_ok:
                safe_original = (query or "").strip()
                safe_parts = (safe_original,) if safe_original else ()
                return QueryDecompositionResult(
                    original_query=query,
                    sub_queries=safe_parts,
                    used_conversation_context=False,
                    decomposition_notes="llm_failure_fallback_original_query",
                )
            parts = llm_parts if llm_parts else self._split_into_sub_queries(rewritten_query)
        else:
            parts = self._split_into_sub_queries(rewritten_query)
        if not parts:
            parts = [rewritten_query]

        return QueryDecompositionResult(
            original_query=query,
            sub_queries=tuple(parts),
            used_conversation_context=rewrite_result.used_conversation_context,
            decomposition_notes="single_query" if len(parts) == 1 else "multi_issue_query",
        )

    def _extract_reference_target(self, context_blob: str) -> str | None:
        for pattern in self._LEGAL_REFERENCE_PATTERNS:
            matches = list(pattern.finditer(context_blob))
            if matches:
                return matches[-1].group(0)
        return None

    def _replace_ambiguous_reference(self, query: str, referent: str) -> str:
        replacements = (
            r"\b(that|this|those|these|previous|prior)\s+clause\b",
            r"\b(that|this|those|these|previous|prior)\s+section\b",
            r"\b(that|this|those|these|previous|prior)\s+case\b",
            r"\b(that|this|those|these|previous|prior)\s+example\b",
            r"\b(that one|this one|it)\b",
        )
        rewritten = query
        for pattern in replacements:
            rewritten = re.sub(pattern, referent, rewritten, flags=re.IGNORECASE)
        return rewritten

    def _llm_rewrite_query(self, query: str, context_blob: str) -> tuple[bool, str | None]:
        prompt = (
            "Rewrite the legal retrieval query by resolving ambiguous references only from context. "
            "Preserve jurisdiction, dates, parties, and legal scope. "
            "Return strict JSON: {\"rewritten_query\": \"...\"}.\n\n"
            f"CONTEXT:\n{context_blob}\n\nQUERY:\n{query}"
        )
        try:
            raw = self.llm_client.complete(prompt)  # type: ignore[union-attr]
            parsed = json.loads(raw)
            rewritten = str(parsed.get("rewritten_query", "")).strip()
            return True, rewritten or None
        except Exception:
            logger.exception("LLM rewrite failed; using heuristic/safe fallback.")
            return False, None

    def _llm_decompose_query(
        self,
        query: str,
        conversation_summary: str | None,
        recent_messages: Sequence[Any] | None,
    ) -> tuple[bool, list[str] | None]:
        context_blob = _build_context_blob(conversation_summary, recent_messages)
        prompt = (
            "Decompose the legal query into minimal non-overlapping retrieval sub-queries. "
            "Do not invent legal issues. Preserve deterministic ordering. "
            "Return strict JSON: {\"sub_queries\": [\"...\"]}.\n\n"
            f"CONTEXT:\n{context_blob}\n\nQUERY:\n{query}"
        )
        try:
            raw = self.llm_client.complete(prompt)  # type: ignore[union-attr]
            parsed = json.loads(raw)
            items = parsed.get("sub_queries")
            if not isinstance(items, list):
                return True, None
            cleaned = [str(item).strip() for item in items if str(item).strip()]
            return True, list(dict.fromkeys(cleaned)) or None
        except Exception:
            logger.exception("LLM decomposition failed; using heuristic/safe fallback.")
            return False, None

    def _split_into_sub_queries(self, query: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", query).strip()
        if not normalized:
            return []

        base_parts = [part.strip(" .") for part in re.split(r"\s*;\s*", normalized) if part.strip()]
        output: list[str] = []
        for part in base_parts:
            if _is_rule_exception_pattern(part):
                split_rule = re.split(r"\band\b", part, maxsplit=1, flags=re.IGNORECASE)
                output.extend(chunk.strip(" .") for chunk in split_rule if chunk.strip())
                continue

            if _should_split_on_and(part):
                split_parts = re.split(r"\band\b", part, flags=re.IGNORECASE)
                output.extend(chunk.strip(" .") for chunk in split_parts if chunk.strip())
                continue

            output.append(part)

        deduped = list(dict.fromkeys(output))
        return deduped


def _build_context_blob(conversation_summary: str | None, recent_messages: Sequence[Any] | None) -> str:
    summary_text = (conversation_summary or "").strip()
    message_texts: list[str] = []
    for message in recent_messages or ():
        if isinstance(message, str):
            text = message.strip()
        elif isinstance(message, Mapping):
            text = str(message.get("content") or message.get("text") or message.get("message") or "").strip()
        else:
            text = str(message).strip()
        if text:
            message_texts.append(text)
    return "\n".join([summary_text, *message_texts]).strip()


def _is_rule_exception_pattern(part: str) -> bool:
    lowered = part.lower()
    return "rule" in lowered and "exception" in lowered and " and " in lowered


def _should_split_on_and(part: str) -> bool:
    lowered = part.lower()
    if " and " not in lowered:
        return False

    anchor_keywords = (
        "what is",
        "what are",
        "compare",
        "difference",
        "definition",
        "enforcement",
        "remedy",
        "elements",
        "defenses",
        "jurisdiction",
        "procedural",
        "substantive",
    )
    matched_keywords = sum(1 for keyword in anchor_keywords if keyword in lowered)
    return matched_keywords >= 2


def _is_complex_query(query: str) -> bool:
    lowered = query.lower()
    if ";" in query:
        return True
    if " and " in lowered and ("rule" in lowered or "definition" in lowered or "compare" in lowered):
        return True
    return False


_DEFAULT_QUERY_TRANSFORMATION_SERVICE = QueryTransformationService()
_DEFAULT_LEGAL_ENTITY_EXTRACTOR = LegalEntityExtractor()


def rewrite_query(
    query: str,
    conversation_summary: str | None = None,
    recent_messages: Sequence[Any] | None = None,
) -> QueryRewriteResult:
    """Rewrite a user query into a retrieval-optimized legal query.

    The function is deterministic and optionally uses conversation context only
    to resolve ambiguous references in follow-up turns.
    """

    return _DEFAULT_QUERY_TRANSFORMATION_SERVICE.rewrite_query(
        query=query,
        conversation_summary=conversation_summary,
        recent_messages=recent_messages,
    )


def decompose_query(
    query: str,
    conversation_summary: str | None = None,
    recent_messages: Sequence[Any] | None = None,
) -> QueryDecompositionResult:
    """Break a legal query into retrieval-focused sub-queries.

    Decomposition is conservative: simple queries return a single sub-query,
    while clearly multi-part legal requests are split deterministically.
    """

    return _DEFAULT_QUERY_TRANSFORMATION_SERVICE.decompose_query(
        query=query,
        conversation_summary=conversation_summary,
        recent_messages=recent_messages,
    )


def extract_legal_entities(query: str) -> LegalEntityExtractionResult:
    """Extract structured legal entities from a user query for retrieval.

    This tool is parsing-only (not retrieval, memory, or answer generation).
    It prioritizes high-precision deterministic extraction and derives filters
    solely from extracted entities for safer downstream retrieval/ranking.
    """

    return _DEFAULT_LEGAL_ENTITY_EXTRACTOR.extract(query=query)
