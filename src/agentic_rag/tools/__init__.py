"""Tooling module interfaces and query-intelligence utilities."""

from .interfaces import Tool, ToolRegistry
from .context_processing import (
    CompressContextResult,
    CompressedParentChunk,
    ParentChunkCompressor,
    compress_context,
)
from .answer_generation import AnswerCitation, GenerateAnswerResult, LegalAnswerSynthesizer, generate_answer
from .answerability import (
    CoverageEvaluation,
    EvidenceStrengthEvaluation,
    AnswerabilityAssessment,
    AnswerabilityAssessor,
    assess_answerability,
    evaluate_coverage,
    evaluate_evidence_strength,
)
from .query_intelligence import (
    LegalEntityExtractionResult,
    LegalEntityExtractor,
    LegalEntityFilters,
    QueryDecompositionResult,
    QueryRewriteResult,
    QueryTransformationService,
    decompose_query,
    extract_legal_entities,
    rewrite_query,
)

__all__ = [
    "Tool",
    "ToolRegistry",
    "CompressedParentChunk",
    "CompressContextResult",
    "ParentChunkCompressor",
    "compress_context",
    "AnswerCitation",
    "GenerateAnswerResult",
    "LegalAnswerSynthesizer",
    "generate_answer",
    "CoverageEvaluation",
    "EvidenceStrengthEvaluation",
    "AnswerabilityAssessment",
    "AnswerabilityAssessor",
    "assess_answerability",
    "evaluate_coverage",
    "evaluate_evidence_strength",
    "QueryRewriteResult",
    "QueryDecompositionResult",
    "QueryTransformationService",
    "LegalEntityFilters",
    "LegalEntityExtractionResult",
    "LegalEntityExtractor",
    "rewrite_query",
    "decompose_query",
    "extract_legal_entities",
]
