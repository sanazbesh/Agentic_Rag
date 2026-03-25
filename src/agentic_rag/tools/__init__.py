"""Tooling module interfaces and query-intelligence utilities."""

from .interfaces import Tool, ToolRegistry
from .context_processing import (
    CompressContextResult,
    CompressedParentChunk,
    ParentChunkCompressor,
    compress_context,
)
from .answer_generation import AnswerCitation, GenerateAnswerResult, LegalAnswerSynthesizer, generate_answer
from .query_intelligence import (
    QueryDecompositionResult,
    QueryRewriteResult,
    QueryTransformationService,
    decompose_query,
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
    "QueryRewriteResult",
    "QueryDecompositionResult",
    "QueryTransformationService",
    "rewrite_query",
    "decompose_query",
]
