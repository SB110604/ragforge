"""ragforge: hybrid-search retrieval-augmented generation with semantic chunking, reranking and cited answers."""

from .chunking import Chunk, SemanticChunker
from .embeddings import FastEmbedEmbedder, HashingEmbedder, OpenAICompatibleEmbedder
from .evaluate import EvalCase, evaluate, load_dataset
from .generate import NOT_FOUND, Answer, Citation, ExtractiveGenerator, LLMGenerator
from .llm import LLMError, OpenAICompatibleLLM
from .loaders import Document, LoaderError, Page, load_document
from .pipeline import RAGPipeline
from .retrieval import CrossEncoderReranker, Hit, HybridRetriever, OverlapReranker, QueryRewriter
from .store import BM25, VectorStore

__version__ = "0.1.0"
__all__ = [
    "Chunk",
    "SemanticChunker",
    "FastEmbedEmbedder",
    "HashingEmbedder",
    "OpenAICompatibleEmbedder",
    "EvalCase",
    "evaluate",
    "load_dataset",
    "NOT_FOUND",
    "Answer",
    "Citation",
    "ExtractiveGenerator",
    "LLMGenerator",
    "LLMError",
    "OpenAICompatibleLLM",
    "Document",
    "LoaderError",
    "Page",
    "load_document",
    "RAGPipeline",
    "CrossEncoderReranker",
    "Hit",
    "HybridRetriever",
    "OverlapReranker",
    "QueryRewriter",
    "BM25",
    "VectorStore",
    "__version__",
]
