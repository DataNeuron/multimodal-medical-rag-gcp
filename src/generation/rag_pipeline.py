# ============================================
# RAG Pipeline Module
# ============================================
"""
Retrieval-Augmented Generation pipeline for medical queries.

This module orchestrates:
    1. Query embedding generation
    2. Vector search for relevant context
    3. Context assembly and prompt construction
    4. LLM-based response generation
    5. Citation extraction and formatting

The pipeline is optimized for medical domain queries with
appropriate guardrails and disclaimers.
"""

import os
from dataclasses import dataclass
from typing import Optional

import vertexai
from vertexai.generative_models import GenerativeModel, Part

from src.retrieval import VectorSearchClient
from src.embeddings import embed_text
from src.utils.gcp_utils import initialize_vertex_ai


@dataclass
class RAGResponse:
    """
    Structured response from the RAG pipeline.

    Attributes:
        query: Original user query
        answer: Generated answer text
        sources: List of source documents used
        confidence: Confidence score (0-1)
        disclaimer: Medical disclaimer text
    """

    query: str
    answer: str
    sources: list[dict]
    confidence: float
    disclaimer: str = (
        "This information is for educational purposes only and should not "
        "be used as a substitute for professional medical advice."
    )


class RAGPipeline:
    """
    Complete RAG pipeline for medical question answering.

    Combines vector search retrieval with LLM generation
    for accurate, context-aware responses.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        max_context_length: int = 8000,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            model_name: Vertex AI LLM model name
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
            max_context_length: Maximum context length for LLM
        """
        self.model_name = model_name or os.getenv(
            "VERTEX_LLM_MODEL", "gemini-1.5-pro"
        )
        self.top_k = int(os.getenv("RAG_TOP_K", top_k))
        self.similarity_threshold = float(
            os.getenv("RAG_SIMILARITY_THRESHOLD", similarity_threshold)
        )
        self.max_context_length = int(
            os.getenv("RAG_MAX_CONTEXT_LENGTH", max_context_length)
        )
        self.temperature = float(os.getenv("RAG_TEMPERATURE", 0.2))

        # Initialize components
        project_id = os.getenv("GCP_PROJECT_ID")
        location = os.getenv("GCP_REGION", "us-central1")
        initialize_vertex_ai(project_id, location)

        self._retriever = VectorSearchClient()
        self._model = GenerativeModel(self.model_name)

    def query(self, query: str) -> RAGResponse:
        """
        Execute the full RAG pipeline.

        Args:
            query: User's medical question

        Returns:
            RAGResponse with answer and sources
        """
        # Step 1: Retrieve relevant context
        retrieved_docs = self._retrieve(query)

        # Step 2: Build context
        context = self._build_context(retrieved_docs)

        # Step 3: Generate response
        answer = self._generate(query, context)

        # Step 4: Calculate confidence
        confidence = self._calculate_confidence(retrieved_docs)

        return RAGResponse(
            query=query,
            answer=answer,
            sources=retrieved_docs,
            confidence=confidence,
        )

    def _retrieve(self, query: str) -> list[dict]:
        """Retrieve relevant documents using vector search."""
        results = self._retriever.search_with_text(query, self.top_k)

        # Filter by similarity threshold
        filtered = [
            r for r in results
            if r.get("distance", 0) >= self.similarity_threshold
        ]

        return filtered

    def _build_context(self, documents: list[dict]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []

        for i, doc in enumerate(documents, 1):
            doc_text = doc.get("text", doc.get("content", ""))
            source = doc.get("source", f"Document {i}")
            context_parts.append(f"[Source {i}: {source}]\n{doc_text}")

        context = "\n\n".join(context_parts)

        # Truncate if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."

        return context

    def _generate(self, query: str, context: str) -> str:
        """Generate response using the LLM."""
        prompt = self._build_prompt(query, context)

        response = self._model.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": 2048,
            },
        )

        return response.text

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the LLM prompt with context and query."""
        return f"""You are a medical information assistant. Answer the question based on the provided context.

IMPORTANT GUIDELINES:
- Only use information from the provided context
- If the context doesn't contain enough information, say so
- Be precise and cite sources when possible
- Include relevant medical terminology
- Do not make up information not in the context

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    def _calculate_confidence(self, documents: list[dict]) -> float:
        """Calculate confidence score based on retrieval quality."""
        if not documents:
            return 0.0

        # Average similarity of top results
        similarities = [d.get("distance", 0) for d in documents[:3]]
        return sum(similarities) / len(similarities)


def query_rag(query: str) -> RAGResponse:
    """
    Convenience function to query the RAG system.

    Args:
        query: Medical question

    Returns:
        RAGResponse with answer and sources
    """
    pipeline = RAGPipeline()
    return pipeline.query(query)


def generate_response(query: str, context: str) -> str:
    """
    Generate response given query and context.

    Args:
        query: User question
        context: Retrieved context text

    Returns:
        Generated answer text
    """
    pipeline = RAGPipeline()
    return pipeline._generate(query, context)


if __name__ == "__main__":
    print("RAG Pipeline for Medical Queries")
    print("Usage: from src.generation import query_rag")
    print("       response = query_rag('What are the findings in the chest X-ray?')")
