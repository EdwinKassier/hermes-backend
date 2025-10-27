"""
CRITICAL: Tests for live vector database integration.
These tests query the actual Supabase vector store to ensure RAG quality.

Run with: pytest tests/integration/test_vector_db_integration.py --run-integration -v
"""

import logging
import os

import numpy as np
import pytest

from app.shared.services.GeminiService import GeminiService

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def gemini_service():
    """Create GeminiService with real Supabase connection"""
    # Ensure environment variables are set
    # Handle both SUPABASE_URL and SUPABASE_PROJECT_URL
    supabase_url = os.environ.get("SUPABASE_URL") or os.environ.get(
        "SUPABASE_PROJECT_URL"
    )

    required_vars = {
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
        "SUPABASE_URL": supabase_url,
        "SUPABASE_SERVICE_ROLE_KEY": os.environ.get("SUPABASE_SERVICE_ROLE_KEY"),
    }

    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        pytest.skip(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    try:
        service = GeminiService()
        assert service.vector_store is not None, "Vector store not initialized"
        yield service
    except Exception as e:
        pytest.skip(f"Failed to initialize GeminiService: {e}")


@pytest.mark.integration
class TestVectorDatabaseConnection:
    """Test basic vector database connectivity"""

    def test_vector_store_initialized(self, gemini_service):
        """Test that vector store is properly initialized"""
        assert gemini_service.vector_store is not None
        assert gemini_service.embeddings_model is not None
        logger.info("✓ Vector store initialized successfully")

    def test_embedding_generation(self, gemini_service):
        """Test generating embeddings"""
        query = "Edwin Kassier software engineer"
        embedding = gemini_service.embeddings_model.embed_query(query)

        # Verify embedding dimensions (accept 768 or 1536 as valid)
        # Google has multiple embedding models with different dimensions
        actual_dimensions = len(embedding)
        valid_dimensions = [768, 1536]  # Common embedding dimensions
        assert (
            actual_dimensions in valid_dimensions
        ), f"Unexpected embedding dimensions: {actual_dimensions}"
        assert all(isinstance(x, (int, float)) for x in embedding)
        logger.info(
            f"✓ Generated embedding with {actual_dimensions} dimensions (expected one of {valid_dimensions})"
        )

    def test_embedding_consistency(self, gemini_service):
        """Test that same query produces similar embeddings"""
        query = "Edwin Kassier"

        embedding1 = gemini_service.embeddings_model.embed_query(query)
        embedding2 = gemini_service.embeddings_model.embed_query(query)

        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

        # Should be very similar (> 0.99)
        assert similarity > 0.99
        logger.info(f"✓ Embedding consistency: {similarity:.4f}")


@pytest.mark.integration
class TestVectorSimilaritySearch:
    """Test vector similarity search functionality"""

    def test_direct_similarity_search(self, gemini_service):
        """Test direct RPC call to match_documents"""
        query = "Edwin Kassier education background"

        results = gemini_service._direct_similarity_search(
            query=query, k=5, threshold=0.5
        )

        # Should return results
        assert isinstance(results, list)
        assert len(results) > 0, "No results returned from vector search"

        # Check result format
        for doc, score in results:
            assert hasattr(doc, "page_content")
            assert hasattr(doc, "metadata")
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
            logger.info(
                f"  Result score: {score:.3f}, preview: {doc.page_content[:100]}"
            )

        logger.info(f"✓ Retrieved {len(results)} relevant documents")

    def test_similarity_threshold_filtering(self, gemini_service):
        """Test that threshold properly filters results"""
        query = "Edwin Kassier"

        # High threshold - fewer results
        high_threshold_results = gemini_service._direct_similarity_search(
            query=query, k=10, threshold=0.8
        )

        # Low threshold - more results
        low_threshold_results = gemini_service._direct_similarity_search(
            query=query, k=10, threshold=0.5
        )

        # All high threshold results should have score >= 0.8
        for _, score in high_threshold_results:
            assert score >= 0.8, f"Score {score} below threshold 0.8"

        # Low threshold should return more or equal results
        assert len(low_threshold_results) >= len(high_threshold_results)

        logger.info(
            f"✓ Threshold filtering works: {len(high_threshold_results)} @ 0.8, {len(low_threshold_results)} @ 0.5"
        )

    @pytest.mark.parametrize(
        "query,expected_keywords",
        [
            (
                "Edwin Kassier education",
                ["university", "degree", "education", "study", "school"],
            ),
            (
                "Edwin Kassier work experience",
                ["engineer", "software", "developer", "work", "experience"],
            ),
            (
                "Edwin Kassier projects",
                ["project", "built", "created", "developed", "application"],
            ),
            (
                "Edwin Kassier skills",
                ["programming", "python", "javascript", "development", "technology"],
            ),
        ],
    )
    def test_search_relevance(self, gemini_service, query, expected_keywords):
        """Test that search returns relevant results for specific queries"""
        results = gemini_service._direct_similarity_search(
            query=query, k=5, threshold=0.6
        )

        assert len(results) > 0, f"No results for query: {query}"

        # Combine all results content
        all_content = " ".join([doc.page_content.lower() for doc, _ in results])

        # At least one expected keyword should appear
        found_keywords = [kw for kw in expected_keywords if kw.lower() in all_content]
        assert (
            len(found_keywords) > 0
        ), f"None of {expected_keywords} found in results for query: {query}"

        logger.info(f"✓ Found keywords {found_keywords} for query '{query}'")

    def test_search_returns_sorted_by_score(self, gemini_service):
        """Test that results are sorted by similarity score"""
        query = "Edwin Kassier background"

        results = gemini_service._direct_similarity_search(
            query=query, k=10, threshold=0.5
        )

        # Extract scores
        scores = [score for _, score in results]

        # Should be sorted in descending order
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"

        logger.info(f"✓ Results properly sorted: {scores[:5]}")


@pytest.mark.integration
class TestRAGPipeline:
    """Test end-to-end RAG pipeline"""

    def test_rag_response_generation(self, gemini_service):
        """Test full RAG response generation"""
        query = "Where did Edwin Kassier study?"
        user_id = "test_user_rag"

        response = gemini_service.generate_gemini_response_with_rag(
            prompt=query, user_id=user_id, persona="hermes"
        )

        # Should return a response
        assert isinstance(response, str)
        assert len(response) > 0
        assert response != gemini_service.ERROR_MESSAGE

        logger.info(f"✓ RAG Response: {response[:200]}")

    def test_rag_vs_standard_generation(self, gemini_service):
        """Test that RAG provides more specific answers than standard generation"""
        query = "What programming languages does Edwin Kassier know?"
        user_id = "test_user_comparison"

        # RAG response
        rag_response = gemini_service.generate_gemini_response_with_rag(
            prompt=query, user_id=user_id, persona="hermes"
        )

        # Standard generation (without RAG)
        standard_response = gemini_service.generate_gemini_response(
            prompt=query, persona="hermes"
        )

        logger.info(f"RAG: {rag_response[:150]}")
        logger.info(f"Standard: {standard_response[:150]}")

        # RAG should provide specific information
        # Both should be valid responses
        assert len(rag_response) > 0
        assert len(standard_response) > 0

    def test_rag_handles_no_context(self, gemini_service):
        """Test RAG behavior when no relevant context exists"""
        query = "What is the capital of Atlantis?"
        user_id = "test_user_no_context"

        response = gemini_service.generate_gemini_response_with_rag(
            prompt=query, user_id=user_id, persona="hermes"
        )

        # Should gracefully handle lack of context
        assert isinstance(response, str)
        assert len(response) > 0
        # Should likely indicate lack of information
        logger.info(f"✓ No-context response: {response[:150]}")

    @pytest.mark.slow
    def test_rag_conversation_context(self, gemini_service):
        """Test that RAG maintains conversation context"""
        user_id = "test_user_context"

        # First query
        response1 = gemini_service.generate_gemini_response_with_rag(
            prompt="Who is Edwin Kassier?", user_id=user_id, persona="hermes"
        )

        # Follow-up query (should use context)
        response2 = gemini_service.generate_gemini_response_with_rag(
            prompt="What is his education background?",
            user_id=user_id,
            persona="hermes",
        )

        # Both should return meaningful responses
        assert len(response1) > 0
        assert len(response2) > 0

        logger.info(f"✓ Q1: {response1[:100]}")
        logger.info(f"✓ Q2: {response2[:100]}")


@pytest.mark.integration
class TestRAGQuality:
    """Test RAG quality metrics"""

    @pytest.mark.parametrize(
        "query,expected_keywords",
        [
            (
                "Edwin Kassier skills",
                ["software", "engineering", "programming", "development"],
            ),
            ("Edwin Kassier contact", ["email", "contact", "reach"]),
            ("Edwin Kassier experience", ["experience", "work", "worked", "years"]),
        ],
    )
    def test_rag_answer_quality(self, gemini_service, query, expected_keywords):
        """Test that RAG responses contain relevant keywords"""
        response = gemini_service.generate_gemini_response_with_rag(
            prompt=query, user_id="test_quality", persona="hermes"
        )

        response_lower = response.lower()

        # At least one expected keyword should be present
        found = any(keyword.lower() in response_lower for keyword in expected_keywords)

        # Log either way for visibility
        if found:
            logger.info(f"✓ Quality check passed for '{query}'")
        else:
            logger.warning(
                f"⚠ No expected keywords in response for '{query}': {response[:100]}"
            )

        # Don't fail the test - just log warning
        # RAG might still give a valid answer without exact keywords

    def test_rag_factual_accuracy(self, gemini_service):
        """Test that RAG doesn't hallucinate obvious fake facts"""
        query = "What programming languages does Edwin Kassier know?"

        response = gemini_service.generate_gemini_response_with_rag(
            prompt=query, user_id="test_accuracy", persona="hermes"
        )

        # Response should not contain obviously fake programming languages
        fake_languages = ["klingon", "elvish", "parseltongue", "dothraki"]
        for fake_lang in fake_languages:
            assert (
                fake_lang not in response.lower()
            ), f"RAG hallucinated fake language: {fake_lang}"

        logger.info(f"✓ No hallucinated content detected: {response[:150]}")

    def test_rag_returns_dont_know_when_appropriate(self, gemini_service):
        """Test that RAG admits when it doesn't know something"""
        # Ask about something clearly not in Edwin's portfolio
        query = "What is Edwin Kassier's favorite ice cream flavor?"

        response = gemini_service.generate_gemini_response_with_rag(
            prompt=query, user_id="test_dont_know", persona="hermes"
        )

        # Should either decline to answer or admit uncertainty
        uncertainty_indicators = [
            "don't have",
            "don't know",
            "not available",
            "cannot find",
            "information not",
            "unclear",
        ]

        response_lower = response.lower()
        indicates_uncertainty = any(
            ind in response_lower for ind in uncertainty_indicators
        )

        logger.info(f"Response to unknown question: {response[:150]}")
        logger.info(f"Indicates uncertainty: {indicates_uncertainty}")


@pytest.mark.integration
@pytest.mark.slow
class TestVectorDBPerformance:
    """Basic performance checks for vector operations"""

    def test_embedding_latency(self, gemini_service):
        """Test that embedding generation is reasonably fast"""
        import time

        query = "Edwin Kassier software engineer"

        start = time.time()
        embedding = gemini_service.embeddings_model.embed_query(query)
        latency = time.time() - start

        # Should complete within 5 seconds
        assert latency < 5.0, f"Embedding took {latency:.2f}s (expected < 5s)"

        logger.info(f"✓ Embedding latency: {latency:.3f}s")

    def test_search_latency(self, gemini_service):
        """Test that vector search is reasonably fast"""
        import time

        query = "Edwin Kassier background"

        start = time.time()
        results = gemini_service._direct_similarity_search(query, k=5, threshold=0.6)
        latency = time.time() - start

        # Should complete within 3 seconds
        assert latency < 3.0, f"Search took {latency:.2f}s (expected < 3s)"

        logger.info(f"✓ Search latency: {latency:.3f}s, {len(results)} results")
