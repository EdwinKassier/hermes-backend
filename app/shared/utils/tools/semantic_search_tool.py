"""
Semantic Search Tool - Vector search and query enrichment using Gemini and Supabase.
"""

import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, ClassVar, Dict, List, Optional, Type

try:
    from langchain.tools import BaseTool
    from langchain_core.documents import Document
    from langchain_core.messages import HumanMessage
    from pydantic import BaseModel, Field, PrivateAttr

    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback/Mock classes for environments without LangChain installed
    class BaseTool:
        def __init__(self):
            self.name = "semantic_search"
            self.description = "Semantic Search tool"
            self.args_schema = None

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(*args, **kwargs):
        return None

    def PrivateAttr(**kwargs):
        return None

    Document = None
    LANGCHAIN_AVAILABLE = False

try:
    import google.generativeai as genai
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
except ImportError:
    genai = None
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None

# Optional Vector Store - Standard LangChain integration
try:
    from langchain_community.vectorstores import SupabaseVectorStore
    from supabase import create_client

    SUPABASE_AVAILABLE = True
except ImportError:
    SupabaseVectorStore = None
    create_client = None
    SUPABASE_AVAILABLE = False

logger = logging.getLogger(__name__)


# Custom exceptions for better error handling
class SemanticSearchError(Exception):
    """Base exception for SemanticSearchTool errors."""

    pass


class EmbeddingError(SemanticSearchError):
    """Raised when embedding generation fails."""

    pass


class VectorStoreError(SemanticSearchError):
    """Raised when vector store operations fail."""

    pass


class SemanticSearchInput(BaseModel):
    """Input schema for Semantic Search Tool."""

    query: str = Field(description="The search query to find relevant documents for.")

    k: int = Field(default=5, description="Number of results to return (default: 5).")

    threshold: float = Field(
        default=0.65, description="Minimum similarity threshold (0-1, default: 0.65)."
    )

    enrich_query: bool = Field(
        default=True,
        description="Whether to use AI to expand and enrich the query before searching (default: True).",
    )

    context: Optional[str] = Field(
        default=None,
        description="Optional conversation context to help with query enrichment.",
    )


class SemanticSearchTool(BaseTool):
    """
    Semantic Search Tool for retrieving relevant information from the vector database.

    Features:
    1. Vector similarity search using Supabase
    2. Query enrichment using Gemini to improve search quality
    3. Custom embedding handling (1536 dimensions)
    """

    name: str = "semantic_search"
    description: str = """
    Search the knowledge base using semantic vector search.
    Use this tool to find information about the project, codebase, or specific topics
    stored in the vector database.
    """
    args_schema: Type[BaseModel] = SemanticSearchInput

    # Configuration constants
    EMBEDDING_MODEL_NAME: ClassVar[str] = "models/embedding-001"
    EMBEDDING_DIMENSIONS: ClassVar[int] = 1536

    # Internal state
    _vector_store: Any = PrivateAttr(default=None)
    _embeddings_model: Any = PrivateAttr(default=None)

    def __init__(self):
        """Initialize Semantic Search Tool."""
        super().__init__()
        self._initialize_resources()

    def _initialize_resources(self):
        """Initialize embeddings and vector store connection."""
        if not LANGCHAIN_AVAILABLE:
            logger.warning(
                "LangChain dependencies not available for SemanticSearchTool"
            )
            return

        if "GOOGLE_API_KEY" not in os.environ:
            logger.warning("GOOGLE_API_KEY not set - SemanticSearchTool disabled")
            return

        # Initialize Gemini embeddings (1536 dimensions)
        try:
            # Configure the Gemini API
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

            # Create custom embeddings wrapper with explicit dimensionality
            _target_dimensionality = self.EMBEDDING_DIMENSIONS

            class CustomDimensionalityEmbeddings(GoogleGenerativeAIEmbeddings):
                """Custom embeddings that enforce 1536 dimensions"""

                def embed_query(self, text: str, **kwargs) -> list[float]:
                    """Override to add output_dimensionality parameter"""
                    try:
                        result = genai.embed_content(
                            model="models/embedding-001",
                            content=text,
                            task_type="retrieval_query",  # Correct task type for query
                            output_dimensionality=_target_dimensionality,
                        )
                        return result["embedding"]
                    except Exception as e:
                        logger.error(
                            "Failed to generate embedding for query: %s", str(e)
                        )
                        raise EmbeddingError(
                            f"Failed to generate embedding: {str(e)}"
                        ) from e

                def embed_documents(
                    self, texts: list[str], **kwargs
                ) -> list[list[float]]:
                    """Override to add output_dimensionality parameter"""
                    try:
                        return [self.embed_query(text) for text in texts]
                    except Exception as e:
                        logger.error(
                            "Failed to generate embeddings for documents: %s", str(e)
                        )
                        raise EmbeddingError(
                            f"Failed to generate document embeddings: {str(e)}"
                        ) from e

            self._embeddings_model = CustomDimensionalityEmbeddings(
                model="models/embedding-001",
                google_api_key=os.environ["GOOGLE_API_KEY"],
                task_type="retrieval_document",
            )

            logger.info(
                "Initialized Gemini API Embeddings with model: %s (%dD)",
                "models/embedding-001",
                self.EMBEDDING_DIMENSIONS,
            )

        except Exception as e:
            logger.error("Failed to initialize Gemini API Embeddings: %s", e)
            self._embeddings_model = None

        # Initialize Supabase vector store
        if not SUPABASE_AVAILABLE:
            logger.warning("Supabase dependencies not available for SemanticSearchTool")
            self._vector_store = None
        else:
            try:
                supabase_url = os.environ.get("SUPABASE_URL") or os.environ.get(
                    "SUPABASE_PROJECT_URL"
                )
                supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

                if not supabase_url or not supabase_key:
                    logger.warning(
                        "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set for SemanticSearchTool"
                    )
                    self._vector_store = None
                else:
                    supabase_client = create_client(supabase_url, supabase_key)
                    self._vector_store = SupabaseVectorStore(
                        client=supabase_client,
                        embedding=self._embeddings_model,
                        table_name="hermes_vectors",
                    )
                    logger.info(
                        "Initialized Supabase vector store for SemanticSearchTool"
                    )

            except Exception as e:
                logger.error("Failed to initialize Supabase vector store: %s", e)
                self._vector_store = None

    def _enrich_query(self, user_query: str, context: str = "") -> str:
        """
        Enrich a user query using Gemini to make it more detailed and suitable for vector search.
        """
        if not LANGCHAIN_AVAILABLE:
            return user_query

        enrichment_prompt = f"""You are a query expansion expert. Your task is to
enrich the following user query to make it more suitable for semantic vector
search in a document database.

Rules:
1. Expand vague queries with context, synonyms, and related terms
2. Preserve the original intent and meaning
3. Add technical terms, domain-specific vocabulary, and common variations
4. Keep the enriched query concise (2-4 sentences max) (Under 300 chars)
5. Focus on searchable keywords and concepts, not conversational fluff
6. If the query is already detailed, return it with minor enhancements only

Conversation Context: {context}

Original Query: {user_query}

Enriched Query:"""

        try:
            # Use a simpler model call without tools for query enrichment
            base_model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3,
                max_retries=2,
                timeout=10,
            )

            message = base_model.invoke([HumanMessage(content=enrichment_prompt)])
            enriched_query = message.content.strip()

            logger.info(f"Enriched query: '{user_query}' -> '{enriched_query}'")
            return enriched_query

        except Exception as e:
            logger.error("Error enriching query: %s", e)
            return user_query

    def _direct_similarity_search(
        self, query: str, k: int = 5, threshold: float = 0.7
    ) -> List[Any]:
        """
        Direct Supabase vector similarity search using RPC.
        """
        if not self._vector_store or not self._embeddings_model:
            logger.warning("Vector store or embeddings not initialized")
            return []

        try:

            def _execute_search_rpc():
                # Generate embedding
                query_embedding = self._embeddings_model.embed_query(query)

                # Call Supabase RPC function directly
                logger.info("Calling match_documents RPC function...")
                return self._vector_store._client.rpc(
                    "match_documents",
                    {
                        "query_embedding": query_embedding,
                        "match_count": k * 2,  # Get more results to filter by threshold
                        "filter": {},  # Empty filter for now
                    },
                ).execute()

            # Execute with timeout using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_execute_search_rpc)
                response = future.result(timeout=25)

            if not response.data:
                logger.warning("No data returned from match_documents RPC")
                return []

            # Convert to LangChain Document format and filter by threshold
            results = []
            for row in response.data:
                similarity = row.get("similarity", 0.0)

                if similarity >= threshold:
                    doc = Document(
                        page_content=row.get("content", ""),
                        metadata=row.get("metadata", {}),
                    )
                    # Add similarity score to metadata for visibility
                    doc.metadata["similarity_score"] = similarity
                    results.append((doc, similarity))

            # Sort by similarity (descending) and limit to k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

        except TimeoutError as e:
            logger.error(f"Vector search timed out: {e}")
            return []
        except Exception as e:
            logger.error("Direct similarity search failed: %s", e)
            logger.error(traceback.format_exc())
            return []

    def _run(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.65,
        enrich_query: bool = True,
        context: Optional[str] = None,
    ) -> str:
        """
        Execute the semantic search.

        Args:
            query: Search query
            k: Number of results
            threshold: Similarity threshold
            enrich_query: Whether to enrich the query
            context: Optional context for enrichment

        Returns:
            Formatted string of search results
        """
        if not self._vector_store:
            return "Error: Vector store is not available. Please check system configuration."

        try:
            search_query = query

            # Enrich query if requested
            if enrich_query:
                search_query = self._enrich_query(query, context or "")

            # Perform search
            results = self._direct_similarity_search(
                query=search_query, k=k, threshold=threshold
            )

            if not results:
                return f"No relevant documents found for query: '{query}' (threshold={threshold})"

            # Format results
            formatted_results = [
                f"Found {len(results)} relevant documents for: '{query}'\n"
            ]

            for i, (doc, score) in enumerate(results, 1):
                source = doc.metadata.get("source", "Unknown source")
                content = doc.page_content.strip()

                formatted_results.append(f"--- Document {i} (Score: {score:.3f}) ---")
                formatted_results.append(f"Source: {source}")
                formatted_results.append(f"Content: {content}\n")

            return "\n".join(formatted_results)

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return f"Error performing semantic search: {str(e)}"

    def _arun(self, **kwargs):
        """Async execution not supported."""
        raise NotImplementedError("SemanticSearchTool does not support async execution")
