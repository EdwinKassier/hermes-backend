"""
Utility class for managing vector store operations with Google Cloud Storage integration.
"""

import logging
import traceback
from typing import List, Optional, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore


class VectorStoreService:
    """Service for managing vector store operations with pre-generated embeddings."""

    def __init__(
        self,
        storage_client,
        embeddings_model,
        chunk_size: int = 300,  # Ultra-fine for maximum accuracy
        chunk_overlap: int = 200,  # Ultra-fine for maximum accuracy
        rag_top_k: int = 20,  # Ultra-fine for maximum accuracy
        pre_generated_embeddings: Optional[Dict[str, List[float]]] = None,
    ):
        """
        Initialize the VectorStoreService.

        Args:
            storage_client: Initialized Google Cloud Storage client (unused)
            embeddings_model: Model to use for embeddings
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between text chunks
            rag_top_k: Number of top relevant chunks to retrieve
            pre_generated_embeddings: Dictionary of pre-generated embeddings
        """
        self.embeddings_model = embeddings_model
        # Ultra-optimized text splitter for small datasets
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],  # More granular splitting
            length_function=len,
        )
        self.rag_top_k = rag_top_k
        self.vector_store: Optional[VectorStore] = None
        self.pre_generated_embeddings = pre_generated_embeddings or {}

        # Initialize vector store with pre-generated embeddings
        if self.pre_generated_embeddings:
            self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        """Initialize vector store with pre-generated embeddings."""
        try:
            # Create documents from pre-generated embeddings
            documents = [
                Document(page_content=text)
                for text in self.pre_generated_embeddings.keys()
            ]
            
            # Create vector store with pre-generated embeddings
            self.vector_store = InMemoryVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings_model,
                embedding_vectors=self.pre_generated_embeddings,
            )
            logging.info(
                f"Initialized vector store with {len(documents)} pre-generated embeddings"
            )
        except Exception as e:
            logging.error(f"Failed to initialize vector store: {e}")
            logging.error(traceback.format_exc())
            self.vector_store = None

    def update_rag_sources(
        self, gcs_bucket_name: str, gcs_folder_path: Optional[str] = None
    ) -> None:
        """
        Update RAG sources (no-op since we use pre-generated embeddings).
        
        Args:
            gcs_bucket_name: Name of the GCS bucket (unused)
            gcs_folder_path: Optional folder path within the bucket (unused)
        """
        logging.info("Using pre-generated embeddings, skipping GCS document fetch")

    def clear_rag_sources(self) -> None:
        """Clear vector store."""
        logging.info("Clearing vector store.")
        self.vector_store = None

    def get_relevant_chunks(self, query: str) -> List[Document]:
        """
        Get relevant document chunks for a query with enhanced retrieval.

        Args:
            query: The query to find relevant chunks for

        Returns:
            List of relevant document chunks
        """
        if not self.vector_store:
            logging.warning("No vector store available for RAG.")
            return []

        try:
            if hasattr(self.vector_store, "as_retriever"):
                # Ultra-optimized retriever for small datasets
                retriever = self.vector_store.as_retriever(
                    search_kwargs={
                        "k": self.rag_top_k,
                        "score_threshold": 0.3,  # Lower threshold for small datasets
                        "fetch_k": self.rag_top_k * 3,  # Fetch many candidates
                    }
                )
                chunks = retriever.invoke(query)
                
                # Enhanced processing for small datasets
                if chunks:
                    # Remove duplicates while preserving diversity
                    unique_chunks = []
                    seen_content = set()
                    
                    # Prioritize chunks from document beginnings
                    for chunk in chunks:
                        content_hash = hash(chunk.page_content[:200])  # Longer hash
                        if content_hash not in seen_content:
                            unique_chunks.append(chunk)
                            seen_content.add(content_hash)
                    
                    # For small datasets, return all relevant chunks
                    return unique_chunks[:self.rag_top_k]
                
                return chunks
            else:
                logging.error(
                    "Vector store is not valid or does not have an 'as_retriever' method."
                )
                return []
        except Exception as e:
            logging.error(f"Error retrieving documents from vector store: {e}")
            return [] 