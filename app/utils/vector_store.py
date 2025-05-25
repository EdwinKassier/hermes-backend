"""
Utility class for managing vector store operations.
"""

import io
import logging
import traceback
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from google.cloud import storage
import pypdf


class VectorStoreService:
    """Service for managing vector store operations with Google Cloud Storage integration."""

    def __init__(
        self,
        storage_client: storage.Client,
        embeddings_model,
        chunk_size: int = 1500,
        chunk_overlap: int = 250,
        rag_top_k: int = 3,
    ):
        """
        Initialize the VectorStoreService.

        Args:
            storage_client: Initialized Google Cloud Storage client
            embeddings_model: Model to use for embeddings
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between text chunks
            rag_top_k: Number of top relevant chunks to retrieve
        """
        self.storage_client = storage_client
        self.embeddings_model = embeddings_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.rag_top_k = rag_top_k
        self.gcs_bucket_name: Optional[str] = None
        self.gcs_folder_path: Optional[str] = None
        self.vector_store: Optional[VectorStore] = None

    def update_rag_sources(
        self, gcs_bucket_name: str, gcs_folder_path: Optional[str] = None
    ) -> None:
        """
        Update RAG sources from a GCS bucket.

        Args:
            gcs_bucket_name: Name of the GCS bucket
            gcs_folder_path: Optional folder path within the bucket
        """
        logging.info(
            f"Updating RAG sources from GCS bucket: gs://{gcs_bucket_name}/{gcs_folder_path or ''}"
        )
        self.gcs_bucket_name = gcs_bucket_name
        if gcs_folder_path and not gcs_folder_path.endswith("/"):
            self.gcs_folder_path = gcs_folder_path + "/"
        elif not gcs_folder_path:
            self.gcs_folder_path = ""
        else:
            self.gcs_folder_path = gcs_folder_path
        self._build_vector_store()

    def clear_rag_sources(self) -> None:
        """Clear RAG sources and vector store."""
        logging.info("Clearing RAG sources and vector store.")
        self.gcs_bucket_name = None
        self.gcs_folder_path = None
        self.vector_store = None

    def get_relevant_chunks(self, query: str) -> List[Document]:
        """
        Get relevant document chunks for a query.

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
                retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": self.rag_top_k}
                )
                return retriever.invoke(query)
            else:
                logging.error(
                    "Vector store is not valid or does not have an 'as_retriever' method."
                )
                return []
        except Exception as e:
            logging.error(f"Error retrieving documents from vector store: {e}")
            return []

    def _build_vector_store(self) -> None:
        """Build the vector store from documents in GCS."""
        if not self.gcs_bucket_name:
            logging.info("No GCS bucket name set. Vector store will not be built.")
            self.vector_store = None
            return

        logging.info(
            f"Building vector store from GCS: gs://{self.gcs_bucket_name}/{self.gcs_folder_path or ''}..."
        )
        documents = self._get_documents_from_gcs_bucket()
        if not documents:
            logging.warning(
                "No documents could be processed from the GCS location. Vector store not built."
            )
            self.vector_store = None
            return

        self.vector_store = self._create_vector_store_from_documents(documents)
        if self.vector_store:
            logging.info(
                f"Successfully built/updated vector store with {len(documents)} source documents."
            )
        else:
            logging.warning("Failed to build vector store.")

    def _get_documents_from_gcs_bucket(self) -> List[Document]:
        """Get documents from GCS bucket."""
        all_documents = []
        if not self.gcs_bucket_name:
            logging.info("No GCS bucket specified for document extraction.")
            return []
        if not self.storage_client:
            logging.error("GCS storage client not initialized. Cannot fetch documents.")
            return []

        try:
            bucket = self.storage_client.bucket(self.gcs_bucket_name)
            blobs = bucket.list_blobs(prefix=self.gcs_folder_path or None)

            found_pdfs = False
            for blob in blobs:
                if blob.name.lower().endswith(".pdf"):
                    if blob.name == self.gcs_folder_path and blob.name.endswith("/"):
                        continue

                    found_pdfs = True
                    full_gcs_path = f"gs://{self.gcs_bucket_name}/{blob.name}"
                    logging.info(f"Fetching and parsing PDF: {full_gcs_path}")
                    try:
                        pdf_bytes = blob.download_as_bytes()
                        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                        extracted_text_content = ""
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                extracted_text_content += page_text + "\n"

                        if extracted_text_content.strip():
                            all_documents.append(
                                Document(
                                    page_content=extracted_text_content.strip(),
                                    metadata={"source": full_gcs_path},
                                )
                            )
                        else:
                            logging.warning(f"No text extracted from PDF: {full_gcs_path}")
                    except Exception as pdf_e:
                        logging.error(
                            f"Error processing PDF file {full_gcs_path}: {pdf_e}"
                        )
                        logging.error(traceback.format_exc())
            if not found_pdfs:
                logging.warning(
                    f"No PDF files found in gs://{self.gcs_bucket_name}/{self.gcs_folder_path or ''}"
                )

        except Exception as e:
            logging.error(
                f"Error listing or accessing GCS bucket gs://{self.gcs_bucket_name}/: {e}"
            )
            logging.error(traceback.format_exc())
        return all_documents

    def _create_vector_store_from_documents(
        self, documents: List[Document]
    ) -> Optional[VectorStore]:
        """Create a vector store from documents."""
        if not documents:
            logging.warning("No documents provided to create vector store.")
            return None
        split_documents = self.text_splitter.split_documents(documents)
        if not split_documents:
            logging.warning("Text splitting resulted in no document chunks.")
            return None
        logging.info(
            f"Split {len(documents)} documents into {len(split_documents)} chunks."
        )
        try:
            vector_store = InMemoryVectorStore.from_documents(
                documents=split_documents,
                embedding=self.embeddings_model,
            )
            logging.info("Successfully created in-memory vector store from documents.")
            return vector_store
        except Exception as e:
            logging.error(f"Failed to create vector store: {e}")
            logging.error(traceback.format_exc())
            return None 