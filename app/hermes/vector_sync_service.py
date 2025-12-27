"""Vector Sync Service - Orchestrates vector store synchronization with GCS bucket.

This service encapsulates the logic for:
- Loading documents from Google Cloud Storage
- Processing and chunking documents (with Gemini OCR for PDFs)
- Generating embeddings using Gemini API
- Upserting embeddings to Supabase vector store
"""

import hashlib
import logging
import mimetypes
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import numpy as np
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from supabase import Client as SupabaseClient
from supabase import create_client
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .exceptions import VectorSyncError

logger = logging.getLogger(__name__)

# Register MIME types
mimetypes.add_type("application/pdf", ".pdf")
mimetypes.add_type("text/plain", ".txt")
mimetypes.add_type("text/markdown", ".md")

# Text splitter configuration
TEXT_SPLITTER_SEPARATORS = [
    "\n\n## ",
    "\n# ",
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ": ",
    " ",
    "-",
    "\t",
    "",
]
MARKDOWN_HEADERS_TO_SPLIT = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]


@dataclass
class SyncConfig:
    """Configuration for vector sync operations."""

    embedding_model: str = "models/embedding-001"
    gemini_vision_model: str = "gemini-2.5-flash"
    embedding_dimensions: int = 1536
    output_dimensionality: int = 1536
    task_type: str = "RETRIEVAL_DOCUMENT"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 50
    max_retries: int = 5
    default_bucket: str = "ashes-project-hermes-training"
    min_text_length: int = 20
    enable_title_extraction: bool = True


@dataclass
class SyncResult:
    """Result of a vector sync operation."""

    status: str
    documents_processed: int = 0
    chunks_generated: int = 0
    embeddings_created: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class EnhancedDocumentSplitter:
    """Splits documents into fine-grained chunks with context and rich metadata."""

    def __init__(self, config: SyncConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=TEXT_SPLITTER_SEPARATORS,
        )
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT
        )

    def split_document(self, document: Document) -> List[Document]:
        """Splits a single document, adding chunk-specific metadata."""
        content = document.page_content
        if not content.strip():
            return []

        is_markdown = bool(re.search(r"^[\s]*#\s+", content, re.MULTILINE))
        doc_type = "markdown" if is_markdown else "text"
        base_metadata = {**document.metadata, "document_type": doc_type}

        chunks = (
            self._split_markdown(content, base_metadata)
            if is_markdown
            else self._split_text(content, base_metadata)
        )

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        logger.info(
            f"Split document '{document.metadata.get('file_name')}' into {len(chunks)} chunks."
        )
        return chunks

    def _split_markdown(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        md_sections = self.markdown_splitter.split_text(content)
        return [
            sub_chunk
            for i, section in enumerate(md_sections, 1)
            for sub_chunk in self.text_splitter.create_documents(
                [section.page_content],
                metadatas=[{**metadata, **section.metadata, "section_index": i}],
            )
        ]

    def _split_text(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        return self.text_splitter.create_documents([content], metadatas=[metadata])


class VectorSyncService:
    """Service for synchronizing vector store with documents from GCS bucket."""

    def __init__(self, config: Optional[SyncConfig] = None):
        """Initialize the vector sync service."""
        self.config = config or SyncConfig()
        self._supabase_client: Optional[SupabaseClient] = None
        self._vertex_initialized = False

        # Validate required environment variables
        self._google_api_key = os.environ.get("GOOGLE_API_KEY")
        self._supabase_url = os.environ.get("SUPABASE_PROJECT_URL") or os.environ.get(
            "SUPABASE_URL"
        )
        self._supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        self._google_project_id = os.environ.get("GOOGLE_PROJECT_ID")
        self._google_location = os.environ.get("GOOGLE_PROJECT_LOCATION", "us-central1")

    def _validate_credentials(self) -> None:
        """Validate that all required credentials are available."""
        missing = []
        if not self._google_api_key:
            missing.append("GOOGLE_API_KEY")
        if not self._supabase_url:
            missing.append("SUPABASE_PROJECT_URL or SUPABASE_URL")
        if not self._supabase_key:
            missing.append("SUPABASE_SERVICE_ROLE_KEY")
        if not self._google_project_id:
            missing.append("GOOGLE_PROJECT_ID")

        if missing:
            raise VectorSyncError(
                f"Missing required environment variables: {', '.join(missing)}",
                details={"missing_variables": missing},
            )

    def _init_clients(self) -> None:
        """Initialize API clients lazily."""
        if self._supabase_client is None:
            self._supabase_client = create_client(
                self._supabase_url, self._supabase_key
            )
            logger.info("Initialized Supabase client")

        if not self._vertex_initialized:
            import vertexai

            vertexai.init(
                project=self._google_project_id, location=self._google_location
            )
            self._vertex_initialized = True
            logger.info(
                f"Initialized Vertex AI for project '{self._google_project_id}'"
            )

        # Configure Gemini API for embeddings
        genai.configure(api_key=self._google_api_key)

    def sync_vectors(
        self,
        bucket_name: Optional[str] = None,
        folder_path: str = "",
        force_refresh: bool = False,
    ) -> SyncResult:
        """
        Synchronize vector store with documents from GCS bucket.

        Args:
            bucket_name: GCS bucket name (defaults to configured bucket)
            folder_path: Folder path within the bucket
            force_refresh: Force re-sync even if documents haven't changed

        Returns:
            SyncResult with operation details
        """
        start_time = datetime.now()
        errors: List[str] = []

        try:
            # Validate and initialize
            self._validate_credentials()
            self._init_clients()

            bucket = bucket_name or self.config.default_bucket
            logger.info(
                f"Starting vector sync from bucket '{bucket}', folder '{folder_path or 'root'}'"
            )

            # Load documents from GCS
            documents_data = self._load_documents_from_gcs(bucket, folder_path)
            if not documents_data:
                return SyncResult(
                    status="completed",
                    documents_processed=0,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    errors=["No documents found in the specified location"],
                )

            # Convert to LangChain documents
            documents = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in documents_data
            ]

            # Split into chunks
            splitter = EnhancedDocumentSplitter(self.config)
            all_chunks = self._process_and_split_documents(documents, splitter)

            if not all_chunks:
                return SyncResult(
                    status="completed",
                    documents_processed=len(documents),
                    chunks_generated=0,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    errors=["No chunks generated from documents"],
                )

            # Process batches and upsert
            total_embeddings = 0
            for i in range(0, len(all_chunks), self.config.batch_size):
                batch = all_chunks[i : i + self.config.batch_size]
                try:
                    self._process_batch(batch)
                    total_embeddings += len(batch)
                except Exception as e:
                    error_msg = (
                        f"Batch {i // self.config.batch_size + 1} failed: {str(e)}"
                    )
                    logger.error(error_msg)
                    errors.append(error_msg)

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Vector sync completed: {len(documents)} docs, "
                f"{len(all_chunks)} chunks, {total_embeddings} embeddings in {duration:.2f}s"
            )

            return SyncResult(
                status="completed" if not errors else "completed_with_errors",
                documents_processed=len(documents),
                chunks_generated=len(all_chunks),
                embeddings_created=total_embeddings,
                duration_seconds=duration,
                errors=errors,
            )

        except VectorSyncError:
            raise
        except Exception as e:
            logger.error(f"Vector sync failed: {e}", exc_info=True)
            raise VectorSyncError(
                f"Vector sync operation failed: {str(e)}",
                details={"error": str(e)},
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _load_documents_from_gcs(
        self, bucket_name: str, folder_path: str
    ) -> List[Dict]:
        """Load and process documents from GCS bucket."""
        from google.cloud import storage

        logger.info(
            f"Loading from GCS bucket '{bucket_name}', folder '{folder_path or 'root'}'..."
        )
        storage_client = storage.Client()
        blobs = list(
            storage_client.bucket(bucket_name).list_blobs(prefix=folder_path or "")
        )

        processed_docs = []
        for blob in blobs:
            if blob.name.endswith("/"):
                continue

            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, os.path.basename(blob.name))
                blob.download_to_filename(file_path)

                doc = self._process_document(file_path)
                if doc:
                    processed_docs.append(doc)

        logger.info(f"Successfully processed {len(processed_docs)} documents from GCS.")
        return processed_docs

    def _process_document(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Process a single document file."""
        file_name = os.path.basename(file_path)
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type or not any(
                mime_type.startswith(p) for p in ["text/", "application/pdf"]
            ):
                logger.warning(
                    f"Skipping unsupported file type '{mime_type}' for: {file_name}"
                )
                return None

            content = ""
            if mime_type == "application/pdf":
                content = self._extract_text_with_gemini(file_path, mime_type)
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

            cleaned_content = self._clean_text(content)
            if len(cleaned_content) < self.config.min_text_length:
                logger.warning(f"Skipping '{file_name}' due to insufficient content.")
                return None

            metadata = {
                "source": file_path,
                "file_name": file_name,
                "file_type": mime_type,
                "file_size": os.path.getsize(file_path),
                "content_hash": hashlib.sha256(
                    cleaned_content.encode("utf-8")
                ).hexdigest(),
                "load_timestamp": datetime.utcnow().isoformat(),
                "embedding_model": self.config.embedding_model,
                "ocr_model": (
                    self.config.gemini_vision_model
                    if mime_type == "application/pdf"
                    else "N/A"
                ),
                "title": (
                    self._extract_title(cleaned_content)
                    if self.config.enable_title_extraction
                    else "N/A"
                ),
            }
            return {"page_content": cleaned_content, "metadata": metadata}

        except Exception as e:
            logger.error(f"Failed to process document {file_name}: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _extract_text_with_gemini(self, file_path: str, mime_type: str) -> str:
        """Extract text from PDF using Gemini multimodal model."""
        from vertexai.generative_models import GenerativeModel, Part

        logger.info(
            f"Extracting text from '{os.path.basename(file_path)}' using Gemini..."
        )
        model = GenerativeModel(self.config.gemini_vision_model)

        with open(file_path, "rb") as f:
            pdf_content = f.read()

        document_part = Part.from_data(pdf_content, mime_type=mime_type)
        prompt = "Extract all text from this document. Preserve the original structure and formatting as much as possible."

        response = model.generate_content([document_part, prompt])
        full_text = "".join(part.text for part in response.candidates[0].content.parts)

        logger.info(f"Successfully extracted text from '{os.path.basename(file_path)}'")
        return full_text

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_title(self, content: str) -> Optional[str]:
        """Heuristically extract a title from the document content."""
        match = re.search(r"^\s*#\s+(.+)", content)
        if match:
            return match.group(1).strip()
        first_line = next(
            (line.strip() for line in content.split("\n") if line.strip()), None
        )
        if first_line and len(first_line) < 150:
            return first_line
        return None

    def _process_and_split_documents(
        self, docs: List[Document], splitter: EnhancedDocumentSplitter
    ) -> List[Document]:
        """Process and split all documents into chunks."""
        return [chunk for doc in docs for chunk in splitter.split_document(doc)]

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using Gemini API."""
        all_embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.config.embedding_model,
                content=text,
                task_type=self.config.task_type.lower(),
                output_dimensionality=self.config.output_dimensionality,
            )
            all_embeddings.append(np.array(result["embedding"], dtype=np.float32))
        return all_embeddings

    def _process_batch(self, batch: List[Document]) -> None:
        """Process a batch of documents: generate embeddings and upsert."""
        texts = [doc.page_content for doc in batch]
        embeddings = self._generate_embeddings(texts)

        class PrecomputedEmbedder:
            def __init__(self, embeddings_list, dimensions):
                self._embeddings = embeddings_list
                self._dimensions = dimensions

            def embed_documents(self, _texts):
                return [e.tolist() for e in self._embeddings]

            def embed_query(self, _text):
                return [0.0] * self._dimensions

        vector_store = SupabaseVectorStore(
            client=self._supabase_client,
            embedding=PrecomputedEmbedder(embeddings, self.config.embedding_dimensions),
            table_name="hermes_vectors",
            query_name="match_documents",
        )
        vector_store.add_documents(batch)
        logger.info(f"Successfully upserted batch of {len(batch)} chunks to Supabase.")


# Singleton instance
_vector_sync_service: Optional[VectorSyncService] = None


def get_vector_sync_service() -> VectorSyncService:
    """Get or create the VectorSyncService singleton."""
    global _vector_sync_service
    if _vector_sync_service is None:
        _vector_sync_service = VectorSyncService()
    return _vector_sync_service
