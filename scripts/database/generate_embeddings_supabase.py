#!/usr/bin/env python3
"""
Ultra-fine Vertex AI embedding generation for Supabase vector store using Gemini OCR.

This script uses Google Cloud's Vertex AI Gemini models for high-accuracy, multimodal OCR processing
of all PDF documents, removing the need for the Document AI API.

Key Features:
- Gemini-Powered OCR: Automatically uses a powerful Gemini model for OCR on PDFs,
  simplifying configuration and removing dependency on Document AI processors.
- High-Accuracy OCR: All PDFs are processed using Google Cloud's advanced multimodal models for
  maximum accuracy on both digital and scanned documents.
- Rich Metadata Enrichment: Automatically extracts titles and adds other contextual metadata.
- Context-aware, Semantic Chunking: Intelligently splits Markdown and plain text.
- Optimized for Supabase: Fine-tuned configuration for HNSW and GIN indexes for
  optimal vector retrieval performance.
"""

import os
import re
import sys
import json
import logging
import hashlib
import mimetypes
import tempfile
import unicodedata
from datetime import datetime
from typing import List, Optional, Any, Dict

from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
import numpy as np
import psycopg2
from psycopg2 import sql

from supabase import create_client, Client as SupabaseClient
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
)
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore

# --------------------- MIME TYPES ---------------------
mimetypes.add_type('application/pdf', '.pdf')
mimetypes.add_type('text/plain', '.txt')
mimetypes.add_type('text/markdown', '.md')

# --------------------- LOGGING ---------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('embedding_generation.log')
    ]
)
logger = logging.getLogger(__name__)

# --------------------- CONFIGURATION ---------------------
class EmbeddingConfig:
    """Configuration for embedding generation."""
    EMBEDDING_MODEL = "text-embedding-005"
    GEMINI_VISION_MODEL = "gemini-2.5-flash" # Using the specified multimodal model for OCR
    EMBEDDING_DIMENSIONS = 768
    TASK_TYPE = "RETRIEVAL_DOCUMENT"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    BATCH_SIZE = 50
    MAX_RETRIES = 5
    DEFAULT_BUCKET = "ashes-project-hermes-training"
    DEFAULT_FOLDER = ""
    MIN_TEXT_LENGTH = 20
    ENABLE_TITLE_EXTRACTION = True
    OCR_PROVIDER = "GEMINI"

    # HNSW index parameters
    HNSW_M = 16
    HNSW_EF_CONSTRUCTION = 64

TEXT_SPLITTER_SEPARATORS = [
    "\n\n## ", "\n# ", "\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", "-", "\t", ""
]
MARKDOWN_HEADERS_TO_SPLIT = [
    ("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")
]

# --------------------- UTILITIES ---------------------
def load_environment() -> Dict[str, str]:
    """Load environment variables from .env and set up auth."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(os.path.join(project_root, '.env'), override=True)

    credentials_path = os.path.join(project_root, 'credentials.json')
    if os.path.exists(credentials_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

    env = {
        'supabase_project_url': os.getenv('SUPABASE_PROJECT_URL'),
        'supabase_service_role_key': os.getenv('SUPABASE_SERVICE_ROLE_KEY'),
        'google_project_id': os.getenv('GOOGLE_PROJECT_ID'),
        'google_project_location': os.getenv('GOOGLE_PROJECT_LOCATION', 'us-central1'),
        'supabase_database_url': os.getenv('SUPABASE_DATABASE_URL'),
    }
    
    missing = [k for k, v in env.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    os.environ.pop('GOOGLE_API_KEY', None)
    env['supabase_url'] = env['supabase_project_url']
    logging.info("Environment configuration loaded successfully.")
    return env

@retry(
    stop=stop_after_attempt(EmbeddingConfig.MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def embed_texts_with_vertexai(
    texts: List[str], model: TextEmbeddingModel
) -> List[np.ndarray]:
    """Generate embeddings for a list of texts using the Vertex AI model."""
    try:
        all_embeddings = []
        for i in range(0, len(texts), EmbeddingConfig.BATCH_SIZE):
            batch = texts[i:i + EmbeddingConfig.BATCH_SIZE]
            inputs = [TextEmbeddingInput(text, EmbeddingConfig.TASK_TYPE) for text in batch]
            response = model.get_embeddings(inputs)
            all_embeddings.extend([np.array(emb.values, dtype=np.float32) for emb in response])
            logger.info(f"Generated embeddings for {min(i + len(batch), len(texts))}/{len(texts)} chunks.")
        return all_embeddings
    except Exception as e:
        logger.error(f"Failed to generate embeddings after multiple retries: {e}")
        raise

# --------------------- DOCUMENT PROCESSING ---------------------
def clean_text(text: str) -> str:
    """Perform pre-processing and cleaning on extracted text."""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_title(content: str) -> Optional[str]:
    """Heuristically extracts a title from the document content."""
    match = re.search(r"^\s*#\s+(.+)", content)
    if match: return match.group(1).strip()
    first_line = next((line.strip() for line in content.split('\n') if line.strip()), None)
    if first_line and len(first_line) < 150: return first_line
    return None

class EnhancedDocumentSplitter:
    """Splits documents into fine-grained chunks with context and rich metadata."""
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=TEXT_SPLITTER_SEPARATORS
        )
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT)

    def split_document(self, document: Document) -> List[Document]:
        """Splits a single document, adding chunk-specific metadata."""
        content = document.page_content
        if not content.strip(): return []
        is_markdown = bool(re.search(r'^[\s]*#\s+', content, re.MULTILINE))
        doc_type = "markdown" if is_markdown else "text"
        base_metadata = {**document.metadata, "document_type": doc_type}
        chunks = self._split_markdown(content, base_metadata) if is_markdown else self._split_text(content, base_metadata)
        for i, chunk in enumerate(chunks): chunk.metadata['chunk_index'] = i
        logger.info(f"Split document '{document.metadata.get('file_name')}' into {len(chunks)} chunks.")
        return chunks

    def _split_markdown(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        md_sections = self.markdown_splitter.split_text(content)
        return [sub_chunk for i, section in enumerate(md_sections, 1)
                for sub_chunk in self.text_splitter.create_documents(
                    [section.page_content], metadatas=[{**metadata, **section.metadata, "section_index": i}]
                )]

    def _split_text(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        return self.text_splitter.create_documents([content], metadatas=[metadata])

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_text_with_gemini(
    file_path: str, mime_type: str, config: EmbeddingConfig
) -> str:
    """Extracts text from a PDF document using a Gemini multimodal model."""
    try:
        logger.info(f"Extracting text from '{os.path.basename(file_path)}' using Gemini...")
        model = GenerativeModel(config.GEMINI_VISION_MODEL)
        
        with open(file_path, "rb") as f:
            pdf_content = f.read()
        
        document_part = Part.from_data(pdf_content, mime_type=mime_type)
        prompt = "Extract all text from this document. Preserve the original structure and formatting as much as possible."
        
        response = model.generate_content([document_part, prompt])
        
        full_text = "".join(part.text for part in response.candidates[0].content.parts)
        
        logger.info(f"Successfully extracted text from '{os.path.basename(file_path)}' using Gemini.")
        return full_text
    except Exception as e:
        logger.error(f"Gemini OCR failed for {os.path.basename(file_path)}: {e}")
        raise

def process_document(file_path: str, config: EmbeddingConfig) -> Optional[Dict[str, Any]]:
    """Reads a file, extracts content via OCR or text read, cleans it, and prepares metadata."""
    file_name = os.path.basename(file_path)
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not any(mime_type.startswith(p) for p in ['text/', 'application/pdf']):
            logger.warning(f"Skipping unsupported file type '{mime_type}' for: {file_name}")
            return None

        content = ""
        if mime_type == 'application/pdf' and config.OCR_PROVIDER == 'GEMINI':
            content = extract_text_with_gemini(file_path, mime_type, config)
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        
        cleaned_content = clean_text(content)
        if len(cleaned_content) < config.MIN_TEXT_LENGTH:
            logger.warning(f"Skipping '{file_name}' due to insufficient content after cleaning.")
            return None

        metadata = {
            "source": file_path, "file_name": file_name, "file_type": mime_type,
            "file_size": os.path.getsize(file_path),
            "content_hash": hashlib.sha256(cleaned_content.encode('utf-8')).hexdigest(),
            "load_timestamp": datetime.utcnow().isoformat(),
            "embedding_model": config.EMBEDDING_MODEL,
            "ocr_model": config.GEMINI_VISION_MODEL if mime_type == 'application/pdf' else 'N/A',
            "title": extract_title(cleaned_content) if config.ENABLE_TITLE_EXTRACTION else "N/A",
            "author": "Unknown", "document_category": "General",
        }
        return {"page_content": cleaned_content, "metadata": metadata}
    except Exception as e:
        logger.error(f"Failed to process document {file_name}: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def load_documents_from_gcs(bucket_name: str, folder_path: str, config: EmbeddingConfig) -> List[Dict]:
    from google.cloud import storage
    logger.info(f"Loading from GCS bucket '{bucket_name}', folder '{folder_path or 'root'}'...")
    storage_client = storage.Client()
    blobs = list(storage_client.bucket(bucket_name).list_blobs(prefix=folder_path or ""))
    processed_docs = []
    for blob in blobs:
        if blob.name.endswith("/"): continue
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, os.path.basename(blob.name))
            blob.download_to_filename(file_path)
            if doc := process_document(file_path, config):
                processed_docs.append(doc)
    logger.info(f"Successfully processed {len(processed_docs)} documents from GCS.")
    return processed_docs

def process_and_split_documents(docs: List[Document], splitter: EnhancedDocumentSplitter) -> List[Document]:
    return [chunk for doc in docs for chunk in splitter.split_document(doc)]

def process_batch(batch: List[Document], supabase: SupabaseClient, model: TextEmbeddingModel) -> None:
    try:
        texts = [doc.page_content for doc in batch]
        embeddings = embed_texts_with_vertexai(texts, model)
        class PrecomputedEmbedder:
            def embed_documents(self, _texts): return [e.tolist() for e in embeddings]
            def embed_query(self, _text): return [0.0] * EmbeddingConfig.EMBEDDING_DIMENSIONS
        vector_store = SupabaseVectorStore(
            client=supabase, embedding=PrecomputedEmbedder(),
            table_name="hermes_vectors", query_name="match_documents"
        )
        vector_store.add_documents(batch)
        logger.info(f"Successfully upserted batch of {len(batch)} chunks to Supabase.")
    except Exception as e:
        logger.error(f"Error processing batch: {e}", exc_info=True)
        raise

def setup_supabase_table(db_url: str, config: EmbeddingConfig) -> None:
    logger.info("Verifying Supabase database setup...")
    try:
        with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
            conn.autocommit = True
            
            # Create vector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create table
            cur.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS hermes_vectors (
                id UUID PRIMARY KEY DEFAULT extensions.uuid_generate_v4(),
                content TEXT, metadata JSONB, embedding VECTOR({})
            );
            """).format(sql.Literal(config.EMBEDDING_DIMENSIONS)))
            
            # Create HNSW index for vector similarity search
            cur.execute(sql.SQL("""
            CREATE INDEX IF NOT EXISTS hermes_vectors_hnsw_idx
            ON hermes_vectors USING hnsw (embedding vector_cosine_ops)
            WITH (m = {}, ef_construction = {});
            """).format(sql.Literal(config.HNSW_M), sql.Literal(config.HNSW_EF_CONSTRUCTION)))
            
            # Create GIN index for metadata filtering
            cur.execute("""
            CREATE INDEX IF NOT EXISTS hermes_vectors_metadata_idx
            ON hermes_vectors USING GIN (metadata);
            """)
            
            # Create or replace the match_documents RPC function
            # This function is required by LangChain's SupabaseVectorStore
            # Drop ALL existing versions of the function to avoid conflicts
            # Query for all match_documents functions and drop them
            cur.execute("""
            DO $$ 
            DECLARE 
                func_signature text;
            BEGIN
                FOR func_signature IN 
                    SELECT oid::regprocedure::text 
                    FROM pg_proc 
                    WHERE proname = 'match_documents'
                LOOP
                    EXECUTE 'DROP FUNCTION IF EXISTS ' || func_signature || ' CASCADE';
                END LOOP;
            END $$;
            """)
            
            cur.execute(sql.SQL("""
            CREATE OR REPLACE FUNCTION match_documents(
                query_embedding VECTOR({}),
                match_count INT DEFAULT 10,
                filter JSONB DEFAULT '{{}}'
            )
            RETURNS TABLE (
                id UUID,
                content TEXT,
                metadata JSONB,
                embedding VECTOR({}),
                similarity FLOAT
            )
            LANGUAGE plpgsql
            AS $$
            BEGIN
                RETURN QUERY
                SELECT
                    hermes_vectors.id,
                    hermes_vectors.content,
                    hermes_vectors.metadata,
                    hermes_vectors.embedding,
                    1 - (hermes_vectors.embedding <=> query_embedding) AS similarity
                FROM hermes_vectors
                WHERE
                    CASE
                        WHEN filter = '{{}}' THEN TRUE
                        ELSE hermes_vectors.metadata @> filter
                    END
                ORDER BY hermes_vectors.embedding <=> query_embedding
                LIMIT match_count;
            END;
            $$;
            """).format(sql.Literal(config.EMBEDDING_DIMENSIONS), sql.Literal(config.EMBEDDING_DIMENSIONS)))
            
        logger.info("Supabase table, indexes, and match_documents function are configured correctly.")
    except Exception as e:
        logger.error(f"Failed to set up Supabase table: {e}")
        raise

# --------------------- MAIN ---------------------
def main():
    start_time = datetime.now()
    logger.info("🚀 Starting Vertex AI embedding pipeline with Gemini OCR...")
    try:
        env = load_environment()
        config = EmbeddingConfig()

        logger.info(f"Initializing Vertex AI: project '{env['google_project_id']}', location '{env['google_project_location']}'...")
        vertexai.init(project=env['google_project_id'], location=env['google_project_location'])
        
        embedding_model = TextEmbeddingModel.from_pretrained(config.EMBEDDING_MODEL)
        logger.info(f"Using embedding model '{config.EMBEDDING_MODEL}' ({config.EMBEDDING_DIMENSIONS} dimensions).")
        if config.OCR_PROVIDER == "GEMINI":
            logger.info(f"Using OCR model '{config.GEMINI_VISION_MODEL}'.")

        setup_supabase_table(env["supabase_database_url"], config)
        supabase_client = create_client(env["supabase_url"], env["supabase_service_role_key"])

        documents_data = load_documents_from_gcs(
            bucket_name=config.DEFAULT_BUCKET,
            folder_path=config.DEFAULT_FOLDER,
            config=config
        )
        if not documents_data:
            logger.warning("No documents were processed from GCS. Exiting.")
            return

        documents = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in documents_data]
        splitter = EnhancedDocumentSplitter(config)
        all_chunks = process_and_split_documents(documents, splitter)

        if not all_chunks:
            logger.warning("No chunks were generated from the documents. Exiting.")
            return

        total_chunks = len(all_chunks)
        logger.info(f"Total chunks to process: {total_chunks}")
        for i in range(0, total_chunks, config.BATCH_SIZE):
            batch = all_chunks[i:i + config.BATCH_SIZE]
            process_batch(batch, supabase_client, embedding_model)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"✅ Pipeline completed successfully! "
            f"Processed {len(documents)} documents, generating {total_chunks} chunks. "
            f"Total time: {duration:.2f} seconds."
        )
    except Exception as e:
        logger.critical(f"A critical error occurred in the pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()