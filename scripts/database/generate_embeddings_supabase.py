#!/usr/bin/env python3
"""
Simplified script to generate embeddings and upload to Supabase vector store.

This script uses LangChain's standard integration with no custom logic:
1. Loads documents from Google Cloud Storage
2. Uses standard LangChain SupabaseVectorStore
3. Uses gemini-embedding-001 (768 dimensions - high quality + Supabase compatible)

Prerequisites:
- Run scripts/supabase_schema.sql in your Supabase SQL Editor
- Set environment variables in .env:
  - SUPABASE_DATABASE_URL
  - SUPABASE_SERVICE_ROLE_KEY
  - GOOGLE_PROJECT_ID
  - GOOGLE_PROJECT_LOCATION
- Install dependencies: pip install -r requirements.txt

Note: Supabase pgvector indexes support max 2000 dimensions.
      768 dimensions provides excellent quality with fast performance.
"""

import os
import logging
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Standard LangChain imports
from langchain_community.document_loaders import GCSDirectoryLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
EMBEDDING_MODEL = "text-embedding-004"  # 768 dimensions by default
EMBEDDING_DIMENSIONS = 768  # Supabase supports max 2000, 768 is optimal
CHUNK_SIZE = 300
CHUNK_OVERLAP = 200

# GCS Configuration
DEFAULT_BUCKET = "ashes-project-hermes-training"
DEFAULT_FOLDER = None  # Set to None to load entire bucket

def load_environment():
    """Load and validate environment variables."""
    load_dotenv()
    
    required_vars = [
        'SUPABASE_DATABASE_URL',
        'SUPABASE_SERVICE_ROLE_KEY',
        'GOOGLE_PROJECT_ID',
        'GOOGLE_PROJECT_LOCATION'
    ]
    
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please add them to your .env file."
        )
    
    return {
        'supabase_url': os.environ['SUPABASE_DATABASE_URL'],
        'supabase_key': os.environ['SUPABASE_SERVICE_ROLE_KEY'],
        'project_id': os.environ['GOOGLE_PROJECT_ID'],
        'location': os.environ['GOOGLE_PROJECT_LOCATION']
    }

def load_documents_from_gcs(bucket_name: str, folder_path: str = None) -> List:
    """Load documents from Google Cloud Storage."""
    logging.info(f"Loading documents from GCS bucket: {bucket_name}")
    if folder_path:
        logging.info(f"Folder path: {folder_path}")
    
    try:
        loader = GCSDirectoryLoader(
            project_name=os.environ['GOOGLE_PROJECT_ID'],
            bucket=bucket_name,
            prefix=folder_path or ""
        )
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} documents from GCS")
        return documents
    except Exception as e:
        logging.error(f"Error loading documents from GCS: {e}")
        raise

def split_documents(documents: List) -> List:
    """Split documents into chunks using LangChain."""
    logging.info(f"Splitting documents (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Created {len(chunks)} chunks")
    return chunks

def main():
    """Main execution function."""
    try:
        # Load environment
        logging.info("Loading environment variables...")
        env = load_environment()
        
        # Initialize embeddings model (text-embedding-004)
        # text-embedding-004 defaults to 768 dimensions which is perfect for Supabase
        logging.info(f"Initializing embeddings model: {EMBEDDING_MODEL}")
        embeddings = VertexAIEmbeddings(
            model_name=EMBEDDING_MODEL,
            project=env['project_id'],
            location=env['location']
            # text-embedding-004 defaults to 768 dimensions
        )
        logging.info(f"Embeddings model initialized ({EMBEDDING_DIMENSIONS} dimensions)")
        
        # Load documents from GCS
        documents = load_documents_from_gcs(DEFAULT_BUCKET, DEFAULT_FOLDER)
        
        if not documents:
            logging.warning("No documents found. Exiting.")
            return
        
        # Split documents into chunks
        chunks = split_documents(documents)
        
        # Initialize Supabase client
        logging.info("Connecting to Supabase...")
        supabase_client = create_client(env['supabase_url'], env['supabase_key'])
        
        # Upload to Supabase using LangChain's standard method
        # Process in batches due to Vertex AI limit of 250 documents per batch
        logging.info(f"Uploading {len(chunks)} chunks to Supabase vector store...")
        logging.info("This may take a few minutes...")
        
        BATCH_SIZE = 200  # Stay well under the 250 limit for safety
        total_chunks = len(chunks)
        
        # Create the vector store with the first batch
        first_batch = chunks[:BATCH_SIZE]
        logging.info(f"Processing batch 1/{((total_chunks - 1) // BATCH_SIZE) + 1} ({len(first_batch)} chunks)...")
        
        vector_store = SupabaseVectorStore.from_documents(
            documents=first_batch,
            embedding=embeddings,
            client=supabase_client,
            table_name="hermes_vectors",
            query_name="match_documents"
        )
        
        # Add remaining batches
        for i in range(BATCH_SIZE, total_chunks, BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = ((total_chunks - 1) // BATCH_SIZE) + 1
            logging.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            vector_store.add_documents(batch)
        
        logging.info("✅ Successfully uploaded all embeddings to Supabase!")
        logging.info(f"Total chunks: {len(chunks)}")
        logging.info(f"Embedding dimensions: {EMBEDDING_DIMENSIONS}")
        logging.info(f"Model: {EMBEDDING_MODEL}")
        logging.info("\nVector store is ready for use!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

