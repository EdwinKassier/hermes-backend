-- ============================================================================
-- SUPABASE VECTOR STORE SCHEMA
-- For gemini-embedding-001 with 768 dimensions (high quality + fast)
-- ============================================================================
-- 
-- Run this SQL in your Supabase SQL Editor to set up the vector store
-- 
-- Prerequisites:
--   - Supabase project created
--   - Run in Supabase SQL Editor
-- 
-- Note: Supabase pgvector indexes support max 2000 dimensions
--       Using 768 dimensions provides excellent quality with fast indexing
-- 
-- ============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create hermes_vectors table with 768 dimensions
CREATE TABLE IF NOT EXISTS hermes_vectors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(768),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create HNSW index for fast similarity search
-- HNSW provides best performance for vectors under 2000 dimensions
CREATE INDEX IF NOT EXISTS idx_hermes_vectors_embedding 
ON hermes_vectors 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Create match_documents function for similarity search
-- This function is called by LangChain's SupabaseVectorStore
CREATE OR REPLACE FUNCTION match_documents (
    query_embedding VECTOR(768),
    match_count INT DEFAULT 20,
    filter JSONB DEFAULT '{}'::jsonb
) RETURNS TABLE (
    id UUID,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
) LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        hermes_vectors.id,
        hermes_vectors.content,
        hermes_vectors.metadata,
        1 - (hermes_vectors.embedding <=> query_embedding) AS similarity
    FROM hermes_vectors
    WHERE metadata @> filter
    ORDER BY hermes_vectors.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Verify setup
SELECT 
    'Setup complete! Table created: hermes_vectors' AS status,
    COUNT(*) AS document_count
FROM hermes_vectors;

-- Show indexes
SELECT 
    schemaname,
    tablename,
    indexname
FROM pg_indexes
WHERE tablename = 'hermes_vectors';

