-- Fix ambiguous column reference in match_documents function
-- Run this in your Supabase SQL Editor

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
        hv.id,
        hv.content,
        hv.metadata,
        1 - (hv.embedding <=> query_embedding) AS similarity
    FROM hermes_vectors hv
    WHERE hv.metadata @> filter
    ORDER BY hv.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

