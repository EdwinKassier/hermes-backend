#!/usr/bin/env python3
"""
Test script to verify embeddings loading and vector store functionality.
"""

import json
import os
from pathlib import Path

def test_embeddings_loading():
    """Test if embeddings can be loaded from the cache file."""
    embeddings_file = "data/embeddings_cache/embeddings.json"
    
    print(f"Testing embeddings file: {embeddings_file}")
    print(f"File exists: {os.path.exists(embeddings_file)}")
    
    if os.path.exists(embeddings_file):
        file_size = os.path.getsize(embeddings_file)
        print(f"File size: {file_size} bytes")
        
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                embeddings = json.load(f)
            
            print(f"Successfully loaded {len(embeddings)} embeddings")
            
            # Show first few keys
            keys = list(embeddings.keys())
            print(f"First 3 text chunks:")
            for i, key in enumerate(keys[:3]):
                print(f"  {i+1}. {key[:100]}...")
            
            # Show embedding dimensions
            if keys:
                first_embedding = embeddings[keys[0]]
                print(f"Embedding dimensions: {len(first_embedding)}")
            
            return True
            
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False
    else:
        print("Embeddings file not found!")
        return False

def test_gemini_service_loading():
    """Test if GeminiService can load the embeddings."""
    try:
        from app.services.GeminiService import GeminiService
        
        print("\nTesting GeminiService embeddings loading...")
        
        # Create a minimal GeminiService instance
        service = GeminiService()
        
        # Check if embeddings were loaded
        if hasattr(service, 'embeddings_dict'):
            print(f"GeminiService loaded {len(service.embeddings_dict)} embeddings")
            
            if service.embeddings_dict:
                print("✅ Embeddings loaded successfully!")
                return True
            else:
                print("❌ Embeddings dictionary is empty")
                return False
        else:
            print("❌ GeminiService has no embeddings_dict attribute")
            return False
            
    except Exception as e:
        print(f"❌ Error testing GeminiService: {e}")
        return False

def test_vector_store_retrieval():
    """Test vector store retrieval directly."""
    print("\n=== Vector Store Retrieval Test ===")
    
    try:
        from app.utils.vector_store import VectorStore
        
        # Initialize vector store
        vector_store = VectorStore()
        print("✅ Vector store initialized successfully")
        
        # Test query
        query = "What is Edwin's current employment?"
        print(f"Query: {query}")
        
        # Get top results
        results = vector_store.search(query, top_k=5)
        print(f"\nTop {len(results)} results:")
        
        for i, (text, score) in enumerate(results, 1):
            # Truncate long text for display
            display_text = text[:200] + "..." if len(text) > 200 else text
            print(f"{i}. Score: {score:.3f}")
            print(f"   Text: {display_text}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Embeddings Test ===")
    
    # Test 1: Direct file loading
    file_test = test_embeddings_loading()
    
    # Test 2: GeminiService loading
    service_test = test_gemini_service_loading()
    
    # Add vector store test
    vector_test = test_vector_store_retrieval()
    
    print("\n=== Summary ===")
    print(f"File loading test: {'✅ PASSED' if file_test else '❌ FAILED'}")
    print(f"Service loading test: {'✅ PASSED' if service_test else '❌ FAILED'}")
    print(f"Vector store test: {'✅ PASSED' if vector_test else '❌ FAILED'}")
    
    if not all([file_test, service_test, vector_test]):
        print("\n⚠️  Some tests failed. Check the output above for issues.")
    else:
        print("\n🎉 All tests passed!") 