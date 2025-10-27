"""
Performance tests for vector search operations.

Run with: pytest tests/performance/ --run-integration --run-slow -v
"""
import pytest
import time
import statistics
import os
from app.shared.services.GeminiService import GeminiService


@pytest.fixture(scope="module")
def gemini_service():
    """Create GeminiService for performance testing"""
    if not all([os.environ.get('GOOGLE_API_KEY'), 
                os.environ.get('SUPABASE_URL'),
                os.environ.get('SUPABASE_SERVICE_ROLE_KEY')]):
        pytest.skip("Missing required environment variables for integration tests")
    
    service = GeminiService()
    return service


@pytest.mark.slow
@pytest.mark.integration
class TestVectorSearchPerformance:
    """Test vector search performance"""
    
    def test_search_latency(self, gemini_service):
        """Test that vector search completes within acceptable time"""
        query = "Edwin Kassier software engineer"
        
        # Warm up (3 runs to stabilize)
        for _ in range(3):
            gemini_service._direct_similarity_search(query, k=5)
        
        time.sleep(0.5)  # Brief pause after warm-up
        
        # Measure latency over 10 runs
        latencies = []
        for i in range(10):
            start = time.time()
            results = gemini_service._direct_similarity_search(query, k=5, threshold=0.6)
            latency = time.time() - start
            latencies.append(latency)
            
            assert len(results) > 0, f"Run {i+1}: No results returned"
        
        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p50_latency = statistics.median(latencies)
        
        print(f"\nVector Search Performance:")
        print(f"  Average: {avg_latency:.3f}s")
        print(f"  Median:  {p50_latency:.3f}s")
        print(f"  P95:     {p95_latency:.3f}s")
        print(f"  Min:     {min(latencies):.3f}s")
        print(f"  Max:     {max(latencies):.3f}s")
        
        # Performance assertions
        assert avg_latency < 2.0, f"Average latency {avg_latency:.2f}s exceeds 2s threshold"
        assert p95_latency < 3.0, f"P95 latency {p95_latency:.2f}s exceeds 3s threshold"
    
    def test_rag_response_latency(self, gemini_service):
        """Test full RAG pipeline latency"""
        query = "What is Edwin Kassier's background?"
        
        # Warm up
        gemini_service.generate_gemini_response_with_rag(
            prompt=query,
            user_id="warmup",
            persona='hermes'
        )
        
        time.sleep(1)
        
        # Measure latency over 5 runs (fewer due to API cost)
        latencies = []
        for i in range(5):
            start = time.time()
            response = gemini_service.generate_gemini_response_with_rag(
                prompt=query,
                user_id=f"perf_test_{i}",
                persona='hermes'
            )
            latency = time.time() - start
            latencies.append(latency)
            
            assert len(response) > 0, f"Run {i+1}: Empty response"
        
        avg_latency = statistics.mean(latencies)
        
        print(f"\nRAG Pipeline Performance:")
        print(f"  Average: {avg_latency:.3f}s")
        print(f"  Min:     {min(latencies):.3f}s")
        print(f"  Max:     {max(latencies):.3f}s")
        
        # Full RAG should complete within 10 seconds on average
        assert avg_latency < 10.0, f"Average RAG latency {avg_latency:.2f}s exceeds 10s threshold"
    
    def test_embedding_generation_throughput(self, gemini_service):
        """Test embedding generation throughput"""
        queries = [
            "Edwin Kassier software engineer",
            "machine learning artificial intelligence",
            "web development full stack",
            "data science analytics",
            "cloud computing infrastructure"
        ]
        
        start = time.time()
        embeddings = [gemini_service.embeddings_model.embed_query(q) for q in queries]
        total_time = time.time() - start
        
        throughput = len(queries) / total_time
        
        print(f"\nEmbedding Generation Throughput:")
        print(f"  {len(queries)} embeddings in {total_time:.3f}s")
        print(f"  {throughput:.2f} embeddings/sec")
        
        # Should generate at least 1 embedding per 2 seconds
        assert throughput > 0.5, f"Throughput {throughput:.2f} emb/s is too low"


@pytest.mark.slow
@pytest.mark.integration
class TestConcurrentPerformance:
    """Test performance under concurrent load"""
    
    def test_concurrent_searches(self, gemini_service):
        """Test multiple concurrent searches"""
        import concurrent.futures
        
        query = "Edwin Kassier"
        num_concurrent = 5
        
        def search():
            start = time.time()
            results = gemini_service._direct_similarity_search(query, k=5, threshold=0.6)
            return time.time() - start, len(results)
        
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(search) for _ in range(num_concurrent)]
            results = [f.result() for f in futures]
        total_time = time.time() - start
        
        latencies = [latency for latency, _ in results]
        avg_latency = statistics.mean(latencies)
        
        print(f"\nConcurrent Search Performance ({num_concurrent} concurrent):")
        print(f"  Total time:    {total_time:.3f}s")
        print(f"  Avg latency:   {avg_latency:.3f}s")
        print(f"  Max latency:   {max(latencies):.3f}s")
        
        # Under concurrent load, should still complete reasonably
        assert avg_latency < 5.0, f"Concurrent avg latency {avg_latency:.2f}s too high"


@pytest.mark.slow  
@pytest.mark.integration
class TestScalability:
    """Test scalability with varying parameters"""
    
    @pytest.mark.parametrize("k", [1, 5, 10, 20])
    def test_search_with_varying_k(self, gemini_service, k):
        """Test search performance with different k values"""
        query = "Edwin Kassier"
        
        start = time.time()
        results = gemini_service._direct_similarity_search(query, k=k, threshold=0.5)
        latency = time.time() - start
        
        print(f"\nSearch with k={k}: {latency:.3f}s, {len(results)} results")
        
        # Latency shouldn't increase dramatically with k
        assert latency < 3.0, f"Search with k={k} took {latency:.2f}s"

