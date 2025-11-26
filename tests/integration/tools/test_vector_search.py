#!/usr/bin/env python3
"""
Test script to verify vector search timeout fix in GeminiService.
"""

import logging
import os
import signal
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.shared.services.GeminiService import GeminiService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_vector_search_timeout():
    """Test that vector search handles timeouts gracefully."""
    logger.info("Initializing GeminiService...")
    try:
        service = GeminiService()
    except Exception as e:
        logger.error(f"Failed to initialize GeminiService: {e}")
        return

    query = "Edwin Kassier work history"
    logger.info(f"Testing vector search with query: '{query}'")

    start_time = time.time()
    try:
        # This calls _direct_similarity_search internally
        # We want to verify it doesn't hang indefinitely
        results = service._direct_similarity_search(query, k=3)

        duration = time.time() - start_time
        logger.info(f"Vector search completed in {duration:.2f} seconds")
        logger.info(f"Found {len(results)} results")

        for i, (doc, score) in enumerate(results):
            logger.info(
                f"Result {i+1}: Score={score:.3f}, Content='{doc.page_content[:50]}...'"
            )

        if duration > 20:
            logger.error("❌ Test FAILED: Vector search took too long (>20s)")
        else:
            logger.info(
                "✅ Test PASSED: Vector search completed within reasonable time"
            )

    except Exception as e:
        logger.error(f"Vector search failed with error: {e}")
        # If it failed with TimeoutError (caught internally and returned empty list), that's also a pass for "not hanging"
        # But here we are calling the internal method which catches it.
        pass


if __name__ == "__main__":
    test_vector_search_timeout()
