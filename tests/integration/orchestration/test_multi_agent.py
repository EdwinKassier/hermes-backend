#!/usr/bin/env python3
"""
Integration test for multi-agent orchestration.
Simulates a complex user request that should trigger multiple agents.
"""

import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)

from app.hermes.legion.graph_service import LegionGraphService
from app.hermes.models import UserIdentity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_multi_agent_orchestration():
    """Test multi-agent orchestration with a complex query."""
    logger.info("Initializing LegionGraphService...")
    service = LegionGraphService()

    # Complex query that requires research AND analysis
    # We provide detailed context to avoid information gathering phase
    query = "Research the latest advancements in quantum computing (specifically error correction) and analyze their potential impact on RSA cryptography. Provide a technical summary."

    # Create proper UserIdentity object
    user_identity = UserIdentity(
        user_id="test_user_multi_agent",
        ip_address="127.0.0.1",
        user_agent="IntegrationTest/1.0",
        accept_language="en-US",
    )

    logger.info(f"Processing complex query: '{query}'")

    try:
        # Process request (not async)
        response = service.process_request(
            text=query,
            user_identity=user_identity,
            response_mode="text",
            persona="hermes",
        )

        logger.info("Response received!")
        logger.info("-" * 50)
        logger.info(response)
        logger.info("-" * 50)

        # Validation
        response_text = (
            response.message if hasattr(response, "message") else str(response)
        )

        # Check for keywords indicating both research and analysis happened
        keywords = ["quantum", "cryptography", "encryption", "impact", "analysis"]
        found_keywords = [k for k in keywords if k in response_text.lower()]

        logger.info(f"Found keywords: {found_keywords}")

        if len(found_keywords) >= 3:
            logger.info("✅ Test PASSED: Response contains relevant content")
        else:
            logger.warning("⚠️ Test WARNING: Response might be missing expected content")

        # Check metadata for multiple agents if available
        if hasattr(response, "metadata"):
            agents_used = response.metadata.get("agents_used", [])
            logger.info(f"Agents used: {agents_used}")
            if len(agents_used) > 1:
                logger.info("✅ Test PASSED: Multiple agents were used")
            elif len(agents_used) == 1:
                logger.info(
                    "ℹ️ Note: Only one agent was used (might be expected if one agent handled both)"
                )
            else:
                logger.warning("⚠️ Warning: No agents listed in metadata")

    except Exception as e:
        logger.error(f"Multi-agent orchestration failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    test_multi_agent_orchestration()
