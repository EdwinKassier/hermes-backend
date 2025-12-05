#!/usr/bin/env python3
"""
Integration test for the full LangGraph orchestration flow.
Simulates a user request that requires agent creation and tool execution.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)

from app.hermes.legion.graph_service import LegionGraphService  # noqa: E402
from app.hermes.models import UserIdentity  # noqa: E402

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_full_orchestration():
    """Test the full orchestration flow with a complex query."""
    logger.info("Initializing LegionGraphService...")
    service = LegionGraphService()

    # Test query that should trigger:
    # 1. Orchestrator -> Identification of task
    # 2. Agent Creation (Research Agent)
    # 3. Tool Allocation (Web Search)
    # 4. Execution
    # 5. Final Response
    query = "What is the current price of Bitcoin and Ethereum? Compare them."

    # Create proper UserIdentity object
    user_identity = UserIdentity(
        user_id="test_user_integration",
        ip_address="127.0.0.1",
        user_agent="IntegrationTest/1.0",
        accept_language="en-US",
    )

    logger.info(f"Processing query: '{query}'")

    try:
        # Process request (not async)
        response = service.process_request(
            text=query,
            user_identity=user_identity,
            response_mode="text",  # Valid values: 'text' or 'tts'
            persona="hermes",
        )

        logger.info("Response received!")
        logger.info("-" * 50)
        logger.info(response)
        logger.info("-" * 50)

        # Basic validation
        response_text = (
            response.message if hasattr(response, "message") else str(response)
        )
        if (
            "Bitcoin" in response_text
            or "Ethereum" in response_text
            or "crypto" in response_text.lower()
        ):
            logger.info("✅ Test PASSED: Response contains expected keywords")
        else:
            logger.warning("⚠️ Test WARNING: Response might be missing expected content")
            logger.info(f"Response was: {response_text[:200]}...")

    except Exception as e:
        logger.error(f"Orchestration failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    # Run test (not async)
    test_full_orchestration()
