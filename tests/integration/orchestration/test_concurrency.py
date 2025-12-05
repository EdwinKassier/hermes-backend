#!/usr/bin/env python3
"""
Integration test for concurrent user sessions.
Verifies that state is isolated between different users.
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


def test_concurrency():
    """Test concurrent user sessions with state isolation."""
    logger.info("Initializing LegionGraphService...")
    service = LegionGraphService()

    # Create two distinct users
    user_a = UserIdentity(
        user_id="user_alice", ip_address="127.0.0.1", user_agent="Test/1.0"
    )

    user_b = UserIdentity(
        user_id="user_bob", ip_address="127.0.0.1", user_agent="Test/1.0"
    )

    logger.info("Step 1: Setting context for User A (Alice)")
    service.process_request(
        text="My name is Alice. Remember this.",
        user_identity=user_a,
        response_mode="text",
        persona="hermes",
    )

    logger.info("Step 2: Setting context for User B (Bob)")
    service.process_request(
        text="My name is Bob. Remember this.",
        user_identity=user_b,
        response_mode="text",
        persona="hermes",
    )

    logger.info("Step 3: Verifying User A context")
    response_a = service.process_request(
        text="What is my name?",
        user_identity=user_a,
        response_mode="text",
        persona="hermes",
    )
    response_text_a = (
        response_a.message if hasattr(response_a, "message") else str(response_a)
    )
    logger.info(f"User A Response: {response_text_a}")

    logger.info("Step 4: Verifying User B context")
    response_b = service.process_request(
        text="What is my name?",
        user_identity=user_b,
        response_mode="text",
        persona="hermes",
    )
    response_text_b = (
        response_b.message if hasattr(response_b, "message") else str(response_b)
    )
    logger.info(f"User B Response: {response_text_b}")

    # Validation
    passed = True

    if "Alice" in response_text_a and "Bob" not in response_text_a:
        logger.info("✅ User A context correct")
    else:
        logger.error("❌ User A context incorrect")
        passed = False

    if "Bob" in response_text_b and "Alice" not in response_text_b:
        logger.info("✅ User B context correct")
    else:
        logger.error("❌ User B context incorrect")
        passed = False

    if passed:
        logger.info("✅ Concurrency Test PASSED: State is isolated")
    else:
        logger.error("❌ Concurrency Test FAILED")
        raise Exception("Concurrency test failed")


if __name__ == "__main__":
    test_concurrency()
