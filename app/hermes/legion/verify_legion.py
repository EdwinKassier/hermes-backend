import asyncio
import logging
import os
import sys

# Add app directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from app.hermes.legion.nodes.orchestration_graph import get_orchestration_graph
from app.hermes.legion.state import OrchestratorState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def verify_graph():
    try:
        logger.info("Initializing graph...")
        graph = get_orchestration_graph()
        logger.info("Graph initialized successfully.")

        # Basic check of graph structure
        logger.info(f"Graph nodes: {graph.nodes.keys()}")

        if "legion_orchestrator" in graph.nodes:
            logger.info("SUCCESS: legion_orchestrator node found.")
        else:
            logger.error("FAILURE: legion_orchestrator node NOT found.")

        if "legion_worker" in graph.nodes:
            logger.info("SUCCESS: legion_worker node found.")
        else:
            logger.error("FAILURE: legion_worker node NOT found.")

    except Exception as e:
        logger.error(f"Graph verification failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(verify_graph())
