import os
import sys

# Add project root to python path
sys.path.append(os.getcwd())

try:
    from app.hermes.legion.graph_service import LegionGraphService

    print("Successfully imported LegionGraphService")
except Exception as e:
    print(f"Failed to import LegionGraphService: {e}")
    import traceback

    traceback.print_exc()
