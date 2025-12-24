import logging
import os

from langchain.chat_models import init_chat_model

# Setup logging
logging.basicConfig(level=logging.INFO)

# Ensure only API key is set, no project
if "GOOGLE_CLOUD_PROJECT" in os.environ:
    del os.environ["GOOGLE_CLOUD_PROJECT"]
if "GCP_PROJECT" in os.environ:
    del os.environ["GCP_PROJECT"]

# Assuming GOOGLE_API_KEY is present or provided
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("WARNING: GOOGLE_API_KEY not found in env")

print(f"Testing init_chat_model with 'gemini-2.5-flash'")

try:
    # 1. Default behavior
    print("\n--- Attempt 1: Default (inferred provider) ---")
    model = init_chat_model("gemini-2.5-flash")
    print(f"Initialized model type: {type(model)}")
    print(f"Provider: {model._llm_type if hasattr(model, '_llm_type') else 'unknown'}")

    # Try to invoke to trigger auth check
    # model.invoke("Hello")  # Might fail if no auth
except Exception as e:
    print(f"Attempt 1 Failed: {e}")

try:
    # 2. Explicit google_genai
    print("\n--- Attempt 2: Explicit model_provider='google_genai' ---")
    model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    print(f"Initialized model type: {type(model)}")
    print(f"Provider: {model._llm_type if hasattr(model, '_llm_type') else 'unknown'}")
except Exception as e:
    print(f"Attempt 2 Failed: {e}")
