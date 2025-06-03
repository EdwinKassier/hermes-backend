import os
import traceback
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

# Google / Gemini / Langchain specific
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)
from langchain_google_vertexai import (
    VertexAIEmbeddings,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import tools and services
from app.utils.vector_store import VectorStoreService
from app.utils.toolhub import get_all_tools
from app.utils.conversation_state import State, ConversationState

# GCS
from google.cloud import storage

import logging

# Setup basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GeminiService:
    MODEL_NAME = "gemini-2.0-flash"  # LLM for generation
    TEMPERATURE = 0.8
    TIMEOUT = 60
    MAX_RETRIES = 3
    ERROR_MESSAGE = "Sorry, the Hermes LLM service encountered an error."

    # Changed to a Vertex AI embedding model
    EMBEDDING_MODEL_NAME = "text-embedding-004"
    TEXT_SPLITTER_CHUNK_SIZE = 1500
    TEXT_SPLITTER_CHUNK_OVERLAP = 250
    RAG_TOP_K = 3

    def __init__(
        self,
        initial_gcs_bucket_name: Optional[str] = "ashes-project-hermes-training",
        initial_gcs_folder_path: Optional[str] = None,
        gcs_credentials_path: Optional[str] = "credentials.json",
        vertex_project: Optional[str] = os.environ["GOOGLE_PROJECT_ID"],
        vertex_location: Optional[str] = os.environ["GOOGLE_PROJECT_LOCATION"],
        conversation_db_path: str = "conversations.db"
    ):
        # For Gemini LLM (ChatGoogleGenerativeAI)
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set for Gemini LLM."
            )

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"

        # Initialize base model
        base_model = ChatGoogleGenerativeAI(
            model=self.MODEL_NAME,
            temperature=self.TEMPERATURE,
            max_retries=self.MAX_RETRIES,
            timeout=self.TIMEOUT,
        )

        # Get all available tools and bind them to the model
        tools = get_all_tools()
        self.model = base_model.bind_tools(tools)
        logging.info(f"Initialized Gemini model with {len(tools)} tools attached")

        self.base_prompt = os.environ.get("BASE_PROMPT", "")

        # Initialize GCS Client
        try:
            if gcs_credentials_path and os.path.exists(gcs_credentials_path):
                self.storage_client = storage.Client.from_service_account_json(
                    gcs_credentials_path
                )
                logging.info(
                    f"Initialized GCS client using credentials from: {gcs_credentials_path}"
                )
            else:
                if gcs_credentials_path:
                    logging.warning(
                        f"GCS credentials file not found at '{gcs_credentials_path}'. "
                        "Using Application Default Credentials."
                    )
                else:
                    logging.info(
                        "No GCS credentials file path provided. "
                        "Using Application Default Credentials."
                    )
                self.storage_client = storage.Client()
                logging.info(
                    "Initialized GCS client using Application Default Credentials."
                )
        except Exception as e:
            logging.error(
                f"Failed to initialize GCS client: {e}. "
                "Ensure GCS credentials are correctly set up."
            )
            raise

        # Initialize VertexAIEmbeddings
        try:
            self.embeddings_model = VertexAIEmbeddings(
                model_name=self.EMBEDDING_MODEL_NAME,
                project=vertex_project,
                location=vertex_location,
            )
            logging.info(
                f"Initialized VertexAIEmbeddings with model: {self.EMBEDDING_MODEL_NAME}"
            )
        except Exception as e:
            logging.error(f"Failed to initialize VertexAIEmbeddings: {e}")
            logging.error(traceback.format_exc())
            raise

        # Initialize VectorStoreService
        self.vector_store_service = VectorStoreService(
            storage_client=self.storage_client,
            embeddings_model=self.embeddings_model,
            chunk_size=self.TEXT_SPLITTER_CHUNK_SIZE,
            chunk_overlap=self.TEXT_SPLITTER_CHUNK_OVERLAP,
            rag_top_k=self.RAG_TOP_K,
        )

        if initial_gcs_bucket_name:
            self.vector_store_service.update_rag_sources(
                initial_gcs_bucket_name, initial_gcs_folder_path
            )

        # Initialize conversation state management
        self.conversation_state = ConversationState(db_path=conversation_db_path)
        
        logging.info(
            f"GeminiService initialized. LLM: {self.MODEL_NAME}"
        )

    def update_rag_sources(
        self, gcs_bucket_name: str, gcs_folder_path: Optional[str] = None
    ):
        self.vector_store_service.update_rag_sources(gcs_bucket_name, gcs_folder_path)

    def clear_rag_sources(self):
        self.vector_store_service.clear_rag_sources()

    def generate_gemini_response(self, prompt: str) -> str:
        prompt_prefix = f"{self.base_prompt}\n\n" if self.base_prompt else ""
        full_prompt = f"{prompt_prefix}User Input: {prompt}"
        if not self.base_prompt:
            full_prompt = prompt
        try:
            message = self.model.invoke([HumanMessage(content=full_prompt)])
            response = message.content.strip()
            if "error" in response.lower() or not response:
                logging.warning(
                    f"Gemini returned error or empty message: '{response}'"
                )
                return (
                    self.ERROR_MESSAGE +
                    " Check safety filters, rate limiting, or prompt."
                )
            logging.info("Generated response successfully")
            return response
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            logging.error(traceback.format_exc())
            return (
                self.ERROR_MESSAGE +
                " Rate limiting or internal error. Try again in 30s."
            )

    def _format_conversation_history(self, messages: List[Dict[str, Any]]) -> str:
        """Format conversation history into a readable string."""
        formatted = []
        for msg in messages:
            role = msg.get('role', 'unknown').capitalize()
            content = msg.get('content', '').strip()
            if content:
                formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def generate_gemini_response_with_rag(
        self, 
        prompt: str, 
        user_id: str,
        **kwargs
    ) -> str:
        """Generate a response using RAG with conversation history.
        
        Args:
            prompt: The user's input prompt
            user_id: Unique identifier for the user/session
            **kwargs: Additional keyword arguments
            
        Returns:
            str: The generated response
        """
        # Get or initialize conversation state
        state = self.conversation_state.get_state(user_id)
        if not state or 'conversation' not in state.data:
            state = State(data={
                'conversation': [],
                'metadata': {
                    'created_at': datetime.utcnow().isoformat(),
                    'last_updated': datetime.utcnow().isoformat(),
                    'message_count': 0
                }
            })

        # Add user message to conversation
        state.data['conversation'].append({
            'role': 'user',
            'content': prompt,
            'timestamp': datetime.utcnow().isoformat()
        })
        state.data['metadata']['last_updated'] = datetime.utcnow().isoformat()
        state.data['metadata']['message_count'] += 1

        # Get relevant context from vector store
        relevant_chunks = self.vector_store_service.get_relevant_chunks(prompt)
        if not relevant_chunks:
            logging.info("No relevant chunks found. Using standard generation.")
            return self.generate_gemini_response(prompt)

        # Format conversation history for context
        conversation_history = self._format_conversation_history(
            state.data['conversation'][-10:]  # Last 10 messages for context
        )
        
        # Prepare context
        prompt_prefix = f"{self.base_prompt}\n\n" if self.base_prompt else ""
        context_str = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        logging.info(f"Using {len(relevant_chunks)} chunks for context")

        # Construct full prompt with conversation history
        full_prompt = (
            f"{prompt_prefix}"
            f"Current date and time: {datetime.utcnow().isoformat()}\n\n"
            f"=== CONVERSATION HISTORY ===\n"
            f"{conversation_history}\n\n"
            f"=== RELEVANT CONTEXT ===\n"
            f"{context_str}\n\n"
            f"=== USER'S LATEST MESSAGE ===\n"
            f"User: {prompt}\n\n"
            f"Please provide a helpful response based on the above conversation and context."
        )

        try:
            # Generate response
            message = self.model.invoke([HumanMessage(content=full_prompt)])
            response = message.content.strip()
            
            if "error" in response.lower() or not response:
                logging.warning(f"RAG response error or empty: '{response}'")
                return (
                    self.ERROR_MESSAGE +
                    " RAG: Check safety filters or rate limiting."
                )
                
            # Add assistant's response to conversation history
            state.data['conversation'].append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.utcnow().isoformat()
            })
            state.data['metadata']['last_updated'] = datetime.utcnow().isoformat()
            
            # Save updated conversation state
            self.conversation_state.save_state(user_id, state)
            
            logging.info("Generated RAG response successfully")
            return response
        except Exception as e:
            logging.error(f"Error generating RAG response: {e}")
            logging.error(traceback.format_exc())
            return self.ERROR_MESSAGE + " RAG: Generation error."
