import os
import traceback
from typing import Optional, List, Dict, Any
from datetime import datetime

# Google / Gemini / Langchain specific
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)
from langchain_google_vertexai import (
    VertexAIEmbeddings,
)
from langchain_core.messages import HumanMessage

# Vector Store - Standard LangChain integration
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

# Import tools and services
from app.shared.utils.toolhub import get_all_tools
from app.shared.utils.conversation_state import State, ConversationState

import logging

# Setup basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GeminiService:
    MODEL_NAME = "gemini-2.5-flash"  # LLM for generation
    TEMPERATURE = 1.0
    TIMEOUT = 60
    MAX_RETRIES = 3
    ERROR_MESSAGE = "Sorry, the Hermes LLM service encountered an error."

    # Vertex AI embedding model (Supabase compatible dimensions)
    EMBEDDING_MODEL_NAME = "text-embedding-004"  # 768 dimensions by default
    EMBEDDING_DIMENSIONS = 768  # Supabase pgvector index limit is 2000
    # Ultra-fine embedding parameters for maximum accuracy
    TEXT_SPLITTER_CHUNK_SIZE = 300  # Ultra-fine chunks
    TEXT_SPLITTER_CHUNK_OVERLAP = 200  # High overlap preserves context
    RAG_TOP_K = 20  # Retrieve many chunks for comprehensive coverage

    def __init__(
        self,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        conversation_db_path: str = "conversations.db"
    ):
        # Set default values for vertex project/location if not provided
        if vertex_project is None:
            vertex_project = os.environ.get("GOOGLE_PROJECT_ID")
        if vertex_location is None:
            vertex_location = os.environ.get("GOOGLE_PROJECT_LOCATION")

        # For Gemini LLM (ChatGoogleGenerativeAI)
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set."
            )

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
        logging.info(
            "Initialized Gemini model with %d tools attached", len(tools)
        )

        # Store multiple base prompts for different personas
        self.base_prompts = {
            'hermes': os.environ.get("BASE_PROMPT", ""),  # Default/backward compatible
            'prisma': os.environ.get("PRISMA_BASE_PROMPT", ""),
            'prism': os.environ.get("PRISM_BASE_PROMPT", "")  # Prism voice agent persona
        }

        # Initialize VertexAIEmbeddings with text-embedding-004
        # text-embedding-004 defaults to 768 dimensions which is perfect for Supabase
        try:
            self.embeddings_model = VertexAIEmbeddings(
                model_name=self.EMBEDDING_MODEL_NAME,
                project=vertex_project,
                location=vertex_location
                # Default 768 dimensions - perfect for Supabase pgvector
            )
            logging.info(
                "Initialized VertexAIEmbeddings with model: %s "
                "(%d dimensions - Supabase compatible)",
                self.EMBEDDING_MODEL_NAME,
                self.EMBEDDING_DIMENSIONS
            )
        except Exception as e:
            logging.error(
                "Failed to initialize VertexAIEmbeddings: %s", e
            )
            logging.error(traceback.format_exc())
            raise

        # Initialize Supabase Vector Store (LangChain Standard)
        try:
            supabase_url = os.environ.get("SUPABASE_DATABASE_URL")
            supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

            if not supabase_url or not supabase_key:
                raise ValueError(
                    "SUPABASE_DATABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
                    "environment variables must be set. "
                    "Please add them to your .env file."
                )

            # Create Supabase client
            supabase_client = create_client(supabase_url, supabase_key)

            # Initialize LangChain's standard SupabaseVectorStore
            self.vector_store = SupabaseVectorStore(
                client=supabase_client,
                embedding=self.embeddings_model,
                table_name="hermes_vectors",
                query_name="match_documents"
            )

            logging.info(
                "Initialized Supabase vector store "
                "using LangChain (standard integration)"
            )

        except Exception as e:
            logging.error(
                "Failed to initialize Supabase vector store: %s", e
            )
            logging.error(traceback.format_exc())
            raise

        # Initialize conversation state management
        self.conversation_state = ConversationState(
            db_path=conversation_db_path
        )

        logging.info(
            "GeminiService initialized. "
            "LLM: %s, Embeddings: %s (%dD), Vector Store: Supabase",
            self.MODEL_NAME,
            self.EMBEDDING_MODEL_NAME,
            self.EMBEDDING_DIMENSIONS
        )

    def generate_gemini_response(self, prompt: str, persona: str = 'hermes') -> str:
        # Select base prompt based on persona
        base_prompt = self.base_prompts.get(persona, self.base_prompts['hermes'])
        
        prompt_prefix = f"{base_prompt}\n\n" if base_prompt else ""
        full_prompt = f"{prompt_prefix}User Input: {prompt}"
        if not base_prompt:
            full_prompt = prompt
        try:
            message = self.model.invoke([HumanMessage(content=full_prompt)])
            response = message.content.strip()
            if "error" in response.lower() or not response:
                logging.warning(
                    "Gemini returned error or empty message: '%s'",
                    response
                )
                return (
                    self.ERROR_MESSAGE +
                    " Check safety filters, rate limiting, or prompt."
                )
            logging.info("Generated response successfully with persona: %s", persona)
            return response
        except Exception as e:
            logging.error("Error generating response: %s", e)
            logging.error(traceback.format_exc())
            return (
                self.ERROR_MESSAGE +
                " Rate limiting or internal error. Try again in 30s."
            )

    def _format_conversation_history(
        self, messages: List[Dict[str, Any]]
    ) -> str:
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
        persona: str = 'hermes'
    ) -> str:
        """Generate a response using RAG optimized for small datasets.

        Args:
            prompt: The user's input prompt
            user_id: Unique identifier for the user/session
            persona: Which persona/base prompt to use ('hermes' or 'prisma')

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

        # Get relevant context from Supabase vector store
        # using LangChain's standard method
        try:
            relevant_chunks = self.vector_store.similarity_search(
                query=prompt,
                k=self.RAG_TOP_K
            )
            if not relevant_chunks:
                logging.info(
                    "No relevant chunks found. Using standard generation."
                )
                return self.generate_gemini_response(prompt, persona)
        except Exception as e:
            logging.error("Error querying vector store: %s", e)
            logging.info("Falling back to standard generation.")
            return self.generate_gemini_response(prompt, persona)

        # Format conversation history for context
        conversation_history = self._format_conversation_history(
            state.data['conversation'][-10:]  # Last 10 messages
        )

        # Select base prompt based on persona
        base_prompt = self.base_prompts.get(persona, self.base_prompts['hermes'])

        # Prepare context
        prompt_prefix = (
            f"{base_prompt}\n\n" if base_prompt else ""
        )
        context_str = "\n\n".join(
            [chunk.page_content for chunk in relevant_chunks]
        )
        logging.info("Using %d chunks for context", len(relevant_chunks))

        # Construct ultra-optimized prompt for small datasets
        full_prompt = (
            f"{prompt_prefix}"
            f"Current date and time: "
            f"{datetime.utcnow().isoformat()}\n\n"
            f"=== CONVERSATION HISTORY ===\n"
            f"{conversation_history}\n\n"
            f"=== RELEVANT CONTEXT "
            f"(SMALL DATASET - MAXIMUM ACCURACY) ===\n"
            f"{context_str}\n\n"
            f"=== USER'S LATEST MESSAGE ===\n"
            f"User: {prompt}\n\n"
            f"IMPORTANT: This is a small dataset with limited documents. "
            f"Please provide a comprehensive response using "
            f"ALL relevant information "
            f"from the context above, especially information marked as "
            f"[IMPORTANT]. "
            f"Be thorough and detailed in your response."
        )

        try:
            # Generate response
            message = self.model.invoke([
                HumanMessage(content=full_prompt)
            ])
            response = message.content.strip()

            if "error" in response.lower() or not response:
                logging.warning(
                    "RAG response error or empty: '%s'", response
                )
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
            state.data['metadata']['last_updated'] = (
                datetime.utcnow().isoformat()
            )

            # Save updated conversation state
            self.conversation_state.save_state(user_id, state)

            logging.info("Generated RAG response successfully")
            return response
        except Exception as e:
            logging.error("Error generating RAG response: %s", e)
            logging.error(traceback.format_exc())
            return self.ERROR_MESSAGE + " RAG: Generation error."
