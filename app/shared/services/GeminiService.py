import logging
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# Optional Google / Gemini / Langchain specific
try:
    import google.generativeai as genai
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )

    LANGCHAIN_AVAILABLE = True
except ImportError:
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None
    HumanMessage = None
    LANGCHAIN_AVAILABLE = False

# Optional Vector Store - Standard LangChain integration
try:
    from langchain_community.vectorstores import SupabaseVectorStore
    from supabase import create_client

    SUPABASE_AVAILABLE = True
except ImportError:
    SupabaseVectorStore = None
    create_client = None
    SUPABASE_AVAILABLE = False

from app.shared.config.posthog_config import posthog_config
from app.shared.utils.conversation_state import ConversationState, State

# Import tools and services
from app.shared.utils.toolhub import get_all_tools

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Custom exceptions for better error handling
class GeminiServiceError(Exception):
    """Base exception for GeminiService errors."""

    pass


class PersonaNotFoundError(GeminiServiceError):
    """Raised when a requested persona is not found."""

    pass


class EmbeddingError(GeminiServiceError):
    """Raised when embedding generation fails."""

    pass


class VectorStoreError(GeminiServiceError):
    """Raised when vector store operations fail."""

    pass


class ToolExecutionError(GeminiServiceError):
    """Raised when tool execution fails."""

    pass


@dataclass
class PersonaConfig:
    """Configuration for a specific persona."""

    name: str
    base_prompt: str
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.3
    timeout: int = 60
    max_retries: int = 3
    allowed_tools: Optional[List[str]] = None  # None means all tools
    error_message_template: Optional[str] = None

    def __post_init__(self):
        """Set default templates if not provided."""
        if self.error_message_template is None:
            self.error_message_template = "Sorry, I encountered an error."


class GeminiService:
    """Extensible Gemini service supporting multiple personas with individual configurations."""

    # Default configuration (can be overridden per persona)
    DEFAULT_MODEL_NAME = "gemini-2.5-flash"
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_TIMEOUT = 60
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_ERROR_MESSAGE = "Sorry, the AI service encountered an error."

    # Using models/embedding-001 with 1536 dimensions (Gemini API)
    # IMPORTANT: Must match the model used to create embeddings in Supabase
    EMBEDDING_MODEL_NAME = "models/embedding-001"  # Gemini API embedding model
    EMBEDDING_DIMENSIONS = 1536  # Higher dimensionality for better accuracy
    TEXT_SPLITTER_CHUNK_SIZE = 1000  # Larger chunks for better context
    TEXT_SPLITTER_CHUNK_OVERLAP = 200  # More overlap for better context retention
    RAG_TOP_K = 5  # Reduced from 30 for better quality
    RAG_SIMILARITY_THRESHOLD = 0.65  # Raised to filter out low-quality matches

    def __init__(
        self,
        conversation_db_path: str = "conversations.db",
        persona_configs: Optional[Dict[str, PersonaConfig]] = None,
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain dependencies not available. "
                "Install with: pip install langchain-google-genai langchain-google-vertexai langchain-core"
            )

        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")

        # Initialize persona configurations
        self.persona_configs = self._initialize_persona_configs(persona_configs)

        # Initialize base model (will be customized per persona)
        self.base_model_class = ChatGoogleGenerativeAI

        # Get all available tools
        self.all_tools = get_all_tools()
        logging.info("Loaded %d tools for persona filtering", len(self.all_tools))

        # Load base prompts and create persona configs
        self.base_prompts = self._load_agent_prompts()
        self._create_persona_configs_from_prompts()

        logging.info(
            "Initialized GeminiService with personas: %s",
            list(self.persona_configs.keys()),
        )

        # Initialize Gemini embeddings (1536 dimensions)
        try:
            # Configure the Gemini API for LLM and embeddings
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

            # Create custom embeddings wrapper with explicit dimensionality
            # MUST use models/embedding-001 with 1536 dimensions to match Supabase embeddings
            _target_dimensionality = self.EMBEDDING_DIMENSIONS  # Store in closure

            class CustomDimensionalityEmbeddings(GoogleGenerativeAIEmbeddings):
                """Custom embeddings that enforce 1536 dimensions"""

                def embed_query(self, text: str, **kwargs) -> list[float]:
                    """Override to add output_dimensionality parameter"""
                    # kwargs ignored - using explicit parameters
                    try:
                        result = genai.embed_content(
                            model="models/embedding-001",  # Use the model name directly
                            content=text,
                            task_type=self.task_type,
                            output_dimensionality=_target_dimensionality,
                        )
                        return result["embedding"]
                    except Exception as e:
                        logging.error(
                            "Failed to generate embedding for query: %s", str(e)
                        )
                        raise EmbeddingError(
                            f"Failed to generate embedding: {str(e)}"
                        ) from e

                def embed_documents(
                    self, texts: list[str], **kwargs
                ) -> list[list[float]]:
                    """Override to add output_dimensionality parameter"""
                    # kwargs ignored - using explicit parameters
                    try:
                        return [self.embed_query(text) for text in texts]
                    except Exception as e:
                        logging.error(
                            "Failed to generate embeddings for documents: %s", str(e)
                        )
                        raise EmbeddingError(
                            f"Failed to generate document embeddings: {str(e)}"
                        ) from e

            self.embeddings_model = CustomDimensionalityEmbeddings(
                model="models/embedding-001",  # Use the model name directly
                google_api_key=os.environ["GOOGLE_API_KEY"],
                task_type="retrieval_document",
            )

            logging.info(
                "Initialized Gemini API Embeddings with model: %s (%dD)",
                "models/embedding-001",
                self.EMBEDDING_DIMENSIONS,
            )
        except (ValueError, KeyError) as e:
            logging.error(
                "Configuration error initializing Gemini API Embeddings: %s", e
            )
            raise EmbeddingError(f"Configuration error: {str(e)}") from e
        except Exception as e:
            logging.error("Failed to initialize Gemini API Embeddings: %s", e)
            logging.error(traceback.format_exc())
            raise EmbeddingError(f"Failed to initialize embeddings: {str(e)}") from e

        # Initialize Supabase vector store
        if not SUPABASE_AVAILABLE:
            logging.warning(
                "Supabase dependencies not available. "
                "Install with: pip install langchain-community supabase"
            )
            self.vector_store = None
        else:
            try:
                supabase_url = os.environ.get("SUPABASE_URL") or os.environ.get(
                    "SUPABASE_PROJECT_URL"
                )
                supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
                if not supabase_url or not supabase_key:
                    raise ValueError(
                        "SUPABASE_URL (or SUPABASE_PROJECT_URL) and SUPABASE_SERVICE_ROLE_KEY must be set."
                    )
                supabase_client = create_client(supabase_url, supabase_key)
                self.vector_store = SupabaseVectorStore(
                    client=supabase_client,
                    embedding=self.embeddings_model,
                    table_name="hermes_vectors",
                )
                logging.info("Initialized Supabase vector store using LangChain")
            except (ValueError, KeyError) as e:
                logging.error(
                    "Configuration error initializing Supabase vector store: %s", e
                )
                raise VectorStoreError(f"Configuration error: {str(e)}") from e
            except Exception as e:
                logging.error("Failed to initialize Supabase vector store: %s", e)
                logging.error(traceback.format_exc())
                raise VectorStoreError(
                    f"Failed to initialize vector store: {str(e)}"
                ) from e

        # Initialize conversation state
        self.conversation_state = ConversationState(db_path=conversation_db_path)

        vector_store_status = (
            "Supabase" if self.vector_store else "None (dependencies missing)"
        )
        logging.info(
            "GeminiService initialized. Embeddings: %s (%dD), Vector Store: %s",
            self.EMBEDDING_MODEL_NAME,
            self.EMBEDDING_DIMENSIONS,
            vector_store_status,
        )

    def _load_agent_prompts(self) -> Dict[str, str]:
        """
        Load persona prompts from markdown files in docs/Personas/.
        Returns a dictionary mapping persona names (lowercase) to their prompt content.
        """
        prompts = {}

        # Get the project root directory (3 levels up from this file)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        personas_dir = os.path.join(project_root, "docs", "Personas")

        if not os.path.exists(personas_dir):
            logging.warning("Personas directory not found: %s", personas_dir)
            return prompts

        # Load all .md files from the personas directory
        try:
            for filename in os.listdir(personas_dir):
                if filename.endswith(".md"):
                    persona_name = filename[
                        :-3
                    ].lower()  # Remove .md extension and lowercase
                    filepath = os.path.join(personas_dir, filename)

                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:
                                prompts[persona_name] = content
                                logging.info(
                                    "Loaded prompt for persona '%s' from %s (%d chars)",
                                    persona_name,
                                    filename,
                                    len(content),
                                )
                            else:
                                logging.warning("Empty prompt file: %s", filename)
                    except Exception as e:
                        logging.error("Failed to load prompt from %s: %s", filepath, e)
        except Exception as e:
            logging.error("Error reading personas directory %s: %s", personas_dir, e)

        return prompts

    def _initialize_persona_configs(
        self, persona_configs: Optional[Dict[str, PersonaConfig]]
    ) -> Dict[str, PersonaConfig]:
        """Initialize persona configurations with defaults."""
        configs = {}

        if persona_configs:
            configs.update(persona_configs)

        # Add default hermes config if not provided
        if "hermes" not in configs:
            configs["hermes"] = PersonaConfig(
                name="hermes", base_prompt=""  # Will be loaded from file
            )

        return configs

    def _create_persona_configs_from_prompts(self):
        """Create persona configs from loaded prompts."""
        for persona_name, base_prompt in self.base_prompts.items():
            if persona_name not in self.persona_configs:
                self.persona_configs[persona_name] = PersonaConfig(
                    name=persona_name, base_prompt=base_prompt
                )
            else:
                # Update existing config with loaded prompt
                self.persona_configs[persona_name].base_prompt = base_prompt

    def _get_persona_config(self, persona: str) -> PersonaConfig:
        """Get persona configuration with fallback."""
        if persona not in self.persona_configs:
            logging.warning("Unknown persona '%s', falling back to 'hermes'", persona)
            if "hermes" not in self.persona_configs:
                raise PersonaNotFoundError(
                    f"Persona '{persona}' not found and no fallback 'hermes' persona available"
                )
            persona = "hermes"

        return self.persona_configs[persona]

    def _create_model_for_persona(self, persona_config: PersonaConfig):
        """Create a model instance configured for a specific persona."""
        model = self.base_model_class(
            model=persona_config.model_name,
            temperature=persona_config.temperature,
            max_retries=persona_config.max_retries,
            timeout=persona_config.timeout,
        )

        # Filter tools based on persona configuration
        tools = self._filter_tools_for_persona(persona_config)
        return model.bind_tools(tools)

    def _filter_tools_for_persona(self, persona_config: PersonaConfig) -> List[Any]:
        """Filter tools based on persona configuration."""
        if persona_config.allowed_tools is None:
            return self.all_tools

        # Filter tools by name
        filtered_tools = []
        for tool in self.all_tools:
            if hasattr(tool, "name") and tool.name in persona_config.allowed_tools:
                filtered_tools.append(tool)

        logging.info(
            "Filtered %d tools for persona '%s'",
            len(filtered_tools),
            persona_config.name,
        )
        return filtered_tools

    def _execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[str]:
        """
        Execute tool calls and return formatted results.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            List of formatted tool result strings
        """
        tool_results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "unknown")
            tool_args = tool_call.get("args", {})
            logging.info("Executing tool: %s with args: %s", tool_name, tool_args)

            # Find and execute the tool
            from app.shared.utils.toolhub import get_tool_by_name

            tool = get_tool_by_name(tool_name)
            if tool:
                try:
                    result = tool._run(**tool_args)
                    tool_results.append(f"{tool_name}: {result}")
                    logging.info("Tool %s result: %s", tool_name, result)
                except (ValueError, TypeError, KeyError) as e:
                    error_msg = f"Error executing {tool_name}: {str(e)}"
                    tool_results.append(error_msg)
                    logging.error("Tool execution error for %s: %s", tool_name, str(e))
                    raise ToolExecutionError(
                        f"Failed to execute tool {tool_name}: {str(e)}"
                    ) from e
                except Exception as e:
                    error_msg = f"Unexpected error executing {tool_name}: {str(e)}"
                    tool_results.append(error_msg)
                    logging.error(
                        "Unexpected tool execution error for %s: %s", tool_name, str(e)
                    )
                    raise ToolExecutionError(
                        f"Unexpected error executing tool {tool_name}: {str(e)}"
                    ) from e
            else:
                tool_results.append(f"Tool {tool_name} not found")
                logging.warning("Tool not found: %s", tool_name)

        return tool_results

    def _create_tool_response_prompt(
        self, tool_results: List[str], original_prompt: str, persona: str
    ) -> str:
        """
        Create a follow-up prompt for incorporating tool results into natural responses.

        Args:
            tool_results: List of tool result strings
            original_prompt: Original user prompt
            persona: Persona name for context

        Returns:
            Formatted follow-up prompt
        """
        tool_results_text = "\n".join(tool_results)

        # Simple, generic template for tool response incorporation
        follow_up_prompt = f"""Based on the tool results below, provide a natural response that incorporates this information conversationally.

Tool Results:
{tool_results_text}

Original User Query: {original_prompt}

Don't just repeat the tool output verbatim - make it sound like a helpful assistant sharing information naturally."""

        return follow_up_prompt

    def _direct_similarity_search(self, query: str, k: int = 5, threshold: float = 0.7):
        """
        Direct Supabase vector similarity search using RPC.
        Bypasses broken LangChain wrapper.

        Args:
            query: Search query
            k: Number of results
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of (Document, score) tuples
        """
        try:

            from concurrent.futures import ThreadPoolExecutor, TimeoutError

            def _execute_search_rpc():
                # Generate embedding
                query_embedding = self.embeddings_model.embed_query(query)

                # Call Supabase RPC function directly
                logging.info("Calling match_documents RPC function...")
                return self.vector_store._client.rpc(
                    "match_documents",
                    {
                        "query_embedding": query_embedding,
                        "match_count": k * 2,  # Get more results to filter by threshold
                        "filter": {},  # Empty filter for now
                    },
                ).execute()

            # Execute with timeout using ThreadPoolExecutor (thread-safe)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_execute_search_rpc)
                response = future.result(timeout=25)  # 25s total timeout

            if not response.data:
                logging.warning("No data returned from match_documents RPC")
                return []

            # Log all similarity scores before filtering
            all_scores = [row.get("similarity", 0.0) for row in response.data]
            if all_scores:
                logging.info(
                    f"Raw similarity scores from DB: "
                    f"count={len(all_scores)}, "
                    f"max={max(all_scores):.3f}, "
                    f"min={min(all_scores):.3f}, "
                    f"avg={sum(all_scores)/len(all_scores):.3f}"
                )

            # Convert to LangChain Document format and filter by threshold
            from langchain_core.documents import Document

            results = []
            for row in response.data:
                similarity = row.get("similarity", 0.0)

                # Filter by threshold
                if similarity >= threshold:
                    doc = Document(
                        page_content=row.get("content", ""),
                        metadata=row.get("metadata", {}),
                    )
                    results.append((doc, similarity))

            logging.info(
                f"After threshold filter: {len(results)}/{len(response.data)} chunks passed (threshold={threshold})"
            )

            # Sort by similarity (descending) and limit to k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

        except TimeoutError as e:
            logging.error(f"Vector search timed out: {e}")
            logging.warning("Falling back to non-RAG response due to timeout")
            return []
        except Exception as e:
            logging.error("Direct similarity search failed: %s", e)
            logging.error(traceback.format_exc())
            return []

    def _create_callback_handler(
        self,
        user_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        persona: str = "hermes",
        operation_type: str = "standard",
    ) -> Optional[Any]:
        """
        Create a PostHog callback handler for tracking LLM events.

        Args:
            user_id: User ID for tracking
            trace_id: Trace ID for grouping events
            persona: Current persona name
            operation_type: Type of operation (standard, rag, enrichment)

        Returns:
            CallbackHandler or None
        """
        properties = {
            "persona": persona,
            "model": self.persona_configs.get(
                persona, self.persona_configs["hermes"]
            ).model_name,
            "operation_type": operation_type,
            "service": "GeminiService",
        }

        return posthog_config.get_callback_handler(
            user_id=user_id, trace_id=trace_id, properties=properties
        )

    def generate_gemini_response(
        self, prompt: str, persona: str = "hermes", user_id: Optional[str] = None
    ) -> str:
        """
        Generate a response using Gemini with optional tool execution.

        This method creates a persona-specific model, processes the user prompt,
        and handles tool calls if the model requests them. It ensures natural
        language responses by re-prompting the model with tool results.

        Args:
            prompt: User input prompt to process
            persona: Persona name for response style and configuration (default: "hermes")
            user_id: Optional user ID for analytics tracking

        Returns:
            Generated response string from the AI model

        Raises:
            PersonaNotFoundError: If the specified persona is not found
            ToolExecutionError: If tool execution fails
            GeminiServiceError: For other service-related errors

        Example:
            >>> service = GeminiService()
            >>> response = service.generate_gemini_response("What's the time?", "hermes")
            >>> print(response)
            "The current time is..."
        """
        persona_config = self._get_persona_config(persona)

        # Create persona-specific model
        model = self._create_model_for_persona(persona_config)

        # Simple prompt composition
        base_prompt = persona_config.base_prompt
        prompt_prefix = f"{base_prompt}\n\n" if base_prompt else ""
        full_prompt = f"{prompt_prefix}User Input: {prompt}" if base_prompt else prompt

        # Setup callbacks
        callback_handler = self._create_callback_handler(
            user_id=user_id, persona=persona, operation_type="standard"
        )
        callbacks = [callback_handler] if callback_handler else []
        config = {"callbacks": callbacks}

        try:
            logging.info(
                f"Invoking Gemini model with prompt length: {len(full_prompt)}"
            )
            message = model.invoke([HumanMessage(content=full_prompt)], config=config)
            logging.info("Gemini model invocation successful")

            content = message.content

            # Handle tool calls
            if hasattr(message, "tool_calls") and message.tool_calls:
                logging.info(f"Tool calls detected: {len(message.tool_calls)}")
                logging.info("Model requested %d tool call(s)", len(message.tool_calls))

                # Execute tools using abstracted method
                tool_results = self._execute_tools(message.tool_calls)

                # Create persona-aware follow-up prompt
                follow_up_prompt = self._create_tool_response_prompt(
                    tool_results, prompt, persona
                )

                # Generate follow-up response with tool results
                # Reuse callbacks for the follow-up generation
                follow_up_message = model.invoke(
                    [HumanMessage(content=follow_up_prompt)],
                    config={"callbacks": callbacks},
                )
                response = follow_up_message.content.strip()

                if not response or "error" in response.lower():
                    logging.warning(
                        "Follow-up response failed, falling back to formatted tool results"
                    )
                    return "\n".join(tool_results)

                logging.info("Generated natural response incorporating tool results")
                return response

            # Handle regular text response
            response = message.content.strip()
            if not response:
                logging.warning("Gemini returned empty response")
                return persona_config.error_message_template

            logging.info("Generated response successfully with persona: %s", persona)
            return response

        except Exception as e:
            logging.error("Error generating response: %s", e)
            logging.error(traceback.format_exc())
            return (
                persona_config.error_message_template
                + " Rate limiting or internal error. Try again in 30s."
            )

    def add_persona(self, persona_config: PersonaConfig) -> None:
        """
        Add a new persona configuration to the service.

        This method allows runtime addition of new personas with custom configurations.
        The persona will be immediately available for use in response generation.

        Args:
            persona_config: PersonaConfig object containing persona settings

        Raises:
            ValueError: If persona_config is invalid or persona name already exists

        Example:
            >>> service = GeminiService()
            >>> custom_persona = PersonaConfig(
            ...     name="customer_service",
            ...     base_prompt="You are a helpful customer service agent.",
            ...     temperature=0.5
            ... )
            >>> service.add_persona(custom_persona)
        """
        self.persona_configs[persona_config.name] = persona_config
        logging.info("Added persona configuration: %s", persona_config.name)

    def remove_persona(self, persona_name: str) -> bool:
        """
        Remove a persona configuration from the service.

        Args:
            persona_name: Name of the persona to remove

        Returns:
            True if persona was removed, False if persona was not found

        Example:
            >>> service = GeminiService()
            >>> success = service.remove_persona("customer_service")
            >>> print(success)
            True
        """
        if persona_name in self.persona_configs:
            del self.persona_configs[persona_name]
            logging.info("Removed persona configuration: %s", persona_name)
            return True
        return False

    def get_available_personas(self) -> List[str]:
        """Get list of available persona names."""
        return list(self.persona_configs.keys())

    def get_persona_info(self, persona_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific persona."""
        if persona_name not in self.persona_configs:
            return None

        config = self.persona_configs[persona_name]
        return {
            "name": config.name,
            "model_name": config.model_name,
            "temperature": config.temperature,
            "timeout": config.timeout,
            "allowed_tools": config.allowed_tools,
            "has_custom_error_message": bool(config.error_message_template),
        }

    def _format_conversation_history(self, messages: List[Dict[str, Any]]) -> str:
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "").strip()
            if content:
                formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def _enrich_query_for_vector_search(
        self,
        user_query: str,
        conversation_history: str = "",
        user_id: Optional[str] = None,
    ) -> str:
        """
        Enrich a user query using Gemini to make it more detailed and suitable for vector search.
        Expands vague queries with context, synonyms, and related terms.

        Args:
            user_query: The original user query
            conversation_history: Optional conversation context
            user_id: Optional user ID for analytics tracking

        Returns:
            Enriched query string optimized for vector similarity search
        """
        enrichment_prompt = f"""You are a query expansion expert. Your task is to
enrich the following user query to make it more suitable for semantic vector
search in a document database.

Rules:
1. Expand vague queries with context, synonyms, and related terms
2. Preserve the original intent and meaning
3. Add technical terms, domain-specific vocabulary, and common variations
4. Keep the enriched query concise (2-4 sentences max) (Under 300 chars)
5. Focus on searchable keywords and concepts, not conversational fluff
6. If the query is already detailed, return it with minor enhancements only

Conversation Context: {conversation_history}

Original Query: {user_query}

Enriched Query:"""

        try:
            # Use a simpler model call without tools for query enrichment
            base_model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3,  # Lower temp for focused enrichment
                max_retries=2,
                timeout=10,  # Shorter timeout for speed
            )

            # Setup callbacks
            callback_handler = self._create_callback_handler(
                user_id=user_id, persona="hermes", operation_type="query_enrichment"
            )
            callbacks = [callback_handler] if callback_handler else []

            message = base_model.invoke(
                [HumanMessage(content=enrichment_prompt)],
                config={"callbacks": callbacks},
            )
            # Validate enrichment - check if it adds meaningful value
            enriched_query = message.content.strip()

            return enriched_query

        except Exception as e:
            logging.error("Error enriching query: %s", e)
            logging.info("Falling back to original query")
            return user_query

    def generate_gemini_response_with_rag(
        self,
        prompt: str,
        user_id: str,
        persona: str = "hermes",
        min_chunk_length: int = 20,
    ) -> str:
        """
        Generate a response using Retrieval-Augmented Generation (RAG).

        This method combines vector similarity search with LLM generation to provide
        contextually relevant responses based on retrieved documents. It maintains
        conversation state and uses persona-specific configurations.

        Args:
            prompt: User input prompt to process
            user_id: Unique identifier for conversation state management
            persona: Persona name for response style and configuration (default: "hermes")
            min_chunk_length: Minimum length for retrieved document chunks (default: 20)

        Returns:
            Generated response string incorporating retrieved context

        Raises:
            PersonaNotFoundError: If the specified persona is not found
            VectorStoreError: If vector store operations fail
            EmbeddingError: If embedding generation fails
            GeminiServiceError: For other service-related errors

        Example:
            >>> service = GeminiService()
            >>> response = service.generate_gemini_response_with_rag(
            ...     "Tell me about Edwin's work", "user123", "hermes"
            ... )
            >>> print(response)
            "Based on the available information, Edwin has worked on..."
        """

        # Retrieve or initialize conversation state
        state = self.conversation_state.get_state(user_id)
        if not state or "conversation" not in state.data:
            state = State(
                data={
                    "conversation": [],
                    "metadata": {
                        "created_at": datetime.utcnow().isoformat(),
                        "last_updated": datetime.utcnow().isoformat(),
                        "message_count": 0,
                    },
                }
            )

        # Add user message
        state.data["conversation"].append(
            {
                "role": "user",
                "content": prompt,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        state.data["metadata"]["last_updated"] = datetime.utcnow().isoformat()
        state.data["metadata"]["message_count"] += 1

        # Vector store fallback
        if self.vector_store is None:
            logging.info("Vector store not available. Using standard generation.")
            return self.generate_gemini_response(prompt, persona, user_id=user_id)

        # Retrieve top relevant chunks using DIRECT RPC call
        try:
            # Add entity context to improve matching
            contextualized_query = f"Edwin Kassier {prompt}"
            logging.info(
                "Searching vector store with query: '%s'", contextualized_query
            )

            # Use direct RPC call instead of broken LangChain wrapper
            relevant_chunks = self._direct_similarity_search(
                query=contextualized_query,
                k=self.RAG_TOP_K,
                threshold=self.RAG_SIMILARITY_THRESHOLD,
            )

            if not relevant_chunks:
                logging.warning(
                    f"No chunks above similarity threshold {self.RAG_SIMILARITY_THRESHOLD}"
                )
                return self.generate_gemini_response(prompt, persona, user_id=user_id)

            # Log similarity scores for debugging
            scores = [score for _, score in relevant_chunks]
            logging.info(
                f"Retrieved {len(relevant_chunks)} high-quality chunks - "
                f"Scores: min={min(scores):.3f}, max={max(scores):.3f}, avg={sum(scores)/len(scores):.3f}"
            )

            # Log top chunks for debugging
            for idx, (doc, score) in enumerate(relevant_chunks[:3]):
                preview = doc.page_content[:100].replace("\n", " ")
                logging.info(
                    f"  Chunk {idx+1}: score={score:.3f}, preview='{preview}...'"
                )

            # Extract documents (already sorted by score)
            docs = [doc for doc, score in relevant_chunks]

            # Filter by minimum length
            docs = [
                doc for doc in docs if len(doc.page_content.strip()) >= min_chunk_length
            ]

            if not docs:
                logging.info(
                    "No chunks passed length filter. Using standard generation."
                )
                return self.generate_gemini_response(prompt, persona, user_id=user_id)

        except Exception as e:
            logging.error("Error in vector search: %s", e)
            logging.error(traceback.format_exc())
            logging.info("Falling back to standard generation.")
            return self.generate_gemini_response(prompt, persona, user_id=user_id)

        # Conversation history
        conversation_history = self._format_conversation_history(
            state.data["conversation"][-10:]
        )

        # Base prompt
        base_prompt = self.base_prompts.get(persona, self.base_prompts["hermes"])
        prompt_prefix = f"{base_prompt}\n\n" if base_prompt else ""

        # Merge chunks
        context_str = "\n\n".join([doc.page_content for doc in docs])
        logging.info(
            "Using %d chunks (%d chars) for context", len(docs), len(context_str)
        )

        # Construct strict RAG prompt
        full_prompt = (
            f"{prompt_prefix}"
            f"Current UTC time: {datetime.utcnow().isoformat()}\n\n"
            f"=== CONVERSATION HISTORY ===\n{conversation_history}\n\n"
            f"=== VERIFIED CONTEXT (YOUR ONLY SOURCE OF TRUTH) ===\n{context_str}\n\n"
            f"=== USER'S MESSAGE ===\nUser: {prompt}\n\n"
            f"CRITICAL INSTRUCTIONS:\n"
            f"1. Answer ONLY using information explicitly stated in the VERIFIED CONTEXT above\n"
            f"2. If the VERIFIED CONTEXT doesn't contain the answer, respond: 'I don't have that information'\n"
            f"3. DO NOT use your general knowledge, training data, or make assumptions\n"
            f"4. DO NOT invent, extrapolate, or fill in missing details\n"
            f"5. When facts conflict, prefer the most recent information in the context\n\n"
            f"Your response:"
        )

        # Invoke LLM
        try:
            # Use persona-aware model creation
            persona_config = self._get_persona_config(persona)
            model = self._create_model_for_persona(persona_config)

            # Setup callbacks
            callback_handler = self._create_callback_handler(
                user_id=user_id, persona=persona, operation_type="rag"
            )
            callbacks = [callback_handler] if callback_handler else []

            message = model.invoke(
                [HumanMessage(content=full_prompt)], config={"callbacks": callbacks}
            )
            response = message.content.strip()

            if not response or "error" in response.lower():
                logging.warning("RAG response error or empty: '%s'", response)
                return (
                    persona_config.error_message_template
                    + " RAG: Check rate limits or context."
                )

            # Add assistant response to conversation state
            state.data["conversation"].append(
                {
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            state.data["metadata"]["last_updated"] = datetime.utcnow().isoformat()

            # Save updated state
            self.conversation_state.save_state(user_id, state)
            logging.info("Generated RAG response successfully")
            return response

        except Exception as e:
            logging.error("Error generating RAG response: %s", e)
            logging.error(traceback.format_exc())
            persona_config = self._get_persona_config(persona)
            return persona_config.error_message_template + " RAG: Generation error."
