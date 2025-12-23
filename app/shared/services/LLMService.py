import logging
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# LangChain imports - provider-agnostic
try:
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    init_chat_model = None
    HumanMessage = None
    LANGCHAIN_AVAILABLE = False

from app.shared.config.langfuse_config import langfuse_config
from app.shared.config.posthog_config import posthog_config
from app.shared.utils.conversation_state import ConversationState

# Import tools and services
from app.shared.utils.toolhub import get_all_tools

# Default model from environment (provider-agnostic)
# Examples: gemini-2.5-flash, gpt-4o, claude-3-sonnet, ollama/llama3
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Custom exceptions for better error handling
class LLMServiceError(Exception):
    """Base exception for LLMService errors."""

    pass


# Backward compatibility alias
GeminiServiceError = LLMServiceError


class PersonaNotFoundError(LLMServiceError):
    """Raised when a requested persona is not found."""

    pass


class ToolExecutionError(LLMServiceError):
    """Raised when tool execution fails."""

    pass


@dataclass
class PersonaConfig:
    """Configuration for a specific persona."""

    name: str
    base_prompt: str
    model_name: str = DEFAULT_LLM_MODEL
    temperature: float = 0.3
    timeout: int = 60
    max_retries: int = 3
    allowed_tools: Optional[List[str]] = None  # None means all tools
    error_message_template: Optional[str] = None

    def __post_init__(self):
        """Set default templates if not provided."""
        if self.error_message_template is None:
            self.error_message_template = "Sorry, I encountered an error."


class LLMService:
    """Provider-agnostic LLM service supporting multiple personas with individual configurations.

    Uses LangChain's init_chat_model() for automatic provider selection based on model name.
    Supported providers: Gemini, OpenAI, Anthropic, Ollama, and more.

    Features:
    - Model caching: Models are cached per-persona to avoid re-initialization overhead
    - Tool binding: Tools are bound to models and cached together
    - Provider-agnostic: Supports multiple LLM providers via init_chat_model()
    """

    # Default configuration (can be overridden per persona)
    DEFAULT_MODEL_NAME = DEFAULT_LLM_MODEL
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_TIMEOUT = 60
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_ERROR_MESSAGE = "Sorry, the AI service encountered an error."

    # Model cache for performance optimization
    _model_cache: Dict[str, Any] = {}

    def __init__(
        self,
        conversation_db_path: str = "conversations.db",
        persona_configs: Optional[Dict[str, PersonaConfig]] = None,
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain dependencies not available. "
                "Install with: pip install langchain langchain-core"
            )

        # Initialize persona configurations
        self.persona_configs = self._initialize_persona_configs(persona_configs)

        # Get all available tools
        self.all_tools = get_all_tools()
        logging.info("Loaded %d tools for persona filtering", len(self.all_tools))

        # Load base prompts and create persona configs
        self.base_prompts = self._load_agent_prompts()
        self._create_persona_configs_from_prompts()

        logging.info(
            "Initialized LLMService with personas: %s (model: %s)",
            list(self.persona_configs.keys()),
            DEFAULT_LLM_MODEL,
        )

        # Initialize conversation state
        self.conversation_state = ConversationState(db_path=conversation_db_path)

        logging.info("LLMService initialized.")

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
        """Create or retrieve a cached model instance for a specific persona.

        Uses LangChain's init_chat_model() for provider-agnostic model initialization.
        Models are cached to avoid re-initialization overhead on every request.

        The provider is automatically detected from the model name prefix:
        - gemini-* → Google GenAI
        - gpt-* → OpenAI
        - claude-* → Anthropic
        - ollama/* → Ollama (local)

        Performance: Caching saves ~50-100ms per request by avoiding:
        - Model re-initialization network calls
        - Tool schema re-processing on bind_tools()
        """
        # Build cache key from persona config values that affect model behavior
        cache_key = f"{persona_config.name}:{persona_config.model_name}:{persona_config.temperature}"

        # Check cache first
        if cache_key in self._model_cache:
            logging.debug(f"Using cached model for persona '{persona_config.name}'")
            return self._model_cache[cache_key]

        # Create new model instance
        logging.info(f"Creating new model instance for persona '{persona_config.name}'")
        model = init_chat_model(
            persona_config.model_name,
            temperature=persona_config.temperature,
            max_retries=persona_config.max_retries,
        )

        # Filter tools based on persona configuration
        tools = self._filter_tools_for_persona(persona_config)
        bound_model = model.bind_tools(tools)

        # Cache for future requests
        self._model_cache[cache_key] = bound_model
        logging.info(
            f"Cached model for persona '{persona_config.name}' (cache key: {cache_key})"
        )

        return bound_model

    def clear_model_cache(self, persona_name: Optional[str] = None) -> int:
        """Clear the model cache to force re-initialization.

        Useful when persona configurations or tools change.

        Args:
            persona_name: If provided, only clear cache for this persona.
                         If None, clear entire cache.

        Returns:
            Number of cache entries cleared
        """
        if persona_name is None:
            count = len(self._model_cache)
            self._model_cache.clear()
            logging.info(f"Cleared entire model cache ({count} entries)")
            return count

        # Clear entries matching persona name
        keys_to_remove = [
            k for k in self._model_cache if k.startswith(f"{persona_name}:")
        ]
        for key in keys_to_remove:
            del self._model_cache[key]

        logging.info(
            f"Cleared {len(keys_to_remove)} cache entries for persona '{persona_name}'"
        )
        return len(keys_to_remove)

    def _filter_tools_for_persona(self, persona_config: PersonaConfig) -> List[Any]:
        """Filter tools based on persona configuration."""
        if persona_config.allowed_tools is None:
            # Use all tools, but filter out degraded ones
            candidates = self.all_tools
        else:
            # Filter tools by name
            candidates = []
            for tool in self.all_tools:
                if hasattr(tool, "name") and tool.name in persona_config.allowed_tools:
                    candidates.append(tool)

        # Apply health check filter
        from app.hermes.legion.utils.tool_registry import get_tool_registry

        registry = get_tool_registry()

        filtered_tools = []
        for tool in candidates:
            tool_name = getattr(tool, "name", str(tool))
            if registry.is_tool_healthy(tool_name):
                filtered_tools.append(tool)
            else:
                logging.warning(
                    f"Excluding degraded tool '{tool_name}' from persona '{persona_config.name}'"
                )

        logging.info(
            "Filtered %d tools for persona '%s' (originally %d)",
            len(filtered_tools),
            persona_config.name,
            len(candidates),
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
        from app.hermes.legion.utils.tool_registry import get_tool_registry
        from app.shared.utils.toolhub import get_tool_by_name

        tool_results = []
        registry = get_tool_registry()

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "unknown")
            tool_args = tool_call.get("args", {})
            logging.info("Executing tool: %s with args: %s", tool_name, tool_args)

            # Find and execute the tool
            tool = registry.get_tool(tool_name)
            if tool:
                try:
                    # Double-check health before execution
                    if not registry.is_tool_healthy(tool_name):
                        msg = (
                            f"Tool '{tool_name}' is currently degraded and unavailable."
                        )
                        tool_results.append(msg)
                        logging.warning(msg)
                        continue

                    result = tool._run(**tool_args)
                    tool_results.append(f"{tool_name}: {result}")
                    logging.info("Tool %s result: %s", tool_name, result)

                    # Mark success
                    registry.mark_tool_success(tool_name)

                except Exception as e:
                    # Sanitize error message length
                    error_str = str(e)
                    if len(error_str) > 500:
                        error_str = error_str[:497] + "..."

                    error_msg = f"Error executing {tool_name}: {error_str}"
                    tool_results.append(error_msg)

                    logging.error(
                        "Tool execution error for %s: %s", tool_name, error_str
                    )

                    # Mark failure and potentially degrade
                    registry.mark_tool_failed(tool_name, error_str)

                    # We do NOT raise ToolExecutionError anymore, allowing the LLM
                    # to see the error and recover/retry
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
        follow_up_prompt = f"""Based on the tool results below, provide a response to the original query.

Tool Results:
{tool_results_text}

Original User Query: {original_prompt}

IMPORTANT: Ensure you fully address the Original User Query requirements, including any formatting, citation, or style instructions provided there."""

        return follow_up_prompt

    def _extract_text_content(self, content: Any, fallback: str = "") -> str:
        """
        Extract text content from various LangChain message content types.

        Handles:
        - String content (direct return, stripped)
        - List of content blocks (extracts text elements, filters non-text)
        - None/empty (returns fallback)
        - Dict objects (attempts to extract text from common keys)
        - Other types (attempts string conversion)

        Args:
            content: The content from AIMessage.content (can be any type)
            fallback: Default string to return if content cannot be extracted

        Returns:
            str: Extracted text content, or fallback if extraction fails

        Examples:
            >>> service._extract_text_content("Hello world")
            'Hello world'
            >>> service._extract_text_content(["Hello", "world"])
            'Hello world'
            >>> service._extract_text_content(None)
            ''
            >>> service._extract_text_content([{"type": "text", "text": "Hello"}])
            'Hello'
        """
        # Handle None/empty
        if content is None:
            return fallback

        # Handle string (most common case - optimize first)
        if isinstance(content, str):
            return content.strip() or fallback

        # Handle list (the problematic case)
        if isinstance(content, list):
            logging.debug(
                f"Extracting text from list content with {len(content)} items"
            )
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    # Extract text from dict (check common LangChain content keys)
                    text = (
                        item.get("text") or item.get("content") or item.get("message")
                    )
                    if text and isinstance(text, str):
                        text_parts.append(text)
                else:
                    # Try string conversion for other types
                    try:
                        str_item = str(item)
                        if str_item and str_item.strip():
                            text_parts.append(str_item)
                    except Exception:
                        pass  # Skip items that can't be converted

            result = " ".join(text_parts).strip()
            return result or fallback

        # Handle dict (less common but possible)
        if isinstance(content, dict):
            text = (
                content.get("text") or content.get("content") or content.get("message")
            )
            if text:
                return str(text).strip() or fallback

        # Fallback for other types
        try:
            result = str(content).strip()
            if result:
                return result
        except Exception as e:
            logging.warning(
                f"Unexpected content type {type(content)} in message.content: {e}. "
                f"Attempting fallback conversion."
            )

        return fallback

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
        self,
        prompt: str,
        persona: str = "hermes",
        user_id: Optional[str] = None,
        trace_id: Optional[str] = None,
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
            trace_id: Optional trace ID for distributed tracing

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
        posthog_handler = self._create_callback_handler(
            user_id=user_id, persona=persona, operation_type="standard"
        )
        langfuse_handler = langfuse_config.get_callback_handler()

        # Combine all available callbacks
        callbacks = []
        if posthog_handler:
            callbacks.append(posthog_handler)
        if langfuse_handler:
            callbacks.append(langfuse_handler)

        # Add metadata for Langfuse trace attributes (v3 pattern)
        # Langfuse v3 supports trace attributes via metadata in config
        metadata = {}
        if user_id:
            metadata["langfuse_user_id"] = user_id
        if trace_id:
            metadata["langfuse_trace_id"] = trace_id

        config = {"callbacks": callbacks, "metadata": metadata}

        try:
            logging.info(
                f"Invoking Gemini model with prompt length: {len(full_prompt)}"
            )
            message = model.invoke([HumanMessage(content=full_prompt)], config=config)
            logging.info("Gemini model invocation successful")

            content = self._extract_text_content(message.content)

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
                    config={"callbacks": callbacks, "metadata": metadata},
                )
                response = self._extract_text_content(follow_up_message.content)

                if not response or "error" in response.lower():
                    logging.warning(
                        "Follow-up response failed, falling back to formatted tool results"
                    )
                    return "\n".join(tool_results)

                logging.info("Generated natural response incorporating tool results")
                return response

            # Handle regular text response
            response = self._extract_text_content(message.content)
            if not response:
                logging.warning(
                    "Gemini returned empty response for persona: %s", persona
                )

                # Return error template as final fallback
                return persona_config.error_message_template

            # Use LLM to clean up and complete the response if needed
            response = self._llm_cleanup_response(response, prompt, persona)

            logging.info("Generated response successfully with persona: %s", persona)
            return response

        except Exception as e:
            logging.error("Error generating response: %s", e)
            logging.error(traceback.format_exc())
            return (
                persona_config.error_message_template
                + " Rate limiting or internal error. Try again in 30s."
            )

    def _llm_cleanup_response(
        self, response: str, original_prompt: str, persona: str
    ) -> str:
        """
        Use LLM to clean up and complete responses that appear incomplete or malformed.

        Simple and effective: if the response looks problematic, ask the LLM to fix it.

        Args:
            response: The generated response from the LLM
            original_prompt: The original prompt for context
            persona: The persona used for generation

        Returns:
            Cleaned up response
        """
        if not response or not response.strip():
            return response

        # Quick check if response needs cleanup
        if self._response_needs_cleanup(response, original_prompt, persona):
            logging.info("Response needs cleanup, using LLM to fix it")
            try:
                cleaned_response = self._ask_llm_to_cleanup(
                    response, original_prompt, persona
                )
                if cleaned_response and len(cleaned_response.strip()) > len(
                    response.strip()
                ):
                    logging.info("LLM cleanup successful, response improved")
                    return cleaned_response
                else:
                    logging.warning("LLM cleanup didn't improve response")
            except Exception as e:
                logging.error("LLM cleanup failed: %s", e)

        return response

    def _response_needs_cleanup(self, response: str, prompt: str, persona: str) -> bool:
        """
        Selective check to determine if response needs LLM cleanup.

        Optimized for latency: multiple early exits for well-formed responses.
        Cleanup is only triggered for short responses with incomplete patterns.
        """
        # Early exit 1: Empty or very short responses
        if not response or len(response.strip()) < 10:
            return False  # Too short to be worth cleaning

        # Early exit 2: Response is long enough to be complete (>200 chars)
        response_stripped = response.strip()
        if len(response_stripped) > 200:
            return False  # Long responses are almost always complete

        # Early exit 3: Response ends with proper punctuation (complete sentence)
        if response_stripped and response_stripped[-1] in ".!?)\"'":
            return False  # Proper sentence ending = complete

        # Early exit 4: Response ends with closing code block (complete code)
        if response_stripped.endswith("```"):
            return False  # Closed code block = complete

        import re

        # Only check for problems in short (<50 char) responses with specific patterns
        if len(response_stripped) >= 50:
            return False  # Medium-length responses don't need cleanup

        # Check for obvious incomplete patterns (short responses only)
        incomplete_patterns = [
            r":\s*$",  # Lines ending with colon (Python blocks)
            r"^\s*\w+\s*\([^)]*$",  # Unclosed function calls
        ]

        return any(
            re.search(pattern, response, re.MULTILINE)
            for pattern in incomplete_patterns
        )

    def _ask_llm_to_cleanup(
        self, response: str, original_prompt: str, persona: str
    ) -> str:
        """
        Ask the LLM to clean up and complete the response.
        """
        # Create a simple cleanup prompt
        cleanup_prompt = f"""Please clean up and complete this response. Make sure it's properly formatted and complete:

Original Request: {original_prompt}

Current Response: {response}

Provide a cleaned up, complete version of the response above. Ensure proper formatting, complete code blocks, and all necessary closing elements."""

        try:
            # Use a simplified model call for cleanup (reuse existing model setup)
            persona_config = self._get_persona_config(persona)

            # Create model for cleanup (reuse the same model instance if possible)
            model = self._create_model_for_persona(persona_config)

            # Simple cleanup call
            cleanup_message = model.invoke([HumanMessage(content=cleanup_prompt)])
            cleaned_content = self._extract_text_content(cleanup_message.content)

            return cleaned_content if cleaned_content else response

        except Exception as e:
            logging.error("Cleanup LLM call failed: %s", e)
            return response  # Return original if cleanup fails

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

    def add_personas(self, persona_configs: Dict[str, PersonaConfig]) -> None:
        """
        Add multiple persona configurations to the service.

        Args:
            persona_configs: Dictionary of persona name to PersonaConfig objects

        Example:
            >>> service = GeminiService()
            >>> personas = {"agent1": config1, "agent2": config2}
            >>> service.add_personas(personas)
        """
        self.persona_configs.update(persona_configs)
        logging.info(f"Added {len(persona_configs)} personas to GeminiService")
        for name, config in persona_configs.items():
            logging.debug("Added persona configuration: %s", name)

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


# Backward compatibility alias
GeminiService = LLMService
