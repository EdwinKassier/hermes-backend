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

from app.shared.config.langfuse_config import langfuse_config
from app.shared.config.posthog_config import posthog_config
from app.shared.utils.conversation_state import ConversationState

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

        # Initialize conversation state
        self.conversation_state = ConversationState(db_path=conversation_db_path)

        logging.info("GeminiService initialized.")

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
        follow_up_prompt = f"""Based on the tool results below, provide a response to the original query.

Tool Results:
{tool_results_text}

Original User Query: {original_prompt}

IMPORTANT: Ensure you fully address the Original User Query requirements, including any formatting, citation, or style instructions provided there."""

        return follow_up_prompt

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
                    config={"callbacks": callbacks, "metadata": metadata},
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
