"""Tool allocation for dynamic agent tool assignment."""

import logging
from typing import Any, Dict, List, Optional

from app.shared.utils.service_loader import get_gemini_service

from .persona_context import get_current_legion_persona

logger = logging.getLogger(__name__)


class ToolAllocator:
    """
    Manages dynamic tool allocation to agents based on task requirements.

    This implements the Magentic pattern's dynamic tool allocation capability.
    """

    def __init__(self):
        """Initialize tool allocator."""
        self._gemini_service = None

        # Tool capability mapping - ONLY tools with unique external capabilities
        # Updated 2025-11-19: Removed file tools (OCR'd) and python_repl (no code execution)
        # Tool capability mapping - ONLY tools with unique external capabilities
        # Updated 2025-11-19: Removed file tools (OCR'd) and python_repl (no code execution)
        # Only keeping tools that provide capabilities LLMs fundamentally cannot perform
        # Map capabilities to tools
        # Note: Legacy tool capability mappings - dynamic agents specify tools directly

        # Validate on initialization
        self._validated = False

    def _validate_tool_capabilities(self):
        """
        Validate that all tools in capability mapping actually exist.

        This prevents runtime errors from referencing non-existent tools.
        Logs warnings for any mismatches.
        """
        if self._validated:
            return

        try:
            available_tools = self.gemini_service.all_tools
            available_names = {self._get_tool_name(t) for t in available_tools}

            logger.info(f"Available tools: {available_names}")
            logger.info(f"Mapped tools: {set(self._tool_capabilities.keys())}")

            # Check for tools in mapping that don't exist
            mapped_names = set(self._tool_capabilities.keys())
            missing_tools = mapped_names - available_names

            if missing_tools:
                logger.error(
                    f"CRITICAL: Tools in capability mapping but don't exist: {missing_tools}. "
                    f"This will cause runtime errors!"
                )

            # Check for tools that exist but aren't mapped
            unmapped_tools = available_names - mapped_names
            if unmapped_tools:
                logger.warning(
                    f"Tools exist but not in capability mapping: {unmapped_tools}. "
                    f"These tools won't be allocated by task type."
                )

            # Log summary
            logger.info(
                f"Tool validation complete: {len(available_names)} available, "
                f"{len(mapped_names)} mapped, {len(missing_tools)} missing, "
                f"{len(unmapped_tools)} unmapped"
            )

            self._validated = True

        except Exception as e:
            logger.error(f"Failed to validate tool capabilities: {e}")

    @property
    def gemini_service(self):
        """Lazy load Gemini service and validate tools."""
        if self._gemini_service is None:
            self._gemini_service = get_gemini_service()
            # Validate tools on first access
            self._validate_tool_capabilities()
        return self._gemini_service

    def allocate_tools_with_ai(
        self,
        task_type: str,
        task_description: str,
        available_tools: List[Any],
    ) -> List[str]:
        """
        Use AI to select appropriate tools for a task using structured JSON output.

        Args:
            task_type: Type of task (research, code, analysis, data)
            task_description: Detailed description of the task
            available_tools: List of available tool objects

        Returns:
            List of tool names to allocate
        """
        try:
            # Build tool descriptions
            tool_descriptions = {}
            for tool in available_tools:
                tool_name = self._get_tool_name(tool)
                tool_desc = self._get_tool_description(tool)
                tool_descriptions[tool_name] = tool_desc

            # Build AI prompt
            prompt = self._build_tool_selection_prompt(
                task_type, task_description, tool_descriptions
            )

            # Get AI recommendation with circuit breaker
            from ..utils.resilience import get_llm_circuit_breaker

            circuit_breaker = get_llm_circuit_breaker()

            response = circuit_breaker.call(
                self.gemini_service.generate_gemini_response,
                prompt,
                persona=get_current_legion_persona(),
            )

            # Parse tool names from response
            selected_tools = self._parse_tool_selection(
                response, set(tool_descriptions.keys())
            )

            logger.info(
                f"AI tool allocation for {task_type}: {selected_tools} "
                f"(from {len(available_tools)} available)"
            )

            return selected_tools

        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Network error in tool allocation: {e}, using fallback")
            return self._fallback_tool_allocation(task_type, available_tools)

        except (ValueError, KeyError) as e:
            logger.error(f"Tool parsing error: {e}, using fallback")
            return self._fallback_tool_allocation(task_type, available_tools)

        except Exception as e:
            logger.exception(
                f"Unexpected error in AI tool allocation: {e}, using fallback"
            )
            return self._fallback_tool_allocation(task_type, available_tools)

    def _build_tool_selection_prompt(
        self, task_type: str, task_description: str, tool_descriptions: dict
    ) -> str:
        """Build AI prompt for tool selection with JSON requirement."""
        tools_list = "\n".join(
            [f"- {name}: {desc}" for name, desc in tool_descriptions.items()]
        )

        return f"""Select the appropriate tools for this task.

Task type: {task_type}
Task description: "{task_description}"

Available tools:
{tools_list}

Select ONLY the tools that are necessary for this specific task.
- Don't select tools that aren't needed
- Do select all tools that will be useful
- Consider the task type and description carefully

Respond with a JSON object containing a list of selected tool names.
Format: {{"selected_tools": ["tool_name1", "tool_name2"]}}
"""

    def _parse_tool_selection(self, response: str, available_tools: set) -> List[str]:
        """
        Parse tool names from AI response (expects JSON format).

        Args:
            response: AI response containing tool selection
            available_tools: Set of valid tool names

        Returns:
            List of selected tool names

        Raises:
            ValueError: If response format is invalid
        """
        from .llm_utils import extract_json_from_llm_response

        try:
            data = extract_json_from_llm_response(response)
            tools = data.get("selected_tools", [])

            if not isinstance(tools, list):
                raise ValueError("'selected_tools' field must be a list")

            # Validate all tools exist
            selected = []
            for tool in tools:
                if tool in available_tools:
                    selected.append(tool)
                else:
                    logger.warning(f"AI selected non-existent tool: {tool}")

            return selected

        except (ValueError, KeyError) as e:
            logger.error(f"Failed to parse tool selection: {e}")
            raise ValueError(f"Invalid tool selection format: {e}")

    def _fallback_tool_allocation(
        self, task_type: str, available_tools: List[Any]
    ) -> List[str]:
        """Fallback to static tool allocation if AI fails."""
        # Use original static mapping logic
        relevant_tools = []

        for tool in available_tools:
            tool_name = self._get_tool_name(tool)
            capabilities = self._tool_capabilities.get(tool_name, [])

            if task_type in capabilities:
                relevant_tools.append(tool_name)
            elif tool_name not in self._tool_capabilities:
                # Include unknown tools (conservative)
                relevant_tools.append(tool_name)

        return (
            relevant_tools
            if relevant_tools
            else [self._get_tool_name(t) for t in available_tools]
        )

    def _get_tool_description(self, tool: Any) -> str:
        """Extract tool description from tool object."""
        if hasattr(tool, "description"):
            return tool.description
        elif hasattr(tool, "__doc__") and tool.__doc__:
            # Use first line of docstring
            return tool.__doc__.split("\n")[0].strip()
        else:
            # Generate description from name
            tool_name = self._get_tool_name(tool)
            return f"Tool for {tool_name.replace('_', ' ')}"

    def allocate_tools_for_task(
        self,
        task_type: str,
        task_description: str,
        persona: str = "hermes",
        required_tools: Optional[List[str]] = None,
        use_ai: bool = True,  # Re-enabled now that we have 6 tools
    ) -> List[Any]:
        """
        Allocate tools for a specific task.

        Args:
            task_type: Type of task (research, code, analysis)
            task_description: Description of the task
            persona: AI persona to filter tools by
            required_tools: Optional list of required tool names
            use_ai: Whether to use AI for tool selection (default: True)
                   Re-enabled now that we have 6 functional tools.

        Returns:
            List of tool objects allocated for this task
        """
        # Get all available tools for the persona
        all_tools = self._get_tools_for_persona(persona)

        if use_ai and task_description:
            # Use AI to select tools
            logger.info("Using AI for tool selection")
            selected_tool_names = self.allocate_tools_with_ai(
                task_type, task_description, all_tools
            )

            # Convert tool names to tool objects
            relevant_tools = [
                tool
                for tool in all_tools
                if self._get_tool_name(tool) in selected_tool_names
            ]
        else:
            # Fallback to static filtering
            logger.info("Using static filtering for tool selection")
            relevant_tools = self._filter_tools_by_task_type(all_tools, task_type)

        # If specific tools are required, ensure they're included
        if required_tools:
            relevant_tools = self._ensure_required_tools(
                relevant_tools, all_tools, required_tools
            )

        logger.info(
            f"Allocated {len(relevant_tools)} tools for {task_type} task: "
            f"{[self._get_tool_name(t) for t in relevant_tools]}"
        )

        return relevant_tools

    def _get_tools_for_persona(self, persona: str) -> List[Any]:
        """Get tools available for a persona."""
        try:
            gemini_service = self.gemini_service

            # Try public method first
            if hasattr(gemini_service, "get_tools_for_persona"):
                return gemini_service.get_tools_for_persona(persona)

            # Fallback to getting persona config and filtering
            persona_config = gemini_service.persona_configs.get(persona)
            if persona_config and hasattr(gemini_service, "_filter_tools_for_persona"):
                return gemini_service._filter_tools_for_persona(persona_config)

            # Last resort: return all tools
            return getattr(gemini_service, "all_tools", [])

        except Exception as e:
            logger.error(f"Failed to get tools for persona '{persona}': {e}")
            return []

    def _filter_tools_by_task_type(self, tools: List[Any], task_type: str) -> List[Any]:
        """Filter tools relevant to task type."""
        if not task_type:
            return tools

        relevant_tools = []

        for tool in tools:
            tool_name = self._get_tool_name(tool)

            # Check if tool is relevant for this task type
            capabilities = self._tool_capabilities.get(tool_name, [])
            if task_type in capabilities:
                relevant_tools.append(tool)
            # If not in mapping, include it anyway (conservative approach)
            elif tool_name not in self._tool_capabilities:
                relevant_tools.append(tool)

        return relevant_tools if relevant_tools else tools

    def _ensure_required_tools(
        self,
        current_tools: List[Any],
        all_tools: List[Any],
        required_tool_names: List[str],
    ) -> List[Any]:
        """Ensure required tools are in the list."""
        current_tool_names = {self._get_tool_name(t) for t in current_tools}

        for required_name in required_tool_names:
            if required_name not in current_tool_names:
                # Find and add the required tool
                for tool in all_tools:
                    if self._get_tool_name(tool) == required_name:
                        current_tools.append(tool)
                        break

        return current_tools

    def _get_tool_name(self, tool: Any) -> str:
        """Extract tool name from tool object."""
        if hasattr(tool, "name"):
            return tool.name
        elif hasattr(tool, "__name__"):
            return tool.__name__
        else:
            return str(tool)

    def register_tool_capability(self, tool_name: str, capabilities: List[str]) -> None:
        """
        Register tool capabilities for better matching.

        Args:
            tool_name: Name of the tool
            capabilities: List of task types this tool supports
        """
        self._tool_capabilities[tool_name] = capabilities
        logger.info(f"Registered capabilities for tool '{tool_name}': {capabilities}")

    def get_tool_capabilities(self) -> Dict[str, List[str]]:
        """Get all registered tool capabilities."""
        return self._tool_capabilities.copy()
