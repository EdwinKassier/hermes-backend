"""Persona generation service for Legion workers."""

import asyncio
import logging
from typing import Dict, List, Optional

from app.shared.utils.service_loader import get_async_llm_service

from .persona_selector import PersonaSelector
from .persona_templates import PersonaTemplateStore
from .persona_validator import PersonaInputValidator

logger = logging.getLogger(__name__)


class LegionPersonaGenerator:
    """Generates appropriate personas for Legion workers."""

    def __init__(self):
        self.llm_service = get_async_llm_service()
        self.template_store = PersonaTemplateStore()
        self.selector = PersonaSelector()
        self.validator = PersonaInputValidator()

    async def generate_persona_for_worker(
        self, role: str, task_description: str, context: Optional[Dict] = None
    ) -> str:
        """Generate a unique persona for a worker based on role and task."""
        try:
            # Input validation
            validated_role = self.validator.validate_role(role)
            validated_task = self.validator.validate_task_description(task_description)

            # Normalize role to match our templates
            normalized_role = self._normalize_role(validated_role)

            # Use predefined templates for common roles
            templates = self.template_store.get_templates_for_role(normalized_role)
            if templates:
                # Select persona based on task content
                persona = self.selector.select_persona_from_templates(
                    templates, validated_task
                )
                logger.debug(
                    f"Generated persona '{persona}' for role '{validated_role}' with task: {validated_task[:50]}..."
                )
                return persona

            # For unknown roles, generate a basic persona
            basic_persona = f"{normalized_role}_specialist"
            logger.debug(
                f"Generated basic persona '{basic_persona}' for unknown role '{validated_role}'"
            )
            return basic_persona

        except Exception as e:
            logger.error(f"Error generating persona for role '{role}': {e}")
            # Structured error with fallback
            try:
                fallback_role = self._normalize_role(self.validator.validate_role(role))
                fallback_persona = f"{fallback_role}_specialist"
            except Exception:
                fallback_persona = "general_specialist"
            logger.warning(f"Using fallback persona: {fallback_persona}")
            return fallback_persona

    def _normalize_role(self, role: str) -> str:
        """Normalize role names to match template keys."""
        role_lower = role.lower()

        # Map common role variations
        role_mappings = {
            "researcher": "research",
            "coder": "code",
            "programmer": "code",
            "analyst": "analysis",
            "data_analyst": "data",
            "scientist": "research",
            "engineer": "code",
            "architect": "code",
            "investigator": "research",
            "evaluator": "analysis",
            "interpreter": "analysis",
            "specialist": "general",
            "expert": "general",
        }

        return role_mappings.get(role_lower, role_lower)

    async def generate_personas_for_workers(
        self, workers: List[Dict], context: Optional[Dict] = None
    ) -> List[Dict]:
        """Generate personas for a list of workers with parallel processing."""
        if not workers:
            return []

        # Create tasks for parallel execution to avoid N+1 queries
        tasks = [
            self.generate_persona_for_worker(
                role=worker.get("role", "general"),
                task_description=worker.get("task_description", ""),
                context=context,
            )
            for worker in workers
        ]

        # Execute all persona generations in parallel
        persona_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle any exceptions
        updated_workers = []
        for worker, persona_result in zip(workers, persona_results):
            if isinstance(persona_result, Exception):
                # Handle generation failure gracefully
                logger.error(
                    f"Failed to generate persona for worker {worker.get('worker_id', 'unknown')}: {persona_result}"
                )
                # Use fallback persona based on role
                fallback_persona = (
                    f"{self._normalize_role(worker.get('role', 'general'))}_specialist"
                )
                persona = fallback_persona
            else:
                persona = persona_result

            # Create updated worker with persona
            worker_copy = worker.copy()
            worker_copy["persona"] = persona
            updated_workers.append(worker_copy)

        logger.info(
            f"Generated personas for {len(updated_workers)} workers (parallel processing)"
        )
        return updated_workers
