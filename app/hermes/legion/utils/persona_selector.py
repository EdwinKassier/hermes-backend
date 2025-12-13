"""Persona selection logic for Legion workers."""

from typing import List


class PersonaSelector:
    """Handles persona selection based on task content."""

    def select_persona_from_templates(
        self, templates: List[str], task_description: str
    ) -> str:
        """Select appropriate persona from templates based on task content."""
        task_lower = task_description.lower()

        # Keywords that influence persona selection
        if any(
            keyword in task_lower
            for keyword in ["research", "investigate", "explore", "discover"]
        ):
            return templates[0]  # thorough_investigator
        elif any(
            keyword in task_lower
            for keyword in ["analyze", "evaluate", "assess", "review"]
        ):
            return templates[1]  # data_driven_analyst
        elif any(
            keyword in task_lower for keyword in ["design", "architecture", "structure"]
        ):
            return templates[2]  # clean_code_specialist or comprehensive_researcher
        elif any(
            keyword in task_lower
            for keyword in ["optimize", "performance", "efficient"]
        ):
            return templates[3]  # performance_optimizer
        elif any(keyword in task_lower for keyword in ["robust", "reliable", "secure"]):
            return templates[4]  # robust_engineer
        else:
            # Default to first template
            return templates[0]
