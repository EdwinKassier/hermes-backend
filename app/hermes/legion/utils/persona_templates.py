"""Persona template definitions for Legion workers."""

from typing import Dict, List


class PersonaTemplateStore:
    """Manages persona templates for different roles."""

    def __init__(self):
        self.role_persona_templates = {
            "research": [
                "thorough_investigator",
                "data_driven_analyst",
                "comprehensive_researcher",
                "methodical_explorer",
                "evidence_based_analyst",
            ],
            "code": [
                "pragmatic_developer",
                "innovative_architect",
                "clean_code_specialist",
                "performance_optimizer",
                "robust_engineer",
            ],
            "analysis": [
                "critical_thinker",
                "systematic_analyzer",
                "logical_reasoner",
                "insightful_evaluator",
                "data_interpreter",
            ],
            "data": [
                "precision_analyst",
                "pattern_recognizer",
                "statistical_expert",
                "data_storyteller",
                "insight_miner",
            ],
            "general": [
                "versatile_solver",
                "adaptive_problem_solver",
                "comprehensive_executor",
                "resourceful_agent",
                "flexible_specialist",
            ],
        }

    def get_templates_for_role(self, role: str) -> List[str]:
        """Get persona templates for a specific role."""
        return self.role_persona_templates.get(role, [])

    def get_all_roles(self) -> List[str]:
        """Get all available roles."""
        return list(self.role_persona_templates.keys())

    def validate_templates(self) -> bool:
        """Validate that all templates are properly structured."""
        for role, templates in self.role_persona_templates.items():
            if not isinstance(templates, list) or len(templates) == 0:
                return False
            if not all(isinstance(template, str) for template in templates):
                return False

        # Check uniqueness across all templates
        all_templates = []
        for templates in self.role_persona_templates.values():
            all_templates.extend(templates)
        return len(all_templates) == len(set(all_templates))
