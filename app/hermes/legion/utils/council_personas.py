"""Council persona definitions and selection logic."""

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class CouncilPersona:
    """Definition of a council member persona."""

    name: str
    description: str
    perspective: str
    strengths: List[str]


# Define available council personas
COUNCIL_PERSONAS = {
    "optimist": CouncilPersona(
        name="optimist",
        description="Focuses on opportunities, benefits, and positive outcomes",
        perspective="What are the best-case scenarios and opportunities?",
        strengths=[
            "opportunity identification",
            "benefit analysis",
            "positive framing",
        ],
    ),
    "pessimist": CouncilPersona(
        name="pessimist",
        description="Identifies risks, challenges, and potential problems",
        perspective="What could go wrong and what are the risks?",
        strengths=["risk assessment", "problem identification", "critical thinking"],
    ),
    "critic": CouncilPersona(
        name="critic",
        description="Evaluates weaknesses, flaws, and areas for improvement",
        perspective="What are the weaknesses and how can this be improved?",
        strengths=["quality assessment", "flaw detection", "constructive criticism"],
    ),
    "pragmatist": CouncilPersona(
        name="pragmatist",
        description="Focuses on practical implementation and realistic outcomes",
        perspective="What is realistically achievable and how do we implement it?",
        strengths=[
            "practical planning",
            "feasibility analysis",
            "implementation focus",
        ],
    ),
    "creative": CouncilPersona(
        name="creative",
        description="Generates innovative ideas and unconventional solutions",
        perspective="What are the creative and innovative approaches?",
        strengths=["ideation", "innovation", "lateral thinking"],
    ),
    "analytical": CouncilPersona(
        name="analytical",
        description="Provides data-driven analysis and logical reasoning",
        perspective="What does the data and logic tell us?",
        strengths=["data analysis", "logical reasoning", "systematic thinking"],
    ),
}


def select_council_personas(question_type: str, council_size: int = 3) -> List[str]:
    """
    Select appropriate personas for a council based on question type.

    Args:
        question_type: Type of question (decision, analysis, creative, etc.)
        council_size: Number of council members (default: 3)

    Returns:
        List of persona names
    """
    # Define persona combinations for different question types
    persona_sets = {
        "decision": ["optimist", "pessimist", "pragmatist"],
        "investment": ["optimist", "pessimist", "analytical"],
        "strategy": ["creative", "pragmatist", "analytical"],
        "problem_solving": ["creative", "critic", "pragmatist"],
        "evaluation": ["critic", "analytical", "pragmatist"],
        "innovation": ["creative", "optimist", "critic"],
        "risk_assessment": ["pessimist", "analytical", "pragmatist"],
    }

    # Get personas for question type, default to balanced council
    selected = persona_sets.get(
        question_type.lower(), ["optimist", "critic", "pragmatist"]  # Balanced default
    )

    # Adjust to requested council size
    if len(selected) > council_size:
        selected = selected[:council_size]
    elif len(selected) < council_size:
        # Add analytical if we need more
        remaining = [p for p in COUNCIL_PERSONAS.keys() if p not in selected]
        selected.extend(remaining[: council_size - len(selected)])

    logger.info(f"Selected council personas for {question_type}: {selected}")
    return selected


def get_persona_prompt_modifier(persona_name: str) -> str:
    """
    Get a prompt modifier to guide the AI's perspective for a given persona.

    Args:
        persona_name: Name of the persona

    Returns:
        Prompt modifier string
    """
    persona = COUNCIL_PERSONAS.get(persona_name)
    if not persona:
        return ""

    return f"""
You are acting as a council member with the following perspective:
- Role: {persona.name.title()}
- Focus: {persona.description}
- Key Question: {persona.perspective}
- Strengths: {', '.join(persona.strengths)}

Provide your analysis from this specific perspective, emphasizing your unique viewpoint.
"""
