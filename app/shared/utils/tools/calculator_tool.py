from typing import Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="The mathematical expression to evaluate (e.g., '2 + 2', '3 * 5')"
    )


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = (
        "Useful for performing mathematical calculations. Input should be a valid mathematical expression."
    )
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """Use the tool."""
        try:
            # Safe evaluation of mathematical expressions
            # Using eval() is generally unsafe, but for a calculator tool in a controlled env it's often used.
            # A safer approach would be to use a library like numexpr or simpleeval.
            # For this demo, we'll strip potentially dangerous characters.
            allowed_chars = "0123456789+-*/().% "
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression. Only numbers and basic operators allowed."

            return str(eval(expression))
        except Exception as e:
            return f"Error calculating {expression}: {str(e)}"

    async def _arun(self, expression: str) -> str:
        """Use the tool asynchronously."""
        # For simple calculation, sync run is fine
        return self._run(expression)
