"""
Simplified LangChain tools without complex field dependencies.
"""
import logging
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""
    expression: str = Field(description="Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')")


class CalculatorTool(BaseTool):
    """Simple calculator tool for basic math operations."""
    
    name: str = "calculator"
    description: str = "Calculate mathematical expressions. Use this for basic arithmetic like addition, subtraction, multiplication, and division."
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args_schema = CalculatorInput
    
    def _run(self, expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            logger.info(f"🔢 Calculating: '{expression}'")
            
            # Simple safety check - only allow basic math operations
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return f"Error: Invalid characters in expression '{expression}'. Only basic math operations are allowed."
            
            # Evaluate the expression
            result = eval(expression)
            
            logger.info(f"✅ Calculation result: {expression} = {result}")
            return f"Calculation result: {expression} = {result}"
            
        except Exception as e:
            logger.error(f"❌ Error calculating '{expression}': {str(e)}")
            return f"Error calculating '{expression}': {str(e)}"
