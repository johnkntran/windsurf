"""
Tools implementation for the demonstration.
"""
from __future__ import annotations
import json
import random

from langchain_core.tools import BaseTool, Tool
from langchain_core.pydantic_v1 import BaseModel, Field

from .models import (
    WeatherResponse,
    CalculationResponse,
    WikipediaResponse,
    ToolName,
)


class ToolManager:
    """Manages the available tools and their execution."""
    
    def __init__(self):
        self._tools = self._initialize_tools()
    
    def _initialize_tools(self) -> dict[str, Tool]:
        """Initialize and return the available tools."""
        return {
            ToolName.WEATHER: Tool(
                name=ToolName.WEATHER,
                func=self._get_weather,
                description=(
                    "Useful for getting weather information for a specific location. "
                    "Input should be a JSON string with 'location' and 'unit' (celsius/fahrenheit) keys."
                ),
                args_schema=WeatherToolInput,
            ),
            ToolName.CALCULATOR: Tool(
                name=ToolName.CALCULATOR,
                func=self._calculate,
                description=(
                    "Useful for performing mathematical calculations. "
                    "Input should be a mathematical expression as a string."
                ),
                args_schema=CalculatorInput,
            ),
            ToolName.WIKIPEDIA: Tool(
                name=ToolName.WIKIPEDIA,
                func=self._search_wikipedia,
                description=(
                    "Useful for searching information on Wikipedia. "
                    "Input should be a search query string."
                ),
                args_schema=WikipediaInput,
            ),
        }
    
    def get_tools(self) -> list[Tool]:
        """Get the list of available tools."""
        return list(self._tools.values())
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a specific tool by name."""
        return self._tools.get(name)
    
    async def execute_tool(self, tool_name: str, tool_input: str | dict[str, Any]) -> Any:
        """Execute a tool with the given input."""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError:
                pass
        
        return await tool.arun(tool_input)
    
    # Tool implementations
    
    async def _get_weather(self, location: str, unit: str = "celsius") -> dict[str, Any]:
        """Get weather information for a location (mock implementation)."""
        # In a real implementation, this would call a weather API
        weather_data = {
            "location": location,
            "temperature": random.uniform(15, 30) if unit == "celsius" else random.uniform(59, 86),
            "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Partly Cloudy"]),
            "humidity": random.uniform(30, 90),
            "wind_speed": random.uniform(0, 30),
            "unit": unit,
        }
        return WeatherResponse(**weather_data).model_dump()
    
    async def _calculate(self, expression: str) -> dict[str, Any]:
        """Evaluate a mathematical expression (mock implementation)."""
        # In a real implementation, this would use a more robust evaluation
        try:
            # WARNING: Using eval is generally unsafe, but we're controlling the input here
            # In production, use a proper expression evaluator
            result = eval(expression, {"__builtins__": {}}, {})
            return CalculationResponse(
                expression=expression,
                result=float(result) if isinstance(result, (int, float)) else result
            ).model_dump()
        except Exception as e:
            return {"error": f"Calculation error: {str(e)}"}
    
    async def _search_wikipedia(self, query: str) -> dict[str, Any]:
        """Search for information on Wikipedia (mock implementation)."""
        # In a real implementation, this would call the Wikipedia API
        return WikipediaResponse(
            topic=query,
            summary=f"This is a mock summary for '{query}'. In a real implementation, "
                   f"this would fetch actual information from Wikipedia.",
            url=f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
        ).model_dump()


# Input schemas for the tools

class WeatherToolInput(BaseModel):
    location: str = Field(..., description="The city and state, e.g., 'San Francisco, CA'")
    unit: str = Field("celsius", description="The unit of temperature, either 'celsius' or 'fahrenheit'")


class CalculatorInput(BaseModel):
    expression: str = Field(..., description="A mathematical expression to evaluate, e.g., '2 + 2' or '3 * 5 / 2'")


class WikipediaInput(BaseModel):
    query: str = Field(..., description="The search query for Wikipedia")
