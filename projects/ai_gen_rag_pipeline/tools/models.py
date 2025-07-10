"""
Pydantic models for structured output in the tools demonstration.
"""
from enum import Enum
from pydantic import BaseModel, Field
from typing import Any


class ToolName(str, Enum):
    """Available tool names for the demonstration."""
    WEATHER = "get_weather"
    CALCULATOR = "calculator"
    WIKIPEDIA = "wikipedia"


class ToolCall(BaseModel):
    """Represents a tool call with its arguments."""
    name: ToolName = Field(..., description="The name of the tool to call")
    args: dict[str, Any] = Field(..., description="The arguments to pass to the tool")


class StructuredOutput(BaseModel):
    """Structured output from the language model."""
    thought: str = Field(..., description="The model's thought process")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="List of tool calls to make")
    final_answer: str | None = Field(None, description="Final answer if no tool calls are needed")


class WeatherResponse(BaseModel):
    """Structured response from the weather tool."""
    location: str = Field(..., description="The location for the weather data")
    temperature: float = Field(..., description="Temperature in Celsius")
    condition: str = Field(..., description="Weather condition")
    humidity: float = Field(..., description="Humidity percentage")
    wind_speed: float = Field(..., description="Wind speed in km/h")


class CalculationResponse(BaseModel):
    """Structured response from the calculator tool."""
    expression: str = Field(..., description="The mathematical expression")
    result: float = Field(..., description="The result of the calculation")


class WikipediaResponse(BaseModel):
    """Structured response from the Wikipedia tool."""
    topic: str = Field(..., description="The topic that was searched")
    summary: str = Field(..., description="A brief summary of the topic")
    url: str = Field(..., description="URL to the full Wikipedia article")
