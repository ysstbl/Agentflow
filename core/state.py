import operator
from typing import Annotated, Literal, TypedDict, Optional
from pydantic import BaseModel, Field


class Classification(BaseModel):
    """A single routing decision: which agent to call with what query."""
    source: Literal["code", "search", "document"]
    query: str

class ClassificationResult(BaseModel):
    """Result of classifying a user query into agent-specific sub-questions."""
    classifications: list[Classification] = Field(
        description="List of agents to invoke with their targeted sub-questions"
    )

class AgentInput(TypedDict):
    """Simple input state for each subagent."""
    query: str
    file_path: Optional[str]

class AgentOutput(TypedDict):
    """Output from each subagent."""
    source: str
    result: str

class RouterState(TypedDict):
    query: str
    file_path: Optional[str]
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]  # Reducer collects parallel results
    final_answer: str
