"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Dict, Any

from langchain_core.runnables import ensure_config
from langgraph.config import get_config

from react_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the multi-agent system."""

    # System-wide settings
    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt for overall system context."
        },
    )
    
    # Agent-specific prompts
    supervisor_prompt: str = field(
        default=prompts.SUPERVISOR_PROMPT,
        metadata={
            "description": "The system prompt for the supervisor agent that routes requests."
        },
    )

    personal_assistant_prompt: str = field(
        default=prompts.PERSONAL_ASSISTANT_PROMPT,
        metadata={
            "description": "The system prompt for the personal assistant agent that handles user interaction."
        },
    )

    feature_request_prompt: str = field(
        default=prompts.FEATURE_REQUEST_PROMPT,
        metadata={
            "description": "The system prompt for the execution enforcer agent."
        },
    )

    deep_research_prompt: str = field(
        default=prompts.DEEP_RESEARCH_PROMPT,
        metadata={
            "description": "The system prompt for the deep research agent."
        },
    )
    
    # Agent-specific models
    supervisor_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4-turbo",
        metadata={
            "description": "The model used for the supervisor agent's routing decisions."
        },
    )
    
    personal_assistant_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The model used for the personal assistant agent's user interactions."
        },
    )
    
    execution_enforcer_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4-turbo",
        metadata={
            "description": "The model used for the execution enforcer agent's planning capabilities."
        },
    )
    
    deep_research_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4-turbo",
        metadata={
            "description": "The model used for the deep research agent's information gathering capabilities."
        },
    )

    web_search_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The model used for web search operations."
        },
    )
    
    # Model-specific parameters
    supervisor_model_params: Dict[str, Any] = field(
        default_factory=lambda: {"temperature": 0.2},  # Low temperature for more consistent routing
        metadata={
            "description": "Parameters for the supervisor model to optimize routing decisions."
        },
    )
    
    personal_assistant_model_params: Dict[str, Any] = field(
        default_factory=lambda: {"temperature": 0.7, "max_tokens": 1024},  # Higher temperature for conversational variety
        metadata={
            "description": "Parameters for the personal assistant model to optimize conversational abilities."
        },
    )
    
    specialist_model_params: Dict[str, Any] = field(
        default_factory=lambda: {"temperature": 0.2, "max_tokens": 2048},  # Lower temperature for analytical precision
        metadata={
            "description": "Parameters for the specialist models to optimize analytical capabilities."
        },
    )
    
    # Other configuration settings
    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )
    
    feature_requests_queue: list = field(
        default_factory=list,
        metadata={
            "description": "Queue of feature requests that have been documented and are ready for development."
        },
    )

    @classmethod
    def from_context(cls) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        try:
            config = get_config()
        except RuntimeError:
            config = None
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})