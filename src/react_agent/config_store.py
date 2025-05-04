"""Configuration store for the multi-agent system."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, ClassVar, Type
from typing_extensions import Annotated

from langgraph.config import ConfigurableFieldSpec


@dataclass
class ConfigStore:
    """Configuration store for the multi-agent system."""
    
    # System-wide settings
    system_prompt: Annotated[
        str,
        ConfigurableFieldSpec(
            description="The system prompt for overall system context."
        )
    ] = "You are a highly efficient AI Personal Assistant system, grounded in the principles of Kaizen."
    
    # Agent-specific prompts
    supervisor_prompt: Annotated[
        str,
        ConfigurableFieldSpec(
            description="The system prompt for the supervisor agent that routes requests."
        )
    ] = "You are the Supervisor Agent in a hierarchical multi-agent system."
    
    personal_assistant_prompt: Annotated[
        str,
        ConfigurableFieldSpec(
            description="The system prompt for the personal assistant agent that handles user interaction."
        )
    ] = "You are a friendly, conversational Personal Assistant AI named Kaizen Assistant."
    
    feature_request_prompt: Annotated[
        str,
        ConfigurableFieldSpec(
            description="The system prompt for the execution enforcer agent."
        )
    ] = "You are an Execution Enforcer Agent responsible for transforming user ideas into actionable implementation plans."
    
    deep_research_prompt: Annotated[
        str,
        ConfigurableFieldSpec(
            description="The system prompt for the deep research agent."
        )
    ] = "You are a Deep Research Specialist tasked with providing comprehensive, accurate information on complex topics."
    
    # Agent-specific models
    supervisor_model: Annotated[
        str,
        ConfigurableFieldSpec(
            description="The model used for the supervisor agent's routing decisions.",
            template_metadata={"kind": "llm"}
        )
    ] = "deepseek/deepseek-v3"
    
    personal_assistant_model: Annotated[
        str,
        ConfigurableFieldSpec(
            description="The model used for the personal assistant agent's user interactions.",
            template_metadata={"kind": "llm"}
        )
    ] = "openai/gpt-4o"
    
    execution_enforcer_model: Annotated[
        str,
        ConfigurableFieldSpec(
            description="The model used for the execution enforcer agent's planning capabilities.",
            template_metadata={"kind": "llm"}
        )
    ] = "openai/gpt-4-turbo"
    
    deep_research_model: Annotated[
        str,
        ConfigurableFieldSpec(
            description="The model used for the deep research agent's information gathering capabilities.",
            template_metadata={"kind": "llm"}
        )
    ] = "openai/gpt-4-turbo"
    
    # Model-specific parameters
    supervisor_model_params: Annotated[
        Dict[str, Any],
        ConfigurableFieldSpec(
            description="Parameters for the supervisor model to optimize routing decisions."
        )
    ] = field(default_factory=lambda: {"temperature": 0.2})
    
    personal_assistant_model_params: Annotated[
        Dict[str, Any],
        ConfigurableFieldSpec(
            description="Parameters for the personal assistant model to optimize conversational abilities."
        )
    ] = field(default_factory=lambda: {"temperature": 0.7, "max_tokens": 1024})
    
    specialist_model_params: Annotated[
        Dict[str, Any],
        ConfigurableFieldSpec(
            description="Parameters for the specialist models to optimize analytical capabilities."
        )
    ] = field(default_factory=lambda: {"temperature": 0.2, "max_tokens": 2048})
    
    # Other configuration settings
    max_search_results: Annotated[
        int,
        ConfigurableFieldSpec(
            description="The maximum number of search results to return for each search query."
        )
    ] = 10
    
    # Static class configuration
    CONFIG_KEY: ClassVar[str] = "kaizen_assistant"
    CONFIG_TYPE: ClassVar[Type] = Dict[str, Any]