"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated

from langchain_core.runnables import ensure_config
from langgraph.config import get_config

from react_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    personal_assistant_prompt: str = field(
        default=prompts.PERSONAL_ASSISTANT_PROMPT,
        metadata={
            "description": "The system prompt for the main Personal Assistant agent. "
            "This agent handles user interaction, rapport building, and routing."
        },
    )

    # supervisor_prompt: str = field(
    #     default=prompts.SUPERVISOR_PROMPT,
    #     metadata={
    #         "description": "[DEPRECATED] The system prompt for the supervisor agent that routes requests. "
    #         "This prompt determines how requests are routed to specialized agents."
    #     },
    # )

    feature_request_prompt: str = field(
        default=prompts.FEATURE_REQUEST_PROMPT,
        metadata={
            "description": "The system prompt for the feature request agent (Execution Enforcer). "
            "This agent identifies and documents feature requests."
        },
    )

    deep_research_prompt: str = field(
        default=prompts.DEEP_RESEARCH_PROMPT,
        metadata={
            "description": "The system prompt for the deep research agent. "
            "This agent provides thorough research on complex topics."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-mini",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

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
