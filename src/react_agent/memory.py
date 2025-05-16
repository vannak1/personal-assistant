"""Define differentiated memory structures for various agent types.

This module provides memory classes tailored to different agent types:
1. Primary agents (Router and Personal Assistant) - rich context and long-term memory
2. Specialized agents (Website, Research, Feature Request) - task-specific context
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from typing_extensions import NotRequired
from datetime import datetime
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState


class UserProfile(TypedDict):
    """User profile information stored in primary agent memory."""
    preferences: Dict[str, Any]
    frequent_requests: List[str]
    last_interactions: List[Dict[str, Any]]


class SessionContext(TypedDict):
    """Session context information stored in primary agent memory."""
    current_topic: str
    active_agents: List[str]
    pending_tasks: List[Any]


class PrimaryAgentMemory(MessagesState):
    """Memory structure for Router and Personal Assistant agents.
    
    These agents need rich context and long-term memory to manage
    user interactions and coordinate other specialized agents.
    """
    
    messages: List[BaseMessage]
    user_profile: NotRequired[UserProfile]
    session_context: NotRequired[SessionContext]
    
    # Additional fields from existing State class
    user_name: NotRequired[Optional[str]]
    user_uid: NotRequired[Optional[str]]
    agent_status: NotRequired[Optional[str]]
    routing_reason: NotRequired[Optional[str]]
    specialist_results: NotRequired[bool]
    first_message: NotRequired[bool]
    original_question: NotRequired[Optional[str]]
    conversation_context: NotRequired[Dict[str, Any]]
    session_state: NotRequired[Optional[str]]


class TaskContext(TypedDict):
    """Task-specific context for specialized agents."""
    query_details: Any
    intermediate_results: List[Any]
    start_time: float
    primary_context_ref: NotRequired[str]


class SpecializedAgentMemory(MessagesState):
    """Memory structure for specialized agents.
    
    These agents only need task-specific context rather than 
    full user history and preferences.
    """
    
    messages: List[BaseMessage]
    task_context: NotRequired[TaskContext]
    
    # Minimal fields from primary context needed for task execution
    active_agent: NotRequired[Optional[str]]
    routing_reason: NotRequired[Optional[str]]


def create_memory_for_agent(agent_type: str) -> Dict[str, Any]:
    """Create appropriate initial memory state for different agent types.
    
    Args:
        agent_type: The type of agent to create memory for
            (router, personal_assistant, website, research, feature_request)
            
    Returns:
        A dictionary representing the initial memory state
    """
    
    base_memory = {"messages": []}
    
    if agent_type in ["router", "personal_assistant"]:
        return {
            **base_memory,
            "user_profile": {
                "preferences": {},
                "frequent_requests": [],
                "last_interactions": []
            },
            "session_context": {
                "current_topic": "",
                "active_agents": [],
                "pending_tasks": []
            },
            "user_name": None,
            "user_uid": None,
            "agent_status": None,
            "routing_reason": None,
            "specialist_results": False,
            "first_message": True,
            "original_question": None,
            "conversation_context": {},
            "session_state": None
        }
    elif agent_type in ["website", "research", "feature_request"]:
        return {
            **base_memory,
            "task_context": {
                "query_details": None,
                "intermediate_results": [],
                "start_time": datetime.now().timestamp(),
                "primary_context_ref": None
            },
            "active_agent": agent_type,
            "routing_reason": None
        }
    else:
        return base_memory