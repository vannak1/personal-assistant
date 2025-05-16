"""React Agent multi-agent personal assistant system.

This module defines a custom reasoning and action agent graph.
It implements a multi-agent system with a supervisor, personal assistant, and specialists,
featuring differentiated memory types, handoff mechanisms, optimized tool calling,
and human-in-the-loop capabilities.
"""

from react_agent.configuration import Configuration
from react_agent.state import State, InputState
from react_agent.tools import TOOLS, search_tool, manage_user_session_tool

# Import enhanced components
from react_agent.memory import (
    PrimaryAgentMemory,
    SpecializedAgentMemory,
    UserProfile,
    SessionContext,
    TaskContext,
    create_memory_for_agent
)

from react_agent.handoff import (
    create_handoff_tool,
    create_handoff_tools,
    extract_relevant_context
)

from react_agent.tool_registry import (
    ToolRegistry,
    create_tool_registry,
    TOOL_REGISTRY
)

from react_agent.memory_manager import MemoryManager

from react_agent.supervisor import create_supervisor_system

from react_agent.human_loop import (
    add_human_in_the_loop,
    get_default_approval_configuration
)

from react_agent.main import (
    create_personal_assistant,
    process_user_query
)

# Legacy graph
from react_agent.graph import (
    graph,
    personal_assistant_agent,
    features_agent,
    deep_research_agent,
    web_search_agent,
    create_enhanced_graph
)

__all__ = [
    # Configuration
    "Configuration",
    
    # State
    "State",
    "InputState",
    
    # Memory
    "PrimaryAgentMemory",
    "SpecializedAgentMemory",
    "UserProfile",
    "SessionContext",
    "TaskContext",
    "create_memory_for_agent",
    
    # Tools
    "TOOLS",
    "search_tool",
    "manage_user_session_tool",
    
    # Handoff
    "create_handoff_tool",
    "create_handoff_tools",
    "extract_relevant_context",
    
    # Tool registry
    "ToolRegistry",
    "create_tool_registry",
    "TOOL_REGISTRY",
    
    # Memory manager
    "MemoryManager",
    
    # Supervisor
    "create_supervisor_system",
    
    # Human-in-the-loop
    "add_human_in_the_loop",
    "get_default_approval_configuration",
    
    # Main exports
    "create_personal_assistant",
    "process_user_query",
    
    # Legacy graph
    "graph",
    "personal_assistant_agent",
    "features_agent",
    "deep_research_agent",
    "web_search_agent",
    "create_enhanced_graph"
]