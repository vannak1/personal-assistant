"""Tool registry to manage tools and their associations with agents.

This module provides a registry to manage tool availability and scoping
for different agent types, avoiding duplication and ensuring proper
tool access control.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Callable
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel

from react_agent.tools import search_tool, manage_user_session_tool
from react_agent.handoff import create_handoff_tools


class ToolRegistry:
    """Registry to manage tools and their associations with agents.
    
    This class manages the assignment of tools to specific agent types
    and ensures proper scoping of tool access.
    """
    
    def __init__(self):
        """Initialize a new tool registry."""
        self.tools_by_name = {}
        self.agent_tool_sets = {}
    
    def register_tool(self, tool: BaseTool, agent_types: Optional[List[str]] = None) -> None:
        """Register a tool and associate it with specified agent types.
        
        Args:
            tool: The tool to register
            agent_types: List of agent types that should have access to this tool
        """
        self.tools_by_name[tool.name] = tool
        
        if agent_types:
            for agent_type in agent_types:
                if agent_type not in self.agent_tool_sets:
                    self.agent_tool_sets[agent_type] = []
                self.agent_tool_sets[agent_type].append(tool)
    
    def register_tools(self, tools: List[BaseTool], agent_types: Optional[List[str]] = None) -> None:
        """Register multiple tools at once.
        
        Args:
            tools: List of tools to register
            agent_types: List of agent types that should have access to these tools
        """
        for tool in tools:
            self.register_tool(tool, agent_types)
    
    def get_tools_for_agent(self, agent_type: str) -> List[BaseTool]:
        """Get all tools registered for a specific agent type.
        
        Args:
            agent_type: The agent type to get tools for
            
        Returns:
            List of tools available to the specified agent type
        """
        return self.agent_tool_sets.get(agent_type, [])
    
    def bind_tools_to_model(self, model: BaseLanguageModel, agent_type: str) -> BaseLanguageModel:
        """Bind the appropriate tools to a language model for a specific agent type.
        
        Args:
            model: The language model to bind tools to
            agent_type: The agent type to get tools for
            
        Returns:
            The language model with bound tools
        """
        tools = self.get_tools_for_agent(agent_type)
        return model.bind_tools(tools) if tools else model


# Create and configure the tool registry
def create_tool_registry() -> ToolRegistry:
    """Create and populate the tool registry with all necessary tools.
    
    Returns:
        Fully configured ToolRegistry
    """
    registry = ToolRegistry()
    
    # Register common tools available to all agents
    common_tools = [
        search_tool,
        manage_user_session_tool,
    ]
    registry.register_tools(common_tools, ["router", "personal_assistant", "research", "website", "feature_request"])
    
    # Register handoff tools - only for router and personal assistant
    handoff_tools = create_handoff_tools()
    registry.register_tools(handoff_tools, ["router", "personal_assistant"])
    
    # Register specialized tools for specific agent types
    
    # Research agent tools
    # Will be implemented with web search and document analysis capabilities
    # But using placeholder for now
    research_tools = [
        search_tool,
    ]
    registry.register_tools(research_tools, ["research"])
    
    # Website agent tools
    # Will be implemented with web browsing and extraction capabilities
    # But using placeholder for now
    website_tools = [
        search_tool,
    ]
    registry.register_tools(website_tools, ["website"])
    
    # Feature request agent tools
    # Will be implemented with code analysis and requirements tools
    # But using placeholder for now
    feature_request_tools = [
        search_tool,
    ]
    registry.register_tools(feature_request_tools, ["feature_request"])
    
    # Personal assistant specific tools (in addition to common tools)
    # Will be implemented with calendar, email tools
    # But using placeholder for now
    personal_assistant_tools = [
        search_tool,
    ]
    registry.register_tools(personal_assistant_tools, ["personal_assistant"])
    
    return registry


# Export for easy access
TOOL_REGISTRY = create_tool_registry()