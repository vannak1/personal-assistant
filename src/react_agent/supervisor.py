"""Supervisor architecture implementation for the multi-agent system.

This module provides the implementation of a supervisor-based multi-agent
system using LangGraph.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, cast
from datetime import datetime, UTC
import uuid
import json

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent, ToolNode

from react_agent.configuration import Configuration
from react_agent.memory import create_memory_for_agent, PrimaryAgentMemory, SpecializedAgentMemory
from react_agent.tool_registry import ToolRegistry
from react_agent.memory_manager import MemoryManager
from react_agent.utils import load_chat_model


def create_supervisor_system(
    model: Optional[BaseLanguageModel] = None,
    tool_registry: Optional[ToolRegistry] = None,
    memory_manager: Optional[MemoryManager] = None,
    configuration: Optional[Configuration] = None,
    checkpointer = None,
    store = None
) -> StateGraph:
    """Create a multi-agent system with a supervisor architecture.
    
    Args:
        model: Optional base language model to use (will be loaded from config if not provided)
        tool_registry: Optional tool registry (will be created if not provided)
        memory_manager: Optional memory manager (will be created if not provided)
        configuration: Optional configuration (will be loaded from context if not provided)
        checkpointer: Optional checkpointer for state persistence
        store: Optional store for additional data storage
        
    Returns:
        Compiled StateGraph for the multi-agent system
    """
    # Load configuration if not provided
    if configuration is None:
        configuration = Configuration.from_context()
    
    # Create model if not provided
    if model is None:
        model = load_chat_model(configuration.supervisor_model)
    
    # Create tool registry if not provided
    if tool_registry is None:
        from react_agent.tool_registry import create_tool_registry
        tool_registry = create_tool_registry()
    
    # Create memory manager if not provided
    if memory_manager is None:
        memory_manager = MemoryManager(checkpointer=checkpointer, store=store)
    
    # Create prompts for each agent type
    prompts = {
        "router": ChatPromptTemplate.from_messages([
            ("system", configuration.supervisor_prompt.format(system_time=datetime.now(tz=UTC).isoformat())),
            MessagesPlaceholder(variable_name="messages"),
        ]),
        
        "personal_assistant": ChatPromptTemplate.from_messages([
            ("system", configuration.personal_assistant_prompt.format(system_time=datetime.now(tz=UTC).isoformat())),
            MessagesPlaceholder(variable_name="messages"),
        ]),
        
        "research": ChatPromptTemplate.from_messages([
            ("system", configuration.deep_research_prompt.format(system_time=datetime.now(tz=UTC).isoformat())),
            MessagesPlaceholder(variable_name="messages"),
        ]),
        
        "website": ChatPromptTemplate.from_messages([
            ("system", """You are a specialized agent for website interaction and extraction.
            You can browse websites, extract information, and interact with web elements.
            Focus on accurately navigating websites and extracting the requested information.
            When your task is complete, provide the extracted information in a structured format."""),
            MessagesPlaceholder(variable_name="messages"),
        ]),
        
        "feature_request": ChatPromptTemplate.from_messages([
            ("system", configuration.feature_request_prompt.format(system_time=datetime.now(tz=UTC).isoformat())),
            MessagesPlaceholder(variable_name="messages"),
        ]),
    }
    
    # Create ReAct agents with appropriate tools and prompts
    agents = {}
    for agent_type, prompt in prompts.items():
        bound_model = tool_registry.bind_tools_to_model(model, agent_type)
        agents[agent_type] = create_react_agent(
            bound_model,
            prompt,
            name=agent_type
        )
    
    # Get tools for the ToolNode
    from react_agent.tools import TOOLS
    
    # Define agent nodes
    async def create_agent_node(agent_type: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent state based on agent type."""
        print(f"--- Running {agent_type.capitalize()} Agent ---")
        agent = agents[agent_type]
        
        # Update agent status
        state["agent_status"] = f"{agent_type}_thinking"
        
        # Invoke the agent with the current state
        response = await agent.ainvoke(state)
        
        if agent_type in ["router", "personal_assistant"]:
            # Save to long-term memory
            if memory_manager and "user_uid" in state and state["user_uid"]:
                await memory_manager.save_memory(
                    agent_type=agent_type,
                    user_id=state.get("user_uid", "unknown"),
                    content=json.dumps(response),
                    context="agent_response",
                    metadata={"agent_type": agent_type}
                )
        
        # Mark completion in agent status
        state["agent_status"] = f"{agent_type}_completed"
        
        if agent_type != "router" and not response.get("next"):
            # Non-router agents should always return to router when done
            response["next"] = "router"
        
        return response
    
    # Create the router node
    async def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return await create_agent_node("router", state)
    
    # Create specialized agent nodes
    async def personal_assistant_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return await create_agent_node("personal_assistant", state)
    
    async def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return await create_agent_node("research", state)
    
    async def website_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return await create_agent_node("website", state)
    
    async def feature_request_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return await create_agent_node("feature_request", state)
    
    # Create the tools node
    tools_node = ToolNode(TOOLS, handle_tool_errors=True)
    
    # Create system graph
    workflow = StateGraph(PrimaryAgentMemory)
    
    # Add nodes for each agent
    workflow.add_node("router", router_node)
    workflow.add_node("personal_assistant", personal_assistant_node)
    workflow.add_node("research", research_node)
    workflow.add_node("website", website_node)
    workflow.add_node("feature_request", feature_request_node)
    workflow.add_node("tools", tools_node)
    
    # Define routing functions
    def route_to_agent_or_end(state: PrimaryAgentMemory) -> str:
        """Routes based on next value or tool calls."""
        last_message = state.messages[-1] if state.messages else None
        
        # Check for tool calls
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            return "tools"
        
        # Check for explicit next routing
        next_node = state.get("next", None)
        if next_node:
            print(f"Router: Explicit routing to {next_node}")
            if next_node == "__end__":
                return END
            return next_node
        
        # Default to end
        return END
    
    def route_tools_output(state: PrimaryAgentMemory) -> str:
        """Routes tool output back to the calling agent."""
        active_agent = state.get("active_agent", None)
        print(f"Routing Tools: Active agent is '{active_agent}'")
        
        # Map state values to node names
        if active_agent == "personal_assistant":
            return "personal_assistant"
        elif active_agent == "research":
            return "research"
        elif active_agent == "website":
            return "website"
        elif active_agent == "feature_request":
            return "feature_request"
        elif active_agent == "router":
            return "router"
        else:
            print(f"WARNING: Unknown active agent '{active_agent}', routing to router")
            return "router"
    
    # Add edges for the router
    workflow.add_conditional_edges(
        "router",
        route_to_agent_or_end,
        {
            "tools": "tools",
            "personal_assistant": "personal_assistant",
            "research": "research",
            "website": "website",
            "feature_request": "feature_request",
            END: END
        }
    )
    
    # Add edges for the personal assistant
    workflow.add_conditional_edges(
        "personal_assistant",
        route_to_agent_or_end,
        {
            "tools": "tools",
            "router": "router",
            "research": "research",
            "website": "website",
            "feature_request": "feature_request",
            END: END
        }
    )
    
    # Add edges for specialized agents
    for agent_type in ["research", "website", "feature_request"]:
        workflow.add_conditional_edges(
            agent_type,
            route_to_agent_or_end,
            {
                "tools": "tools",
                "router": "router",
                "personal_assistant": "personal_assistant",
                END: END
            }
        )
    
    # Add edges for the tools node
    workflow.add_conditional_edges(
        "tools",
        route_tools_output,
        {
            "router": "router",
            "personal_assistant": "personal_assistant",
            "research": "research",
            "website": "website",
            "feature_request": "feature_request"
        }
    )
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Compile the graph with the provided checkpointer and store
    return workflow.compile(checkpointer=checkpointer, store=store)