"""Human-in-the-loop capabilities for the multi-agent system.

This module provides functionality for adding human approval and feedback
nodes to the agent graph, allowing human users to be involved in
critical decision points.
"""

from __future__ import annotations

from typing import Dict, Any, Callable, Optional, List, Union
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END

from react_agent.memory import PrimaryAgentMemory


def add_human_in_the_loop(
    graph: StateGraph,
    agent_types: List[str] = None,
    approval_required_for: Optional[Dict[str, List[str]]] = None
) -> StateGraph:
    """Add human-in-the-loop nodes to the agent graph.
    
    Args:
        graph: The StateGraph to modify
        agent_types: Which agent types to add human-in-the-loop for (default: all)
        approval_required_for: Dict mapping agent types to tool names requiring approval
    
    Returns:
        Modified StateGraph with human-in-the-loop capability
    """
    agent_types = agent_types or ["router", "personal_assistant", "research", "website", "feature_request"]
    approval_required_for = approval_required_for or {}
    
    # Add human approval node
    async def human_approval_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Node that requests human approval for certain actions."""
        print("--- Running Human Approval Node ---")
        
        # Get the last AI message with tool calls
        messages = state.get("messages", [])
        
        for i in range(len(messages) - 1, -1, -1):
            if (isinstance(messages[i], AIMessage) and 
                hasattr(messages[i], "tool_calls") and 
                messages[i].tool_calls):
                
                # Check if any tool calls require approval
                agent_type = state.get("active_agent", "unknown")
                tools_needing_approval = approval_required_for.get(agent_type, [])
                
                critical_tool_calls = []
                for tool_call in messages[i].tool_calls:
                    if tool_call.get("name") in tools_needing_approval:
                        critical_tool_calls.append(tool_call)
                
                if critical_tool_calls:
                    # Format a message requesting approval
                    approval_request = HumanMessage(
                        content=f"Please approve the following action(s):\n" + 
                                "\n".join([
                                    f"- {tc['name']}: {tc.get('args', {})}" 
                                    for tc in critical_tool_calls
                                ]) +
                                "\nType 'approve' to proceed, or provide alternative instructions."
                    )
                    
                    # Return updated state with approval request
                    return {
                        "messages": messages + [approval_request], 
                        "awaiting_approval": critical_tool_calls,
                        "agent_status": "awaiting_human_approval"
                    }
        
        # No actions requiring approval
        return state
    
    async def human_feedback_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Node that processes human feedback after agent responses."""
        print("--- Running Human Feedback Node ---")
        
        # This would collect feedback on agent responses
        messages = state.get("messages", [])
        
        # Find the last AI message
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage):
                # Add a feedback request
                feedback_request = HumanMessage(
                    content="Was this response helpful? Provide any feedback or corrections."
                )
                return {
                    "messages": messages + [feedback_request], 
                    "collecting_feedback": True,
                    "agent_status": "collecting_human_feedback"
                }
                
        return state
    
    # Add the human nodes to the graph
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("human_feedback", human_feedback_node)
    
    # Add conditional edges to determine when human approval is needed
    def needs_approval(state: PrimaryAgentMemory) -> bool:
        """Check if the current action needs approval."""
        # Get the current agent type
        agent_type = state.get("active_agent", "unknown")
        
        # Get the list of tools requiring approval for this agent
        tools_needing_approval = approval_required_for.get(agent_type, [])
        
        if not tools_needing_approval:
            return False
            
        # Check the last AI message for tool calls that need approval
        messages = state.get("messages", [])
        for i in range(len(messages) - 1, -1, -1):
            if (isinstance(messages[i], AIMessage) and 
                hasattr(messages[i], "tool_calls") and 
                messages[i].tool_calls):
                
                for tool_call in messages[i].tool_calls:
                    if tool_call.get("name") in tools_needing_approval:
                        return True
                
                break  # Only check the last AI message
                
        return False
    
    def should_collect_feedback(state: PrimaryAgentMemory) -> bool:
        """Check if we should collect feedback from the human."""
        # Implementation to determine if feedback should be collected
        # This might be based on configuration, frequency settings, etc.
        
        # Example implementation: collect feedback every 5 exchanges
        exchange_count = sum(1 for msg in state.get("messages", []) 
                             if isinstance(msg, AIMessage))
        
        return exchange_count % 5 == 0 and exchange_count > 0
    
    # Process human response to approval request
    def process_human_response(state: PrimaryAgentMemory) -> str:
        """Process human response to approval request."""
        messages = state.get("messages", [])
        if not messages:
            return "continue"
            
        # Check if we're awaiting approval and the last message is from a human
        if (state.get("awaiting_approval") and 
            isinstance(messages[-1], HumanMessage)):
            
            human_response = messages[-1].content.lower()
            
            if "approve" in human_response:
                # Approval given, continue to tools
                return "approved"
            else:
                # Approval denied or alternate instructions given
                # Route back to the original agent for reconsideration
                return "reconsider"
        
        # Check if we're collecting feedback
        if (state.get("collecting_feedback") and 
            isinstance(messages[-1], HumanMessage)):
            
            # Feedback received, continue to original agent
            return "feedback_received"
            
        return "continue"
    
    # Add routing for human approval and feedback
    for agent_type in agent_types:
        # Add conditional edge from agent to human approval
        original_edges = graph.get_edges(agent_type)
        
        # For each edge, modify to possibly route through approval
        for target, condition in original_edges:
            if target == "tools":
                # Replace the direct edge to tools with conditional routing
                graph.remove_edge(agent_type, target)
                
                # Add conditional routing through approval when needed
                graph.add_conditional_edges(
                    agent_type,
                    lambda state: "human_approval" if needs_approval(state) else "tools",
                    {
                        "human_approval": "human_approval",
                        "tools": "tools"
                    },
                    condition=condition  # Preserve original condition
                )
        
        # Add conditional edge for feedback collection
        graph.add_conditional_edges(
            agent_type,
            lambda state: "human_feedback" if should_collect_feedback(state) else END,
            {
                "human_feedback": "human_feedback",
                END: END
            },
            # Only apply when the agent is about to end
            condition=lambda state: state.get("next") == END
        )
    
    # Add edges from human approval node based on response
    graph.add_conditional_edges(
        "human_approval",
        process_human_response,
        {
            "approved": "tools",  # Proceed to tools if approved
            "reconsider": lambda state: state.get("active_agent", "router"),  # Back to agent if denied
            "continue": "tools"  # Default path
        }
    )
    
    # Add edges from human feedback node
    graph.add_conditional_edges(
        "human_feedback",
        process_human_response,
        {
            "feedback_received": lambda state: state.get("active_agent", "router"),
            "continue": END
        }
    )
    
    return graph


def get_default_approval_configuration() -> Dict[str, List[str]]:
    """Get default configuration for actions requiring human approval.
    
    Returns:
        Dictionary mapping agent types to tool names requiring approval
    """
    return {
        "personal_assistant": ["manage_user_session"],
        "research": ["search"],
        "website": ["search"],
        "feature_request": ["search"]
    }