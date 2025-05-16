"""Enhanced handoff mechanism for transferring control between agents.

This module provides tools and functions for transferring control between
different specialized agents with appropriate context extraction.
"""

from __future_ import annotations

from typing import Annotated, Dict, Any, List, Optional
from datetime import datetime
import uuid
import re
import json

from langchain_core.tools import tool, BaseTool
from langchain_core.messages import ToolMessage, HumanMessage, BaseMessage
from langgraph.prebuilt import InjectedState
# InjectedToolCallId was removed or moved; using tool_call_id: str as per LangGraph updates
from langgraph.types import Command # Ensure Command is imported if not already

from react_agent.memory import TaskContext, UserProfile, SessionContext


def create_handoff_tool(agent_name: str, description: Optional[str] = None) -> BaseTool:
    """Create a handoff tool that transfers control to another agent.
    
    Args:
        agent_name: The name of the agent to transfer control to
        description: Optional custom description for the tool
        
    Returns:
        A BaseTool that can be used to transfer control
    """
    
    name = f"transfer_to_{agent_name}"
    desc = description or f"Transfer control to the {agent_name} agent to handle specialized tasks."
    
    @tool(name=name, description=desc)
    def handoff_to_agent(
        task_description: str,
        state: Annotated[Dict[str, Any], InjectedState],
        tool_call_id: str,  # Using regular string as InjectedToolCallId is not available
    ):
        """Transfer the current task to another agent with relevant context.
        
        Args:
            task_description: Description of the task to transfer
            state: Current state (injected)
            tool_call_id: Tool call ID
            
        Returns:
            Command object to specify next node and update state
        """
        
        tool_message = ToolMessage(
            content=f"Transferring to {agent_name} agent to handle: {task_description}",
            name=name,
            tool_call_id=tool_call_id,
        )
        
        # Extract only relevant context based on agent type
        relevant_context = extract_relevant_context(state, agent_name, task_description)
        
        # Return Command object to specify next node and update state
        return Command(
            goto=agent_name,
            update={
                "messages": state["messages"] + [tool_message],
                "task_context": {
                    "query_details": task_description,
                    "intermediate_results": [],
                    "start_time": datetime.now().timestamp(),
                    **relevant_context
                },
                "active_agent": agent_name,
                "routing_reason": f"transferred_to_{agent_name}"
            }
        )
    
    return handoff_to_agent


def extract_relevant_context(state: Dict[str, Any], target_agent: str, task_description: str) -> Dict[str, Any]:
    """Extract only the relevant context needed by the target agent.
    
    Args:
        state: Current state of the conversation
        target_agent: The agent to extract context for
        task_description: Description of the task being transferred
        
    Returns:
        Dictionary of relevant context for the target agent
    """
    
    relevant_context = {}
    
    # Current user information if available
    if "user_profile" in state and state["user_profile"]:
        relevant_context["user_preferences"] = state["user_profile"].get("preferences", {})
    
    # Add user name if available
    if "user_name" in state and state["user_name"]:
        relevant_context["user_name"] = state["user_name"]
    
    # Extract specific context based on agent type
    if target_agent == "research":
        # For research agent, extract search queries and previous related results
        relevant_context["search_queries"] = extract_search_queries_from_messages(
            state.get("messages", []),
            task_description
        )
        
    elif target_agent == "website":
        # For website agent, extract URLs and navigation history
        relevant_context["urls"] = extract_urls_from_messages(
            state.get("messages", []),
            task_description
        )
        
    elif target_agent == "feature_request":
        # For feature request agent, extract feature descriptions and specifications
        relevant_context["feature_specs"] = extract_feature_specs(
            state.get("messages", []),
            task_description
        )
    
    # Add reference to primary context if needed
    if "session_context" in state:
        relevant_context["primary_context_ref"] = state.get("user_uid", str(uuid.uuid4()))
    
    return relevant_context


# Helper functions for context extraction
def extract_search_queries_from_messages(messages: List[BaseMessage], task_description: str) -> List[str]:
    """Extract potential search queries from message history and task description.
    
    Args:
        messages: List of conversation messages
        task_description: Description of the task being transferred
        
    Returns:
        List of potential search queries
    """
    queries = []
    
    # Extract from task description
    if task_description:
        # Basic extraction logic 
        queries.append(task_description)
    
    # Analyze recent messages for potential queries
    for message in messages[-5:]:  # Consider last 5 messages
        if hasattr(message, "content") and isinstance(message.content, str):
            # Simple extraction for question-like content
            if "?" in message.content or "search" in message.content.lower():
                queries.append(message.content)
    
    return queries[:3]  # Return top 3 potential queries


def extract_urls_from_messages(messages: List[BaseMessage], task_description: str) -> List[str]:
    """Extract URLs from message history and task description.
    
    Args:
        messages: List of conversation messages
        task_description: Description of the task being transferred
        
    Returns:
        List of URLs found in the messages and task description
    """
    # Simple URL regex pattern
    url_pattern = r'https?://\S+'
    urls = []
    
    # Extract from task description
    if task_description:
        urls.extend(re.findall(url_pattern, task_description))
    
    # Extract from recent messages
    for message in messages[-5:]:  # Last 5 messages
        if hasattr(message, "content") and isinstance(message.content, str):
            urls.extend(re.findall(url_pattern, message.content))
    
    return list(set(urls))  # Remove duplicates


def extract_feature_specs(messages: List[BaseMessage], task_description: str) -> Dict[str, Any]:
    """Extract feature specifications from messages and task description.
    
    Args:
        messages: List of conversation messages
        task_description: Description of the task being transferred
        
    Returns:
        Dictionary containing feature specifications
    """
    feature_specs = {
        "description": task_description,
        "requirements": [],
        "acceptance_criteria": []
    }
    
    # Extract specific requirements from recent messages
    for message in messages[-5:]:
        if hasattr(message, "content") and isinstance(message.content, str):
            content = message.content.lower()
            
            # Look for requirement-like statements
            if "should" in content or "must" in content or "need" in content:
                sentences = re.split(r'[.!?]+', content)
                for sentence in sentences:
                    if "should" in sentence or "must" in sentence or "need" in sentence:
                        feature_specs["requirements"].append(sentence.strip())
    
    return feature_specs


# Create handoff tools for each agent type
def create_handoff_tools() -> List[BaseTool]:
    """Create handoff tools for all agent types.
    
    Returns:
        List of handoff tools
    """
    handoff_tools = [
        create_handoff_tool("personal_assistant", 
                           "Transfer to the Personal Assistant agent for user interaction, simple queries, and result presentation."),
        create_handoff_tool("research", 
                           "Transfer to the Research agent for in-depth information gathering and analysis."),
        create_handoff_tool("website", 
                           "Transfer to the Website agent for web interactions and information extraction."),
        create_handoff_tool("feature_request", 
                           "Transfer to the Feature Request agent for planning and documenting feature requests.")
    ]
    
    return handoff_tools