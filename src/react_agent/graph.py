"""Define a hierarchical multi-agent system.

This implements a supervisor agent that routes requests to specialized agent teams:
1. Feature Request Agent: Documents feature requests and pain points
2. Deep Research Agent: Provides in-depth research on complex topics
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, TypedDict, cast

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import State, InputState
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model


class Command(TypedDict, total=False):
    """A command to update the state and possibly go to a specified node."""

    update: Dict
    goto: str


# Supervisor Agent Implementation
async def supervisor_agent(state: State) -> Dict:
    """
    Supervisor agent that analyzes the user query and routes to the appropriate specialized agent.
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        Dict: State update with routing information.
    """
    configuration = Configuration.from_context()
    
    # Only process if the last message is from the user
    last_message = state.messages[-1]
    if not isinstance(last_message, HumanMessage):
        # If it's a response from one of our agents, just pass it through to the user
        return {"next": "__end__"}
    
    # Initialize the model for the supervisor agent
    model = load_chat_model(configuration.model)
    
    # Format the supervisor prompt
    system_message = configuration.supervisor_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )
    
    # Prepare messages for the supervisor agent
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Please analyze this request and route to the appropriate team: {last_message.content}"}
    ]
    
    # Get the supervisor's decision
    response = await model.ainvoke(messages)
    routing_decision = response.content.lower()
    
    # Route based on keywords in the response
    if "feature request" in routing_decision or "pain point" in routing_decision:
        return {"next": "feature_request_agent", "active_agent": "feature_request"}
    else:
        # Default to deep research if not clearly a feature request
        return {"next": "deep_research_agent", "active_agent": "deep_research"}


# Feature Request Agent Implementation
async def feature_request_agent(state: State) -> Command:
    """
    Feature request agent that documents pain points and feature requests.
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        Command: State update with agent response.
    """
    configuration = Configuration.from_context()
    
    # Initialize the model with tool binding
    model = load_chat_model(configuration.model).bind_tools(TOOLS)
    
    # Format the feature request prompt
    system_message = configuration.feature_request_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )
    
    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )
    
    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "update": {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="I couldn't complete processing this feature request in the specified number of steps.",
                    )
                ]
            },
            "goto": "supervisor"
        }
    
    # If there are tool calls, we need to execute them and continue the conversation
    if response.tool_calls:
        return {
            "update": {"messages": [response]},
            "goto": "tools"
        }
    else:
        # If the model has finished, capture any feature request data
        # This is a simplified approach - in a real system, you'd store this in a persistent database
        if "feature request" in response.content.lower() or "requirements" in response.content.lower():
            # Extract feature request details - in production, use a more robust approach
            feature_request = {
                "description": response.content,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "status": "queued"
            }
            
            current_requests = state.feature_requests.copy()
            current_requests.append(feature_request)
            
            return {
                "update": {
                    "messages": [response],
                    "feature_requests": current_requests
                },
                "goto": "supervisor"
            }
        
        # Just return the response if no feature request identified
        return {
            "update": {"messages": [response]},
            "goto": "supervisor"
        }


# Deep Research Agent Implementation
async def deep_research_agent(state: State) -> Command:
    """
    Deep research agent that provides in-depth information on complex topics.
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        Command: State update with agent response.
    """
    configuration = Configuration.from_context()
    
    # Initialize the model with tool binding
    model = load_chat_model(configuration.model).bind_tools(TOOLS)
    
    # Format the deep research prompt
    system_message = configuration.deep_research_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )
    
    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )
    
    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "update": {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="I couldn't complete my deep research in the specified number of steps.",
                    )
                ]
            },
            "goto": "supervisor"
        }
    
    # If there are tool calls, we need to execute them and continue the conversation
    if response.tool_calls:
        return {
            "update": {"messages": [response]},
            "goto": "tools"
        }
    else:
        return {
            "update": {"messages": [response]},
            "goto": "supervisor"
        }


# Build the hierarchical multi-agent graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add the supervisor and specialized agent nodes
builder.add_node("supervisor", supervisor_agent)
builder.add_node("feature_request_agent", feature_request_agent)
builder.add_node("deep_research_agent", deep_research_agent)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as the supervisor
builder.add_edge("__start__", "supervisor")

# Add conditional edges from the supervisor to the specialized agents
builder.add_conditional_edges(
    "supervisor",
    lambda state: state.next,
    {
        "feature_request_agent": "feature_request_agent",
        "deep_research_agent": "deep_research_agent",
        "__end__": "__end__"
    }
)

# Add edges from specialized agents back to supervisor
builder.add_edge("feature_request_agent", "tools")
builder.add_edge("deep_research_agent", "tools")
builder.add_edge("tools", "supervisor")

# Compile the graph
graph = builder.compile(name="Hierarchical Assistant")
