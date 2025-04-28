"""Define a hierarchical multi-agent system.

This implements a supervisor agent that routes requests to specialized agent teams:
1. Feature Request Agent: Documents feature requests and pain points
2. Deep Research Agent: Provides in-depth research on complex topics
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, TypedDict, cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import State, InputState
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model


# Supervisor Agent Implementation
async def supervisor_agent(state: State) -> Dict:
    """
    Supervisor agent that analyzes the user query and routes to the appropriate specialized agent.
    """
    configuration = Configuration.from_context()
    
    last_message = state.messages[-1]
    
    # If the last message is an AIMessage from a completed sub-agent, route to end.
    if isinstance(last_message, AIMessage) and state.active_agent is None:
        print("Supervisor: Agent finished. Routing to __end__.")
        return {"next": "__end__"}

    # If it's a ToolMessage, it means tools ran but routing failed somehow. End flow.
    if isinstance(last_message, ToolMessage):
         print("Supervisor: Received unexpected ToolMessage. Routing to __end__.")
         return {"next": "__end__"}

    # If it's not a HumanMessage at this point, end the flow.
    if not isinstance(last_message, HumanMessage):
         print(f"Supervisor: Received unexpected message type {type(last_message)}. Routing to __end__.")
         return {"next": "__end__"}

    # It's a HumanMessage, proceed with routing
    print("Supervisor: Routing HumanMessage.")
    model = load_chat_model(configuration.model)
    system_message = configuration.supervisor_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )
    # Use invoke, assuming supervisor doesn't need async here, adjust if needed
    response = model.invoke([
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Please analyze this request and route to the appropriate team: {last_message.content}"}
    ])
    routing_decision = response.content.lower()

    if "feature request" in routing_decision or "pain point" in routing_decision:
        print("Supervisor: Routing to Feature Request Agent.")
        return {"next": "feature_request_agent", "active_agent": "feature_request"}
    else:
        print("Supervisor: Routing to Deep Research Agent.")
        # Default to deep research if not clearly a feature request
        return {"next": "deep_research_agent", "active_agent": "deep_research"}


# Feature Request Agent Implementation
async def feature_request_agent(state: State) -> Dict:
    """
    Feature request agent that documents pain points and feature requests.
    Calls LLM, updates state with AIMessage. Routing decided by edges based on tool_calls.
    """
    print("--- Running Feature Request Agent ---")
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model).bind_tools(TOOLS)
    system_message = configuration.feature_request_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )
    
    # Filter messages to avoid passing supervisor routing instructions
    agent_messages = [msg for msg in state.messages if not getattr(msg, 'name', None) == 'supervisor'] 

    response = cast(
        AIMessage,
        await model.ainvoke(
             [{"role": "system", "content": system_message}, *agent_messages] 
        ),
    )

    # Check if the agent is done (no tool calls)
    if not response.tool_calls:
         print("Feature Request Agent: Finished (no tool calls).")
         # Add feature request to the queue (simple in-memory version)
         feature_request = {
             "description": response.content,
             "timestamp": datetime.now(tz=UTC).isoformat(),
             "status": "queued"
         }
         # Ensure feature_requests is initialized if None
         current_requests = state.feature_requests if state.feature_requests is not None else []
         updated_requests = current_requests + [feature_request]
         
         # Signal completion by setting active_agent to None
         # The 'next' state for routing back to supervisor is handled by the edge logic
         return {
             "messages": [response], 
             "feature_requests": updated_requests,
             "active_agent": None # Mark as done
        }
    else:
        # Agent needs to use tools
        print("Feature Request Agent: Requesting tools.")
        # Keep active_agent set, return message with tool calls
        return {"messages": [response]} 


# Deep Research Agent Implementation
async def deep_research_agent(state: State) -> Dict:
    """
    Deep research agent that provides in-depth information on complex topics.
    Calls LLM, updates state with AIMessage. Routing decided by edges based on tool_calls.
    """
    print("--- Running Deep Research Agent ---")
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model).bind_tools(TOOLS)
    system_message = configuration.deep_research_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Filter messages
    agent_messages = [msg for msg in state.messages if not getattr(msg, 'name', None) == 'supervisor']

    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *agent_messages] 
        ),
    )

    # Check if the agent is done (no tool calls)
    if not response.tool_calls:
        print("Deep Research Agent: Finished (no tool calls).")
        # Signal completion
        return {
            "messages": [response], 
            "active_agent": None # Mark as done
        }
    else:
        # Agent needs to use tools
        print("Deep Research Agent: Requesting tools.")
        # Keep active_agent set, return message with tool calls
        return {"messages": [response]} 


# Build the hierarchical multi-agent graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add the nodes
builder.add_node("supervisor", supervisor_agent)
builder.add_node("feature_request_agent", feature_request_agent)
builder.add_node("deep_research_agent", deep_research_agent)
builder.add_node("tools", ToolNode(TOOLS))

# Entry point
builder.add_edge("__start__", "supervisor")

# Supervisor routing edge
builder.add_conditional_edges(
    "supervisor",
    # The supervisor node returns the name of the next node (agent or __end__) in the 'next' field
    lambda state: state.next, 
    {
        "feature_request_agent": "feature_request_agent",
        "deep_research_agent": "deep_research_agent",
        "__end__": "__end__"
    }
)

# Conditional edges from agents based on tool calls in the last message
def route_agent_output(state: State) -> Literal["tools", "supervisor"]:
    """Routes agent output to tools if tool calls exist, otherwise back to supervisor."""
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            print("Agent output: Has tool calls. Routing to tools.")
            return "tools"
        else:
            # Agent finished its turn without tool calls
            print("Agent output: No tool calls. Routing to supervisor.")
            return "supervisor" 
    # Should not happen in normal flow after an agent runs
    print("Agent output: Last message not AIMessage. Routing to supervisor (fallback).")
    return "supervisor" 

builder.add_conditional_edges(
    "feature_request_agent",
    route_agent_output,
    {"tools": "tools", "supervisor": "supervisor"}
)

builder.add_conditional_edges(
    "deep_research_agent",
    route_agent_output,
    {"tools": "tools", "supervisor": "supervisor"}
)


# Edge from tools back to the active agent
def route_tools_output(state: State) -> Literal["feature_request_agent", "deep_research_agent"]:
    """Routes tool output back to the agent that called the tools."""
    active_agent = state.active_agent
    print(f"Tools output: Routing back to active agent: {active_agent}")
    if active_agent == "feature_request":
        return "feature_request_agent"
    elif active_agent == "deep_research":
        return "deep_research_agent"
    else:
        # This indicates a state error, potentially end the graph or raise an error
        print(f"ERROR: No active agent found after tool execution. State: {state}")
        # Returning supervisor as a fallback, but this should be investigated
        # raise ValueError("No active agent found after tool execution.") 
        return "supervisor" # Fallback to prevent crash, but indicates an issue

builder.add_conditional_edges(
    "tools",
    route_tools_output,
    {
        "feature_request_agent": "feature_request_agent",
        "deep_research_agent": "deep_research_agent",
        "supervisor": "supervisor" # Include fallback route
    }
)

# Compile the graph
graph = builder.compile(name="Hierarchical Assistant")

# Optional: Print graph structure for debugging
# try:
#     from langchain_core.runnables.graph import print_ascii
#     print_ascii(graph)
# except ImportError:
#     print("Install 'pip install runnable-graph' to print the graph structure.")
