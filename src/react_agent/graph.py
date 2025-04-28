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


# Personal Assistant Agent Implementation (Replaces Supervisor)
async def personal_assistant_agent(state: State) -> Dict:
    \"\"\"
    The main agent interacting with the user, grounded in Kaizen.
    Handles simple requests directly, builds rapport, and routes complex requests
    to specialized agents (Execution Enforcer, Deep Research).
    \"\"\"
    print("--- Running Personal Assistant Agent ---")
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model).bind_tools(TOOLS) # Bind tools for direct handling
    
    last_message = state.messages[-1]

    # If the last message is an AIMessage from a completed specialist agent, synthesize and respond.
    if isinstance(last_message, AIMessage) and state.active_agent is None and last_message.name != "PersonalAssistant":
        print("PA: Specialist agent finished. Synthesizing response.")
        # Simple synthesis for now: just pass the specialist's message through.
        # In a more advanced version, the PA could rephrase or add context.
        # We need to return a new AIMessage attributed to the PA.
        synthesis_prompt = f\"A specialist provided this information: '{last_message.content}'. Please present this back to the user in a helpful and friendly tone, maintaining your Personal Assistant persona.\"
        
        pa_model = load_chat_model(configuration.model) # Use a non-tool-bound model for synthesis
        final_response = await pa_model.ainvoke([
            {\"role\": \"system\", \"content\": configuration.personal_assistant_prompt.format(system_time=datetime.now(tz=UTC).isoformat())},
            {\"role\": \"user\", \"content\": synthesis_prompt}
        ])
        
        return {
            "messages": [AIMessage(content=final_response.content, name="PersonalAssistant")],
            "next": "__end__" # End the flow after synthesizing
        }

    # If it's a ToolMessage, route back to the PA itself if it called the tool
    if isinstance(last_message, ToolMessage) and state.active_agent == "personal_assistant":
        print("PA: Received tool result. Continuing.")
        # The graph edge will route back to this node based on active_agent
        # Need to call the model again with the tool result
        system_message = configuration.personal_assistant_prompt.format(
            system_time=datetime.now(tz=UTC).isoformat()
        )
        response = cast(
            AIMessage,
            await model.ainvoke(
                [{"role": "system", "content": system_message}, *state.messages]
            ),
        )
        response.name = "PersonalAssistant" # Ensure response is named
        # Decide next step based on the new response
        if not response.tool_calls:
            print("PA: Finished handling directly after tool use.")
            return {"messages": [response], "active_agent": None, "next": "__end__"}
        else:
            print("PA: Needs more tools after initial tool use.")
            # Keep active_agent as PA, graph edge will route to tools
            return {"messages": [response]}

    # If it's not a HumanMessage at this point (and not a handled AI/Tool message), end the flow.
    if not isinstance(last_message, HumanMessage):
         print(f"PA: Received unexpected message type {type(last_message)} or state. Routing to __end__.")
         return {"next": "__end__"}

    # It's a new HumanMessage, PA needs to decide how to handle it.
    print("PA: Processing new HumanMessage.")
    system_message = configuration.personal_assistant_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )
    
    # Add a specific instruction for the PA to decide routing or direct handling
    decision_prompt = f\"User request: '{last_message.content}'.\n\nBased on this request and our conversation history, decide the best course of action:\n1. Handle Directly: If it's a simple request, greeting, or question you can answer, generate the response directly. You can use tools if needed.
2. Route to Execution Enforcer: If it involves planning, feature requests, or turning ideas into actions.
3. Route to Deep Research: If it requires in-depth information, complex explanations, or research.

Respond ONLY with 'HANDLE_DIRECTLY', 'ROUTE_EXECUTION', or 'ROUTE_RESEARCH'. Do not add any other text.\"
    
    # Use a separate, non-tool-bound model for the routing decision to avoid premature tool calls
    decision_model = load_chat_model(configuration.model)
    decision_response = await decision_model.ainvoke([
        {\"role\": \"system\", \"content\": system_message},
        *state.messages[:-1], # History without the latest user message
        {\"role\": \"user\", \"content\": decision_prompt}
    ])
    decision = decision_response.content.strip()
    print(f"PA Decision: {decision}")

    if "ROUTE_EXECUTION" in decision:
        print("PA: Routing to Execution Enforcer.")
        # Add a message indicating handover (optional, for clarity)
        # handover_msg = AIMessage(content="Let me consult our planning specialist for this.", name="PersonalAssistant")
        return {
            # "messages": [handover_msg],
             "next": "execution_enforcer_agent", 
             "active_agent": "execution_enforcer"
        }
    elif "ROUTE_RESEARCH" in decision:
        print("PA: Routing to Deep Research Agent.")
        # handover_msg = AIMessage(content="Let me bring in our research expert to look into this.", name="PersonalAssistant")
        return {
            # "messages": [handover_msg],
            "next": "deep_research_agent", 
            "active_agent": "deep_research"
        }
    else: # HANDLE_DIRECTLY (default)
        print("PA: Handling request directly.")
        # Call the main model (with tools) to generate the direct response
        response = cast(
            AIMessage,
            await model.ainvoke(
                [{"role": "system", "content": system_message}, *state.messages]
            ),
        )
        response.name = "PersonalAssistant" # Name the response
        
        if not response.tool_calls:
            print("PA: Finished handling directly.")
            # No tools needed, respond and end
            return {"messages": [response], "active_agent": None, "next": "__end__"}
        else:
            print("PA: Handling directly, needs tools.")
            # Tools needed, set PA as active agent and let edge route to tools
            return {"messages": [response], "active_agent": "personal_assistant"}


# Feature Request Agent Implementation (Execution Enforcer)
async def execution_enforcer_agent(state: State) -> Dict:
    \"\"\"
    Execution Enforcer Agent: Turns ideas into actionable plans.
    Renamed from feature_request_agent for clarity.
    \"\"\"
    print("--- Running Execution Enforcer Agent ---")
    configuration = Configuration.from_context()
    # Use the specific prompt for this agent
    model = load_chat_model(configuration.model).bind_tools(TOOLS)
    system_message = configuration.feature_request_prompt.format( # Still uses feature_request_prompt config key
        system_time=datetime.now(tz=UTC).isoformat()
    )
    
    # Filter messages: Include history but maybe exclude PA's routing messages if added
    agent_messages = [msg for msg in state.messages if getattr(msg, 'name', None) != 'PersonalAssistant']

    response = cast(
        AIMessage,
        await model.ainvoke(
             [{"role": "system", "content": system_message}, *agent_messages] 
        ),
    )
    response.name = "ExecutionEnforcer" # Name the response

    if not response.tool_calls:
         print("Execution Enforcer: Finished (no tool calls).")
         feature_request = {
             "description": response.content,
             "timestamp": datetime.now(tz=UTC).isoformat(),
             "status": "queued"
         }
         current_requests = state.feature_requests if state.feature_requests is not None else []
         updated_requests = current_requests + [feature_request]
         return {
             "messages": [response], 
             "feature_requests": updated_requests,
             "active_agent": None # Mark as done
        }
    else:
        print("Execution Enforcer: Requesting tools.")
        # Keep active_agent set (execution_enforcer), return message with tool calls
        return {"messages": [response]} 


# Deep Research Agent Implementation
async def deep_research_agent(state: State) -> Dict:
    \"\"\"
    Deep research agent that provides in-depth information on complex topics.
    \"\"\"
    print("--- Running Deep Research Agent ---")
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model).bind_tools(TOOLS)
    system_message = configuration.deep_research_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Filter messages
    agent_messages = [msg for msg in state.messages if getattr(msg, 'name', None) != 'PersonalAssistant']

    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *agent_messages] 
        ),
    )
    response.name = "DeepResearch" # Name the response

    if not response.tool_calls:
        print("Deep Research Agent: Finished (no tool calls).")
        return {
            "messages": [response], 
            "active_agent": None # Mark as done
        }
    else:
        print("Deep Research Agent: Requesting tools.")
        # Keep active_agent set (deep_research), return message with tool calls
        return {"messages": [response]} 


# Build the hierarchical multi-agent graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add the nodes
builder.add_node("personal_assistant_agent", personal_assistant_agent)
builder.add_node("execution_enforcer_agent", execution_enforcer_agent) # Renamed node
builder.add_node("deep_research_agent", deep_research_agent)
builder.add_node("tools", ToolNode(TOOLS))

# Entry point is the Personal Assistant
builder.add_edge("__start__", "personal_assistant_agent")

# Routing from Personal Assistant
builder.add_conditional_edges(
    "personal_assistant_agent",
    # Decision is made within the node, output in 'next'
    lambda state: state.next, 
    {
        "execution_enforcer_agent": "execution_enforcer_agent",
        "deep_research_agent": "deep_research_agent",
        "tools": "tools", # Added route for PA direct handling needing tools
        "__end__": "__end__"
    }
)

# Conditional edges from specialist agents based on tool calls
def route_specialist_output(state: State) -> Literal["tools", "personal_assistant_agent"]:
    \"\"\"Routes specialist agent output to tools or back to the Personal Assistant."\"\"
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            print(f"Specialist ({state.active_agent}) output: Has tool calls. Routing to tools.")
            return "tools"
        else:
            # Specialist finished its turn without tool calls
            print(f"Specialist ({state.active_agent}) output: No tool calls. Routing to Personal Assistant for synthesis.")
            # Clear active_agent here before routing back to PA
            state.active_agent = None 
            return "personal_assistant_agent" 
    print(f"Specialist ({state.active_agent}) output: Last message not AIMessage. Routing to PA (fallback).")
    state.active_agent = None
    return "personal_assistant_agent" 

builder.add_conditional_edges(
    "execution_enforcer_agent",
    route_specialist_output,
    {"tools": "tools", "personal_assistant_agent": "personal_assistant_agent"}
)

builder.add_conditional_edges(
    "deep_research_agent",
    route_specialist_output,
    {"tools": "tools", "personal_assistant_agent": "personal_assistant_agent"}
)


# Edge from tools back to the active agent
def route_tools_output(state: State) -> Literal["personal_assistant_agent", "execution_enforcer_agent", "deep_research_agent"]:
    \"\"\"Routes tool output back to the agent that called the tools."\"\"
    active_agent = state.active_agent
    print(f"Tools output: Routing back to active agent: {active_agent}")
    if active_agent == "personal_assistant":
        return "personal_assistant_agent"
    elif active_agent == "execution_enforcer":
        return "execution_enforcer_agent"
    elif active_agent == "deep_research":
        return "deep_research_agent"
    else:
        print(f"ERROR: No active agent found after tool execution. State: {state}")
        # Fallback to PA
        return "personal_assistant_agent"

builder.add_conditional_edges(
    "tools",
    route_tools_output,
    {
        "personal_assistant_agent": "personal_assistant_agent",
        "execution_enforcer_agent": "execution_enforcer_agent",
        "deep_research_agent": "deep_research_agent",
    }
)

# Compile the graph
graph = builder.compile(name="Kaizen Personal Assistant")

# Optional: Print graph structure for debugging
# try:
#     from langchain_core.runnables.graph import print_ascii
#     print_ascii(graph)
# except ImportError:
#     print("Install 'pip install runnable-graph' to print the graph structure.")
