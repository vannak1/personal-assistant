"""Define a hierarchical multi-agent system.

This implements a supervisor agent that routes requests to specialized agent teams:
1. Personal Assistant Agent: Handles user interaction and simple queries
2. Execution Enforcer Agent: Creates detailed implementation plans
3. Deep Research Agent: Provides in-depth research on complex topics
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, TypedDict, cast
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import State, InputState
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model

# Define specific session states
SESSION_STATE_CHECKING = "session_checking"
SESSION_STATE_WAITING_FOR_NAME = "session_waiting_for_name"

# Supervisor Agent Implementation
async def supervisor_agent(state: State) -> Dict:
    """
    The supervisor agent that analyzes requests and routes them to specialized agents.
    """
    print(f"--- Running Supervisor Agent (State: {state.active_agent}) ---")
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.supervisor_model)
    
    last_message = state.messages[-1]
    
    # If this is the first message and we don't have user information yet,
    # route directly to personal assistant to handle session
    if isinstance(last_message, HumanMessage) and state.first_message and not state.user_name:
        print("Supervisor: First message, routing to PA for session handling")
        state.first_message = False  # Mark that we've processed the first message
        state.routing_reason = "session_management"
        return {
            "next": "personal_assistant_agent",
            "active_agent": "personal_assistant"
        }
    
    # If we're receiving results from a specialist agent, route to PA for presentation
    if isinstance(last_message, AIMessage) and last_message.name in ["ExecutionEnforcer", "DeepResearch"]:
        print(f"Supervisor: Received results from {last_message.name}, routing to PA for presentation")
        # Reset active agent since the specialist is done
        state.active_agent = None
        state.routing_reason = "present_results"
        state.specialist_results = True
        return {
            "next": "personal_assistant_agent"
        }
    
    # For normal routing decisions, analyze the request
    user_request_message = next((msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)), None)
    if not user_request_message:
        # Fallback if we can't find a user message
        state.routing_reason = "fallback"
        return {"next": "personal_assistant_agent"}
    
    # Get user name if available
    current_user_name = state.user_name or "User"
    
    # Analyze the request to determine routing
    system_message = configuration.supervisor_prompt.format(system_time=datetime.now(tz=UTC).isoformat())
    routing_prompt = f"""
User ({current_user_name}) request: '{user_request_message.content}'.

Based on this request, determine which specialized agent should handle it:

1. PERSONAL_ASSISTANT: For user interaction, simple questions, direct handling, small talk, capability explanations, or result presentation.
2. EXECUTION_ENFORCER: For planning, implementation, project breakdown, or feasibility analysis.
3. DEEP_RESEARCH: For in-depth information, complex analysis, or comprehensive research.

Respond ONLY with 'PERSONAL_ASSISTANT', 'EXECUTION_ENFORCER', or 'DEEP_RESEARCH'.
"""
    
    # Get relevant history for context
    history = [msg for msg in state.messages if not isinstance(msg, ToolMessage)][-5:]  # Last 5 non-tool messages
    
    routing_response = await model.ainvoke([
        {"role": "system", "content": system_message},
        *history[:-1],  # Previous messages for context
        {"role": "user", "content": routing_prompt}
    ])
    
    routing_decision = routing_response.content.strip().upper()
    print(f"Supervisor Routing Decision: {routing_decision}")
    
    # Map the decision to the appropriate agent
    if "EXECUTION_ENFORCER" in routing_decision:
        # Prepare context for execution enforcer
        execution_context = f"""
User: {current_user_name}
Request: {user_request_message.content}

SPECIALIST INSTRUCTIONS:
You are receiving this request because it requires planning, feature development, or turning ideas into actions.
Based on our analysis, this request needs detailed implementation planning.
"""
        # Add context note for the specialist
        state.messages.append(AIMessage(
            content=execution_context,
            name="Supervisor_ContextNote",
            additional_kwargs={"is_context": True}
        ))
        
        state.routing_reason = "planning_needed"
        return {
            "next": "execution_enforcer_agent", 
            "active_agent": "execution_enforcer"
        }
        
    elif "DEEP_RESEARCH" in routing_decision:
        # Prepare context for deep research
        research_context = f"""
User: {current_user_name}
Request: {user_request_message.content}

SPECIALIST INSTRUCTIONS:
You are receiving this request because it requires in-depth research or complex information gathering.
Based on our analysis, this request needs comprehensive research and explanation.
"""
        # Add context note for the specialist
        state.messages.append(AIMessage(
            content=research_context,
            name="Supervisor_ContextNote",
            additional_kwargs={"is_context": True}
        ))
        
        state.routing_reason = "research_needed"
        return {
            "next": "deep_research_agent", 
            "active_agent": "deep_research"
        }
        
    else:  # Default to PERSONAL_ASSISTANT
        state.routing_reason = "direct_handling"
        return {
            "next": "personal_assistant_agent", 
            "active_agent": "personal_assistant"
        }


# Personal Assistant Agent Implementation
async def personal_assistant_agent(state: State) -> Dict:
    """
    The personal assistant agent that handles user interaction, simple queries,
    session management, and result presentation.
    """
    print(f"--- Running Personal Assistant Agent (State: {state.active_agent}) ---")
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.personal_assistant_model).bind_tools(TOOLS)
    
    last_message = state.messages[-1]
    current_user_name = state.user_name
    routing_reason = state.routing_reason or "unknown"
    
    print(f"PA: Received task with routing reason: {routing_reason}")
    
    # --- Session & User Management Logic ---
    
    # A. New user, need to get session information
    if not current_user_name and routing_reason == "session_management":
        print("PA: New user. Checking session.")
        state.session_state = SESSION_STATE_CHECKING
        return {
            "messages": [AIMessage(
                content="", 
                tool_calls=[{
                    "name": "manage_user_session", 
                    "args": {}, 
                    "id": f"tool_call_{uuid.uuid4()}"
                }],
                name="PersonalAssistant"
            )],
            "active_agent": "personal_assistant"
        }
    
    # B. Received session check result
    if isinstance(last_message, ToolMessage) and state.session_state == SESSION_STATE_CHECKING:
        # Process session information
        session_info = last_message.content
        if isinstance(session_info, str):
            try:
                import json
                session_info = json.loads(session_info)
            except json.JSONDecodeError:
                session_info = {"user_name": None, "user_uid": None}
                
        found_name = session_info.get("user_name")
        found_uid = session_info.get("user_uid")
        
        if found_name:
            # Existing user found
            state.user_name = found_name
            state.user_uid = found_uid
            state.session_state = None
            greeting_message = AIMessage(
                content=f"Welcome back, {found_name}! How can I help you today?",
                name="PersonalAssistant"
            )
            greeting_message.additional_kwargs = {"status": "greeting_returning_user"}
            return {
                "messages": [greeting_message],
                "next": "supervisor_agent" # Return to supervisor for next action
            }
        else:
            # New user, ask for name
            state.session_state = SESSION_STATE_WAITING_FOR_NAME
            greeting_message = AIMessage(
                content="Hello there! I'm your personal assistant. I don't believe we've met, what's your name?",
                name="PersonalAssistant"
            )
            greeting_message.additional_kwargs = {"status": "greeting_new_user"}
            return {
                "messages": [greeting_message],
                "next": "__end__" # Wait for user response
            }
    
    # C. User provided their name
    if isinstance(last_message, HumanMessage) and state.session_state == SESSION_STATE_WAITING_FOR_NAME:
        # Save name
        extracted_name = last_message.content.strip()
        state.session_state = SESSION_STATE_CHECKING
        return {
            "messages": [AIMessage(
                content="", 
                tool_calls=[{
                    "name": "manage_user_session", 
                    "args": {"user_name_to_set": extracted_name}, 
                    "id": f"tool_call_{uuid.uuid4()}"
                }],
                name="PersonalAssistant"
            )],
            "active_agent": "personal_assistant"
        }
    
    # --- Specialist Result Presentation Logic ---
    if routing_reason == "present_results" and state.specialist_results:
        # Get the specialist results (last AI message that's not from PersonalAssistant)
        specialist_results = next(
            (msg for msg in reversed(state.messages) 
             if isinstance(msg, AIMessage) and msg.name not in ["PersonalAssistant", "Supervisor"]),
            None
        )
        
        if not specialist_results:
            # Fallback if we can't find specialist results
            return {
                "messages": [AIMessage(
                    content=f"I've analyzed your request, {current_user_name}. Is there anything specific you'd like me to help with?",
                    name="PersonalAssistant"
                )],
                "next": "supervisor_agent"
            }
        
        # Prepare a prompt to present the results
        specialist_type = specialist_results.name
        presentation_prompt = f"""
The {specialist_type} agent has provided this information:
'{specialist_results.content}'

Present this information back to {current_user_name} in a friendly, conversational way:
1. Use a warm tone and natural language
2. Relate the information to their original request
3. Ask if this meets their needs or if they need any clarification
4. Offer to help with next steps
"""
        
        response = await model.ainvoke([
            {"role": "system", "content": configuration.personal_assistant_prompt.format(system_time=datetime.now(tz=UTC).isoformat())},
            {"role": "user", "content": presentation_prompt}
        ])
        
        result_message = AIMessage(content=response.content, name="PersonalAssistant")
        result_message.additional_kwargs = {
            "status": "presenting_results",
            "specialist_source": specialist_type
        }
        
        # Reset the specialist results flag
        state.specialist_results = False
        
        return {
            "messages": [result_message],
            "next": "supervisor_agent" # Return to supervisor for next action
        }
    
    # --- Direct Handling Logic (for simple requests) ---
    
    # Get the user's request message
    user_request_message = next((msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)), None)
    if not user_request_message:
        # Fallback if we can't find a user message
        return {
            "messages": [AIMessage(
                content=f"How can I help you today, {current_user_name or 'there'}?",
                name="PersonalAssistant"
            )],
            "next": "supervisor_agent"
        }
    
    # Direct handling prompt
    direct_prompt = f"""
User ({current_user_name}): {user_request_message.content}

Handle this request directly in a friendly, conversational manner. You can use tools if needed.
"""
    
    # Get relevant history for context
    history = [msg for msg in state.messages if not isinstance(msg, ToolMessage)][-5:]  # Last 5 non-tool messages
    
    response = cast(AIMessage, await model.ainvoke([
        {"role": "system", "content": configuration.personal_assistant_prompt.format(system_time=datetime.now(tz=UTC).isoformat())},
        *history[:-1],  # Previous messages for context
        {"role": "user", "content": direct_prompt}
    ]))
    
    response.name = "PersonalAssistant"
    response.additional_kwargs = {"status": "direct_handling"}
    
    if not response.tool_calls:
        # No tools needed, just respond directly
        return {
            "messages": [response],
            "next": "supervisor_agent" # Return to supervisor for next action
        }
    else:
        # Tools needed for direct handling
        return {
            "messages": [response],
            "active_agent": "personal_assistant"  # Keep PA active for tool loop
        }


# Execution Enforcer Agent Implementation
async def execution_enforcer_agent(state: State) -> Dict:
    """
    Execution Enforcer Agent: Turns ideas into actionable plans.
    """
    print("--- Running Execution Enforcer Agent ---")
    configuration = Configuration.from_context()
    # Use the specific model for this agent
    model = load_chat_model(configuration.execution_enforcer_model).bind_tools(TOOLS)
    system_message = configuration.feature_request_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )
    
    # Filter messages: Include history, context notes, but exclude general messages
    agent_messages = []
    context_notes = []
    
    for msg in state.messages:
        # Collect context notes separately
        if getattr(msg, 'name', None) in ['Supervisor_ContextNote', 'PersonalAssistant_ContextNote']:
            context_notes.append(msg)
        # Include all user messages, but not PA or Supervisor responses
        elif isinstance(msg, HumanMessage) or (isinstance(msg, AIMessage) and 
                                              getattr(msg, 'name', None) not in ['PersonalAssistant', 'Supervisor']):
            agent_messages.append(msg)
    
    # Add context notes at the beginning for better context
    agent_messages = context_notes + agent_messages

    response = cast(
        AIMessage,
        await model.ainvoke(
             [{"role": "system", "content": system_message}, *agent_messages] 
        ),
    )
    response.name = "ExecutionEnforcer" # Name the response
    
    # Add status information
    response.additional_kwargs = {
        "status": "specialist_active",
        "specialist": "ExecutionEnforcer"
    }

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
             "next": "supervisor_agent" # Return to supervisor for presentation
        }
    else:
        print("Execution Enforcer: Requesting tools.")
        # Keep active_agent set, return message with tool calls
        return {"messages": [response]} 


# Deep Research Agent Implementation
async def deep_research_agent(state: State) -> Dict:
    """
    Deep research agent that provides in-depth information on complex topics.
    """
    print("--- Running Deep Research Agent ---")
    configuration = Configuration.from_context()
    # Use the specific model for this agent
    model = load_chat_model(configuration.deep_research_model).bind_tools(TOOLS)
    system_message = configuration.deep_research_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Filter messages: Include history, context notes, but exclude general messages
    agent_messages = []
    context_notes = []
    
    for msg in state.messages:
        # Collect context notes separately
        if getattr(msg, 'name', None) in ['Supervisor_ContextNote', 'PersonalAssistant_ContextNote']:
            context_notes.append(msg)
        # Include all user messages, but not PA or Supervisor responses
        elif isinstance(msg, HumanMessage) or (isinstance(msg, AIMessage) and 
                                              getattr(msg, 'name', None) not in ['PersonalAssistant', 'Supervisor']):
            agent_messages.append(msg)
    
    # Add context notes at the beginning for better context
    agent_messages = context_notes + agent_messages

    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *agent_messages] 
        ),
    )
    response.name = "DeepResearch" # Name the response
    
    # Add status information
    response.additional_kwargs = {
        "status": "specialist_active",
        "specialist": "DeepResearch"
    }

    if not response.tool_calls:
        print("Deep Research Agent: Finished (no tool calls).")
        return {
            "messages": [response], 
            "next": "supervisor_agent" # Return to supervisor for presentation
        }
    else:
        print("Deep Research Agent: Requesting tools.")
        # Keep active_agent set, return message with tool calls
        return {"messages": [response]} 


# Build the hierarchical multi-agent graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add the nodes
builder.add_node("supervisor_agent", supervisor_agent)
builder.add_node("personal_assistant_agent", personal_assistant_agent)
builder.add_node("execution_enforcer_agent", execution_enforcer_agent)
builder.add_node("deep_research_agent", deep_research_agent)
builder.add_node("tools", ToolNode(TOOLS))

# Entry point is now the Supervisor
builder.add_edge("__start__", "supervisor_agent")

# Routing from Supervisor to specialized agents
def route_supervisor_output(state: State) -> Literal["personal_assistant_agent", "execution_enforcer_agent", "deep_research_agent"]:
    """Routes supervisor output to the appropriate agent."""
    # Use the next field set by supervisor
    return state.next

builder.add_conditional_edges(
    "supervisor_agent",
    route_supervisor_output,
    {
        "personal_assistant_agent": "personal_assistant_agent",
        "execution_enforcer_agent": "execution_enforcer_agent",
        "deep_research_agent": "deep_research_agent",
    }
)

# Routing from Personal Assistant
def route_pa_output(state: State) -> Literal["tools", "supervisor_agent", "__end__"]:
    """Routes PA output based on tools, completion, or next destination."""
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # PA needs to call tools
        return "tools"
    
    # If PA set a specific next field (e.g., for session management)
    if state.next == "__end__":
        return "__end__"
    elif state.next == "supervisor_agent":
        return "supervisor_agent"
    
    # If PA is still in session management
    if state.session_state in [SESSION_STATE_CHECKING, SESSION_STATE_WAITING_FOR_NAME]:
        # Stay with PA
        return "personal_assistant_agent"
    
    # Otherwise, back to supervisor
    return "supervisor_agent"

builder.add_conditional_edges(
    "personal_assistant_agent",
    route_pa_output,
    {
        "tools": "tools",
        "personal_assistant_agent": "personal_assistant_agent",
        "supervisor_agent": "supervisor_agent",
        "__end__": "__end__"
    }
)

# Routing from specialist agents
def route_specialist_output(state: State) -> Literal["tools", "supervisor_agent"]:
    """Routes specialist output to tools or back to supervisor."""
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Specialist needs tools
        return "tools"
    
    # If specialist set a specific next field
    if state.next == "supervisor_agent":
        return "supervisor_agent"
    
    # Otherwise, back to supervisor for presenting results
    return "supervisor_agent"

builder.add_conditional_edges(
    "execution_enforcer_agent",
    route_specialist_output,
    {"tools": "tools", "supervisor_agent": "supervisor_agent"}
)

builder.add_conditional_edges(
    "deep_research_agent",
    route_specialist_output,
    {"tools": "tools", "supervisor_agent": "supervisor_agent"}
)

# Edge from tools back to the active agent
def route_tools_output(state: State) -> Literal["personal_assistant_agent", "execution_enforcer_agent", "deep_research_agent"]:
    """Routes tool output back to the agent that called the tools."""
    active_agent = state.active_agent
    print(f"Tools output: Routing back to active agent: {active_agent}")
    
    if active_agent == "personal_assistant":
        return "personal_assistant_agent"
    elif active_agent == "execution_enforcer":
        return "execution_enforcer_agent"
    elif active_agent == "deep_research":
        return "deep_research_agent"
    else:
        # Fallback to PA
        print(f"ERROR: Unknown active agent '{active_agent}' after tool execution. Routing to PA.")
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
graph = builder.compile(name="Kaizen Multi-Agent System")