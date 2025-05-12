# graph.py
"""Define a hierarchical multi-agent system.

This implements a personal assistant agent as the main interaction point
that handles user interaction, simple queries, session management, result presentation,
and routes requests to specialized agent teams when necessary:
1. Personal Assistant Agent: Handles user interaction, session, routing, presentation.
2. Features Agent: Documents and tracks feature requests.
3. Deep Research Agent: Provides in-depth research on complex topics.
4. Web Search Agent: Handles web searches and information retrieval via tools.
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, TypedDict, cast, Optional, Any
import uuid
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError
from langgraph.pregel.retry import RetryPolicy

from react_agent.configuration import Configuration
from react_agent.state import State, InputState
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model

# Define specific session states (constants)
SESSION_STATE_CHECKING = "session_checking"
SESSION_STATE_WAITING_FOR_NAME = "session_waiting_for_name"
SESSION_STATE_NAME_RECEIVED = "session_name_received"

# --- Agent Implementations ---

async def personal_assistant_agent(state: State) -> Dict:
    """
    The personal assistant agent handles user interaction, simple queries,
    session management, result presentation, and routing to specialist agents.
    """
    print(f"--- Running Personal Assistant Agent ---")
    configuration = Configuration.from_context()
    # Bind tools for potential direct use by PA
    model = load_chat_model(configuration.personal_assistant_model).bind_tools(TOOLS)

    last_message = state.messages[-1] if state.messages else None
    current_user_name = state.user_name or "User"
    routing_reason = state.routing_reason or "unknown"

    print(f"PA: Current state - session_state: {state.session_state}, user_name: {state.user_name}")
    print(f"PA: Last message type: {type(last_message).__name__ if last_message else 'None'}")

    # --- Session & User Management Logic ---
    # A. First message, need to check session
    if state.first_message and not state.user_name and state.session_state is None:
        print("PA: First message, initiating session check.")
        state.first_message = False
        state.session_state = SESSION_STATE_CHECKING
        tool_call_id = f"tool_call_{uuid.uuid4()}"
        
        return {
            "messages": [AIMessage(
                content="",
                tool_calls=[{
                    "name": "manage_user_session",
                    "args": {},
                    "id": tool_call_id
                }],
                name="PersonalAssistant"
            )],
            "session_state": SESSION_STATE_CHECKING,
            "active_agent": "personal_assistant",
            "first_message": False
        }

    # B. Received session check result
    if isinstance(last_message, ToolMessage) and state.session_state == SESSION_STATE_CHECKING:
        print("PA: Processing session check result.")
        try:
            content = last_message.content
            session_info = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, Exception) as e:
            print(f"PA: Error parsing session info: {e}")
            session_info = {"user_name": None, "user_uid": None}

        found_name = session_info.get("user_name")
        found_uid = session_info.get("user_uid")

        if found_name:
            # Existing user found
            print(f"PA: Found existing user: {found_name}")
            greeting_message = AIMessage(
                content=f"Welcome back, {found_name}! How can I help you today?",
                name="PersonalAssistant"
            )
            return {
                "messages": [greeting_message],
                "user_name": found_name,
                "user_uid": found_uid,
                "session_state": None,
                "active_agent": None,
                "next": "__end__"
            }
        else:
            # New user, ask for name
            print("PA: New user, asking for name.")
            greeting_message = AIMessage(
                content="Hello there! I'm your personal assistant. It seems we haven't met before. What's your name?",
                name="PersonalAssistant"
            )
            return {
                "messages": [greeting_message],
                "session_state": SESSION_STATE_WAITING_FOR_NAME,
                "active_agent": None,
                "next": "__end__"
            }

    # C. User provided their name
    if isinstance(last_message, HumanMessage) and state.session_state == SESSION_STATE_WAITING_FOR_NAME:
        print("PA: User provided name, saving session.")
        user_name = last_message.content.strip()
        state.session_state = SESSION_STATE_NAME_RECEIVED
        tool_call_id = f"tool_call_{uuid.uuid4()}"
        
        return {
            "messages": [AIMessage(
                content="",
                tool_calls=[{
                    "name": "manage_user_session",
                    "args": {"user_name_to_set": user_name},
                    "id": tool_call_id
                }],
                name="PersonalAssistant"
            )],
            "session_state": SESSION_STATE_NAME_RECEIVED,
            "active_agent": "personal_assistant"
        }

    # D. Session save confirmation
    if isinstance(last_message, ToolMessage) and state.session_state == SESSION_STATE_NAME_RECEIVED:
        print("PA: Session saved, greeting new user.")
        try:
            content = last_message.content
            session_info = json.loads(content) if isinstance(content, str) else content
            user_name = session_info.get("user_name")
            user_uid = session_info.get("user_uid")
        except (json.JSONDecodeError, Exception) as e:
            print(f"PA: Error parsing saved session: {e}")
            user_name = "friend"
            user_uid = None

        greeting_message = AIMessage(
            content=f"Nice to meet you, {user_name}! I'm here to help. What can I do for you today?",
            name="PersonalAssistant"
        )
        return {
            "messages": [greeting_message],
            "user_name": user_name,
            "user_uid": user_uid,
            "session_state": None,
            "active_agent": None,
            "next": "__end__"
        }

    # --- Specialist Result Presentation Logic ---
    if isinstance(last_message, AIMessage) and getattr(last_message, 'additional_kwargs', {}).get('status') == 'results_obtained':
        specialist_results_msg = last_message
        specialist_type = specialist_results_msg.name
        print(f"PA: Processing results from {specialist_type}")

        if specialist_type == "WebSearchAgent":
            original_query = specialist_results_msg.additional_kwargs.get("original_query", "your recent query")
            search_results_raw = specialist_results_msg.additional_kwargs.get("search_results")

            formatted_results = f"Search results for query: '{original_query}'\n\n"

            if isinstance(search_results_raw, str):
                try:
                    search_results_raw = json.loads(search_results_raw)
                except json.JSONDecodeError:
                    search_results_raw = [{"title": "Parser Error", "url": "", "content": f"Error parsing search results"}]

            if isinstance(search_results_raw, list) and search_results_raw:
                if isinstance(search_results_raw[0], dict) and search_results_raw[0].get("title") == "Search Error":
                    formatted_results += f"An error occurred during the search: {search_results_raw[0].get('content', 'Unknown error')}"
                else:
                    for i, result in enumerate(search_results_raw, 1):
                        title = result.get("title", "Untitled")
                        url = result.get("url", "No URL")
                        content = result.get("content", "No content available")
                        formatted_results += f"{i}. {title} ({url})\n   Summary: {content[:250]}...\n\n"
            else:
                formatted_results += "No search results were found or returned for the query."

            search_presentation_prompt = f"""
            You are presenting web search results to the user, {current_user_name}.
            Their original query was: '{original_query}'

            The following search results were retrieved from the web:
            --- BEGIN SEARCH RESULTS ---
            {formatted_results}
            --- END SEARCH RESULTS ---

            Based ONLY on the provided search results, synthesize a helpful and concise answer.
            """

            print("PA: Calling LLM to synthesize search results...")
            context_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)][-2:]

            response = await model.ainvoke([
                {"role": "system", "content": configuration.personal_assistant_prompt.format(system_time=datetime.now(tz=UTC).isoformat())},
                *context_messages,
                {"role": "user", "content": search_presentation_prompt}
            ])

            result_message = AIMessage(content=response.content, name="PersonalAssistant")
            return {
                "messages": [result_message],
                "active_agent": None,
                "routing_reason": None,
                "specialist_results": False,
                "next": "__end__"
            }

    # --- Routing Logic for new messages ---
    if isinstance(last_message, HumanMessage) and state.user_name and state.session_state is None:
        print("PA: New human message, deciding route.")
        user_request = last_message.content

        routing_prompt = f"""
        User ({current_user_name}) request: '{user_request}'.

        Analyze this request. Which agent is BEST suited? Choose ONE:
        - PERSONAL_ASSISTANT: Simple questions, chat, clarification.
        - FEATURES_AGENT: Planning features, software development tasks.
        - DEEP_RESEARCH: In-depth analysis, complex topics.
        - WEB_SEARCH: Current events, facts needing up-to-date info.

        Respond ONLY with the agent name.
        """

        routing_response = await model.ainvoke([
            {"role": "system", "content": "You are a routing assistant."},
            {"role": "user", "content": routing_prompt}
        ])
        routing_decision = routing_response.content.strip().upper()
        print(f"PA Routing Decision: {routing_decision}")

        context_note_content = f"""
        User: {current_user_name}
        Request: {user_request}
        Routing Reason: This request was identified as needing your specific capabilities.
        """
        context_note = AIMessage(
            content=context_note_content,
            name="PersonalAssistant_ContextNote"
        )

        if "FEATURES_AGENT" in routing_decision:
            return {
                "messages": [context_note],
                "routing_reason": "feature_planning_needed",
                "active_agent": "features_agent",
                "next": "features_agent"
            }
        elif "DEEP_RESEARCH" in routing_decision:
            return {
                "messages": [context_note],
                "routing_reason": "research_needed",
                "active_agent": "deep_research",
                "next": "deep_research_agent"
            }
        elif "WEB_SEARCH" in routing_decision:
            return {
                "messages": [context_note],
                "routing_reason": "web_search_needed",
                "active_agent": "web_search",
                "next": "web_search_agent"
            }
        else:
            # Handle directly
            print("PA: Handling request directly.")
            direct_prompt = f"""
            User ({current_user_name}): {user_request}
            
            Handle this request directly. Maintain a helpful, conversational tone.
            """
            
            response = await model.ainvoke([
                {"role": "system", "content": configuration.personal_assistant_prompt.format(system_time=datetime.now(tz=UTC).isoformat())},
                {"role": "user", "content": direct_prompt}
            ])
            
            response.name = "PersonalAssistant"
            
            if not response.tool_calls:
                return {
                    "messages": [response],
                    "active_agent": None,
                    "next": "__end__"
                }
            else:
                return {
                    "messages": [response],
                    "active_agent": "personal_assistant"
                }

    # Handle tool responses for PA
    if isinstance(last_message, ToolMessage) and state.active_agent == "personal_assistant":
        print("PA: Received tool result for direct handling.")
        
        direct_prompt = f"Based on the tool result, provide a response to {current_user_name}."
        history = state.messages[-5:]
        
        response = await model.ainvoke(history)
        response.name = "PersonalAssistant"
        
        return {
            "messages": [response],
            "active_agent": None,
            "next": "__end__"
        }

    # Fallback
    print("PA: Reached fallback state.")
    fallback_message = AIMessage(
        content="I'm here to help. Please let me know what you need.",
        name="PersonalAssistant"
    )
    return {
        "messages": [fallback_message],
        "active_agent": None,
        "next": "__end__"
    }


async def features_agent(state: State) -> Dict:
    """Features Agent: Documents and plans feature requests."""
    print("--- Running Features Agent ---")
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.execution_enforcer_model).bind_tools(TOOLS)
    
    system_message = configuration.feature_request_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    agent_messages: List[BaseMessage] = [{"role": "system", "content": system_message}]
    
    # Include relevant messages
    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            agent_messages.append(msg)
        elif isinstance(msg, ToolMessage) and state.active_agent == "features_agent":
            agent_messages.append(msg)
        elif getattr(msg, 'name', None) == 'PersonalAssistant_ContextNote':
            agent_messages.append(msg)

    response = await model.ainvoke(agent_messages)
    response.name = "FeaturesAgent"

    if not response.tool_calls:
        print("Features Agent: Task complete.")
        return {
            "messages": [response],
            "active_agent": None,
            "next": "personal_assistant_agent"
        }
    else:
        print("Features Agent: Requesting tools.")
        return {
            "messages": [response],
            "active_agent": "features_agent"
        }


async def deep_research_agent(state: State) -> Dict:
    """Deep research agent: Provides in-depth information."""
    print("--- Running Deep Research Agent ---")
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.deep_research_model).bind_tools(TOOLS)
    
    system_message = configuration.deep_research_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    agent_messages: List[BaseMessage] = [{"role": "system", "content": system_message}]
    
    # Include relevant messages
    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            agent_messages.append(msg)
        elif isinstance(msg, ToolMessage) and state.active_agent == "deep_research":
            agent_messages.append(msg)
        elif getattr(msg, 'name', None) == 'PersonalAssistant_ContextNote':
            agent_messages.append(msg)

    response = await model.ainvoke(agent_messages)
    response.name = "DeepResearch"

    if not response.tool_calls:
        print("Deep Research Agent: Task complete.")
        return {
            "messages": [response],
            "active_agent": None,
            "next": "personal_assistant_agent"
        }
    else:
        print("Deep Research Agent: Requesting tools.")
        return {
            "messages": [response],
            "active_agent": "deep_research"
        }


async def web_search_agent(state: State) -> Dict:
    """Web search agent: Handles search queries using the 'search' tool."""
    print("--- Running Web Search Agent ---")
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.web_search_model).bind_tools(TOOLS)

    # Extract original query
    original_user_request = None
    context_note = next((msg for msg in reversed(state.messages) 
                        if getattr(msg, 'name', None) == 'PersonalAssistant_ContextNote'), None)

    if context_note and isinstance(context_note.content, str) and "Request:" in context_note.content:
        request_line = [line for line in context_note.content.split('\n') if "Request:" in line]
        if request_line:
            original_user_request = request_line[0].replace("Request:", "").strip()

    if not original_user_request:
        user_request_message = next((msg for msg in reversed(state.messages) 
                                   if isinstance(msg, HumanMessage)), None)
        if user_request_message:
            original_user_request = user_request_message.content

    search_query = original_user_request if original_user_request else "general information"

    # Handle tool result
    last_message = state.messages[-1] if state.messages else None
    if isinstance(last_message, ToolMessage) and state.active_agent == "web_search":
        print(f"Web Search Agent: Processing tool result")
        
        try:
            search_results_raw = json.loads(last_message.content)
        except json.JSONDecodeError:
            search_results_raw = [{"title": "Parser Error", "url": "", 
                                 "content": f"Error parsing search results"}]

        result_carrier_message = AIMessage(
            content=f"Search results obtained for query: '{search_query}'",
            name="WebSearchAgent",
            additional_kwargs={
                "status": "results_obtained",
                "original_query": search_query,
                "search_results": search_results_raw
            }
        )

        return {
            "messages": [result_carrier_message],
            "active_agent": None,
            "next": "personal_assistant_agent"
        }
    else:
        # Initiate search
        tool_call_id = f"tool_call_{uuid.uuid4()}"
        print(f"Web Search Agent: Initiating search for '{search_query}'")

        search_request = AIMessage(
            content="I'll search for information about that topic.",
            tool_calls=[{
                "name": "search",
                "args": {"query": search_query},
                "id": tool_call_id
            }],
            name="WebSearchAgent"
        )

        return {
            "messages": [search_request],
            "active_agent": "web_search"
        }


# --- Graph Construction ---
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add nodes
builder.add_node("personal_assistant_agent", personal_assistant_agent, 
                retry=RetryPolicy(max_attempts=3))
builder.add_node("features_agent", features_agent)
builder.add_node("deep_research_agent", deep_research_agent)
builder.add_node("web_search_agent", web_search_agent)

# Create ToolNode with error handling
# Explicitly list each tool to ensure they are properly registered
from react_agent.tools import search_tool, manage_user_session_tool
tool_node = ToolNode([search_tool, manage_user_session_tool], handle_tool_errors=True)
builder.add_node("tools", tool_node)

# Set entry point
builder.set_entry_point("personal_assistant_agent")

# --- Routing Logic ---

def route_pa_output(state: State) -> str:
    """Routes PA output based on next state or tool calls."""
    last_message = state.messages[-1] if state.messages else None

    # Check for tool calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # Check for explicit next routing
    next_node = state.next
    if next_node == "features_agent":
        return "features_agent"
    elif next_node == "deep_research_agent":
        return "deep_research_agent"
    elif next_node == "web_search_agent":
        return "web_search_agent"
    elif next_node == "__end__":
        return "__end__"

    # Default to end
    return "__end__"

builder.add_conditional_edges(
    "personal_assistant_agent",
    route_pa_output,
    {
        "tools": "tools",
        "features_agent": "features_agent",
        "deep_research_agent": "deep_research_agent",
        "web_search_agent": "web_search_agent",
        "__end__": "__end__"
    }
)

def route_specialist_output(state: State) -> str:
    """Routes specialist output."""
    last_message = state.messages[-1] if state.messages else None

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return "personal_assistant_agent"

# Apply routing to all specialists
builder.add_conditional_edges("features_agent", route_specialist_output, 
                            {"tools": "tools", "personal_assistant_agent": "personal_assistant_agent"})
builder.add_conditional_edges("deep_research_agent", route_specialist_output, 
                            {"tools": "tools", "personal_assistant_agent": "personal_assistant_agent"})
builder.add_conditional_edges("web_search_agent", route_specialist_output, 
                            {"tools": "tools", "personal_assistant_agent": "personal_assistant_agent"})

def route_tools_output(state: State) -> str:
    """Routes tool output back to the calling agent."""
    active_agent = state.active_agent
    print(f"Routing Tools: Active agent is '{active_agent}'")

    # Map state values to node names
    if active_agent == "personal_assistant":
        return "personal_assistant_agent"
    elif active_agent == "features_agent":
        return "features_agent"
    elif active_agent == "deep_research":
        return "deep_research_agent"
    elif active_agent == "web_search":
        return "web_search_agent"
    else:
        print(f"WARNING: Unknown active agent '{active_agent}', routing to PA")
        return "personal_assistant_agent"

builder.add_conditional_edges(
    "tools",
    route_tools_output,
    {
        "personal_assistant_agent": "personal_assistant_agent",
        "features_agent": "features_agent",
        "deep_research_agent": "deep_research_agent",
        "web_search_agent": "web_search_agent",
    }
)

# Compile with recursion limit
graph = builder.compile(checkpointer=None)

print("Graph compiled successfully.")