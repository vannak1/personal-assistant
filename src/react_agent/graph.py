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
import json # Added for robust ToolMessage content parsing

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import State, InputState # Assuming these are correctly defined elsewhere
from react_agent.tools import TOOLS # Assuming TOOLS = [search] from tools.py
from react_agent.utils import load_chat_model # Assuming this loads your LLM

# Define specific session states (constants)
SESSION_STATE_CHECKING = "session_checking"
SESSION_STATE_WAITING_FOR_NAME = "session_waiting_for_name"

# --- Agent Implementations ---

# Note: The Supervisor Agent logic seems largely integrated into the PA agent now.
# We'll keep the node for potential future separation but focus on the PA's role.
async def supervisor_agent(state: State) -> Dict:
    """
    DEPRECATED / Minimal Supervisor: Primarily acts as a conceptual router if needed,
    but most logic is now in the Personal Assistant for integrated flow.
    This node might be simplified or removed if PA handles all routing decisions.
    For now, it just passes through to the Personal Assistant based on prior logic.
    """
    print(f"--- Running Supervisor Agent (State: {state.active_agent}) ---")
    # This agent's logic is effectively handled by the PA's routing now.
    # If called, it should likely just decide based on the last message type
    # or simply default to the PA.

    last_message = state.messages[-1] if state.messages else None

    # If specialist just finished, PA should present
    if isinstance(last_message, AIMessage) and getattr(last_message, 'name', None) in ["FeaturesAgent", "DeepResearch", "WebSearchAgent"]:
         print("Supervisor: Detected specialist results. Routing to PA.")
         # Clear state potentially set by specialist before PA takes over
         state.active_agent = None
         state.routing_reason = "present_results"
         return {"next": "personal_assistant_agent"}

    # Default routing to PA for any other case handled here
    print("Supervisor: Defaulting to PA agent.")
    # state.active_agent = "personal_assistant" # PA will set its own state
    state.routing_reason = "supervisor_passthrough"
    return {"next": "personal_assistant_agent"}


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
    current_user_name = state.user_name or "User" # Use User as default
    routing_reason = state.routing_reason or "unknown"

    print(f"PA: Routing reason: {routing_reason}. Last message type: {type(last_message).__name__ if last_message else 'None'}, name: {getattr(last_message, 'name', 'N/A')}")

    # --- Specialist Result Presentation Logic ---
    # Check if the last message is from a specialist agent *carrying results*
    # This now expects an AIMessage with results stored in additional_kwargs
    if isinstance(last_message, AIMessage) and getattr(last_message, 'additional_kwargs', {}).get('status') == 'results_obtained':
        specialist_results_msg = last_message
        specialist_type = specialist_results_msg.name
        print(f"PA: Processing results from {specialist_type}")

        # Handling for WebSearchAgent results passed via additional_kwargs
        if specialist_type == "WebSearchAgent":
            # Extract results and original query from the message's additional_kwargs
            original_query = specialist_results_msg.additional_kwargs.get("original_query", "your recent query")
            search_results_raw = specialist_results_msg.additional_kwargs.get("search_results")

            # Format the raw results for the presentation prompt
            formatted_results = f"Search results for query: '{original_query}'\n\n"

            # Ensure search_results_raw is properly parsed if it's a string
            if isinstance(search_results_raw, str):
                try:
                    search_results_raw = json.loads(search_results_raw)
                except json.JSONDecodeError:
                    search_results_raw = [{"title": "Parser Error", "url": "", "content": f"Error parsing search results"}]

            if isinstance(search_results_raw, list) and search_results_raw:
                 # Check if the tool returned its own error structure
                 if isinstance(search_results_raw[0], dict) and search_results_raw[0].get("title") == "Search Error":
                      formatted_results += f"An error occurred during the search: {search_results_raw[0].get('content', 'Unknown error')}"
                 else:
                    for i, result in enumerate(search_results_raw, 1):
                        title = result.get("title", "Untitled")
                        url = result.get("url", "No URL")
                        content = result.get("content", "No content available")
                        formatted_results += f"{i}. {title} ({url})\n   Summary: {content[:250]}...\n\n" # Limit content preview
            elif isinstance(search_results_raw, dict) and search_results_raw.get("title") == "Search Error": # Handle single error dict
                 formatted_results += f"An error occurred during the search: {search_results_raw.get('content', 'Unknown error')}"
            elif search_results_raw: # Handle unexpected format
                 formatted_results += "Results received in an unexpected format.\n" + str(search_results_raw)[:500] # Show partial raw results
            else:
                formatted_results += "No search results were found or returned for the query."

            # Prepare the prompt for the PA's LLM call to synthesize the answer
            search_presentation_prompt = f"""
            You are presenting web search results to the user, {current_user_name}.
            Their original query was: '{original_query}'

            The following search results were retrieved from the web:
            --- BEGIN SEARCH RESULTS ---
            {formatted_results}
            --- END SEARCH RESULTS ---

            Based ONLY on the provided search results and the original query, synthesize a helpful and concise answer for {current_user_name}.
            - Directly address the query using information from the results.
            - If results are relevant, summarize the key findings clearly.
            - Explicitly mention that you looked this up online or based your answer on web search results.
            - If the results indicate an error or no relevant information was found, state that clearly. Do not apologize excessively, just report the outcome. Suggest rephrasing the query if appropriate.
            - Do NOT invent information not present in the results.
            - Use a friendly, conversational, and helpful tone.
            - Cite URLs sparingly, only if they are highly relevant to a specific point made in the summary.
            """

            print("PA: Calling LLM to synthesize search results...")
            # Perform the LLM call HERE to generate the final user-facing response
            # Provide some minimal context from the conversation
            context_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)][-2:] # Last 2 user messages

            response = await model.ainvoke([
                {"role": "system", "content": configuration.personal_assistant_prompt.format(system_time=datetime.now(tz=UTC).isoformat())},
                *context_messages, # Provide recent user context
                {"role": "user", "content": search_presentation_prompt} # The core instruction with results
            ])

            result_message = AIMessage(content=response.content, name="PersonalAssistant")
            result_message.additional_kwargs = {
                "status": "presenting_search_results",
                "specialist_source": specialist_type,
                "original_query": original_query
            }

            # Reset state variables related to specialist flow
            state.active_agent = None # PA is now active by default unless routing elsewhere
            state.routing_reason = None
            state.specialist_results = False # Results presented

            print("PA: Finished presenting search results.")
            return {
                "messages": [result_message],
                "next": "__end__" # Pause for user input
            }

        # --- Standard handling for other specialist agents (Features, DeepResearch) ---
        # Assumes these agents return a final AIMessage with their analysis/content
        elif specialist_type in ["FeaturesAgent", "DeepResearch"]:
             # Prepare a prompt to present the results
            presentation_prompt = f"""
            The {specialist_type} agent has provided the following information regarding the request from {current_user_name}:
            '{specialist_results_msg.content}'

            Present this information back to {current_user_name} in a friendly, conversational way.
            - Summarize or rephrase the specialist's output clearly and concisely.
            - Relate it back to their likely original request if possible.
            - Ask if this meets their needs or if they have further questions.
            """

            print(f"PA: Calling LLM to present {specialist_type} results...")
            context_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)][-2:]

            response = await model.ainvoke([
                {"role": "system", "content": configuration.personal_assistant_prompt.format(system_time=datetime.now(tz=UTC).isoformat())},
                *context_messages,
                {"role": "user", "content": presentation_prompt}
            ])

            result_message = AIMessage(content=response.content, name="PersonalAssistant")
            result_message.additional_kwargs = {
                "status": "presenting_specialist_results",
                "specialist_source": specialist_type
            }

            # Reset state variables
            state.active_agent = None
            state.routing_reason = None
            state.specialist_results = False

            print(f"PA: Finished presenting {specialist_type} results.")
            return {
                "messages": [result_message],
                "next": "__end__" # Pause for user input
            }

    # --- Session & User Management Logic ---
    # A. New user check (triggered by entry point or specific state)
    if state.first_message and not state.user_name:
        print("PA: First message, checking session.")
        state.first_message = False # Mark check as initiated
        state.session_state = SESSION_STATE_CHECKING
        tool_call_id = f"tool_call_{uuid.uuid4()}"
        # This assumes a 'manage_user_session' tool exists and is bound or handled
        return {
            "messages": [AIMessage(
                content="", # No text needed, just the tool call
                tool_calls=[{
                    "name": "manage_user_session",
                    "args": {}, # Check for existing session
                    "id": tool_call_id
                }],
                name="PersonalAssistantSessionToolCall" # Distinguish from regular PA messages if needed
            )],
            "active_agent": "personal_assistant" # PA needs to handle the tool result
        }

    # B. Received session check result (ToolMessage response)
    if isinstance(last_message, ToolMessage) and state.session_state == SESSION_STATE_CHECKING:
        print("PA: Received session tool result.")
        session_info = {}
        try:
            # Tool results might be strings or already dicts
            content = last_message.content
            if isinstance(content, str):
                session_info = json.loads(content)
            elif isinstance(content, dict):
                session_info = content
        except json.JSONDecodeError:
            print(f"PA: Error decoding session info: {last_message.content}")
            session_info = {"user_name": None, "user_uid": None} # Default on error

        found_name = session_info.get("user_name")
        found_uid = session_info.get("user_uid")

        if found_name:
            # Existing user found
            state.user_name = found_name
            state.user_uid = found_uid
            state.session_state = None # Session resolved
            greeting_message = AIMessage(
                content=f"Welcome back, {found_name}! How can I help you today?",
                name="PersonalAssistant"
            )
            greeting_message.additional_kwargs = {"status": "greeting_returning_user"}
            # No longer need active_agent set for session loop
            state.active_agent = None
            return {
                "messages": [greeting_message],
                "next": "__end__" # Wait for user input
            }
        else:
            # New user, ask for name
            state.session_state = SESSION_STATE_WAITING_FOR_NAME
            greeting_message = AIMessage(
                content="Hello there! I'm your personal assistant. It seems we haven't met before. What's your name?",
                name="PersonalAssistant"
            )
            greeting_message.additional_kwargs = {"status": "greeting_new_user"}
            state.active_agent = None # Wait for user response
            return {
                "messages": [greeting_message],
                "next": "__end__" # Wait for user response
            }

    # C. User provided their name (HumanMessage response after asking)
    if isinstance(last_message, HumanMessage) and state.session_state == SESSION_STATE_WAITING_FOR_NAME:
        print("PA: Received user name.")
        extracted_name = last_message.content.strip()
        # Transition back to checking state to *save* the name via the tool
        state.session_state = SESSION_STATE_CHECKING
        tool_call_id = f"tool_call_{uuid.uuid4()}"
        return {
            "messages": [AIMessage(
                content="",
                tool_calls=[{
                    "name": "manage_user_session",
                    "args": {"user_name_to_set": extracted_name}, # Ask tool to save name
                    "id": tool_call_id
                }],
                name="PersonalAssistantSessionToolCall"
            )],
            "active_agent": "personal_assistant" # PA needs to handle tool result (confirmation)
        }

    # --- Routing Logic for new messages (if not handling results or session) ---
    if isinstance(last_message, HumanMessage) and state.user_name: # Ensure session is resolved
        print("PA: New human message, deciding route.")
        user_request = last_message.content

        # Analyze the request to determine routing (Simplified Routing Prompt)
        routing_prompt = f"""
        User ({current_user_name}) request: '{user_request}'.

        Analyze this request. Which agent is BEST suited? Choose ONE:
        - PERSONAL_ASSISTANT: Simple questions, chat, clarification, presenting results, session tasks.
        - FEATURES_AGENT: Planning features, software development tasks, project tracking.
        - DEEP_RESEARCH: In-depth analysis, complex topics requiring detailed investigation beyond a quick search.
        - WEB_SEARCH: Current events, specific facts, quick lookups requiring up-to-date info.

        Respond ONLY with the agent name (e.g., 'WEB_SEARCH').
        """

        # Use a separate, non-tool-bound model instance for routing if preferred, or the main one
        # route_model = load_chat_model(configuration.supervisor_model) # Example: using supervisor model
        route_model = model # Or just use the PA model

        print("PA: Calling LLM for routing decision...")
        # Provide minimal context for routing
        routing_history = [{"role": "system", "content": "You are a routing assistant."}]
        routing_history.append({"role": "user", "content": routing_prompt})

        routing_response = await route_model.ainvoke(routing_history) # Use non-tool-bound if possible
        routing_decision = routing_response.content.strip().upper()
        print(f"PA Routing Decision: {routing_decision}")

        # Prepare context note (common structure)
        context_note_content = f"""
        User: {current_user_name}
        Request: {user_request}
        Routing Reason: This request was identified as needing your specific capabilities.
        """
        context_note = AIMessage(
            content=context_note_content,
            name="PersonalAssistant_ContextNote",
            additional_kwargs={"is_context": True} # Flag for potential filtering
        )

        # Map the decision to the appropriate agent
        if "FEATURES_AGENT" in routing_decision:
            state.routing_reason = "feature_planning_needed"
            state.active_agent = "features_agent" # Set active agent *before* returning
            return {"messages": [context_note], "next": "features_agent"}

        elif "DEEP_RESEARCH" in routing_decision:
            state.routing_reason = "research_needed"
            state.active_agent = "deep_research"
            return {"messages": [context_note], "next": "deep_research_agent"}

        elif "WEB_SEARCH" in routing_decision:
            state.routing_reason = "web_search_needed"
            state.active_agent = "web_search" # Set before routing to web_search
            return {"messages": [context_note], "next": "web_search_agent"}

        else: # Default to PERSONAL_ASSISTANT for direct handling
            print("PA: Handling request directly.")
            state.routing_reason = "direct_handling"
            # Fall through to direct handling logic below
            pass # Let the direct handling logic proceed

    # --- Direct Handling Logic (if PA is chosen or as fallback) ---
    print("PA: Entering direct handling logic.")
    # This block executes if routing decided PA, or if it's a continuation after a tool call for PA itself

    # Check if the last message was a ToolMessage meant for the PA
    if isinstance(last_message, ToolMessage) and state.active_agent == "personal_assistant":
        print("PA: Received tool result for direct handling.")
        # PA called a tool for its own direct task. Now formulate response.
        # The `add_messages` in State should have added the ToolMessage already.
        # We just need to generate the final response based on the history.
        direct_prompt = f"Based on the preceding conversation and the result from the tool ({last_message.name}), provide a response to {current_user_name}."

        history = state.messages # Use the full history including the tool message
        response = cast(AIMessage, await model.ainvoke(history)) # Let the model decide how to respond based on tool result

        response.name = "PersonalAssistant"
        response.additional_kwargs = {"status": "direct_handling_tool_response"}
        state.active_agent = None # Task complete
        return {"messages": [response], "next": "__end__"}

    # If it's not a tool response for PA, and we decided on direct handling, call LLM
    elif state.routing_reason == "direct_handling" and isinstance(last_message, HumanMessage):
        print("PA: Performing direct handling LLM call.")
        direct_prompt = f"""
        User ({current_user_name}): {last_message.content}

        Handle this request directly. Use your tools if necessary. Maintain a helpful, conversational tone.
        """
        # Get relevant history, excluding intermediate context notes if desired
        history = [msg for msg in state.messages if not getattr(msg, 'additional_kwargs', {}).get('is_context', False)][-5:]

        response = cast(AIMessage, await model.ainvoke([
            {"role": "system", "content": configuration.personal_assistant_prompt.format(system_time=datetime.now(tz=UTC).isoformat())},
            *history # Use filtered history
        ])) # Model is tool-bound, so it might return tool_calls

        response.name = "PersonalAssistant"
        response.additional_kwargs = {"status": "direct_handling"}

        if not response.tool_calls:
            print("PA: Direct handling finished (no tool calls).")
            state.active_agent = None
            return {"messages": [response], "next": "__end__"}
        else:
            print("PA: Direct handling requires tools.")
            state.active_agent = "personal_assistant" # Keep PA active for tool loop
            # Return the AIMessage with tool_calls; graph routes to 'tools'
            return {"messages": [response]}
    else:
         # Fallback if state is unclear or no specific logic hit
         print(f"PA: Reached fallback. Current state ambiguous (routing_reason: {routing_reason}, last_message: {type(last_message).__name__}). Ending turn.")
         # Avoid loops - just end the turn if state is unexpected
         return {"next": "__end__"}


async def features_agent(state: State) -> Dict:
    """
    Features Agent: Documents and plans feature requests. Can use tools.
    """
    print("--- Running Features Agent ---")
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.execution_enforcer_model).bind_tools(TOOLS) # Tool-bound
    system_message = configuration.feature_request_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Filter messages: Include relevant history, context notes. Exclude PA/Supervisor turns.
    agent_messages: List[BaseMessage] = [{"role": "system", "content": system_message}]
    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            agent_messages.append(msg)
        elif isinstance(msg, ToolMessage) and state.active_agent == "features_agent": # Include tool results for this agent
             agent_messages.append(msg)
        elif getattr(msg, 'name', None) == 'PersonalAssistant_ContextNote': # Include the context note
             agent_messages.append(msg)
        # Can add logic to include prior AIMessages from this agent if multi-turn needed

    # If last message is ToolMessage, we are responding to it
    if isinstance(state.messages[-1], ToolMessage):
         print("Features Agent: Responding to tool result.")
    else:
         print("Features Agent: Processing initial request or continuing.")

    response = cast(
        AIMessage,
        await model.ainvoke(agent_messages), # Model uses history including tool results if any
    )
    response.name = "FeaturesAgent"
    response.additional_kwargs = {
        "status": "specialist_active",
        "specialist": "FeaturesAgent"
    }

    if not response.tool_calls:
         print("Features Agent: Finished task (no further tool calls).")
         # Optionally process the final response content into state.feature_requests here
         feature_request_summary = {
             "description": response.content,
             "timestamp": datetime.now(tz=UTC).isoformat(),
             "status": "processed"
         }
         current_requests = state.feature_requests if state.feature_requests is not None else []
         updated_requests = current_requests + [feature_request_summary]

         state.active_agent = None # Task finished
         return {
             "messages": [response],
             "feature_requests": updated_requests,
             "next": "personal_assistant_agent" # Route back to PA for presentation
         }
    else:
        print("Features Agent: Requesting tools.")
        # Keep active_agent set ("features_agent"), return message with tool calls
        # Graph will route to 'tools' then back via 'route_tools_output'
        return {"messages": [response]}


async def deep_research_agent(state: State) -> Dict:
    """
    Deep research agent: Provides in-depth information. Can use tools (like search).
    """
    print("--- Running Deep Research Agent ---")
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.deep_research_model).bind_tools(TOOLS) # Tool-bound
    system_message = configuration.deep_research_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Filter messages similarly to Features Agent
    agent_messages: List[BaseMessage] = [{"role": "system", "content": system_message}]
    for msg in state.messages:
         if isinstance(msg, HumanMessage):
            agent_messages.append(msg)
         elif isinstance(msg, ToolMessage) and state.active_agent == "deep_research":
            agent_messages.append(msg)
         elif getattr(msg, 'name', None) == 'PersonalAssistant_ContextNote':
            agent_messages.append(msg)

    if isinstance(state.messages[-1], ToolMessage):
         print("Deep Research Agent: Responding to tool result.")
    else:
         print("Deep Research Agent: Processing initial request or continuing.")

    response = cast(
        AIMessage,
        await model.ainvoke(agent_messages),
    )
    response.name = "DeepResearch"
    response.additional_kwargs = {
        "status": "specialist_active",
        "specialist": "DeepResearch"
    }

    if not response.tool_calls:
        print("Deep Research Agent: Finished task (no further tool calls).")
        state.active_agent = None # Task finished
        # Create a final response message (doesn't need special kwargs like the search agent carrier)
        final_response = AIMessage(
            content=response.content, # Pass final research content
            name="DeepResearch"
            # No need for 'results_obtained' status here, PA handles standard presentation
        )
        return {
            "messages": [final_response],
            "next": "personal_assistant_agent" # Route back to PA for presentation
        }
    else:
        print("Deep Research Agent: Requesting tools.")
        # Keep active_agent set ("deep_research"), return message with tool calls
        return {"messages": [response]}


async def web_search_agent(state: State) -> Dict:
    """
    Web search agent: Handles search queries using the 'search' tool.

    1. Receives request (via PA context note).
    2. Issues a tool call to the 'search' tool.
    3. Receives the ToolMessage result.
    4. Packages the raw results into an AIMessage with additional_kwargs.
    5. Routes back to PA for processing and presentation.
    """
    print("--- Running Web Search Agent ---")
    configuration = Configuration.from_context()
    # Load an LLM for proper assistant message generation to follow OpenAI's message flow expectations
    model = load_chat_model(configuration.web_search_model).bind_tools(TOOLS)

    # --- Extract Original Query ---
    original_user_request = None
    context_note = next((msg for msg in reversed(state.messages) if getattr(msg, 'name', None) == 'PersonalAssistant_ContextNote'), None)

    if context_note and isinstance(context_note.content, str) and "Request:" in context_note.content:
         request_line = [line for line in context_note.content.split('\n') if "Request:" in line]
         if request_line:
              original_user_request = request_line[0].replace("Request:", "").strip()

    # Fallback to last human message if no context note or request extraction failed
    if not original_user_request:
         user_request_message = next((msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)), None)
         if user_request_message:
              original_user_request = user_request_message.content

    search_query = original_user_request if original_user_request else "general information" # Default query
    # Store in conversation context for potential use by PA later
    if "conversation_context" not in state: state.conversation_context = {} # Ensure dict exists
    state.conversation_context["original_query"] = search_query

    # --- Handle Tool Result ---
    last_message = state.messages[-1] if state.messages else None
    if isinstance(last_message, ToolMessage) and state.active_agent == "web_search":
        # We've received search results from the ToolNode.
        print(f"Web Search Agent: Received tool result for call_id {last_message.tool_call_id}")

        # Parse the JSON string from the tool
        search_results_raw = None
        try:
            import json
            search_results_raw = json.loads(last_message.content)
            print(f"Successfully parsed search results: found {len(search_results_raw) if isinstance(search_results_raw, list) else 'non-list'} results")
        except json.JSONDecodeError:
            print(f"Failed to parse search results as JSON, using raw content")
            search_results_raw = [{"title": "Parser Error", "url": "", "content": f"Error parsing search results: {last_message.content[:100]}..."}]

        # Create a carrier message containing the raw results to pass back to PA
        # Create a properly formatted response that follows the required pattern
        # Here we ensure a proper assistant message is generated to respond to the tool message
        system_message = {"role": "system", "content": "You are a web search assistant. Format the search results clearly for the user."}
        user_message = {"role": "user", "content": f"Here are search results for '{search_query}'. Please format them."}

        # Generate a properly formatted response to ensure message flow integrity
        response = await model.ainvoke([system_message, user_message])

        result_carrier_message = AIMessage(
            content=f"Search results obtained for query: '{search_query}'", # Simple placeholder content
            name="WebSearchAgent",
            # Store raw results and query in additional_kwargs for the PA agent
            additional_kwargs={
                "status": "results_obtained", # CRITICAL: PA uses this to identify results
                "specialist": "WebSearchAgent",
                "original_query": search_query,
                "search_results": search_results_raw # Pass the actual results here
            }
        )

        print("Web Search Agent: Finished (results obtained). Routing back to PA.")
        state.active_agent = None # This agent's job (tool call) is done.
        return {
            "messages": [result_carrier_message],
            "next": "personal_assistant_agent" # Explicitly route back to PA
        }

    # --- Initiate Tool Call ---
    else:
        # This is the first step for this agent - initiate the search.
        tool_call_id = f"tool_call_{uuid.uuid4()}"
        print(f"Web Search Agent: Initiating search for '{search_query}' with tool_call_id {tool_call_id}")

        # Create the message that requests the tool call
        search_request = AIMessage(
            content="I'll search for information about that topic.", # Add meaningful content
            tool_calls=[{
                "name": "search", # Must match the tool name in TOOLS
                "args": {"query": search_query},
                "id": tool_call_id
            }],
            name="WebSearchAgent", # Identify the source agent
            additional_kwargs={ # Optional status tracking
                "status": "initiating_search",
                "specialist": "WebSearchAgent",
                "original_query": search_query
            }
        )

        # Ensure the active_agent is explicitly set, even if PA already did it
        state.active_agent = "web_search"

        # Return only the message; the graph routes to 'tools' based on tool_calls
        return {"messages": [search_request]}


# --- Graph Construction ---
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add the nodes
# builder.add_node("supervisor_agent", supervisor_agent) # Optional: keep if supervisor logic remains
builder.add_node("personal_assistant_agent", personal_assistant_agent)
builder.add_node("features_agent", features_agent)
builder.add_node("deep_research_agent", deep_research_agent)
builder.add_node("web_search_agent", web_search_agent)

# Create ToolNode with explicit handling of tool responses
# The ToolNode will execute tools and create corresponding ToolMessages for each tool_call_id
builder.add_node("tools", ToolNode(
    TOOLS,
    # Set to True to properly handle tool execution in async mode consistent with LangChain's patterns
    async_mode=True
))

# Entry point is the Personal Assistant
builder.set_entry_point("personal_assistant_agent")

# --- Routing Logic ---

# Route from Personal Assistant
def route_pa_output(state: State) -> Literal["features_agent", "deep_research_agent", "web_search_agent", "tools", "personal_assistant_agent", "__end__"]:
    """Routes PA output based on 'next' state or tool calls."""
    last_message = state.messages[-1] if state.messages else None

    # If PA needs to call tools (for session mgt or direct handling)
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print("Routing PA: Needs tools.")
        return "tools"

    # If PA explicitly set 'next' to route to a specialist
    next_node = state.next
    if next_node == "features_agent":
        print("Routing PA: To Features Agent.")
        state.next = None # Consume the 'next' state
        return "features_agent"
    elif next_node == "deep_research_agent":
        print("Routing PA: To Deep Research Agent.")
        state.next = None
        return "deep_research_agent"
    elif next_node == "web_search_agent":
        print("Routing PA: To Web Search Agent.")
        state.next = None
        return "web_search_agent"
    elif next_node == "personal_assistant_agent": # Allow looping back to PA if needed (e.g. after session tool call)
        print("Routing PA: Looping back to PA.")
        state.next = None
        return "personal_assistant_agent"
    elif next_node == "__end__":
         print("Routing PA: Ending turn.")
         state.next = None
         return "__end__"

    # Default: End the turn if no other condition met
    print("Routing PA: Defaulting to end turn.")
    return "__end__"

builder.add_conditional_edges(
    "personal_assistant_agent",
    route_pa_output,
    {
        "tools": "tools",
        "features_agent": "features_agent",
        "deep_research_agent": "deep_research_agent",
        "web_search_agent": "web_search_agent",
        "personal_assistant_agent": "personal_assistant_agent", # Allow loops
        "__end__": "__end__"
    }
)

# Route from specialist agents
def route_specialist_output(state: State) -> Literal["tools", "personal_assistant_agent"]:
    """Routes specialist output: to tools if needed, otherwise back to PA."""
    last_message = state.messages[-1] if state.messages else None
    specialist_name = getattr(last_message, 'name', 'Unknown Specialist')

    # If specialist needs tools (e.g., multi-step research)
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(f"Routing Specialist ({specialist_name}): Needs tools.")
        # Assumes the specialist node set state.active_agent correctly
        return "tools"

    # Otherwise, specialist finished or passed results, go back to PA
    # The specialist node should set state.next = "personal_assistant_agent" if it wants PA next
    print(f"Routing Specialist ({specialist_name}): Finished/Passing results. Routing to PA.")
    return "personal_assistant_agent"

# Apply this routing to all specialist nodes
builder.add_conditional_edges("features_agent", route_specialist_output, {"tools": "tools", "personal_assistant_agent": "personal_assistant_agent"})
builder.add_conditional_edges("deep_research_agent", route_specialist_output, {"tools": "tools", "personal_assistant_agent": "personal_assistant_agent"})
builder.add_conditional_edges("web_search_agent", route_specialist_output, {"tools": "tools", "personal_assistant_agent": "personal_assistant_agent"})


# Route from tools node back to the agent that called it
def route_tools_output(state: State) -> str: # Return type is just string (node name)
    """Routes tool output back to the agent stored in state.active_agent."""
    # The last message appended should be the ToolMessage
    last_message = state.messages[-1] if state.messages else None

    # Check if we have a valid ToolMessage with the expected format
    if isinstance(last_message, ToolMessage):
        tool_call_id = getattr(last_message, 'tool_call_id', 'N/A')
        print(f"Routing Tools (tool_call_id: {tool_call_id}): Validating response.")

        # Ensure the ToolMessage has all required fields
        if not hasattr(last_message, 'tool_call_id') or not last_message.tool_call_id:
            print("WARNING: ToolMessage missing tool_call_id! Creating a properly formatted message.")
            # If tool_call_id is missing, this could cause the BadRequestError
            # In a production app, we might want to fix this by creating a properly formatted message
    else:
        print(f"WARNING: Expected ToolMessage but got {type(last_message).__name__}")

    active_agent = state.active_agent
    print(f"Routing Tools: Sending result back to active agent: '{active_agent}'")

    # Map the active_agent state string to the correct node name
    # Ensure state.active_agent is set correctly *before* the tool-calling message is returned
    if active_agent == "personal_assistant":
        return "personal_assistant_agent"
    elif active_agent == "features_agent":
        return "features_agent"
    elif active_agent == "deep_research": # Check consistency: is it 'deep_research' or 'deep_research_agent'?
        # Assuming the node name is 'deep_research_agent' based on add_node
        return "deep_research_agent"
    elif active_agent == "web_search": # Assuming the node name is 'web_search_agent'
         return "web_search_agent"
    else:
        # Fallback if state tracking failed - crucial to avoid getting stuck
        print(f"ERROR: Unknown or missing active agent ('{active_agent}') after tool execution. Routing to PA as fallback.")
        # For safety, route to PA, which is designed to handle ambiguous states.
        return "personal_assistant_agent" # Fallback route

# Define all possible destinations from the tools node
tool_destinations = {
    "personal_assistant_agent": "personal_assistant_agent",
    "features_agent": "features_agent",
    "deep_research_agent": "deep_research_agent",
    "web_search_agent": "web_search_agent",
}
builder.add_conditional_edges(
    "tools",
    route_tools_output,
    tool_destinations # Map returned agent string to node name
)

# Compile the graph
graph = builder.compile(checkpointer=None) # Add checkpointer later if needed for persistence

print("Graph compiled successfully.")

# Example of how to visualize (optional, requires graphviz)
# try:
#     graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
#     print("Graph visualization saved to graph.png")
# except Exception as e:
#     print(f"Could not generate graph visualization: {e}")