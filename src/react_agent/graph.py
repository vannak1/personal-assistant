"""Define a hierarchical multi-agent system.

This implements a personal assistant agent as the main supervisor that routes requests to specialized agent teams:
1. Personal Assistant Agent: Handles user interaction, simple queries, and supervises routing
2. Features Agent: Documents and tracks feature requests
3. Deep Research Agent: Provides in-depth research on complex topics
4. Web Search Agent: Handles web searches and information retrieval
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
    if isinstance(last_message, AIMessage) and last_message.name in ["FeaturesAgent", "DeepResearch", "WebSearchAgent"]:
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
    2. FEATURES_AGENT: For planning, implementation, project breakdown, feature tracking, or feasibility analysis.
    3. DEEP_RESEARCH: For in-depth information, complex analysis, or comprehensive research.
    4. WEB_SEARCH: For current events, fact checking, or search-specific queries.

    Respond ONLY with 'PERSONAL_ASSISTANT', 'FEATURES_AGENT', 'DEEP_RESEARCH', or 'WEB_SEARCH'.
    
    Before responding, apply the NLU process to break down the task:
    1. Intent Recognition: What is the primary intent?
    2. Entity Extraction: What key entities are involved?
    3. Task Decomposition: How can this be broken down?
    4. Goal Analysis: What is the underlying goal?
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
    if "FEATURES_AGENT" in routing_decision:
        # Prepare context for features agent
        features_context = f"""
        User: {current_user_name}
        Request: {user_request_message.content}
        
        SPECIALIST INSTRUCTIONS:
        You are receiving this request because it requires feature documentation, planning, or development.
        Based on our analysis, this request needs detailed implementation planning and tracking.
        """
        # Add context note for the specialist
        state.messages.append(AIMessage(
            content=features_context,
            name="Supervisor_ContextNote",
            additional_kwargs={"is_context": True}
        ))
        
        state.routing_reason = "feature_planning_needed"
        return {
            "next": "features_agent", 
            "active_agent": "features_agent"
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
        
    elif "WEB_SEARCH" in routing_decision:
        # Prepare context for web search
        search_context = f"""
        User: {current_user_name}
        Request: {user_request_message.content}
        
        SPECIALIST INSTRUCTIONS:
        You are receiving this request because it requires web search or current information.
        Based on our analysis, this request needs up-to-date information from the web.
        """
        # Add context note for the specialist
        state.messages.append(AIMessage(
            content=search_context,
            name="Supervisor_ContextNote",
            additional_kwargs={"is_context": True}
        ))
        
        state.routing_reason = "web_search_needed"
        return {
            "next": "web_search_agent", 
            "active_agent": "web_search"
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
    session management, result presentation, and routing to specialist agents.
    This agent now acts as the primary supervisor for the system.
    """
    print(f"--- Running Personal Assistant Agent (State: {state.active_agent}) ---")
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.personal_assistant_model).bind_tools(TOOLS)

    last_message = state.messages[-1]
    current_user_name = state.user_name
    routing_reason = state.routing_reason or "unknown"

    print(f"PA: Received task with routing reason: {routing_reason}")

    # --- Routing Logic for new messages ---
    # If this is a new human message and we're not in session management, analyze for routing
    if isinstance(last_message, HumanMessage) and not state.session_state and state.user_name:
        # Get user request for routing analysis
        user_request = last_message.content
        current_user_name = state.user_name or "User"

        # Analyze the request to determine routing
        routing_prompt = f"""
        User ({current_user_name}) request: '{user_request}'.

        Based on this request, determine which specialized agent should handle it:

        1. PERSONAL_ASSISTANT: For user interaction, simple questions, direct handling, small talk, capability explanations, or result presentation.
        2. FEATURES_AGENT: For planning, implementation, project breakdown, feature tracking, or feasibility analysis.
        3. DEEP_RESEARCH: For in-depth information, complex analysis, or comprehensive research.
        4. WEB_SEARCH: For current events, fact checking, or search-specific queries.

        Respond ONLY with 'PERSONAL_ASSISTANT', 'FEATURES_AGENT', 'DEEP_RESEARCH', or 'WEB_SEARCH'.

        Before responding, analyze the task:
        1. Intent Recognition: What is the primary intent?
        2. Entity Extraction: What key entities are involved?
        3. Task Decomposition: How can this be broken down?
        4. Goal Analysis: What is the underlying goal?
        """

        # Get relevant history for context
        history = [msg for msg in state.messages if not isinstance(msg, ToolMessage)][-5:]  # Last 5 non-tool messages

        routing_response = await model.ainvoke([
            {"role": "system", "content": configuration.personal_assistant_prompt.format(system_time=datetime.now(tz=UTC).isoformat())},
            *history[:-1],  # Previous messages for context
            {"role": "user", "content": routing_prompt}
        ])

        routing_decision = routing_response.content.strip().upper()
        print(f"PA Routing Decision: {routing_decision}")

        # Map the decision to the appropriate agent
        if "FEATURES_AGENT" in routing_decision:
            # Prepare context for features agent
            features_context = f"""
            User: {current_user_name}
            Request: {user_request}

            SPECIALIST INSTRUCTIONS:
            You are receiving this request because it requires feature documentation, planning, or development.
            Based on our analysis, this request needs detailed implementation planning and tracking.
            """
            # Add context note for the specialist
            state.messages.append(AIMessage(
                content=features_context,
                name="PersonalAssistant_ContextNote",
                additional_kwargs={"is_context": True}
            ))

            state.routing_reason = "feature_planning_needed"
            state.active_agent = "features_agent"
            return {
                "next": "features_agent"
            }

        elif "DEEP_RESEARCH" in routing_decision:
            # Prepare context for deep research
            research_context = f"""
            User: {current_user_name}
            Request: {user_request}

            SPECIALIST INSTRUCTIONS:
            You are receiving this request because it requires in-depth research or complex information gathering.
            Based on our analysis, this request needs comprehensive research and explanation.
            """
            # Add context note for the specialist
            state.messages.append(AIMessage(
                content=research_context,
                name="PersonalAssistant_ContextNote",
                additional_kwargs={"is_context": True}
            ))

            state.routing_reason = "research_needed"
            state.active_agent = "deep_research"
            return {
                "next": "deep_research_agent"
            }

        elif "WEB_SEARCH" in routing_decision:
            # Prepare context for web search
            search_context = f"""
            User: {current_user_name}
            Request: {user_request}

            SPECIALIST INSTRUCTIONS:
            You are receiving this request because it requires web search or current information.
            Based on our analysis, this request needs up-to-date information from the web.
            """
            # Add context note for the specialist
            state.messages.append(AIMessage(
                content=search_context,
                name="PersonalAssistant_ContextNote",
                additional_kwargs={"is_context": True}
            ))

            state.routing_reason = "web_search_needed"
            state.active_agent = "web_search"
            return {
                "next": "web_search_agent"
            }

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
                "next": "__end__" # Wait for user input before continuing
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
    # Check if the last message is from a specialist agent
    if isinstance(last_message, AIMessage) and last_message.name in ["FeaturesAgent", "DeepResearch", "WebSearchAgent"]:
        # Specialist has completed its work, process results
        specialist_results = last_message
        specialist_type = specialist_results.name

        # Special handling for WebSearchAgent results
        if specialist_type == "WebSearchAgent":
            # Extract the original query from additional_kwargs or state context
            original_query = specialist_results.additional_kwargs.get("original_query")
            if not original_query and "original_query" in state.conversation_context:
                original_query = state.conversation_context.get("original_query")

            # Check if the response content contains the ORIGINAL_QUERY marker
            response_content = specialist_results.content
            if "ORIGINAL_QUERY:" in response_content:
                parts = response_content.split("ORIGINAL_QUERY:")
                search_results = parts[0].strip()
                if len(parts) > 1 and not original_query:
                    original_query = parts[1].strip()
            else:
                search_results = response_content

            if original_query:
                # Prepare a specialized prompt for processing web search results
                search_presentation_prompt = f"""
                The user originally asked: '{original_query}'

                The Web Search Agent found these results:
                '{search_results}'

                Based on both the original query and the search results, provide a comprehensive answer that:
                1. Directly addresses the user's question using the search results
                2. Synthesizes information from multiple sources if available
                3. Provides a nuanced, thoughtful response that goes beyond summarizing
                4. Uses a warm, conversational tone appropriate for {current_user_name}
                5. Cites sources when appropriate
                6. Addresses any limitations in the search results

                IMPORTANT: Make sure to reference that you searched the web for this information.
                """

                # Process the search results to generate a comprehensive response
                response = await model.ainvoke([
                    {"role": "system", "content": configuration.personal_assistant_prompt.format(system_time=datetime.now(tz=UTC).isoformat())},
                    {"role": "user", "content": search_presentation_prompt}
                ])

                # Ensure we mention searching the web if not already included
                content = response.content
                if not any(phrase in content.lower() for phrase in ["i searched", "searched the web", "search results", "found online"]):
                    content = f"Based on my web search about '{original_query}', I found the following information:\n\n{content}"

                result_message = AIMessage(content=content, name="PersonalAssistant")
                result_message.additional_kwargs = {
                    "status": "presenting_search_results",
                    "specialist_source": specialist_type,
                    "original_query": original_query
                }

                # Reset the active agent back to personal assistant
                state.active_agent = "personal_assistant"

                return {
                    "messages": [result_message],
                    "next": "__end__" # Pause for user input before continuing
                }

        # Standard handling for other specialist agents
        # Prepare a prompt to present the results
        presentation_prompt = f"""
        The {specialist_type} agent has provided this information:
        '{specialist_results.content}'

        Present this information back to {current_user_name} in a friendly, conversational way:
        1. Use a warm tone and natural language
        2. Relate the information to their original request
        3. Ask if this meets their needs or if they need any clarification
        4. Offer to help with next steps
        5. Apply psychological techniques (mirroring, active listening, positive reinforcement)
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

        # Reset the active agent back to personal assistant
        state.active_agent = "personal_assistant"

        return {
            "messages": [result_message],
            "next": "__end__" # Pause for user input before continuing
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
            "next": "__end__" # Wait for user input before continuing
        }
    
    # Direct handling prompt
    direct_prompt = f"""
    User ({current_user_name}): {user_request_message.content}
    
    Handle this request directly in a friendly, conversational manner. You can use tools if needed.
    Remember to apply psychological techniques to build rapport and understand deeper motivations.
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
            "next": "__end__" # Wait for user input before continuing
        }
    else:
        # Tools needed for direct handling
        return {
            "messages": [response],
            "active_agent": "personal_assistant"  # Keep PA active for tool loop
        }


# Features Agent Implementation
async def features_agent(state: State) -> Dict:
    """
    Features Agent: Documents and plans feature requests.
    """
    print("--- Running Features Agent ---")
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
    response.name = "FeaturesAgent" # Name the response
    
    # Add status information
    response.additional_kwargs = {
        "status": "specialist_active",
        "specialist": "FeaturesAgent"
    }

    if not response.tool_calls:
         print("Features Agent: Finished (no tool calls).")
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
        print("Features Agent: Requesting tools.")
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


# Web Search Agent Implementation
async def web_search_agent(state: State) -> Dict:
    """
    Web search agent that handles search queries and information retrieval.

    When search results are found, the agent will pass the results along with the original
    query back to the personal assistant for further processing.

    This agent uses the proper tool calling flow to ensure responses match tool calls.
    """
    print("--- Running Web Search Agent ---")
    configuration = Configuration.from_context()
    # Use the specific model for this agent WITH tool binding
    model = load_chat_model(configuration.deep_research_model).bind_tools(TOOLS)

    # Create a custom prompt for web search
    web_search_prompt = """You are a Web Search Agent specializing in finding accurate and current information.

    Your primary role is to:
    1. Formulate effective search queries based on user requests
    2. Process and present search results in a clear, organized format
    3. Verify information when possible using multiple sources
    4. Provide direct answers with citations to sources

    Present your findings in a clear, structured format with:
    - A direct answer to the user's question
    - Supporting evidence or details
    - Source citations (URLs)
    - Any limitations or caveats about the information

    IMPORTANT: Use the 'search' tool to look up information. DO NOT attempt to answer without searching first.

    System time: {system_time}"""

    # Filter messages: Include history, context notes, but exclude general messages
    agent_messages = []
    context_notes = []
    original_user_request = None

    for msg in state.messages:
        # Collect context notes separately
        if getattr(msg, 'name', None) in ['Supervisor_ContextNote', 'PersonalAssistant_ContextNote']:
            context_notes.append(msg)
            # Extract the original user request from context notes
            if original_user_request is None and hasattr(msg, 'content'):
                content = msg.content
                if isinstance(content, str) and "Request:" in content:
                    request_line = [line for line in content.split('\n') if "Request:" in line]
                    if request_line:
                        original_user_request = request_line[0].replace("Request:", "").strip()
        # Include all user messages, but not PA or Supervisor responses
        elif isinstance(msg, HumanMessage) or (isinstance(msg, AIMessage) and
                                             getattr(msg, 'name', None) not in ['PersonalAssistant', 'Supervisor']):
            agent_messages.append(msg)
            # If we haven't found a request yet, use the last human message
            if original_user_request is None and isinstance(msg, HumanMessage):
                original_user_request = msg.content

    # Store the original query in the state for later use
    if original_user_request:
        state.conversation_context["original_query"] = original_user_request

    search_query = original_user_request if original_user_request else "unknown query"

    # Check if this is a ToolMessage response to a previous search request
    last_message = state.messages[-1]
    if isinstance(last_message, ToolMessage):
        # We've received search results, now we need to process them
        search_results_raw = last_message.content

        # Format the search results into a readable format
        search_results = "Search results for: " + search_query + "\n\n"

        if search_results_raw:
            try:
                # Convert string to dict if needed
                if isinstance(search_results_raw, str):
                    import json
                    search_results_raw = json.loads(search_results_raw)

                # Format the results
                for i, result in enumerate(search_results_raw, 1):
                    title = result.get("title", "Untitled")
                    url = result.get("url", "")
                    content = result.get("content", "")

                    search_results += f"{i}. {title}\n"
                    search_results += f"   URL: {url}\n"
                    search_results += f"   {content[:200]}...\n\n"
            except (json.JSONDecodeError, TypeError):
                search_results += "Error processing search results format. Raw results: " + str(search_results_raw)
        else:
            search_results = "No search results found for your query."

        # Add context notes at the beginning for better context
        agent_messages = context_notes + agent_messages

        # Prepare a prompt for processing the search results
        search_processing_prompt = f"""
        The user asked: '{search_query}'

        I searched the web and found these results:

        {search_results}

        Based on these search results, provide a comprehensive answer to the user's question.
        Include relevant information from the search results and cite sources when appropriate.
        """

        # Process the search results into a coherent response
        response = cast(
            AIMessage,
            await model.ainvoke(
                [{"role": "system", "content": web_search_prompt.format(system_time=datetime.now(tz=UTC).isoformat())},
                 *agent_messages,
                 {"role": "user", "content": search_processing_prompt}]
            ),
        )

        # Create a new message with the processed results
        result_message = AIMessage(content=response.content, name="WebSearchAgent")

        # Add status information and the original query
        result_message.additional_kwargs = {
            "status": "specialist_active",
            "specialist": "WebSearchAgent",
            "original_query": original_user_request
        }

        print("Web Search Agent: Finished (search results processed).")
        return {
            "messages": [result_message],
            "next": "supervisor_agent" # Return to supervisor for presentation
        }
    else:
        # This is the first step - we need to initiate the search
        # Send a message with a tool call to the search tool
        tool_call_id = f"tool_call_{uuid.uuid4()}"

        search_request = AIMessage(
            content="",
            tool_calls=[{
                "name": "search",
                "args": {"query": search_query},
                "id": tool_call_id
            }],
            name="WebSearchAgent"
        )
        search_request.additional_kwargs = {
            "status": "searching",
            "specialist": "WebSearchAgent",
            "original_query": original_user_request
        }

        print(f"Web Search Agent: Initiating search for '{search_query}' with tool_call_id {tool_call_id}")
        return {
            "messages": [search_request],
            "active_agent": "web_search"  # Maintain agent state for tool response
        }


# Build the hierarchical multi-agent graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add the nodes
builder.add_node("supervisor_agent", supervisor_agent)
builder.add_node("personal_assistant_agent", personal_assistant_agent)
builder.add_node("features_agent", features_agent)
builder.add_node("deep_research_agent", deep_research_agent)
builder.add_node("web_search_agent", web_search_agent)
builder.add_node("tools", ToolNode(TOOLS))

# Entry point is now the Personal Assistant
builder.add_edge("__start__", "personal_assistant_agent")

# Routing from Personal Assistant
def route_pa_output(state: State) -> Literal["features_agent", "deep_research_agent", "web_search_agent", "supervisor_agent", "tools", "personal_assistant_agent", "__end__"]:
    """Routes PA output to the appropriate specialist agent or ends the conversation turn."""
    last_message = state.messages[-1]

    # If PA needs to call tools
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # If PA wants to explicitly end the turn for user input
    if state.next == "__end__":
        return "__end__"

    # If PA is handling session management
    if state.session_state in [SESSION_STATE_CHECKING, SESSION_STATE_WAITING_FOR_NAME]:
        return "personal_assistant_agent"

    # If PA wants to route to a specialist agent
    if state.next == "features_agent":
        return "features_agent"
    elif state.next == "deep_research_agent":
        return "deep_research_agent"
    elif state.next == "web_search_agent":
        return "web_search_agent"
    elif state.next == "supervisor_agent":
        return "supervisor_agent"

    # Default is to end the turn and wait for user input
    return "__end__"

builder.add_conditional_edges(
    "personal_assistant_agent",
    route_pa_output,
    {
        "tools": "tools",
        "personal_assistant_agent": "personal_assistant_agent",
        "features_agent": "features_agent",
        "deep_research_agent": "deep_research_agent",
        "web_search_agent": "web_search_agent",
        "supervisor_agent": "supervisor_agent",
        "__end__": "__end__"
    }
)

# Routing from specialist agents
def route_specialist_output(state: State) -> Literal["tools", "personal_assistant_agent"]:
    """Routes specialist output to tools or back to personal assistant."""
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Specialist needs tools
        return "tools"

    # Otherwise, back to personal assistant for presenting results
    return "personal_assistant_agent"

builder.add_conditional_edges(
    "features_agent",
    route_specialist_output,
    {"tools": "tools", "personal_assistant_agent": "personal_assistant_agent"}
)

builder.add_conditional_edges(
    "deep_research_agent",
    route_specialist_output,
    {"tools": "tools", "personal_assistant_agent": "personal_assistant_agent"}
)

builder.add_conditional_edges(
    "web_search_agent",
    route_specialist_output,
    {"tools": "tools", "personal_assistant_agent": "personal_assistant_agent"}
)

# Edge from tools back to the active agent
def route_tools_output(state: State) -> Literal["personal_assistant_agent", "features_agent", "deep_research_agent", "web_search_agent"]:
    """Routes tool output back to the agent that called the tools."""
    active_agent = state.active_agent
    print(f"Tools output: Routing back to active agent: {active_agent}")

    if active_agent == "personal_assistant":
        return "personal_assistant_agent"
    elif active_agent == "features_agent":
        return "features_agent"
    elif active_agent == "deep_research":
        return "deep_research_agent"
    elif active_agent == "web_search":
        return "web_search_agent"
    else:
        # Fallback to PA since it's now the supervisor
        print(f"ERROR: Unknown active agent '{active_agent}' after tool execution. Routing to PA.")
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

# Compile the graph
graph = builder.compile(name="Multi-Agent Personal Assistant")