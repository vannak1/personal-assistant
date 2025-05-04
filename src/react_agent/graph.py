"""Define a hierarchical multi-agent system.

This implements a supervisor agent that routes requests to specialized agent teams:
1. Feature Request Agent: Documents feature requests and pain points
2. Deep Research Agent: Provides in-depth research on complex topics
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, TypedDict, cast
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import State, InputState
from react_agent.tools import TOOLS, manage_user_session # Import the specific tool
from react_agent.utils import load_chat_model

# Define specific states for clarity in the PA logic
PA_STATE_CHECKING_SESSION = "personal_assistant_checking_session"
PA_STATE_WAITING_FOR_NAME = "personal_assistant_waiting_for_name"
PA_STATE_PROCESSING = "personal_assistant_processing"

# Personal Assistant Agent Implementation (Replaces Supervisor)
async def personal_assistant_agent(state: State) -> Dict:
    \"\"\"
    The main agent interacting with the user, grounded in Kaizen.
    Handles session check, name gathering, simple requests directly, builds rapport, 
    and routes complex requests to specialized agents.
    \"\"\"
    print(f"--- Running Personal Assistant Agent (State: {state.active_agent}) ---")
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model).bind_tools(TOOLS)
    
    last_message = state.messages[-1]
    current_user_name = state.user_name

    # --- Session & Name Handling Logic --- 

    # A. Just received user input, need to check session/name if unknown
    if isinstance(last_message, HumanMessage) and not current_user_name and state.active_agent != PA_STATE_WAITING_FOR_NAME:
        print("PA: New input, user name unknown. Checking session.")
        # Call the session tool to check for existing session
        tool_call = manage_user_session.invoke({"user_name_to_set": None})
        # We need to manually create the ToolMessage structure expected by the graph
        # In a real LangGraph setup, this might be handled differently, but for simulation:
        # This direct call bypasses graph's tool node, so we process result immediately.
        # Let's adjust to use the graph's tool calling mechanism instead.
        return {
            "messages": [AIMessage( # Create an AI message requesting the tool
                content="", 
                tool_calls=[{
                    "name": "manage_user_session", 
                    "args": {}, 
                    "id": f"tool_call_{uuid.uuid4()}"
                }],
                name="PersonalAssistant"
            )],
            "active_agent": PA_STATE_CHECKING_SESSION # Mark state as checking session
            # 'next' will be determined by the edge routing based on tool call
        }

    # B. Received result from initial session check
    if isinstance(last_message, ToolMessage) and state.active_agent == PA_STATE_CHECKING_SESSION:
        print("PA: Received session check result.")
        session_info = last_message.content # Assuming content is the dict from the tool
        if isinstance(session_info, str):
            try: # Handle potential stringified JSON if ToolNode stringifies
                import json
                session_info = json.loads(session_info)
            except json.JSONDecodeError:
                 print(f"PA: Error decoding session tool result: {session_info}")
                 # Fallback: Ask for name
                 session_info = {"user_name": None, "user_uid": None}

        found_name = session_info.get("user_name")
        found_uid = session_info.get("user_uid")

        if found_name:
            print(f"PA: Session found for {found_name}. Updating state.")
            state.user_name = found_name
            state.user_uid = found_uid
            # Now proceed to process the *original* user message (which is the second to last)
            original_user_message = next((msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)), None)
            if original_user_message:
                # Re-enter the processing logic, but now with the name set
                state.active_agent = PA_STATE_PROCESSING 
                # Fall through to the main processing logic below
                print("PA: Proceeding with original request.")
                # *** The rest of the function below handles the actual request ***
            else:
                 # Should not happen in normal flow, but handle gracefully
                 print("PA: Could not find original user message after session check.")
                 # Create welcome back message with status
                 welcome_message = AIMessage(
                     content=f"Welcome back, {found_name}! What can I help you with today?", 
                     name="PersonalAssistant"
                 )
                 welcome_message.additional_kwargs = {"status": "greeting_returning_user"}
                 
                 return {"messages": [welcome_message], "active_agent": None, "next": "__end__"}
        else:
            print("PA: No session found. Asking for user's name.")
            # Ask for name
            # Create greeting message with status
            greeting_message = AIMessage(
                content="Hello there! I'm your personal assistant. I don't believe we've met, what's your name?", 
                name="PersonalAssistant"
            )
            greeting_message.additional_kwargs = {"status": "greeting_new_user"}
            
            return {
                "messages": [greeting_message],
                "active_agent": PA_STATE_WAITING_FOR_NAME, # Mark state as waiting for name
                "next": "__end__" # Wait for user input
            }

    # C. Waiting for name, and received user input
    if isinstance(last_message, HumanMessage) and state.active_agent == PA_STATE_WAITING_FOR_NAME:
        print("PA: Received input, likely the user's name.")
        # Simple extraction: assume the message *is* the name (can be improved)
        extracted_name = last_message.content.strip()
        print(f"PA: Attempting to set name to: {extracted_name}")
        # Call tool to save the name and get UID
        return {
            "messages": [AIMessage( # Create an AI message requesting the tool
                content="", 
                tool_calls=[{
                    "name": "manage_user_session", 
                    "args": {"user_name_to_set": extracted_name}, 
                    "id": f"tool_call_{uuid.uuid4()}"
                }],
                name="PersonalAssistant"
            )],
            "active_agent": PA_STATE_CHECKING_SESSION # Reuse state, tool result handles update
        }

    # D. Received result after *setting* the name
    # This case is handled by B, as active_agent is PA_STATE_CHECKING_SESSION again.
    # If name was successfully set, B will find it and proceed.

    # --- Main Request Processing Logic (Only runs if name is known) ---
    current_user_name = state.user_name # Re-check state after potential update
    if not current_user_name:
        # This should only happen if something went wrong in the name-getting flow
        print("PA: ERROR - Name still unknown, ending flow.")
        # Or force asking for name again
        return {
            "messages": [AIMessage(content="Sorry, something went wrong with getting your name. Could you please tell me your name?", name="PersonalAssistant")],
            "active_agent": PA_STATE_WAITING_FOR_NAME,
            "next": "__end__"
        }

    # Get the actual user request message (might not be the last message if session check happened)
    user_request_message = next((msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)), None)
    if not user_request_message:
        print("PA: Could not find user request message.")
        # Maybe the interaction was just setting the name
        return {"messages": [AIMessage(content=f"Nice to meet you, {current_user_name}! How can I help?", name="PersonalAssistant")], "active_agent": None, "next": "__end__"}

    print(f"PA: Processing request for {current_user_name}: '{user_request_message.content[:50]}...'")
    state.active_agent = PA_STATE_PROCESSING # Ensure state is correct

    # --- Existing PA Logic (Routing / Direct Handling) --- 
    # (Slightly modified to use current_user_name and user_request_message)

    # If the last message is an AIMessage from a completed specialist agent, synthesize and respond.
    if isinstance(last_message, AIMessage) and state.active_agent is None and last_message.name != "PersonalAssistant":
        print("PA: Specialist agent finished. Synthesizing response.")
        synthesis_prompt = f\"A specialist provided this information: '{last_message.content}'. Please present this back to {current_user_name} in a helpful and friendly tone, maintaining your Personal Assistant persona.\"
        pa_model = load_chat_model(configuration.model)
        final_response = await pa_model.ainvoke([
            {\"role\": \"system\", \"content\": configuration.personal_assistant_prompt.format(system_time=datetime.now(tz=UTC).isoformat())},
            {\"role\": \"user\", \"content\": synthesis_prompt}
        ])
        
        # Create the response with status info
        synthesized_response = AIMessage(content=final_response.content, name="PersonalAssistant")
        synthesized_response.additional_kwargs = {
            "status": "specialist_complete",
            "specialist": last_message.name
        }
        
        return {
            "messages": [synthesized_response],
            "next": "__end__"
        }

    # If it's a ToolMessage for the PA itself (for direct handling)
    if isinstance(last_message, ToolMessage) and state.active_agent == "personal_assistant": # Note: state might be PA_STATE_PROCESSING now
        print("PA: Received tool result for direct handling. Continuing.")
        system_message = configuration.personal_assistant_prompt.format(system_time=datetime.now(tz=UTC).isoformat())
        response = cast(AIMessage, await model.ainvoke([{"role": "system", "content": system_message}, *state.messages]))
        response.name = "PersonalAssistant"
        
        # Add status information for UI
        if not hasattr(response, "additional_kwargs") or response.additional_kwargs is None:
            response.additional_kwargs = {}
        response.additional_kwargs["status"] = "tool_usage"
        if not response.tool_calls:
            print("PA: Finished handling directly after tool use.")
            return {"messages": [response], "active_agent": None, "next": "__end__"}
        else:
            print("PA: Needs more tools after initial tool use.")
            return {"messages": [response], "active_agent": "personal_assistant"} # Keep PA active for tool loop

    # --- Decision Logic (Handle Directly or Route) --- 
    system_message = configuration.personal_assistant_prompt.format(system_time=datetime.now(tz=UTC).isoformat())
    decision_prompt = f\"User ({current_user_name}) request: '{user_request_message.content}'.\n\nAssess this request and our conversation history to determine what approach to take:\n\n1. GATHER_MORE_INFO: If you need more details about goals, constraints, preferences, or examples before proceeding.\n\n2. HANDLE_DIRECTLY: If it's a straightforward request you can handle with available information.\n\n3. DISCUSS_EXECUTION_APPROACH: If it involves planning or feature development that would benefit from specialist help, but you should first discuss the approach with the user.\n\n4. DISCUSS_RESEARCH_APPROACH: If it requires in-depth research that would benefit from specialist help, but you should first discuss the approach with the user.\n\nRespond ONLY with 'GATHER_MORE_INFO', 'HANDLE_DIRECTLY', 'DISCUSS_EXECUTION_APPROACH', or 'DISCUSS_RESEARCH_APPROACH'. Do not add any other text.\"
    
    decision_model = load_chat_model(configuration.model)
    # Provide relevant history for decision
    decision_history = [msg for msg in state.messages if not isinstance(msg, ToolMessage)] # Exclude tool messages for brevity
    decision_response = await decision_model.ainvoke([
        {\"role\": \"system\", \"content\": system_message},
        *decision_history[:-1], # History up to the last user message
        {\"role\": \"user\", \"content\": decision_prompt}
    ])
    decision = decision_response.content.strip()
    print(f"PA Decision: {decision}")

    # Get the number of messages from this user to determine conversation stage
    user_message_count = sum(1 for msg in state.messages if isinstance(msg, HumanMessage))
    first_interaction = user_message_count <= 2  # First or second message from user

    if "GATHER_MORE_INFO" in decision:
        print("PA: Need to gather more information.")
        # Create a prompt specifically for asking clarifying questions in a conversational way
        clarity_prompt = f\"System time: {datetime.now(tz=UTC).isoformat()}\nUser name: {current_user_name}\n\nThe user's request requires more clarity. In a friendly, conversational tone, ask specific questions to better understand their needs. Focus on one question at a time. If this is one of your first interactions, start with a warm greeting. Ask about:\n1. What they're hoping to accomplish\n2. Any preferences or requirements they have\n3. How urgent this is for them\n4. Any similar experiences they've had before\n\nMake the conversation feel natural and supportive, not like an interrogation.\"
        
        response = cast(AIMessage, await model.ainvoke([{"role": "system", "content": configuration.personal_assistant_prompt + "\n" + clarity_prompt}, *state.messages]))
        response.name = "PersonalAssistant"
        
        # Add status information for UI
        if not hasattr(response, "additional_kwargs") or response.additional_kwargs is None:
            response.additional_kwargs = {}
        response.additional_kwargs["status"] = "gathering_more_info"
        return {"messages": [response], "active_agent": None, "next": "__end__"}
        
    elif "DISCUSS_EXECUTION_APPROACH" in decision:
        print("PA: Discussing execution approach before routing.")
        
        # Create a prompt for discussing the approach and confirming with user
        discussion_prompt = f\"System time: {datetime.now(tz=UTC).isoformat()}\nUser name: {current_user_name}\n\nBefore handing off to a specialist, discuss your approach with the user in a friendly tone:\n\n1. Acknowledge their request about '{user_request_message.content[:50]}...'\n2. Explain that their request involves planning or implementation that would benefit from specialist expertise\n3. Briefly share how you think a plan should be developed\n4. Ask if that approach sounds good to them\n5. Explicitly mention that you'll hand off to a specialist once they confirm\n\nUse language like 'I can connect you with our planning specialist who can create a detailed implementation plan for this. Does that sound helpful?'\"
        
        response = cast(AIMessage, await model.ainvoke([{"role": "system", "content": configuration.personal_assistant_prompt + "\n" + discussion_prompt}, *state.messages]))
        response.name = "PersonalAssistant"
        
        # Add status information for UI and set state flag to indicate we're waiting for confirmation
        response.additional_kwargs = {
            "awaiting_execution_confirmation": True,
            "status": "discussing_approach",
            "specialist_type": "execution"
        }
        return {"messages": [response], "active_agent": None, "next": "__end__"}
        
    elif "DISCUSS_RESEARCH_APPROACH" in decision:
        print("PA: Discussing research approach before routing.")
        
        # Create a prompt for discussing the approach and confirming with user
        discussion_prompt = f\"System time: {datetime.now(tz=UTC).isoformat()}\nUser name: {current_user_name}\n\nBefore handing off to a specialist, discuss your approach with the user in a friendly tone:\n\n1. Acknowledge their question about '{user_request_message.content[:50]}...'\n2. Explain that their question requires in-depth research that would benefit from specialist expertise\n3. Briefly share what key aspects you think should be researched\n4. Ask if that approach sounds good to them\n5. Explicitly mention that you'll hand off to a research specialist once they confirm\n\nUse language like 'I can connect you with our research specialist who can provide comprehensive information on this topic. Would that be helpful?'\"
        
        response = cast(AIMessage, await model.ainvoke([{"role": "system", "content": configuration.personal_assistant_prompt + "\n" + discussion_prompt}, *state.messages]))
        response.name = "PersonalAssistant"
        
        # Add status information for UI and set state flag to indicate we're waiting for confirmation
        response.additional_kwargs = {
            "awaiting_research_confirmation": True,
            "status": "discussing_approach",
            "specialist_type": "research"
        }
        return {"messages": [response], "active_agent": None, "next": "__end__"}
        
    # Check if user confirmed a specialist handoff (look for confirmation in the last message)
    elif isinstance(last_message, HumanMessage):
        # Check if the previous AI message was awaiting confirmation
        previous_ai_messages = [msg for msg in reversed(state.messages[:-1]) if isinstance(msg, AIMessage)]
        if previous_ai_messages and previous_ai_messages[0].additional_kwargs:
            if previous_ai_messages[0].additional_kwargs.get("awaiting_execution_confirmation"):
                # Check if user's response indicates agreement
                confirmation_prompt = f\"Does the message '{last_message.content}' indicate that the user AGREES to have their request handled by a specialist for execution planning? Respond with only YES or NO.\"
                confirmation_response = await decision_model.ainvoke([
                    {\"role\": \"system\", \"content\": \"You analyze messages to determine if they indicate agreement or disagreement.\"},
                    {\"role\": \"user\", \"content\": confirmation_prompt}
                ])
                if "YES" in confirmation_response.content.upper():
                    print("PA: User confirmed execution specialist. Routing to Execution Enforcer.")
                    # Send a handoff message first
                    handoff_message = f"Thanks for confirming, {current_user_name}! I'll connect you with our planning specialist now who will create a detailed implementation plan for you. They'll have all the context from our conversation."
                    state.messages.append(AIMessage(content=handoff_message, name="PersonalAssistant"))
                    
                    # Prepare comprehensive context for the specialist
                    execution_context = f"""
User: {current_user_name}
Request: {user_request_message.content}

SPECIALIST INSTRUCTIONS:
You are receiving this request because it requires planning, feature development, or turning ideas into actions.
Based on our conversation, we have gathered these key details:
- User has provided sufficient context about their goals and requirements
- This appears to be a well-defined task requiring execution planning
- Please create a detailed, actionable plan addressing all aspects of the request
"""
                    # Add execution context as a system note (not visible to user)
                    state.messages.append(AIMessage(
                        content=execution_context, 
                        name="PersonalAssistant_ContextNote",
                        additional_kwargs={"is_context": True, "status": "specialist_routing", "specialist": "ExecutionEnforcer"}
                    ))
                    return {"next": "execution_enforcer_agent", "active_agent": "execution_enforcer"}
                
            elif previous_ai_messages[0].additional_kwargs.get("awaiting_research_confirmation"):
                # Check if user's response indicates agreement
                confirmation_prompt = f\"Does the message '{last_message.content}' indicate that the user AGREES to have their question handled by a research specialist? Respond with only YES or NO.\"
                confirmation_response = await decision_model.ainvoke([
                    {\"role\": \"system\", \"content\": \"You analyze messages to determine if they indicate agreement or disagreement.\"},
                    {\"role\": \"user\", \"content\": confirmation_prompt}
                ])
                if "YES" in confirmation_response.content.upper():
                    print("PA: User confirmed research specialist. Routing to Deep Research Agent.")
                    # Send a handoff message first
                    handoff_message = f"Great, {current_user_name}! I'll connect you with our research specialist now who will provide comprehensive information on this topic. They'll have all the context from our conversation."
                    state.messages.append(AIMessage(content=handoff_message, name="PersonalAssistant"))
                    
                    # Prepare comprehensive context for the specialist
                    research_context = f"""
User: {current_user_name}
Request: {user_request_message.content}

SPECIALIST INSTRUCTIONS:
You are receiving this request because it requires in-depth research or complex information gathering.
Based on our conversation, we have gathered these key details:
- User has provided sufficient context about what information they need
- This appears to be a well-defined request requiring thorough research
- Please provide comprehensive, accurate information addressing all aspects of the query
"""
                    # Add research context as a system note (not visible to user)
                    state.messages.append(AIMessage(
                        content=research_context, 
                        name="PersonalAssistant_ContextNote",
                        additional_kwargs={"is_context": True, "status": "specialist_routing", "specialist": "DeepResearch"}
                    ))
                    return {"next": "deep_research_agent", "active_agent": "deep_research"}
    
    # If we reach here, either it's a direct handling case or the user didn't confirm specialist
    print("PA: Handling request directly.")
    # Use user name in prompt if desired, or just proceed
    direct_handling_prompt = f\"System time: {datetime.now(tz=UTC).isoformat()}\nUser name: {current_user_name}\" # Example context
    response = cast(AIMessage, await model.ainvoke([{"role": "system", "content": configuration.personal_assistant_prompt + "\n" + direct_handling_prompt}, *state.messages]))
    response.name = "PersonalAssistant"
    
    # Add status information for UI
    if not hasattr(response, "additional_kwargs") or response.additional_kwargs is None:
        response.additional_kwargs = {}
    response.additional_kwargs["status"] = "thinking_complete"
    if not response.tool_calls:
        print("PA: Finished handling directly.")
        return {"messages": [response], "active_agent": None, "next": "__end__"}
    else:
        print("PA: Handling directly, needs tools.")
        return {"messages": [response], "active_agent": "personal_assistant"} # PA handles its own tools


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
    
    # Filter messages: Include history, context notes, but exclude general PA messages
    agent_messages = []
    context_notes = []
    
    for msg in state.messages:
        # Collect context notes separately
        if getattr(msg, 'name', None) == 'PersonalAssistant_ContextNote':
            context_notes.append(msg)
        # Include all user messages and specialist messages, but not PA responses
        elif isinstance(msg, HumanMessage) or getattr(msg, 'name', None) not in ['PersonalAssistant']:
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
    
    # Add status information for UI
    if not hasattr(response, "additional_kwargs") or response.additional_kwargs is None:
        response.additional_kwargs = {}
    response.additional_kwargs["status"] = "specialist_active"
    response.additional_kwargs["specialist"] = "ExecutionEnforcer"

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

    # Filter messages: Include history, context notes, but exclude general PA messages
    agent_messages = []
    context_notes = []
    
    for msg in state.messages:
        # Collect context notes separately
        if getattr(msg, 'name', None) == 'PersonalAssistant_ContextNote':
            context_notes.append(msg)
        # Include all user messages and specialist messages, but not PA responses
        elif isinstance(msg, HumanMessage) or getattr(msg, 'name', None) not in ['PersonalAssistant']:
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
    
    # Add status information for UI
    if not hasattr(response, "additional_kwargs") or response.additional_kwargs is None:
        response.additional_kwargs = {}
    response.additional_kwargs["status"] = "specialist_active"
    response.additional_kwargs["specialist"] = "DeepResearch"

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
def route_pa_output(state: State) -> Literal["tools", "execution_enforcer_agent", "deep_research_agent", "__end__"]:
    \"\"\"Routes PA output based on whether it needs tools, is routing, or ending.\"\"\"
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # PA needs to call tools (either for session check or direct handling)
        return "tools"
    # If PA decided to route, 'next' field is set
    if state.next in ["execution_enforcer_agent", "deep_research_agent", "__end__"]:
        return state.next
    # Default fallback or if PA finished directly without tools
    return "__end__"

builder.add_conditional_edges(
    "personal_assistant_agent",
    route_pa_output,
    {
        "tools": "tools", # Added route for PA direct handling needing tools
        "execution_enforcer_agent": "execution_enforcer_agent",
        "deep_research_agent": "deep_research_agent",
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
    # Handle the specific states used by PA for session checking
    if active_agent in [PA_STATE_CHECKING_SESSION, PA_STATE_WAITING_FOR_NAME, "personal_assistant", PA_STATE_PROCESSING]:
        return "personal_assistant_agent"
    elif active_agent == "execution_enforcer":
        return "execution_enforcer_agent"
    elif active_agent == "deep_research":
        return "deep_research_agent"
    else:
        print(f"ERROR: Unknown active agent '{active_agent}' after tool execution. Routing to PA.")
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
