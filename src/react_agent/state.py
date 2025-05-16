"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated

# Import new memory structures for compatibility
from react_agent.memory import PrimaryAgentMemory, SpecializedAgentMemory, UserProfile, SessionContext, TaskContext


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """


@dataclass
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    
    Note: For the enhanced implementation with differentiated memory, consider using
    PrimaryAgentMemory or SpecializedAgentMemory instead of this class directly.
    """

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    next: Optional[str] = field(default=None)
    """
    Indicates the next node to route to in the hierarchical agent system.
    Used by the supervisor agent to route requests to specialized agents.
    """

    active_agent: Optional[str] = field(default=None)
    """
    Tracks which specialized agent is currently handling the request.
    """

    feature_requests: List[Dict[str, Any]] = field(default_factory=list)
    """
    Stores documented feature requests that have been processed by the Feature Request agent.
    Each feature request is a dictionary with details about the request.
    """

    user_name: Optional[str] = field(default=None)
    """
    Stores the user's name retrieved from the session.
    """

    user_uid: Optional[str] = field(default=None)
    """
    Stores the unique user ID retrieved or generated from the session.
    """

    agent_status: Optional[str] = field(default=None)
    """
    Stores the current status of the agent for UI display purposes.
    Possible values include: thinking, routing, gathering_more_info, etc.
    """

    # New fields for the supervisor-PA architecture
    routing_reason: Optional[str] = field(default=None)
    """
    Stores the reason why the supervisor routed to a specific agent.
    Used to provide context for agents about why they were selected.
    """

    specialist_results: bool = field(default=False)
    """
    Flag indicating if there are specialist results waiting to be presented.
    Used by the PA agent to know when to present results.
    """

    first_message: bool = field(default=True)
    """
    Flag indicating if this is the first message in the conversation.
    Used for session initialization.
    """

    original_question: Optional[str] = field(default=None)
    """
    Stores the user's original question when they first interact with the system
    without having a session. This allows answering their question after onboarding.
    """

    conversation_context: Dict[str, Any] = field(default_factory=dict)
    """
    Stores additional conversation context that might be useful for agents.
    Can include things like detected topics, user preferences, etc.
    """

    # Session management states
    session_state: Optional[str] = field(default=None)
    """
    Tracks the current state of the session management process.
    Possible values: checking, waiting_for_name, etc.
    """
    
    # Fields for human-in-the-loop functionality
    awaiting_approval: List[Dict[str, Any]] = field(default_factory=list)
    """
    Stores tool calls that are awaiting human approval.
    """
    
    collecting_feedback: bool = field(default=False)
    """
    Flag indicating if the system is currently collecting feedback from the user.
    """
    
    # Fields for compatibility with new memory structures
    user_profile: Optional[UserProfile] = field(default=None)
    """
    User profile information for primary agents.
    This field allows compatibility with PrimaryAgentMemory.
    """
    
    session_context: Optional[SessionContext] = field(default=None)
    """
    Session context information for primary agents.
    This field allows compatibility with PrimaryAgentMemory.
    """
    
    task_context: Optional[TaskContext] = field(default=None)
    """
    Task-specific context for specialized agents.
    This field allows compatibility with SpecializedAgentMemory.
    """
    
    # Method to convert to/from memory classes
    def to_primary_memory(self) -> PrimaryAgentMemory:
        """Convert this state to a PrimaryAgentMemory instance."""
        # Create user profile if not exists
        if not self.user_profile:
            self.user_profile = {
                "preferences": {},
                "frequent_requests": [],
                "last_interactions": []
            }
        
        # Create session context if not exists
        if not self.session_context:
            self.session_context = {
                "current_topic": "",
                "active_agents": [],
                "pending_tasks": []
            }
        
        # Convert to PrimaryAgentMemory
        return PrimaryAgentMemory(
            messages=self.messages,
            user_profile=self.user_profile,
            session_context=self.session_context,
            user_name=self.user_name,
            user_uid=self.user_uid,
            agent_status=self.agent_status,
            routing_reason=self.routing_reason,
            specialist_results=self.specialist_results,
            first_message=self.first_message,
            original_question=self.original_question,
            conversation_context=self.conversation_context,
            session_state=self.session_state
        )
    
    def to_specialized_memory(self) -> SpecializedAgentMemory:
        """Convert this state to a SpecializedAgentMemory instance."""
        # Create task context if not exists
        if not self.task_context:
            import time
            self.task_context = {
                "query_details": self.original_question or "",
                "intermediate_results": [],
                "start_time": time.time(),
                "primary_context_ref": self.user_uid
            }
        
        # Convert to SpecializedAgentMemory
        return SpecializedAgentMemory(
            messages=self.messages,
            task_context=self.task_context,
            active_agent=self.active_agent,
            routing_reason=self.routing_reason
        )
    
    @classmethod
    def from_primary_memory(cls, memory: PrimaryAgentMemory) -> State:
        """Create a State instance from a PrimaryAgentMemory object."""
        state = cls(
            messages=memory.messages,
            user_name=memory.get("user_name"),
            user_uid=memory.get("user_uid"),
            agent_status=memory.get("agent_status"),
            routing_reason=memory.get("routing_reason"),
            specialist_results=memory.get("specialist_results", False),
            first_message=memory.get("first_message", True),
            original_question=memory.get("original_question"),
            conversation_context=memory.get("conversation_context", {}),
            session_state=memory.get("session_state")
        )
        
        # Add memory-specific fields
        state.user_profile = memory.get("user_profile")
        state.session_context = memory.get("session_context")
        
        return state
    
    @classmethod
    def from_specialized_memory(cls, memory: SpecializedAgentMemory) -> State:
        """Create a State instance from a SpecializedAgentMemory object."""
        state = cls(
            messages=memory.messages,
            active_agent=memory.get("active_agent"),
            routing_reason=memory.get("routing_reason")
        )
        
        # Add memory-specific fields
        state.task_context = memory.get("task_context")
        
        return state