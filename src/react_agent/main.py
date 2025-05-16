"""Main integration module for the personal assistant.

This module provides the complete implementation of the personal assistant,
integrating all the components into a single, cohesive system.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import asyncio
import uuid

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from react_agent.configuration import Configuration
from react_agent.memory import create_memory_for_agent
from react_agent.tool_registry import create_tool_registry
from react_agent.memory_manager import MemoryManager
from react_agent.supervisor import create_supervisor_system
from react_agent.human_loop import add_human_in_the_loop, get_default_approval_configuration
from react_agent.utils import load_chat_model


def create_personal_assistant(
    configuration: Optional[Configuration] = None,
    model: Optional[BaseLanguageModel] = None,
    embeddings = None,
    vector_store = None,
    checkpointer = None,
    store = None,
    with_human_in_loop: bool = True
) -> tuple:
    """Create a complete personal assistant system.
    
    Args:
        configuration: Optional configuration (will be loaded from context if not provided)
        model: Optional base language model (will be loaded from config if not provided)
        embeddings: Optional embeddings model (will be created if not provided)
        vector_store: Optional vector store (will be created if not provided)
        checkpointer: Optional checkpointer for state persistence (will be created if not provided)
        store: Optional store for additional data storage (will be created if not provided)
        with_human_in_loop: Whether to include human-in-the-loop capabilities
        
    Returns:
        Tuple of (supervisor_system, memory_manager)
    """
    # Load configuration if not provided
    if configuration is None:
        configuration = Configuration.from_context()
    
    # Create model if not provided
    if model is None:
        model = load_chat_model(configuration.supervisor_model)
    
    # Create embeddings if not provided
    if embeddings is None:
        try:
            embeddings = OpenAIEmbeddings()
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            embeddings = None
    
    # Create vector store if not provided
    if vector_store is None and embeddings is not None:
        try:
            vector_store = Chroma(embedding_function=embeddings)
        except Exception as e:
            print(f"Error creating vector store: {e}")
            vector_store = None
    
    # Create checkpointer if not provided
    if checkpointer is None:
        checkpointer = InMemorySaver()
    
    # Create store if not provided
    if store is None:
        store = InMemoryStore()
    
    # Create tool registry
    tool_registry = create_tool_registry()
    
    # Create memory manager
    memory_manager = MemoryManager(
        vector_store=vector_store,
        embeddings=embeddings,
        checkpointer=checkpointer,
        store=store
    )
    
    # Create the supervisor system
    supervisor_system = create_supervisor_system(
        model=model,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        configuration=configuration,
        checkpointer=checkpointer,
        store=store
    )
    
    # Add human-in-the-loop capabilities if requested
    if with_human_in_loop:
        supervisor_system = add_human_in_the_loop(
            graph=supervisor_system,
            approval_required_for=get_default_approval_configuration()
        )
    
    return supervisor_system, memory_manager


async def process_user_query(
    supervisor, 
    memory_manager, 
    thread_id: str, 
    user_input: str
) -> Dict[str, Any]:
    """Process a user query through the personal assistant.
    
    Args:
        supervisor: The supervisor system
        memory_manager: The memory manager
        thread_id: Thread ID for this conversation
        user_input: User input to process
        
    Returns:
        Result dictionary from processing the query
    """
    # Get existing state or create new state
    state = memory_manager.get_thread_state(thread_id)
    
    if state is None:
        # Initial state based on the router agent's memory needs
        state = create_memory_for_agent("router")
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Add the user input to the messages
    user_message = HumanMessage(content=user_input)
    state["messages"].append(user_message)
    
    # Invoke the system
    result = await supervisor.ainvoke(state, config=config)
    
    # Save the updated state
    memory_manager.save_thread_state(thread_id, result)
    
    return result


async def main() -> None:
    """Run an example conversation with the personal assistant."""
    # Create the personal assistant
    supervisor, memory_manager = create_personal_assistant()
    
    # Use a thread ID to maintain state across interactions
    thread_id = f"user123_thread_{uuid.uuid4()}"
    
    # Process a user query
    user_input = "Can you help me research the latest developments in AI and then create a simple React component to display that information?"
    
    print(f"User: {user_input}")
    
    result = await process_user_query(supervisor, memory_manager, thread_id, user_input)
    
    # Process and display results
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            print(f"Assistant: {message.content}")
        elif isinstance(message, HumanMessage):
            if message.content != user_input:  # Skip the original input
                print(f"Human: {message.content}")
    
    # Example of handling human-in-the-loop
    if result.get("awaiting_approval"):
        human_approval = input("Do you approve? (yes/no): ")
        if "yes" in human_approval.lower():
            next_result = await process_user_query(supervisor, memory_manager, thread_id, "approve")
            
            for message in next_result["messages"]:
                if isinstance(message, AIMessage):
                    print(f"Assistant: {message.content}")
    
    # Example of handling feedback
    if result.get("collecting_feedback"):
        feedback = input("Feedback: ")
        next_result = await process_user_query(supervisor, memory_manager, thread_id, feedback)
        
        for message in next_result["messages"]:
            if isinstance(message, AIMessage):
                print(f"Assistant: {message.content}")


if __name__ == "__main__":
    asyncio.run(main())