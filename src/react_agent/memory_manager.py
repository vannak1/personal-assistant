"""Memory manager for different memory types and persistence strategies.

This module provides a memory manager to handle different memory types
and persistence strategies for the multi-agent system.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import uuid
import json

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from react_agent.memory import UserProfile, SessionContext, TaskContext


class MemoryManager:
    """Manager for different memory types and persistence strategies.
    
    This class handles the storage, retrieval, and management of different
    types of memory for the multi-agent system.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embeddings: Optional[Embeddings] = None,
        checkpointer = None,
        store = None
    ):
        """Initialize a new memory manager.
        
        Args:
            vector_store: Optional vector store for semantic search
            embeddings: Optional embeddings model for vector encoding
            checkpointer: Optional checkpointer for state persistence
            store: Optional store for additional data storage
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.checkpointer = checkpointer
        self.store = store
        
        # In-memory fallback if no vector store is provided
        self._memory_store: Dict[str, Dict[str, Any]] = {}
    
    async def save_memory(
        self,
        agent_type: str,
        user_id: str,
        content: str,
        context: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a memory to the appropriate store with agent-specific settings.
        
        Args:
            agent_type: Type of agent this memory belongs to
            user_id: User ID for this memory
            content: The memory content to save
            context: Context for this memory
            metadata: Additional metadata for this memory
            
        Returns:
            ID of the saved memory
        """
        metadata = metadata or {}
        
        # Get namespace for this agent type
        namespace = self._get_namespace_for_agent(agent_type, user_id)
        
        # Set TTL based on agent type
        ttl = self._get_memory_ttl(agent_type)
        expiration = datetime.now() + timedelta(seconds=ttl)
        
        memory_id = str(uuid.uuid4())
        
        # Store with appropriate metadata
        if self.vector_store and self.embeddings:
            # Create embedding
            embedding = await self.embeddings.aembed_query(content)
            
            # Store with vector store
            try:
                memory_id = await self.vector_store.aadd(
                    embedding=embedding,
                    document=content,
                    metadata={
                        **metadata,
                        "context": context,
                        "agent_type": agent_type,
                        "timestamp": datetime.now().isoformat(),
                        "expiration": expiration.isoformat(),
                        "user_id": user_id
                    },
                    namespace=namespace
                )
            except Exception as e:
                print(f"Error storing in vector store: {e}")
                # Fall back to in-memory storage
                self._store_in_memory(namespace, memory_id, content, context, metadata, agent_type, user_id, expiration)
        else:
            # Use in-memory storage
            self._store_in_memory(namespace, memory_id, content, context, metadata, agent_type, user_id, expiration)
        
        return memory_id
    
    def _store_in_memory(
        self, 
        namespace: str, 
        memory_id: str, 
        content: str, 
        context: str, 
        metadata: Dict[str, Any],
        agent_type: str,
        user_id: str,
        expiration: datetime
    ) -> None:
        """Store memory in the in-memory fallback store.
        
        Args:
            namespace: Namespace to store the memory in
            memory_id: ID for the memory
            content: Content of the memory
            context: Context for the memory
            metadata: Additional metadata
            agent_type: Type of agent
            user_id: User ID
            expiration: Expiration time
        """
        if namespace not in self._memory_store:
            self._memory_store[namespace] = {}
        
        self._memory_store[namespace][memory_id] = {
            "content": content,
            "metadata": {
                **metadata,
                "context": context,
                "agent_type": agent_type,
                "timestamp": datetime.now().isoformat(),
                "expiration": expiration.isoformat(),
                "user_id": user_id
            }
        }
    
    async def search_memory(
        self,
        agent_type: str,
        user_id: str,
        query: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search memory with agent-specific scoping and strategies.
        
        Args:
            agent_type: Type of agent to search memories for
            user_id: User ID to search memories for
            query: Search query
            filter_metadata: Additional metadata filters
            limit: Maximum number of results to return
            
        Returns:
            List of matching memories
        """
        filter_metadata = filter_metadata or {}
        
        # Get namespace for this agent type
        namespace = self._get_namespace_for_agent(agent_type, user_id)
        
        # Apply default filters for this agent type
        combined_filter = {
            **filter_metadata,
            **self._get_default_filter_for_agent(agent_type)
        }
        
        # Perform search with appropriate strategy
        if self.vector_store and self.embeddings:
            try:
                # Get search strategy for this agent type
                search_strategy = self._get_search_strategy(agent_type)
                
                results = await self.vector_store.asimilarity_search(
                    query=query,
                    k=limit,
                    filter=combined_filter,
                    namespace=namespace,
                    search_type=search_strategy
                )
                
                return results
            except Exception as e:
                print(f"Error searching vector store: {e}")
                # Fall back to in-memory search
                return self._search_in_memory(namespace, query, combined_filter, limit)
        else:
            # Use in-memory search
            return self._search_in_memory(namespace, query, combined_filter, limit)
    
    def _search_in_memory(
        self, 
        namespace: str, 
        query: str, 
        filter_metadata: Dict[str, Any], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search the in-memory fallback store.
        
        Args:
            namespace: Namespace to search in
            query: Search query
            filter_metadata: Metadata to filter by
            limit: Maximum number of results
            
        Returns:
            List of matching memories
        """
        if namespace not in self._memory_store:
            return []
        
        # Very simple search implementation - just check if query is in content
        results = []
        for memory_id, memory in self._memory_store[namespace].items():
            # Check if memory matches filter
            metadata_match = True
            for key, value in filter_metadata.items():
                if key not in memory["metadata"] or memory["metadata"][key] != value:
                    metadata_match = False
                    break
            
            if metadata_match and query.lower() in memory["content"].lower():
                results.append({
                    "id": memory_id,
                    "content": memory["content"],
                    "metadata": memory["metadata"]
                })
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_thread_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve thread state from the checkpointer.
        
        Args:
            thread_id: ID of the thread to retrieve state for
            
        Returns:
            Thread state or None if not found
        """
        if self.checkpointer:
            try:
                return self.checkpointer.get(thread_id)
            except Exception as e:
                print(f"Error retrieving thread state: {e}")
        return None
    
    def save_thread_state(self, thread_id: str, state: Dict[str, Any]) -> None:
        """Save thread state to the checkpointer.
        
        Args:
            thread_id: ID of the thread to save state for
            state: State to save
        """
        if self.checkpointer:
            try:
                self.checkpointer.put(thread_id, state)
            except Exception as e:
                print(f"Error saving thread state: {e}")
    
    def _get_namespace_for_agent(self, agent_type: str, user_id: str) -> str:
        """Get appropriate namespace for agent type.
        
        Args:
            agent_type: Type of agent
            user_id: User ID
            
        Returns:
            Namespace string
        """
        if agent_type in ["router", "personal_assistant"]:
            return f"{user_id}:long-term"
        else:
            return f"{user_id}:{agent_type}:short-term"
    
    def _get_memory_ttl(self, agent_type: str) -> int:
        """Get appropriate TTL (in seconds) for agent memory.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            TTL in seconds
        """
        HOUR = 3600
        DAY = 24 * HOUR
        WEEK = 7 * DAY
        MONTH = 30 * DAY
        
        if agent_type in ["router", "personal_assistant"]:
            return 6 * MONTH  # Long-term memory
        elif agent_type == "research":
            return 1 * WEEK   # Medium-term memory
        elif agent_type in ["website", "feature_request"]:
            return 1 * DAY    # Short-term memory
        else:
            return 1 * HOUR   # Very short-term memory
    
    def _get_default_filter_for_agent(self, agent_type: str) -> Dict[str, Any]:
        """Get default filter for each agent type.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            Filter dictionary
        """
        now = datetime.now()
        
        if agent_type == "router":
            # Router only needs high-level summary info
            return {
                "content_type": "summary",
                "timestamp": {"$gt": (now - timedelta(days=30)).isoformat()}
            }
        elif agent_type == "personal_assistant":
            # Personal assistant needs comprehensive history
            return {
                "timestamp": {"$gt": (now - timedelta(days=90)).isoformat()}
            }
        # Add filters for other agent types
        return {}
    
    def _get_search_strategy(self, agent_type: str) -> str:
        """Get search strategy for each agent type.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            Search strategy string
        """
        if agent_type == "research":
            return "mmr"  # Maximum Marginal Relevance for diversity
        elif agent_type == "personal_assistant":
            return "similarity_score_threshold"  # Similarity with threshold
        else:
            return "similarity"  # Default semantic similarity