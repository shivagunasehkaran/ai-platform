"""Memory module for conversation history management."""

import logging
from collections import deque
from typing import Literal

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant"]


class ConversationMemory:
    """Manages conversation history with a fixed-size buffer.
    
    Uses a deque to maintain a rolling window of messages,
    automatically discarding oldest messages when capacity is reached.
    
    Example:
        memory = ConversationMemory(max_messages=10)
        memory.add("user", "Hello!")
        memory.add("assistant", "Hi there!")
        history = memory.get_history()
    """
    
    def __init__(self, max_messages: int = 10) -> None:
        """Initialize conversation memory.
        
        Args:
            max_messages: Maximum number of messages to retain.
        """
        self._max_messages = max_messages
        self._messages: deque[dict] = deque(maxlen=max_messages)
        logger.debug(f"Initialized ConversationMemory with max_messages={max_messages}")
    
    def add(self, role: Role, content: str) -> None:
        """Add a message to the conversation history.
        
        Args:
            role: The role of the message sender ('system', 'user', or 'assistant').
            content: The message content.
        """
        message = {"role": role, "content": content}
        self._messages.append(message)
        logger.debug(f"Added message: role={role}, length={len(content)}")
    
    def get_history(self) -> list[dict]:
        """Get the full conversation history.
        
        Returns:
            list[dict]: List of message dictionaries with 'role' and 'content' keys.
        """
        return list(self._messages)
    
    def clear(self) -> None:
        """Clear all messages from the conversation history."""
        self._messages.clear()
        logger.debug("Cleared conversation history")
    
    def __len__(self) -> int:
        """Return the number of messages in history."""
        return len(self._messages)
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"ConversationMemory(messages={len(self)}, max={self._max_messages})"
