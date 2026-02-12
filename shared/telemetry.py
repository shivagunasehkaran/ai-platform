"""Telemetry module for metrics collection and cost tracking."""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from shared.config import PRICE_INPUT_PER_1K, PRICE_OUTPUT_PER_1K

logger = logging.getLogger(__name__)


class Timer:
    """Context manager for measuring execution latency.
    
    Example:
        with Timer() as t:
            do_something()
        print(f"Took {t.elapsed_ms}ms")
    """
    
    def __init__(self) -> None:
        """Initialize timer."""
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_ms: float = 0.0
    
    def __enter__(self) -> "Timer":
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Stop the timer and calculate elapsed time."""
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD for token usage.
    
    Args:
        prompt_tokens: Number of input/prompt tokens.
        completion_tokens: Number of output/completion tokens.
        
    Returns:
        float: Total cost in USD.
    """
    input_cost = (prompt_tokens / 1000) * PRICE_INPUT_PER_1K
    output_cost = (completion_tokens / 1000) * PRICE_OUTPUT_PER_1K
    return input_cost + output_cost


def format_stats(
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
    latency_ms: float,
) -> str:
    """Format usage statistics as a string.
    
    Args:
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens.
        cost: Cost in USD.
        latency_ms: Latency in milliseconds.
        
    Returns:
        str: Formatted statistics string.
    """
    return (
        f"[stats] prompt={prompt_tokens} "
        f"completion={completion_tokens} "
        f"cost=${cost:.6f} "
        f"latency={latency_ms:.0f}ms"
    )


@dataclass
class MetricsStore:
    """Store and persist metrics for chat, retrieval, and agent operations."""
    
    chat_metrics: list[dict] = field(default_factory=list)
    retrieval_metrics: list[dict] = field(default_factory=list)
    agent_metrics: list[dict] = field(default_factory=list)
    
    def log_chat_metrics(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        latency_ms: float,
    ) -> None:
        """Log metrics for a chat completion.
        
        Args:
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
            cost: Cost in USD.
            latency_ms: Latency in milliseconds.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
            "latency_ms": latency_ms,
        }
        self.chat_metrics.append(entry)
        logger.debug(f"Logged chat metrics: {entry}")
    
    def log_retrieval_metrics(
        self,
        query: str,
        latency_ms: float,
        recall: float,
        mrr: float,
    ) -> None:
        """Log metrics for a retrieval operation.
        
        Args:
            query: The search query.
            latency_ms: Latency in milliseconds.
            recall: Recall score (0-1).
            mrr: Mean Reciprocal Rank score.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "latency_ms": latency_ms,
            "recall": recall,
            "mrr": mrr,
        }
        self.retrieval_metrics.append(entry)
        logger.debug(f"Logged retrieval metrics: {entry}")
    
    def log_agent_metrics(
        self,
        task: str,
        success: bool,
        tool_calls: int,
        cost: float,
    ) -> None:
        """Log metrics for an agent task execution.
        
        Args:
            task: Description of the task.
            success: Whether the task completed successfully.
            tool_calls: Number of tool calls made.
            cost: Total cost in USD.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "success": success,
            "tool_calls": tool_calls,
            "cost": cost,
        }
        self.agent_metrics.append(entry)
        logger.debug(f"Logged agent metrics: {entry}")
    
    def save_metrics(self, filepath: str | Path) -> None:
        """Save all metrics to a JSON file, merging with existing data.
        
        Args:
            filepath: Path to save the metrics file.
        """
        filepath = Path(filepath)
        
        # Load existing metrics first to merge
        existing_data = {
            "chat_metrics": [],
            "retrieval_metrics": [],
            "agent_metrics": [],
        }
        
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass  # Use empty if file is corrupted
        
        # Merge existing with new (append new entries)
        merged_data = {
            "chat_metrics": existing_data.get("chat_metrics", []) + self.chat_metrics,
            "retrieval_metrics": existing_data.get("retrieval_metrics", []) + self.retrieval_metrics,
            "agent_metrics": existing_data.get("agent_metrics", []) + self.agent_metrics,
        }
        
        with open(filepath, "w") as f:
            json.dump(merged_data, f, indent=2)
        
        # Clear current session metrics after saving to prevent duplicates
        self.chat_metrics = []
        self.retrieval_metrics = []
        self.agent_metrics = []
        
        logger.info(f"Saved metrics to {filepath}")
    
    def load_metrics(self, filepath: str | Path) -> None:
        """Load metrics from a JSON file.
        
        Args:
            filepath: Path to the metrics file.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"Metrics file not found: {filepath}")
            return
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        self.chat_metrics = data.get("chat_metrics", [])
        self.retrieval_metrics = data.get("retrieval_metrics", [])
        self.agent_metrics = data.get("agent_metrics", [])
        
        logger.info(f"Loaded metrics from {filepath}")


# Global metrics store instance
metrics_store = MetricsStore()
