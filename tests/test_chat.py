"""Tests for chat functionality."""

import pytest

from shared.memory import ConversationMemory
from shared.telemetry import calculate_cost
from shared.config import PRICE_INPUT_PER_1K, PRICE_OUTPUT_PER_1K


class TestConversationMemory:
    """Tests for ConversationMemory class."""

    def test_memory_limit(self) -> None:
        """Verify that memory respects the 10 message limit."""
        memory = ConversationMemory(max_messages=10)
        
        # Add 15 messages (more than the limit)
        for i in range(15):
            memory.add("user", f"Message {i}")
        
        # Should only have 10 messages
        history = memory.get_history()
        assert len(history) == 10
        
        # Should have the most recent 10 messages (5-14)
        assert history[0]["content"] == "Message 5"
        assert history[-1]["content"] == "Message 14"

    def test_memory_add_and_retrieve(self) -> None:
        """Verify messages are added and retrieved correctly."""
        memory = ConversationMemory(max_messages=10)
        
        memory.add("user", "Hello")
        memory.add("assistant", "Hi there!")
        
        history = memory.get_history()
        
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi there!"}

    def test_memory_clear(self) -> None:
        """Verify memory can be cleared."""
        memory = ConversationMemory(max_messages=10)
        
        memory.add("user", "Hello")
        memory.add("assistant", "Hi!")
        
        assert len(memory) == 2
        
        memory.clear()
        
        assert len(memory) == 0
        assert memory.get_history() == []


class TestCostCalculation:
    """Tests for cost calculation."""

    def test_cost_calculation(self) -> None:
        """Verify cost formula: input=$0.005/1K, output=$0.015/1K."""
        # Test with known values
        prompt_tokens = 1000
        completion_tokens = 1000
        
        cost = calculate_cost(prompt_tokens, completion_tokens)
        
        # Expected: (1000/1000 * 0.005) + (1000/1000 * 0.015) = 0.02
        expected_cost = 0.005 + 0.015
        assert cost == pytest.approx(expected_cost)

    def test_cost_calculation_zero_tokens(self) -> None:
        """Verify cost is zero when no tokens used."""
        cost = calculate_cost(0, 0)
        assert cost == 0.0

    def test_cost_calculation_only_prompt(self) -> None:
        """Verify cost with only prompt tokens."""
        cost = calculate_cost(2000, 0)
        
        # Expected: (2000/1000 * 0.005) = 0.01
        expected_cost = 2 * PRICE_INPUT_PER_1K
        assert cost == pytest.approx(expected_cost)

    def test_cost_calculation_only_completion(self) -> None:
        """Verify cost with only completion tokens."""
        cost = calculate_cost(0, 2000)
        
        # Expected: (2000/1000 * 0.015) = 0.03
        expected_cost = 2 * PRICE_OUTPUT_PER_1K
        assert cost == pytest.approx(expected_cost)

    def test_cost_calculation_realistic(self) -> None:
        """Verify cost with realistic token counts."""
        # Typical chat: ~500 prompt tokens, ~200 completion tokens
        prompt_tokens = 500
        completion_tokens = 200
        
        cost = calculate_cost(prompt_tokens, completion_tokens)
        
        # Expected: (500/1000 * 0.005) + (200/1000 * 0.015)
        # = 0.0025 + 0.003 = 0.0055
        expected_cost = (500 / 1000 * PRICE_INPUT_PER_1K) + (200 / 1000 * PRICE_OUTPUT_PER_1K)
        assert cost == pytest.approx(expected_cost)
