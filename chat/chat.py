"""CLI chat application with streaming and telemetry."""

import logging
import sys

from shared.config import MODEL_NAME, validate_config
from shared.llm import get_client, stream_chat
from shared.memory import ConversationMemory
from shared.telemetry import Timer, calculate_cost, format_stats, metrics_store

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# System prompt for the assistant
SYSTEM_PROMPT = """You are a helpful AI assistant. Be concise and informative."""


def estimate_tokens(text: str) -> int:
    """Estimate token count using character-based approximation.
    
    A rough approximation: ~4 characters per token for English text.
    
    Args:
        text: The text to estimate tokens for.
        
    Returns:
        int: Estimated token count.
    """
    return max(1, len(text) // 4)


def chat_loop() -> None:
    """Run the main chat loop with streaming responses."""
    # Validate configuration
    if not validate_config():
        print("❌ Error: Missing OPENAI_BASE_URL or OPENAI_API_KEY")
        print("Please set environment variables or create .env file")
        sys.exit(1)
    
    # Initialize memory
    memory = ConversationMemory(max_messages=10)
    
    print(f"🤖 AI Chat (Model: {MODEL_NAME})")
    print("Type 'quit' or 'exit' to end. Press Ctrl+C to interrupt.\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ("quit", "exit", "q"):
                print("\nGoodbye! 👋")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Add user message to memory
            memory.add("user", user_input)
            
            # Build messages list with system prompt and history
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(memory.get_history())
            
            # Estimate prompt tokens
            prompt_text = SYSTEM_PROMPT + " ".join(
                msg["content"] for msg in memory.get_history()
            )
            prompt_tokens = estimate_tokens(prompt_text)
            
            # Stream response with timing
            print("Assistant: ", end="", flush=True)
            
            response_tokens: list[str] = []
            
            with Timer() as timer:
                for token in stream_chat(messages):
                    print(token, end="", flush=True)
                    response_tokens.append(token)
            
            # Complete the response line
            print()
            
            # Assemble full response
            full_response = "".join(response_tokens)
            
            # Add assistant response to memory
            memory.add("assistant", full_response)
            
            # Calculate metrics
            completion_tokens = estimate_tokens(full_response)
            cost = calculate_cost(prompt_tokens, completion_tokens)
            
            # Print stats
            stats = format_stats(prompt_tokens, completion_tokens, cost, timer.elapsed_ms)
            print(f"{stats}\n")
            
            # Log metrics
            metrics_store.log_chat_metrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
                latency_ms=timer.elapsed_ms,
            )
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n❌ Error: {e}\n")


def main() -> None:
    """Entry point for the chat application."""
    try:
        chat_loop()
    finally:
        # Save metrics on exit
        try:
            from shared.config import METRICS_FILE
            metrics_store.save_metrics(METRICS_FILE)
        except Exception:
            pass  # Ignore errors saving metrics


if __name__ == "__main__":
    main()
