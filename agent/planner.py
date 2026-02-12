"""Autonomous planning agent with tool calling for travel itineraries."""

import json
import logging
import sys
from datetime import datetime, timedelta

from shared.llm import get_client
from shared.config import MODEL_NAME
from shared.telemetry import Timer, calculate_cost, metrics_store

from agent.tools import TOOL_REGISTRY, TOOL_DEFINITIONS
from agent.schemas import Itinerary, ItineraryDay, FlightResult, HotelResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Maximum iterations to prevent infinite loops
MAX_ITERATIONS = 10


SYSTEM_PROMPT = """You are an expert travel planner assistant. Your job is to create detailed travel itineraries within the user's budget.

When planning a trip:
1. First, search for flights to the destination
2. Then, search for hotels within the remaining budget
3. Check weather forecasts for the travel dates
4. Find attractions and activities
5. Create a day-by-day itinerary that fits the budget

Always be cost-conscious and try to stay within budget. Include a mix of paid attractions and free activities.

After gathering all information, provide a complete itinerary summary in JSON format with:
- Total costs breakdown
- Day-by-day schedule
- Budget comparison

If the trip cannot be done within budget, explain why and suggest alternatives."""


def log_reasoning(message: str) -> None:
    """Log agent reasoning to console."""
    print(f"[reasoning] {message}")
    logger.debug(f"Reasoning: {message}")


def log_tool_call(name: str, arguments: dict) -> None:
    """Log tool call to console."""
    args_str = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    print(f"[tool_call] {name}({args_str})")
    logger.info(f"Tool call: {name}({args_str})")


def log_tool_result(name: str, result: str) -> None:
    """Log tool result summary to console."""
    # Truncate long results
    summary = result[:200] + "..." if len(result) > 200 else result
    print(f"[tool_result] {name} → {summary}")
    logger.debug(f"Tool result: {name} → {result}")


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return JSON result.
    
    Args:
        name: Tool name.
        arguments: Tool arguments.
        
    Returns:
        str: JSON string of tool result.
    """
    if name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {name}"})
    
    try:
        tool_fn = TOOL_REGISTRY[name]
        result = tool_fn(**arguments)
        
        # Convert Pydantic models to dict
        if isinstance(result, list):
            result = [r.model_dump() if hasattr(r, 'model_dump') else r for r in result]
        elif hasattr(result, 'model_dump'):
            result = result.model_dump()
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return json.dumps({"error": str(e)})


def parse_trip_request(prompt: str) -> dict:
    """Extract trip parameters from user prompt.
    
    Args:
        prompt: User's trip request.
        
    Returns:
        dict: Parsed parameters (destination, days, budget, etc.)
    """
    # Default values
    params = {
        "destination": "Auckland",
        "origin": "Wellington",
        "days": 2,
        "budget": 500,
        "start_date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
    }
    
    prompt_lower = prompt.lower()
    
    import re
    
    # Parse destination
    cities = ["auckland", "wellington", "christchurch", "queenstown", "rotorua"]
    for city in cities:
        if city in prompt_lower:
            params["destination"] = city.title()
            break
    
    # Parse duration (e.g., "2-day", "3 day", "2 days")
    days_match = re.search(r"(\d+)[- ]?days?", prompt_lower)
    if days_match:
        params["days"] = int(days_match.group(1))
    
    # Parse budget (e.g., "$500", "NZ$500", "500 dollars", "under 500")
    # Look for currency patterns first
    budget_match = re.search(r"(?:nz)?\s*\$\s*(\d+)", prompt_lower)
    if not budget_match:
        # Try "under X" or "budget X" patterns
        budget_match = re.search(r"(?:under|budget|within)\s+(\d+)", prompt_lower)
    if not budget_match:
        # Try "X dollars" pattern
        budget_match = re.search(r"(\d{3,})\s*(?:dollars?|nzd)?", prompt_lower)
    
    if budget_match:
        params["budget"] = int(budget_match.group(1))
    
    return params


def run_agent(prompt: str) -> dict:
    """Run the planning agent with tool calling.
    
    Args:
        prompt: User's trip planning request.
        
    Returns:
        dict: Final itinerary or error.
    """
    client = get_client()
    
    # Parse trip parameters
    params = parse_trip_request(prompt)
    log_reasoning(f"Parsed request: {params['days']}-day trip to {params['destination']} with ${params['budget']} budget")
    
    # Build enhanced prompt with parsed parameters
    enhanced_prompt = f"""{prompt}

Trip details:
- Destination: {params['destination']}
- Origin: {params['origin']}
- Duration: {params['days']} days
- Budget: NZ${params['budget']}
- Start date: {params['start_date']}

Please plan this trip using the available tools."""

    # Initialize conversation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": enhanced_prompt},
    ]
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    tool_calls_count = 0
    
    with Timer() as total_timer:
        for iteration in range(MAX_ITERATIONS):
            log_reasoning(f"Iteration {iteration + 1}/{MAX_ITERATIONS}")
            
            # Call LLM with tools
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )
            
            # Track tokens
            if response.usage:
                total_prompt_tokens += response.usage.prompt_tokens
                total_completion_tokens += response.usage.completion_tokens
            
            message = response.choices[0].message
            
            # Check if model wants to call tools
            if message.tool_calls:
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in message.tool_calls:
                    name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    log_tool_call(name, arguments)
                    tool_calls_count += 1
                    
                    result = execute_tool(name, arguments)
                    log_tool_result(name, result)
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
            
            else:
                # Model finished - return final response
                log_reasoning("Agent completed planning")
                
                # Calculate cost
                cost = calculate_cost(total_prompt_tokens, total_completion_tokens)
                
                # Log and save metrics
                metrics_store.log_agent_metrics(
                    task=prompt[:50],
                    success=True,
                    tool_calls=tool_calls_count,
                    cost=cost,
                )
                
                from shared.config import METRICS_FILE
                metrics_store.save_metrics(METRICS_FILE)
                
                return {
                    "success": True,
                    "response": message.content,
                    "tool_calls": tool_calls_count,
                    "iterations": iteration + 1,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "cost": cost,
                    "latency_ms": total_timer.elapsed_ms,
                }
        
        # Max iterations reached
        log_reasoning("Max iterations reached - stopping")
        
        cost = calculate_cost(total_prompt_tokens, total_completion_tokens)
        metrics_store.log_agent_metrics(
            task=prompt[:50],
            success=False,
            tool_calls=tool_calls_count,
            cost=cost,
        )
        
        from shared.config import METRICS_FILE
        metrics_store.save_metrics(METRICS_FILE)
        
        return {
            "success": False,
            "error": "Max iterations reached",
            "tool_calls": tool_calls_count,
            "iterations": MAX_ITERATIONS,
            "cost": cost,
        }


def main() -> None:
    """CLI entry point for the planning agent."""
    if len(sys.argv) < 2:
        print("Usage: python -m agent.planner \"<trip request>\"")
        print("Example: python -m agent.planner \"Plan a 2-day trip to Auckland for under NZ$500\"")
        sys.exit(1)
    
    prompt = sys.argv[1]
    
    print("\n" + "=" * 70)
    print("🗺️  TRAVEL PLANNING AGENT")
    print("=" * 70)
    print(f"\n📝 Request: {prompt}\n")
    print("-" * 70)
    
    result = run_agent(prompt)
    
    print("-" * 70)
    
    if result["success"]:
        print(f"\n✅ Planning complete!")
        print(f"\n📋 ITINERARY:\n")
        print(result["response"])
        print("\n" + "-" * 70)
        print(f"📊 Stats:")
        print(f"   Tool calls: {result['tool_calls']}")
        print(f"   Iterations: {result['iterations']}")
        print(f"   Tokens: {result['prompt_tokens']} prompt + {result['completion_tokens']} completion")
        print(f"   Cost: ${result['cost']:.4f}")
        print(f"   Latency: {result['latency_ms']:.0f}ms")
    else:
        print(f"\n❌ Planning failed: {result.get('error', 'Unknown error')}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
