"""Self-healing code generation with repair loop for multiple languages."""

import logging
import sys
import tempfile
from pathlib import Path

from shared.llm import get_client
from shared.config import MODEL_NAME, METRICS_FILE
from shared.telemetry import Timer, calculate_cost, metrics_store

from code_assistant.runner import (
    detect_language,
    setup_project,
    write_code,
    run_tests,
    parse_errors,
    extract_code_from_response,
    get_code_filepath,
    LANGUAGES,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Maximum repair attempts
MAX_ATTEMPTS = 3


# Language-specific system prompts
SYSTEM_PROMPTS = {
    "python": """You are a Python code generator. Generate complete, working code with pytest test cases.

Requirements:
- Include the main function/class implementation
- Include at least 3 pytest test cases
- Use descriptive test names (test_*)
- Handle edge cases
- Import pytest at the top

Output ONLY the Python code, no explanations.""",

    "rust": """You are a Rust code generator. Generate complete, working code with tests.

Requirements:
- Include the main function/struct implementation
- Include at least 3 test functions in a #[cfg(test)] module
- Use descriptive test names
- Handle edge cases
- Use #[test] attribute for each test function

IMPORTANT Rust-specific rules:
- Use references (&, &mut) to avoid ownership issues
- For sorting functions, use `&mut [T]` and mutate in-place
- Use `.clone()` in tests if you need to reuse values
- Use `where T: Ord + Clone` for generic comparison functions
- For swapping elements, use `slice.swap(i, j)` method

Output ONLY the Rust code, no explanations.

Example for a sorting function:
```rust
pub fn sort<T: Ord>(arr: &mut [T]) {
    // implementation using arr.swap(i, j)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let mut arr: Vec<i32> = vec![];
        sort(&mut arr);
        assert_eq!(arr, vec![]);
    }

    #[test]
    fn test_sorted() {
        let mut arr = vec![3, 1, 2];
        sort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3]);
    }
}
```""",

    "javascript": """You are a JavaScript code generator. Generate complete, working code with Jest tests.

Requirements:
- Include the main function/class implementation
- Include at least 3 test cases using Jest (describe/test/expect)
- Use descriptive test names
- Handle edge cases
- Export functions using module.exports

Output ONLY the JavaScript code, no explanations.

Example structure:
```javascript
function functionName() {
    // implementation
}

module.exports = { functionName };

describe('functionName', () => {
    test('should do something', () => {
        expect(functionName()).toBe(expected);
    });
});
```""",
}


REPAIR_PROMPT_TEMPLATE = """The code failed with these errors:
{errors}

Original code:
```{language}
{code}
```

Fix the code to pass all tests. Output ONLY the corrected {language_name} code."""


def generate_code(task: str, language: str) -> tuple[str, dict]:
    """Generate code for a task using LLM.
    
    Args:
        task: Natural language task description.
        language: Target programming language.
        
    Returns:
        tuple: (generated_code, usage_dict)
    """
    client = get_client()
    
    system_prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["python"])
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )
    
    content = response.choices[0].message.content or ""
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    
    # Extract code from response
    code = extract_code_from_response(content, language)
    
    return code, usage


def repair_code(code: str, errors: str, language: str) -> tuple[str, dict]:
    """Attempt to repair code based on errors.
    
    Args:
        code: The failing code.
        errors: Error messages from test run.
        language: Programming language.
        
    Returns:
        tuple: (repaired_code, usage_dict)
    """
    client = get_client()
    
    language_name = LANGUAGES.get(language, LANGUAGES["python"]).name
    
    repair_prompt = REPAIR_PROMPT_TEMPLATE.format(
        errors=errors,
        code=code,
        language=language,
        language_name=language_name,
    )
    
    system_prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["python"])
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": repair_prompt},
    ]
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )
    
    content = response.choices[0].message.content or ""
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    
    # Extract code from response
    code = extract_code_from_response(content, language)
    
    return code, usage


def repair_loop(task: str, language: str | None = None) -> dict:
    """Run the code generation and repair loop.
    
    Args:
        task: Natural language task description.
        language: Programming language (auto-detected if None).
        
    Returns:
        dict: Results including success, code, attempts, etc.
    """
    # Auto-detect language if not specified
    if language is None:
        language = detect_language(task)
    
    lang_config = LANGUAGES.get(language)
    if not lang_config:
        return {
            "success": False,
            "error": f"Unsupported language: {language}",
        }
    
    print(f"[language] Detected: {lang_config.name}")
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    # Create temp directory for project
    temp_dir = Path(tempfile.mkdtemp(prefix=f"code_assistant_{language}_"))
    
    # Setup project structure
    print(f"[setup] Creating {lang_config.name} project in {temp_dir}")
    setup_project(language, temp_dir)
    
    # Get code filepath
    filepath = get_code_filepath(temp_dir, language)
    
    print(f"\n[step 1] Generating initial {lang_config.name} code...")
    
    errors = ""
    
    with Timer() as total_timer:
        # Generate initial code
        code, usage = generate_code(task, language)
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]
        
        print(f"[generated] {len(code)} characters of {lang_config.name} code")
        
        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"\n[step 2.{attempt}] Writing and testing code (attempt {attempt}/{MAX_ATTEMPTS})...")
            
            # Write code to file
            write_code(code, str(filepath))
            
            # Run tests
            print(f"[testing] Running {lang_config.test_command[0]}...")
            success, output = run_tests(str(filepath), language)
            
            if success:
                print(f"[success] All tests passed on attempt {attempt}!")
                
                cost = calculate_cost(total_prompt_tokens, total_completion_tokens)
                
                # Log metrics
                metrics_store.log_agent_metrics(
                    task=f"[CodeAssist] {task[:40]}",
                    success=True,
                    tool_calls=attempt,  # Using attempts as "tool_calls"
                    cost=cost,
                )
                metrics_store.save_metrics(METRICS_FILE)
                
                return {
                    "success": True,
                    "language": language,
                    "code": code,
                    "filepath": str(filepath),
                    "attempts": attempt,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "cost": cost,
                    "latency_ms": total_timer.elapsed_ms,
                }
            
            # Parse errors
            errors = parse_errors(output, language)
            print(f"[failed] Tests failed. Errors:\n{errors[:500]}")
            
            if attempt < MAX_ATTEMPTS:
                print(f"\n[step 3.{attempt}] Attempting repair...")
                
                # Repair code
                code, usage = repair_code(code, errors, language)
                total_prompt_tokens += usage["prompt_tokens"]
                total_completion_tokens += usage["completion_tokens"]
                
                print(f"[repaired] Generated {len(code)} characters of repaired code")
        
        # All attempts failed
        cost = calculate_cost(total_prompt_tokens, total_completion_tokens)
        
        # Log metrics
        metrics_store.log_agent_metrics(
            task=f"[CodeAssist] {task[:40]}",
            success=False,
            tool_calls=MAX_ATTEMPTS,
            cost=cost,
        )
        metrics_store.save_metrics(METRICS_FILE)
        
        return {
            "success": False,
            "language": language,
            "code": code,
            "filepath": str(filepath),
            "attempts": MAX_ATTEMPTS,
            "last_errors": errors,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "cost": cost,
            "latency_ms": total_timer.elapsed_ms,
        }


def main() -> None:
    """CLI entry point for the code assistant."""
    if len(sys.argv) < 2:
        print("Usage: python -m code_assistant.repair_loop \"<task description>\"")
        print("\nExamples:")
        print("  python -m code_assistant.repair_loop \"Write a function to check if a string is palindrome\"")
        print("  python -m code_assistant.repair_loop \"Write quicksort in Rust\"")
        print("  python -m code_assistant.repair_loop \"Write a factorial function in JavaScript\"")
        sys.exit(1)
    
    task = sys.argv[1]
    
    print("\n" + "=" * 70)
    print("🔧 CODE ASSISTANT - Self-Healing Code Generation")
    print("=" * 70)
    print(f"\n📝 Task: {task}")
    print("-" * 70)
    
    result = repair_loop(task)
    
    print("\n" + "-" * 70)
    
    if result.get("success"):
        lang = result.get("language", "unknown")
        print(f"\n✅ {LANGUAGES[lang].name} code generated successfully!")
        print(f"\n📄 Generated Code:\n")
        print("-" * 40)
        print(result["code"])
        print("-" * 40)
        print(f"\n📁 Saved to: {result['filepath']}")
    else:
        if "error" in result:
            print(f"\n❌ Error: {result['error']}")
        else:
            print(f"\n❌ Failed after {result['attempts']} attempts")
            print(f"\n📄 Last Code Attempt:\n")
            print("-" * 40)
            print(result.get("code", "No code generated"))
            print("-" * 40)
            print(f"\n❗ Last Errors:\n{result.get('last_errors', 'Unknown')[:500]}")
    
    if "cost" in result:
        print(f"\n📊 Stats:")
        print(f"   Language: {result.get('language', 'unknown')}")
        print(f"   Attempts: {result.get('attempts', 0)}")
        print(f"   Tokens: {result.get('prompt_tokens', 0)} prompt + {result.get('completion_tokens', 0)} completion")
        print(f"   Cost: ${result.get('cost', 0):.4f}")
        print(f"   Time: {result.get('latency_ms', 0):.0f}ms")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
