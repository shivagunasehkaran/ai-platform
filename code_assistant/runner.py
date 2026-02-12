"""Code execution and testing utilities for multiple languages."""

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

Language = Literal["python", "rust", "javascript"]


@dataclass
class LanguageConfig:
    """Configuration for a programming language."""
    name: str
    extension: str
    test_command: list[str]
    setup_commands: list[list[str]] | None = None
    project_structure: dict[str, str] | None = None


# Language configurations
LANGUAGES: dict[str, LanguageConfig] = {
    "python": LanguageConfig(
        name="Python",
        extension=".py",
        test_command=[sys.executable, "-m", "pytest", "-v", "--tb=short"],
    ),
    "rust": LanguageConfig(
        name="Rust",
        extension=".rs",
        test_command=["cargo", "test"],
        setup_commands=[["cargo", "init", "--name", "generated_code"]],
        project_structure={
            "Cargo.toml": """[package]
name = "generated_code"
version = "0.1.0"
edition = "2021"

[dependencies]
""",
        },
    ),
    "javascript": LanguageConfig(
        name="JavaScript",
        extension=".js",
        test_command=["npm", "test"],
        setup_commands=[["npm", "init", "-y"], ["npm", "install", "--save-dev", "jest"]],
        project_structure={
            "package.json": """{
  "name": "generated_code",
  "version": "1.0.0",
  "scripts": {
    "test": "jest"
  },
  "devDependencies": {
    "jest": "^29.0.0"
  }
}
""",
        },
    ),
}


def detect_language(prompt: str) -> str:
    """Detect programming language from prompt.
    
    Args:
        prompt: User's task description.
        
    Returns:
        str: Detected language (python, rust, javascript).
    """
    prompt_lower = prompt.lower()
    
    # Check for explicit language mentions
    if any(word in prompt_lower for word in ["rust", "cargo", ".rs"]):
        return "rust"
    elif any(word in prompt_lower for word in ["javascript", "js", "node", "jest"]):
        return "javascript"
    elif any(word in prompt_lower for word in ["python", "pytest", ".py"]):
        return "python"
    
    # Default to Python
    return "python"


def setup_project(language: str, project_dir: Path) -> bool:
    """Setup project structure for a language.
    
    Args:
        language: Programming language.
        project_dir: Project directory path.
        
    Returns:
        bool: True if setup successful.
    """
    config = LANGUAGES.get(language)
    if not config:
        return False
    
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Write project structure files
    if config.project_structure:
        for filename, content in config.project_structure.items():
            filepath = project_dir / filename
            filepath.write_text(content)
            logger.debug(f"Created {filepath}")
    
    # Run setup commands
    if config.setup_commands:
        for cmd in config.setup_commands:
            try:
                # Skip cargo init if Cargo.toml exists
                if cmd[0] == "cargo" and cmd[1] == "init" and (project_dir / "Cargo.toml").exists():
                    continue
                    
                subprocess.run(
                    cmd,
                    cwd=project_dir,
                    capture_output=True,
                    timeout=60,
                )
            except Exception as e:
                logger.warning(f"Setup command failed: {cmd}, error: {e}")
    
    return True


def write_code(code: str, filepath: str) -> Path:
    """Write code to a file.
    
    Args:
        code: Source code to write.
        filepath: Path to write the file.
        
    Returns:
        Path: The path to the written file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    path.write_text(code, encoding="utf-8")
    logger.info(f"Wrote code to {path}")
    
    return path


def run_tests(filepath: str, language: str) -> tuple[bool, str]:
    """Run tests for a file based on language.
    
    Args:
        filepath: Path to the source file.
        language: Programming language.
        
    Returns:
        tuple: (success: bool, output: str)
    """
    path = Path(filepath)
    config = LANGUAGES.get(language)
    
    if not config:
        return False, f"Unsupported language: {language}"
    
    if not path.exists():
        return False, f"File not found: {filepath}"
    
    # Determine working directory and test command
    if language == "python":
        cwd = path.parent
        cmd = config.test_command + [str(path)]
    elif language == "rust":
        cwd = path.parent.parent  # Rust: src/main.rs -> project root
        cmd = config.test_command
    elif language == "javascript":
        cwd = path.parent
        cmd = config.test_command
    else:
        cwd = path.parent
        cmd = config.test_command
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        output = result.stdout + result.stderr
        success = result.returncode == 0
        
        logger.info(f"Tests {'passed' if success else 'failed'} for {path}")
        
        return success, output
        
    except subprocess.TimeoutExpired:
        return False, "Test execution timed out (60s limit)"
    except FileNotFoundError as e:
        return False, f"Test runner not found: {e}. Make sure {config.test_command[0]} is installed."
    except Exception as e:
        return False, f"Error running tests: {str(e)}"


def parse_errors(output: str, language: str) -> str:
    """Extract error messages from test output.
    
    Args:
        output: Raw test output.
        language: Programming language.
        
    Returns:
        str: Cleaned error messages for LLM feedback.
    """
    errors = []
    
    if language == "python":
        # Extract pytest failures
        failed_tests = re.findall(r"FAILED\s+[\w:]+::(test_\w+)", output)
        if failed_tests:
            errors.append(f"Failed tests: {', '.join(failed_tests)}")
        
        # Extract exceptions
        exceptions = re.findall(
            r"((?:AssertionError|TypeError|ValueError|NameError|SyntaxError|"
            r"IndentationError|AttributeError|KeyError|IndexError):\s*.*?)(?:\n|$)",
            output
        )
        errors.extend(exc.strip() for exc in exceptions)
        
        # Extract E lines
        e_lines = re.findall(r"^E\s+(.+)$", output, re.MULTILINE)
        errors.extend(line.strip() for line in e_lines[:5])
        
    elif language == "rust":
        # Extract Rust compiler errors
        rust_errors = re.findall(r"error(?:\[E\d+\])?:\s*(.+?)(?:\n|$)", output)
        errors.extend(err.strip() for err in rust_errors[:10])
        
        # Extract failed tests
        failed_tests = re.findall(r"test\s+(\w+)\s+\.\.\.\s+FAILED", output)
        if failed_tests:
            errors.append(f"Failed tests: {', '.join(failed_tests)}")
        
        # Extract assertion failures
        assertions = re.findall(r"assertion .+ failed.*", output)
        errors.extend(assertions[:5])
        
    elif language == "javascript":
        # Extract Jest failures
        jest_errors = re.findall(r"(expect\(.+\)\..+)", output)
        errors.extend(jest_errors[:5])
        
        # Extract other errors
        js_errors = re.findall(r"((?:TypeError|ReferenceError|SyntaxError):\s*.*?)(?:\n|$)", output)
        errors.extend(err.strip() for err in js_errors)
    
    # Fallback: return truncated output
    if not errors:
        return output[-1000:] if len(output) > 1000 else output
    
    return "\n".join(errors)


def extract_code_from_response(response: str, language: str) -> str:
    """Extract code from LLM response.
    
    Args:
        response: LLM response that may contain markdown code blocks.
        language: Expected programming language.
        
    Returns:
        str: Extracted source code.
    """
    # Language aliases for markdown
    lang_aliases = {
        "python": ["python", "py"],
        "rust": ["rust", "rs"],
        "javascript": ["javascript", "js"],
    }
    
    aliases = lang_aliases.get(language, [language])
    
    # Try to extract from markdown code block with language tag
    for alias in aliases:
        pattern = rf"```{alias}\s*\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Try generic code block
    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # No code block - return cleaned response
    return response.strip()


def get_code_filepath(project_dir: Path, language: str) -> Path:
    """Get the appropriate file path for generated code.
    
    Args:
        project_dir: Project directory.
        language: Programming language.
        
    Returns:
        Path: File path for the code.
    """
    config = LANGUAGES.get(language)
    ext = config.extension if config else ".txt"
    
    if language == "rust":
        # Rust requires src/main.rs or src/lib.rs
        src_dir = project_dir / "src"
        src_dir.mkdir(exist_ok=True)
        return src_dir / f"lib{ext}"
    elif language == "javascript":
        return project_dir / f"code.test{ext}"
    else:
        return project_dir / f"generated_code{ext}"
