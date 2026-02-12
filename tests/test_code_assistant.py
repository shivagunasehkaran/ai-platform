"""Tests for the code assistant module."""

import tempfile
from pathlib import Path

import pytest

from code_assistant.runner import (
    write_code,
    run_tests,
    parse_errors,
    extract_code_from_response,
    detect_language,
)
from code_assistant.repair_loop import MAX_ATTEMPTS


class TestWriteCode:
    """Tests for write_code function."""

    def test_write_code_creates_file(self) -> None:
        """Verify write_code creates a file with correct content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_file.py"
            code = "print('hello world')"
            
            result = write_code(code, str(filepath))
            
            assert result.exists()
            assert result.read_text() == code

    def test_write_code_creates_directories(self) -> None:
        """Verify write_code creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "subdir" / "nested" / "test_file.py"
            code = "x = 1"
            
            result = write_code(code, str(filepath))
            
            assert result.exists()
            assert result.parent.exists()

    def test_write_code_overwrites_existing(self) -> None:
        """Verify write_code overwrites existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_file.py"
            
            write_code("old content", str(filepath))
            write_code("new content", str(filepath))
            
            assert filepath.read_text() == "new content"


class TestRunTestsSuccess:
    """Tests for run_tests function with successful tests."""

    def test_run_tests_success(self) -> None:
        """Verify run_tests returns success for passing tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_passing.py"
            
            # Write a simple passing test
            code = """
def add(a, b):
    return a + b

def test_add():
    assert add(1, 2) == 3

def test_add_negative():
    assert add(-1, 1) == 0
"""
            filepath.write_text(code)
            
            success, output = run_tests(str(filepath), "python")
            
            assert success is True
            assert "passed" in output.lower()

    def test_run_tests_failure(self) -> None:
        """Verify run_tests returns failure for failing tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_failing.py"
            
            # Write a failing test
            code = """
def test_will_fail():
    assert 1 == 2
"""
            filepath.write_text(code)
            
            success, output = run_tests(str(filepath), "python")
            
            assert success is False
            assert "FAILED" in output

    def test_run_tests_file_not_found(self) -> None:
        """Verify run_tests handles missing files."""
        success, output = run_tests("/nonexistent/file.py", "python")
        
        assert success is False
        assert "not found" in output.lower()

    def test_run_tests_syntax_error(self) -> None:
        """Verify run_tests handles syntax errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_syntax.py"
            
            # Write code with syntax error
            code = """
def test_broken(:
    pass
"""
            filepath.write_text(code)
            
            success, output = run_tests(str(filepath), "python")
            
            assert success is False


class TestMaxRetries:
    """Tests for max retry limit."""

    def test_max_attempts_constant(self) -> None:
        """Verify MAX_ATTEMPTS is set to 3."""
        assert MAX_ATTEMPTS == 3

    def test_max_attempts_type(self) -> None:
        """Verify MAX_ATTEMPTS is an integer."""
        assert isinstance(MAX_ATTEMPTS, int)


class TestParseErrors:
    """Tests for error parsing."""

    def test_parse_errors_assertion(self) -> None:
        """Verify parse_errors extracts assertion errors."""
        output = """
FAILED test_example.py::test_add
E       AssertionError: assert 3 == 4
"""
        errors = parse_errors(output, "python")
        
        assert "AssertionError" in errors

    def test_parse_errors_type_error(self) -> None:
        """Verify parse_errors extracts type errors."""
        output = """
TypeError: unsupported operand type(s)
"""
        errors = parse_errors(output, "python")
        
        assert "TypeError" in errors

    def test_parse_errors_rust_compiler(self) -> None:
        """Verify parse_errors extracts Rust compiler errors."""
        output = """
error[E0382]: borrow of moved value: `x`
"""
        errors = parse_errors(output, "rust")
        
        assert "borrow of moved value" in errors


class TestDetectLanguage:
    """Tests for language detection."""

    def test_detect_python_explicit(self) -> None:
        """Verify detection of Python from prompt."""
        assert detect_language("Write a function in Python") == "python"

    def test_detect_rust_explicit(self) -> None:
        """Verify detection of Rust from prompt."""
        assert detect_language("Write quicksort in Rust") == "rust"

    def test_detect_javascript_explicit(self) -> None:
        """Verify detection of JavaScript from prompt."""
        assert detect_language("Write a function in JavaScript") == "javascript"

    def test_detect_default_python(self) -> None:
        """Verify default to Python when no language specified."""
        assert detect_language("Write a sorting function") == "python"

    def test_detect_rust_cargo(self) -> None:
        """Verify detection of Rust from cargo mention."""
        assert detect_language("Create a cargo project") == "rust"

    def test_detect_js_node(self) -> None:
        """Verify detection of JavaScript from node mention."""
        assert detect_language("Write a node script") == "javascript"


class TestExtractCode:
    """Tests for code extraction from LLM responses."""

    def test_extract_from_markdown_block(self) -> None:
        """Verify extraction from markdown code block."""
        response = """Here's the code:

```python
def hello():
    print("world")
```

That's it!"""
        
        code = extract_code_from_response(response, "python")
        
        assert "def hello():" in code
        assert "```" not in code

    def test_extract_from_rust_block(self) -> None:
        """Verify extraction from Rust code block."""
        response = """```rust
fn main() {
    println!("Hello");
}
```"""
        
        code = extract_code_from_response(response, "rust")
        
        assert "fn main()" in code

    def test_extract_no_block(self) -> None:
        """Verify extraction when no code block present."""
        response = """def add(a, b):
    return a + b"""
        
        code = extract_code_from_response(response, "python")
        
        assert "def add" in code
