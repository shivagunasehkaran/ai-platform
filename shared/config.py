"""Configuration module for AI platform settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI Configuration
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Gpt4o")

# GPT-4o Pricing (USD per 1K tokens)
PRICE_INPUT_PER_1K: float = 0.005
PRICE_OUTPUT_PER_1K: float = 0.015

# Chunking Configuration for RAG
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50

# Embedding Model
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

# Project Paths
PROJECT_ROOT: Path = Path(__file__).parent.parent
CORPUS_DIR: Path = PROJECT_ROOT / "corpus"
DATA_DIR: Path = PROJECT_ROOT / "data"
METRICS_FILE: Path = PROJECT_ROOT / "metrics.json"


def validate_config() -> bool:
    """Validate that required configuration is present.
    
    Returns:
        bool: True if configuration is valid, False otherwise.
    """
    if not OPENAI_BASE_URL:
        return False
    if not OPENAI_API_KEY:
        return False
    return True
