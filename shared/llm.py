"""LLM client module for Azure OpenAI and local embeddings."""

import logging
from typing import Generator

from openai import OpenAI
from fastembed import TextEmbedding

from shared.config import (
    OPENAI_BASE_URL,
    OPENAI_API_KEY,
    MODEL_NAME,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

# Singleton instances
_client: OpenAI | None = None
_embedding_model: TextEmbedding | None = None


def get_client() -> OpenAI:
    """Get OpenAI client singleton.
    
    Returns:
        OpenAI: Configured OpenAI client for Azure endpoint.
    """
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
        )
        logger.debug("Initialized OpenAI client")
    return _client


def get_embedding_model() -> TextEmbedding:
    """Get fastembed model singleton.
    
    Returns:
        TextEmbedding: Configured fastembed model.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
        logger.debug(f"Initialized embedding model: {EMBEDDING_MODEL}")
    return _embedding_model


def stream_chat(messages: list[dict]) -> Generator[str, None, None]:
    """Stream chat completion tokens.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
        
    Yields:
        str: Individual tokens from the response.
    """
    client = get_client()
    
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        logger.error(f"Stream chat error: {e}")
        raise


def chat_with_usage(messages: list[dict]) -> tuple[str, dict]:
    """Get chat completion with token usage statistics.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
        
    Returns:
        tuple: (response_text, usage_dict) where usage_dict contains
               'prompt_tokens', 'completion_tokens', and 'total_tokens'.
    """
    client = get_client()
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )
        
        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }
        
        logger.debug(f"Chat completed: {usage}")
        return content, usage
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise


def embed(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using fastembed.
    
    Args:
        texts: List of text strings to embed.
        
    Returns:
        list[list[float]]: List of embedding vectors.
    """
    model = get_embedding_model()
    
    try:
        embeddings = list(model.embed(texts))
        # Convert numpy arrays to lists
        embeddings = [emb.tolist() for emb in embeddings]
        logger.debug(f"Generated {len(embeddings)} embeddings")
        return embeddings
        
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise
