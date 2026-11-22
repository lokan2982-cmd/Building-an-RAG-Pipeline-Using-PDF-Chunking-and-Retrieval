"""
embeddings.py
=============
Initialises local Ollama embeddings using LangChain's OllamaEmbeddings.

Runs completely offline once the embedding model is pulled via Ollama:
    ollama pull nomic-embed-text

Configuration (via .env or environment variables):
  - OLLAMA_BASE_URL        : Ollama server URL (default: http://localhost:11434)
  - OLLAMA_EMBEDDING_MODEL : Embedding model name (default: nomic-embed-text)

No OpenAI or Azure OpenAI API keys are required.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

# Load .env into environment (safe to call multiple times)
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


def get_ollama_base_url(base_url: Optional[str] = None) -> str:
    """Return the configured Ollama server base URL (without trailing slash)."""
    url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    return url.strip().rstrip("/")


def get_embedding_model_name(model: Optional[str] = None) -> str:
    """Return the configured Ollama embedding model name."""
    name = model or os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return name.strip()


def get_embedding_identifier(model: Optional[str] = None) -> str:
    """
    Return a stable identifier for the active embedding model.
    Stored alongside the FAISS index so the index is rebuilt automatically
    if the embedding model changes.
    """
    return f"ollama:{get_embedding_model_name(model)}"


def list_installed_ollama_models(
    base_url: Optional[str] = None,
    timeout: float = 5.0,
) -> List[str]:
    """
    Query the local Ollama server for installed model names.

    Args:
        base_url (str, optional): Ollama server URL.
        timeout  (float): HTTP timeout in seconds.

    Returns:
        List[str]: Installed model names (e.g., ['nomic-embed-text:latest', 'llama3.2:latest']).

    Raises:
        ConnectionError: If the Ollama server cannot be reached.
    """
    url = f"{get_ollama_base_url(base_url)}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in payload.get("models", []) if m.get("name")]
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ConnectionError(
            f"Could not connect to local Ollama server at {get_ollama_base_url(base_url)}.\n"
            "Make sure Ollama is installed and running (try: `ollama serve`)."
        ) from exc


def is_model_installed(model_name: str, installed_models: List[str]) -> bool:
    """Return True if `model_name` matches an installed Ollama model (with or without tag)."""
    target = model_name.strip().lower()
    for installed in installed_models:
        inst_lower = installed.strip().lower()
        if inst_lower == target or inst_lower.split(":")[0] == target.split(":")[0]:
            return True
    return False


def get_embeddings(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OllamaEmbeddings:
    """
    Create and return a configured OllamaEmbeddings instance.

    Args:
        model    (str, optional): Ollama embedding model (default: nomic-embed-text).
        base_url (str, optional): Ollama server URL (default: http://localhost:11434).

    Returns:
        OllamaEmbeddings: Ready-to-use LangChain Ollama embeddings client.

    Raises:
        ValueError: If model name or base URL is empty.
    """
    resolved_model = get_embedding_model_name(model)
    resolved_url = get_ollama_base_url(base_url)

    if not resolved_model:
        raise ValueError(
            "OLLAMA_EMBEDDING_MODEL cannot be empty. "
            "Set it to 'nomic-embed-text' in your .env file."
        )
    if not resolved_url.startswith(("http://", "https://")):
        raise ValueError(
            f"Invalid OLLAMA_BASE_URL='{resolved_url}'. "
            "Expected a URL such as 'http://localhost:11434'."
        )

    logger.info(
        f"Initialising OllamaEmbeddings: model='{resolved_model}', base_url='{resolved_url}'"
    )
    return OllamaEmbeddings(model=resolved_model, base_url=resolved_url)


def validate_embedding_config(check_server: bool = False) -> Dict[str, Any]:
    """
    Return the active local Ollama embedding configuration.

    Args:
        check_server (bool): If True, also check if the Ollama server is reachable
                             and whether the configured embedding model is pulled.

    Returns:
        dict: Configuration details.
    """
    model = get_embedding_model_name()
    base_url = get_ollama_base_url()
    config: Dict[str, Any] = {
        "provider": "ollama",
        "model": model,
        "base_url": base_url,
        "api_key_required": False,
    }
    if check_server:
        try:
            installed = list_installed_ollama_models(base_url=base_url)
            config["ollama_running"] = True
            config["model_installed"] = is_model_installed(model, installed)
        except ConnectionError:
            config["ollama_running"] = False
            config["model_installed"] = False
    return config


if __name__ == "__main__":
    config = validate_embedding_config(check_server=True)
    print("Ollama Embedding Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
