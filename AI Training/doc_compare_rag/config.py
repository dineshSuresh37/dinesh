"""
config.py – Central configuration loaded from environment variables / .env file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(override=False)
except ImportError:
    pass


@dataclass
class Config:
    llm_provider: str = field(
        default_factory=lambda: os.environ.get("LLM_PROVIDER", "openai")
    )
    openai_api_key: str | None = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )
    openai_model: str = field(
        default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-4o")
    )
    anthropic_model: str = field(
        default_factory=lambda: os.environ.get(
            "ANTHROPIC_MODEL", "claude-sonnet-4-6"
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    embedding_cache_dir: str = field(
        default_factory=lambda: os.environ.get(
            "EMBEDDING_CACHE_DIR", ".cache/embeddings"
        )
    )
    vector_store: str = field(
        default_factory=lambda: os.environ.get("VECTOR_STORE", "faiss")
    )
    data_dir: str = field(
        default_factory=lambda: os.environ.get("DATA_DIR", "data")
    )
    chroma_persist_dir: str = field(
        default_factory=lambda: os.environ.get("CHROMA_PERSIST_DIR", "chroma_db")
    )
    chunk_size: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_SIZE", "500"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP", "50"))
    )

    def validate(self) -> None:
        if self.llm_provider not in {"openai", "anthropic"}:
            raise EnvironmentError(
                f"Unknown LLM_PROVIDER: {self.llm_provider!r}. "
                "Must be 'openai' or 'anthropic'."
            )
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai."
            )
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic."
            )


_default_config: Config | None = None


def get_config() -> Config:
    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config
