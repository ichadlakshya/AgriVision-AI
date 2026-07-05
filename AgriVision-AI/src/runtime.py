"""Shared runtime helpers for configuration, environment loading, and logging."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in local/dev environments
    load_dotenv = None


@lru_cache(maxsize=1)
def load_environment() -> None:
    """Load environment variables from a local .env file once per process."""

    if load_dotenv is None:
        return

    load_dotenv()


def project_root() -> Path:
    """Return the repository root path."""

    return Path(__file__).resolve().parents[1]


def env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable after loading .env values."""

    load_environment()
    return os.getenv(name, default)


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""

    raw_value = env(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: Iterable[str] | None = None) -> list[str]:
    """Read a comma separated environment variable into a list."""

    raw_value = env(name)
    if not raw_value:
        return list(default or [])
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def configure_logging(name: str, level: str | None = None) -> logging.Logger:
    """Return a configured application logger."""

    load_environment()
    logger = logging.getLogger(name)
    resolved_level = (level or env("LOG_LEVEL", "INFO") or "INFO").upper()

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=getattr(logging, resolved_level, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        logger.setLevel(getattr(logging, resolved_level, logging.INFO))

    return logger