"""Cache backend shim with an in-memory fallback when diskcache is unavailable."""

from __future__ import annotations

from typing import Any

from ai_bias_search.utils.logging import configure_logging

LOGGER = configure_logging()

try:  # pragma: no cover - exercised when diskcache is installed
    from diskcache import Cache as Cache  # type: ignore[no-redef]
except ModuleNotFoundError:  # pragma: no cover - covered in environments without diskcache
    _GLOBAL_STORES: dict[str, dict[Any, Any]] = {}
    _WARNED: set[str] = set()

    class Cache:  # type: ignore[no-redef]
        """Minimal `diskcache.Cache`-compatible fallback for tests/local runs."""

        def __init__(self, directory: Any | None = None) -> None:
            self.directory = directory
            key = str(directory or "__default__")
            self._store = _GLOBAL_STORES.setdefault(key, {})
            if key not in _WARNED:
                _WARNED.add(key)
                LOGGER.warning(
                    "diskcache not installed; falling back to in-memory cache for %s",
                    directory,
                )

        def __enter__(self) -> "Cache":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            self.close()

        def get(self, key: Any, default: Any = None) -> Any:
            return self._store.get(key, default)

        def set(self, key: Any, value: Any, expire: int | None = None) -> bool:
            self._store[key] = value
            return True

        def close(self) -> None:
            return None
