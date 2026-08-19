import os
from collections.abc import Iterator

import pytest

from tutor.config import settings

PREFIX = "REASONING_"


def _drop_reasoning_vars() -> None:
    for name in [key for key in os.environ if key.startswith(PREFIX)]:
        del os.environ[name]


@pytest.fixture(autouse=True)
def reasoning_env_isolation() -> Iterator[None]:
    saved = {key: value for key, value in os.environ.items() if key.startswith(PREFIX)}
    _drop_reasoning_vars()
    settings.cache_clear()
    try:
        yield
    finally:
        _drop_reasoning_vars()
        os.environ.update(saved)
        settings.cache_clear()
