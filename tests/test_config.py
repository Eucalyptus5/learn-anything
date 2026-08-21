import logging
import traceback
from pathlib import Path

import httpx2
import openai
import pytest
from pydantic import SecretStr, ValidationError

from tutor.config import Settings

REPO_ROOT = Path(__file__).resolve().parent.parent
FAKE_BASE = "https://reasoning.invalid/v1"
FAKE_KEY = "sk-test-not-a-real-key"


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


def _unauthorized(request: httpx2.Request) -> httpx2.Response:
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    return httpx2.Response(
        401, json={"error": {"message": "invalid key", "type": "invalid_request"}}
    )


async def _rejected_call(tmp_path: Path) -> openai.AuthenticationError:
    path = _write(
        tmp_path, "canary.env", f"REASONING_API_BASE={FAKE_BASE}\nREASONING_API_KEY={FAKE_KEY}\n"
    )
    cfg = Settings(_env_file=path)
    client = openai.AsyncOpenAI(
        api_key=cfg.reasoning_api_key.get_secret_value(),
        base_url=cfg.reasoning_api_base,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(_unauthorized)),
        max_retries=0,
    )
    try:
        with pytest.raises(openai.AuthenticationError) as caught:
            await client.chat.completions.create(
                model=cfg.reasoning_model,
                messages=[{"role": "user", "content": "explain the pool"}],
                max_tokens=cfg.reasoning_max_tokens,
            )
    finally:
        await client.close()
    return caught.value


def test_reads_env_file(tmp_path: Path) -> None:
    path = _write(
        tmp_path, "full.env", f"REASONING_API_BASE={FAKE_BASE}\nREASONING_API_KEY={FAKE_KEY}\n"
    )

    cfg = Settings(_env_file=path)

    assert cfg.reasoning_api_base == FAKE_BASE
    assert isinstance(cfg.reasoning_api_key, SecretStr)
    assert cfg.reasoning_api_key.get_secret_value() == FAKE_KEY


def test_blank_or_missing_key_raises(tmp_path: Path) -> None:
    blank = _write(tmp_path, "blank.env", f"REASONING_API_BASE={FAKE_BASE}\nREASONING_API_KEY=\n")
    absent = _write(tmp_path, "absent.env", f"REASONING_API_BASE={FAKE_BASE}\n")

    with pytest.raises(ValidationError):
        Settings(_env_file=blank)

    with pytest.raises(ValidationError):
        Settings(_env_file=absent)


def test_shipped_template_does_not_validate() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=REPO_ROOT / ".env.example")


def test_filled_template_yields_defaults(tmp_path: Path) -> None:
    template = (REPO_ROOT / ".env.example").read_text()
    filled = template.replace("REASONING_API_BASE=\n", f"REASONING_API_BASE={FAKE_BASE}\n").replace(
        "REASONING_API_KEY=\n", f"REASONING_API_KEY={FAKE_KEY}\n"
    )
    path = _write(tmp_path, "filled.env", filled)

    cfg = Settings(_env_file=path)

    assert cfg.reasoning_api_base == FAKE_BASE
    assert cfg.reasoning_model == "glm-5.3-flash"
    assert cfg.reasoning_effort == "low"
    assert cfg.reasoning_max_tokens == 400


async def test_key_absent_from_the_error_text(tmp_path: Path) -> None:
    exc = await _rejected_call(tmp_path)

    assert FAKE_KEY not in str(exc)
    assert FAKE_KEY not in repr(exc)


async def test_key_absent_from_the_traceback(tmp_path: Path) -> None:
    exc = await _rejected_call(tmp_path)

    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    assert FAKE_KEY not in formatted


async def test_key_absent_from_debug_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    await _rejected_call(tmp_path)

    assert FAKE_KEY not in caplog.text
