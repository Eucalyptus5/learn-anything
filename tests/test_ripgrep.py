from pathlib import Path

import pytest

from tutor.tools.ripgrep import RipgrepUnavailable, probe_ripgrep

FAKE_RG_DIR = str(Path(__file__).parent / "data" / "fake_rg")


def test_probe_ripgrep_returns_version_from_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", FAKE_RG_DIR)

    assert probe_ripgrep() == "15.2.0"


def test_probe_ripgrep_rejects_version_below_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", FAKE_RG_DIR)
    monkeypatch.setenv("FAKE_RG_VERSION", "12.0.0")

    with pytest.raises(RipgrepUnavailable, match="12.0.0"):
        probe_ripgrep()


def test_probe_ripgrep_raises_on_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")

    with pytest.raises(RipgrepUnavailable, match="rg"):
        probe_ripgrep()
