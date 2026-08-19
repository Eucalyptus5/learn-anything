import tomllib
from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement

import tutor

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
DECLARED_FLOORS = {"openai": "3.8", "pydantic-settings": "2.6", "httpx2": "2.12"}


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _runtime_requirements() -> dict[str, Requirement]:
    return {
        requirement.name: requirement
        for requirement in map(Requirement, _pyproject()["project"]["dependencies"])
    }


def test_package_imports() -> None:
    assert tutor.__name__ == "tutor"


def test_reasoning_deps_are_declared_at_their_floors() -> None:
    runtime = _runtime_requirements()

    for name, floor in DECLARED_FLOORS.items():
        assert str(runtime[name].specifier) == f">={floor}"


def test_pydantic_is_declared_once() -> None:
    names = [Requirement(entry).name for entry in _pyproject()["project"]["dependencies"]]

    assert names.count("pydantic") == 1


def test_openai_and_httpx2_are_in_no_dependency_group() -> None:
    for group, entries in _pyproject()["dependency-groups"].items():
        names = {Requirement(entry).name for entry in entries}

        assert "openai" not in names, group
        assert "httpx2" not in names, group


def test_installed_versions_satisfy_the_declared_floors() -> None:
    for name, requirement in _runtime_requirements().items():
        assert requirement.specifier.contains(version(name)), name
