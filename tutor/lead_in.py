import re
from collections.abc import Iterator

from tutor.tools.models import SearchMatch, SearchResult

_DEFINITION = re.compile(r"\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")


def _symbol(text: str) -> str | None:
    found = _DEFINITION.match(text)
    return found.group(1) if found else None


def _span(match: SearchMatch) -> tuple[int, int] | None:
    lines = sorted(
        {context.line for context in match.before}
        | {match.line}
        | {context.line for context in match.after}
    )
    if len(lines) < 2 or lines[-1] - lines[0] + 1 != len(lines):
        return None
    return lines[0], lines[-1]


def _paths(results: list[SearchResult]) -> list[str]:
    ordered: dict[str, None] = {}
    for result in results:
        for match in result.matches:
            ordered[match.path] = None
    return list(ordered)


def _match_sentence(match: SearchMatch) -> str:
    span = _span(match)
    where = (
        f"{match.path} lines {span[0]} to {span[1]}" if span else f"{match.path} line {match.line}"
    )
    symbol = _symbol(match.text)
    if symbol:
        return f"The definition of {symbol} is in {where}."
    return f"The match is in {where}."


def lead_in_stages(result: SearchResult) -> Iterator[str]:
    for match in result.matches:
        yield _match_sentence(match)


def lead_in_sentence(results: list[SearchResult]) -> str:
    paths = _paths(results)
    if not paths:
        return "Nothing turned up for that."
    if len(paths) == 1:
        first = next(match for result in results for match in result.matches)
        return _match_sentence(first)
    if len(paths) == 2:
        listed = f"{paths[0]} and {paths[1]}"
    else:
        listed = ", ".join(paths[:-1]) + f", and {paths[-1]}"
    return f"It shows up in {listed}."
