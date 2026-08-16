import base64
from collections.abc import Sequence
from pathlib import Path

from tutor.tools.models import ContextLine, SearchBudget, SearchMatch, SearchResult
from tutor.tools.ripgrep import run_ripgrep


def _decode(field: dict) -> str | None:
    if "text" in field:
        return field["text"]
    try:
        return base64.b64decode(field["bytes"]).decode()
    except UnicodeDecodeError:
        return None


async def search(
    query: str, globs: Sequence[str], root: Path, budget: SearchBudget
) -> SearchResult:
    records, byte_count, truncated, oversized = await run_ripgrep(query, globs, root, budget)

    hits: list[tuple[str, int, str]] = []
    context: dict[tuple[str, int], str] = {}
    for record in records:
        data = record["data"]
        path = _decode(data["path"])
        text = _decode(data["lines"])
        if path is None or text is None:
            continue
        path = path.removeprefix("./")
        text = text.removesuffix("\n")
        if record["type"] == "match":
            hits.append((path, data["line_number"], text))
        else:
            context[(path, data["line_number"])] = text
    hits.sort(key=lambda hit: (hit[0], hit[1]))

    taken: set[tuple[str, int]] = set()
    matches: list[SearchMatch] = []
    for path, line, text in hits:
        before: list[ContextLine] = []
        for number in range(line - budget.context_lines, line):
            key = (path, number)
            if key in context and key not in taken:
                taken.add(key)
                before.append(ContextLine(line=number, text=context[key]))

        after: list[ContextLine] = []
        for number in range(line + 1, line + budget.context_lines + 1):
            key = (path, number)
            if key in context and key not in taken:
                taken.add(key)
                after.append(ContextLine(line=number, text=context[key]))

        matches.append(SearchMatch(path=path, line=line, text=text, before=before, after=after))

    return SearchResult(
        tool="search",
        query=query,
        globs=list(globs),
        matches=matches,
        truncated=truncated,
        oversized=oversized,
        byte_count=byte_count,
    )
