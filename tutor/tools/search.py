import base64
from collections.abc import Sequence
from pathlib import Path

from tutor.tools.models import SearchBudget, SearchMatch, SearchResult
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

    matches: list[SearchMatch] = []
    for record in records:
        if record["type"] != "match":
            continue
        data = record["data"]
        path = _decode(data["path"])
        text = _decode(data["lines"])
        if path is None or text is None:
            continue
        matches.append(
            SearchMatch(
                path=path.removeprefix("./"),
                line=data["line_number"],
                text=text.removesuffix("\n"),
            )
        )
    matches.sort(key=lambda match: (match.path, match.line))

    return SearchResult(
        tool="search",
        query=query,
        globs=list(globs),
        matches=matches,
        truncated=truncated,
        oversized=oversized,
        byte_count=byte_count,
    )
