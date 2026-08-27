from collections.abc import Sequence
from pathlib import Path

from tutor.tools.models import ContextLine, SearchBudget, SearchMatch, SearchResult
from tutor.tools.ripgrep import decode_field, record_size, run_ripgrep


async def search(
    query: str, globs: Sequence[str], root: Path, budget: SearchBudget
) -> SearchResult:
    records, _, truncated, oversized = await run_ripgrep(query, globs, root, budget)

    hits: list[tuple[str, int, str, dict]] = []
    context: dict[tuple[str, int], tuple[str, dict]] = {}
    for record in records:
        data = record["data"]
        path = decode_field(data["path"])
        text = decode_field(data["lines"])
        if path is None or text is None:
            continue
        path = path.removeprefix("./")
        text = text.removesuffix("\n")
        if record["type"] == "match":
            hits.append((path, data["line_number"], text, record))
        else:
            context[(path, data["line_number"])] = (text, record)
    hits.sort(key=lambda hit: (hit[0], hit[1]))
    if len(hits) > budget.max_matches:
        truncated = True
        hits = hits[: budget.max_matches]

    taken: set[tuple[str, int]] = set()
    matches: list[SearchMatch] = []
    byte_count = 0
    for path, line, text, record in hits:
        byte_count += record_size(record)

        before: list[ContextLine] = []
        for number in range(line - budget.context_lines, line):
            key = (path, number)
            if key in context and key not in taken:
                taken.add(key)
                context_text, context_record = context[key]
                byte_count += record_size(context_record)
                before.append(ContextLine(line=number, text=context_text))

        after: list[ContextLine] = []
        for number in range(line + 1, line + budget.context_lines + 1):
            key = (path, number)
            if key in context and key not in taken:
                taken.add(key)
                context_text, context_record = context[key]
                byte_count += record_size(context_record)
                after.append(ContextLine(line=number, text=context_text))

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
