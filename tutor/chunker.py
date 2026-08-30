import re
from collections import Counter
from collections.abc import AsyncIterator

_BOUNDARY_CHARS = ".?!;:,"
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "Dr.")
_WORD = re.compile(r"\S+")
_TAG = re.compile(r"</?[a-z]+>|<[a-z/]{0,15}\Z")
_JSON_BOUND = 8192


def split_clauses(text: str, min_words: int, max_words: int) -> tuple[list[str], str]:
    clauses: list[str] = []
    start = 0
    count = 0
    prev_end = 0

    for match in _WORD.finditer(text):
        word_end = match.end()

        if count == max_words:
            clauses.append(text[start:prev_end].strip())
            start = prev_end + 1
            count = 0

        count += 1
        prev_end = word_end
        word = match.group()

        if word[-1] not in _BOUNDARY_CHARS:
            continue
        if word_end == len(text) or not text[word_end].isspace():
            continue
        if word[-1] == "." and word in _ABBREVIATIONS:
            continue
        if count < min_words:
            continue

        clauses.append(text[start:word_end].strip())
        start = word_end + 1
        count = 0

    return clauses, text[start:]


async def clause_chunks(
    tokens: AsyncIterator[str],
    *,
    first_min_words: int = 3,
    min_words: int = 8,
    max_words: int = 12,
) -> AsyncIterator[str]:
    buffer = ""
    released = False
    async for token in tokens:
        buffer += token
        if not released:
            first, _ = split_clauses(buffer, first_min_words, max_words)
            if not first:
                continue
            released = True
            yield first[0]
            buffer = buffer.lstrip()[len(first[0]) :]
        clauses, buffer = split_clauses(buffer, min_words, max_words)
        for clause in clauses:
            yield clause
    remainder = buffer.strip()
    if remainder:
        yield remainder


class Scrubber:
    def __init__(self) -> None:
        self.dropped: Counter[str] = Counter()
        self.dropped_chars: Counter[str] = Counter()
        self._mode, self._need, self._block, self._held = "speech", 0, 0, ""

    def feed(self, delta: str) -> str:
        text, self._held, i = self._held + delta, "", 0
        out: list[str] = []
        while i < len(text):
            i = self._step(text, i, out)
        return "".join(out)

    def flush(self) -> str:
        held, self._held = self._held, ""
        if self._mode != "speech":
            self._mark(self._mode, self._block)
        elif held.startswith("`"):
            self._mark("backtick", len(held))
            held = ""
        return held

    def _mark(self, key: str, chars: int) -> None:
        self.dropped[key] += 1
        self.dropped_chars[key] += chars

    def _step(self, text: str, i: int, out: list[str]) -> int:
        if self._mode != "speech":
            for j in range(i, len(text)):
                self._block += 1
                if self._mode == "json":
                    self._need += (text[j] == "{") - (text[j] == "}")
                else:
                    self._need = self._need - 1 if text[j] == "`" else 3
                if self._need == 0 or (self._mode == "json" and self._block == _JSON_BOUND):
                    if self._need:
                        self._mark("json_unbalanced", self._block)
                    self._mark(self._mode, self._block)
                    self._mode, self._block = "speech", 0
                    return j + 1
            return len(text)
        nxt = min((k for k in (text.find(c, i) for c in "`{*<") if k >= 0), default=len(text))
        out.append(text[i:nxt])
        if nxt == len(text):
            return nxt
        rest, match = text[nxt:], _TAG.match(text, nxt)
        if rest in ("`", "``", "*") or rest.rstrip() == "{" or (match and ">" not in match.group()):
            self._held = rest
            return len(text)
        if rest.startswith("```") or (rest[0] == "{" and rest[1:].lstrip().startswith('"')):
            self._mode, self._need = ("fence", 3) if rest[0] == "`" else ("json", 1)
            self._block = 3 if rest[0] == "`" else len(rest) - len(rest[1:].lstrip())
            out.append(" ")
            return nxt + self._block
        if match or rest[0] == "`" or rest.startswith("**"):
            width = match.end() - nxt if match else 1 if rest[0] == "`" else 2
            self._mark("tag" if match else "backtick" if rest[0] == "`" else "bold", width)
            return nxt + width
        out.append(rest[0])
        return nxt + 1


async def spoken_text(tokens: AsyncIterator[str], scrubber: Scrubber) -> AsyncIterator[str]:
    async for token in tokens:
        if text := scrubber.feed(token):
            yield text
    if tail := scrubber.flush():
        yield tail
