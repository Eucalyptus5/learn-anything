import re

_BOUNDARY_CHARS = ".?!;:,"
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "Dr.")
_WORD = re.compile(r"\S+")


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
