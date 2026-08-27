import logging
import posixpath
import re
from dataclasses import dataclass

from tutor.tools.models import PATH_EXTENSIONS, GroundingVerdict, Position, SearchResult

logger = logging.getLogger(__name__)

_LEADING = "`\"'([{"
_TRAILING = "`\"')]}.,;:!?"
_LINE_SUFFIX = re.compile(r":(\d+)(?:-(\d+))?$")
_DIGITS = re.compile(r"^\d+$")
_DIGIT_ORDINAL = re.compile(r"^(\d+)(?:st|nd|rd|th)$", re.IGNORECASE)
# Model text reaches this scanner. A number this wide cannot be a recorded line, so an over-long
# run is clamped to a value nothing can match rather than parsed whole: int() on an unbounded
# digit run raises.
_VALUE_DIGITS = 20
_NUMBER_WORD_LIMIT = 16
_SEAM_LIMIT = 32
_SHORT_EXTENSION = re.compile(r"\.[A-Za-z0-9]{1,6}$")
_RANGE_WORDS = frozenset({"to", "through", "and"})
_LINE_WORDS = frozenset({"line", "lines"})

_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_TEENS = (
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_ORDINAL_ONES = (
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
)
_ORDINAL_TEENS = (
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
)
_ORDINAL_TENS = (
    "twentieth",
    "thirtieth",
    "fortieth",
    "fiftieth",
    "sixtieth",
    "seventieth",
    "eightieth",
    "ninetieth",
)

_NUMBER_WORDS: dict[str, tuple[int, str, bool]] = (
    {word: (value, "ones", False) for value, word in enumerate(_ONES)}
    | {word: (value, "teens", False) for value, word in enumerate(_TEENS, start=10)}
    | {word: (20 + step * 10, "tens", False) for step, word in enumerate(_TENS)}
    | {word: (value, "ones", True) for value, word in enumerate(_ORDINAL_ONES, start=1)}
    | {word: (value, "teens", True) for value, word in enumerate(_ORDINAL_TEENS, start=10)}
    | {word: (20 + step * 10, "tens", True) for step, word in enumerate(_ORDINAL_TENS)}
    | {"hundred": (100, "hundred", False), "thousand": (1000, "thousand", False)}
)


@dataclass(frozen=True)
class _Unit:
    kind: str
    index: int
    end: int
    path: str = ""
    value: int = 0
    ordinal: bool = False
    owner: int | None = None
    mirrors: int | None = None


def _strip(token: str) -> str:
    return token.lstrip(_LEADING).rstrip(_TRAILING)


def _is_path(token: str) -> bool:
    if not token or "://" in token:
        return False
    segment = token.rsplit("/", 1)[-1]
    dot = segment.rfind(".")
    if dot != -1 and segment[dot:] in PATH_EXTENSIONS:
        return True
    return "/" in token and _SHORT_EXTENSION.search(segment) is not None


def _value(digits: str) -> int:
    return int(digits[:_VALUE_DIGITS])


def _digit_value(piece: str) -> tuple[int, bool] | None:
    if _DIGITS.match(piece):
        return _value(piece), False
    ordinal = _DIGIT_ORDINAL.match(piece)
    if ordinal:
        return _value(ordinal.group(1)), True
    return None


def _compose(run: list[tuple[int, str, bool]]) -> tuple[int, bool]:
    segments: list[int] = []
    multiplied = False
    index = 0
    while index < len(run):
        value, tier, _ = run[index]
        index += 1
        if tier in ("hundred", "thousand"):
            multiplied = True
            if not segments:
                segments = [value]
            elif tier == "hundred":
                segments[-1] *= value
            else:
                segments = [sum(segments) * value]
            continue
        if tier == "tens" and index < len(run):
            next_value, next_tier, _ = run[index]
            if next_tier == "ones":
                value += next_value
                index += 1
        segments.append(value)

    ordinal = any(is_ordinal for _, _, is_ordinal in run)
    if multiplied:
        return _value(str(sum(segments))), ordinal
    return _value("".join(str(segment) for segment in segments)), ordinal


def _scan(text: str, offset: int = 0) -> tuple[list[_Unit], list[str]]:
    stripped = [_strip(token) for token in text.split()]
    units: list[_Unit] = []
    run: list[tuple[int, str, bool]] = []
    run_at: list[int] = []
    run_start = 0
    run_end = 0

    def flush() -> None:
        nonlocal run, run_at
        if not run:
            return
        value, ordinal = _compose(run)
        merged = len(units)
        units.append(
            _Unit(kind="number", index=run_start, end=run_end, value=value, ordinal=ordinal)
        )
        # A run the cut lands inside was heard whole only if the donating chunk was spoken, so the
        # tail on its own is a second reading of the same words and both have to be grounded.
        if run_start < offset <= run_end:
            tail = [word for word, at in zip(run, run_at) if at >= offset]
            value, ordinal = _compose(tail)
            units.append(
                _Unit(
                    kind="number",
                    index=next(at for at in run_at if at >= offset),
                    end=run_end,
                    value=value,
                    ordinal=ordinal,
                    mirrors=merged,
                )
            )
        run = []
        run_at = []

    def number(index: int, value: int, ordinal: bool = False, owner: int | None = None) -> None:
        units.append(
            _Unit(kind="number", index=index, end=index, value=value, ordinal=ordinal, owner=owner)
        )

    for index, token in enumerate(stripped):
        suffix = _LINE_SUFFIX.search(token)
        head = token[: suffix.start()] if suffix else token
        if suffix:
            flush()
            start, end = suffix.groups()
            if _is_path(head):
                owner = len(units)
                units.append(_Unit(kind="path", index=index, end=index, path=head))
                number(index, _value(start), owner=owner)
                if end:
                    units.append(_Unit(kind="range", index=index, end=index))
                    number(index, _value(end), owner=owner)
            elif _DIGITS.match(head):
                number(index, _value(head))
                units.append(_Unit(kind="range", index=index, end=index))
                number(index, _value(start))
                if end:
                    units.append(_Unit(kind="range", index=index, end=index))
                    number(index, _value(end))
            continue

        if _is_path(head):
            flush()
            units.append(_Unit(kind="path", index=index, end=index, path=head))
            continue

        pieces = token.split("-")
        numerals = [_digit_value(piece) for piece in pieces]
        if all(numeral is not None for numeral in numerals):
            flush()
            for step, numeral in enumerate(numerals):
                if step:
                    units.append(_Unit(kind="range", index=index, end=index))
                value, ordinal = numeral
                number(index, value, ordinal=ordinal)
            continue

        words = [_NUMBER_WORDS.get(piece.lower()) for piece in pieces]
        if all(word is not None for word in words):
            for word in words:
                if len(run) == _NUMBER_WORD_LIMIT:
                    flush()
                if not run:
                    run_start = index
                run_end = index
                run.append(word)
                run_at.append(index)
            continue

        flush()
        if token.lower() in _RANGE_WORDS:
            units.append(_Unit(kind="range", index=index, end=index))

    flush()
    return units, stripped


def _ordinal_mentions(units: list[_Unit], stripped: list[str]) -> list[int]:
    evidence = [-1] * len(units)
    for position in reversed(range(len(units))):
        unit = units[position]
        if unit.kind != "number" or not unit.ordinal:
            continue
        after = unit.end + 1
        following = stripped[after].lower() if after < len(stripped) else ""
        if following in _LINE_WORDS:
            evidence[position] = after
        elif (
            following in _RANGE_WORDS
            and position + 2 < len(units)
            and units[position + 1].kind == "range"
        ):
            evidence[position] = evidence[position + 2]
    return evidence


def _mentions(units: list[_Unit], stripped: list[str]) -> list[int]:
    # Per unit, the word index that marks it as a line number, or -1 when nothing does. Which side
    # of the cut that index falls on decides whether a number the previous chunk ended on is a
    # mention this chunk makes or one it already made.
    ordinals = _ordinal_mentions(units, stripped)
    evidence = [-1] * len(units)
    for position, unit in enumerate(units):
        if unit.kind != "number":
            continue
        sources = [ordinals[position]]
        if unit.owner is not None:
            sources.append(unit.index)
        if unit.index > 0 and stripped[unit.index - 1].lower() in _LINE_WORDS:
            sources.append(unit.index - 1)
        if (
            position >= 2
            and units[position - 1].kind == "range"
            and units[position - 2].kind == "number"
        ):
            sources.append(evidence[position - 2])
        evidence[position] = max(sources)

    for position, unit in enumerate(units):
        if unit.mirrors is not None:
            shared = max(evidence[position], evidence[unit.mirrors])
            evidence[position] = evidence[unit.mirrors] = shared

    return evidence


def _nearest(units: list[_Unit], paths: list[int], index: int) -> int:
    best = paths[0]
    best_score = (abs(units[best].index - index), units[best].index > index)
    for candidate in paths[1:]:
        score = (abs(units[candidate].index - index), units[candidate].index > index)
        if score < best_score:
            best, best_score = candidate, score
    return best


def _voiced(unit: _Unit, evidence: int, offset: int) -> bool:
    if unit.end >= offset:
        return True
    # A number the previous chunk ended on is reported again only when the word that makes it a
    # line number sits on the far side of the cut.
    return unit.kind == "number" and unit.end == offset - 1 and evidence >= offset


def _bind(
    units: list[_Unit], evidence: list[int], carry: str | None, offset: int
) -> tuple[list[Position], str | None]:
    paths = [
        position
        for position, unit in enumerate(units)
        if unit.kind == "path" and unit.index >= offset
    ]
    claimed: set[int] = set()
    targets: dict[int, str] = {}

    for position, unit in enumerate(units):
        if evidence[position] < 0 or not _voiced(unit, evidence[position], offset):
            continue
        if unit.owner is not None:
            claimed.add(unit.owner)
            targets[position] = units[unit.owner].path
        elif not paths:
            targets[position] = carry or ""
        else:
            owner = _nearest(units, paths, unit.index)
            claimed.add(owner)
            targets[position] = units[owner].path

    positions: list[Position] = []
    for position, unit in enumerate(units):
        if not _voiced(unit, evidence[position], offset):
            continue
        if evidence[position] >= 0:
            positions.append(Position(path=targets[position], line=unit.value))
        elif unit.kind == "path" and position not in claimed:
            positions.append(Position(path=unit.path))

    return positions, units[paths[-1]].path if paths else None


def _seam_words(units: list[_Unit], stripped: list[str]) -> str:
    # The seam reaches back over the whole chain the scanner walks to flag a number rather than a
    # fixed count of words, or a chain wider than the window is lost at the cut.
    start = len(stripped)
    for unit in reversed(units):
        if unit.kind not in ("number", "range"):
            break
        start = unit.index
    if start and stripped[start - 1].lower() in _LINE_WORDS:
        start -= 1
    return " ".join(stripped[max(start, len(stripped) - _SEAM_LIMIT) :])


def extract_positions(text: str) -> list[Position]:
    units, stripped = _scan(text)
    positions, _ = _bind(units, _mentions(units, stripped), None, 0)
    return positions


class TurnRegistry:
    def __init__(self) -> None:
        self._turn_id: str | None = None
        self._paths: dict[str, set[int]] = {}
        self._keys: set[tuple[str, int | None]] = set()
        self._carry: dict[str, str] = {}
        self._seam: dict[str, str] = {}

    def open_turn(self, turn_id: str) -> None:
        self._turn_id = turn_id
        self._paths = {}
        self._keys = set()
        self._carry = {}
        self._seam = {}

    def record(self, turn_id: str, result: SearchResult) -> None:
        if turn_id != self._turn_id:
            logger.warning("record_ignored turn_id=%s open_turn=%s", turn_id, self._turn_id)
            return

        for position in result.positions():
            self._paths.setdefault(position.path, set()).add(position.line)
        self._rederive_keys()

    def abandon(self, turn_id: str) -> None:
        if turn_id != self._turn_id:
            return
        self._turn_id = None
        self._paths = {}
        self._keys = set()
        self._carry = {}
        self._seam = {}

    def known(self, turn_id: str, position: Position) -> bool:
        return turn_id == self._turn_id and position.key() in self._keys

    def verify(self, turn_id: str, text: str) -> GroundingVerdict:
        return self._verdict(turn_id, extract_positions(text))

    def verify_chunk(self, turn_id: str, text: str, source: str = "model") -> GroundingVerdict:
        open_turn = turn_id == self._turn_id
        seam = self._seam.get(source, "") if open_turn else ""
        window = f"{seam} {text}" if seam else text
        offset = len(seam.split())
        units, stripped = _scan(window, offset)
        carry = self._carry.get(source) if open_turn else None
        positions, last = _bind(units, _mentions(units, stripped), carry, offset)
        verdict = self._verdict(turn_id, positions)
        if open_turn:
            self._seam[source] = _seam_words(units, stripped)
            if verdict.ok and last is not None:
                self._carry[source] = last
        return verdict

    def _verdict(self, turn_id: str, positions: list[Position]) -> GroundingVerdict:
        ungrounded = [position for position in positions if not self.known(turn_id, position)]
        return GroundingVerdict(ok=not ungrounded, ungrounded=ungrounded)

    def _rederive_keys(self) -> None:
        basename_owners: dict[str, list[str]] = {}
        for path in self._paths:
            basename_owners.setdefault(posixpath.basename(path), []).append(path)

        keys: set[tuple[str, int | None]] = set()
        for path, lines in self._paths.items():
            for line in lines:
                keys.add((path, line))
            keys.add((path, None))

        for basename, owners in basename_owners.items():
            if len(owners) != 1:
                continue
            path = owners[0]
            for line in self._paths[path]:
                keys.add((basename, line))
            keys.add((basename, None))

        self._keys = keys
