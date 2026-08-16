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
)


@dataclass(frozen=True)
class _Unit:
    kind: str
    index: int
    path: str = ""
    value: int = 0
    ordinal: bool = False
    owner: int | None = None


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


def _digit_value(piece: str) -> tuple[int, bool] | None:
    if _DIGITS.match(piece):
        return int(piece), False
    ordinal = _DIGIT_ORDINAL.match(piece)
    if ordinal:
        return int(ordinal.group(1)), True
    return None


def _compose(run: list[tuple[int, str, bool]]) -> tuple[int, bool]:
    digits = ""
    index = 0
    while index < len(run):
        value, tier, _ = run[index]
        if tier == "tens" and index + 1 < len(run) and run[index + 1][1] == "ones":
            value += run[index + 1][0]
            index += 2
        else:
            index += 1
        digits += str(value)
    return int(digits), any(word[2] for word in run)


def _scan(text: str) -> tuple[list[_Unit], list[str]]:
    stripped = [_strip(token) for token in text.split()]
    units: list[_Unit] = []
    run: list[tuple[int, str, bool]] = []
    run_index = 0

    def flush() -> None:
        nonlocal run
        if not run:
            return
        value, ordinal = _compose(run)
        units.append(_Unit(kind="number", index=run_index, value=value, ordinal=ordinal))
        run = []

    for index, token in enumerate(stripped):
        suffix = _LINE_SUFFIX.search(token)
        if suffix:
            flush()
            head = token[: suffix.start()]
            if _is_path(head):
                owner = len(units)
                units.append(_Unit(kind="path", index=index, path=head))
                start, end = suffix.groups()
                units.append(_Unit(kind="number", index=index, value=int(start), owner=owner))
                if end:
                    units.append(_Unit(kind="range", index=index))
                    units.append(_Unit(kind="number", index=index, value=int(end), owner=owner))
            continue

        if _is_path(token):
            flush()
            units.append(_Unit(kind="path", index=index, path=token))
            continue

        pieces = token.split("-")
        numerals = [_digit_value(piece) for piece in pieces]
        if all(numeral is not None for numeral in numerals):
            flush()
            for offset, numeral in enumerate(numerals):
                if offset:
                    units.append(_Unit(kind="range", index=index))
                value, ordinal = numeral
                units.append(_Unit(kind="number", index=index, value=value, ordinal=ordinal))
            continue

        words = [_NUMBER_WORDS.get(piece.lower()) for piece in pieces]
        if all(word is not None for word in words):
            if not run:
                run_index = index
            run.extend(words)
            continue

        flush()
        if token.lower() in _RANGE_WORDS:
            units.append(_Unit(kind="range", index=index))

    flush()
    return units, stripped


def _mentions(units: list[_Unit], stripped: list[str]) -> list[bool]:
    flags = [False] * len(units)
    for position, unit in enumerate(units):
        if unit.kind != "number":
            continue
        after_line_word = unit.index > 0 and stripped[unit.index - 1].lower() in _LINE_WORDS
        continues_range = (
            position >= 2
            and units[position - 1].kind == "range"
            and units[position - 2].kind == "number"
            and flags[position - 2]
        )
        flags[position] = (
            unit.owner is not None or unit.ordinal or after_line_word or continues_range
        )
    return flags


def _nearest(units: list[_Unit], paths: list[int], index: int) -> int:
    best = paths[0]
    best_score = (abs(units[best].index - index), units[best].index > index)
    for candidate in paths[1:]:
        score = (abs(units[candidate].index - index), units[candidate].index > index)
        if score < best_score:
            best, best_score = candidate, score
    return best


def _bind(
    units: list[_Unit], flags: list[bool], carry: str | None
) -> tuple[list[Position], str | None]:
    paths = [position for position, unit in enumerate(units) if unit.kind == "path"]
    claimed: set[int] = set()
    targets: dict[int, str] = {}

    for position, unit in enumerate(units):
        if not flags[position]:
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
        if flags[position]:
            positions.append(Position(path=targets[position], line=unit.value))
        elif unit.kind == "path" and position not in claimed:
            positions.append(Position(path=unit.path))

    return positions, units[paths[-1]].path if paths else None


def extract_positions(text: str) -> list[Position]:
    units, stripped = _scan(text)
    positions, _ = _bind(units, _mentions(units, stripped), None)
    return positions


class TurnRegistry:
    def __init__(self) -> None:
        self._turn_id: str | None = None
        self._paths: dict[str, set[int]] = {}
        self._keys: set[tuple[str, int | None]] = set()
        self._carry: str | None = None

    def open_turn(self, turn_id: str) -> None:
        self._turn_id = turn_id
        self._paths = {}
        self._keys = set()
        self._carry = None

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
        self._carry = None

    def known(self, turn_id: str, position: Position) -> bool:
        return turn_id == self._turn_id and position.key() in self._keys

    def verify(self, turn_id: str, text: str) -> GroundingVerdict:
        return self._verdict(turn_id, extract_positions(text))

    def verify_chunk(self, turn_id: str, text: str) -> GroundingVerdict:
        units, stripped = _scan(text)
        open_turn = turn_id == self._turn_id
        carry = self._carry if open_turn else None
        positions, last = _bind(units, _mentions(units, stripped), carry)
        if open_turn and last is not None:
            self._carry = last
        return self._verdict(turn_id, positions)

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
