import logging
import re
from typing import Literal, NamedTuple

logger = logging.getLogger(__name__)

TAG_NAMES = ("step", "scene", "set", "point", "look", "orbit", "hand", "take", "draw", "clear")
TAG_MAX_CHARS = 240
_OPENS = re.compile(r"<(?:" + "|".join(TAG_NAMES) + r")(?![a-z])", re.IGNORECASE)
_MARKER = re.compile(r"(step|scene) ([1-9][0-9]?)")
_CLOSERS = "\"')]"
_SENTENCE_END = re.compile(r"[.?!][\"')\]]*(?=\s)")


class RawTag(NamedTuple):
    text: str


class Marker(NamedTuple):
    kind: Literal["step", "scene"]
    n: int


def _could_open(text: str) -> bool:
    return any(name.startswith(text[1:].lower()) for name in TAG_NAMES)


def _close(text: str, quoted: bool, escaped: bool) -> tuple[int | None, bool, bool]:
    for at, char in enumerate(text):
        if escaped:
            escaped = False
        elif quoted:
            if char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == ">":
            return at + 1, quoted, escaped
    return None, quoted, escaped


class TagSplitter:
    def __init__(self) -> None:
        self._held = ""
        self._skipped = 0
        self._quoted = False
        self._escaped = False
        self._deferred: list[RawTag] = []
        self._last = ""
        self._ended = False
        self._gap = False

    def feed(self, delta: str) -> list[str | RawTag]:
        if self._skipped:
            end, self._quoted, self._escaped = _close(delta, self._quoted, self._escaped)
            if end is None:
                self._skipped += len(delta)
                return []
            logger.info("tag.unterminated chars=%d", self._skipped + end)
            self._skipped = 0
            delta = delta[end:]
        out: list[str | RawTag] = []
        text, self._held = self._held + delta, ""
        while text:
            at = text.find("<")
            if at < 0:
                self._text(out, text)
                break
            if at:
                self._text(out, text[:at])
                text = text[at:]
            if _OPENS.match(text):
                end, quoted, escaped = _close(text, False, False)
                if end is None and len(text) <= TAG_MAX_CHARS:
                    self._held = text
                    break
                if end is None:
                    self._skipped, self._quoted, self._escaped = len(text), quoted, escaped
                    break
                if end > TAG_MAX_CHARS:
                    logger.info("tag.unterminated chars=%d", end)
                else:
                    self._tag(out, RawTag(text[1 : end - 1]))
                text = text[end:]
            elif _could_open(text):
                self._held = text
                break
            else:
                self._text(out, "<")
                text = text[1:]
        return out

    def finish(self) -> list[str | RawTag]:
        dropped = self._skipped or len(self._held)
        if dropped:
            logger.info("tag.unterminated chars=%d", dropped)
        deferred = self._deferred
        self._held, self._skipped, self._quoted, self._escaped = "", 0, False, False
        self._deferred, self._last, self._ended, self._gap = [], "", False, False
        return deferred

    def _text(self, out: list[str | RawTag], text: str) -> None:
        if self._gap and not text[0].isspace():
            text = " " + text
        self._gap = False
        if self._deferred and self._ended and text[0].isspace():
            out.extend(self._deferred)
            self._deferred = []
        end = _SENTENCE_END.search(text) if self._deferred else None
        if end is None:
            out.append(text)
        else:
            out.append(text[: end.end()])
            out.extend(self._deferred)
            self._deferred = []
            if rest := text[end.end() :]:
                out.append(rest)
        if stripped := text.rstrip().rstrip(_CLOSERS).rstrip():
            self._last = stripped[-1]
        if body := text.rstrip(_CLOSERS):
            self._ended = body[-1] in ".?!"

    def _tag(self, out: list[str | RawTag], tag: RawTag) -> None:
        if self._last in ("", ".", "?", "!"):
            out.extend(self._deferred)
            self._deferred = []
            out.append(tag)
            return
        self._deferred.append(tag)
        self._gap = True
        logger.info("tag.deferred name=%s", tag_name(tag))


def tag_name(tag: RawTag) -> str:
    name = re.split(r"[^a-z]", tag.text.lower(), maxsplit=1)[0]
    return name if name in TAG_NAMES else "unknown"


def parse_marker(tag: RawTag) -> Marker | str:
    match = _MARKER.fullmatch(tag.text)
    if match is not None:
        return Marker(match.group(1), int(match.group(2)))
    return "malformed" if tag_name(tag) in ("step", "scene") else "unsupported"
