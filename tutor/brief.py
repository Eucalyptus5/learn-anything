import logging
from enum import Enum, auto
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

BRIEF_MARKER = "<visual>"
BRIEF_END = "</visual>"
BRIEF_LIMIT = 600


class VisualBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["diagram", "app", "none"]
    title: str = Field(max_length=80)
    show: str = Field(max_length=400)


def parse_brief(text: str) -> VisualBrief | None:
    try:
        return VisualBrief.model_validate_json(text)
    except ValidationError:
        logger.info("brief.unparsed chars=%d", len(text))
        return None


class _State(Enum):
    BEFORE = auto()
    HEAD = auto()
    CLOSING = auto()
    AFTER = auto()


class BriefSplitter:
    def __init__(self) -> None:
        self.brief: VisualBrief | None = None
        self._held = ""
        self._state = _State.BEFORE
        self._skip_blank = False
        self._oversized = False
        self._scanned = 0
        self._started = False
        self._plain = False
        self._depth = 0
        self._in_string = False
        self._escaped = False

    def feed(self, delta: str) -> str:
        if self._state is _State.AFTER:
            if not self._skip_blank:
                return delta
            delta = delta.lstrip()
            if delta:
                self._skip_blank = False
            return delta
        self._held += delta
        if self._state is _State.BEFORE:
            text = self._held.lstrip()
            if not text:
                return ""
            if text.startswith(BRIEF_MARKER):
                self._state = _State.HEAD
                self._held = text[len(BRIEF_MARKER) :]
            elif BRIEF_MARKER.startswith(text):
                return ""
            else:
                return self._release(self._held)
        if self._state is _State.HEAD:
            end = self._object_end()
            at = self._held.find(BRIEF_END)
            if at >= 0 and (end is None or at <= end):
                self._parse(self._held[:at])
                return self._release(self._held[at + len(BRIEF_END) :])
            if end is None:
                if len(self._held) >= BRIEF_LIMIT:
                    if not self._oversized:
                        self._oversized = True
                        logger.info("brief.oversized chars=%d", len(self._held))
                    self._held = self._held[1 - len(BRIEF_END) :]
                    self._scanned = len(self._held)
                return ""
            self._parse(self._held[:end])
            self._held = self._held[end:]
            self._state = _State.CLOSING
        text = self._held.lstrip()
        if not text:
            return ""
        if text.startswith(BRIEF_END):
            return self._release(text[len(BRIEF_END) :])
        if BRIEF_END.startswith(text):
            return ""
        return self._release(self._held)

    def finish(self) -> str:
        held, self._held = self._held, ""
        if self._state is _State.AFTER:
            return ""
        if self._state is _State.BEFORE:
            return held
        if self._state is _State.HEAD:
            logger.info("brief.unterminated chars=%d", len(held))
            return ""
        text = held.lstrip()
        return "" if BRIEF_END.startswith(text) else text

    def _parse(self, body: str) -> None:
        if not self._oversized:
            self.brief = parse_brief(body.strip())

    def _release(self, rest: str) -> str:
        rest = rest.lstrip()
        self._state = _State.AFTER
        self._held = ""
        self._skip_blank = not rest
        return rest

    def _object_end(self) -> int | None:
        if self._plain:
            return None
        held = self._held
        while self._scanned < len(held):
            char = held[self._scanned]
            self._scanned += 1
            if not self._started:
                if char.isspace():
                    continue
                if char != "{":
                    self._plain = True
                    return None
                self._started = True
                self._depth = 1
            elif self._in_string:
                if self._escaped:
                    self._escaped = False
                elif char == "\\":
                    self._escaped = True
                elif char == '"':
                    self._in_string = False
            elif char == '"':
                self._in_string = True
            elif char == "{":
                self._depth += 1
            elif char == "}":
                self._depth -= 1
                if self._depth == 0:
                    return self._scanned
        return None
