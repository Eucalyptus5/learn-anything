import logging
import posixpath

from tutor.tools.citations import _window, extract_positions
from tutor.tools.models import GroundingVerdict, Position, SearchResult

logger = logging.getLogger(__name__)


class TurnRegistry:
    def __init__(self) -> None:
        self._turn_id: str | None = None
        self._paths: dict[str, set[int]] = {}
        self._keys: set[tuple[str, int | None]] = set()
        self._carry: dict[str, str] = {}
        self._seam: dict[str, str] = {}
        self._heard: dict[str, str] = {}

    def open_turn(self, turn_id: str) -> None:
        self._turn_id = turn_id
        self._paths = {}
        self._keys = set()
        self._carry = {}
        self._seam = {}
        self._heard = {}

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
        self._heard = {}

    def known(self, turn_id: str, position: Position) -> bool:
        return turn_id == self._turn_id and position.key() in self._keys

    def verify(self, turn_id: str, text: str) -> GroundingVerdict:
        return self._verdict(turn_id, extract_positions(text))

    def verify_chunk(self, turn_id: str, text: str, source: str = "model") -> GroundingVerdict:
        open_turn = turn_id == self._turn_id
        heard = self._heard.get(source, "") if open_turn else ""
        seam = self._seam.get(source, "") if open_turn else ""
        carry = self._carry.get(source) if open_turn else None

        positions, last, heard_tail = _window(heard, text, carry)
        verdict = self._verdict(turn_id, positions)
        seam_tail = heard_tail
        # A withheld chunk still asserted its line words, so the scanned tail can withhold on a
        # number they mark; only the spoken tail may bind one to a path or advance the carry.
        if seam != heard:
            scanned, _, seam_tail = _window(seam, text, carry)
            ungrounded = list(verdict.ungrounded)
            for position in scanned:
                if not self.known(turn_id, position) and position not in ungrounded:
                    ungrounded.append(position)
            if ungrounded:
                verdict = GroundingVerdict(ok=False, ungrounded=ungrounded)

        if open_turn:
            self._seam[source] = seam_tail
            if verdict.ok:
                self._heard[source] = heard_tail
                if last is not None:
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
