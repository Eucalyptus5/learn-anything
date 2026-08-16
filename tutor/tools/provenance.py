import logging
import posixpath

from tutor.tools.models import Position, SearchResult

logger = logging.getLogger(__name__)


class TurnRegistry:
    def __init__(self) -> None:
        self._turn_id: str | None = None
        self._paths: dict[str, set[int]] = {}
        self._keys: set[tuple[str, int | None]] = set()

    def open_turn(self, turn_id: str) -> None:
        self._turn_id = turn_id
        self._paths = {}
        self._keys = set()

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

    def known(self, turn_id: str, position: Position) -> bool:
        return turn_id == self._turn_id and position.key() in self._keys

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
