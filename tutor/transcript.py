from tutor.prompt import Message


class Transcript:
    def __init__(self, turns: int) -> None:
        self._turns = turns
        self._order: list[str] = []
        self._learner: dict[str, str] = {}
        self._tutor: dict[str, list[str]] = {}
        self._head: dict[str, str] = {}
        self._tail: dict[str, str] = {}

    def learner(self, turn_id: str, text: str) -> None:
        if turn_id in self._learner:
            return
        self._order.append(turn_id)
        self._learner[turn_id] = text

    def head(self, turn_id: str, line: str) -> None:
        self._head[turn_id] = line

    def tutor(self, turn_id: str, clause: str) -> None:
        self._tutor.setdefault(turn_id, []).append(clause)

    def tail(self, turn_id: str, line: str) -> None:
        self._tail[turn_id] = line

    def history(self, before: str) -> list[Message]:
        cut = self._order.index(before) if before in self._order else len(self._order)
        closed = self._order[:cut]
        messages: list[Message] = []
        for turn_id in closed[max(len(closed) - self._turns, 0) :]:
            messages.append(Message(role="user", content=self._learner[turn_id]))
            parts = [
                self._head.get(turn_id, ""),
                " ".join(self._tutor.get(turn_id, [])),
                self._tail.get(turn_id, ""),
            ]
            content = "\n".join(part for part in parts if part)
            if content:
                messages.append(Message(role="assistant", content=content))
        return messages
