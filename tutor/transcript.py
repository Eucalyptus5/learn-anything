from tutor.prompt import Message


class Transcript:
    def __init__(self, turns: int) -> None:
        self._turns = turns
        self._order: list[str] = []
        self._learner: dict[str, str] = {}
        self._tutor: dict[str, list[str]] = {}

    def learner(self, turn_id: str, text: str) -> None:
        if turn_id in self._learner:
            return
        self._order.append(turn_id)
        self._learner[turn_id] = text

    def tutor(self, turn_id: str, clause: str) -> None:
        self._tutor.setdefault(turn_id, []).append(clause)

    def latest(self) -> str | None:
        return self._order[-1] if self._order else None

    def history(self, before: str) -> list[Message]:
        cut = self._order.index(before) if before in self._order else len(self._order)
        return self._messages(self._order[:cut], self._turns)

    def since(self, after: str | None, turns: int) -> list[Message]:
        start = self._order.index(after) + 1 if after in self._order else 0
        return self._messages(self._order[start:], turns)

    def _messages(self, order: list[str], turns: int) -> list[Message]:
        messages: list[Message] = []
        for turn_id in order[max(len(order) - turns, 0) :]:
            messages.append(Message(role="user", content=self._learner[turn_id]))
            content = " ".join(self._tutor.get(turn_id, []))
            if content:
                messages.append(Message(role="assistant", content=content))
        return messages
