from pydantic import BaseModel


class TurnUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    reasoning_chars: int = 0


# promotional pricing; list price is double this and the promotion expires 2026-09-09.
INPUT_USD_PER_MTOK = 0.075
OUTPUT_USD_PER_MTOK = 0.250
LIST_PRICE_MULTIPLIER = 2.0


def turn_cost_usd(usage: TurnUsage, list_price: bool = False) -> float:
    cost = (
        usage.prompt_tokens * INPUT_USD_PER_MTOK / 1e6
        + usage.completion_tokens * OUTPUT_USD_PER_MTOK / 1e6
    )
    return cost * LIST_PRICE_MULTIPLIER if list_price else cost


class UsageLedger:
    def __init__(self) -> None:
        self.turns = 0
        self.total = TurnUsage(prompt_tokens=0, completion_tokens=0)

    def add(self, usage: TurnUsage) -> None:
        self.turns += 1
        self.total = TurnUsage(
            prompt_tokens=self.total.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self.total.completion_tokens + usage.completion_tokens,
            cached_tokens=self.total.cached_tokens + usage.cached_tokens,
            reasoning_chars=self.total.reasoning_chars + usage.reasoning_chars,
        )

    def cost_usd(self, list_price: bool = False) -> float:
        return turn_cost_usd(self.total, list_price)
