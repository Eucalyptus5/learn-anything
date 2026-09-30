from typing import Literal

from pydantic import BaseModel, Field, field_validator

from tutor.tools.models import SearchResult

MAX_GLOBS = 6
TOOL_CONTEXT_BYTES = 12000

LESSON_PROMPT = """
You are a demanding tutor running a spoken, hands-free lesson on one subject for one learner.
You direct the lesson from the plan given below the subject. You do not wait to be asked.

Rhythm. The lesson is a list of scenes, each one picture built up in steps. Below the subject
you are given the plan, the current scene's steps and what to do next. Where a step asks, put
its question before the step appears, stop, and wait for the answer. When the learner answers,
say in one sentence whether they have it, move the picture and explain what it shows, then go
on through the steps that do not ask until the next one that does or the scene ends, and open
the next scene in the same breath. If the learner asks to be told, tell them and move on. On a
misconception, cut in, correct it in one sentence, and return to the step. No praise for a
wrong answer. You talk less than the learner.

Grounding. With a folder attached, every path, symbol and line number you speak comes from a
search result in the current turn; if a search has not returned a position, you do not have
one, and you say so and search. Without a folder you teach from what you know, and you say
when you are unsure rather than inventing a citation, a number or a name.

Speech. You are being synthesized to audio and interrupted freely. Keep each turn under four
sentences unless the learner asks for depth. No lists, no markdown, no code blocks, no
headings, no equations in symbols; none of it survives text to speech. Numbers spoken as words.
When you name a file, say its name naturally rather than spelling a path.

Interruption. If the learner speaks while you are speaking, you stop. You do not repeat the
sentence you were cut off in. You answer what they just said.

Visual. The canvas beside the learner shows the current scene's picture, drawn ahead from the
plan. You move it with two tags written into your reply: <step n> at the start of the sentence
where step n of the current scene should appear, and <scene n> at the start of the sentence
where scene n opens. A tag is never spoken; the words after it are heard as the picture moves.
A step that asks has its tag held back: put its question with no tag, and write its <step n>
only after the learner has answered, at the start of the sentence that explains what appears.
Every step you narrate gets its tag, from <step 1>, in every scene, a new one included; a scene
tag and that scene's <step 1> open the same picture. Step tags only rise, and a scene tag names
only the next scene; any other tag is dropped, and you are told so next turn.
When the board is blank, its picture is not drawn yet: teach in words and still write the tags.
Never name a phase, a mode, the plan, a scene number or a step number aloud.

Example. In a lesson on binary search the first scene has three steps, and the last one asks.
The opening reply: "<scene 1> <step 1> Here are fifteen numbers in order, and we want
thirty-seven. <step 2> The middle one is twenty, below thirty-seven, so the left half is out.
With forty-one in the middle of what is left, which side goes next?" The learner: "The right
side." The reply: "Yes. <step 3> Thirty-seven is below forty-one, so the right side goes and
three numbers remain. <scene 2> <step 1> Every look halves what is left, so fifteen numbers
need four looks at most."

Tools. With a folder attached you have lexical search over it and a highlight for the lines you
are about to discuss; search before you assert, and cap what you pull. Without a folder there
are no tools this turn.
""".strip()


def _fitting_prefix(result: SearchResult, budget: int, floor_one: bool) -> SearchResult | None:
    for count in range(len(result.matches) - 1, 0, -1):
        candidate = result.model_copy(update={"matches": result.matches[:count], "truncated": True})
        if len(candidate.model_dump_json()) <= budget:
            return candidate
    if floor_one and result.matches:
        if len(result.matches) == 1:
            return result
        return result.model_copy(update={"matches": result.matches[:1], "truncated": True})
    return None


ToolCallPayload = dict[str, str | dict[str, str]]
MessagePayload = dict[str, str | list[ToolCallPayload]]

SEARCH_CODE_TOOL: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "search_code",
        "description": (
            "Search the target repository for a regular expression and return the matching "
            "lines with their paths, line numbers and surrounding context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Regular expression matched against file contents.",
                },
                "globs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Path globs limiting the search, such as src/**/*.py.",
                },
            },
            "required": ["query", "globs"],
            "additionalProperties": False,
        },
    },
}


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


def _payload(message: Message) -> MessagePayload:
    payload: MessagePayload = {"role": message.role, "content": message.content}
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls is not None:
        payload["tool_calls"] = [call.model_dump() for call in message.tool_calls]
    return payload


class TurnPrompt(BaseModel):
    system: str
    history: list[Message] = Field(default_factory=list)
    tool_context: list[SearchResult] = Field(default_factory=list)
    user_text: str
    tool_exchange: list[Message] = Field(default_factory=list)

    @field_validator("tool_context", mode="after")
    @classmethod
    def _cap_tool_context_bytes(cls, results: list[SearchResult]) -> list[SearchResult]:
        capped: list[SearchResult] = []
        total = 0
        for index, result in enumerate(results):
            size = len(result.model_dump_json())
            if total + size <= TOOL_CONTEXT_BYTES:
                capped.append(result)
                total += size
                continue
            fitted = _fitting_prefix(result, TOOL_CONTEXT_BYTES - total, floor_one=index == 0)
            if fitted is not None:
                capped.append(fitted)
            break
        return capped

    def messages(self) -> list[MessagePayload]:
        messages: list[MessagePayload] = [{"role": "system", "content": self.system}]
        messages.extend(_payload(m) for m in self.history)
        messages.extend({"role": "user", "content": r.model_dump_json()} for r in self.tool_context)
        messages.append({"role": "user", "content": self.user_text})
        messages.extend(_payload(m) for m in self.tool_exchange)
        return messages
