import asyncio
import logging
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from tutor.tools.models import PATH_EXTENSIONS, SearchResult
from tutor.tools.ripgrep import READ_CHUNK_BYTES, reap

logger = logging.getLogger(__name__)

MAX_GLOBS = 6
WALK_TIMEOUT_MS = 2000
TOOL_CONTEXT_BYTES = 12000

SYSTEM_PROMPT = """
You are a demanding tutor running a spoken, hands-free lesson on one subject for one learner.
You direct the curriculum. You do not wait to be asked.

Rhythm. You move through three phases and you name the phase you are in.
Teach: introduce one mechanism, why it exists and how it works, one idea per turn, and end
with a question the learner can answer from what you just said. Never ask about a term you
have not introduced.
Concrete: make the mechanism tangible. A worked example with numbers, one step of the
derivation, a plot, a trace of one iteration; with a folder attached, the exact lines. Never
read syntax aloud; say what it does.
Interrogate: stop lecturing and test. Pose an edge case, a failure mode or a limit, and have
the learner explain the mechanism back. At most two probes per gap, then explain. If the
learner asks to be told, tell them and move on. On a misconception, cut in, correct it in one
sentence, and return to Concrete on the counterexample that settles it. No praise for a wrong
answer. No softening a gap into a near miss.

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

Visual. The canvas beside the learner is drawn by a second author from a brief you write.
Open every reply with exactly one line, <visual>{"kind": ..., "title": ..., "show": ...}</visual>,
before any spoken word. kind is diagram for a topology, a lifecycle or who calls whom; app for
a plot, an animation, a worked example or a typeset equation; none when what is on the canvas
still fits or nothing would help. title is under eight words. show is one or two sentences
saying exactly what the picture must contain, with the numbers and the case. Then speak; the
picture lands while you talk, and you may refer to it. Draw for the mechanism, never for
decoration.

Tools. With a folder attached you have lexical search over it and a highlight for the lines you
are about to discuss; search before you assert, and cap what you pull. Without a folder there
are no tools this turn.
""".strip()

TOKEN_TRIM = "\"'`()[]{}<>,.;:!?"
WORD = re.compile(r"[a-z]+")

# "go" and "c" are ordinary English words in a spoken turn; the tree walk covers those trees.
LANGUAGE_SUFFIXES: dict[str, list[str]] = {
    "golang": [".go"],
    "python": [".py", ".pyi"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx"],
    "rust": [".rs"],
    "java": [".java"],
    "kotlin": [".kt"],
    "ruby": [".rb"],
    "swift": [".swift"],
    "html": [".html"],
    "css": [".css"],
    "sql": [".sql"],
    "markdown": [".md"],
    "yaml": [".yaml", ".yml"],
    "toml": [".toml"],
    "json": [".json"],
}

_walk_cache: dict[Path, list[str]] = {}


def _named_files(user_text: str) -> list[str]:
    globs = []
    for token in user_text.split():
        path = PurePosixPath(token.strip(TOKEN_TRIM))
        if path.suffix not in PATH_EXTENSIONS or not path.stem:
            continue
        parent = path.parent
        globs.append(f"**/*{path.suffix}" if str(parent) == "." else f"{parent}/**/*{path.suffix}")
    return list(dict.fromkeys(globs))


def _language_words(user_text: str) -> list[str]:
    globs = []
    for word in WORD.findall(user_text.lower()):
        for suffix in LANGUAGE_SUFFIXES.get(word, ()):
            globs.append(f"**/*{suffix}")
    return list(dict.fromkeys(globs))


def _ranked(counts: Counter[str]) -> list[str]:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [f"**/*{suffix}" for suffix, _ in ordered] or ["**/*"]


async def _walk(root: Path) -> list[str]:
    if root in _walk_cache:
        return _walk_cache[root]

    proc = await asyncio.create_subprocess_exec(
        "rg",
        "--files",
        "--no-require-git",
        cwd=root,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    counts: Counter[str] = Counter()
    buffer = b""
    try:
        try:
            async with asyncio.timeout(WALK_TIMEOUT_MS / 1000):
                while True:
                    chunk = await proc.stdout.read(READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    *lines, buffer = (buffer + chunk).split(b"\n")
                    for line in lines:
                        suffix = PurePosixPath(line.decode(errors="replace")).suffix
                        if suffix in PATH_EXTENSIONS:
                            counts[suffix] += 1
        except TimeoutError:
            await reap(proc)
            logger.warning("walk_timeout root=%s suffixes=%d", root, len(counts))
            return _ranked(counts)
        await proc.wait()
    except asyncio.CancelledError:
        await reap(proc)
        raise

    _walk_cache[root] = _ranked(counts)
    return _walk_cache[root]


async def derive_globs(user_text: str, root: Path) -> list[str]:
    globs = _named_files(user_text) or _language_words(user_text) or await _walk(root)
    return globs[:MAX_GLOBS]


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
