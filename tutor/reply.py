import re
from collections.abc import Mapping
from typing import Literal, NamedTuple

from tutor.lesson import OPENING_TEXT, Cursor, LessonPlan, Scene, ScriptChunk
from tutor.prompt import Message
from tutor.tags import Marker

LABELS = (
    "answered_right",
    "answered_wrong",
    "answered_partly",
    "side_question",
    "go_on",
    "tell_me",
    "other",
)
ANSWERS = LABELS[:3]
BRIDGED = ("go_on", "answered_right", "tell_me")
RELAXED = re.compile(r"\s*(?:label\s*:\s*)?(" + "|".join(LABELS) + r")\b\s*:?", re.IGNORECASE)
STARTS = (*LABELS, *(f"label:{label}" for label in LABELS))
STRIP = re.compile(
    r"\blabel\s*:|\bother\s*:|\b(?:"
    + "|".join(label for label in LABELS if "_" in label)
    + r")\b\s*:?",
    re.IGNORECASE,
)
RIGHT_BRIDGES = ("Yes, that's right.", "Exactly right.", "Right, well done.")
TELL_ME_BRIDGE = "Sure, here it is."
RECENT = 6

LIVE_PROMPT = """
You are the live voice of a spoken lesson, taught aloud to one learner beside a picture. The
learner has just spoken. You label what they said, and for some labels you write the tutor's
short reaction. The lesson's next part is prepared already and plays straight after you; you
never see it, so never narrate it.

The first word of your reply is exactly one of the labels below, written as it is here, with
nothing before it.

The labels:
answered_right: a question is pending and the learner answers it correctly.
answered_wrong: a question is pending and the learner's answer is wrong.
answered_partly: a question is pending and the answer is partly right or incomplete.
side_question: the learner asks a question of their own instead of answering or moving on.
go_on: the learner acknowledges, agrees, comments, or asks to continue, and is not answering a
pending question.
tell_me: the learner asks to be told the answer, or to have the rest explained without
questions.
other: anything else.

On go_on, answered_right and tell_me, write the label and nothing else: the tutor's words there
are prepared. On side_question, answered_wrong, answered_partly and other, write the label, a
new line, then the tutor's reaction, which is spoken aloud as you write it.

The reaction. On a wrong or partial answer, write one kind sentence saying it is not quite
right, with no hint, nothing like "think back" or "think again", and without handing the turn
back: the answer plays straight after your sentence. On a side question, answer it briefly, then
put the pending question again in words close to it; with no question pending, just answer it.
Otherwise, reply briefly and naturally, in one to three short sentences.

You are shown what the picture will show once the pending question is answered, only so you
can judge the answer. Never describe it, never say what comes next in the lesson, and never
give the pending question's answer unless the learner has already given it. Write plain spoken
English in ASCII only: no symbols, no markdown, no angle brackets, no tags, and no label name
inside the reaction. Never name a scene number or a step number aloud.
""".strip()


class Question(NamedTuple):
    scene_id: str
    n: int


class Piece(NamedTuple):
    cue: Marker | None
    text: str
    question: Question | None


class Missing(NamedTuple):
    n: int


def read_label(text: str, ended: bool) -> tuple[Literal["label", "none", "wait"], str | None, int]:
    found = RELAXED.match(text)
    if found is not None and (
        found.group(1).lower() != "other" or found.end(1) < len(text) or ended
    ):
        return "label", found.group(1).lower(), found.end()
    first = text.lstrip().split("\n", 1)
    head = "".join(first[0].split()).lower()
    if ended or len(first) > 1 or not any(start.startswith(head) for start in STARTS):
        return "none", None, 0
    return "wait", None, 0


def live_text(
    subject: str, scene: Scene, pending: int | None, history: list[Message], learner: str
) -> str:
    lines = [f"Subject: {subject}", f"Current scene: {scene.title}"]
    if pending is None:
        lines.append("Pending question: none.")
    else:
        step = scene.steps[pending - 1]
        lines.append(f"Pending question: {step.ask}")
        lines.append(
            f"What the picture will show once it is answered, for judging only: {step.show}"
        )
    said = [m for m in history if not (m.role == "user" and m.content == OPENING_TEXT)]
    lines += ["", "Recent conversation, oldest first:"]
    lines += [
        f"{'Tutor' if m.role == 'assistant' else 'Learner'}: {m.content}" for m in said[-RECENT:]
    ]
    lines += ["", f"The learner now says: {learner}"]
    return "\n".join(lines)


def bridge_for(label: str | None, pending: int | None, rights: int) -> tuple[str, int]:
    if label == "tell_me":
        return TELL_ME_BRIDGE, rights
    if label == "answered_right" and pending is not None:
        return RIGHT_BRIDGES[rights % len(RIGHT_BRIDGES)], rights + 1
    return "", rights


def _pick(script: list[ScriptChunk], step: int, question: bool) -> str:
    return next(chunk.text for chunk in script if (chunk.step, chunk.question) == (step, question))


def _cue(n: int, step: int, opened: bool) -> Marker:
    return Marker("scene", n) if opened and step == 1 else Marker("step", step)


def _play_from(
    scene: Scene, n: int, script: list[ScriptChunk], first: int, answered: bool, opened: bool
) -> list[Piece]:
    pieces: list[Piece] = []
    for step in range(first, len(scene.steps) + 1):
        if scene.steps[step - 1].ask and not (answered and step == first):
            question = Question(scene.id, step)
            return [*pieces, Piece(None, _pick(script, step, True), question)]
        pieces.append(Piece(_cue(n, step, opened), _pick(script, step, False), None))
    return pieces


def _open(
    plan: LessonPlan, scripts: Mapping[str, list[ScriptChunk]], n: int
) -> tuple[list[Piece], Missing | None]:
    scene = plan.scenes[n - 1]
    script = scripts.get(scene.id)
    if script is None:
        return [], Missing(n)
    return _play_from(scene, n, script, 1, False, True), None


def direct(
    plan: LessonPlan,
    scripts: Mapping[str, list[ScriptChunk]],
    move: str,
    at: Cursor,
    pending: int | None,
) -> tuple[list[Piece], Missing | None]:
    n, step = at.scene, at.step
    if move == "open" or (n == 0 and pending is None):
        return _open(plan, scripts, 1)
    if move not in (*ANSWERS, "go_on", "tell_me"):
        return [], None
    # with nothing open, or at a scene's end, a pending question is the next scene's step 1
    opened = n == 0 or step >= len(plan.scenes[n - 1].steps)
    if opened and pending is None:
        return ([], None) if n == len(plan.scenes) else _open(plan, scripts, n + 1)
    if opened:
        n, step = n + 1, 0
    scene = plan.scenes[n - 1]
    script = scripts.get(scene.id)
    if script is None:
        return [], Missing(n)
    if move == "tell_me":
        told = range(step + 1, len(scene.steps) + 1)
        return [Piece(_cue(n, k, opened), _pick(script, k, False), None) for k in told], None
    pieces = _play_from(scene, n, script, step + 1, move in ANSWERS and pending is not None, opened)
    if pieces[-1].question is not None or n == len(plan.scenes):
        return pieces, None
    following, missing = _open(plan, scripts, n + 1)
    return [*pieces, *following], missing
