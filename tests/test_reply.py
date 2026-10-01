import hashlib

import pytest

from tests.test_lesson import ASKS_FIRST, DIRECTED
from tests.test_session import LESSON, SCRIPTS
from tutor.lesson import OPENING_TEXT, Cursor, LessonPlan, ScriptChunk
from tutor.prompt import Message
from tutor.reply import (
    ANSWERS,
    LIVE_PROMPT,
    STRIP,
    Missing,
    Piece,
    Question,
    bridge_for,
    direct,
    live_text,
    read_label,
)
from tutor.tags import Marker

SUBJECT = "The clipped objective in PPO"
HISTORY = [
    Message(role="user", content=OPENING_TEXT),
    Message(role="assistant", content="Here are the old and the new policy over three actions."),
    Message(role="user", content="okay"),
    Message(role="assistant", content="Each bar is the chance one of them gives an action."),
    Message(role="user", content="got it"),
    Message(role="assistant", content="Divide the new chance by the old one."),
    Message(role="user", content="sure"),
    Message(role="assistant", content="What is the ratio where they agree?"),
]
RECENT_LINES = [
    "Recent conversation, oldest first:",
    "Learner: okay",
    "Tutor: Each bar is the chance one of them gives an action.",
    "Learner: got it",
    "Tutor: Divide the new chance by the old one.",
    "Learner: sure",
    "Tutor: What is the ratio where they agree?",
    "",
    "The learner now says: one",
]


def chunk(
    scene_id: str, step: int, question: bool, scripts: dict[str, list[ScriptChunk]] = SCRIPTS
) -> str:
    return next(
        each.text for each in scripts[scene_id] if each.step == step and each.question == question
    )


ASKED_AGAIN = Piece(None, chunk("ratio", 2, True), Question("ratio", 2))


@pytest.mark.parametrize(
    ("text", "ended", "expected"),
    [
        ("go_", False, ("wait", None, 0)),
        ("go_on", False, ("label", "go_on", 5)),
        ("label: answered_right", False, ("label", "answered_right", 21)),
        ("\n\nGo_On:\nx", False, ("label", "go_on", 8)),
        ("other", False, ("wait", None, 0)),
        ("other", True, ("label", "other", 5)),
        ("other.", False, ("label", "other", 5)),
        ("otherwise", False, ("none", None, 0)),
        ("ans", False, ("wait", None, 0)),
        ("Hello", False, ("none", None, 0)),
    ],
)
def test_the_label_is_read_as_soon_as_its_word_ends(
    text: str, ended: bool, expected: tuple[str, str | None, int]
) -> None:
    assert read_label(text, ended) == expected


def test_unlabelled_text_loses_its_label_words_but_keeps_the_word_other() -> None:
    spoken = STRIP.sub("", "label: go_on Sure. other: fine, other things")

    assert spoken == " Sure.  fine, other things"


def test_the_live_prompt_is_the_prototypes() -> None:
    digest = hashlib.sha256(LIVE_PROMPT.encode()).hexdigest()

    assert digest == "9d96c0e73012195512b4d1d0dbc949ca7c4e070290e7c5dcbb7efa61d09e7602"


def test_the_live_input_is_the_prototypes() -> None:
    ratio = LESSON.scenes[0]

    pending = live_text(SUBJECT, ratio, 2, HISTORY, "one")
    settled = live_text(SUBJECT, ratio, None, HISTORY, "one")

    assert pending == "\n".join(
        [
            "Subject: The clipped objective in PPO",
            "Current scene: The ratio",
            "Pending question: What is the ratio where they agree?",
            (
                "What the picture will show once it is answered, for judging only: "
                "Their ratio at one action"
            ),
            "",
            *RECENT_LINES,
        ]
    )
    assert settled == "\n".join(
        [
            "Subject: The clipped objective in PPO",
            "Current scene: The ratio",
            "Pending question: none.",
            "",
            *RECENT_LINES,
        ]
    )


def test_the_director_opens_scene_one_to_its_first_question() -> None:
    played = direct(LESSON, SCRIPTS, "open", Cursor(), None)

    assert played == (
        [
            Piece(Marker("scene", 1), chunk("ratio", 1, False), None),
            ASKED_AGAIN,
        ],
        None,
    )


@pytest.mark.parametrize(
    ("move", "pending"), [("go_on", 2), ("go_on", None), ("answered_right", None)]
)
def test_go_on_before_an_asked_step_puts_its_question_again(move: str, pending: int | None) -> None:
    played = direct(LESSON, SCRIPTS, move, Cursor(scene=1, step=1), pending)

    assert played == ([ASKED_AGAIN], None)


@pytest.mark.parametrize("move", ["side_question", "other"])
def test_a_reaction_label_plays_no_piece(move: str) -> None:
    assert direct(LESSON, SCRIPTS, move, Cursor(scene=1, step=1), 2) == ([], None)


def test_a_missing_script_is_named() -> None:
    assert direct(LESSON, {}, "open", Cursor(), None) == ([], Missing(1))


def scripted(plan: LessonPlan) -> dict[str, list[ScriptChunk]]:
    return {
        each.id: [
            ScriptChunk(
                step=n, question=question, text=f"{each.id} {'asks' if question else 'shows'} {n}."
            )
            for n, step in enumerate(each.steps, start=1)
            for question in ((True, False) if step.ask else (False,))
        ]
        for each in plan.scenes
    }


DIRECTED_SCRIPTS = scripted(DIRECTED)
ASKS_FIRST_SCRIPTS = scripted(ASKS_FIRST)


def spelled(
    plan: LessonPlan, scripts: dict[str, list[ScriptChunk]], scene: int, pieces: str
) -> list[Piece]:
    played: list[Piece] = []
    for word in [] if pieces == "none" else pieces.split(", "):
        if word.startswith("q("):
            n, step = (int(part) for part in word[2:-1].split("."))
            scene_id = plan.scenes[n - 1].id
            text = chunk(scene_id, step, True, scripts)
            played.append(Piece(None, text, Question(scene_id, step)))
            continue
        kind, n = word.split()
        scene = int(n) if kind == "scene" else scene
        step = 1 if kind == "scene" else int(n)
        text = chunk(plan.scenes[scene - 1].id, step, False, scripts)
        played.append(Piece(Marker(kind, int(n)), text, None))
    return played


@pytest.mark.parametrize(
    ("move", "at", "pending", "pieces"),
    [
        *[(move, (1, 2), 3, "step 3, step 4, scene 2, q(2.2)") for move in ANSWERS],
        ("answered_right", (1, 1), None, "step 2, q(1.3)"),
        ("go_on", (1, 2), 3, "q(1.3)"),
        ("answered_right", (2, 1), 2, "step 2, q(2.3)"),
        ("tell_me", (2, 1), 2, "step 2, step 3"),
        *[(move, (1, 4), None, "scene 2, q(2.2)") for move in ("tell_me", "go_on")],
        ("go_on", (2, 3), None, "scene 3, step 2, step 3"),
        *[(move, (1, 2), 3, "none") for move in ("side_question", "other")],
        *[
            (move, (0, 0), None, "scene 1, step 2, q(1.3)")
            for move in ("go_on", "side_question", "tell_me")
        ],
    ],
)
def test_the_director_follows_the_prototype(
    move: str, at: tuple[int, int], pending: int | None, pieces: str
) -> None:
    played = direct(DIRECTED, DIRECTED_SCRIPTS, move, Cursor(scene=at[0], step=at[1]), pending)

    assert played == (spelled(DIRECTED, DIRECTED_SCRIPTS, at[0], pieces), None)


@pytest.mark.parametrize(
    ("move", "at", "pending", "pieces"),
    [
        ("open", (0, 0), None, "q(1.1)"),
        ("go_on", (0, 0), 1, "q(1.1)"),
        *[(move, (0, 0), 1, "scene 1, step 2, step 3, q(2.1)") for move in ANSWERS],
        ("tell_me", (0, 0), 1, "scene 1, step 2, step 3"),
        *[(move, (0, 0), 1, "none") for move in ("side_question", "other")],
        *[(move, (1, 3), None, "q(2.1)") for move in ("go_on", "tell_me", "answered_right")],
        ("go_on", (1, 3), 1, "q(2.1)"),
        ("answered_right", (1, 3), 1, "scene 2, step 2, q(2.3)"),
        ("tell_me", (1, 3), 1, "scene 2, step 2, step 3"),
        *[(move, (1, 3), 1, "none") for move in ("side_question", "other")],
    ],
)
def test_a_scene_that_asks_at_its_first_step_opens_on_the_answer(
    move: str, at: tuple[int, int], pending: int | None, pieces: str
) -> None:
    played = direct(ASKS_FIRST, ASKS_FIRST_SCRIPTS, move, Cursor(scene=at[0], step=at[1]), pending)

    assert played == (spelled(ASKS_FIRST, ASKS_FIRST_SCRIPTS, at[0], pieces), None)


def test_bridges_rotate_on_right_answers_only() -> None:
    walked = [bridge_for("answered_right", 3, rights) for rights in range(4)]

    assert walked == [
        ("Yes, that's right.", 1),
        ("Exactly right.", 2),
        ("Right, well done.", 3),
        ("Yes, that's right.", 4),
    ]
    assert bridge_for("answered_right", None, 5) == ("", 5)
    assert bridge_for("tell_me", 3, 2) == ("Sure, here it is.", 2)
    for label in ("go_on", "answered_wrong", "answered_partly", "side_question", "other"):
        assert bridge_for(label, 3, 0) == ("", 0)


def test_a_script_needed_only_for_the_next_scene_is_named() -> None:
    second = DIRECTED.scenes[1].id
    scripts = {
        scene_id: chunks for scene_id, chunks in DIRECTED_SCRIPTS.items() if scene_id != second
    }

    assert direct(DIRECTED, scripts, "go_on", Cursor(scene=1, step=4), None) == ([], Missing(2))
    assert direct(DIRECTED, scripts, "answered_right", Cursor(scene=1, step=2), 3) == (
        spelled(DIRECTED, DIRECTED_SCRIPTS, 1, "step 3, step 4"),
        Missing(2),
    )
