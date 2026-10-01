import hashlib

import pytest

from tests.test_session import LESSON, SCRIPTS
from tutor.lesson import OPENING_TEXT, Cursor
from tutor.prompt import Message
from tutor.reply import (
    LIVE_PROMPT,
    STRIP,
    Missing,
    Piece,
    Question,
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


def chunk(scene_id: str, step: int, question: bool) -> str:
    return next(
        each.text for each in SCRIPTS[scene_id] if each.step == step and each.question == question
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
