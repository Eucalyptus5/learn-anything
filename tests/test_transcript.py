import json

from tutor.brief import BRIEF_END, BRIEF_MARKER, VisualBrief
from tutor.prompt import Message
from tutor.session import OUTCOME_MARKER
from tutor.transcript import Transcript

HEAD = (
    BRIEF_MARKER
    + VisualBrief(kind="diagram", title="Clipped objective", show="ratio vs clip").model_dump_json()
    + BRIEF_END
)
TAIL_LINE = OUTCOME_MARKER + json.dumps({"signal": "covered", "settling": ""})


def test_history_is_empty_before_the_first_turn() -> None:
    assert Transcript(10).history(before="turn-1") == []


def test_learner_and_tutor_entries_alternate_as_user_and_assistant_messages() -> None:
    t = Transcript(10)
    t.learner("turn-1", "teach me ppo")
    t.tutor("turn-1", "PPO is a policy gradient method,")
    t.tutor("turn-1", "with a clipped objective.")
    t.learner("turn-2", "why clip")
    assert t.history(before="turn-2") == [
        Message(role="user", content="teach me ppo"),
        Message(
            role="assistant", content="PPO is a policy gradient method, with a clipped objective."
        ),
    ]


def test_history_before_a_turn_excludes_that_turn_and_later_ones() -> None:
    t = Transcript(10)
    t.learner("turn-1", "one")
    t.tutor("turn-1", "one.")
    t.learner("turn-2", "two")
    t.tutor("turn-2", "two.")
    t.learner("turn-3", "three")
    t.tutor("turn-3", "three.")
    assert [m.content for m in t.history(before="turn-2")] == ["one", "one."]


def test_history_keeps_only_the_last_n_turns() -> None:
    t = Transcript(2)
    for n in range(1, 5):
        t.learner(f"turn-{n}", f"q{n}")
        t.tutor(f"turn-{n}", f"a{n}.")
    assert [m.content for m in t.history(before="turn-5")] == ["q3", "a3.", "q4", "a4."]


def test_a_turn_with_no_tutor_clause_yields_only_the_user_message() -> None:
    t = Transcript(10)
    t.learner("turn-1", "hello")
    t.learner("turn-2", "again")
    assert t.history(before="turn-2") == [Message(role="user", content="hello")]


def test_zero_turns_yields_no_history() -> None:
    t = Transcript(0)
    t.learner("turn-1", "one")
    t.tutor("turn-1", "one.")
    assert t.history(before="turn-2") == []


def test_a_second_learner_text_for_the_same_turn_is_ignored() -> None:
    t = Transcript(10)
    t.learner("turn-1", "first")
    t.learner("turn-1", "second")
    assert t.history(before="turn-2") == [Message(role="user", content="first")]


def test_a_head_precedes_the_clauses_on_its_own_line() -> None:
    t = Transcript(10)
    t.learner("turn-1", "teach me ppo")
    t.head("turn-1", HEAD)
    t.tutor("turn-1", "PPO is a policy gradient method,")
    t.tutor("turn-1", "with a clipped objective.")
    assert t.history(before="turn-2") == [
        Message(role="user", content="teach me ppo"),
        Message(
            role="assistant",
            content=HEAD + "\nPPO is a policy gradient method, with a clipped objective.",
        ),
    ]


def test_a_head_with_no_clauses_still_yields_the_assistant_message() -> None:
    t = Transcript(10)
    t.learner("turn-1", "teach me ppo")
    t.head("turn-1", HEAD)
    assert t.history(before="turn-2") == [
        Message(role="user", content="teach me ppo"),
        Message(role="assistant", content=HEAD),
    ]


def test_a_head_for_a_turn_that_never_opens_is_never_emitted() -> None:
    t = Transcript(10)
    t.learner("turn-8", "eight")
    t.head("turn-9", HEAD)
    assert t.history(before="turn-10") == [Message(role="user", content="eight")]


def test_a_second_head_for_the_same_turn_replaces_the_first() -> None:
    t = Transcript(10)
    t.learner("turn-1", "teach me ppo")
    t.head("turn-1", BRIEF_MARKER + "{}" + BRIEF_END)
    t.head("turn-1", HEAD)
    t.tutor("turn-1", "PPO clips.")
    assert t.history(before="turn-2") == [
        Message(role="user", content="teach me ppo"),
        Message(role="assistant", content=HEAD + "\nPPO clips."),
    ]


def test_a_tail_follows_the_clauses_on_its_own_line() -> None:
    t = Transcript(10)
    t.learner("turn-1", "teach me ppo")
    t.tutor("turn-1", "PPO is a policy gradient method,")
    t.tutor("turn-1", "with a clipped objective.")
    t.tail("turn-1", TAIL_LINE)
    assert t.history(before="turn-2") == [
        Message(role="user", content="teach me ppo"),
        Message(
            role="assistant",
            content="PPO is a policy gradient method, with a clipped objective.\n" + TAIL_LINE,
        ),
    ]


def test_head_clauses_and_tail_stack_in_that_order() -> None:
    t = Transcript(10)
    t.learner("turn-1", "teach me ppo")
    t.tail("turn-1", TAIL_LINE)
    t.tutor("turn-1", "PPO clips.")
    t.head("turn-1", HEAD)
    assert t.history(before="turn-2") == [
        Message(role="user", content="teach me ppo"),
        Message(role="assistant", content=HEAD + "\nPPO clips.\n" + TAIL_LINE),
    ]


def test_a_tail_for_a_turn_that_never_opens_is_never_emitted() -> None:
    t = Transcript(10)
    t.learner("turn-8", "eight")
    t.tail("turn-9", TAIL_LINE)
    assert t.history(before="turn-10") == [Message(role="user", content="eight")]
