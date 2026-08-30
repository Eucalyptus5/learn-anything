import asyncio
from collections import Counter
from collections.abc import AsyncIterator

import pytest

from tutor.chunker import Scrubber, clause_chunks, split_clauses, spoken_text


def test_splits_on_period_followed_by_whitespace() -> None:
    clauses, remainder = split_clauses("This is a full sentence. ", min_words=3, max_words=50)
    assert clauses == ["This is a full sentence."]
    assert remainder == ""


def test_splits_on_question_mark_followed_by_whitespace() -> None:
    clauses, remainder = split_clauses("Is this the right function? ", min_words=3, max_words=50)
    assert clauses == ["Is this the right function?"]
    assert remainder == ""


def test_splits_on_exclamation_mark_followed_by_whitespace() -> None:
    clauses, remainder = split_clauses("Watch out for that bug! ", min_words=3, max_words=50)
    assert clauses == ["Watch out for that bug!"]
    assert remainder == ""


def test_splits_on_semicolon_followed_by_whitespace() -> None:
    clauses, remainder = split_clauses(
        "First we parse the tokens; then we emit clauses ", min_words=3, max_words=50
    )
    assert clauses == ["First we parse the tokens;"]
    assert remainder == "then we emit clauses "


def test_splits_on_colon_followed_by_whitespace() -> None:
    clauses, remainder = split_clauses(
        "Here is the plan: read the file first ", min_words=3, max_words=50
    )
    assert clauses == ["Here is the plan:"]
    assert remainder == "read the file first "


def test_splits_on_comma_followed_by_whitespace() -> None:
    clauses, remainder = split_clauses(
        "After the buffer fills, we flush it downstream ", min_words=3, max_words=50
    )
    assert clauses == ["After the buffer fills,"]
    assert remainder == "we flush it downstream "


def test_does_not_split_inside_eg_abbreviation() -> None:
    text = "Some languages e.g. Python are dynamically typed. "
    clauses, remainder = split_clauses(text, min_words=3, max_words=50)
    assert clauses == ["Some languages e.g. Python are dynamically typed."]
    assert remainder == ""


def test_does_not_split_inside_ie_abbreviation() -> None:
    text = "Use the fast path i.e. skip the cache entirely. "
    clauses, remainder = split_clauses(text, min_words=3, max_words=50)
    assert clauses == ["Use the fast path i.e. skip the cache entirely."]
    assert remainder == ""


def test_does_not_split_inside_etc_abbreviation() -> None:
    text = "It handles files sockets pipes etc. without issue. "
    clauses, remainder = split_clauses(text, min_words=3, max_words=50)
    assert clauses == ["It handles files sockets pipes etc. without issue."]
    assert remainder == ""


def test_does_not_split_inside_vs_abbreviation() -> None:
    text = "This is a tradeoff of latency vs. throughput here. "
    clauses, remainder = split_clauses(text, min_words=3, max_words=50)
    assert clauses == ["This is a tradeoff of latency vs. throughput here."]
    assert remainder == ""


def test_does_not_split_inside_dr_abbreviation() -> None:
    text = "Dr. Smith wrote the original paper on this. "
    clauses, remainder = split_clauses(text, min_words=1, max_words=50)
    assert clauses == ["Dr. Smith wrote the original paper on this."]
    assert remainder == ""


def test_does_not_split_inside_decimal_number() -> None:
    text = "The constant is 3.14 and it never changes. "
    clauses, remainder = split_clauses(text, min_words=3, max_words=50)
    assert clauses == ["The constant is 3.14 and it never changes."]
    assert remainder == ""


def test_does_not_split_inside_version_number() -> None:
    text = "We pinned the dependency to 1.15.0 for now. "
    clauses, remainder = split_clauses(text, min_words=3, max_words=50)
    assert clauses == ["We pinned the dependency to 1.15.0 for now."]
    assert remainder == ""


def test_does_not_split_inside_module_path() -> None:
    text = "The fix lives in connection_pool.py near the top. "
    clauses, remainder = split_clauses(text, min_words=3, max_words=50)
    assert clauses == ["The fix lives in connection_pool.py near the top."]
    assert remainder == ""


def test_does_not_split_inside_repo_relative_path() -> None:
    text = "Open tutor/tts.py and look at the queue. "
    clauses, remainder = split_clauses(text, min_words=3, max_words=50)
    assert clauses == ["Open tutor/tts.py and look at the queue."]
    assert remainder == ""


def test_does_not_split_inside_attribute_access() -> None:
    text = "The call to self.acquire blocks until it succeeds. "
    clauses, remainder = split_clauses(text, min_words=3, max_words=50)
    assert clauses == ["The call to self.acquire blocks until it succeeds."]
    assert remainder == ""


def test_returns_tail_as_remainder_when_no_boundary_present() -> None:
    clauses, remainder = split_clauses("still waiting on more tokens", min_words=3, max_words=50)
    assert clauses == []
    assert remainder == "still waiting on more tokens"


def test_under_min_words_whole_text_keeps_everything_in_remainder() -> None:
    clauses, remainder = split_clauses("Yes, ", min_words=3, max_words=50)
    assert clauses == []
    assert remainder == "Yes, "


def test_short_fragment_merges_forward_into_next_clause() -> None:
    text = "Yes, that is exactly right. "
    clauses, remainder = split_clauses(text, min_words=3, max_words=50)
    assert clauses == ["Yes, that is exactly right."]
    assert remainder == ""


def test_forced_split_at_max_words_keeps_chunk_within_limit() -> None:
    words = [f"word{i}" for i in range(12)]
    text = " ".join(words) + " more tokens after the break"
    clauses, remainder = split_clauses(text, min_words=3, max_words=10)

    assert len(clauses) == 1
    forced = clauses[0]
    assert len(forced.split()) <= 10
    assert forced == " ".join(words[:10])
    assert remainder == " ".join(words[10:]) + " more tokens after the break"


def test_forced_split_remainder_can_hold_a_partial_word() -> None:
    text = "one two three four five six seven eight nine te"
    clauses, remainder = split_clauses(text, min_words=3, max_words=9)

    assert clauses == ["one two three four five six seven eight nine"]
    assert remainder == "te"


def test_trailing_punctuation_with_no_whitespace_stays_in_remainder() -> None:
    clauses, remainder = split_clauses("The value is 3.", min_words=3, max_words=50)
    assert clauses == []
    assert remainder == "The value is 3."


def test_return_shape_is_list_and_string_tuple() -> None:
    result = split_clauses("Hello there friend. ", min_words=1, max_words=50)
    assert isinstance(result, tuple)
    assert len(result) == 2
    clauses, remainder = result
    assert isinstance(clauses, list)
    assert isinstance(remainder, str)


async def _stream(pieces: list[str]) -> AsyncIterator[str]:
    for piece in pieces:
        yield piece


async def test_mid_word_token_cuts_still_yield_whole_clauses() -> None:
    pieces = [
        "The sys",
        "tem logs each reque",
        "st,",
        " then it wri",
        "tes the response to di",
        "sk. ",
    ]
    clauses = [
        clause
        async for clause in clause_chunks(
            _stream(pieces), first_min_words=3, min_words=3, max_words=50
        )
    ]
    assert clauses == ["The system logs each request,", "then it writes the response to disk."]


async def test_first_chunk_uses_first_min_words_then_switches_to_min_words() -> None:
    text = "Yes, that is right, and then also, we look at the second clause, which is longer. "
    tokens = [f"{word} " for word in text.split()]

    clauses = [
        clause
        async for clause in clause_chunks(
            _stream(tokens), first_min_words=3, min_words=8, max_words=50
        )
    ]

    assert clauses == [
        "Yes, that is right,",
        "and then also, we look at the second clause,",
        "which is longer.",
    ]


async def test_trailing_remainder_is_flushed_stripped_at_source_exhaustion() -> None:
    tokens = ["This ", "finishes ", "quickly. ", "Wrap up"]

    clauses = [
        clause
        async for clause in clause_chunks(
            _stream(tokens), first_min_words=3, min_words=8, max_words=50
        )
    ]

    assert clauses == ["This finishes quickly.", "Wrap up"]


async def test_empty_source_yields_nothing() -> None:
    clauses = [clause async for clause in clause_chunks(_stream([]))]
    assert clauses == []


async def test_cancelling_the_consumer_propagates_and_does_not_flush_the_buffer() -> None:
    entered = asyncio.Event()
    blocked = asyncio.Event()

    async def source() -> AsyncIterator[str]:
        entered.set()
        yield "partial fragment without a boundary "
        await blocked.wait()
        yield "unreachable "

    received: list[str] = []

    async def consume() -> None:
        async for clause in clause_chunks(source()):
            received.append(clause)

    task = asyncio.create_task(consume())
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert received == []
    assert task.cancelled()


async def test_one_token_with_several_boundaries_only_shortens_the_first_chunk() -> None:
    text = "Yes, that is right, and then also, we look at the second clause, which is longer. "

    clauses = [
        clause
        async for clause in clause_chunks(
            _stream([text]), first_min_words=3, min_words=8, max_words=50
        )
    ]

    assert clauses == [
        "Yes, that is right,",
        "and then also, we look at the second clause,",
        "which is longer.",
    ]


async def test_boundary_free_run_is_forced_into_max_words_chunks() -> None:
    tokens = [f"w{i} " for i in range(30)]

    clauses = [
        clause
        async for clause in clause_chunks(
            _stream(tokens), first_min_words=3, min_words=8, max_words=12
        )
    ]

    assert clauses == [
        " ".join(f"w{i}" for i in range(12)),
        " ".join(f"w{i}" for i in range(12, 24)),
        " ".join(f"w{i}" for i in range(24, 30)),
    ]
    assert all(len(clause.split()) <= 12 for clause in clauses)


def _scrub(deltas: list[str]) -> tuple[str, Scrubber]:
    scrubber = Scrubber()
    text = "".join(scrubber.feed(delta) for delta in deltas) + scrubber.flush()
    return text, scrubber


def test_a_fenced_block_is_dropped_whole() -> None:
    text, scrubber = _scrub(
        ["lifecycle.\n\n``", '`json\n{"type":', '"diagram"}\n``', "`\n\nPhase: Teach."]
    )
    assert text == "lifecycle.\n\n \n\nPhase: Teach."
    assert scrubber.dropped == Counter({"fence": 1})


def test_a_json_payload_is_dropped_to_its_closing_brace() -> None:
    text, scrubber = _scrub(['attempts.{"diag', 'ram":{"a":1}}We', "'re in"])
    assert text == "attempts. We're in"
    assert scrubber.dropped == Counter({"json": 1})


def test_an_unbalanced_payload_gives_the_turn_back() -> None:
    text, scrubber = _scrub(['{"' + "x" * 9000, " and so on."])
    assert text.endswith(" and so on.")
    assert scrubber.dropped == Counter({"json": 1, "json_unbalanced": 1})


def test_inline_markers_are_removed_from_prose() -> None:
    text, scrubber = _scrub(["the `acq", "uire` method is **free", "** now"])
    assert text == "the acquire method is free now"
    assert scrubber.dropped == Counter({"backtick": 2, "bold": 2})


def test_a_tag_shaped_token_is_dropped() -> None:
    text, scrubber = _scrub(["path.\n<", "/turn>"])
    assert text == "path.\n"
    assert scrubber.dropped == Counter({"tag": 1})

    text, scrubber = _scrub(["a < b and", " c > d"])
    assert text == "a < b and c > d"
    assert scrubber.dropped == Counter()


def test_a_lone_trailing_backtick_is_held_then_released() -> None:
    scrubber = Scrubber()
    fed = scrubber.feed("ends here `")
    assert fed == "ends here "
    assert scrubber.flush() == ""
    assert scrubber.dropped == Counter({"backtick": 1})


def test_the_first_delta_can_open_a_fence() -> None:
    text, scrubber = _scrub(["```", "json\n{}\n```", "Speech."])
    assert text == " Speech."
    assert scrubber.dropped == Counter({"fence": 1})


def test_a_payload_with_a_space_after_the_brace_is_dropped() -> None:
    text, scrubber = _scrub(['forever.\n\n{ "d', 'iagram": 1}Next.'])
    assert text == "forever.\n\n Next."
    assert scrubber.dropped == Counter({"json": 1})


def test_dropped_chars_counts_what_each_class_removed() -> None:
    text, scrubber = _scrub(['a ```x``` b {"k":1} c </turn> `d` **e**'])
    assert text == "a   b   c  d e"
    assert scrubber.dropped == Counter({"fence": 1, "json": 1, "tag": 1, "backtick": 2, "bold": 2})
    assert scrubber.dropped_chars == Counter(
        {"fence": 7, "json": 7, "tag": 7, "backtick": 2, "bold": 4}
    )


async def test_spoken_text_feeds_the_chunker_clean_clauses() -> None:
    pieces = [
        "The pool is a free list. ",
        '```json\n{"a":1}\n```',
        " The acquire path takes a slot from it.",
    ]
    clauses = [clause async for clause in clause_chunks(spoken_text(_stream(pieces), Scrubber()))]
    assert clauses == ["The pool is a free list.", "The acquire path takes a slot from it."]
