from tutor.chunker import split_clauses


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
