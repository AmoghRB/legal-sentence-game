"""Tests for the five milestones in README.md, one section each."""
import pytest

from app import core


# Milestone 1: Prepare a legal text corpus for n-gram model training.
def test_milestone_1_cleaning_splitting_and_tokens():
    assert core.clean_text("9.2  This  Agreement “terms”") == 'This Agreement "terms"'
    assert core.clean_text("Section 4. (a) Each party shall") == "Each party shall"
    assert core.clean_text("30 days after notice") == "30 days after notice"  # a number is not numbering

    text = "Acme Inc. Seller shall pay. The U.S. Buyer shall deliver. Done."
    assert core.split_sentences(text) == ["Acme Inc. Seller shall pay.", "The U.S. Buyer shall deliver.", "Done."]

    tokens = core.tokenize("Section 9.2, the Company’s $1,000 fee (net).")
    assert tokens == ["section", "9.2", ",", "the", "company's", "$1,000", "fee", "(", "net", ")", "."]
    assert core.detokenize(tokens) == "section 9.2, the company's $1,000 fee (net)."


def test_milestone_1_build_corpus_keeps_one_clean_sentence_per_provision():
    law = "1.1 This Agreement shall be governed by the laws of the State of New York without regard to conflicts. More."
    records = [
        (law, "Governing Laws"),
        (law, "Governing Laws"),                                           # duplicate
        ("THIS AGREEMENT SHALL BE GOVERNED BY THE LAWS OF THE STATE OF NEW YORK IN ALL RESPECTS.", "Governing Laws"),
        ("Notices shall be in writing.", "Notices"),                       # too short
        ("The Executive shall receive a base salary of one hundred dollars per year for services.", "Base Salary"),
    ] + [(f"All notices under this Agreement shall be in writing and delivered to party number {i} by hand.",
          "Notices") for i in range(4)]
    rows = core.build_corpus(records, play_every=2)

    assert rows[0] == {"text": law.removeprefix("1.1 ").removesuffix(" More."), "label": "Governing Law",
                       "split": "train"}
    assert [row["label"] for row in rows] == ["Governing Law"] + ["Notices"] * 4
    assert [row["split"] for row in rows[1:]] == ["train", "play", "train", "play"]


def test_milestone_1_committed_corpus_covers_every_clause_type():
    rows = core.load_corpus()
    labels = {row["label"] for row in rows}
    assert labels == set(core.CLAUSE_TYPES.values())
    for label in labels:
        assert sum(row["label"] == label for row in rows) >= 100
    train, play = core.load_corpus(split="train"), core.load_corpus(split="play")
    assert len(train) + len(play) == len(rows) and len(play) >= 8 * 10
    assert not {r["text"] for r in train} & {r["text"] for r in play}
    assert all(tokens for tokens in core.corpus_tokens(train))


# Milestone 2: Implement the n-gram next-word predictor.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Load the fine-tuned GPT-2 model for legal clause classification.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Create a CLI game interface presenting partial legal sentences for completion.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: After user completion, classify the full sentence and provide feedback.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")
