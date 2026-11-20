"""Tests for the five milestones in README.md, one section each."""
import json

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
def test_milestone_2_ngram_counts_backoff_and_generation():
    sentences = [core.tokenize(s) for s in ["The party shall pay the fee.", "The party shall give notice.",
                                            "The company shall pay the fee."]]
    model = core.NGramModel(3).fit(sentences)

    context = ["the", "party"]
    assert abs(sum(model.prob(w, context) for w in model.vocab | {"<unknown>"}) - 1) < 1e-9
    assert model.predict(context, 1)[0][0] == "shall"
    assert {w for w, _ in model.predict(["party", "shall"], 2)} == {"pay", "give"}
    assert model.predict(["never", "shall"], 1)[0][0] == "pay"          # unseen trigram backs off to "shall pay"
    assert model.generate(["the", "company"]) == ["shall", "pay", "the", "fee", "."]
    assert model.perplexity(sentences) < model.perplexity([core.tokenize("fee the pay shall party the.")])


def test_milestone_2_ngram_predicts_legal_language():
    model = core.train_ngram()
    assert model.predict(core.tokenize("This Agreement constitutes the entire"), 1)[0][0] == "agreement"
    assert model.predict(core.tokenize("construed in accordance with the"), 1)[0][0] == "laws"
    ending = core.detokenize(model.generate(core.tokenize("If any provision of this Agreement is held")))
    assert ending.startswith("to be invalid")
    held_out = core.corpus_tokens(core.load_corpus(split="play"))
    assert model.perplexity(held_out) < core.train_ngram(n=2).perplexity(held_out)


# Milestone 3: Load the fine-tuned GPT-2 model for legal clause classification.
def test_milestone_3_fine_tuned_gpt2_loads_and_classifies(tiny_model_dir):
    classifier = core.load_classifier(tiny_model_dir, device="cpu")
    assert type(classifier.model).__name__ == "GPT2ForSequenceClassification"
    assert classifier.labels == sorted(core.CLAUSE_TYPES.values())

    probs = classifier.predict_proba("This Agreement shall be governed by the laws of the State of New York.")
    assert set(probs) == set(classifier.labels)
    assert abs(sum(probs.values()) - 1) < 1e-5
    assert list(probs.values()) == sorted(probs.values(), reverse=True)
    label, prob = classifier.classify("Any notice under this Agreement shall be in writing.")
    assert label in classifier.labels and 0 < prob <= 1

    metrics = json.loads((tiny_model_dir / "metrics.json").read_text())
    assert metrics["labels"] == classifier.labels and len(metrics["losses"]) == 1


def test_milestone_3_missing_model_says_how_to_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="app.cli train"):
        core.load_classifier(tmp_path / "no-model-here")


# Milestone 4: Create a CLI game interface presenting partial legal sentences for completion.
LAW = {"text": "This Agreement shall be governed by the laws of the State of New York without regard to its "
                "conflicts of law rules.", "label": "Governing Law", "split": "play"}
SEVERABILITY = {"text": "If any provision of this Agreement is held invalid, the remaining provisions shall continue "
                        "in full force and effect.", "label": "Severability", "split": "play"}


def test_milestone_4_rounds_show_the_start_of_a_held_out_clause():
    easy, medium, hard = (core.make_round(LAW, level) for level in ("easy", "medium", "hard"))
    assert LAW["text"].startswith(hard.prefix) and hard.target == "Governing Law" and hard.original == LAW["text"]
    assert 3 <= len(hard.prefix.split()) < len(medium.prefix.split()) < len(easy.prefix.split())
    assert len(easy.prefix.split()) <= len(LAW["text"].split()) - 3                  # always something left to write

    play_rows = core.load_corpus(split="play")
    rounds = core.pick_rounds(play_rows, n=8, seed=1)
    assert len({r.target for r in rounds}) == 8                                       # one of each clause type
    assert core.pick_rounds(play_rows, n=8, seed=1) == rounds                         # same seed, same game
    assert {r.original for r in rounds} <= {row["text"] for row in play_rows}         # never a training sentence

    assert core.join_completion("is held", "  to be invalid. ") == "is held to be invalid."
    assert core.join_completion("of New York", ", without regard") == "of New York, without regard"


def test_milestone_4_play_loop_shows_hints_and_reads_endings():
    ngram = core.train_ngram()
    rounds = [core.make_round(SEVERABILITY, "medium"), core.make_round(LAW, "medium")]
    answers = iter(["?", "x", "to be invalid, the rest of this Agreement stays in force.", "q"])
    said = []
    results = core.play(rounds, ngram, ask=lambda prompt: next(answers), say=said.append)
    screen = "\n".join(said)

    assert "Round 1 of 2" in screen and "write a Severability clause" in screen
    assert rounds[0].prefix in screen
    assert "n-gram hints for the next word" in screen and "n-gram suggests" in screen
    assert "at least two words" in screen                                              # "x" was rejected
    assert len(results) == 1                                                           # quit in round 2
    assert results[0]["hint_used"] is True
    assert results[0]["sentence"] == rounds[0].prefix + " to be invalid, the rest of this Agreement stays in force."


# Milestone 5: After user completion, classify the full sentence and provide feedback.
class FixedClassifier:
    """Stands in for GPT-2 with fixed probabilities."""

    def __init__(self, probs):
        self.probs = probs

    def predict_proba(self, text):
        return dict(sorted(self.probs.items(), key=lambda pair: -pair[1]))


def test_milestone_5_points_and_feedback():
    gpt2 = FixedClassifier({"Severability": 0.8, "Termination": 0.15, "Notices": 0.05})
    hit = core.judge(gpt2, "If any provision is held invalid, the rest survives.", "Severability")
    assert hit.matched and hit.predicted == "Severability" and hit.points == 8
    assert core.judge(gpt2, "If any provision is held invalid.", "Severability", hint_used=True).points == 4

    miss = core.judge(gpt2, "If any provision is held invalid.", "Termination")
    assert not miss.matched and miss.points == 2
    screen = "\n".join(core.feedback_lines(miss, {"Termination": ["terminate", "terminated"]}))
    assert "GPT-2 reads it as: Severability (80%)" in screen
    assert "you were asked for Termination (15%)" in screen
    assert "terminate, terminated" in screen and "+2 points" in screen


def test_milestone_5_keywords_explain_each_clause_type():
    keywords = core.clause_keywords(core.load_corpus(split="train"))
    assert set(keywords) == set(core.CLAUSE_TYPES.values())
    assert {"governed", "laws"} <= set(keywords["Governing Law"])
    assert {"invalid", "unenforceable"} <= set(keywords["Severability"])
    assert "confidential" in keywords["Confidentiality"] and "indemnify" in keywords["Indemnification"]


def test_milestone_5_full_game_with_the_fine_tuned_gpt2(tiny_model_dir, monkeypatch, tmp_path, capsys):
    from app.cli import main
    from app.storage import Store

    monkeypatch.setenv("DATA_PATH", str(tmp_path / "app.json"))
    answers = iter(["shall remain in full force and effect.", "?", "by written notice to the other party."])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    main(["play", "--rounds", "2", "--seed", "3", "--model", str(tiny_model_dir)])
    screen = capsys.readouterr().out
    assert screen.count("GPT-2 reads it as:") == 2
    assert "n-gram suggests" in screen and "Score:" in screen

    game = Store().get("games")[-1]
    assert game["rounds"] == 2 and game["max"] == 20 and 0 <= game["score"] <= 20
    main(["history"])
    assert "Best:" in capsys.readouterr().out
