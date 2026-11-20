# Legal Sentence Game

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/ynrdq3wmdc/badge.svg)](https://msrit.ask.thili.ai/b/ynrdq3wmdc)
[![tests](https://github.com/AmoghRB/legal-sentence-game/actions/workflows/ci.yml/badge.svg)](https://github.com/AmoghRB/legal-sentence-game/actions/workflows/ci.yml)

A command-line game for law students. Each round shows the start of a real contract clause and asks you to
finish it as a given clause type (Governing Law, Severability, Notices...). An n-gram model trained on real
contracts suggests next words as you go. When you finish, a GPT-2 fine-tuned to tag clause types reads your
sentence and tells you what kind of clause you actually wrote.

```text
Round 2 of 3 · write an Assignment clause
  The Employee may not assign this Agreement or any rights ...
  n-gram hints for the next word: or · , · and
Your ending (? = n-gram suggestion, q = quit) > hereunder without the prior written consent of the Company.
  Your clause: The Employee may not assign this Agreement or any rights hereunder without the prior written consent of the Company.
  GPT-2 reads it as: Assignment (100%)  ✓ the clause you were asked for
  Next best: Notices 0% · Indemnification 0%
  +10 points
  n-gram would end it: ... or obligations hereunder shall be in writing and shall be deemed to be invalid, ...
  Real clause: The Employee may not assign this Agreement or any rights hereunder to any other party or parties without the prior written consent of the Company.
```

A miss shows what went wrong and how to fix it:

```text
  GPT-2 reads it as: Governing Law (72%)  ✗ you were asked for Notices (0%)
  Tip: Notices clauses usually say: notices, given, writing, notice, mail
```

## How it works

| Step | What happens | Code |
|---|---|---|
| 1. Corpus | 1,927 first sentences of real contract provisions from [LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue) (SEC filings), 8 clause types. Numbering and odd quotes are cleaned off, sentences of 10–45 words kept, and every 10th sentence per type is held back for game rounds. | `build_corpus`, `corpus/clauses.jsonl` |
| 2. N-gram | A from-scratch 4-gram model. Probabilities mix the 4-, 3-, 2- and 1-gram estimates (Jelinek-Mercer interpolation), so an unseen context backs off to a shorter one. It gives the next-word hints and the "n-gram would end it" line. | `NGramModel` |
| 3. GPT-2 | `GPT2ForSequenceClassification` fine-tuned on the training sentences (3 epochs, AdamW, lr 5e-5). | `train_classifier`, `load_classifier` |
| 4. Game | Shows 60% / 40% / 25% of a held-out clause (easy / medium / hard) with n-gram hints; you type the ending, `?` for the n-gram's whole ending, `q` to quit. | `pick_rounds`, `play` |
| 5. Feedback | GPT-2 classifies your full sentence. Points = 10 × its probability for the type you were asked for (halved if you used `?`). On a miss it lists the words that mark the right type. | `judge`, `feedback_lines` |

Clause types: Assignment, Confidentiality, Entire Agreement, Governing Law, Indemnification, Notices,
Severability, Termination.

## Results

Measured on the 192 held-out game sentences, which neither model saw in training. Each model is set next to
something simple, so the numbers mean something. Reproduce with `python -m app.cli evaluate`.

| Task | Method | Score |
|---|---|---|
| Clause type | Baseline: always guess one type | 13.0% |
| Clause type | Baseline: pick the type whose marker words appear most | 83.3% |
| Clause type | **Fine-tuned GPT-2** | **94.3%** |
| Next word in top 3 guesses | Baseline: the 3 most common words | 16.0% |
| Next word in top 3 guesses | 2-gram | 51.0% |
| Next word in top 3 guesses | 3-gram | 64.4% |
| Next word in top 3 guesses | **4-gram (used in the game)** | **67.5%** |

GPT-2 makes about a third as many mistakes as the keyword baseline (5.7% vs 16.7% wrong). It reached 96.0%
on its validation split after 3 epochs (94.2% after 1). The 4-gram also has the lowest held-out perplexity:
15.8, against 18.7 for the 3-gram and 36.5 for the 2-gram.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.cli train      # fine-tunes GPT-2 into models/clause-gpt2 (downloads gpt2 once, ~500 MB)
python -m app.cli play       # 5 rounds, medium
```

Fine-tuning takes about 2 minutes on an Apple M3 Pro GPU (MPS); it also runs on CUDA or CPU. Other commands:

```bash
python -m app.cli play --rounds 8 --difficulty hard --seed 1   # same seed, same clauses
python -m app.cli predict "This Agreement shall be governed by"  # n-gram next words
python -m app.cli classify "Any notice shall be in writing and delivered by hand."
python -m app.cli history                                       # past scores
python -m app.cli evaluate                                      # models vs baselines
python -m app.cli build-corpus                                  # rebuild the corpus from LEDGAR
pytest
```

No API key or server is needed: everything runs on your laptop.

## Tests

`pytest` runs 16 tests: at least one per milestone, one that checks each model beats its baseline, and the plumbing. The GPT-2 tests fine-tune a tiny, randomly initialised GPT-2
(same architecture and files as the real one), so they run offline in a few seconds. GitHub Actions runs them on
every push.

## Milestones

- [x] **1. Prepare a legal text corpus for n-gram model training.** `build_corpus`, `tokenize`, `corpus/clauses.jsonl`
- [x] **2. Implement the n-gram next-word predictor.** `NGramModel`, `predict` command
- [x] **3. Load the fine-tuned GPT-2 model for legal clause classification.** `train_classifier`, `load_classifier`
- [x] **4. Create a CLI game interface presenting partial legal sentences for completion.** `play` command
- [x] **5. After user completion, classify the full sentence and provide feedback.** `judge`, `feedback_lines`

Stretch ideas, also done: scoring from GPT-2's probability, three difficulty levels, and feedback that shows the
n-gram's own ending and the words that mark the clause type you missed.

## Layout

```text
app/core.py        all five milestones, one section each, then the baselines
app/storage.py     saves your scores to data/app.json
app/cli.py         commands: play, train, classify, predict, evaluate, history, build-corpus
corpus/            the prepared clause sentences (JSON lines: text, label, split)
models/            your fine-tuned GPT-2 (created by `train`, not in git)
tests/             test_core.py (milestones, baselines), test_smoke.py (store and commands)
```

## Data

Sentences come from LEDGAR (Tuggener et al., LREC 2020) as packaged in LexGLUE (Chalkidis et al., ACL 2022),
released under CC BY 4.0. They are contract provisions from public SEC EDGAR filings.
