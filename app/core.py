"""Legal Sentence Game: one section per milestone, built in order."""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "corpus" / "clauses.jsonl"

# The clause types the game uses: LEDGAR label -> name shown to the player.
CLAUSE_TYPES = {
    "Governing Laws": "Governing Law",
    "Confidentiality": "Confidentiality",
    "Terminations": "Termination",
    "Indemnifications": "Indemnification",
    "Notices": "Notices",
    "Severability": "Severability",
    "Assignments": "Assignment",
    "Entire Agreements": "Entire Agreement",
}


# --- Milestone 1: Prepare a legal text corpus for n-gram model training.
# Real contract provisions from LEDGAR (LexGLUE), cut down to one clean sentence each.

_ABBREVIATIONS = {"inc", "corp", "co", "ltd", "no", "nos", "sec", "secs", "u.s", "e.g", "i.e", "etc", "l.l.c",
                  "llc", "l.p", "n.a", "mr", "ms", "mrs", "dr", "st", "jr", "v", "vs", "art", "para", "cal", "del"}
_ENUMERATION = re.compile(r"^(?:(?:section|article|clause)\s+)?(?:\(?[0-9]+[a-z]?\)|\([a-z]{1,4}\)|(?:[0-9]+\.)+[0-9]*)\s+",
                          re.IGNORECASE)
_TOKEN = re.compile(r"\$?[0-9]+(?:[.,][0-9]+)*%?|[a-z]+(?:[.'-][a-z]+)*|[^\w\s]")
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-",
                         " ": " "})


def clean_text(text: str) -> str:
    """Straighten quotes, squash whitespace and drop leading numbering such as '9.2', '(a)' or 'Section 4.'."""
    text = re.sub(r"\s+", " ", text.translate(_QUOTES)).strip()
    previous = None
    while previous != text:
        previous = text
        text = _ENUMERATION.sub("", text, count=1).strip()
    return text


def _ends_with_abbreviation(sentence: str) -> bool:
    last = sentence.rsplit(" ", 1)[-1].lower().rstrip(".")
    return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? followed by a capital letter, without breaking after 'Inc.', 'U.S.' and similar."""
    sentences: list[str] = []
    for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"])", text):
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] += " " + piece
        else:
            sentences.append(piece)
    return [s.strip() for s in sentences if s.strip()]


def tokenize(text: str) -> list[str]:
    """Lower-case word and punctuation tokens: 'Section 9.2, the Company's' -> ['section', '9.2', ',', ...]."""
    return _TOKEN.findall(text.translate(_QUOTES).lower())


def detokenize(tokens: Iterable[str]) -> str:
    """Join tokens back into readable text (no space before , . ; : ) and none after ( $)."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?)\]%])", r"\1", text)
    return re.sub(r"([(\[$])\s+", r"\1", text)


def is_good_sentence(sentence: str, min_words: int = 10, max_words: int = 45) -> bool:
    """A usable game sentence: the right length, mostly lower case (not a SHOUTED disclaimer), ends like a sentence."""
    words = sentence.split()
    letters = [c for c in sentence if c.isalpha()]
    if not (min_words <= len(words) <= max_words) or not letters:
        return False
    if sum(c.isupper() for c in letters) / len(letters) > 0.25:
        return False
    return sentence.endswith((".", ";")) and sentence[0].isupper()


def build_corpus(records: Iterable[tuple[str, str]], per_label: int = 250, play_every: int = 10,
                 min_words: int = 10, max_words: int = 45) -> list[dict]:
    """Turn (provision text, LEDGAR label) pairs into game rows {text, label, split}.

    Keeps the first sentence of each provision whose label is in CLAUSE_TYPES, drops duplicates, caps each label at
    `per_label`, and holds back every `play_every`-th sentence of a label as split="play" (used for game rounds,
    never for training), the rest as split="train".
    """
    rows: list[dict] = []
    seen: set[str] = set()
    kept = {label: 0 for label in CLAUSE_TYPES.values()}
    for text, ledgar_label in records:
        label = CLAUSE_TYPES.get(ledgar_label)
        if label is None or kept[label] >= per_label:
            continue
        sentences = split_sentences(clean_text(text))
        if not sentences or not is_good_sentence(sentences[0], min_words, max_words):
            continue
        sentence = sentences[0]
        key = " ".join(tokenize(sentence))
        if key in seen:
            continue
        seen.add(key)
        kept[label] += 1
        split = "play" if kept[label] % play_every == 0 else "train"
        rows.append({"text": sentence, "label": label, "split": split})
    return rows


def download_ledgar(split: str = "train+validation+test") -> Iterable[tuple[str, str]]:
    """Stream (text, label name) pairs from LEDGAR on the Hugging Face Hub (needs internet once)."""
    from datasets import load_dataset

    data = load_dataset("coastalcph/lex_glue", "ledgar", split=split)
    names = data.features["label"].names
    return ((row["text"], names[row["label"]]) for row in data)


def save_corpus(rows: list[dict], path: str | Path = CORPUS_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return path


def load_corpus(path: str | Path = CORPUS_PATH, split: str | None = None) -> list[dict]:
    """Read the corpus rows, optionally only one split ('train' or 'play')."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [row for row in rows if split is None or row["split"] == split]


def corpus_tokens(rows: Iterable[dict]) -> list[list[str]]:
    """The n-gram training input: one token list per sentence."""
    return [tokenize(row["text"]) for row in rows]


# --- Milestone 2: Implement the n-gram next-word predictor.
# Counts of every 1..n-gram, mixed with fixed weights (Jelinek-Mercer interpolation) so unseen trigrams back off.

BOS, EOS = "<s>", "</s>"


class NGramModel:
    """An n-gram language model over tokens. P(w | u, v) = l3*P(w|u,v) + l2*P(w|v) + l1*P(w)."""

    def __init__(self, n: int = 3, weights: tuple[float, ...] | None = None):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        # weights[k] is the weight of the order with a context of k words; longer contexts weigh more
        self.weights = weights or tuple(2 ** k for k in range(n))
        self.counts: list[dict[tuple, Counter]] = [defaultdict(Counter) for _ in range(n)]
        self.totals: list[dict[tuple, int]] = [{} for _ in range(n)]

    def fit(self, sentences: Iterable[list[str]]) -> "NGramModel":
        for tokens in sentences:
            padded = [BOS] * (self.n - 1) + list(tokens) + [EOS]
            for i in range(self.n - 1, len(padded)):
                for k in range(self.n):
                    self.counts[k][tuple(padded[i - k:i])][padded[i]] += 1
        self.totals = [{ctx: sum(c.values()) for ctx, c in order.items()} for order in self.counts]
        return self

    @property
    def vocab(self) -> set[str]:
        return set(self.counts[0][()])

    def _context(self, context: list[str]) -> tuple:
        padded = [BOS] * (self.n - 1) + list(context)
        return tuple(padded[len(padded) - (self.n - 1):])

    def prob(self, word: str, context: list[str]) -> float:
        """Interpolated probability of `word` after `context`. Orders whose context was never seen are skipped."""
        ctx = self._context(context)
        vocab_size = len(self.counts[0][()]) + 1                  # +1 leaves room for unknown words
        total = self.totals[0].get((), 0)
        p = self.weights[0] * (self.counts[0][()][word] + 1) / (total + vocab_size)
        used = self.weights[0]
        for k in range(1, self.n):
            sub = ctx[len(ctx) - k:]
            seen = self.totals[k].get(sub)
            if seen:
                p += self.weights[k] * self.counts[k][sub][word] / seen
                used += self.weights[k]
        return p / used

    def predict(self, context: list[str], k: int = 3) -> list[tuple[str, float]]:
        """The k most likely next tokens after `context`, with their probabilities."""
        ctx = self._context(context)
        candidates: set[str] = set()
        for order in range(self.n - 1, 0, -1):
            candidates |= set(self.counts[order].get(ctx[len(ctx) - order:], ()))
        if len(candidates) < k:
            candidates |= {w for w, _ in self.counts[0][()].most_common(k + 1)}
        scored = [(w, self.prob(w, context)) for w in candidates if w != BOS]
        return sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:k]

    def generate(self, context: list[str], max_words: int = 30) -> list[str]:
        """Greedy continuation of `context` until the end-of-sentence token, never repeating an n-gram."""
        tokens, new = list(context), []
        seen = {tuple(tokens[i:i + self.n]) for i in range(len(tokens) - self.n + 1)}
        for _ in range(max_words):
            for word, _ in self.predict(tokens, k=10):
                gram = tuple((tokens + [word])[-self.n:])
                if gram not in seen:
                    break
            else:
                break
            if word == EOS:
                break
            seen.add(gram)
            tokens.append(word)
            new.append(word)
        return new

    def perplexity(self, sentences: Iterable[list[str]]) -> float:
        """How surprised the model is by `sentences` (lower is better)."""
        log_sum, count = 0.0, 0
        for tokens in sentences:
            for i, word in enumerate(list(tokens) + [EOS]):
                log_sum += math.log(self.prob(word, tokens[:i]))
                count += 1
        return math.exp(-log_sum / max(count, 1))


def train_ngram(rows: Iterable[dict] | None = None, n: int = 4) -> NGramModel:
    """Fit an n-gram model on the corpus training sentences (game sentences are held out).

    n=4 by default: on the held-out sentences it scores perplexity 15.8 against 18.7 for n=3 and 36.5 for n=2.
    """
    rows = load_corpus(split="train") if rows is None else rows
    return NGramModel(n).fit(corpus_tokens(rows))


# --- Milestone 3: Load the fine-tuned GPT-2 model for legal clause classification.
# GPT-2 with a classification head (GPT2ForSequenceClassification), fine-tuned on the corpus' training sentences.

MODEL_DIR = ROOT / "models" / "clause-gpt2"


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClauseClassifier:
    """A fine-tuned GPT-2 that reads a sentence and says which clause type it is."""

    def __init__(self, model, tokenizer, device: str = "cpu", max_length: int = 96):
        self.model, self.tokenizer, self.device, self.max_length = model, tokenizer, device, max_length

    @property
    def labels(self) -> list[str]:
        return [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def predict_proba(self, text: str) -> dict[str, float]:
        """Probability of every clause type, most likely first."""
        import torch

        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        with torch.no_grad():
            logits = self.model(**encoded.to(self.device)).logits[0]
        probs = logits.float().softmax(-1).tolist()
        return dict(sorted(zip(self.labels, probs), key=lambda pair: -pair[1]))

    def classify(self, text: str) -> tuple[str, float]:
        """The most likely clause type and its probability."""
        return next(iter(self.predict_proba(text).items()))

    def accuracy(self, rows: Iterable[dict]) -> float:
        rows = list(rows)
        return sum(self.classify(row["text"])[0] == row["label"] for row in rows) / max(len(rows), 1)


def _quiet_transformers() -> None:
    """Hide the progress bars and load reports transformers prints, so the game screen stays clean."""
    from transformers.utils import logging

    logging.set_verbosity_error()
    logging.disable_progress_bar()


def load_classifier(model_dir: str | Path = MODEL_DIR, device: str | None = None) -> ClauseClassifier:
    """Load the fine-tuned GPT-2 clause classifier saved by `train_classifier`."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _quiet_transformers()
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"No fine-tuned model in {model_dir}. Run `python -m app.cli train` first.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.config.pad_token_id = tokenizer.pad_token_id
    device = device or pick_device()
    model.to(device).eval()
    return ClauseClassifier(model, tokenizer, device)


def train_classifier(rows: Iterable[dict] | None = None, base_model: str = "gpt2", out_dir: str | Path = MODEL_DIR,
                     epochs: int = 3, batch_size: int = 16, lr: float = 5e-5, max_length: int = 96,
                     val_fraction: float = 0.1, seed: int = 0, device: str | None = None, log=print) -> dict:
    """Fine-tune GPT-2 to tag clause types, save it to `out_dir`, and return the training metrics."""
    import random

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = load_corpus(split="train") if rows is None else list(rows)
    labels = sorted({row["label"] for row in rows})
    label2id = {label: i for i, label in enumerate(labels)}
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction))
    val, train = rows[:n_val], rows[n_val:]

    _quiet_transformers()
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token          # GPT-2 has no pad token; reuse end-of-text
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id)
    model.config.pad_token_id = tokenizer.pad_token_id
    device = device or pick_device()
    model.to(device)

    steps = epochs * math.ceil(len(train) / batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1 - step / steps))
    classifier = ClauseClassifier(model, tokenizer, device, max_length)
    rng = random.Random(seed)
    losses, val_accuracy = [], 0.0
    for epoch in range(epochs):
        model.train()
        rng.shuffle(train)
        epoch_loss = 0.0
        for start in range(0, len(train), batch_size):
            batch = train[start:start + batch_size]
            encoded = tokenizer([row["text"] for row in batch], padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt").to(device)
            targets = torch.tensor([label2id[row["label"]] for row in batch], device=device)
            loss = model(**encoded, labels=targets).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item() * len(batch)
        losses.append(epoch_loss / len(train))
        model.eval()
        val_accuracy = classifier.accuracy(val)
        log(f"epoch {epoch + 1}/{epochs}: train loss {losses[-1]:.3f}, validation accuracy {val_accuracy:.1%}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    metrics = {"base_model": base_model, "labels": labels, "train_size": len(train), "val_size": len(val),
               "epochs": epochs, "losses": losses, "val_accuracy": val_accuracy}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# --- Milestone 4: Create a CLI game interface presenting partial legal sentences for completion.
# Each round shows the start of a real (held-out) clause plus n-gram hints; the player writes the ending.

DIFFICULTY = {"easy": 0.6, "medium": 0.4, "hard": 0.25}     # share of the real sentence the player is shown


@dataclass
class Round:
    prefix: str        # the start of a real clause, shown to the player
    target: str        # the clause type the player should write
    original: str      # the whole real sentence, revealed after the round


def make_round(row: dict, difficulty: str = "medium") -> Round:
    """Cut a corpus sentence down to its first words; harder levels show fewer words (always 3+ to write)."""
    words = row["text"].split()
    shown = min(max(3, round(len(words) * DIFFICULTY[difficulty])), len(words) - 3)
    return Round(" ".join(words[:shown]), row["label"], row["text"])


def pick_rounds(rows: Iterable[dict], n: int = 5, difficulty: str = "medium", seed: int | None = None) -> list[Round]:
    """`n` rounds from `rows`, taking clause types in turn so one game mixes them."""
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    labels = sorted(by_label)
    rng.shuffle(labels)
    for label in labels:
        rng.shuffle(by_label[label])
    rounds: list[Round] = []
    while len(rounds) < n and any(by_label.values()):
        for label in labels:
            if by_label[label] and len(rounds) < n:
                rounds.append(make_round(by_label[label].pop(), difficulty))
    return rounds


def join_completion(prefix: str, ending: str) -> str:
    """The player's full sentence: the shown start plus what they typed."""
    ending = " ".join(ending.split())
    if not ending:
        return prefix
    return prefix + ("" if ending[0] in ",.;:)" else " ") + ending


def ngram_hints(model: NGramModel, text: str, k: int = 3) -> list[str]:
    """The k next words the n-gram model expects after `text`."""
    return ["(end)" if word == EOS else word for word, _ in model.predict(tokenize(text), k)]


def ngram_ending(model: NGramModel, text: str, max_words: int = 30) -> str:
    """How the n-gram model would finish `text`."""
    return detokenize(model.generate(tokenize(text), max_words))


def _read_ending(game_round: Round, ngram: NGramModel, ask, say) -> tuple[str | None, bool]:
    """Ask until the player types an ending; returns (ending or None if they quit, whether they used '?')."""
    hint_used = False
    while True:
        try:
            answer = ask("Your ending (? = n-gram suggestion, q = quit) > ").strip()
        except EOFError:
            answer = "q"
        if answer.lower() == "q":
            return None, hint_used
        if answer == "?":
            hint_used = True
            say(f"  n-gram suggests: ... {ngram_ending(ngram, game_round.prefix)}")
        elif len(answer.split()) < 2:
            say("  Write at least two words to finish the sentence.")
        else:
            return answer, hint_used


def play(rounds: list[Round], ngram: NGramModel, classifier=None, ask=None, say=None,
         keywords: dict[str, list[str]] | None = None) -> list[dict]:
    """Run the game in the terminal. With a `classifier`, GPT-2 marks each sentence (milestone 5).
    `ask` and `say` default to input() and print(); tests swap them to play the game."""
    ask, say = ask or input, say or print
    results = []
    for number, game_round in enumerate(rounds, 1):
        article = "an" if game_round.target[0] in "AEIOU" else "a"
        say(f"\nRound {number} of {len(rounds)} · write {article} {game_round.target} clause")
        say(f"  {game_round.prefix} ...")
        say(f"  n-gram hints for the next word: {' · '.join(ngram_hints(ngram, game_round.prefix))}")
        answer, hint_used = _read_ending(game_round, ngram, ask, say)
        if answer is None:
            break
        sentence = join_completion(game_round.prefix, answer)
        say(f"  Your clause: {sentence}")
        result = {"target": game_round.target, "sentence": sentence, "hint_used": hint_used}
        if classifier is not None:
            feedback = judge(classifier, sentence, game_round.target, hint_used)
            for line in feedback_lines(feedback, keywords):
                say(line)
            result.update(predicted=feedback.predicted, matched=feedback.matched, points=feedback.points,
                          confidence=round(feedback.probs[feedback.predicted], 4))
        if not hint_used:
            say(f"  n-gram would end it: ... {ngram_ending(ngram, game_round.prefix)}")
        say(f"  Real clause: {game_round.original}")
        results.append(result)
    if classifier is not None and results:
        say("\n" + score_summary(results))
    return results


# --- Milestone 5: After user completion, classify the full sentence and provide feedback.
# GPT-2 tags the player's sentence; points = 10 x the probability it gave the clause type that was asked for.

_STOPWORDS = {"this", "that", "shall", "such", "with", "from", "have", "been", "which", "under", "other", "will",
              "than", "into", "upon", "each", "their", "there", "these", "those", "then", "them", "they", "would",
              "only", "also", "were", "being", "hereby", "herein", "pursuant", "respect"}


def clause_keywords(rows: Iterable[dict], top: int = 5) -> dict[str, list[str]]:
    """Words that mark each clause type: common in its sentences, rare in the others (weighted log-odds)."""
    doc_freq: dict[str, Counter] = defaultdict(Counter)
    docs: Counter = Counter()
    for row in rows:
        docs[row["label"]] += 1
        doc_freq[row["label"]].update({t for t in tokenize(row["text"])
                                       if t.isalpha() and len(t) > 3 and t not in _STOPWORDS})
    overall: Counter = sum(doc_freq.values(), Counter())
    total = sum(docs.values())
    keywords = {}
    for label, freq in doc_freq.items():
        def score(word: str) -> float:
            inside = (freq[word] + 1) / (docs[label] + 2)
            outside = (overall[word] - freq[word] + 1) / (total - docs[label] + 2)
            return inside * math.log(inside / outside)
        keywords[label] = sorted(freq, key=lambda word: (-score(word), word))[:top]
    return keywords


@dataclass
class Feedback:
    sentence: str
    target: str                 # the clause type the player was asked for
    probs: dict[str, float]     # GPT-2's probability for every clause type, most likely first
    hint_used: bool = False

    @property
    def predicted(self) -> str:
        return next(iter(self.probs))

    @property
    def matched(self) -> bool:
        return self.predicted == self.target

    @property
    def points(self) -> int:
        points = round(10 * self.probs.get(self.target, 0.0))
        return points // 2 if self.hint_used else points


def judge(classifier, sentence: str, target: str, hint_used: bool = False) -> Feedback:
    """Classify the player's finished sentence with the fine-tuned GPT-2."""
    return Feedback(sentence, target, classifier.predict_proba(sentence), hint_used)


def feedback_lines(feedback: Feedback, keywords: dict[str, list[str]] | None = None) -> list[str]:
    """What the player sees after a round: the verdict, the runners-up, tips on a miss, and the points."""
    ranked = list(feedback.probs.items())
    verdict = f"  GPT-2 reads it as: {feedback.predicted} ({ranked[0][1]:.0%})"
    if feedback.matched:
        lines = [verdict + "  ✓ the clause you were asked for"]
    else:
        lines = [verdict + f"  ✗ you were asked for {feedback.target} "
                           f"({feedback.probs.get(feedback.target, 0.0):.0%})"]
        if keywords and keywords.get(feedback.target):
            lines.append(f"  Tip: {feedback.target} clauses usually say: {', '.join(keywords[feedback.target])}")
    lines.append("  Next best: " + " · ".join(f"{label} {prob:.0%}" for label, prob in ranked[1:3]))
    lines.append(f"  +{feedback.points} points" + (" (halved for the n-gram suggestion)" if feedback.hint_used else ""))
    return lines


def score_summary(results: list[dict]) -> str:
    score = sum(result["points"] for result in results)
    matched = sum(result["matched"] for result in results)
    return (f"Score: {score} / {10 * len(results)} · "
            f"{matched} of {len(results)} clauses read as the type you were asked for")


# --- How well it works: each model next to a simple baseline (`python -m app.cli evaluate`).

def next_word_accuracy(model: NGramModel, sentences: Iterable[list[str]], k: int = 1) -> float:
    """Share of held-out next words (end of sentence included) that are in the model's top-k guesses."""
    hits = total = 0
    for tokens in sentences:
        for i, word in enumerate(list(tokens) + [EOS]):
            hits += word in {guess for guess, _ in model.predict(tokens[:i], k)}
            total += 1
    return hits / max(total, 1)


class KeywordBaseline:
    """The simple rival to GPT-2: pick the clause type whose marker words appear most in the sentence."""

    def __init__(self, rows: Iterable[dict], top: int = 10):
        rows = list(rows)
        self.keywords = {label: set(words) for label, words in clause_keywords(rows, top).items()}
        self.fallback = Counter(row["label"] for row in rows).most_common(1)[0][0]

    def classify(self, text: str) -> tuple[str, float]:
        words = set(tokenize(text))
        hits = {label: len(words & marks) for label, marks in sorted(self.keywords.items())}
        best = max(hits, key=hits.get)
        return (best, hits[best] / len(self.keywords[best])) if hits[best] else (self.fallback, 0.0)

    def accuracy(self, rows: Iterable[dict]) -> float:
        rows = list(rows)
        return sum(self.classify(row["text"])[0] == row["label"] for row in rows) / max(len(rows), 1)


def evaluate(classifier=None, train_rows: list[dict] | None = None,
             test_rows: list[dict] | None = None) -> list[tuple[str, str, float]]:
    """(task, method, score) rows on the held-out game sentences: baselines first, then the real models."""
    train_rows = load_corpus(split="train") if train_rows is None else train_rows
    test_rows = load_corpus(split="play") if test_rows is None else test_rows
    majority = Counter(row["label"] for row in train_rows).most_common(1)[0][0]
    results = [
        ("clause type accuracy", f"baseline: always '{majority}'",
         sum(row["label"] == majority for row in test_rows) / len(test_rows)),
        ("clause type accuracy", "baseline: count marker words", KeywordBaseline(train_rows).accuracy(test_rows)),
    ]
    if classifier is not None:
        results.append(("clause type accuracy", "fine-tuned GPT-2", classifier.accuracy(test_rows)))
    sentences = corpus_tokens(test_rows)
    for n, name in [(1, "baseline: most common word"), (2, "2-gram"), (3, "3-gram"), (4, "4-gram (used in the game)")]:
        model = train_ngram(train_rows, n=n)
        results.append(("next word in top 3", name, next_word_accuracy(model, sentences, k=3)))
    return results
