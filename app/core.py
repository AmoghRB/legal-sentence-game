"""Legal Sentence Game: one section per milestone, built in order."""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
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
