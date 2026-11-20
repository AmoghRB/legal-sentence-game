"""Shared fixtures. The GPT-2 tests use a tiny, randomly initialised GPT-2 so they run offline in seconds;
the real fine-tuned model (`python -m app.cli train`) has the same architecture and files."""
import pytest

from app import core


def _tiny_gpt2(path, texts):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=400, special_tokens=["<|endoftext|>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|endoftext|>", bos_token="<|endoftext|>")
    config = GPT2Config(vocab_size=len(tokenizer), n_positions=128, n_embd=32, n_layer=2, n_head=2,
                        bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    GPT2LMHeadModel(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def small_corpus():
    """A few training sentences of every clause type."""
    rows, per_label = [], {}
    for row in core.load_corpus(split="train"):
        if per_label.get(row["label"], 0) < 6:
            per_label[row["label"]] = per_label.get(row["label"], 0) + 1
            rows.append(row)
    return rows


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory, small_corpus):
    """A tiny GPT-2 fine-tuned for one epoch on `small_corpus`, saved like the real one."""
    base = _tiny_gpt2(tmp_path_factory.mktemp("tiny-gpt2"), [row["text"] for row in small_corpus])
    out = tmp_path_factory.mktemp("clause-gpt2")
    core.train_classifier(small_corpus, base_model=str(base), out_dir=out, epochs=1, batch_size=8,
                          device="cpu", log=lambda _: None)
    return out
