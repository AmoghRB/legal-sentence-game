"""Command-line entry point: `python -m app.cli --help`."""
import argparse
from collections import Counter

from app import core


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="legal-sentence-game", description="Law students play a CLI game to practice drafting legal sentences using n-gram prediction, receiving instant feedback on generated clause types from a fine-tuned GPT-2 model.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    build = sub.add_parser("build-corpus", help="milestone 1: rebuild corpus/clauses.jsonl from LEDGAR")
    build.add_argument("--per-label", type=int, default=250, help="sentences kept per clause type")
    build.add_argument("--out", default=str(core.CORPUS_PATH))

    predict = sub.add_parser("predict", help="milestone 2: n-gram next-word predictions for a sentence start")
    predict.add_argument("text", help='e.g. "This Agreement shall be governed by"')
    predict.add_argument("-k", type=int, default=5, help="how many next words to show")

    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Legal Sentence Game is ready. Open README.md for your milestones.")
    elif args.command == "build-corpus":
        rows = core.build_corpus(core.download_ledgar(), per_label=args.per_label)
        path = core.save_corpus(rows, args.out)
        counts = Counter(row["label"] for row in rows)
        print(f"Wrote {len(rows)} sentences to {path}")
        for label, n in sorted(counts.items()):
            print(f"  {label:<18} {n}")
    elif args.command == "predict":
        model = core.train_ngram()
        tokens = core.tokenize(args.text)
        for word, prob in model.predict(tokens, args.k):
            print(f"  {word:<15} {prob:.1%}")
        print(f"Most likely ending: {args.text} {core.detokenize(model.generate(tokens))}")


if __name__ == "__main__":
    main()
