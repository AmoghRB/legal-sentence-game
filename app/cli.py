"""Command-line entry point: `python -m app.cli --help`."""
import argparse
from collections import Counter
from datetime import datetime

from app import core
from app.storage import Store


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

    train = sub.add_parser("train", help="milestone 3: fine-tune GPT-2 to tag clause types")
    train.add_argument("--base", default="gpt2", help="model to start from (gpt2, distilgpt2 or a local folder)")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--out", default=str(core.MODEL_DIR))

    classify = sub.add_parser("classify", help="milestone 3: clause type of a sentence, by the fine-tuned GPT-2")
    classify.add_argument("text")
    classify.add_argument("--model", default=str(core.MODEL_DIR))

    play = sub.add_parser("play", help="play the game")
    play.add_argument("--rounds", type=int, default=5)
    play.add_argument("--difficulty", choices=list(core.DIFFICULTY), default="medium",
                      help="how much of each clause you are shown: easy 60%%, medium 40%%, hard 25%%")
    play.add_argument("--seed", type=int, default=None, help="same seed, same clauses")
    play.add_argument("--model", default=str(core.MODEL_DIR), help="the fine-tuned GPT-2 folder")
    sub.add_parser("history", help="your past games and best score")

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
    elif args.command == "train":
        print(f"Fine-tuning {args.base} on {core.CORPUS_PATH.name} (training split)...")
        metrics = core.train_classifier(base_model=args.base, out_dir=args.out, epochs=args.epochs,
                                        batch_size=args.batch_size)
        held_out = core.load_classifier(args.out).accuracy(core.load_corpus(split="play"))
        print(f"Saved to {args.out}. Accuracy on the held-out game sentences: {held_out:.1%}")
    elif args.command == "classify":
        try:
            classifier = core.load_classifier(args.model)
        except FileNotFoundError as error:
            raise SystemExit(str(error))
        for label, prob in list(classifier.predict_proba(args.text).items())[:3]:
            print(f"  {label:<18} {prob:.1%}")
    elif args.command == "play":
        try:
            classifier = core.load_classifier(args.model)
        except FileNotFoundError as error:
            raise SystemExit(str(error))
        train_rows = core.load_corpus(split="train")
        ngram, keywords = core.train_ngram(train_rows), core.clause_keywords(train_rows)
        rounds = core.pick_rounds(core.load_corpus(split="play"), args.rounds, args.difficulty, args.seed)
        print(f"Legal Sentence Game · {len(rounds)} rounds · {args.difficulty}")
        print("Finish each clause so the fine-tuned GPT-2 reads it as the clause type asked for.")
        print("Points per round: 10 x GPT-2's probability for that type (halved if you ask for the n-gram's ending).")
        try:
            results = core.play(rounds, ngram, classifier, keywords=keywords)
        except KeyboardInterrupt:
            print()
            return
        if results:
            store = Store()
            store.put("games", store.get("games", []) + [{
                "date": datetime.now().isoformat(timespec="minutes"), "difficulty": args.difficulty,
                "rounds": len(results), "score": sum(r["points"] for r in results), "max": 10 * len(results)}])
    elif args.command == "history":
        games = Store().get("games", [])
        if not games:
            print("No games yet. Run: python -m app.cli play")
        for game in games[-10:]:
            print(f"  {game['date']}  {game['difficulty']:<6}  {game['score']:>3} / {game['max']}")
        if games:
            best = max(games, key=lambda game: game["score"] / game["max"])
            print(f"Best: {best['score']} / {best['max']} on {best['date']} ({best['difficulty']})")

if __name__ == "__main__":
    main()
