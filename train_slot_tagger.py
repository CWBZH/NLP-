import argparse
import json

from generate_training_data import generate_bio_training_data
from slot_tagger import BiLSTMCRFSlotTagger


def load_jsonl(path: str):
    samples = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BiLSTM-CRF path slot tagger.")
    parser.add_argument("--data", default=None, help="Optional JSONL BIO training data path.")
    parser.add_argument("--output", default="slot_tagger.pkl")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    samples = load_jsonl(args.data) if args.data else generate_bio_training_data(args.limit)
    tagger = BiLSTMCRFSlotTagger(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )
    tagger.fit(samples, epochs=args.epochs)
    tagger.save(args.output)
    print(f"Saved slot tagger to {args.output} with {len(samples)} samples.")


if __name__ == "__main__":
    main()
