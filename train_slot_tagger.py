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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.005)
    args = parser.parse_args()

    samples = load_jsonl(args.data) if args.data else generate_bio_training_data(args.limit)
    tagger = BiLSTMCRFSlotTagger(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    tagger.fit(samples, epochs=args.epochs, learning_rate=args.lr, verbose=True)
    tagger.save(args.output)
    print(f"Saved slot tagger to {args.output} with {len(samples)} samples.")


if __name__ == "__main__":
    main()
