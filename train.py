import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import ModelConfig
from dataset import LanguageModelDataset
from datasets import load_dataset
from model import MiniLlamaLanguageModel
from tokenizer import BPETokenizer
from utils import count_parameters, set_seed


DEFAULT_CORPUS = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
    "Attention helps a language model connect related tokens.",
    "Grouped query attention shares key and value heads.",
    "Rotary embeddings represent relative token positions.",
    "The feed forward network uses a SwiGLU gate.",
    "A transformer contains attention and feed forward layers.",
    "The model learns to predict the next token in a sequence.",
]


def load_corpus(path=None):
    print("Downloading TinyStories (5000 stories)...")

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train[:5000]"
    )

    corpus = dataset["text"]

    print(f"Loaded {len(corpus)} stories.")

    return corpus

def train(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(args.corpus)

    tokenizer = BPETokenizer(num_merges=args.num_merges)
    tokenizer.fit(corpus)
    tokenizer.save(str(output_dir / "tokenizer.json"))

    all_token_ids = []

    for document in corpus:
        all_token_ids.extend(tokenizer.encode(document))

    dataset = LanguageModelDataset(
        token_ids=all_token_ids,
        sequence_length=args.sequence_length,
        pad_token_id=tokenizer.pad_token_id,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.sequence_length,
    )

    config.save(str(output_dir / "config.json"))

    model = MiniLlamaLanguageModel(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("Device:", device)
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Dataset samples:", len(dataset))
    print("Trainable parameters:", count_parameters(model))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                input_ids=input_ids,
                labels=labels,
            )

            loss = outputs["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch}/{args.epochs} - loss: {average_loss:.4f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
    }

    torch.save(
        checkpoint,
        output_dir / "model.pt",
    )

    print("Saved model to:", output_dir / "model.pt")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the upgraded mini Llama-style language model."
    )

    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-merges", type=int, default=80)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--num-key-value-heads", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=352)
    parser.add_argument("--seed", type=int, default=42)

    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
