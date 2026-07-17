import argparse
from pathlib import Path

import torch

from config import ModelConfig
from model import MiniLlamaLanguageModel
from tokenizer import BPETokenizer
from utils import set_seed


def generate(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    artifact_dir = Path(args.artifact_dir)

    tokenizer = BPETokenizer.load(
        str(artifact_dir / "tokenizer.json")
    )

    config = ModelConfig.load(
        str(artifact_dir / "config.json")
    )

    checkpoint = torch.load(
        artifact_dir / "model.pt",
        map_location=device,
    )

    model = MiniLlamaLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_ids = tokenizer.encode(
        args.prompt,
        add_special_tokens=True,
    )

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    generated = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(
        generated[0].tolist()
    )

    print(text)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate text from the trained mini Llama model."
    )

    parser.add_argument("--artifact-dir", type=str, default="artifacts")
    parser.add_argument("--prompt", type=str, default="This document")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    return parser


if __name__ == "__main__":
    generate(build_parser().parse_args())
