import torch

from config import ModelConfig
from model import MiniLlamaLanguageModel
from tokenizer import BPETokenizer


def main():
    corpus = [
        "This is a small tokenizer test.",
        "This is a small transformer test.",
    ]

    tokenizer = BPETokenizer(num_merges=20)
    tokenizer.fit(corpus)

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        hidden_size=64,
        num_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=32,
    )

    model = MiniLlamaLanguageModel(config)

    token_ids = tokenizer.encode(corpus[0])[:32]

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
    )

    labels = input_ids.clone()
    outputs = model(input_ids, labels=labels)

    assert outputs["logits"].shape == (
        1,
        input_ids.shape[1],
        tokenizer.vocab_size,
    )

    assert outputs["loss"] is not None

    generated = model.generate(
        input_ids,
        max_new_tokens=2,
        temperature=0,
        eos_token_id=tokenizer.eos_token_id,
    )

    print("Smoke test passed.")
    print("Logits shape:", outputs["logits"].shape)
    print("Loss:", float(outputs["loss"]))
    print("Generated shape:", generated.shape)


if __name__ == "__main__":
    main()
