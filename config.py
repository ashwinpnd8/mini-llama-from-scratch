from dataclasses import dataclass, asdict
import json


@dataclass
class ModelConfig:
    vocab_size: int
    pad_token_id: int
    hidden_size: int = 128
    num_layers: int = 2
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    intermediate_size: int = 352
    max_position_embeddings: int = 128
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    use_qk_norm: bool = True

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(asdict(self), file, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as file:
            return cls(**json.load(file))
