import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.pow(2).mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return (self.weight * normalized).to(input_dtype)


class QKNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * torch.rsqrt(
            hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )


def repeat_kv(hidden_states: torch.Tensor, repetitions: int) -> torch.Tensor:
    batch_size, key_value_heads, sequence_length, head_dim = hidden_states.shape

    if repetitions == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size,
        key_value_heads,
        repetitions,
        sequence_length,
        head_dim,
    )

    return hidden_states.reshape(
        batch_size,
        key_value_heads * repetitions,
        sequence_length,
        head_dim,
    )


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even.")

        inverse_frequency = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )

        positions = torch.arange(
            max_position_embeddings,
            dtype=torch.float32,
        )

        frequencies = torch.outer(positions, inverse_frequency)

        self.register_buffer(
            "cosine",
            frequencies.cos(),
            persistent=False,
        )
        self.register_buffer(
            "sine",
            frequencies.sin(),
            persistent=False,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cosine = self.cosine[position_ids].unsqueeze(1)
        sine = self.sine[position_ids].unsqueeze(1)

        def rotate(hidden_states: torch.Tensor) -> torch.Tensor:
            even = hidden_states[..., 0::2]
            odd = hidden_states[..., 1::2]
            rotated_even = even * cosine - odd * sine
            rotated_odd = even * sine + odd * cosine
            return torch.stack(
                (rotated_even, rotated_odd),
                dim=-1,
            ).flatten(-2)

        return rotate(query), rotate(key)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads."
            )

        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads."
            )

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = (
            config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.attention_dropout = config.attention_dropout
        self.use_qk_norm = config.use_qk_norm

        self.query_projection = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.key_projection = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.value_projection = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.output_projection = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )

        self.rotary_embedding = RotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

        if self.use_qk_norm:
            self.query_key_norm = QKNorm()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.query_projection(hidden_states)
        key = self.key_projection(hidden_states)
        value = self.value_projection(hidden_states)

        query = query.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)

        key = key.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        value = value.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        query, key = self.rotary_embedding(
            query,
            key,
            position_ids,
        )

        if self.use_qk_norm:
            query = self.query_key_norm(query)
            key = self.query_key_norm(key)

        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        ) / math.sqrt(self.head_dim)

        attention_scores = attention_scores + attention_mask

        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)

        attention_weights = F.dropout(
            attention_weights,
            p=self.attention_dropout,
            training=self.training,
        )

        attention_output = torch.matmul(
            attention_weights,
            value,
        )

        attention_output = attention_output.transpose(
            1,
            2,
        ).contiguous()

        attention_output = attention_output.view(
            batch_size,
            sequence_length,
            self.hidden_size,
        )

        return (
            self.output_projection(attention_output),
            attention_weights,
        )


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_projection = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_projection = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_projection = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_projection(hidden_states))
        up = self.up_projection(hidden_states)
        return self.down_projection(gate * up)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.attention = GroupedQueryAttention(config)
        self.post_attention_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.feed_forward = SwiGLUFeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_output, attention_weights = self.attention(
            self.input_norm(hidden_states),
            attention_mask,
            position_ids,
        )

        hidden_states = hidden_states + attention_output

        feed_forward_output = self.feed_forward(
            self.post_attention_norm(hidden_states)
        )

        hidden_states = hidden_states + feed_forward_output

        return hidden_states, attention_weights


class MiniLlamaLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )

        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        self.final_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.language_model_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.language_model_head.weight = self.token_embedding.weight

    def _create_attention_mask(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        device = input_ids.device

        causal_mask = torch.triu(
            torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                device=device,
            ),
            diagonal=1,
        )

        causal_mask = causal_mask.view(
            1,
            1,
            sequence_length,
            sequence_length,
        )

        padding_mask = input_ids.eq(
            self.config.pad_token_id
        ).view(
            batch_size,
            1,
            1,
            sequence_length,
        )

        padding_values = torch.zeros(
            (
                batch_size,
                1,
                sequence_length,
                sequence_length,
            ),
            device=device,
        )

        padding_values = padding_values.masked_fill(
            padding_mask,
            float("-inf"),
        )

        return causal_mask + padding_values

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.config.max_position_embeddings:
            raise ValueError(
                "Sequence length exceeds max_position_embeddings."
            )

        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device,
        ).unsqueeze(0).expand(batch_size, -1)

        attention_mask = self._create_attention_mask(input_ids)
        hidden_states = self.token_embedding(input_ids)
        attention_maps = []

        for layer in self.layers:
            hidden_states, attention_weights = layer(
                hidden_states,
                attention_mask,
                position_ids,
            )
            attention_maps.append(attention_weights)

        logits = self.language_model_head(
            self.final_norm(hidden_states)
        )

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=self.config.pad_token_id,
            )

        return {
            "logits": logits,
            "loss": loss,
            "attention_maps": attention_maps,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 30,
        temperature: float = 0.8,
        top_k: int = 20,
        eos_token_id: int = None,
    ) -> torch.Tensor:
        self.eval()

        for _ in range(max_new_tokens):
            model_input = input_ids[
                :,
                -self.config.max_position_embeddings:,
            ]

            outputs = self(model_input)
            logits = outputs["logits"][:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True,
                )
            else:
                logits = logits / temperature

                if top_k > 0:
                    values, indices = torch.topk(
                        logits,
                        k=min(top_k, logits.size(-1)),
                    )

                    filtered = torch.full_like(
                        logits,
                        float("-inf"),
                    )

                    filtered.scatter_(1, indices, values)
                    logits = filtered

                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                )

            input_ids = torch.cat(
                (input_ids, next_token),
                dim=1,
            )

            if (
                eos_token_id is not None
                and torch.all(next_token.eq(eos_token_id))
            ):
                break

        return input_ids
