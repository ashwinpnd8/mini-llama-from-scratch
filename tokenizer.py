import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple


class BPETokenizer:
    def __init__(self, num_merges: int = 100):
        self.num_merges = num_merges
        self.end_of_word = "</w>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.merges: List[Tuple[str, str]] = []
        self.vocab: List[str] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    def _split_text(self, text: str) -> List[str]:
        return re.findall(r"\S+", text)

    def _get_pair_stats(
        self,
        splits: Dict[Tuple[str, ...], int],
    ) -> Dict[Tuple[str, str], int]:
        pair_counts = defaultdict(int)

        for symbols, frequency in splits.items():
            for index in range(len(symbols) - 1):
                pair_counts[(symbols[index], symbols[index + 1])] += frequency

        return dict(pair_counts)

    def _merge_pair(
        self,
        pair: Tuple[str, str],
        splits: Dict[Tuple[str, ...], int],
    ) -> Dict[Tuple[str, ...], int]:
        first, second = pair
        merged = first + second
        updated = defaultdict(int)

        for symbols, frequency in splits.items():
            new_symbols = []
            index = 0

            while index < len(symbols):
                if (
                    index < len(symbols) - 1
                    and symbols[index] == first
                    and symbols[index + 1] == second
                ):
                    new_symbols.append(merged)
                    index += 2
                else:
                    new_symbols.append(symbols[index])
                    index += 1

            updated[tuple(new_symbols)] += frequency

        return dict(updated)

    def fit(self, corpus: List[str]) -> None:
        word_splits = defaultdict(int)
        characters = set()

        for document in corpus:
            for character in document:
                if not character.isspace():
                    characters.add(character)

            for word in self._split_text(document):
                word_splits[tuple(list(word) + [self.end_of_word])] += 1

        current_splits = dict(word_splits)
        self.merges = []

        for _ in range(self.num_merges):
            pair_stats = self._get_pair_stats(current_splits)

            if not pair_stats:
                break

            best_pair = max(
                pair_stats,
                key=lambda pair: (pair_stats[pair], pair),
            )

            self.merges.append(best_pair)
            current_splits = self._merge_pair(best_pair, current_splits)

        learned_tokens = set(characters)
        learned_tokens.add(self.end_of_word)

        for first, second in self.merges:
            learned_tokens.add(first + second)

        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]

        self.vocab = special_tokens + sorted(learned_tokens)
        self.token_to_id = {
            token: index for index, token in enumerate(self.vocab)
        }
        self.id_to_token = {
            index: token for token, index in self.token_to_id.items()
        }

    def _encode_word(self, word: str) -> List[str]:
        symbols = list(word) + [self.end_of_word]

        for first, second in self.merges:
            merged = first + second
            updated = []
            index = 0

            while index < len(symbols):
                if (
                    index < len(symbols) - 1
                    and symbols[index] == first
                    and symbols[index + 1] == second
                ):
                    updated.append(merged)
                    index += 2
                else:
                    updated.append(symbols[index])
                    index += 1

            symbols = updated

        return symbols

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        if not self.token_to_id:
            raise RuntimeError("Tokenizer must be trained or loaded before encoding.")

        token_ids = []

        if add_special_tokens:
            token_ids.append(self.bos_token_id)

        for word in self._split_text(text):
            for token in self._encode_word(word):
                token_ids.append(
                    self.token_to_id.get(
                        token,
                        self.unk_token_id,
                    )
                )

        if add_special_tokens:
            token_ids.append(self.eos_token_id)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        words = []
        current_word = ""

        for token_id in token_ids:
            token = self.id_to_token.get(token_id, self.unk_token)

            if token in {
                self.pad_token,
                self.unk_token,
                self.bos_token,
            }:
                continue

            if token == self.eos_token:
                break

            current_word += token

            while self.end_of_word in current_word:
                word, current_word = current_word.split(
                    self.end_of_word,
                    1,
                )
                words.append(word)

        if current_word:
            words.append(current_word)

        return " ".join(words)

    def save(self, path: str) -> None:
        payload = {
            "num_merges": self.num_merges,
            "merges": self.merges,
            "vocab": self.vocab,
        }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        tokenizer = cls(num_merges=payload["num_merges"])
        tokenizer.merges = [tuple(pair) for pair in payload["merges"]]
        tokenizer.vocab = payload["vocab"]
        tokenizer.token_to_id = {
            token: index for index, token in enumerate(tokenizer.vocab)
        }
        tokenizer.id_to_token = {
            index: token for token, index in tokenizer.token_to_id.items()
        }

        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.eos_token]
