from typing import List
import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):
    def __init__(
        self,
        token_ids: List[int],
        sequence_length: int,
        pad_token_id: int,
    ):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2.")

        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
        self.samples = []

        stride = sequence_length

        for start in range(0, len(token_ids), stride):
            chunk = token_ids[start:start + sequence_length + 1]

            if len(chunk) < 2:
                continue

            input_ids = chunk[:-1]
            labels = chunk[1:]

            input_ids = input_ids + [
                pad_token_id
            ] * (sequence_length - len(input_ids))

            labels = labels + [
                pad_token_id
            ] * (sequence_length - len(labels))

            self.samples.append(
                (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]
