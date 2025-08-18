import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast


DEVICE = torch.device("cuda")

class NextTokenDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=7):
        self.samples = []

        for line in texts:
            if len(line.split()) < seq_len:
                continue

            token_ids = tokenizer.encode(
                line, add_special_tokens=False, max_length=512, truncation=True)

            if len(token_ids) < seq_len:
                continue

            # Go through tokenized sequence with sliding window of size seq_len
            for head in range(0, len(token_ids) - seq_len):
                tail = min(head + seq_len, len(token_ids)) - 1
                context = token_ids[head:tail] + [tokenizer.mask_token_id]
                target = token_ids[tail]
                # Append sequence + expected prediction to samples
                self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x).to(DEVICE), torch.tensor(y).to(DEVICE)


TOKENIZER = BertTokenizerFast.from_pretrained("bert-base-uncased")


def tokenizer() -> BertTokenizerFast:
    return TOKENIZER


def prepare_data(text: list[str], seq_len: int = 7, shuffle: bool = False) -> tuple[NextTokenDataset, DataLoader]:
    dataset = NextTokenDataset(text, TOKENIZER, seq_len)
    loader = DataLoader(dataset, batch_size=64, shuffle=shuffle)
    return dataset, loader
