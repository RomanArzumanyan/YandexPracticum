import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm

TOKENIZER = BertTokenizerFast.from_pretrained("bert-base-uncased")
DEVICE = torch.device("cuda")
BATCH_SIZE = 1024
SEQ_LEN = 6


class NextTokenDataset(Dataset):
    def __init__(self, texts):
        self.samples = []

        for line in tqdm(texts):
            ret = tokenize_line(line)
            if ret:
                self.samples.extend(ret)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x).to(DEVICE), torch.tensor(y).to(DEVICE)


def tokenize_line(line: str) -> list[tuple[list[int], int]]:
    if len(line.split()) < SEQ_LEN:
        return None

    token_ids = TOKENIZER.encode(
        line, add_special_tokens=False, max_length=512, truncation=True)

    if len(token_ids) < SEQ_LEN:
        return None

    # Go through tokenized sequence with sliding window of size SEQ_LEN
    ret = []
    for head in range(0, len(token_ids) - SEQ_LEN):
        tail = min(head + SEQ_LEN, len(token_ids)) - 1
        context = token_ids[head:tail] + [TOKENIZER.mask_token_id]
        target = token_ids[tail]
        ret.append((context, target))

    return ret


def prepare_data(text: list[str], shuffle: bool = False) -> tuple[NextTokenDataset, DataLoader]:
    dataset = NextTokenDataset(text)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    return dataset, loader
