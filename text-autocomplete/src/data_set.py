import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

TOKENIZER = BertTokenizerFast.from_pretrained("bert-base-uncased")
DEVICE = torch.device("cuda")
BATCH_SIZE = 256
# Min sentence length in tokens
MIN_LEN = 4
# Max sentence length in tokens
MAX_LEN = 80


class TwitterDataset(Dataset):
    def __init__(self, texts):
        self.samples = []

        for line in tqdm(texts):
            ret = tokenize_line(line)
            if ret:
                self.samples.append(ret)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return {
            'context': torch.tensor(x, dtype=torch.long).to(DEVICE),
            'token': torch.tensor(y, dtype=torch.long).to(DEVICE)
        }


def tokenize_line(line: str) -> tuple[list[int], int]:
    token_ids = TOKENIZER.encode(
        line, add_special_tokens=False, max_length=MAX_LEN, truncation=True)

    if len(token_ids) < MIN_LEN:
        return None

    head = 0
    tail = min(len(token_ids), MAX_LEN) - 1
    context = token_ids[head:tail] + [TOKENIZER.mask_token_id]
    target = token_ids[tail]
    return context, target


def collate(batch) -> dict:
    contexts = [item['context'] for item in batch]
    tokens = [item['token'] for item in batch]
    lengths = [len(ctx) for ctx in contexts]
    padded_contexts = pad_sequence(contexts, batch_first=True, padding_value=0)

    return {
        'contexts': padded_contexts,
        'lengths': lengths,
        'tokens': tokens}


def prepare_data(text: list[str], shuffle: bool = False) -> tuple[TwitterDataset, DataLoader]:
    dataset = TwitterDataset(text)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=shuffle, collate_fn=collate)
    return dataset, loader
