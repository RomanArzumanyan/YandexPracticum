import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

TOKENIZER = BertTokenizerFast.from_pretrained("bert-base-uncased")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 256
MIN_LEN = 4
MAX_LEN = 80


def tokenize(line: str, num_targets: int = 1) -> tuple[list[int], int]:
    """
    Tokenize sentence

    Args:
        line (str): sentence to be tokenized
        num_targets (int, optional): number of tokens to take as prediction targets. Defaults to 1.

    Returns:
        tuple[list[int], int]: tuple, first element is tokenized context, second is tokenized target.
    """
    assert (num_targets >= 0)

    token_ids = TOKENIZER.encode(
        line, add_special_tokens=False, max_length=MAX_LEN, truncation=True)
    tok_len = len(token_ids)

    if (tok_len < MIN_LEN) or (tok_len > MAX_LEN) or (num_targets >= tok_len):
        return None

    tail = min(len(token_ids), MAX_LEN) - num_targets
    context = token_ids[:tail] + [TOKENIZER.mask_token_id]
    target = token_ids[tail:] if num_targets > 0 else -1
    return context, target


def collate(batch) -> dict:
    """
    Custom collate function

    Returns:
        dict: dictionary with padded context, lengths of contexts and target tokens.
    """
    contexts = [item['context'] for item in batch]
    tokens = [item['token'] for item in batch]
    lengths = [len(ctx) for ctx in contexts]
    padded_contexts = pad_sequence(contexts, batch_first=True, padding_value=0)

    return {
        'contexts': padded_contexts,
        'lengths': lengths,
        'tokens': tokens}


class TwitterDataset(Dataset):
    """
    Torch dataset
    """

    def __init__(self, texts, num_targets: int = 1, shuffle: bool = False):
        self.samples = []

        for line in tqdm(texts):
            ret = tokenize(line, num_targets)
            if ret:
                self.samples.append(ret)

        self.loader = DataLoader(
            self, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collate)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return {
            'context': torch.tensor(x, dtype=torch.long).to(DEVICE),
            'token': torch.tensor(y, dtype=torch.long).to(DEVICE)
        }

    def get_loader(self):
        return self.loader
