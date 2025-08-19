import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Compile dataset cleanup patterns to make preprocessing faster.
PATTERNS = [
    # Remove hypertext
    re.compile(r'https?://\S+|www\.\S+'),
    # Leave only alphanumericals
    re.compile(r'[^a-z0-9\s]'),
    # Remove whitespace sequences
    re.compile(r'\s+')
]


def load_dataset(path: str, cap: int = -1) -> str:
    with open(path, "r") as f_in:
        dataset = [f_in.readline() for i in range(0, cap)] if cap > 0 else [
            line for line in f_in]
    return dataset


def clean_up(texts: list[str]) -> list[str]:
    clean_texts = []
    for text in tqdm(texts):
        text = text.lower()
        for pattern in PATTERNS:
            text = pattern.sub(' ', text)
        clean_texts.append(text.strip())

    return clean_texts


def split_dataset(text: list[str]) -> dict:
    # train, validation and test subsets size
    _ = 0.8
    val = 0.1
    test = 0.1

    train_set, val_set = train_test_split(
        text, test_size=val + test, random_state=42)

    val_set, test_set = train_test_split(
        val_set, test_size=0.5, random_state=42)

    print(f"train set len:       {len(train_set)}")
    print(f"validataion set len: {len(val_set)}")
    print(f"test set len:        {len(test_set)}")

    return {"train": train_set, "val": val_set, "test": test_set}
