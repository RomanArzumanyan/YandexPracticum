import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import requests


# Compile dataset cleanup patterns to make preprocessing faster.
PATTERNS = [
    # Remove hypertext
    re.compile(r'https?://\S+|www\.\S+'),
    # Leave only alphanumericals
    re.compile(r'[^a-z0-9\s]'),
    # Remove whitespace sequences
    re.compile(r'\s+')
]


def download_from_url(url: str, filename: str) -> None:
    """
    Download file.

    Args:
        url (str): input URL
        filename (str): destination path
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def load_dataset(path: str, cap: int = -1) -> str:
    """
    Load dataset from file

    Args:
        path (str): filepath
        cap (int, optional): number of string to read. Defaults to -1.

    Returns:
        str: string with text
    """
    with open(path, "r") as f_in:
        dataset = [f_in.readline() for i in range(0, cap)] if cap > 0 else [
            line for line in f_in]
    return dataset


def clean_up(texts: list[str]) -> list[str]:
    """
    Cleanup dataset, leaving only alphanumericals

    Args:
        texts (list[str]): list of sentences

    Returns:
        list[str]: list of "clean" sentences
    """
    clean_texts = []
    for text in tqdm(texts):
        text = text.lower()
        for pattern in PATTERNS:
            text = pattern.sub(' ', text)
        clean_texts.append(text.strip())

    return clean_texts


def split_dataset(
        text: list[str],
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1) -> dict:
    """
    Split dataset into 3 parts: train, val, test

    Returns:
        dict: dictionary with 3 datasets
    """
    assert (train + val + test == 1.0)
    assert (train * val * test > 0.0)

    # Split into [train] and [val + train]
    a = train
    b = 1 - a
    train_set, val_set = train_test_split(
        text, test_size=b, train_size=a, random_state=42)

    # Split [val + train] into [val] and [train]
    a = val / (val + test)
    b = 1 - a
    val_set, test_set = train_test_split(
        val_set, test_size=b, train_size=a, random_state=42)

    print(f"train set len:       {len(train_set)}")
    print(f"validataion set len: {len(val_set)}")
    print(f"test set len:        {len(test_set)}")

    return {"train": train_set, "val": val_set, "test": test_set}
