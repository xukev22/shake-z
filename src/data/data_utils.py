import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def load_slang_dataset(csv_path):
    """
    Load the Gen Z slang dataset.
    Expects columns: 'slang', 'description', 'example', 'context'.
    We treat 'description' as the source (standard English) and 'slang' as the target.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Description", "Slang"])
    # Build (source, target) pairs
    pairs = list(
        zip(df["Description"].astype(str).tolist(), df["Slang"].astype(str).tolist())
    )
    return pairs


def load_parallel_dataset(csv_path):
    """
    Load parallel Shakespeare→GenZ dataset.
    Expects columns: 'source', 'target'.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["source", "target"])
    return list(zip(df["source"].astype(str), df["target"].astype(str)))


def split_data(pairs, test_size=0.1, val_size=0.1, random_state=42):
    """
    Split a list of (source, target) pairs into train, val, and test sets.
    """
    train_val, test = train_test_split(
        pairs, test_size=test_size, random_state=random_state
    )
    # Compute val size relative to the remaining data
    val_relative = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_relative, random_state=random_state
    )
    return train, val, test


class TranslationDataset(Dataset):
    """
    PyTorch Dataset that tokenizes on the fly.

    Each item is a dict with:
      - input_ids:         LongTensor [max_length]
      - attention_mask:    LongTensor [max_length]
      - labels:            LongTensor [max_length]  (with pad tokens masked to -100)
    """

    def __init__(self, data_pairs, tokenizer, max_length=50):
        """
        Args:
            data_pairs (List[Tuple[str,str]]): list of (source, target) strings
            tokenizer (PreTrainedTokenizer): e.g. T5Tokenizer
            max_length (int): maximum sequence length for both source & target
        """
        self.pairs = data_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        # encode source
        enc = self.tokenizer(
            src,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # encode target
        dec = self.tokenizer(
            tgt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # prepare labels (mask pad tokens as -100 so they’re ignored in loss)
        labels = dec.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }


def create_dataloaders(
    train_pairs,
    val_pairs,
    test_pairs,
    tokenizer,
    max_length,
    batch_size=32,
    shuffle=True,
):
    """
    Build PyTorch DataLoaders for train/val/test splits.

    Args:
        train_pairs (List[Tuple[str,str]]): (source, target) for training.
        val_pairs   (List[Tuple[str,str]]): for validation.
        test_pairs  (List[Tuple[str,str]]): for testing.
        tokenizer   (PreTrainedTokenizer): e.g. T5Tokenizer.
        max_length  (int): max sequence length for both source & target.
        batch_size  (int): batch size.
        shuffle     (bool): whether to shuffle the train split.

    Returns:
        train_loader, val_loader, test_loader: three DataLoader objects,
        each yielding dicts with keys "input_ids", "attention_mask", "labels".
    """
    # wrap each split in our on‑the‑fly tokenizing Dataset
    train_ds = TranslationDataset(train_pairs, tokenizer, max_length)
    val_ds = TranslationDataset(val_pairs, tokenizer, max_length)
    test_ds = TranslationDataset(test_pairs, tokenizer, max_length)

    # default collate will batch the fixed-size tensors
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_data(config):
    path = config["data_path"]
    dataset = config.get("dataset", "shakez")
    if dataset == "shakez":
        csv_path = os.path.join(path, "mappings/shakez.csv")
        pairs = load_parallel_dataset(csv_path)
    elif dataset == "sonnetz":
        csv_path = os.path.join(path, "mappings/sonnetz.csv")
        pairs = load_parallel_dataset(csv_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    test_size = config.get("test_size", 0.1)
    val_size = config.get("val_size", 0.1)
    return split_data(pairs, test_size=test_size, val_size=val_size)
