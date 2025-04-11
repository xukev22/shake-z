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
    Simple PyTorch Dataset for translation pairs.
    Each item is a dict: {'source': str, 'target': str}.
    """

    def __init__(self, data_pairs):
        self.data = data_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return {"source": src, "target": tgt}


def create_dataloaders(train_pairs, val_pairs, test_pairs, batch_size=32, shuffle=True):
    """
    Wrap train/val/test pairs in DataLoaders.
    Returns: train_loader, val_loader, test_loader
    """
    train_ds = TranslationDataset(train_pairs)
    val_ds = TranslationDataset(val_pairs)
    test_ds = TranslationDataset(test_pairs)

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
