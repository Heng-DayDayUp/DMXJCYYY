# data.py
import os
from pathlib import Path
import requests
from typing import Tuple, List


def download_tiny_shakespeare(target_path: str):
    """Download tiny_shakespeare from a known URL if not present."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    target = Path(target_path)
    if target.exists():
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading tiny_shakespeare to {target} ...")
    import requests
    r = requests.get(url)
    r.raise_for_status()
    target.write_text(r.text, encoding='utf-8')
    return str(target)


class CharDataset:
    """
    Simple character-level dataset.
    For LM mode: returns (x, y) where y is x shifted by one (language modeling).
    For Seq2Seq autoencoding: returns (src, tgt) with src = x, tgt = y (same shifting).
    If you want true parallel data, you can extend this class or add another loader.
    """
    def __init__(self, file_path: str, seq_len: int = 128, mode: str = 'lm'):
        """
        mode: 'lm' or 'seq2seq'
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"{file_path} not found. Run download_tiny_shakespeare() first.")
        text = p.read_text(encoding='utf-8')
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.data = [self.char2idx[ch] for ch in text]
        self.seq_len = seq_len
        self.mode = mode

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + 1 + self.seq_len]
        # Cast to lists (they will be converted to tensors in collate)
        if self.mode == 'lm':
            return x, y
        elif self.mode == 'seq2seq':
            # For seq2seq autoencoding: source is x, target is y (shifted)
            # In real seq2seq you would have separate src/tgt preprocess.
            return x, y
        else:
            raise ValueError("mode must be 'lm' or 'seq2seq'")

    def collate_fn(self, batch):
        import torch
        xs = [b[0] for b in batch]
        ys = [b[1] for b in batch]
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


if __name__ == '__main__':
    # quick test downloader
    p = download_tiny_shakespeare('data/tiny_shakespeare.txt')
    ds = CharDataset(p, seq_len=64)
    print('Vocab size:', ds.vocab_size, 'Dataset len:', len(ds))
    x, y = ds[0]
    print('Example lengths', len(x), len(y))
