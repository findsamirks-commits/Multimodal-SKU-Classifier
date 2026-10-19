"""Dataset and preprocessing utilities for the Multimodal Product Classifier.
Handles text tokenization, vocabulary management, image augmentations,
fault-tolerant image loading, and custom batch collation.
"""

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class SimpleVocab:
    """Lightweight vocabulary builder with special tokens and frequency thresholds.

    Strictly fitted on training texts to prevent data leakage.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self):
        self.token_to_id: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.id_to_token: Dict[int, str] = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN,
        }

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Convert text into lowercase alphanumeric token sequence."""
        if not isinstance(text, str):
            return []
        return re.findall(r"\b\w+\b", text.lower())

    def fit_on_texts(
        self,
        texts: List[str],
        min_freq: int = 2,
        max_vocab_size: Optional[int] = 10000,
    ) -> "SimpleVocab":
        """Build vocabulary from a list of training texts."""
        counter: Counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        sorted_tokens = [
            token
            for token, freq in counter.most_common()
            if freq >= min_freq and token not in self.token_to_id
        ]

        if max_vocab_size is not None:
            allowed_new_tokens = max(0, max_vocab_size - len(self.token_to_id))
            sorted_tokens = sorted_tokens[:allowed_new_tokens]

        for token in sorted_tokens:
            new_id = len(self.token_to_id)
            self.token_to_id[token] = new_id
            self.id_to_token[new_id] = token

        return self

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Convert string to list of token IDs."""
        tokens = self.tokenize(text)
        ids = [self.token_to_id.get(tok, self.UNK_IDX) for tok in tokens]
        if not ids:
            ids = [self.UNK_IDX]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def __len__(self) -> int:
        return len(self.token_to_id)

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist vocabulary mapping to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SimpleVocab":
        """Load vocabulary from saved JSON file."""
        vocab = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        vocab.token_to_id = token_to_id
        vocab.id_to_token = {int(v): k for k, v in token_to_id.items()}
        return vocab


def get_image_transforms(
    is_train: bool,
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """Build standardized Torchvision transformations for RGB images."""
    if is_train:
        return T.Compose(
            [
                T.Resize((image_size + 32, image_size + 32)),
                T.RandomCrop((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.CenterCrop((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


class ProductDataset(Dataset):
    """PyTorch Dataset loading multimodal product title & image pairs.

    Features robust error recovery for missing or corrupted images.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: SimpleVocab,
        transform: Optional[Callable] = None,
        max_title_len: int = 32,
        title_col: str = "title",
        image_col: str = "image_path",
        label_col: Optional[str] = "label",
        label_to_id: Optional[Dict[str, int]] = None,
        tfidf_features: Optional[np.ndarray] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.transform = transform
        self.max_title_len = max_title_len
        self.title_col = title_col
        self.image_col = image_col
        self.label_col = label_col
        self.label_to_id = label_to_id
        self.tfidf_features = tfidf_features

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: Union[str, Path]) -> Image.Image:
        """Safely load and convert image to RGB, falling back to a neutral image if corrupt."""
        try:
            if path and Path(path).exists():
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    return img.convert("RGB")
        except Exception:
            pass
        # Fallback 224x224 gray placeholder if file is missing or corrupted
        return Image.new("RGB", (224, 224), color=(128, 128, 128))

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        row = self.df.iloc[idx]
        title = str(row.get(self.title_col, ""))
        img_path = row.get(self.image_col, "")

        # 1. Image loading & transform
        pil_img = self._load_image(img_path)
        if self.transform:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = T.ToTensor()(pil_img)

        # 2. Text tokenization
        token_ids = self.vocab.encode(title, max_length=self.max_title_len)

        item: Dict[str, Any] = {
            "image": img_tensor,
            "tokens": token_ids,
            "title": title,
        }

        if self.tfidf_features is not None:
            item["tfidf"] = torch.tensor(
                self.tfidf_features[idx], dtype=torch.float32
            )

        # 3. Label encoding
        if self.label_col and self.label_col in row:
            raw_label = row[self.label_col]
            if self.label_to_id is not None:
                item["label"] = self.label_to_id[raw_label]
            else:
                item["label"] = int(raw_label)

        return item


def multimodal_collate_fn(
    batch: List[Dict[str, Any]], pad_idx: int = SimpleVocab.PAD_IDX
) -> Dict[str, torch.Tensor]:
    """Custom collator assembling padded text tensors, 1D flattened tokens with

    offsets for nn.EmbeddingBag, stacked image tensors, and labels.
    """
    images = torch.stack([b["image"] for b in batch])

    # Build 1D flattened tokens and offsets for nn.EmbeddingBag
    flat_tokens: List[int] = []
    offsets: List[int] = []
    current_offset = 0

    max_len = max(len(b["tokens"]) for b in batch)
    padded_tokens = []

    for b in batch:
        toks = b["tokens"]
        offsets.append(current_offset)
        flat_tokens.extend(toks)
        current_offset += len(toks)

        # Padded version
        pad_size = max_len - len(toks)
        padded_tokens.append(toks + [pad_idx] * pad_size)

    output = {
        "images": images,
        "flat_tokens": torch.tensor(flat_tokens, dtype=torch.long),
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "padded_tokens": torch.tensor(padded_tokens, dtype=torch.long),
        "titles": [b["title"] for b in batch],
    }

    if "tfidf" in batch[0]:
        output["tfidf"] = torch.stack([b["tfidf"] for b in batch])

    if "label" in batch[0]:
        output["labels"] = torch.tensor(
            [b["label"] for b in batch], dtype=torch.long
        )

    return output
