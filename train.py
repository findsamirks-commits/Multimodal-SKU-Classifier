"""Training pipeline for the Multimodal Product Classifier.
Features differential learning rates, early stopping, validation tracking,
and checkpointing.
"""

import json
from pathlib import Path
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config, default_config
from dataset import (
    ProductDataset,
    SimpleVocab,
    get_image_transforms,
    multimodal_collate_fn,
)
from model import MultimodalProductClassifier


def set_seed(seed: int = 42):
    """Ensure reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_datasets(
    df: pd.DataFrame,
    cfg: Config,
    title_col: str = "title",
    image_col: str = "image_path",
    label_col: str = "label",
) -> Tuple[ProductDataset, ProductDataset, ProductDataset, SimpleVocab, Dict[str, int]]:
    """Split dataset and fit vocabulary strictly on the training set to prevent data leakage."""
    # Build label mapping
    unique_labels = sorted(df[label_col].astype(str).unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

    # Stratified Train/Val/Test Split (70% train, 15% val, 15% test)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=cfg.SEED, stratify=df[label_col]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=cfg.SEED, stratify=temp_df[label_col]
    )

    print(
        f"[Data Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}"
    )
    print(f"[Classes] {len(unique_labels)} categories: {unique_labels}")

    # Build vocabulary ONLY on train split
    vocab = SimpleVocab()
    train_titles = train_df[title_col].tolist()
    vocab.fit_on_texts(
        train_titles,
        min_freq=cfg.MIN_WORD_FREQ,
        max_vocab_size=cfg.MAX_VOCAB_SIZE,
    )
    print(f"[Vocabulary] Built vocabulary with {len(vocab)} tokens from training set.")

    train_transform = get_image_transforms(
        is_train=True,
        image_size=cfg.IMAGE_SIZE,
        mean=cfg.IMAGE_MEAN,
        std=cfg.IMAGE_STD,
    )
    eval_transform = get_image_transforms(
        is_train=False,
        image_size=cfg.IMAGE_SIZE,
        mean=cfg.IMAGE_MEAN,
        std=cfg.IMAGE_STD,
    )

    train_ds = ProductDataset(
        df=train_df,
        vocab=vocab,
        transform=train_transform,
        max_title_len=cfg.MAX_TITLE_LEN,
        title_col=title_col,
        image_col=image_col,
        label_col=label_col,
        label_to_id=label_to_id,
    )

    val_ds = ProductDataset(
        df=val_df,
        vocab=vocab,
        transform=eval_transform,
        max_title_len=cfg.MAX_TITLE_LEN,
        title_col=title_col,
        image_col=image_col,
        label_col=label_col,
        label_to_id=label_to_id,
    )

    test_ds = ProductDataset(
        df=test_df,
        vocab=vocab,
        transform=eval_transform,
        max_title_len=cfg.MAX_TITLE_LEN,
        title_col=title_col,
        image_col=image_col,
        label_col=label_col,
        label_to_id=label_to_id,
    )

    return train_ds, val_ds, test_ds, vocab, label_to_id


def build_optimizer(
    model: MultimodalProductClassifier, cfg: Config
) -> torch.optim.Optimizer:
    """Create optimizer with differential learning rates for pre-trained CNN vs head."""
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "image_encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer_grouped_parameters = [
        {"params": backbone_params, "lr": cfg.LR_BACKBONE},
        {"params": head_params, "lr": cfg.LR_HEAD},
    ]

    return torch.optim.AdamW(
        optimizer_grouped_parameters, weight_decay=cfg.WEIGHT_DECAY
    )


def train_one_epoch(
    model: MultimodalProductClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Train model across a single epoch."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        images = batch["images"].to(device)
        flat_tokens = batch["flat_tokens"].to(device)
        offsets = batch["offsets"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(images=images, tokens=flat_tokens, offsets=offsets)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_epoch(
    model: MultimodalProductClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, float]:
    """Evaluate model and calculate loss, accuracy, and macro F1 score."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        images = batch["images"].to(device)
        flat_tokens = batch["flat_tokens"].to(device)
        offsets = batch["offsets"].to(device)
        labels = batch["labels"].to(device)

        logits = model(images=images, tokens=flat_tokens, offsets=offsets)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, f1


def train_multimodal_classifier(
    df: pd.DataFrame,
    cfg: Optional[Config] = None,
    title_col: str = "title",
    image_col: str = "image_path",
    label_col: str = "label",
) -> Tuple[MultimodalProductClassifier, Dict[str, Any]]:
    """Complete training loop with validation, early stopping, and artifact saving."""
    if cfg is None:
        cfg = default_config

    set_seed(cfg.SEED)
    device = cfg.DEVICE
    print(f"Using computing device: {device}")

    # Prepare datasets
    train_ds, val_ds, test_ds, vocab, label_to_id = prepare_datasets(
        df=df,
        cfg=cfg,
        title_col=title_col,
        image_col=image_col,
        label_col=label_col,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=multimodal_collate_fn,
        num_workers=cfg.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
        num_workers=cfg.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
        num_workers=cfg.NUM_WORKERS,
    )

    # Instantiate model
    num_classes = len(label_to_id)
    model = MultimodalProductClassifier(
        num_classes=num_classes,
        vocab_size=len(vocab),
        image_backbone=cfg.IMAGE_BACKBONE,
        pretrained_image=cfg.IMAGE_PRETRAINED,
        text_encoder_type=cfg.TEXT_ENCODER_TYPE,
        text_embed_dim=cfg.TEXT_EMBED_DIM,
        dropout=cfg.DROPOUT,
        use_batch_norm=cfg.USE_BATCH_NORM,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    best_val_f1 = -1.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    print("\n" + "=" * 60)
    print("Starting Multimodal Training")
    print("=" * 60)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = evaluate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch [{epoch:02d}/{cfg.NUM_EPOCHS:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        # Checkpoint if best validation Macro-F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0

            cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "label_to_id": label_to_id,
                "id_to_label": {v: k for k, v in label_to_id.items()},
                "config": {
                    "num_classes": num_classes,
                    "vocab_size": len(vocab),
                    "image_backbone": cfg.IMAGE_BACKBONE,
                    "text_encoder_type": cfg.TEXT_ENCODER_TYPE,
                    "text_embed_dim": cfg.TEXT_EMBED_DIM,
                    "dropout": cfg.DROPOUT,
                    "use_batch_norm": cfg.USE_BATCH_NORM,
                    "image_size": cfg.IMAGE_SIZE,
                },
            }
            torch.save(checkpoint, cfg.CHECKPOINT_PATH)
            vocab.save(cfg.VOCAB_PATH)

            with open(cfg.LABEL_MAP_PATH, "w", encoding="utf-8") as f:
                json.dump(label_to_id, f, indent=2)

            print(
                f"  >>> Best model saved! (Checkpoint: {cfg.CHECKPOINT_PATH.name})"
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(
                    f"\n[Early Stopping] No improvement in {cfg.PATIENCE} consecutive epochs."
                )
                break

    print("\n" + "=" * 60)
    print("Training Complete! Evaluating on Test Set...")
    print("=" * 60)

    # Load best checkpoint for test evaluation
    best_ckpt = torch.load(cfg.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_acc, test_f1 = evaluate_epoch(
        model, test_loader, criterion, device
    )
    print(
        f"[Test Set Results] Loss: {test_loss:.4f} | Accuracy: {test_acc * 100:.2f}% | Macro-F1: {test_f1:.4f}"
    )

    return model, {
        "history": history,
        "test_results": {
            "loss": test_loss,
            "accuracy": test_acc,
            "macro_f1": test_f1,
        },
        "vocab": vocab,
        "label_to_id": label_to_id,
        "test_loader": test_loader,
    }


if __name__ == "__main__":
    if default_config.DATA_CSV.exists():
        df = pd.read_csv(default_config.DATA_CSV)
        train_multimodal_classifier(df, default_config)
    else:
        print(f"Data file not found at: {default_config.DATA_CSV}")
        print("Run generate_sample_data.py to create sample data first.")
