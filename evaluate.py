"""Evaluation and Modality Ablation Suite for the Multimodal Product Classifier.
Evaluates the model on test data, generates classification reports,
confusion matrices, and performs ablation studies (Multimodal vs Text-Only vs Image-Only).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader

from config import Config, default_config
from dataset import (
    ProductDataset,
    SimpleVocab,
    get_image_transforms,
    multimodal_collate_fn,
)
from model import MultimodalProductClassifier


def load_trained_model(
    checkpoint_path: Path,
    device: str = "cpu",
) -> Tuple[MultimodalProductClassifier, Dict, SimpleVocab, Dict[str, int]]:
    """Restore trained model, configuration, vocabulary, and label mappings from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg_dict = checkpoint["config"]
    label_to_id = checkpoint["label_to_id"]

    model = MultimodalProductClassifier(
        num_classes=cfg_dict["num_classes"],
        vocab_size=cfg_dict["vocab_size"],
        image_backbone=cfg_dict["image_backbone"],
        pretrained_image=False,  # Weights restored from checkpoint
        text_encoder_type=cfg_dict["text_encoder_type"],
        text_embed_dim=cfg_dict["text_embed_dim"],
        dropout=cfg_dict.get("dropout", 0.3),
        use_batch_norm=cfg_dict.get("use_batch_norm", True),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load associated vocab
    vocab_path = checkpoint_path.parent / "vocab.json"
    vocab = SimpleVocab.load(vocab_path)

    return model, cfg_dict, vocab, label_to_id


@torch.no_grad()
def run_evaluation(
    model: MultimodalProductClassifier,
    dataloader: DataLoader,
    device: str,
    ablate_image: bool = False,
    ablate_text: bool = False,
) -> Tuple[List[int], List[int], np.ndarray]:
    """Execute prediction over dataloader under optional modality ablation conditions."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for batch in dataloader:
        images = batch["images"].to(device)
        flat_tokens = batch["flat_tokens"].to(device)
        offsets = batch["offsets"].to(device)

        logits = model(
            images=images,
            tokens=flat_tokens,
            offsets=offsets,
            ablate_image=ablate_image,
            ablate_text=ablate_text,
        )

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

        all_probs.extend(probs)
        all_preds.extend(preds)
        if "labels" in batch:
            all_targets.extend(batch["labels"].cpu().numpy())

    return all_preds, all_targets, np.array(all_probs)


def perform_ablation_study(
    model: MultimodalProductClassifier,
    dataloader: DataLoader,
    device: str,
    id_to_label: Dict[int, str],
) -> pd.DataFrame:
    """Compare performance across Multimodal, Text-Only, and Image-Only conditions."""
    from sklearn.metrics import accuracy_score, f1_score

    conditions = [
        {"name": "Multimodal (Image + Text)", "ablate_img": False, "ablate_txt": False},
        {"name": "Text-Only (Zeroed Image)", "ablate_img": True, "ablate_txt": False},
        {"name": "Image-Only (Zeroed Text)", "ablate_img": False, "ablate_txt": True},
    ]

    results = []
    for cond in conditions:
        preds, targets, _ = run_evaluation(
            model=model,
            dataloader=dataloader,
            device=device,
            ablate_image=cond["ablate_img"],
            ablate_text=cond["ablate_txt"],
        )
        acc = accuracy_score(targets, preds) * 100
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(targets, preds, average="weighted", zero_division=0)

        results.append(
            {
                "Modality Condition": cond["name"],
                "Accuracy (%)": round(acc, 2),
                "Macro F1": round(macro_f1, 4),
                "Weighted F1": round(weighted_f1, 4),
            }
        )

    df_results = pd.DataFrame(results)
    best_single = max(results[1]["Macro F1"], results[2]["Macro F1"])
    multimodal_f1 = results[0]["Macro F1"]
    lift = (multimodal_f1 - best_single) / (best_single + 1e-9) * 100
    df_results["Lift vs Best Unimodal (%)"] = [
        f"{lift:+.2f}%" if i == 0 else "-" for i in range(len(results))
    ]

    return df_results


def evaluate_checkpoint(
    checkpoint_path: Optional[Path] = None,
    test_df: Optional[pd.DataFrame] = None,
    cfg: Optional[Config] = None,
):
    """Full evaluation report generation on test dataset."""
    if cfg is None:
        cfg = default_config
    if checkpoint_path is None:
        checkpoint_path = cfg.CHECKPOINT_PATH

    device = cfg.DEVICE
    print(f"Loading checkpoint from: {checkpoint_path}")
    model, cfg_dict, vocab, label_to_id = load_trained_model(
        checkpoint_path, device=device
    )
    id_to_label = {v: k for k, v in label_to_id.items()}

    # If no test_df provided, load from data CSV with fixed random seed
    if test_df is None:
        from sklearn.model_selection import train_test_split

        full_df = pd.read_csv(cfg.DATA_CSV)
        _, temp_df = train_test_split(
            full_df, test_size=0.30, random_state=cfg.SEED, stratify=full_df["label"]
        )
        _, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=cfg.SEED, stratify=temp_df["label"]
        )

    test_transform = get_image_transforms(
        is_train=False,
        image_size=cfg_dict.get("image_size", cfg.IMAGE_SIZE),
        mean=cfg.IMAGE_MEAN,
        std=cfg.IMAGE_STD,
    )
    test_ds = ProductDataset(
        df=test_df,
        vocab=vocab,
        transform=test_transform,
        label_to_id=label_to_id,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
        num_workers=cfg.NUM_WORKERS,
    )

    print("\n" + "=" * 65)
    print("RUNNING MULTIMODAL EVALUATION & CLASSIFICATION REPORT")
    print("=" * 65)

    preds, targets, _ = run_evaluation(model, test_loader, device=device)
    target_names = [id_to_label[i] for i in range(len(id_to_label))]

    print("\nClassification Report:")
    print(
        classification_report(
            targets, preds, target_names=target_names, zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    cm = confusion_matrix(targets, preds)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)

    print("\n" + "=" * 65)
    print("RUNNING MODALITY ABLATION STUDY (Image vs Text vs Multimodal)")
    print("=" * 65)
    ablation_df = perform_ablation_study(
        model, test_loader, device=device, id_to_label=id_to_label
    )
    print(ablation_df.to_string(index=False))


if __name__ == "__main__":
    evaluate_checkpoint()
